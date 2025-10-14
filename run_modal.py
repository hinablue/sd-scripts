'''

kohya-ss/sd-scripts on https://modal.com
Run training with the following command:
modal run run_modal.py -t flux --config /root/sd-scripts/datasets/config.toml

'''

import os
import subprocess
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import modal
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, "/root/sd-scripts")
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# define the volume for storing model outputs, using "creating volumes lazily": https://modal.com/docs/guide/volumes
# you will find your model, samples and optimizer stored in: https://modal.com/storage/your-username/main/flux-lora-models
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)

# modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_DIR = "/root/sd-scripts/modal_output"  # modal_output, due to "cannot mount volume on non-empty path" requirement

# define modal app
image = (
    modal.Image.debian_slim(python_version="3.11")
    # install required system and pip packages, more about this modal approach: https://modal.com/docs/examples/dreambooth_app
    .apt_install("libgl1", "libglib2.0-0", "git", "wget")
    .pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        index_url="https://download.pytorch.org/whl/cu126"
    )
    .pip_install_from_requirements(
        "/Volumes/HinaDisk/sd-scripts/requirements_modal.txt",
        extra_options="-U"
    )
    .pip_install(
        "xformers==0.0.30",
        index_url="https://download.pytorch.org/whl/cu126"
    )
)

# mount for the entire sd-scripts directory
# example: "/Users/username/sd-scripts" is the local directory, "/root/sd-scripts" is the remote directory
image = image.add_local_dir(
    "/Volumes/HinaDisk/sd-scripts",
    "/root/sd-scripts",
    ignore=["__pycache__", "*.git", "*.github", "*.egg-info", "*.ai", "build", ".vscode", "wandb", "CLAUDE.md", "GEMINI.md", ".claude", ".gemini", "tests", "docs", "bitsandbytes_windows"]
)

# create the Modal app with the necessary mounts and volumes
app = modal.App(name="flux-lora-training", image=image, volumes={MOUNT_DIR: model_volume})

# Check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # Set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)

import argparse

@app.function(
    # request a GPU with at least 24GB VRAM
    # more about modal GPU's: https://modal.com/docs/guide/gpu
    gpu="A100", # gpu="H100"
    # more about modal timeouts: https://modal.com/docs/guide/timeouts
    timeout=7200,  # 2 hours, increase or decrease if needed
    # Add your huggingface read token: https://modal.com/docs/guide/secrets
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def main(
    config_file: str = None,
    name: str = None,
    additional_args: str = None,
    train_type: str = "flux",
    pretrained_model: str = None
):
    if config_file is None and additional_args is None:
        raise ValueError("Config file or additional_args are required")

    if train_type is None:
        raise ValueError("Train type is required")

    import toml

    hf_token = os.environ["HF_TOKEN"]

    os.system(f"cd /root/sd-scripts && pip install -e .")

    if additional_args is None:
        additional_args = ""
    else:
        additional_args = f"{additional_args}"

    config_args = None

    if config_file is None:
        config_file = ""
    else:
        config_args = toml.load(config_file)
        config_file = f"--config_file=\"{config_file}\""

    output_dir = MOUNT_DIR

    if config_args is not None:
        output_dir = f"{MOUNT_DIR}/{config_args['output_name']}"

        if pretrained_model is None and config_args['pretrained_model_name_or_path'] != "":
            pretrained_model = config_args['pretrained_model_name_or_path']
            if not pretrained_model.startswith("http"):
                raise ValueError("Pretrained model name or path must be a valid URL")

    if name is not None:
        additional_args = f"{additional_args} --output_dir=\"{MOUNT_DIR}/{name}\" --output_name=\"{name}\""
    else:
        additional_args = f"{additional_args} --output_dir=\"{output_dir}\""

    print(f"Preparing pretrained/clip/vae/t5xxl models...")

    subprocess.run(f"mkdir -p {MOUNT_DIR}/models", shell=True, check=True)

    print(f"Pretrained model: {pretrained_model}")
    print(f"Train type: {train_type}")

    train_network_script = ""

    # 如果 pretrained_model 符合 http，则下载模型
    if pretrained_model is not None:
        if pretrained_model.startswith("http"):
            model_name = os.path.basename(pretrained_model)

            if not os.path.exists(f"{MOUNT_DIR}/models/{model_name}"):
                subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' {pretrained_model} -O {MOUNT_DIR}/models/{model_name}", shell=True, check=True)
                additional_args = f"{additional_args} --pretrained_model_name_or_path=\"{MOUNT_DIR}/models/{model_name}\""

            print(f"Model name: {model_name}")

    if train_type == "sdxl":
        train_network_script = "sdxl_train_network.py"

        # download vae file from huggingface
        if not os.path.exists(f"{MOUNT_DIR}/models/sdxl_vae_fp16_fix.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sdxl_vae_fp16_fix.safetensors -O {MOUNT_DIR}/models/sdxl_vae_fp16_fix.safetensors", shell=True, check=True)

        print(f"VAE downloaded")

        # override vae path
        additional_args = f"{additional_args} --vae=\"{MOUNT_DIR}/models/sdxl_vae_fp16_fix.safetensors\""

        if pretrained_model is None:
            if not os.path.exists(f"{MOUNT_DIR}/models/sd_xl_base_1.0.safetensors"):
                subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -O {MOUNT_DIR}/models/sd_xl_base_1.0.safetensors", shell=True, check=True)

            additional_args = f"{additional_args} --pretrained_model_name_or_path=\"{MOUNT_DIR}/models/sd_xl_base_1.0.safetensors\""

            print(f"Pretrained model downloaded")
    elif train_type == "sd3" or train_type == "sd3_medium":
        train_network_script = "sd3_train_network.py"

        # download clip_l file from huggingface
        if not os.path.exists(f"{MOUNT_DIR}/models/sd35_clip_l.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/hinablue/modals-for-sd-scripts/resolve/main/sd35_clip_l.safetensors -O {MOUNT_DIR}/models/clip_l.safetensors", shell=True, check=True)

        print(f"Clip_l downloaded")
        if not os.path.exists(f"{MOUNT_DIR}/models/sd35_clip_g.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/hinablue/modals-for-sd-scripts/resolve/main/sd35_clip_g.safetensors -O {MOUNT_DIR}/models/clip_g.safetensors", shell=True, check=True)

        print(f"Clip_g downloaded")
        if not os.path.exists(f"{MOUNT_DIR}/models/t5xxl_fp8_e4m3fn.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/hinablue/modals-for-sd-scripts/resolve/main/t5xxl_fp8_e4m3fn.safetensors -O {MOUNT_DIR}/models/t5xxl_fp8_e4m3fn.safetensors", shell=True, check=True)

        print(f"T5xxl downloaded")

        # override clip_l path
        additional_args = f"{additional_args} --clip_l=\"{MOUNT_DIR}/models/sd3_clip_l.safetensors\""
        # override clip_g path
        additional_args = f"{additional_args} --clip_g=\"{MOUNT_DIR}/models/sd3_clip_g.safetensors\""
        # override t5xxl path
        additional_args = f"{additional_args} --t5xxl=\"{MOUNT_DIR}/models/t5xxl_fp8_e4m3fn.safetensors\""

        if pretrained_model is None:
            if train_type == "sd3_medium":
                if not os.path.exists(f"{MOUNT_DIR}/models/sd3.5_medium.safetensors"):
                    subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/resolve/main/sd3.5_medium.safetensors -O {MOUNT_DIR}/models/sd3.5_medium.safetensors", shell=True, check=True)
                additional_args = f"{additional_args} --pretrained_model_name_or_path=\"{MOUNT_DIR}/models/sd3.5_medium.safetensors\""

                print(f"{train_type} Pretrained model downloaded")
            else:
                if not os.path.exists(f"{MOUNT_DIR}/models/sd3.5_large.safetensors"):
                    subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors -O {MOUNT_DIR}/models/sd3.5_large.safetensors", shell=True, check=True)
                additional_args = f"{additional_args} --pretrained_model_name_or_path=\"{MOUNT_DIR}/models/sd3.5_large.safetensors\""

                print(f"{train_type} Pretrained model downloaded")
    elif train_type == "flux" or train_type == "flux_krea":
        train_network_script = "flux_train_network.py"

        # download vae file from huggingface
        if not os.path.exists(f"{MOUNT_DIR}/models/clip_l.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/hinablue/modals-for-sd-scripts/resolve/main/clip_l.safetensors -O {MOUNT_DIR}/models/clip_l.safetensors", shell=True, check=True)

        print(f"Clip_l downloaded")

        if not os.path.exists(f"{MOUNT_DIR}/models/flux_vae.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/hinablue/modals-for-sd-scripts/resolve/main/flux_vae.safetensors -O {MOUNT_DIR}/models/flux_vae.safetensors", shell=True, check=True)

        print(f"Flux_vae downloaded")

        if not os.path.exists(f"{MOUNT_DIR}/models/t5xxl_fp8_e4m3fn.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/hinablue/modals-for-sd-scripts/resolve/main/t5xxl_fp8_e4m3fn.safetensors -O {MOUNT_DIR}/models/t5xxl_fp8_e4m3fn.safetensors", shell=True, check=True)

        print(f"T5xxl downloaded")

        # override ae path
        additional_args = f"{additional_args} --ae=\"{MOUNT_DIR}/models/flux_vae.safetensors\""
        # override clip_l path
        additional_args = f"{additional_args} --clip_l=\"{MOUNT_DIR}/models/clip_l.safetensors\""
        # override t5xxl path
        additional_args = f"{additional_args} --t5xxl=\"{MOUNT_DIR}/models/t5xxl_fp8_e4m3fn.safetensors\""

        if pretrained_model is None:
            if train_type == "flux_krea":
                if not os.path.exists(f"{MOUNT_DIR}/models/flux1-krea-dev.safetensors"):
                    subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev/resolve/main/flux1-krea-dev.safetensors -O {MOUNT_DIR}/models/flux1-krea-dev.safetensors", shell=True, check=True)
                additional_args = f"{additional_args} --pretrained_model_name_or_path=\"{MOUNT_DIR}/models/flux1-krea-dev.safetensors\""

                print(f"{train_type} Pretrained model downloaded")
            else:
                if not os.path.exists(f"{MOUNT_DIR}/models/flux1-dev.safetensors"):
                    subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors -O {MOUNT_DIR}/models/flux1-dev.safetensors", shell=True, check=True)
                additional_args = f"{additional_args} --pretrained_model_name_or_path=\"{MOUNT_DIR}/models/flux1-dev.safetensors\""

                print(f"{train_type} Pretrained model downloaded")
    elif train_type == "lumina":
        train_network_script = "lumina_train_network.py"

        if not os.path.exists(f"{MOUNT_DIR}/models/flux_vae.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/hinablue/modals-for-sd-scripts/resolve/main/flux_vae.safetensors -O {MOUNT_DIR}/models/flux_vae.safetensors", shell=True, check=True)

        print(f"Flux_vae downloaded")

        if not os.path.exists(f"{MOUNT_DIR}/models/gemma-2-2b-fp16.safetensors"):
            subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/hinablue/modals-for-sd-scripts/resolve/main/gemma-2-2b-fp16.safetensors -O {MOUNT_DIR}/models/gemma-2-2b-fp16.safetensors", shell=True, check=True)

        print(f"Gemma-2-2b-fp16 downloaded")

        # override ae path
        additional_args = f"{additional_args} --ae=\"{MOUNT_DIR}/models/flux_vae.safetensors\""
        # override gemma2 path
        additional_args = f"{additional_args} --gemma2=\"{MOUNT_DIR}/models/gemma-2-2b-fp16.safetensors\""

        if pretrained_model is None:
            if not os.path.exists(f"{MOUNT_DIR}/models/lumina-image-2.safetensors"):
                subprocess.run(f"wget -q -c --header='Authorization: Bearer {hf_token}' https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/resolve/3648256a142a83759fc9b8599113780a02f425c1/lumina-image-2.safetensors -O {MOUNT_DIR}/models/lumina-image-2.safetensors", shell=True, check=True)
            additional_args = f"{additional_args} --pretrained_model_name_or_path=\"{MOUNT_DIR}/models/lumina-image-2.safetensors\""

            print(f"Pretrained model downloaded")
    else:
        raise ValueError("Train network script must be sdxl or sd3 or flux or lumina")

    model_volume.commit()

    run_cmd = f"accelerate launch --config_file /root/sd-scripts/datasets/modal_accelerate_config.yaml /root/sd-scripts/{train_network_script} {config_file} {additional_args}"

    print(f"Running command: {run_cmd}")

    subprocess.run(run_cmd, shell=True, check=True)

    model_volume.commit()
    model.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # optional name replacement for config file
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )

    # config file
    parser.add_argument(
        '-c', '--config_file',
        type=str,
        default=None,
        help='Config file (toml) to use'
    )

    # train network script
    parser.add_argument(
        '-t', '--train_type',
        type=str,
        default=None,
        help='Train network script to use, default is flux'
    )

    parser.add_argument(
        '-m', '--pretrained_model',
        type=str,
        default=None,
        help='Pretrained model name or path'
    )

    # additional arguments
    parser.add_argument(
        '-a', '--additional_args',
        type=str,
        default=None,
        help='Additional arguments to override the config file'
    )

    args = parser.parse_args()

    main.call(config_file=args.config_file, name=args.name, additional_args=args.additional_args, train_type=args.train_type, pretrained_model=args.pretrained_model)
