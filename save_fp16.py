import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

# Specify the path to your .safetensors file
safetensors_path = "/Volumes/HinaDisk/modals-for-sd-scripts/gemma-2-2b.safetensors"

# Load the state_dict from the safetensors file
state_dict = load_file(safetensors_path)

# Create a new state_dict for fp16
fp16_state_dict = {}

# Convert each tensor in the state_dict to float16
for key, value in state_dict.items():
    if isinstance(value, torch.Tensor):
        fp16_state_dict[key] = value.to(torch.float16)
    else:
        fp16_state_dict[key] = value # Keep non-tensor items as they are

# Specify the path for the new fp16 safetensors file
fp16_safetensors_path = "/Volumes/HinaDisk/modals-for-sd-scripts/gemma-2-2b-fp16.safetensors"

# Save the fp16 state_dict to a new safetensors file
save_file(fp16_state_dict, fp16_safetensors_path)

print(f"Model loaded from '{safetensors_path}' and saved as fp16 to '{fp16_safetensors_path}'")
