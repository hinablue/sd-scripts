import argparse
import os
from typing import Optional

import safetensors
from safetensors.torch import safe_open
import torch

from library.utils import setup_logging, str_to_dtype
from library.safetensors_utils import load_safetensors, mem_eff_save_file


setup_logging()
import logging


logger = logging.getLogger(__name__)


def convert_flux_safetensors_dtype(
    input_path: str,
    output_path: str,
    save_precision: Optional[str] = None,
    device: str = "cpu",
    mem_eff_load_save: bool = False,
):
    """
    Convert a Flux (or compatible) .safetensors checkpoint to a specified dtype, preserving metadata.

    Args:
        input_path: Path to the source .safetensors file.
        output_path: Path to save the converted .safetensors file.
        save_precision: Target dtype string (e.g., 'fp16', 'bf16', 'float32').
        device: Device to load tensors on during conversion (typically 'cpu').
        mem_eff_load_save: If True, uses memory-efficient load/save utilities.
    """

    if os.path.splitext(input_path)[1].lower() != ".safetensors":
        raise ValueError("input_path must be a .safetensors file")

    if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target_dtype: Optional[torch.dtype] = str_to_dtype(save_precision) if save_precision else None

    # Read metadata first (if present)
    metadata = None
    try:
        with safe_open(input_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata:
                logger.info(f"Found metadata: {metadata}")
    except Exception as e:
        logger.warning(f"Failed to read metadata from input: {e}")

    # Load tensors, casting if requested
    logger.info(f"Loading input tensors from {input_path}")
    state_dict = load_safetensors(
        input_path,
        device=device,
        disable_mmap=mem_eff_load_save,
        dtype=target_dtype,
    )

    # Ensure only floating tensors are cast (load_safetensors already applies dtype globally)
    # Re-apply selective cast in case some non-float tensors were inadvertently attempted
    if target_dtype is not None:
        for k, v in list(state_dict.items()):
            if torch.is_floating_point(v) and v.dtype != target_dtype:
                state_dict[k] = v.to(target_dtype)

    # Save converted tensors
    logger.info(f"Saving converted tensors to {output_path} ({len(state_dict)} keys)")
    if mem_eff_load_save:
        mem_eff_save_file(state_dict, output_path, metadata)
    else:
        safetensors.torch.save_file(state_dict, output_path, metadata=metadata)

    logger.info("Conversion completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Flux .safetensors to a specified dtype")
    parser.add_argument("--input", required=True, help="Path to input Flux .safetensors file")
    parser.add_argument("--output", required=True, help="Path to save converted .safetensors file")
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        help=(
            "Precision to save weights in (e.g., float32, fp16, bf16, fp8 (fp8_e4m3fn), fp8_e4m3fn, fp8_e4m3fnuz, fp8_e5m2, fp8_e5m2fnuz). "
            "If omitted, keeps original dtypes."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to load tensors on during conversion")
    parser.add_argument(
        "--mem_eff_load_save",
        action="store_true",
        help="Use memory-efficient load/save (recommended for large checkpoints)",
    )
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    convert_flux_safetensors_dtype(
        input_path=args.input,
        output_path=args.output,
        save_precision=args.save_precision,
        device=args.device,
        mem_eff_load_save=args.mem_eff_load_save,
    )


if __name__ == "__main__":
    main()


