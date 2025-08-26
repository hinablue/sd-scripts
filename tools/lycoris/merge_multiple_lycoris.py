#!/usr/bin/env python3
"""
Merge multiple LoKr/LoHa models into a single model.

This tool allows you to combine multiple LoKr/LoHa models with different weights
into a single model, similar to how LoRA models can be merged.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from lycoris.utils import merge
from library.model_util import load_file


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple LoKr/LoHa models into a single model"
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="List of LoKr/LoHa model paths to merge",
        type=str
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Weight for each model (must match number of models). Default: equal weights",
        default=None
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output model path",
        default="./merged_lycoris.safetensors",
        type=str
    )
    parser.add_argument(
        "--device",
        help="Device to use for merging",
        default="cpu",
        type=str
    )
    parser.add_argument(
        "--dtype",
        help="Data type for output model",
        default="float16",
        type=str
    )
    parser.add_argument(
        "--metadata",
        help="Path to save metadata about the merge",
        default=None,
        type=str
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the merged model by checking weight statistics"
    )
    return parser.parse_args()


def load_lycoris_model(model_path: str) -> Dict[str, torch.Tensor]:
    """Load a LoKr/LoHa model from file."""
    if model_path.endswith(".safetensors"):
        return load_file(model_path)
    else:
        return torch.load(model_path, map_location="cpu")


def get_model_type(state_dict: Dict[str, torch.Tensor]) -> str:
    """Determine if the model is LoKr or LoHa based on weight names."""
    keys = list(state_dict.keys())

    # Check for LoKr weights
    lokr_keys = [k for k in keys if any(w in k for w in ["lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t1", "lokr_t2"])]

    # Check for LoHa weights
    loha_keys = [k for k in keys if any(w in k for w in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2"])]

    if lokr_keys and not loha_keys:
        return "LoKr"
    elif loha_keys and not lokr_keys:
        return "LoHa"
    elif lokr_keys and loha_keys:
        return "Mixed"  # Some models might have both
    else:
        return "Unknown"


def validate_models_compatibility(models: List[Dict[str, torch.Tensor]]) -> Tuple[str, bool]:
    """Validate that all models are compatible for merging."""
    if not models:
        return "No models provided", False

    # Get model types
    model_types = [get_model_type(model) for model in models]

    # Check if all models are of the same type (or mixed)
    unique_types = set(model_types)
    if len(unique_types) == 1:
        return f"All models are {model_types[0]}", True
    elif "Mixed" in unique_types:
        return "Models contain mixed LoKr/LoHa types", True
    elif len(unique_types) == 2 and "Unknown" not in unique_types:
        return f"Models are {', '.join(unique_types)} - merging may work but verify results", True
    else:
        return f"Incompatible model types: {', '.join(unique_types)}", False

    # Check if all models have the same structure
    first_model = models[0]
    for i, model in enumerate(models[1:], 1):
        if set(model.keys()) != set(first_model.keys()):
            return f"Model {i} has different structure than first model", False

    return "Models are compatible", True


def merge_lycoris_weights(
    models: List[Dict[str, torch.Tensor]],
    weights: List[float],
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Merge multiple LoKr/LoHa models with given weights."""
    if len(models) != len(weights):
        raise ValueError(f"Number of models ({len(models)}) must match number of weights ({len(weights)})")

    # Initialize merged model with first model
    merged_model = {}
    first_model = models[0]

    # Get all unique keys
    all_keys = set()
    for model in models:
        all_keys.update(model.keys())

    print(f"Merging {len(models)} models with weights: {[f'{w:.3f}' for w in weights]}")
    print(f"Total unique keys: {len(all_keys)}")

    # Merge each key
    for key in all_keys:
        if key in ["alpha", "dora_scale"]:
            # For alpha and dora_scale, use weighted sum (not normalized)
            values = []
            for model, weight in zip(models, weights):
                if key in model:
                    values.append(model[key] * weight)

            if values:
                merged_model[key] = torch.stack(values).sum(dim=0)
        else:
            # For weight matrices, use weighted sum
            merged_tensor = None
            for model, weight in zip(models, weights):
                if key in model:
                    if merged_tensor is None:
                        merged_tensor = model[key] * weight
                    else:
                        merged_tensor = merged_tensor + model[key] * weight

            if merged_tensor is not None:
                merged_model[key] = merged_tensor

    return merged_model


def save_merged_model(
    merged_model: Dict[str, torch.Tensor],
    output_path: str,
    dtype: str = "float16"
) -> None:
    """Save the merged model to file."""
    # Convert dtype
    dtype_map = {
        "float": torch.float,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }

    target_dtype = dtype_map.get(dtype, torch.float16)

    # Convert model to target dtype
    converted_model = {}
    for key, value in merged_model.items():
        converted_model[key] = value.to(target_dtype)

    # Save model
    if output_path.endswith(".safetensors"):
        try:
            from safetensors.torch import save_file
            save_file(converted_model, output_path)
            print(f"Saved merged model as safetensors: {output_path}")
        except ImportError:
            print("Warning: safetensors not available, saving as .pt file")
            torch.save(converted_model, output_path.replace(".safetensors", ".pt"))
    else:
        torch.save(converted_model, output_path)
        print(f"Saved merged model: {output_path}")


def verify_merged_model(
    merged_model: Dict[str, torch.Tensor],
    original_models: List[Dict[str, torch.Tensor]]
) -> None:
    """Verify the merged model by checking weight statistics."""
    print("\n=== Model Verification ===")

    # Check weight statistics for merged model
    print("Merged model statistics:")
    for key, value in merged_model.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape={list(value.shape)}, "
                  f"mean={value.float().mean().item():.6f}, "
                  f"std={value.float().std().item():.6f}, "
                  f"min={value.float().min().item():.6f}, "
                  f"max={value.float().max().item():.6f}")

    # Compare with original models
    print("\nComparison with original models:")
    for i, original_model in enumerate(original_models):
        print(f"\nOriginal model {i+1}:")
        for key, value in original_model.items():
            if key in merged_model and torch.is_tensor(value):
                orig_mean = value.float().mean().item()
                merged_mean = merged_model[key].float().mean().item()
                diff = abs(orig_mean - merged_mean)
                print(f"  {key}: mean_diff={diff:.6f}")


def save_metadata(
    models: List[str],
    weights: List[float],
    output_path: str,
    metadata_path: str
) -> None:
    """Save metadata about the merge operation."""
    metadata = {
        "merge_info": {
            "timestamp": torch.datetime.now().isoformat(),
            "num_models": len(models),
            "models": models,
            "weights": weights,
            "output_path": output_path
        },
        "model_types": [get_model_type(load_lycoris_model(model)) for model in models]
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved merge metadata: {metadata_path}")


@torch.no_grad()
def main():
    args = get_args()

    # Validate arguments
    if args.weights and len(args.weights) != len(args.models):
        print(f"Error: Number of weights ({len(args.weights)}) must match number of models ({len(args.models)})")
        return 1

    # Set default weights if not provided
    if not args.weights:
        args.weights = [1.0] * len(args.models)

    print(f"Loading {len(args.models)} models...")

    # Load all models
    models = []
    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return 1

        try:
            model = load_lycoris_model(model_path)
            models.append(model)
            model_type = get_model_type(model)
            print(f"  Loaded {model_path} ({model_type})")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return 1

    # Validate model compatibility
    compatibility_msg, is_compatible = validate_models_compatibility(models)
    print(f"\nCompatibility check: {compatibility_msg}")

    if not is_compatible:
        print("Warning: Models may not be compatible. Proceed with caution.")

    # Merge models
    print(f"\nMerging models...")
    try:
        merged_model = merge_lycoris_weights(models, args.weights, args.device)
    except Exception as e:
        print(f"Error during merging: {e}")
        return 1

    # Save merged model
    print(f"\nSaving merged model...")
    try:
        save_merged_model(merged_model, args.output, args.dtype)
    except Exception as e:
        print(f"Error saving merged model: {e}")
        return 1

    # Save metadata if requested
    if args.metadata:
        try:
            save_metadata(args.models, args.weights, args.output, args.metadata)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

    # Verify merged model if requested
    if args.verify:
        try:
            verify_merged_model(merged_model, models)
        except Exception as e:
            print(f"Warning: Could not verify merged model: {e}")

    print(f"\nMerge completed successfully!")
    print(f"Output: {args.output}")
    print(f"Total parameters: {sum(p.numel() for p in merged_model.values() if torch.is_tensor(p)):,}")

    return 0


if __name__ == "__main__":
    exit(main())
