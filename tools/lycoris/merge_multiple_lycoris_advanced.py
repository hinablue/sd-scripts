#!/usr/bin/env python3
"""
Advanced merge tool for multiple LoKr/LoHa models.

This tool provides advanced merging strategies including:
- Smart weight fusion based on model characteristics
- Layer-wise weight adjustment
- Automatic weight optimization
- Support for different fusion algorithms
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import json
import numpy as np
from enum import Enum

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from lycoris.utils import merge
from library.model_util import load_file


class FusionStrategy(Enum):
    """Different fusion strategies for merging models."""
    WEIGHTED_SUM = "weighted_sum"
    WEIGHTED_AVERAGE = "weighted_average"
    LAYER_ADAPTIVE = "layer_adaptive"
    SMART_FUSION = "smart_fusion"
    MIN_MAX_NORM = "min_max_norm"


class AdvancedLycorisMerger:
    """Advanced merger for LoKr/LoHa models with multiple fusion strategies."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def load_model(self, model_path: str) -> Dict[str, torch.Tensor]:
        """Load a LoKr/LoHa model from file."""
        if model_path.endswith(".safetensors"):
            return load_file(model_path)
        else:
            return torch.load(model_path, map_location=self.device)

    def get_model_type(self, state_dict: Dict[str, torch.Tensor]) -> str:
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
            return "Mixed"
        else:
            return "Unknown"

    def analyze_model_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Analyze weight statistics for each layer."""
        analysis = {}

        for key, value in state_dict.items():
            if torch.is_tensor(value):
                analysis[key] = {
                    "shape": list(value.shape),
                    "mean": value.float().mean().item(),
                    "std": value.float().std().item(),
                    "min": value.float().min().item(),
                    "max": value.float().max().item(),
                    "norm": value.float().norm().item(),
                    "numel": value.numel()
                }

        return analysis

    def calculate_layer_importance(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate importance score for each layer based on weight magnitude."""
        importance = {}

        for key, value in state_dict.items():
            if torch.is_tensor(value):
                # Use L2 norm as importance measure
                importance[key] = value.float().norm().item()

        # Normalize importance scores for stable fusion
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance

    def smart_fusion_weights(
        self,
        models: List[Dict[str, torch.Tensor]],
        base_weights: List[float]
    ) -> List[Dict[str, float]]:
        """Calculate smart fusion weights based on model characteristics."""
        if len(models) != len(base_weights):
            raise ValueError("Number of models must match number of base weights")

        # Analyze each model
        model_analyses = [self.analyze_model_weights(model) for model in models]
        model_importances = [self.calculate_layer_importance(model) for model in models]

        # Calculate adaptive weights for each layer
        layer_weights = {}
        all_keys = set()
        for model in models:
            all_keys.update(model.keys())

        for key in all_keys:
            # Get importance scores for this layer across all models
            layer_importance = []
            for i, importance_dict in enumerate(model_importances):
                if key in importance_dict:
                    layer_importance.append((i, importance_dict[key]))
                else:
                    layer_importance.append((i, 0.0))

            # Sort by importance
            layer_importance.sort(key=lambda x: x[1], reverse=True)

            # Calculate adaptive weights
            adaptive_weights = []
            for i, base_weight in enumerate(base_weights):
                # Find importance rank for this model
                rank = next(idx for idx, _ in layer_importance if idx == i)
                importance_score = next(imp for idx, imp in layer_importance if idx == i)

                # Adjust weight based on importance
                if importance_score > 0:
                    adjusted_weight = base_weight * (1 + importance_score)
                else:
                    adjusted_weight = base_weight * 0.5  # Reduce weight for less important layers

                adaptive_weights.append(adjusted_weight)

            # Normalize weights for this layer to prevent extreme values, then multiply by user weights
            total_weight = sum(adaptive_weights)
            if total_weight > 0:
                # First normalize to prevent extreme values
                normalized_weights = [w / total_weight for w in adaptive_weights]
                # Then multiply by user's base weights to maintain their intended proportions
                layer_weights[key] = [norm_w * base_w for norm_w, base_w in zip(normalized_weights, base_weights)]
            else:
                # Fallback: use user weights directly
                layer_weights[key] = base_weights

        return layer_weights

    def merge_with_strategy(
        self,
        models: List[Dict[str, torch.Tensor]],
        weights: List[float],
        strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Merge models using the specified strategy."""

        if strategy == FusionStrategy.WEIGHTED_SUM:
            return self._merge_weighted_sum(models, weights)
        elif strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._merge_weighted_average(models, weights)
        elif strategy == FusionStrategy.LAYER_ADAPTIVE:
            return self._merge_layer_adaptive(models, weights)
        elif strategy == FusionStrategy.SMART_FUSION:
            return self._merge_smart_fusion(models, weights)
        elif strategy == FusionStrategy.MIN_MAX_NORM:
            return self._merge_min_max_norm(models, weights, **kwargs)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def _merge_weighted_sum(
        self,
        models: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Simple weighted sum fusion."""
        merged_model = {}
        all_keys = set()
        for model in models:
            all_keys.update(model.keys())

        for key in all_keys:
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

    def _merge_weighted_average(
        self,
        models: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Weighted average fusion (using original weights, not normalized)."""
        merged_model = {}
        all_keys = set()
        for model in models:
            all_keys.update(model.keys())

        for key in all_keys:
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

    def _merge_layer_adaptive(
        self,
        models: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Layer-adaptive fusion based on weight characteristics."""
        merged_model = {}
        all_keys = set()
        for model in models:
            all_keys.update(model.keys())

        for key in all_keys:
            # Get all values for this layer
            values = []
            valid_weights = []

            for model, weight in zip(models, weights):
                if key in model:
                    values.append(model[key])
                    valid_weights.append(weight)

            if not values:
                continue

            # Calculate adaptive weights based on value characteristics
            if len(values) > 1:
                # Calculate variance across models for this layer
                stacked = torch.stack(values)
                variance = torch.var(stacked, dim=0)
                mean_variance = variance.mean().item()

                # Adjust weights based on variance (lower variance = higher weight)
                if mean_variance > 0:
                    adjusted_weights = []
                    for i, value in enumerate(values):
                        layer_variance = variance.mean().item()
                        adjustment = 1.0 / (1.0 + layer_variance)
                        adjusted_weights.append(valid_weights[i] * adjustment)

                    # Use adjusted weights directly without normalization
                else:
                    adjusted_weights = valid_weights
            else:
                adjusted_weights = valid_weights

            # Apply adjusted weights
            merged_tensor = None
            for value, weight in zip(values, adjusted_weights):
                if merged_tensor is None:
                    merged_tensor = value * weight
                else:
                    merged_tensor = merged_tensor + value * weight

            if merged_tensor is not None:
                merged_model[key] = merged_tensor

        return merged_model

    def _merge_smart_fusion(
        self,
        models: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Smart fusion using layer importance analysis."""
        # Calculate smart fusion weights
        layer_weights = self.smart_fusion_weights(models, weights)

        merged_model = {}
        all_keys = set()
        for model in models:
            all_keys.update(model.keys())

        for key in all_keys:
            if key in layer_weights:
                key_weights = layer_weights[key]
                merged_tensor = None

                for i, (model, weight) in enumerate(zip(models, key_weights)):
                    if key in model:
                        if merged_tensor is None:
                            merged_tensor = model[key] * weight
                        else:
                            merged_tensor = merged_tensor + model[key] * weight

                if merged_tensor is not None:
                    merged_model[key] = merged_tensor

        return merged_model

    def _merge_min_max_norm(
        self,
        models: List[Dict[str, torch.Tensor]],
        weights: List[float],
        norm_threshold: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Fusion with min-max normalization to prevent extreme values."""
        merged_model = {}
        all_keys = set()
        for model in models:
            all_keys.update(model.keys())

        for key in all_keys:
            # Get all values for this layer
            values = []
            valid_weights = []

            for model, weight in zip(models, weights):
                if key in model:
                    values.append(model[key])
                    valid_weights.append(weight)

            if not values:
                continue

            # Apply weighted sum
            merged_tensor = None
            for value, weight in zip(values, valid_weights):
                if merged_tensor is None:
                    merged_tensor = value * weight
                else:
                    merged_tensor = merged_tensor + value * weight

            if merged_tensor is not None:
                # Apply min-max scaling to prevent extreme values (this is not weight normalization)
                min_val = merged_tensor.min()
                max_val = merged_tensor.max()

                if max_val > min_val:
                    # Scale to [-norm_threshold, norm_threshold]
                    scaled = 2 * norm_threshold * (merged_tensor - min_val) / (max_val - min_val) - norm_threshold
                    merged_model[key] = scaled
                else:
                    merged_model[key] = merged_tensor

        return merged_model

    def save_model(
        self,
        model: Dict[str, torch.Tensor],
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
        for key, value in model.items():
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


def main():
    parser = argparse.ArgumentParser(
        description="Advanced merge tool for multiple LoKr/LoHa models"
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
        "--strategy",
        choices=[s.value for s in FusionStrategy],
        default="smart_fusion",
        help="Fusion strategy to use"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output model path",
        default="./merged_lycoris_advanced.safetensors",
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
    parser.add_argument(
        "--norm-threshold",
        type=float,
        default=1.0,
        help="Normalization threshold for min_max_norm strategy"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.weights and len(args.weights) != len(args.models):
        print(f"Error: Number of weights ({len(args.weights)}) must match number of models ({len(args.models)})")
        return 1

    # Set default weights if not provided
    if not args.weights:
        args.weights = [1.0] * len(args.models)

    print(f"Loading {len(args.models)} models...")

    # Initialize merger
    merger = AdvancedLycorisMerger(device=args.device)

    # Load all models
    models = []
    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return 1

        try:
            model = merger.load_model(model_path)
            models.append(model)
            model_type = merger.get_model_type(model)
            print(f"  Loaded {model_path} ({model_type})")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return 1

    # Merge models using specified strategy
    print(f"\nMerging models using {args.strategy} strategy...")
    try:
        if args.strategy == "min_max_norm":
            merged_model = merger.merge_with_strategy(
                models, args.weights,
                FusionStrategy.MIN_MAX_NORM,
                norm_threshold=args.norm_threshold
            )
        else:
            merged_model = merger.merge_with_strategy(
                models, args.weights,
                FusionStrategy(args.strategy)
            )
    except Exception as e:
        print(f"Error during merging: {e}")
        return 1

    # Save merged model
    print(f"\nSaving merged model...")
    try:
        merger.save_model(merged_model, args.output, args.dtype)
    except Exception as e:
        print(f"Error saving merged model: {e}")
        return 1

    # Save metadata if requested
    if args.metadata:
        try:
            metadata = {
                "merge_info": {
                    "strategy": args.strategy,
                    "num_models": len(args.models),
                    "models": args.models,
                    "weights": args.weights,
                    "output_path": args.output,
                    "device": args.device,
                    "dtype": args.dtype
                },
                "model_types": [merger.get_model_type(model) for model in models]
            }

            if args.strategy == "min_max_norm":
                metadata["merge_info"]["norm_threshold"] = args.norm_threshold

            with open(args.metadata, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"Saved merge metadata: {args.metadata}")
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

    # Verify merged model if requested
    if args.verify:
        try:
            print("\n=== Model Verification ===")
            analysis = merger.analyze_model_weights(merged_model)

            print("Merged model statistics:")
            for key, stats in analysis.items():
                print(f"  {key}: shape={stats['shape']}, "
                      f"mean={stats['mean']:.6f}, "
                      f"std={stats['std']:.6f}, "
                      f"min={stats['min']:.6f}, "
                      f"max={stats['max']:.6f}, "
                      f"norm={stats['norm']:.6f}")
        except Exception as e:
            print(f"Warning: Could not verify merged model: {e}")

    print(f"\nMerge completed successfully!")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output}")
    print(f"Total parameters: {sum(p.numel() for p in merged_model.values() if torch.is_tensor(p)):,}")

    return 0


if __name__ == "__main__":
    exit(main())
