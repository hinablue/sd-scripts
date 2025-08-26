#!/usr/bin/env python3
"""
Example usage of LoKr/LoHa merge tools.

This script demonstrates how to use the merge tools programmatically.
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

try:
    from merge_multiple_lycoris import merge_lycoris_weights, get_model_type
    from merge_multiple_lycoris_advanced import AdvancedLycorisMerger, FusionStrategy
    print("✓ Successfully imported merge tools")
except ImportError as e:
    print(f"✗ Failed to import merge tools: {e}")
    sys.exit(1)


def create_sample_models():
    """Create sample LoKr and LoHa models for demonstration."""
    print("Creating sample models...")

    # Create LoKr model 1 (style-focused)
    lokr_style = {}
    lokr_style["alpha"] = torch.tensor(1.0)
    lokr_style["lora_te_model1_lokr_w1_a"] = torch.randn(4, 768) * 0.1
    lokr_style["lora_te_model1_lokr_w1_b"] = torch.randn(768, 4) * 0.1
    lokr_style["lora_te_model1_lokr_w2_a"] = torch.randn(4, 768) * 0.1
    lokr_style["lora_te_model1_lokr_w2_b"] = torch.randn(768, 4) * 0.1
    lokr_style["lora_unet_model1_lokr_w1_a"] = torch.randn(4, 1280) * 0.1
    lokr_style["lora_unet_model1_lokr_w1_b"] = torch.randn(1280, 4) * 0.1
    lokr_style["lora_unet_model1_lokr_w2_a"] = torch.randn(4, 1280) * 0.1
    lokr_style["lora_unet_model1_lokr_w2_b"] = torch.randn(1280, 4) * 0.1

    # Create LoKr model 2 (content-focused)
    lokr_content = {}
    lokr_content["alpha"] = torch.tensor(1.0)
    lokr_content["lora_te_model2_lokr_w1_a"] = torch.randn(4, 768) * 0.15
    lokr_content["lora_te_model2_lokr_w1_b"] = torch.randn(768, 4) * 0.15
    lokr_content["lora_te_model2_lokr_w2_a"] = torch.randn(4, 768) * 0.15
    lokr_content["lora_te_model2_lokr_w2_b"] = torch.randn(768, 4) * 0.15
    lokr_content["lora_unet_model2_lokr_w1_a"] = torch.randn(4, 1280) * 0.15
    lokr_content["lora_unet_model2_lokr_w1_b"] = torch.randn(1280, 4) * 0.15
    lokr_content["lora_unet_model2_lokr_w2_a"] = torch.randn(4, 1280) * 0.15
    lokr_content["lora_unet_model2_lokr_w2_b"] = torch.randn(1280, 4) * 0.15

    # Create LoHa model (detail-focused)
    loha_detail = {}
    loha_detail["alpha"] = torch.tensor(1.0)
    loha_detail["lora_te_model3_hada_w1_a"] = torch.randn(768, 4) * 0.12
    loha_detail["lora_te_model3_hada_w1_b"] = torch.randn(4, 768) * 0.12
    loha_detail["lora_te_model3_hada_w2_a"] = torch.randn(768, 4) * 0.12
    loha_detail["lora_te_model3_hada_w2_b"] = torch.randn(4, 768) * 0.12
    loha_detail["lora_unet_model3_hada_w1_a"] = torch.randn(1280, 4) * 0.12
    loha_detail["lora_unet_model3_hada_w1_b"] = torch.randn(4, 1280) * 0.12
    loha_detail["lora_unet_model3_hada_w2_a"] = torch.randn(1280, 4) * 0.12
    loha_detail["lora_unet_model3_hada_w2_b"] = torch.randn(4, 1280) * 0.12

    print(f"✓ Created LoKr style model: {get_model_type(lokr_style)}")
    print(f"✓ Created LoKr content model: {get_model_type(lokr_content)}")
    print(f"✓ Created LoHa detail model: {get_model_type(loha_detail)}")

    return lokr_style, lokr_content, loha_detail


def demonstrate_basic_merge():
    """Demonstrate basic merge functionality."""
    print("\n=== Basic Merge Demonstration ===")

    # Create sample models
    lokr_style, lokr_content, loha_detail = create_sample_models()

    # Merge LoKr models with different weights
    print("\nMerging LoKr models (style: 0.6, content: 0.4)...")
    lokr_weights = [0.6, 0.4]
    lokr_merged = merge_lycoris_weights([lokr_style, lokr_content], lokr_weights)

    print(f"✓ LoKr merge successful, output keys: {len(lokr_merged)}")

    # Check weight application
    key = "lora_te_model1_lokr_w1_a"
    if key in lokr_merged and key in lokr_style and key in lokr_content:
        expected = lokr_style[key] * 0.6 + lokr_content[key] * 0.4
        diff = torch.abs(lokr_merged[key] - expected).max().item()
        print(f"✓ Weight application verified (max diff: {diff:.2e})")

    return lokr_merged


def demonstrate_advanced_merge():
    """Demonstrate advanced merge functionality."""
    print("\n=== Advanced Merge Demonstration ===")

    # Create sample models
    lokr_style, lokr_content, loha_detail = create_sample_models()

    # Initialize advanced merger
    merger = AdvancedLycorisMerger()

    # Test different fusion strategies
    strategies = [
        ("Weighted Average", FusionStrategy.WEIGHTED_AVERAGE),
        ("Layer Adaptive", FusionStrategy.LAYER_ADAPTIVE),
        ("Smart Fusion", FusionStrategy.SMART_FUSION)
    ]

    models = [lokr_style, lokr_content, loha_detail]
    weights = [0.5, 0.3, 0.2]

    for strategy_name, strategy in strategies:
        print(f"\nTesting {strategy_name} strategy...")
        try:
            merged = merger.merge_with_strategy(models, weights, strategy)
            print(f"✓ {strategy_name} successful, output keys: {len(merged)}")

            # Analyze merged model
            analysis = merger.analyze_model_weights(merged)
            total_params = sum(stats["numel"] for stats in analysis.values())
            print(f"  Total parameters: {total_params:,}")

        except Exception as e:
            print(f"✗ {strategy_name} failed: {e}")

    return merger


def demonstrate_file_operations():
    """Demonstrate file save/load operations."""
    print("\n=== File Operations Demonstration ===")

    # Create a sample merged model
    lokr_style, lokr_content, _ = create_sample_models()
    merged = merge_lycoris_weights([lokr_style, lokr_content], [0.7, 0.3])

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        merger = AdvancedLycorisMerger()

        # Test safetensors save/load
        safetensors_path = os.path.join(temp_dir, "merged_model.safetensors")
        try:
            merger.save_model(merged, safetensors_path, "float16")
            print(f"✓ Saved model to: {safetensors_path}")

            # Load and verify
            loaded_model = merger.load_model(safetensors_path)
            if len(loaded_model) == len(merged):
                print("✓ Loaded model successfully")

                # Check data integrity
                key = list(merged.keys())[0]
                if key in loaded_model:
                    diff = torch.abs(merged[key] - loaded_model[key]).max().item()
                    print(f"✓ Data integrity verified (max diff: {diff:.2e})")
            else:
                print("✗ Loaded model size mismatch")

        except Exception as e:
            print(f"✗ Safetensors operation failed: {e}")

        # Test .pt save/load
        pt_path = os.path.join(temp_dir, "merged_model.pt")
        try:
            merger.save_model(merged, pt_path, "float16")
            print(f"✓ Saved model to: {pt_path}")

            # Load and verify
            loaded_model = merger.load_model(pt_path)
            if len(loaded_model) == len(merged):
                print("✓ Loaded PT model successfully")
            else:
                print("✗ Loaded PT model size mismatch")

        except Exception as e:
            print(f"✗ PT operation failed: {e}")


def demonstrate_compatibility_check():
    """Demonstrate model compatibility checking."""
    print("\n=== Compatibility Check Demonstration ===")

    from merge_multiple_lycoris import validate_models_compatibility

    # Create compatible models
    lokr_style, lokr_content, _ = create_sample_models()

    # Check compatibility
    msg, is_compatible = validate_models_compatibility([lokr_style, lokr_content])
    print(f"Compatibility check: {msg}")
    print(f"Models are compatible: {is_compatible}")

    # Create incompatible model (different structure)
    incompatible_model = lokr_style.copy()
    incompatible_model["extra_key"] = torch.tensor(1.0)

    msg, is_compatible = validate_models_compatibility([lokr_style, incompatible_model])
    print(f"Compatibility check with extra key: {msg}")
    print(f"Models are compatible: {is_compatible}")


def demonstrate_weight_analysis():
    """Demonstrate weight analysis functionality."""
    print("\n=== Weight Analysis Demonstration ===")

    merger = AdvancedLycorisMerger()

    # Create sample models
    lokr_style, lokr_content, loha_detail = create_sample_models()

    # Analyze individual models
    print("\nAnalyzing individual models...")
    for i, (name, model) in enumerate([("LoKr Style", lokr_style), ("LoKr Content", lokr_content), ("LoHa Detail", loha_detail)]):
        analysis = merger.analyze_model_weights(model)
        importance = merger.calculate_layer_importance(model)

        print(f"\n{name}:")
        print(f"  Total layers: {len(analysis)}")
        print(f"  Total parameters: {sum(stats['numel'] for stats in analysis.values()):,}")

        # Show top 3 most important layers
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 3 important layers:")
        for j, (key, imp) in enumerate(sorted_importance[:3]):
            print(f"    {j+1}. {key}: {imp:.4f}")

    # Analyze merged model
    print("\nAnalyzing merged model...")
    merged = merger.merge_with_strategy([lokr_style, lokr_content, loha_detail], [0.5, 0.3, 0.2], FusionStrategy.SMART_FUSION)
    analysis = merger.analyze_model_weights(merged)

    print(f"  Total layers: {len(analysis)}")
    print(f"  Total parameters: {sum(stats['numel'] for stats in analysis.values()):,}")

    # Show weight statistics
    print(f"  Weight statistics:")
    for key, stats in list(analysis.items())[:3]:  # Show first 3 layers
        print(f"    {key}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, norm={stats['norm']:.6f}")


def main():
    """Run all demonstrations."""
    print("LoKr/LoHa Merge Tools - Usage Examples")
    print("=" * 50)

    try:
        # Run demonstrations
        demonstrate_basic_merge()
        demonstrate_advanced_merge()
        demonstrate_file_operations()
        demonstrate_compatibility_check()
        demonstrate_weight_analysis()

        print("\n" + "=" * 50)
        print("🎉 All demonstrations completed successfully!")
        print("\nKey takeaways:")
        print("- Basic merge: Simple weighted fusion of models")
        print("- Advanced merge: Multiple strategies with intelligent weight adjustment")
        print("- File operations: Support for safetensors and .pt formats")
        print("- Compatibility: Automatic model structure validation")
        print("- Analysis: Detailed weight statistics and importance calculation")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
