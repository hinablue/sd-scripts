#!/usr/bin/env python3
"""
Fourier Loss 實用範例配置
Practical Example Configurations for Fourier Loss

此文件包含各種應用場景的 Fourier Loss 配置範例
This file contains Fourier Loss configuration examples for various application scenarios
"""

from library.fourier_loss import apply_fourier_loss_to_args, get_fourier_loss_config


# =============================================================================
# 基本使用範例 Basic Usage Examples
# =============================================================================

def apply_basic_fourier_config(args):
    """
    基礎 Fourier Loss 配置
    適用於初學者和快速測試
    """
    apply_fourier_loss_to_args(args, mode="conservative")
    print("✅ 已應用基礎 Fourier Loss 配置")


def apply_recommended_fourier_config(args):
    """
    推薦 Fourier Loss 配置
    適用於大多數訓練場景
    """
    apply_fourier_loss_to_args(args, mode="balanced")
    print("⭐ 已應用推薦 Fourier Loss 配置")


def apply_advanced_fourier_config(args):
    """
    進階 Fourier Loss 配置
    適用於高品質需求場景
    """
    apply_fourier_loss_to_args(args, mode="aggressive")
    print("🚀 已應用進階 Fourier Loss 配置")


# =============================================================================
# 應用場景專用配置 Application-Specific Configurations
# =============================================================================

def configure_for_image_generation(args):
    """
    圖像生成專用配置
    重點：提升細節質量和紋理豐富度
    """
    args.loss_type = "fourier"
    args.fourier_weight = 0.03
    args.fourier_mode = "weighted"
    args.fourier_high_freq_weight = 1.5
    args.fourier_warmup_steps = 150
    args.fourier_norm = "l2"

    print("🎨 已配置圖像生成專用 Fourier Loss")
    return args


def configure_for_super_resolution(args):
    """
    超分辨率專用配置
    重點：多尺度特徵學習，提升邊緣清晰度
    """
    args.loss_type = "fourier"
    args.fourier_weight = 0.08
    args.fourier_mode = "multiscale"
    args.fourier_scales = [1, 2, 4]
    args.fourier_scale_weights = [1.0, 0.7, 0.4]
    args.fourier_warmup_steps = 200
    args.fourier_norm = "l2"

    print("🔍 已配置超分辨率專用 Fourier Loss")
    return args


def configure_for_style_transfer(args):
    """
    風格轉換專用配置
    重點：自適應權重，保持細節一致性
    """
    args.loss_type = "fourier"
    args.fourier_weight = 0.05
    args.fourier_mode = "adaptive"
    args.fourier_adaptive_max_weight = 2.5
    args.fourier_adaptive_min_weight = 0.8
    args.fourier_warmup_steps = 300
    args.fourier_norm = "l1"

    print("🎭 已配置風格轉換專用 Fourier Loss")
    return args


def configure_for_image_restoration(args):
    """
    圖像修復專用配置
    重點：邊緣和紋理恢復
    """
    args.loss_type = "fourier"
    args.fourier_weight = 0.06
    args.fourier_mode = "weighted"
    args.fourier_high_freq_weight = 2.0
    args.fourier_warmup_steps = 250
    args.fourier_norm = "l2"

    print("🖼️ 已配置圖像修復專用 Fourier Loss")
    return args


# =============================================================================
# 記憶體優化配置 Memory-Optimized Configurations
# =============================================================================

def configure_for_low_memory(args):
    """
    低記憶體配置
    適用於記憶體受限的環境
    """
    args.loss_type = "fourier"
    args.fourier_weight = 0.04
    args.fourier_mode = "basic"  # 最輕量模式
    args.fourier_warmup_steps = 100
    args.fourier_norm = "l2"

    print("💾 已配置低記憶體 Fourier Loss")
    return args


def configure_for_large_images(args):
    """
    大圖像專用配置
    適用於高解析度圖像訓練
    """
    args.loss_type = "fourier"
    args.fourier_weight = 0.03  # 降低權重減少計算負擔
    args.fourier_mode = "weighted"
    args.fourier_high_freq_weight = 1.8
    args.fourier_warmup_steps = 400  # 增加預熱期
    args.fourier_norm = "l2"

    print("🖼️ 已配置大圖像專用 Fourier Loss")
    return args


# =============================================================================
# 訓練階段配置 Training Phase Configurations
# =============================================================================

def configure_for_early_training(args):
    """
    訓練初期配置
    重點：保守權重，避免訓練不穩定
    """
    args.loss_type = "fourier"
    args.fourier_weight = 0.02
    args.fourier_mode = "weighted"
    args.fourier_high_freq_weight = 1.5
    args.fourier_warmup_steps = 500  # 長預熱期
    args.fourier_norm = "l2"

    print("🌱 已配置訓練初期 Fourier Loss")
    return args


def configure_for_fine_tuning(args):
    """
    微調階段配置
    重點：提升細節質量
    """
    args.loss_type = "fourier"
    args.fourier_weight = 0.07
    args.fourier_mode = "weighted"
    args.fourier_high_freq_weight = 2.2
    args.fourier_warmup_steps = 100  # 短預熱期
    args.fourier_norm = "l2"

    print("🔧 已配置微調階段 Fourier Loss")
    return args


# =============================================================================
# 動態配置調整 Dynamic Configuration Adjustment
# =============================================================================

def adjust_fourier_weight_by_epoch(args, current_epoch, total_epochs):
    """
    根據訓練進度動態調整 Fourier 權重

    Args:
        args: 訓練參數
        current_epoch: 當前 epoch
        total_epochs: 總 epoch 數
    """
    progress = current_epoch / total_epochs

    if progress < 0.3:
        # 訓練初期：使用較小權重
        base_weight = 0.02
    elif progress < 0.7:
        # 訓練中期：逐漸增加權重
        base_weight = 0.02 + (0.05 - 0.02) * ((progress - 0.3) / 0.4)
    else:
        # 訓練後期：使用較高權重進行細節優化
        base_weight = 0.05 + (0.08 - 0.05) * ((progress - 0.7) / 0.3)

    args.fourier_weight = base_weight
    print(f"📈 Epoch {current_epoch}/{total_epochs}: Fourier weight = {base_weight:.4f}")

    return args


def adjust_fourier_weight_by_loss_ratio(args, fourier_loss, base_loss):
    """
    根據損失比例動態調整 Fourier 權重

    Args:
        args: 訓練參數
        fourier_loss: 當前 Fourier 損失值
        base_loss: 當前基礎損失值
    """
    if base_loss > 0:
        ratio = fourier_loss / base_loss

        if ratio > 5.0:
            # Fourier 損失過大，降低權重
            args.fourier_weight *= 0.8
            print(f"⬇️ Fourier 損失過大 (ratio={ratio:.2f})，降低權重至 {args.fourier_weight:.4f}")
        elif ratio < 0.1:
            # Fourier 損失過小，增加權重
            args.fourier_weight *= 1.2
            args.fourier_weight = min(args.fourier_weight, 0.15)  # 限制最大值
            print(f"⬆️ Fourier 損失過小 (ratio={ratio:.2f})，增加權重至 {args.fourier_weight:.4f}")
        else:
            print(f"✅ Fourier 損失比例正常 (ratio={ratio:.2f})")

    return args


# =============================================================================
# 配置驗證和測試 Configuration Validation and Testing
# =============================================================================

def validate_fourier_config(args):
    """
    驗證 Fourier Loss 配置是否合理

    Args:
        args: 訓練參數

    Returns:
        bool: 配置是否有效
    """
    issues = []

    # 檢查權重範圍
    if hasattr(args, 'fourier_weight'):
        if args.fourier_weight <= 0:
            issues.append("fourier_weight 必須大於 0")
        elif args.fourier_weight > 0.2:
            issues.append("fourier_weight 過大 (> 0.2)，可能導致訓練不穩定")

    # 檢查模式參數
    if hasattr(args, 'fourier_mode'):
        valid_modes = ["basic", "weighted", "multiscale", "adaptive"]
        if args.fourier_mode not in valid_modes:
            issues.append(f"fourier_mode 必須是 {valid_modes} 之一")

    # 檢查高頻權重
    if hasattr(args, 'fourier_high_freq_weight'):
        if args.fourier_high_freq_weight < 1.0:
            issues.append("fourier_high_freq_weight 不能小於 1.0")
        elif args.fourier_high_freq_weight > 5.0:
            issues.append("fourier_high_freq_weight 過大 (> 5.0)")

    # 檢查尺度設置
    if hasattr(args, 'fourier_scales') and args.fourier_scales:
        if 1 not in args.fourier_scales:
            issues.append("fourier_scales 應該包含原始尺度 (1)")

    if issues:
        print("❌ 配置驗證失敗:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ 配置驗證通過")
        return True


def print_fourier_config(args):
    """
    打印當前 Fourier Loss 配置

    Args:
        args: 訓練參數
    """
    print("\n📋 當前 Fourier Loss 配置:")
    print("=" * 40)

    fourier_attrs = [
        'fourier_weight', 'fourier_mode', 'fourier_norm',
        'fourier_high_freq_weight', 'fourier_scales', 'fourier_scale_weights',
        'fourier_adaptive_max_weight', 'fourier_adaptive_min_weight',
        'fourier_eps', 'fourier_warmup_steps'
    ]

    for attr in fourier_attrs:
        if hasattr(args, attr):
            value = getattr(args, attr)
            print(f"{attr:25}: {value}")

    print("=" * 40)


# =============================================================================
# 使用範例 Usage Examples
# =============================================================================

if __name__ == "__main__":
    # 這是一個使用範例，展示如何使用上述配置函數

    class MockArgs:
        """模擬訓練參數對象"""
        pass

    # 創建模擬參數對象
    args = MockArgs()

    print("🚀 Fourier Loss 配置範例")
    print("=" * 50)

    # 範例 1: 圖像生成配置
    print("\n1. 圖像生成配置範例:")
    configure_for_image_generation(args)
    print_fourier_config(args)
    validate_fourier_config(args)

    # 範例 2: 超分辨率配置
    print("\n2. 超分辨率配置範例:")
    args = MockArgs()  # 重新初始化
    configure_for_super_resolution(args)
    print_fourier_config(args)
    validate_fourier_config(args)

    # 範例 3: 動態權重調整
    print("\n3. 動態權重調整範例:")
    args = MockArgs()
    configure_for_image_generation(args)

    # 模擬不同訓練階段
    for epoch in [1, 10, 20, 30]:
        adjust_fourier_weight_by_epoch(args, epoch, 30)

    print("\n✅ 所有範例執行完成!")