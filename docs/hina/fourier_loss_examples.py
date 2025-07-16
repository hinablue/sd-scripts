#!/usr/bin/env python3
"""
Fourier Loss 實用範例配置
Practical Example Configurations for Fourier Loss

此文件包含各種應用場景的 Fourier Loss 配置範例，包括最新的整合型配置
This file contains Fourier Loss configuration examples for various application scenarios, including the latest unified configurations
"""

from library.fourier_loss import apply_fourier_loss_to_args, get_fourier_loss_config


# =============================================================================
# 🌟 最新整合模式範例 Latest Unified Mode Examples
# =============================================================================

def apply_unified_basic_config(args):
    """
    基礎整合配置 - 輕量級，資源友好
    適用於：快速測試、資源受限環境
    """
    apply_fourier_loss_to_args(args, mode="unified_basic")
    print("📱 已應用基礎整合 Fourier Loss 配置")
    print("   - 模式: unified_basic")
    print("   - 權重: 0.03")
    print("   - 特點: 輕量級，適合快速測試")


def apply_unified_balanced_config(args):
    """
    平衡整合配置 - 效果與效率的最佳平衡 ⭐ 推薦
    適用於：日常訓練、大多數應用場景
    """
    apply_fourier_loss_to_args(args, mode="unified_balanced")
    print("🎯 已應用平衡整合 Fourier Loss 配置 (推薦)")
    print("   - 模式: unified_balanced")
    print("   - 權重: 0.06")
    print("   - 特點: 平衡效果與效率，適合大多數場景")


def apply_unified_detail_config(args):
    """
    細節增強配置 - 最高品質，細節豐富
    適用於：高品質生成、超分辨率、細節重建
    """
    apply_fourier_loss_to_args(args, mode="unified_detail")
    print("🔍 已應用細節增強 Fourier Loss 配置")
    print("   - 模式: unified_detail")
    print("   - 權重: 0.08")
    print("   - 特點: 高品質細節，三尺度處理")


def apply_unified_adaptive_config(args):
    """
    自適應策略配置 - 智能調整，策略靈活
    適用於：長期訓練、復雜場景、需要動態調整的任務
    """
    apply_fourier_loss_to_args(args, mode="unified_adaptive")
    print("🧠 已應用自適應策略 Fourier Loss 配置")
    print("   - 模式: unified_adaptive")
    print("   - 權重: 0.07")
    print("   - 特點: 智能自適應，指數衰減策略")


def apply_custom_unified_config(args):
    """
    自定義整合配置 - 完全控制
    適用於：特殊需求、高級用戶
    """
    # 自定義配置
    args.loss_type = "fourier"
    args.fourier_mode = "unified"
    args.fourier_weight = 0.08
    args.fourier_warmup_steps = 300

    # 整合模式特定參數
    args.enable_multiscale = True
    args.enable_frequency_weighting = True
    args.enable_adaptive = True
    args.scales = [1, 2, 4]
    args.adaptive_mode = "cosine"
    args.max_weight = 2.8
    args.min_weight = 0.8
    args.multiscale_weight = 0.7
    args.weighted_weight = 0.3

    print("🔧 已應用自定義整合 Fourier Loss 配置")
    print("   - 模式: unified (自定義)")
    print("   - 尺度: [1, 2, 4]")
    print("   - 自適應: cosine")
    print("   - 權重比例: 多尺度(0.7) + 加權(0.3)")


# =============================================================================
# 經典模式範例 Classic Mode Examples
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
    推薦 Fourier Loss 配置（經典模式）
    適用於大多數訓練場景
    """
    apply_fourier_loss_to_args(args, mode="balanced")
    print("⭐ 已應用推薦 Fourier Loss 配置（經典）")


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
    特點：平衡細節與穩定性
    """
    # 日常圖像生成 - 使用平衡整合模式
    apply_fourier_loss_to_args(args, mode="unified_balanced")

    # 微調參數
    args.fourier_weight = 0.05  # 稍微保守，確保穩定
    args.fourier_warmup_steps = 300

    print("🎨 圖像生成配置已應用")
    print("   - 重點：平衡細節與訓練穩定性")
    print("   - 權重：0.05 (保守)")


def configure_for_super_resolution(args):
    """
    超分辨率專用配置
    特點：強調細節和邊緣重建
    """
    # 使用細節增強模式
    apply_fourier_loss_to_args(args, mode="unified_detail")

    # 針對超分辨率優化
    args.fourier_weight = 0.10  # 更高權重強調細節
    args.fourier_scales = [1, 2, 4, 8]  # 更多尺度
    args.fourier_high_freq_weight = 2.8  # 強調高頻

    print("🔍 超分辨率配置已應用")
    print("   - 重點：細節重建和邊緣銳化")
    print("   - 尺度：[1, 2, 4, 8]")
    print("   - 高頻權重：2.8")


def configure_for_style_transfer(args):
    """
    風格轉換專用配置
    特點：自適應策略，保持內容細節
    """
    # 使用自適應策略模式
    apply_fourier_loss_to_args(args, mode="unified_adaptive")

    # 風格轉換優化
    args.fourier_weight = 0.06
    args.adaptive_mode = "cosine"  # 平滑過渡
    args.max_weight = 2.5
    args.min_weight = 1.0

    print("🎭 風格轉換配置已應用")
    print("   - 重點：內容保持與風格遷移平衡")
    print("   - 自適應：餘弦模式")


def configure_for_image_restoration(args):
    """
    圖像修復專用配置
    特點：最高品質細節恢復
    """
    # 使用自定義配置以獲得最大控制
    args.loss_type = "fourier"
    args.fourier_mode = "unified"
    args.fourier_weight = 0.12  # 較高權重

    # 修復任務特定設置
    args.enable_multiscale = True
    args.enable_frequency_weighting = True
    args.enable_adaptive = True
    args.scales = [1, 2, 4]
    args.freq_weight_per_scale = [2.0, 2.5, 3.0]  # 遞增權重
    args.adaptive_mode = "linear"
    args.max_weight = 3.0
    args.min_weight = 1.5
    args.multiscale_weight = 0.8  # 強調多尺度

    print("🖼️ 圖像修復配置已應用")
    print("   - 重點：最高品質細節恢復")
    print("   - 權重：0.12 (較高)")
    print("   - 特殊：遞增尺度權重")


def configure_for_quick_prototyping(args):
    """
    快速原型開發配置
    特點：快速測試，低資源消耗
    """
    apply_fourier_loss_to_args(args, mode="unified_basic")

    # 原型開發優化
    args.fourier_weight = 0.02  # 低權重，快速收斂
    args.fourier_warmup_steps = 100  # 短預熱

    print("⚡ 快速原型配置已應用")
    print("   - 重點：快速測試和驗證")
    print("   - 資源：低消耗模式")


# =============================================================================
# 硬件配置優化 Hardware-Optimized Configurations
# =============================================================================

def configure_for_limited_memory(args):
    """
    記憶體受限環境配置
    適用於：GPU 記憶體 < 8GB
    """
    apply_fourier_loss_to_args(args, mode="unified_basic")

    # 記憶體優化
    args.fourier_weight = 0.04
    args.scales = [1, 2]  # 減少尺度
    args.enable_multiscale = True  # 保持功能但減少複雜度

    print("💾 記憶體優化配置已應用")
    print("   - 適用：< 8GB GPU 記憶體")
    print("   - 優化：減少尺度數量")


def configure_for_high_performance(args):
    """
    高性能環境配置
    適用於：GPU 記憶體 >= 24GB，追求最佳效果
    """
    # 使用細節增強模式
    apply_fourier_loss_to_args(args, mode="unified_detail")

    # 高性能優化
    args.fourier_weight = 0.10
    args.scales = [1, 2, 4, 8]  # 全尺度
    args.freq_weight_per_scale = [1.8, 2.2, 2.6, 3.0]
    args.adaptive_mode = "cosine"

    print("🚀 高性能配置已應用")
    print("   - 適用：>= 24GB GPU 記憶體")
    print("   - 特點：全尺度處理，最佳效果")


# =============================================================================
# 動態配置範例 Dynamic Configuration Examples
# =============================================================================

def apply_progressive_fourier_config(args, current_epoch, total_epochs):
    """
    漸進式 Fourier Loss 配置
    根據訓練進度動態調整配置
    """
    progress = current_epoch / total_epochs

    if progress < 0.3:  # 早期階段 (0-30%)
        apply_fourier_loss_to_args(args, mode="unified_balanced")
        args.fourier_weight = 0.08  # 較高權重，學習細節
        print(f"🌱 早期階段配置 (進度: {progress:.1%})")

    elif progress < 0.7:  # 中期階段 (30-70%)
        apply_fourier_loss_to_args(args, mode="unified_adaptive")
        args.fourier_weight = 0.06  # 中等權重
        print(f"🌿 中期階段配置 (進度: {progress:.1%})")

    else:  # 後期階段 (70-100%)
        apply_fourier_loss_to_args(args, mode="unified_detail")
        args.fourier_weight = 0.04  # 較低權重，精細調整
        print(f"🌳 後期階段配置 (進度: {progress:.1%})")


def apply_adaptive_by_loss(args, current_loss, target_loss):
    """
    基於損失值的自適應配置
    根據當前損失動態調整策略
    """
    loss_ratio = current_loss / target_loss

    if loss_ratio > 2.0:  # 損失較高，需要更多指導
        apply_fourier_loss_to_args(args, mode="unified_detail")
        args.fourier_weight = 0.10
        print(f"📈 高損失模式 (比例: {loss_ratio:.2f})")

    elif loss_ratio > 1.2:  # 損失適中
        apply_fourier_loss_to_args(args, mode="unified_balanced")
        args.fourier_weight = 0.06
        print(f"📊 標準模式 (比例: {loss_ratio:.2f})")

    else:  # 損失較低，接近目標
        apply_fourier_loss_to_args(args, mode="unified_adaptive")
        args.fourier_weight = 0.04
        print(f"📉 精細調整模式 (比例: {loss_ratio:.2f})")


# =============================================================================
# 比較測試範例 Comparison Test Examples
# =============================================================================

def run_configuration_comparison(args):
    """
    運行配置比較測試
    幫助選擇最適合的配置
    """
    print("🔬 開始配置比較測試...")

    configs_to_test = [
        ("unified_basic", "基礎整合"),
        ("unified_balanced", "平衡整合"),
        ("unified_detail", "細節增強"),
        ("unified_adaptive", "自適應策略")
    ]

    for mode, name in configs_to_test:
        print(f"\n📋 測試配置: {name} ({mode})")

        # 備份原始配置
        original_mode = getattr(args, 'fourier_mode', None)
        original_weight = getattr(args, 'fourier_weight', None)

        # 應用測試配置
        apply_fourier_loss_to_args(args, mode=mode)

        # 這裡可以添加實際的測試邏輯
        print(f"   - 權重: {args.fourier_weight}")
        print(f"   - 預熱步數: {args.fourier_warmup_steps}")
        print(f"   - 建議使用場景: {get_use_case_for_mode(mode)}")

        # 恢復原始配置（如果需要）
        if original_mode:
            args.fourier_mode = original_mode
        if original_weight:
            args.fourier_weight = original_weight


def get_use_case_for_mode(mode):
    """獲取模式的適用場景"""
    use_cases = {
        "unified_basic": "快速測試、資源受限",
        "unified_balanced": "日常訓練、通用場景",
        "unified_detail": "高品質生成、細節重建",
        "unified_adaptive": "長期訓練、復雜場景"
    }
    return use_cases.get(mode, "未知")


# =============================================================================
# 實用工具函數 Utility Functions
# =============================================================================

def print_current_fourier_config(args):
    """
    打印當前的 Fourier Loss 配置
    """
    print("\n📋 當前 Fourier Loss 配置:")
    print(f"   損失類型: {getattr(args, 'loss_type', 'N/A')}")
    print(f"   Fourier 模式: {getattr(args, 'fourier_mode', 'N/A')}")
    print(f"   Fourier 權重: {getattr(args, 'fourier_weight', 'N/A')}")
    print(f"   預熱步數: {getattr(args, 'fourier_warmup_steps', 'N/A')}")

    # 整合模式特定參數
    if hasattr(args, 'scales'):
        print(f"   尺度: {args.scales}")
    if hasattr(args, 'adaptive_mode'):
        print(f"   自適應模式: {args.adaptive_mode}")


def validate_fourier_config(args):
    """
    驗證 Fourier Loss 配置的有效性
    """
    issues = []

    # 檢查基本配置
    if not hasattr(args, 'loss_type') or args.loss_type != 'fourier':
        issues.append("loss_type 應設置為 'fourier'")

    if not hasattr(args, 'fourier_mode'):
        issues.append("缺少 fourier_mode 設置")

    if hasattr(args, 'fourier_weight'):
        if args.fourier_weight < 0.001:
            issues.append("fourier_weight 過小，可能無效果")
        elif args.fourier_weight > 0.15:
            issues.append("fourier_weight 過大，可能不穩定")

    # 打印結果
    if issues:
        print("⚠️ 配置問題:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ 配置驗證通過")


# =============================================================================
# 範例使用 Example Usage
# =============================================================================

if __name__ == "__main__":
    # 模擬 args 對象
    class Args:
        pass

    args = Args()

    print("🎯 Fourier Loss 配置範例演示")
    print("=" * 50)

    # 演示不同配置
    print("\n1️⃣ 最新整合模式範例:")
    apply_unified_balanced_config(args)
    print_current_fourier_config(args)

    print("\n2️⃣ 應用場景配置範例:")
    configure_for_super_resolution(args)
    print_current_fourier_config(args)

    print("\n3️⃣ 配置驗證:")
    validate_fourier_config(args)

    print("\n✨ 更多範例請參考函數文檔")
    print("📚 詳細說明請查看 FOURIER_LOSS_GUIDE.md")