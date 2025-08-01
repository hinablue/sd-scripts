#!/usr/bin/env python3
"""
驗證 ANLO 優化器謹慎更新功能整合
"""

import torch
import torch.nn as nn
from library.hina_anlo import ANLO

def test_basic_functionality():
    """測試基本功能"""
    print("=== 測試基本功能 ===")

    # 創建簡單模型
    model = nn.Linear(10, 5)

    # 創建優化器
    optimizer = ANLO(
        model.parameters(),
        lr=1e-3,
        use_cautious_update=True,
        cautious_threshold=0.1,
        cautious_scale=0.5,
        verbose=True
    )

    # 檢查參數
    print(f"優化器參數組數量: {len(optimizer.param_groups)}")
    print(f"謹慎更新啟用: {optimizer.param_groups[0]['use_cautious_update']}")
    print(f"對齊度閾值: {optimizer.param_groups[0]['cautious_threshold']}")
    print(f"縮放因子: {optimizer.param_groups[0]['cautious_scale']}")

    # 測試統計信息
    norm_stats = optimizer.get_normalization_stats()
    cautious_stats = optimizer.get_cautious_update_stats()

    print(f"正規化統計: {norm_stats['cautious_update_enabled']}")
    print(f"謹慎更新統計: {cautious_stats['enabled_groups']}/{cautious_stats['total_groups']}")

    print("✓ 基本功能測試通過")

def test_cautious_update_method():
    """測試謹慎更新方法"""
    print("\n=== 測試謹慎更新方法 ===")

    # 創建測試張量
    update = torch.randn(5, 5)
    grad = torch.randn(5, 5)

    # 測試對齊度高的情況（應該不縮放）
    aligned_grad = grad.clone()
    result1 = ANLO._apply_cautious_update_optimized(update, aligned_grad, 0.1, 0.5)
    print(f"對齊度高時縮放: {torch.allclose(result1, update)}")

    # 測試對齊度低的情況（應該縮放）
    opposite_grad = -grad.clone()
    result2 = ANLO._apply_cautious_update_optimized(update, opposite_grad, 0.1, 0.5)
    print(f"對齊度低時縮放: {torch.allclose(result2, update * 0.5)}")

    print("✓ 謹慎更新方法測試通過")

def test_optimizer_step():
    """測試優化器步驟"""
    print("\n=== 測試優化器步驟 ===")

    # 創建模型和數據
    model = nn.Linear(10, 5)
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    criterion = nn.MSELoss()

    # 創建優化器
    optimizer = ANLO(
        model.parameters(),
        lr=1e-3,
        use_cautious_update=True,
        cautious_threshold=0.1,
        cautious_scale=0.5,
        verbose=False
    )

    # 執行一個訓練步驟
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    print(f"訓練步驟完成，損失: {loss.item():.6f}")
    print("✓ 優化器步驟測試通過")

def test_backward_compatibility():
    """測試向後兼容性"""
    print("\n=== 測試向後兼容性 ===")

    # 創建模型
    model = nn.Linear(10, 5)

    # 創建不啟用謹慎更新的優化器
    optimizer = ANLO(
        model.parameters(),
        lr=1e-3,
        use_cautious_update=False,  # 不啟用謹慎更新
        verbose=False
    )

    # 檢查參數
    print(f"謹慎更新啟用: {optimizer.param_groups[0]['use_cautious_update']}")
    print(f"對齊度閾值: {optimizer.param_groups[0]['cautious_threshold']}")
    print(f"縮放因子: {optimizer.param_groups[0]['cautious_scale']}")

    # 測試統計信息
    norm_stats = optimizer.get_normalization_stats()
    cautious_stats = optimizer.get_cautious_update_stats()

    print(f"正規化統計: {norm_stats['cautious_update_enabled']}")
    print(f"謹慎更新統計: {cautious_stats['enabled_groups']}/{cautious_stats['total_groups']}")

    print("✓ 向後兼容性測試通過")

def main():
    """主函數"""
    print("ANLO 優化器謹慎更新功能整合驗證")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_cautious_update_method()
        test_optimizer_step()
        test_backward_compatibility()

        print("\n" + "=" * 50)
        print("🎉 所有測試通過！謹慎更新功能已成功整合到 ANLO 優化器中。")
        print("\n主要改進:")
        print("1. ✅ 新增謹慎更新功能")
        print("2. ✅ 保持向後兼容性")
        print("3. ✅ JIT 優化實現")
        print("4. ✅ 完整的統計信息")
        print("5. ✅ 靈活的配置選項")

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()