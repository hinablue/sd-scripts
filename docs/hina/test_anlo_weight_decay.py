#!/usr/bin/env python3
"""
ANLO 優化器權重衰減測試腳本

測試權重衰減是否正確地在參數更新之後應用
"""

import torch
import torch.nn as nn
import sys
import os

# 添加項目根目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from library.hina_anlo import ANLO


def test_weight_decay_order():
    """測試權重衰減的執行順序"""
    print("=== ANLO 優化器權重衰減測試 ===")

    # 創建一個簡單的模型
    model = nn.Linear(10, 5)

    # 創建 ANLO 優化器，設置較大的權重衰減以便觀察
    optimizer = ANLO(
        model.parameters(),
        lr=0.1,
        weight_decay=0.1,  # 較大的權重衰減
        normalize_frequency=10,  # 減少歸一化頻率以便觀察
        verbose=True
    )

    # 記錄初始參數值
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()

    print(f"初始參數值:")
    for name, param in initial_params.items():
        print(f"  {name}: {param.mean().item():.6f}")

    # 創建虛擬損失和梯度
    x = torch.randn(5, 10)
    target = torch.randn(5, 5)

    # 前向傳播
    output = model(x)
    loss = nn.MSELoss()(output, target)

    # 反向傳播
    loss.backward()

    # 記錄梯度值
    print(f"\n梯度值:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad.mean().item():.6f}")

    # 執行優化步驟
    print(f"\n執行優化步驟...")
    optimizer.step()

    # 記錄更新後的參數值
    print(f"\n更新後的參數值:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.data.mean().item():.6f}")

    # 計算參數變化
    print(f"\n參數變化:")
    for name, param in model.named_parameters():
        change = param.data.mean().item() - initial_params[name].mean().item()
        print(f"  {name}: {change:.6f}")

    # 驗證權重衰減是否正確應用
    print(f"\n=== 驗證結果 ===")

    # 檢查權重是否被正確衰減（應該變小）
    weight_param = model.weight
    weight_change = weight_param.data.mean().item() - initial_params['weight'].mean().item()

    if weight_change < 0:
        print("✅ 權重衰減正確應用：權重值減小")
    else:
        print("❌ 權重衰減可能未正確應用：權重值未減小")

    print(f"權重變化: {weight_change:.6f}")

    # 檢查偏置是否被正確衰減
    bias_param = model.bias
    bias_change = bias_param.data.mean().item() - initial_params['bias'].mean().item()

    if bias_change < 0:
        print("✅ 偏置衰減正確應用：偏置值減小")
    else:
        print("❌ 偏置衰減可能未正確應用：偏置值未減小")

    print(f"偏置變化: {bias_change:.6f}")

    print(f"\n測試完成！")


if __name__ == "__main__":
    test_weight_decay_order()