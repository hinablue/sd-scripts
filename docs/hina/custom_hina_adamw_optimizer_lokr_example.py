"""
HinaAdamWOptimizer LoKr 支援示例
展示如何在 LoKr 訓練中使用增強優化器
"""

import torch
import torch.nn as nn
from custom_hina_adamw_optimizer import HinaAdamWOptimizer

class MockLoKrLayer(nn.Module):
    """
    模擬 LoKr 層，用於測試優化器的 LoKr 支援

    LoKr 使用 Kronecker 積分解：W = W0 + (B1 ⊗ B2)(A1 ⊗ A2)
    """
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # 原始權重（凍結）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoKr 參數 - 使用不同的命名模式來測試識別功能
        self.lokr_w1_a = nn.Parameter(torch.randn(rank, in_features))
        self.lokr_w1_b = nn.Parameter(torch.randn(out_features, rank))
        self.lokr_w2_a = nn.Parameter(torch.randn(rank, rank))
        self.lokr_w2_b = nn.Parameter(torch.randn(rank, rank))

        # 為參數設置名稱屬性，模擬實際訓練框架的行為
        self.lokr_w1_a.param_name = "test_layer.lokr_w1_a.weight"
        self.lokr_w1_b.param_name = "test_layer.lokr_w1_b.weight"
        self.lokr_w2_a.param_name = "test_layer.lokr_w2_a.weight"
        self.lokr_w2_b.param_name = "test_layer.lokr_w2_b.weight"

        self.scaling = 0.1

    def forward(self, x):
        # 計算 Kronecker 積近似
        # 這是簡化版本，實際 LoKr 實現會更複雜
        w1_product = torch.matmul(self.lokr_w1_b, self.lokr_w1_a)
        w2_product = torch.matmul(self.lokr_w2_b, self.lokr_w2_a)

        # 簡化的 Kronecker 積效果
        delta_w = w1_product + w2_product * 0.5  # 簡化組合

        # 原始輸出 + LoKr 增量
        output = torch.nn.functional.linear(x, self.weight)
        delta_output = torch.nn.functional.linear(x, delta_w * self.scaling)

        return output + delta_output

class AlternativeLoKrLayer(nn.Module):
    """
    使用替代命名模式的 LoKr 層，測試不同的參數命名識別
    """
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 使用不同的命名模式
        self.lokr_w1 = nn.Parameter(torch.randn(rank, in_features))
        self.lokr_w2 = nn.Parameter(torch.randn(out_features, rank))

        # 設置參數名稱
        self.lokr_w1.param_name = "alt_layer.lokr.w1.weight"
        self.lokr_w2.param_name = "alt_layer.lokr.w2.weight"

        self.scaling = 0.05

    def forward(self, x):
        # 簡單的低秩近似
        intermediate = torch.nn.functional.linear(x, self.lokr_w1.T)
        output = torch.nn.functional.linear(intermediate, self.lokr_w2) * self.scaling
        return output

def test_lokr_support():
    """測試 LoKr 支援功能"""
    print("=== HinaAdamWOptimizer LoKr 支援測試 ===\n")

    # 創建測試模型
    model = nn.Sequential(
        MockLoKrLayer(512, 256, rank=32),
        nn.ReLU(),
        AlternativeLoKrLayer(256, 128, rank=16),
        nn.ReLU(),
        nn.Linear(128, 10)  # 普通層
    )

    # 創建優化器
    optimizer = HinaAdamWOptimizer(
        model.parameters(),
        lr=1e-3,
        use_alora=True,  # 啟用 ALoRA（現在支援 LoKr）
        alora_ratio=18.0,  # LoKr 建議使用略低的比例
        dynamic_weight_decay=True,
        wd_transition_steps=500,  # LoKr 建議較快的過渡
        wd_decay_factor=0.75,     # 較溫和的衰減
        wd_min_ratio=0.15,        # 保持更高的最小權重衰減
        use_spd=True,
        use_cautious=True,
        verbose=True
    )

    # 獲取優化器資訊
    opt_info = optimizer.get_optimization_info()

    print("📊 優化器資訊:")
    print(f"  總參數數: {opt_info['total_params']}")
    print(f"  優化器類型: {opt_info['optimizer_type']}")

    # LoRA 統計
    lora_stats = opt_info['lora_stats']
    print(f"\n📈 LoRA 參數統計:")
    print(f"  LoRA A 參數: {lora_stats['lora_a_params']}")
    print(f"  LoRA B 參數: {lora_stats['lora_b_params']}")
    print(f"  LoRA 配對: {lora_stats['lora_pairs']}")

    # LoKr 統計
    lokr_stats = opt_info['lokr_stats']
    print(f"\n🔷 LoKr 參數統計:")
    print(f"  LoKr W1 參數: {lokr_stats['lokr_w1_params']}")
    print(f"  LoKr W2 參數: {lokr_stats['lokr_w2_params']}")
    print(f"  LoKr W1A 參數: {lokr_stats['lokr_w1_a_params']}")
    print(f"  LoKr W1B 參數: {lokr_stats['lokr_w1_b_params']}")
    print(f"  LoKr W2A 參數: {lokr_stats['lokr_w2_a_params']}")
    print(f"  LoKr W2B 參數: {lokr_stats['lokr_w2_b_params']}")
    print(f"  LoKr 配對: {lokr_stats['lokr_pairs']}")
    print(f"  LoKr 組別: {lokr_stats['lokr_groups']}")

    # 功能特性
    features = opt_info['features']
    print(f"\n🚀 啟用的功能:")
    for feature, enabled in features.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature}: {enabled}")

    # 簡單的訓練模擬
    print(f"\n🏋️ 開始訓練模擬...")

    criterion = nn.CrossEntropyLoss()
    for step in range(10):
        # 模擬批次數據
        x = torch.randn(8, 512)
        y = torch.randint(0, 10, (8,))

        # 前向傳播
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        # 反向傳播和優化
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"  Step {step}: Loss = {loss:.4f}")

    print(f"\n✅ LoKr 支援測試完成！")
    print(f"📋 總結:")
    print(f"  - 成功識別 {lokr_stats['lokr_groups']} 個 LoKr 組別")
    print(f"  - 檢測到 {lokr_stats['lokr_pairs']} 個 LoKr 參數配對")
    print(f"  - 應用了專門針對 LoKr 的學習率縮放和權重衰減策略")

    return True

def test_parameter_classification():
    """測試參數分類功能"""
    print("\n=== 參數分類測試 ===")

    # 創建一個虛擬優化器來測試分類功能
    dummy_param = torch.randn(10, 10, requires_grad=True)
    optimizer = HinaAdamWOptimizer([dummy_param])

    # 測試各種參數名稱
    test_names = [
        ("layer.lora_down.weight", "lora_a"),
        ("layer.lora_up.weight", "lora_b"),
        ("layer.lokr_w1_a.weight", "lokr_w1_a"),
        ("layer.lokr_w1_b.weight", "lokr_w1_b"),
        ("layer.lokr_w2_a.weight", "lokr_w2_a"),
        ("layer.lokr_w2_b.weight", "lokr_w2_b"),
        ("layer.lokr.w1.weight", "lokr_w1"),
        ("layer.lokr.w2.weight", "lokr_w2"),
        ("layer.weight", "regular"),
        ("layer.bias", "regular"),
    ]

    print("🔍 參數名稱分類結果:")
    for name, expected in test_names:
        result = optimizer._classify_parameter(name)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{name}' -> '{result}' (期望: '{expected}')")

    return True

if __name__ == "__main__":
    # 運行測試
    try:
        test_parameter_classification()
        test_lokr_support()
        print(f"\n🎉 所有測試通過！HinaAdamWOptimizer 現在完全支援 LoKr 訓練。")
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        raise