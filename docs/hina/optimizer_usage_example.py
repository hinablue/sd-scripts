"""
改進版 Automagic_CameAMP 優化器使用範例
專門針對 LoRA 訓練和減少邊緣、背景過擬合問題
"""

import torch
import torch.nn as nn
import math
from automagic_cameamp_improved import Automagic_CameAMP_Improved, ImprovedOptimizerConfig

def create_lora_model():
    """創建一個簡單的 LoRA 模型範例."""
    class SimpleLoRA(nn.Module):
        def __init__(self, in_features=512, out_features=512, rank=16):
            super().__init__()
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            self.scaling = 0.1

            # 初始化 LoRA 權重
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            return self.lora_B(self.lora_A(x)) * self.scaling

    return SimpleLoRA()

def main():
    """主要使用範例."""

    # === 1. 創建模型 ===
    model = create_lora_model()
    print("✅ 模型創建完成")

    # === 2. 配置改進版優化器 ===
    # 針對 LoRA 訓練和減少過擬合的優化配置
    config = ImprovedOptimizerConfig(
        # 基本學習率設定
        lr=1e-4,                    # 基礎學習率
        min_lr=1e-6,               # 最小學習率
        max_lr=1e-3,               # 最大學習率
        lr_bump=5e-6,              # 學習率調整幅度

        # 邊緣和背景過擬合控制
        edge_suppression=True,      # 啟用邊緣抑制
        edge_penalty=0.15,         # 邊緣懲罰強度（建議 0.1-0.2）
        edge_threshold=0.5,        # 邊緣檢測閾值
        background_regularization=True,  # 啟用背景正則化
        spatial_awareness=True,     # 啟用空間感知
        frequency_penalty=0.08,    # 頻率懲罰強度
        detail_preservation=0.85,   # 細節保存因子

        # LoRA 特定優化
        lora_rank_penalty=True,     # 啟用 LoRA 低秩懲罰
        rank_penalty_strength=0.02, # 低秩懲罰強度
        low_rank_emphasis=1.3,      # 低秩方向強調因子

        # 其他重要參數
        warmup_steps=800,          # Warmup 步數（針對 LoRA 可以適當增加）
        weight_decay=1e-4,         # 權重衰減
        came=True,                 # 啟用 CAME 優化
        full_finetune=False,       # LoRA 模式
        verbose=True               # 顯示詳細信息
    )

    # === 3. 創建優化器 ===
    optimizer = Automagic_CameAMP_Improved(model.parameters(), **config.__dict__)
    print("✅ 改進版優化器創建完成")

    # === 4. 訓練範例 ===
    criterion = nn.MSELoss()

    print("\n🚀 開始訓練範例...")
    for epoch in range(5):
        # 模擬批次數據
        batch_size = 32
        input_data = torch.randn(batch_size, 512)
        target = torch.randn(batch_size, 512)

        # 前向傳播
        output = model(input_data)
        loss = criterion(output, target)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()

        # 優化步驟
        optimizer.step()

        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.6f}")

    print("✅ 訓練完成")

    # === 5. 顯示優化器狀態 ===
    print("\n📊 優化器狀態摘要:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"參數群組 {i+1}:")
        print(f"  - 當前學習率: {group['lr']:.2e}")
        print(f"  - 邊緣抑制: {'啟用' if group['edge_suppression'] else '停用'}")
        print(f"  - LoRA 低秩懲罰: {'啟用' if group['lora_rank_penalty'] else '停用'}")
        print(f"  - 空間感知: {'啟用' if group['spatial_awareness'] else '停用'}")

def advanced_configuration_example():
    """進階配置範例：針對不同任務類型的優化配置."""

    print("\n🔧 進階配置範例:")

    # === 配置 1：主要物體訓練（減少邊緣過擬合） ===
    main_object_config = ImprovedOptimizerConfig(
        lr=2e-4,
        edge_suppression=True,
        edge_penalty=0.2,           # 較強的邊緣抑制
        background_regularization=True,
        frequency_penalty=0.1,      # 較強的高頻抑制
        detail_preservation=0.9,    # 保留更多細節
        lora_rank_penalty=True,
        rank_penalty_strength=0.015,
        warmup_steps=1000,
        verbose=False
    )
    print("✅ 主要物體訓練配置 - 強化邊緣控制")

    # === 配置 2：背景/風格訓練（減少背景過擬合） ===
    background_style_config = ImprovedOptimizerConfig(
        lr=1e-4,
        edge_suppression=True,
        edge_penalty=0.1,           # 較輕的邊緣抑制
        background_regularization=True,
        frequency_penalty=0.05,     # 較輕的高頻抑制
        detail_preservation=0.7,    # 允許更多平滑化
        lora_rank_penalty=True,
        rank_penalty_strength=0.025, # 更強的低秩約束
        low_rank_emphasis=1.5,      # 更強調低秩
        warmup_steps=1200,          # 更長的 warmup
        verbose=False
    )
    print("✅ 背景/風格訓練配置 - 強化背景控制")

    # === 配置 3：細節保留訓練（平衡模式） ===
    detail_preserving_config = ImprovedOptimizerConfig(
        lr=8e-5,
        edge_suppression=True,
        edge_penalty=0.12,          # 中等邊緣抑制
        background_regularization=True,
        frequency_penalty=0.06,     # 中等高頻抑制
        detail_preservation=0.8,    # 平衡的細節保存
        spatial_awareness=True,
        lora_rank_penalty=True,
        rank_penalty_strength=0.018,
        low_rank_emphasis=1.2,
        warmup_steps=600,
        verbose=False
    )
    print("✅ 細節保留訓練配置 - 平衡模式")

def troubleshooting_tips():
    """故障排除和調優建議."""

    print("\n🛠️  調優建議和故障排除:")
    print("""
    📌 邊緣過擬合問題：
       - 增加 edge_penalty (0.1 → 0.2)
       - 降低 edge_threshold (0.6 → 0.4)
       - 增加 frequency_penalty (0.05 → 0.1)

    📌 背景過擬合問題：
       - 啟用 background_regularization=True
       - 增加 rank_penalty_strength (0.01 → 0.03)
       - 增加 low_rank_emphasis (1.2 → 1.5)

    📌 細節丟失問題：
       - 增加 detail_preservation (0.8 → 0.9)
       - 降低 frequency_penalty (0.08 → 0.03)
       - 降低 edge_penalty (0.15 → 0.08)

    📌 訓練不穩定：
       - 增加 warmup_steps (500 → 1000)
       - 降低初始學習率
       - 啟用 verbose=True 監控

    📌 記憶體使用過多：
       - 降低 context_window (30 → 15)
       - 關閉部分功能 (spatial_awareness=False)
    """)

if __name__ == "__main__":
    # 執行主要範例
    main()

    # 顯示進階配置
    advanced_configuration_example()

    # 顯示調優建議
    troubleshooting_tips()

    print("\n🎉 範例執行完成！請根據您的具體需求調整配置參數。")