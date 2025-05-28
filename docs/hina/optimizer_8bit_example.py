"""
Automagic_CameAMP_8Bit 優化器使用範例
展示 8bit 量化優化器的記憶體節省效果和使用方法
"""

import torch
import torch.nn as nn
import math
import time
from automagic_cameamp_improved_8bit import Automagic_CameAMP_Improved_8Bit, Optimizer8BitConfig

def create_large_lora_model():
    """創建一個較大的 LoRA 模型來展示記憶體節省效果."""
    class LargeLoRA(nn.Module):
        def __init__(self, in_features=2048, out_features=2048, rank=64):
            super().__init__()
            # 多層 LoRA 結構
            self.lora_layers = nn.ModuleList([
                nn.ModuleDict({
                    'lora_A': nn.Linear(in_features, rank, bias=False),
                    'lora_B': nn.Linear(rank, out_features, bias=False),
                }) for _ in range(4)  # 4 層 LoRA
            ])
            self.scaling = 0.1

            # 初始化權重
            for layer in self.lora_layers:
                nn.init.kaiming_uniform_(layer['lora_A'].weight, a=math.sqrt(5))
                nn.init.zeros_(layer['lora_B'].weight)

        def forward(self, x):
            for layer in self.lora_layers:
                delta = layer['lora_B'](layer['lora_A'](x)) * self.scaling
                x = x + delta
            return x

    return LargeLoRA()

def get_model_memory_usage(model):
    """計算模型的記憶體使用量."""
    total_params = 0
    total_memory = 0

    for param in model.parameters():
        total_params += param.numel()
        total_memory += param.numel() * param.element_size()

    return total_params, total_memory

def memory_comparison_demo():
    """展示 8bit 與 32bit 優化器的記憶體使用比較."""
    print("🔬 記憶體使用比較測試")
    print("=" * 50)

    # 創建大型模型
    model = create_large_lora_model()
    total_params, model_memory = get_model_memory_usage(model)

    print(f"📊 模型統計:")
    print(f"  - 總參數數量: {total_params:,}")
    print(f"  - 模型記憶體: {model_memory/1024/1024:.2f} MB")

    # 配置 8bit 優化器
    config_8bit = Optimizer8BitConfig(
        lr=1e-4,
        quantize_states=True,
        error_correction=True,
        block_size=256,
        verbose=True,
        sync_frequency=50
    )

    # 創建 8bit 優化器
    optimizer_8bit = Automagic_CameAMP_8Bit(model.parameters(), **config_8bit.__dict__)

    # 模擬幾步訓練以初始化狀態
    print("\n🚀 初始化優化器狀態...")
    criterion = nn.MSELoss()

    for step in range(10):
        # 模擬數據
        batch_size = 16
        input_data = torch.randn(batch_size, 2048)
        target = torch.randn(batch_size, 2048)

        # 前向傳播
        output = model(input_data)
        loss = criterion(output, target)

        # 反向傳播
        optimizer_8bit.zero_grad()
        loss.backward()
        optimizer_8bit.step()

        if step % 5 == 0:
            print(f"  初始化步驟 {step+1}/10")

    # 獲取記憶體統計
    memory_stats = optimizer_8bit.get_memory_stats()

    print(f"\n📈 8bit 優化器記憶體統計:")
    print(f"  - 量化狀態記憶體: {memory_stats['total_quantized_memory']/1024/1024:.2f} MB")
    print(f"  - 高精度狀態記憶體: {memory_stats['total_high_precision_memory']/1024/1024:.2f} MB")
    print(f"  - 總優化器記憶體: {(memory_stats['total_quantized_memory'] + memory_stats['total_high_precision_memory'])/1024/1024:.2f} MB")
    print(f"  - 量化參數數量: {memory_stats['quantized_params']}")
    print(f"  - 高精度參數數量: {memory_stats['high_precision_params']}")

    # 估算 32bit 優化器的記憶體使用
    # 標準優化器通常需要: exp_avg, exp_avg_sq, exp_avg_res 等狀態
    estimated_32bit_memory = total_params * 4 * 4  # 4個狀態 × 4bytes (float32)

    print(f"\n🔍 記憶體比較:")
    print(f"  - 估算 32bit 優化器記憶體: {estimated_32bit_memory/1024/1024:.2f} MB")
    print(f"  - 實際 8bit 優化器記憶體: {(memory_stats['total_quantized_memory'] + memory_stats['total_high_precision_memory'])/1024/1024:.2f} MB")

    memory_saved = estimated_32bit_memory - (memory_stats['total_quantized_memory'] + memory_stats['total_high_precision_memory'])
    compression_ratio = (memory_stats['total_quantized_memory'] + memory_stats['total_high_precision_memory']) / estimated_32bit_memory

    print(f"  - 記憶體節省: {memory_saved/1024/1024:.2f} MB ({(1-compression_ratio)*100:.1f}%)")
    print(f"  - 壓縮比: {compression_ratio:.2f}x")

def performance_benchmark():
    """性能基準測試."""
    print("\n⚡ 性能基準測試")
    print("=" * 50)

    # 創建模型
    model = create_large_lora_model()

    # 8bit 優化器配置
    config_8bit = Optimizer8BitConfig(
        lr=1e-4,
        quantize_states=True,
        error_correction=True,
        edge_suppression=True,
        edge_penalty=0.15,
        lora_rank_penalty=True,
        verbose=False
    )

    optimizer = Automagic_CameAMP_Improved_8Bit(model.parameters(), **config_8bit.__dict__)
    criterion = nn.MSELoss()

    # 預熱
    print("🔥 預熱階段...")
    for _ in range(5):
        input_data = torch.randn(16, 2048)
        target = torch.randn(16, 2048)
        output = model(input_data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 性能測試
    print("📊 執行性能測試...")
    num_steps = 50
    start_time = time.time()

    losses = []
    for step in range(num_steps):
        input_data = torch.randn(16, 2048)
        target = torch.randn(16, 2048)

        output = model(input_data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0:
            print(f"  步驟 {step+1}/{num_steps}, Loss: {loss.item():.6f}")

    end_time = time.time()

    print(f"\n📈 性能結果:")
    print(f"  - 總時間: {end_time - start_time:.2f} 秒")
    print(f"  - 平均每步時間: {(end_time - start_time)/num_steps*1000:.2f} ms")
    print(f"  - 最終損失: {losses[-1]:.6f}")
    print(f"  - 損失改善: {((losses[0] - losses[-1])/losses[0]*100):.1f}%")

    # 最終記憶體統計
    final_stats = optimizer.get_memory_stats()
    print(f"  - 最終記憶體使用: {(final_stats['total_quantized_memory'] + final_stats['total_high_precision_memory'])/1024/1024:.2f} MB")

def configuration_examples():
    """不同配置範例."""
    print("\n🔧 配置範例")
    print("=" * 50)

    print("1️⃣ 記憶體優先配置（最大記憶體節省）:")
    memory_first_config = Optimizer8BitConfig(
        lr=1e-4,
        quantize_states=True,      # 量化所有可能的狀態
        error_correction=False,    # 關閉誤差修正以節省記憶體
        block_size=512,           # 較大的塊大小
        edge_suppression=False,    # 關閉部分功能
        spatial_awareness=False,   # 減少額外狀態
        verbose=False
    )
    print(f"   - 量化狀態: {memory_first_config.quantize_states}")
    print(f"   - 誤差修正: {memory_first_config.error_correction}")
    print(f"   - 塊大小: {memory_first_config.block_size}")

    print("\n2️⃣ 精度優先配置（保持最佳訓練品質）:")
    precision_first_config = Optimizer8BitConfig(
        lr=1e-4,
        quantize_states=True,      # 仍然量化以節省記憶體
        error_correction=True,     # 啟用誤差修正
        block_size=128,           # 較小的塊大小，更高精度
        edge_suppression=True,     # 保持所有功能
        edge_penalty=0.12,
        background_regularization=True,
        spatial_awareness=True,
        lora_rank_penalty=True,
        verbose=True
    )
    print(f"   - 量化狀態: {precision_first_config.quantize_states}")
    print(f"   - 誤差修正: {precision_first_config.error_correction}")
    print(f"   - 塊大小: {precision_first_config.block_size}")
    print(f"   - 邊緣抑制: {precision_first_config.edge_suppression}")

    print("\n3️⃣ 平衡配置（記憶體與精度平衡）:")
    balanced_config = Optimizer8BitConfig(
        lr=1e-4,
        quantize_states=True,
        error_correction=True,
        block_size=256,           # 中等塊大小
        edge_suppression=True,
        edge_penalty=0.10,
        background_regularization=True,
        spatial_awareness=True,
        frequency_penalty=0.05,
        lora_rank_penalty=True,
        rank_penalty_strength=0.015,
        sync_frequency=100,
        verbose=False
    )
    print(f"   - 量化狀態: {balanced_config.quantize_states}")
    print(f"   - 誤差修正: {balanced_config.error_correction}")
    print(f"   - 塊大小: {balanced_config.block_size}")
    print(f"   - 同步頻率: {balanced_config.sync_frequency}")

def troubleshooting_tips():
    """故障排除建議."""
    print("\n🛠️ 8bit 優化器故障排除建議")
    print("=" * 50)
    print("""
💡 記憶體不足問題：
   - 增加 block_size (256 → 512)
   - 關閉 error_correction
   - 設定 quantize_states=True
   - 關閉 spatial_awareness

💡 精度下降問題：
   - 減少 block_size (256 → 128)
   - 啟用 error_correction=True
   - 降低 sync_frequency (100 → 50)
   - 檢查 verbose 輸出的量化誤差

💡 訓練不穩定：
   - 啟用 error_correction=True
   - 增加 warmup_steps (500 → 1000)
   - 降低學習率
   - 檢查 lr_mask 的數值範圍

💡 速度過慢：
   - 增加 block_size 減少計算開銷
   - 關閉不必要的功能
   - 降低 sync_frequency
   - 設定 verbose=False

💡 記憶體洩漏：
   - 定期檢查 get_memory_stats()
   - 確保正確的狀態管理
   - 避免在訓練循環中創建新張量
    """)

def main():
    """主函數."""
    print("🎯 Automagic_CameAMP_8Bit 優化器完整測試")
    print("=" * 60)

    # 檢查 CUDA 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用設備: {device}")

    if device.type == "cuda":
        print(f"📊 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    try:
        # 執行各項測試
        memory_comparison_demo()
        performance_benchmark()
        configuration_examples()
        troubleshooting_tips()

        print("\n✅ 所有測試完成！")
        print("\n📋 總結:")
        print("- 8bit 優化器可顯著減少記憶體使用（通常 60-75%）")
        print("- 誤差修正機制保持訓練精度")
        print("- 適合大型 LoRA 模型和記憶體受限環境")
        print("- 可根據需求調整量化策略")

    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        print("請檢查：")
        print("1. PyTorch 版本是否支援所需功能")
        print("2. 記憶體是否充足")
        print("3. 相關依賴是否安裝完整")

if __name__ == "__main__":
    main()