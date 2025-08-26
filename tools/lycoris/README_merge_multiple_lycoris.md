# LoKr/LoHa 多模型融合工具

这个工具集提供了将多个 LoKr/LoHa 模型融合为单个模型的功能，类似于 LoRA 模型的融合。

## 工具概览

### 1. 基础融合工具 (`merge_multiple_lycoris.py`)

基础的 LoKr/LoHa 模型融合工具，支持简单的加权融合。

### 2. 高级融合工具 (`merge_multiple_lycoris_advanced.py`)

高级融合工具，提供多种融合策略和智能权重调整。

## 安装依赖

确保已安装以下依赖：

```bash
pip install torch safetensors
```

## 基础工具使用方法

### 基本语法

```bash
python merge_multiple_lycoris.py model1.safetensors model2.safetensors model3.safetensors [选项]
```

### 参数说明

- `models`: 要融合的模型文件路径（支持多个）
- `--weights`: 每个模型的权重（可选，默认等权重）
- `--output` / `-o`: 输出文件路径
- `--device`: 使用的设备（默认：cpu）
- `--dtype`: 输出模型的数据类型（默认：float16）
- `--metadata`: 保存元数据的文件路径（可选）
- `--verify`: 验证融合后的模型

### 使用示例

#### 1. 等权重融合三个模型

```bash
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    model3.safetensors \
    --output merged_model.safetensors
```

#### 2. 自定义权重融合

```bash
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    model3.safetensors \
    --weights 0.5 0.3 0.2 \
    --output merged_model.safetensors
```

#### 3. 保存元数据并验证

```bash
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    --weights 0.7 0.3 \
    --output merged_model.safetensors \
    --metadata merge_info.json \
    --verify
```

## 高级工具使用方法

### 基本语法

```bash
python merge_multiple_lycoris_advanced.py model1.safetensors model2.safetensors [选项]
```

### 融合策略

高级工具提供以下融合策略：

1. **weighted_sum**: 简单加权求和
2. **weighted_average**: 加权平均
3. **layer_adaptive**: 基于层特性的自适应融合
4. **smart_fusion**: 智能融合（默认）
5. **min_max_norm**: 最小-最大缩放融合

### 参数说明

- `--strategy`: 选择融合策略
- `--norm-threshold`: 缩放阈值（用于 min_max_norm 策略）
- 其他参数与基础工具相同

### 使用示例

#### 1. 使用智能融合策略

```bash
python merge_multiple_lycoris_advanced.py \
    model1.safetensors \
    model2.safetensors \
    --strategy smart_fusion \
    --weights 0.6 0.4 \
    --output smart_merged.safetensors
```

#### 2. 使用层自适应融合

```bash
python merge_multiple_lycoris_advanced.py \
    model1.safetensors \
    model2.safetensors \
    model3.safetensors \
    --strategy layer_adaptive \
    --weights 0.5 0.3 0.2 \
    --output adaptive_merged.safetensors
```

#### 3. 使用归一化融合

```bash
python merge_multiple_lycoris_advanced.py \
    model1.safetensors \
    model2.safetensors \
    --strategy min_max_norm \
    --norm-threshold 0.8 \
    --weights 0.7 0.3 \
    --output normalized_merged.safetensors
```

## 融合策略详解

### 1. 加权求和 (weighted_sum)
- 直接对权重进行加权求和
- 适用于权重差异较大的情况
- 可能产生较大的权重值

### 2. 加权平均 (weighted_average)
- 直接使用原始权重进行加权求和
- 保持权重的原始比例，不进行归一化
- 适用于需要精确控制权重比例的情况

### 3. 层自适应 (layer_adaptive)
- 根据每层的权重特性自动调整融合权重
- 考虑权重方差，方差小的层获得更高权重
- **不进行权重归一化，直接使用调整后的权重**
- 适用于模型间差异较大的情况

### 4. 智能融合 (smart_fusion)
- 基于层重要性分析进行融合
- 自动计算每层的最优融合权重
- 综合考虑权重大小和模型特性
- **先归一化确保稳定性，再乘以用户权重保持比例**
- 推荐用于大多数情况

### 5. 最小-最大缩放 (min_max_norm)
- 融合后进行数值范围缩放（不是权重归一化）
- 防止权重值过大或过小
- 适用于需要控制权重范围的情况

## 注意事项

### 1. 模型兼容性
- 确保所有模型具有相同的结构
- 支持混合 LoKr/LoHa 模型
- 工具会自动检测模型类型

### 2. 权重设置
- 权重数量必须与模型数量匹配
- 权重可以是任意正数或负数
- **注意**: 不同策略的权重处理方式不同：
  - **weighted_sum** 和 **weighted_average**: 权重不会被归一化，直接使用原始值
  - **layer_adaptive**: 不进行权重归一化，直接使用调整后的权重
- **smart_fusion**: 先归一化确保稳定性，再乘以用户权重保持比例
  - **min_max_norm**: 权重不会被归一化，但会对最终结果进行数值范围缩放

### 3. 内存使用
- 大模型融合需要足够的内存
- 建议使用 CPU 进行融合以节省 GPU 内存
- 可以使用 `--device cpu` 强制使用 CPU

### 4. 输出格式
- 支持 safetensors 和 .pt 格式
- 建议使用 safetensors 格式（更安全）
- 自动处理数据类型转换

## 故障排除

### 常见错误

1. **模型文件不存在**
   ```
   Error: Model file not found: model.safetensors
   ```
   解决：检查文件路径是否正确

2. **权重数量不匹配**
   ```
   Error: Number of weights (2) must match number of models (3)
   ```
   解决：确保权重数量与模型数量相同

3. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：使用 `--device cpu` 或减少同时加载的模型数量

4. **模型结构不兼容**
   ```
   Warning: Models may not be compatible. Proceed with caution.
   ```
   解决：检查模型是否来自相同的训练配置

### 性能优化建议

1. **批量处理**: 对于大量模型，可以分批融合
2. **数据类型**: 使用 float16 可以节省内存和加速处理
3. **设备选择**: 小模型用 GPU，大模型用 CPU
4. **验证模式**: 生产环境建议使用 `--verify` 选项

## 高级用法

### 1. 批量融合脚本

```bash
#!/bin/bash
# 批量融合多个模型组合

models=("model1.safetensors" "model2.safetensors" "model3.safetensors")
weights=(0.5 0.3 0.2)

python merge_multiple_lycoris_advanced.py \
    "${models[@]}" \
    --weights "${weights[@]}" \
    --strategy smart_fusion \
    --output "batch_merged_$(date +%Y%m%d_%H%M%S).safetensors" \
    --metadata "batch_merge_$(date +%Y%m%d_%H%M%S).json" \
    --verify
```

### 2. 渐进式融合

```bash
# 先融合两个模型
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    --weights 0.7 0.3 \
    --output temp_merged.safetensors

# 再与第三个模型融合
python merge_multiple_lycoris.py \
    temp_merged.safetensors \
    model3.safetensors \
    --weights 0.8 0.2 \
    --output final_merged.safetensors
```

### 3. 条件融合

```bash
# 根据模型类型选择不同策略
if [[ "$model_type" == "LoKr" ]]; then
    strategy="smart_fusion"
else
    strategy="weighted_average"
fi

python merge_multiple_lycoris_advanced.py \
    model1.safetensors \
    model2.safetensors \
    --strategy "$strategy" \
    --output merged.safetensors
```

## 技术原理

### LoKr 融合原理

LoKr (Low-rank Kronecker) 模型使用 Kronecker 积分解：
```
W = W1 ⊗ W2
```

融合时：
```
W_merged = Σ(w_i * W1_i ⊗ W2_i)
```

### LoHa 融合原理

LoHa (Low-rank Hadamard) 模型使用 Hadamard 积：
```
W = (W1a @ W1b) ⊙ (W2a @ W2b)
```

融合时：
```
W_merged = Σ(w_i * (W1a_i @ W1b_i) ⊙ (W2a_i @ W2b_i))
```

### 权重调整策略

1. **重要性权重**: 基于 L2 范数计算层重要性
2. **方差调整**: 根据跨模型方差调整权重
3. **自适应融合**: 动态计算最优融合权重

## 总结

这两个工具提供了从基础到高级的 LoKr/LoHa 模型融合功能：

- **基础工具**: 简单易用，适合快速融合
- **高级工具**: 功能丰富，支持多种策略和智能优化

选择合适的工具和策略取决于具体需求：
- 简单融合：使用基础工具
- 复杂融合：使用高级工具
- 性能要求高：选择智能融合策略
- 稳定性要求高：选择加权平均策略

通过合理使用这些工具，可以有效地将多个 LoKr/LoHa 模型融合为单个模型，实现模型能力的组合和优化。
