# LoKr/LoHa 融合工具快速开始指南

这个指南将帮助你在5分钟内开始使用 LoKr/LoHa 融合工具。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch safetensors
```

### 2. 基础融合（推荐新手）

```bash
# 融合两个模型，等权重
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    --output merged.safetensors

# 融合三个模型，自定义权重
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    model3.safetensors \
    --weights 0.5 0.3 0.2 \
    --output merged.safetensors
```

### 3. 高级融合（推荐进阶用户）

```bash
# 使用智能融合策略
python merge_multiple_lycoris_advanced.py \
    model1.safetensors \
    model2.safetensors \
    --strategy smart_fusion \
    --weights 0.7 0.3 \
    --output smart_merged.safetensors

# 使用层自适应策略
python merge_multiple_lycoris_advanced.py \
    model1.safetensors \
    model2.safetensors \
    model3.safetensors \
    --strategy layer_adaptive \
    --weights 0.5 0.3 0.2 \
    --output adaptive_merged.safetensors
```

## 📚 常用场景

### 场景1：风格融合
```bash
# 将两个不同风格的模型融合
# 注意：权重不会被归一化，直接使用你设置的值
python merge_multiple_lycoris.py \
    anime_style.safetensors \
    realistic_style.safetensors \
    --weights 0.6 0.4 \
    --output mixed_style.safetensors
```

### 场景2：能力增强
```bash
# 将多个专门化模型融合为全能模型
# 注意：权重不会被归一化，直接使用你设置的值
python merge_multiple_lycoris_advanced.py \
    portrait_model.safetensors \
    landscape_model.safetensors \
    object_model.safetensors \
    --strategy smart_fusion \
    --weights 0.4 0.3 0.3 \
    --output universal_model.safetensors
```

### 场景3：渐进式融合
```bash
# 先融合两个模型
python merge_multiple_lycoris.py \
    base_model.safetensors \
    enhancement1.safetensors \
    --weights 0.8 0.2 \
    --output temp_merged.safetensors

# 再与第三个模型融合
python merge_multiple_lycoris.py \
    temp_merged.safetensors \
    enhancement2.safetensors \
    --weights 0.9 0.1 \
    --output final_model.safetensors
```

## ⚡ 性能优化技巧

### 内存优化
```bash
# 使用CPU进行融合（节省GPU内存）
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    --device cpu \
    --output merged.safetensors
```

### 数据类型优化
```bash
# 使用float16节省内存和加速
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    --dtype float16 \
    --output merged.safetensors
```

### 批量处理
```bash
# 创建批量融合脚本
#!/bin/bash
models=("model1.safetensors" "model2.safetensors" "model3.safetensors")
weights=(0.5 0.3 0.2)

python merge_multiple_lycoris_advanced.py \
    "${models[@]}" \
    --weights "${weights[@]}" \
    --strategy smart_fusion \
    --output "batch_merged_$(date +%Y%m%d).safetensors"
```

## 🔍 验证和调试

### 验证融合结果
```bash
# 启用验证模式
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    --verify \
    --output merged.safetensors
```

### 保存元数据
```bash
# 保存融合信息
python merge_multiple_lycoris.py \
    model1.safetensors \
    model2.safetensors \
    --metadata merge_info.json \
    --output merged.safetensors
```

### 测试工具功能
```bash
# 运行测试套件
python test_merge_tools.py

# 运行示例演示
python example_usage.py
```

## 🎯 策略选择指南

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **weighted_sum** | 简单融合 | 快速、直观 | 可能产生极端值 |
| **weighted_average** | 平衡融合 | 稳定、可控 | 可能稀释特征 |
| **layer_adaptive** | 差异较大模型 | 自动调整权重 | 计算复杂度高 |
| **smart_fusion** | 通用场景 | 智能、平衡 | 需要更多计算 |
| **min_max_norm** | 控制权重范围 | 防止极端值 | 可能丢失细节 |

## 🚨 常见问题

### Q: 融合后模型效果不好怎么办？
A: 尝试调整权重比例，或使用不同的融合策略

### Q: 内存不足怎么办？
A: 使用 `--device cpu` 或 `--dtype float16`

### Q: 如何选择合适的权重？
A: 权重处理方式取决于选择的融合策略：
- **weighted_sum/weighted_average**: 权重不会被归一化，直接使用你设置的数值
- **layer_adaptive**: 不进行权重归一化，直接使用调整后的权重
- **smart_fusion**: 先归一化确保稳定性，再乘以用户权重保持比例
- **min_max_norm**: 权重不会被归一化，但会对最终结果进行数值范围缩放

建议从等权重开始，根据效果逐步调整

### Q: 支持哪些模型格式？
A: 支持 .safetensors 和 .pt 格式

### Q: 可以融合不同类型的模型吗？
A: 可以，但建议使用相同结构的模型

## 📖 深入学习

- 查看完整文档：`README_merge_multiple_lycoris.md`
- 运行测试脚本：`test_merge_tools.py`
- 查看示例代码：`example_usage.py`

## 🆘 获取帮助

如果遇到问题：
1. 检查错误信息
2. 运行测试脚本验证安装
3. 查看完整文档
4. 检查模型文件格式和结构

---

**提示**: 建议先用小模型测试，确认效果后再处理大模型！
