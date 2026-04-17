# 量化

## 1. 什么是量化，量化具体过程是什么？

量化就是把原本高精度的权重、激活从 `FP32 / FP16` 映射到更低比特表示，如 `INT8 / INT4`，以减少显存、带宽和推理延迟。

核心过程可以概括为：

1. 确定量化粒度，如 tensor-wise、per-channel、per-head
2. 统计数值范围，得到 `scale` 和 `zero-point`
3. 把浮点数映射成整数
4. 推理时再反量化，或直接用量化算子计算

常见线性量化公式：

```text
q = round(x / scale) + zero_point
x ≈ scale * (q - zero_point)
```

量化的收益：

- 模型更小
- 显存更省
- 推理吞吐更高

代价是：

- 会有精度损失
- 对异常值和激活分布更敏感

---

## 2. QAT和PTQ有什么区别，QAT具体过程是什么 Per-Channel，Per-Head量化是什么，为什么这么做

PTQ 是 Post-Training Quantization，先训完浮点模型，再做量化校准。

QAT 是 Quantization-Aware Training，在训练或微调阶段就把量化误差模拟进去，让模型学会适应低比特。

区别：

| 方法 | 时机 | 优点 | 缺点 |
| --- | --- | --- | --- |
| PTQ | 训练后 | 简单、成本低 | 精度损失可能更大 |
| QAT | 训练中 | 精度通常更好 | 训练更复杂、成本更高 |

QAT 的典型过程：

1. 在前向里插入 fake quant 节点，模拟量化和反量化
2. 反向仍用近似梯度更新参数
3. 训练结束后导出真实低比特权重

Per-Channel 量化是每个输出通道单独有一个 `scale`。

- 适合卷积、线性层权重
- 因为不同通道的数值分布差异很大

Per-Head 量化是每个 attention head 单独量化。

- 适合 Q/K/V 或 attention 输出
- 因为不同 head 的激活范围和功能可能不同

这样做的原因是：粒度更细，能更贴合真实分布，减少统一量化带来的误差。
