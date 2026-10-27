# 10 推理

## 简介

Inference 是一个每天需要持续付出的成本。

在 chatbot 中 ai 是为了人类服务，在 agent 中推理速度直接关系着整个任务的加速。

关于推理速度的 metrics 包含一下几种：

1. Time to first token (TTFT)：从输入完成，到第一个 token 的等待时间。
2. Latency (second/token)：对于一个 query（交互式为主）中 token 出现的速度。
3. Throughput (tokens/second)：常用在批处理，多 query 下 token 的速度。

在推理中，因为自回归的特性导致 GPU 较难实现并行化，这也是 inference 的难点。

## Transformer 推理优化

下图详细展示了 Transformer 的维度变化，红色维度代表同时出现并消失，蓝色代表同时出现不消失。

![Transformer 架构](https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png)

arithmetic intensity （算数强度）：$\frac{浮点运算数 FLOPs }{搬运的内存数}$，一定程度上决定了 GPU 的 utilization。

在 Forward 过程中，对于一个 $T$ 序列长度，一个 token 的时间复杂度是 $O(T^2)$，自回归模型需要顺序地生成 $T$ 个 token，所以整个推理的复杂度是 $O(T^3)$

### KV Cache

在生成下一个 token 的时候，不再需要重新计算之前生成的 token 数，而是从 cache 拿出之前计算的值。

![KV cache](https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp)

KV Cache 对每一个 Sequence (B), token (S), layer (L), head (K) 都会存储一个 H dimension 的向量。

此时推理会被分成两部分：

1. prefill：将所有的 prompt 输入进去并 encoder（可以并行化）
2. generation：自回归地生成每一个 token

在这个阶段，有一些方法可以应用：

1. Group query attention (GQA) 来让多个 Q 共享 KV，减少内存占用，加快运算过程。
2. Multi-head latent attention (MLA)  将 KV 值压缩映射到新空间，并在使用时恢复。
3. Cross layer attention (CLA) 共享 KV 机制，下一层共享上一层的 KV 的值。

### MLP

MLP 在推理的时候没有那么严重的问题，MLP 的运算强度是 $B \times T$ （详细推导参见原课程）

因此 MLP 的运算强度取决于输入的 batch 数量，也就是说在批处理场景下是可控的：

1. prefill：可以通过扩大 batch 来增强运算强度。
2. generation：每次生成一个 token，所以 $T=1$，另外 B 不可控，取决于用户的数量。

### Trade-off

batchsize 决定了 throughout 吞吐量，bs 越大吞吐量越大，但是相应的也会带来 TTFT 变慢，在 KV cache 构建时需要花费更长的时间，同时也会导致更多的内存占用。

### Speculative sampling

在上文中提到的一些优化大多是压缩 KV 的有损压缩，还有一种方法，Speculative sampling

1. 先用一个 cheap 的模型来预测很多个 token
2. 再用一个大模型来判断，并给出输出是否足够好

这么做主要是 prefill 的时候可以并行化，但是 generation 的时候只能串行一个个下去，这么做就是把昂贵的大模型的输出改成可并行化的方式。