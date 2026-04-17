# Transformer 结构与注意力

## 1. 它讲了什么，有什么贡献，相对于 CNN、RNN 的优势在哪里，为什么 Transformer 要引入注意力机制？

`Attention Is All You Need` 提出了以自注意力为核心的序列建模框架，用并行 Attention + FFN 取代了 RNN 的递归结构。

- 相对 `RNN`：并行性更强，长程依赖路径更短，不需要一步步传状态
- 相对 `CNN`：不依赖固定卷积核感受野，更容易建模远距离 token 关系
- 引入注意力机制的原因：不同 token 对当前 token 的重要性不同，模型需要按内容动态聚合上下文，而不是只靠固定窗口或最后一个隐藏状态

一句话总结：Transformer 的核心贡献是把“序列建模”改成了“内容相关的全局信息交互”。

---

## 2. 详细解释 Transformer 的整体架构，编码器和解码器分别由哪些模块组成？

标准 Transformer 是 `encoder-decoder` 结构。

- 输入先经过 `token embedding + position encoding`
- `Encoder` 堆叠多层，每层通常是：
  - Multi-Head Self-Attention
  - Add & Norm
  - FFN
  - Add & Norm
- `Decoder` 堆叠多层，每层通常是：
  - Masked Multi-Head Self-Attention
  - Add & Norm
  - Cross-Attention
  - Add & Norm
  - FFN
  - Add & Norm

职责区别：

- `Encoder` 负责把源序列编码成上下文表示
- `Decoder` 一边看历史输出，一边看编码结果，逐步生成目标序列

---

## 3. MHA 的运行机制是什么？常见注意力机制有哪些，为什么要用多头注意力？

MHA 就是把输入投影成多组 `Q / K / V`，每个头各自做一次注意力，再把各头结果拼接并线性映射。

单头公式：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

为什么用多头：

- 不同头可以关注不同关系，比如语法、位置、实体、局部搭配
- 把一个大空间拆成多个子空间，表达更丰富
- 比单头更容易学习多种相关性，而不是把所有关系混在一张注意力图里

常见注意力机制：

- Self-Attention：Q/K/V 来自同一序列
- Cross-Attention：Q 来自当前序列，K/V 来自另一序列
- Causal Attention：只能看当前位置之前的 token
- Sparse / Local Attention：只看部分位置，降低复杂度

---

## 4. Transformer 的整体时间复杂度是什么？单层 Transformer 内部各模块的时间复杂度分别是多少，Attention 的占比如何？

设序列长度为 `n`，隐藏维度为 `d`。

单层主要复杂度：

| 模块 | 时间复杂度 |
| --- | --- |
| Q/K/V 线性投影 | `O(nd^2)` |
| Attention 分数 `QK^T` | `O(n^2 d)` |
| Attention 加权求和 | `O(n^2 d)` |
| 输出投影 | `O(nd^2)` |
| FFN | `O(ndd_ff)`，若 `d_ff≈4d`，可记作 `O(nd^2)` |

所以单层总体通常写成：

$$
O(n^2 d + nd^2)
$$

结论：

- 当 `n` 很长时，`Attention` 的 `n^2` 项会成为瓶颈
- 当 `n` 不长但 `d` 很大时，线性层和 FFN 也很重
- LLM 长上下文场景下，真正先炸的一般是 Attention 的显存和访存

---

## 5. 三角位置编码（sin / cos）的优势和用处是什么？为什么通常使用 10000 作为基底，对于 123 这种硬编码的好处是？为什么需要位置嵌入，常见做法为什么更偏向加法而不是直接拼接？

Transformer 本身对输入顺序是置换不变的，所以必须显式注入位置信息。

`sin / cos` 位置编码的优势：

- 无需学习参数，外推到更长长度时更稳
- 不同频率覆盖长短期位置关系
- 相对位移可以由三角函数关系间接表达，便于模型学习相对位置

为什么常用 `10000`：

- 它是一个经验上足够大的尺度，让不同维度覆盖从高频到低频的多尺度周期
- 太小会周期过短，远距离位置容易混淆；太大又会让相邻位置变化过慢

对比“123 这种硬编码”的好处：

- 固定公式可复现、可泛化，不依赖手工指定每个位置值
- 新长度不需要重新学表

为什么更偏向加法不是拼接：

- 加法不改变 hidden size，后续层结构不需要改
- 位置和语义会在同一表示空间中融合
- 拼接会增大维度、参数和计算量，也会让残差结构更麻烦

---

## 6. ROPE是相对位置编码还是绝对位置编码，原理是什么

RoPE 从注入方式看是“按绝对位置给 `Q/K` 加旋转角度”，但从注意力结果看，内积主要依赖相对位移，所以通常归到“相对位置编码家族”。

原理：

- 把向量按两维一组看成复平面上的点
- 对第 `m` 个位置施加一个与位置相关的旋转角
- `Q_m` 和 `K_n` 做内积时，会自然带出 `m-n` 的相对位置信息

优点：

- 不需要显式构造相对位置矩阵
- 和 Attention 结合自然
- 长上下文扩展通常比绝对位置 embedding 更稳

---

## 7. MHA 的后续发展是什么呢（MQA, GQA），它们和 MHA 的区别、作用分别是什么？

三者主要差在 `K/V` 的共享方式。

| 结构 | Q 头数 | K/V 头数 | 特点 |
| --- | --- | --- | --- |
| MHA | 多 | 多 | 表达能力强，但 KV Cache 大 |
| MQA | 多 | 1 | 所有 Q 共享一组 K/V，显著降低 KV Cache |
| GQA | 多 | 少量组 | 在 MHA 和 MQA 之间折中 |

作用：

- `MQA` 主要解决推理时 KV Cache 太大、带宽压力太强的问题
- `GQA` 在保持较好效果的同时，大幅降低显存和访存开销

现代 LLM 更常见的是 `GQA`，因为它通常比纯 `MQA` 更稳。

---

## 8. Transformer 掩码注意力机制具体如何做的，为什么 Decoder 自注意力需要 causal mask，三种注意力里哪些需要 mask，如何实现？

做法是在 attention score 上对“不允许看见的位置”加一个极小值，比如 `-inf`，再做 softmax，这样这些位置的权重就近似为 0。

为什么 Decoder Self-Attention 需要 `causal mask`：

- 训练时是并行喂入整段目标序列
- 如果不遮住未来 token，就会看到答案，形成标签泄漏

三种注意力的 mask：

| 注意力类型 | 常见 mask |
| --- | --- |
| Encoder Self-Attention | padding mask |
| Decoder Self-Attention | causal mask + padding mask |
| Cross-Attention | 对 encoder 侧做 padding mask |

实现上通常是：

- 先构造布尔矩阵或上三角矩阵
- 把非法位置填成 `-inf`
- 再做 `softmax`

---

## 9. 为什么用 Layer Norm 而不是 Batch Norm，Transformer / LLaMA 中归一化设计有什么区别，RMSNorm 又是什么

Transformer 更适合 `LayerNorm`，不是 `BatchNorm`。

原因：

- 序列长度可变，NLP 中 batch 统计不稳定
- 推理时 batch size 经常变化，BN 训练和推理分布容易不一致
- LN 按 token 的特征维做归一化，不依赖 batch 维，更适合自回归生成

设计差异：

- 原始 Transformer 论文常见 `Post-Norm`：子层后做 LN
- 现代大模型更常见 `Pre-Norm`：子层前做 LN，训练更稳定，深层更容易收敛
- LLaMA 系列进一步常用 `RMSNorm`

`RMSNorm` 是什么：

- 只按均方根做缩放，不减均值
- 公式更简单，计算略轻
- 实际大模型中常能保持和 LN 接近甚至更好的稳定性

---

## 10. Self-Attention 和 Cross Attention 的区别，分别应用在哪里？Cross Attention 的 Q / K / V 分别来自哪里？

区别在于 Q/K/V 是否来自同一序列。

- `Self-Attention`：Q/K/V 都来自当前序列，用于序列内部建模
- `Cross-Attention`：Q 来自当前解码序列，K/V 来自另一序列，用于跨模态或跨序列对齐

应用：

- Encoder 内部通常用 Self-Attention
- Decoder 的第一层注意力常是 Masked Self-Attention
- Encoder-Decoder、图文模型、语音文本对齐里常用 Cross-Attention

Cross-Attention 中：

- `Q` 来自 decoder hidden states
- `K/V` 来自 encoder 输出或另一模态特征

---

## 11. Q、K、V 机制和注意力公式是什么？为什么要除以 $\sqrt{d_k}$，这和数值稳定性 / 梯度有什么关系？

可以把它理解成：

- `Q`：当前 token 想找什么
- `K`：每个 token 提供什么索引
- `V`：每个 token 真正携带的内容

公式：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

为什么除以 `\sqrt{d_k}`：

- 如果 `d_k` 很大，点积值方差会变大
- softmax 输入过大容易饱和，概率接近 one-hot
- 一旦饱和，梯度会变小，训练不稳定

所以缩放的本质是：

- 控制 logits 尺度
- 提高数值稳定性
- 让 softmax 处在更有梯度的区间

---

## 12. 什么是kv cache，为什么需要kv cache

`KV Cache` 是在自回归生成时，把历史 token 的 `K/V` 保存起来，下一个 token 解码时直接复用。

为什么需要：

- 没有 cache 时，每生成一个 token 都要把历史序列重新算一遍
- 有了 cache，只需对新 token 计算一次 `Q/K/V`，再和历史 `K/V` 做注意力

效果：

- 计算从“重复全序列”变成“增量计算”
- decode 延迟显著下降
- 代价是显存占用增加，尤其长上下文和大 batch 时很明显

---

## 13. 有哪些优化注意力计算的方法？Flash Attention 和 DeepSpeed 分别在优化什么，GPU 运算瓶颈在哪里？

常见优化方向：

- Kernel fusion，减少中间张量落显存
- 分块计算，降低峰值显存
- 稀疏 / 局部 / 线性注意力，减少 `n^2`
- MQA / GQA，减少 KV Cache
- Paged KV Cache、连续批处理，提升推理吞吐

`FlashAttention` 主要优化的是注意力内核本身：

- 用 IO-aware 的方式分块计算 `QK^T` 和 `softmax(V)`
- 尽量少读写 HBM
- 降低显存峰值，同时提速

`DeepSpeed` 更偏系统级优化：

- 训练侧的 ZeRO、通信与显存分片
- 大模型训练和推理的并行、内存管理、kernel 优化
- 它不只优化 attention，而是整体训练/部署效率

GPU 瓶颈常见在：

- HBM 带宽
- KV Cache 读写
- kernel launch 和数据搬运
- 多卡通信，而不一定只是算力 FLOPs 不够

---

## 14. self Attention 如何处理长文本权重分散？

长文本里容易出现“注意力太分散”或远距离信息被噪声淹没的问题。

常见处理方法：

- 局部窗口 + 全局 token，限制每个 token 的可见范围
- 稀疏注意力，让关键位置保留全局连接
- 用更适合长上下文的相对位置编码，如 RoPE / ALiBi
- 检索增强，把外部相关片段拉回来，而不是全靠一个注意力矩阵记
- 分块、滑窗、summary memory、压缩历史上下文

本质上是在做两件事：

- 降低无效 token 对注意力的干扰
- 把有限算力集中到真正相关的位置

---

## 15. 位置编码问题如何影响意图判别？

如果位置编码设计不好，模型会更像“词袋”，只知道有哪些词，不知道它们的顺序关系。

影响主要体现在：

- 否定词位置错了，语义会反过来
- 主谓宾或修饰关系错了，意图分类会偏
- 长句里前后条件、转折、时间顺序识别不准
- 超过训练长度后，位置外推失真，意图判断容易漂移

所以意图判别不仅看“词有没有出现”，还看“这些词在什么顺序和结构里出现”。

---

## 16. QKV 为什么要使用独立的线性投影，而不是输入嵌入？

因为三个角色不同，不能直接共用原始 embedding。

- `Q` 负责发起匹配
- `K` 负责被匹配
- `V` 负责提供内容

独立投影的好处：

- 让模型在不同子空间里学习“查什么”“怎么匹配”“取什么内容”
- 提高表达能力
- 允许不同头学习不同模式

如果直接用输入 embedding：

- 表达过于受限
- 匹配空间和内容空间耦合太死
- 很难学到复杂的非对称关系

---

## 17. 计算注意力得分的时候 QK 都是归一化之后的吗？为什么需要 token 位置编码？

标准 Transformer 里，`Q/K` 一般不是先做 L2 归一化后再算点积，而是直接线性投影后做缩放点积注意力。

补充：

- 有些变体会做 `QK-Norm` 或 cosine attention，用归一化控制尺度
- 但 vanilla 做法不是“先归一化再点积”

为什么需要 token 位置编码：

- Self-Attention 天然对输入顺序不敏感
- 没有位置编码时，模型只能看到 token 集合，无法区分“AB”和“BA”
- 位置编码让模型知道顺序、距离和相对关系

---

## 18. Transformer / LLaMA 等模型中的 FFN 设计有什么变化？ReLU、GELU、SwiGLU 有什么区别？

FFN 本质是“逐 token 的非线性特征变换”，不做 token 间交互。

演化大致是：

- 原始 Transformer：两层线性 + `ReLU`
- BERT / GPT 系：大量使用 `GELU`
- LLaMA 系：常用 `SwiGLU` 或类似 gated FFN

对比：

| 激活 | 特点 |
| --- | --- |
| ReLU | 简单、便宜，但 0 以下直接截断 |
| GELU | 更平滑，保留小负值信息，语言模型里常更稳 |
| SwiGLU | 引入门控，表达更强，现代 LLM 很常见 |

趋势：

- FFN 不只是“升维再降维”
- 现代 LLM 更强调 gated 结构和更高参数利用率

---

## 19. Transformer 训练一次时，哪些参数会参与更新？embedding、QKV 投影、FFN、LayerNorm、输出层分别如何反传？

如果是全参数训练，所有可学习参数都会参与更新，包括：

- token embedding
- 位置相关参数（若是可学习位置编码）
- Q/K/V/O 投影
- FFN 各层权重
- LayerNorm / RMSNorm 的可学习缩放参数
- 输出层 `lm_head`

反传路径是：

`loss -> 输出 logits -> 最后一层 hidden state -> 各层 attention / FFN / norm -> embedding`

分别理解：

- `embedding`：通过输入 token 对应的隐藏状态梯度回传
- `QKV` 投影：来自 attention 输出对输入和权重的梯度
- `FFN`：来自残差后的误差回传
- `LayerNorm`：对归一化前激活和缩放参数都回传梯度
- `输出层`：直接由 logits 和 label 的损失求梯度

如果用了参数冻结、LoRA 或 PEFT，那就不是所有参数都更新。

---

## 20. Encoder 中每个 token 都能看到全部 token，这种全局感受野一定越大越好吗？什么时候更大的或更小的感受野更合适？

不一定越大越好。

全局感受野的好处：

- 能建模远距离依赖
- 适合机器翻译、长文理解、跨段落推理

缺点：

- 计算和显存开销高
- 容易把不相关 token 也纳入，增加噪声
- 小样本任务里更容易过拟合

什么时候更大更合适：

- 文档级理解
- 长程依赖明显
- 需要全局一致性

什么时候更小更合适：

- 局部模式主导，如语音局部帧、部分时序预测
- 实时系统，对延迟敏感
- 希望抑制远处噪声

---

## 21. encoder only、decoder only 和 encoder-decoder 有什么区别？分别适合什么任务？

| 架构 | 特点 | 常见任务 |
| --- | --- | --- |
| Encoder-only | 双向看上下文，擅长理解 | 分类、检索、NER、句向量 |
| Decoder-only | 自回归生成，擅长续写 | 对话、写作、代码生成 |
| Encoder-Decoder | 源序列编码 + 目标序列生成 | 翻译、摘要、语音识别、条件生成 |

理解方式：

- `Encoder-only` 强在表征学习
- `Decoder-only` 强在统一生成接口
- `Encoder-Decoder` 强在输入输出结构明确的 seq2seq 任务

---

## 22. BERT 的预训练任务有哪些？输入 embedding 是如何构成的？

BERT 经典预训练任务有两个：

- `MLM`：Masked Language Modeling，随机遮住部分 token，让模型预测
- `NSP`：Next Sentence Prediction，判断两句是否为上下文相邻

补充：

- 后续很多工作发现 `NSP` 不是必须，比如 RoBERTa 就去掉了它

BERT 的输入 embedding 一般由三部分相加：

- `Token Embedding`
- `Position Embedding`
- `Segment Embedding`

其中 `Segment Embedding` 用来区分句子 A / 句子 B。
