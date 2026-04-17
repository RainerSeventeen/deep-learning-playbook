# ViT & CLIP

## 1. CLIP 如何构造正负样本对，如何执行对比学习

CLIP 的训练数据是大规模图文对 `(image, text)`。

正负样本的构造方式很直接：

- 正样本：同一条数据里的图片和文本
- 负样本：同一个 batch 里其他不匹配的图文组合

一个 batch 有 `N` 对样本时，会得到一个 `N x N` 的相似度矩阵：

$$
s_{ij} = \frac{\langle v_i, t_j \rangle}{\tau}
$$

其中：

- `v_i` 是图像编码向量
- `t_j` 是文本编码向量
- `\tau` 是温度参数

训练时通常做**双向对比学习**：

- image-to-text：给定图片，匹配正确文本
- text-to-image：给定文本，匹配正确图片

损失本质上是对角线做分类，常写成对称的 InfoNCE / Cross Entropy。

一句话记忆：`CLIP 就是在一个 batch 内，让正确图文更近，让错误图文更远。`

---

## 2. 详细画一下 vit 的架构图

ViT 可以理解成“把图像切成 patch，再当成 token 喂给 Transformer”。

结构图可以这样画：

```text
Image
  -> Patch Partition
  -> Flatten
  -> Linear Projection
  -> Patch Embeddings
  -> Add [CLS] Token
  -> Add Position Embedding
  -> Transformer Encoder x L
       -> MSA
       -> MLP
       -> Residual + LayerNorm
  -> 取 [CLS] 或所有 patch 特征
  -> Classification Head / Downstream Head
```

每一步在做什么：

- `Patch Partition`：把图像切成固定大小的小块，比如 `16x16`
- `Linear Projection`：把每个 patch 映射到统一 hidden size
- `Position Embedding`：告诉模型 patch 的空间位置
- `Transformer Encoder`：建模不同 patch 之间的全局关系
- `Head`：做分类、检测、分割或多模态对齐

和 CNN 的核心区别：

- CNN 靠局部卷积逐层扩大感受野
- ViT 一开始就把图像离散成 token，用全局 self-attention 建模关系

面试里常补一句：ViT 的代价是更吃数据和算力，但全局建模能力更强。
