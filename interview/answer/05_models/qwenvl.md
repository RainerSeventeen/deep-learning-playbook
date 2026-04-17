# QwenVL

## 1. Qwen 2.5 VL 以及 Qwen 3 VL 对比

按 Qwen 官方在 2025 年公开的资料来看，Qwen3-VL 是在 Qwen2.5-VL 之上的新一代视觉语言模型，主要升级点不是“只换了一个更大的 backbone”，而是把长上下文、多图视频建模和 agent 能力一起加强了。

可以这样概括：

| 维度 | Qwen2.5-VL | Qwen3-VL |
| --- | --- | --- |
| 核心定位 | 强视觉理解、OCR、定位、长视频 | 更强多模态推理、长上下文、视觉 agent |
| 上下文能力 | 已支持长视频与复杂文档 | 原生更长的 interleaved context |
| 位置建模 | 动态分辨率 + MRoPE / 时间建模 | 增强版 interleaved-MRoPE |
| 视觉语言对齐 | 动态分辨率 ViT + 多阶段对齐 | 进一步强化多层视觉特征利用 |
| 工程特性 | 文档理解、定位、结构化抽取很强 | 在复杂推理、跨图跨视频关联和执行能力上更进一步 |

面试表达建议：

- `Qwen2.5-VL` 更强调“看得清、读得准、定位准”
- `Qwen3-VL` 更强调“看懂之后还能长程推理、跨模态关联和执行”

如果被追问具体 benchmark，不要硬背分数，重点说清**架构升级方向**。

---

## 2. Qwen 2.5 VL 怎么做的动态分辨率处理的图片或者视频，对比 intervl 两者有什么不一样的

Qwen2.5-VL 的关键点是**原生动态分辨率**。

它的思路不是先把所有图片强行缩放到同一个固定尺寸，而是：

- 尽量保留原图尺度信息
- 让不同大小的图像映射成不同数量的视觉 token
- 通过 Window Attention 降低高分辨率输入时的计算开销

这样做的好处：

- 文档、图表、长图里的细节保留更好
- 小图不会被无意义地放大
- 大图也不必粗暴压缩丢失信息

和 InternVL 常见做法的区别可以概括为：

- `InternVL` 更常见的是把图像切成若干 tile，再送入视觉编码器
- `Qwen2.5-VL` 更强调 native dynamic resolution，让 token 数量随输入分辨率自然变化

直观区别：

- tile 方案更像“先切块，再拼接理解”
- dynamic resolution 更像“按原始尺度直接编码，只是在算力上做窗口化优化”

视频上也类似，本质是把帧序列映射成带时间信息的视觉 token，再交给多模态主干处理。

---

## 3. Qwen 2.4 VL 里面的 MROPE 怎么实现的

这一题很多时候实际指的是 Qwen-VL 系列里的 `M-RoPE / MRoPE` 设计，核心思想是把传统一维 RoPE 扩展到多模态场景。

实现直觉：

- 文本 token 只有一维位置：序列位置
- 图像 token 至少有二维位置：`h, w`
- 视频 token 还有时间维：`t, h, w`

所以 MRoPE 通常会给视觉 token 构造多维 position id，再把 RoPE 作用到这些维度上，让注意力同时感知：

- 文本顺序
- 图像空间位置
- 视频时间顺序

可以把它理解成：

```text
text:  position = (seq)
image: position = (h, w)
video: position = (t, h, w)
```

这样做的价值是：

- 图像里“左上”和“右下”不会只被当成普通序列 token
- 视频里前后帧也能被显式区分
- 多图、多帧和文本混排时，位置关系更自然

面试里不必陷入实现细节，重点说清：`MRoPE 是把 RoPE 从一维文本位置扩展到多维视觉/时空位置。`

---

## 4. Qwen 2.5 VL 的图像 ViT 部分是什么结构，图文 embedding 怎么做对齐的？

按官方技术说明，Qwen2.5-VL 的视觉侧是一个**从头训练的动态分辨率 ViT**，并专门做了高分辨率优化。

可以抓住几个关键词：

- native dynamic-resolution ViT
- 只有少数层使用 full attention
- 大部分层使用 window attention 降低复杂度
- 在结构上引入了更接近 LLM 的 `RMSNorm` 和 `SwiGLU`

图文对齐通常不是一步做完，而是多阶段：

- 先做类似 CLIP 的视觉语义预训练，学图文粗对齐
- 再做 vision-language alignment，把视觉 token 投影到 LLM 可消费的表示空间
- 最后做端到端联合训练，让模型学会图文问答、OCR、定位、视频理解等任务

所以“图文 embedding 怎么对齐”可以答成：

- 视觉 encoder 先提特征
- 通过 projector / alignment 模块把视觉特征映射到语言空间
- 再靠多模态指令训练把跨模态语义真正对齐

面试中这题更重要的是说清**三阶段思路**，而不是死背某层 hidden size。
