# Deep Learning Playbook

面向深度学习 / AI 算法岗位准备的学习仓库，按「原理笔记」「论文笔记」「面试问题库」「手撕代码」四条主线组织，避免把所有内容堆进一个超长问答文件里。代码练习部分保留仓库内的最小实现，判题能力通过 `libs/pyre-code` 子模块接入。

## 仓库结构

- `foundations/`：主题型知识笔记，回答”原理是什么”（平铺文件，无子目录）
- `papers/`：论文 / 模型精读，回答”论文解决了什么、怎么做、为什么有效”
  - `transformers/`：Transformer 系列
  - `multimodal/`：多模态模型
  - `training/`：训练技术（微调、对齐、表示学习）
- `interview/`：面试问题库
- `coding/`：最小可运行实现（Python 脚本 + notebooks）
- `libs/`：外部依赖与子模块，当前包含 `pyre-code` 代码判题能力
- `references/`：外部资源索引（待补充）

## 推荐学习路径

1. 先看 `foundations/`，补齐激活函数、归一化、损失函数、Embedding、位置编码等基础。
2. 再看 `papers/transformers/`、`papers/multimodal/`、`papers/training/`，把经典模型放回论文语境中理解。
3. 用 `interview/questions.md` 做题目入口，练习面试问答。
4. 遇到需要推导或代码解释的点，再回到 `coding/` 看最小实现。

## 快速入口

- 基础原理：
  - [Activation Function](foundations/Activation%20Function.md)
  - [Normalization](foundations/Normalization.md)
  - [Position Encoding](foundations/Position%20Encoding.md)
  - [Embedding](foundations/Embedding.md)
  - [Loss Function](foundations/Loss%20Function.md)
- 论文精读：
  - Transformers：[Attention Is All You Need](papers/transformers/attention.md)、[BERT](papers/transformers/bert.md)、[GPT](papers/transformers/gpt.md)、[ViT & CLIP](papers/transformers/vit-clip.md)
  - Multimodal：[BLIP-2](papers/multimodal/blip2.md)、[LLaVA](papers/multimodal/llava.md)、[Qwen-VL](papers/multimodal/qwen-vl.md)
  - Training：[LoRA](papers/training/lora.md)、[RL / PPO / DPO](papers/training/rl-ppo-dpo.md)、[Sentence-BERT](papers/training/sentence-bert.md)
- 面试问题：
  - [questions.md](interview/questions.md)
- 代码实现：
  - [attention.py](coding/attention.py)
  - [normalization.py](coding/normalization.py)
  - [position_encoding.py](coding/position_encoding.py)

## 子模块初始化

首次克隆仓库后，如果需要使用 `pyre-code`：

```bash
git submodule update --init --recursive
```

## 当前覆盖范围

- 5 篇基础笔记
- 10 篇论文笔记
- 1 个面试问题库
- 3 个最小可运行代码实现

后续扩展时，优先把问题挂到 `interview/questions.md`；只有当内容足够稳定、会被多处复用时，再拆成独立知识文档。
