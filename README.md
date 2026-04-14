# Deep Learning Playbook

面向深度学习 / AI 算法岗位准备的学习仓库，按「原理笔记」「论文笔记」「面试问题库」「手撕代码」四条主线组织，避免把所有内容堆进一个超长问答文件里。

## 仓库结构

- `foundations/`：主题型知识笔记，回答“原理是什么”
- `papers/`：论文 / 模型精读，回答“论文解决了什么、怎么做、为什么有效”
- `interview/`：面试问题库与专题页，回答“怎么答、怎么追问”
- `coding/`：最小可运行实现，回答“怎么写、怎么验证”
- `references/`：外部资源索引

## 推荐学习路径

1. 先看 `foundations/architectures/` 和 `foundations/training/`，补齐 Transformer、归一化、损失函数、Embedding 等基础。
2. 再看 `papers/transformers/`、`papers/multimodal/`、`papers/training/`，把经典模型放回论文语境中理解。
3. 用 `interview/index.md` 做题目入口，从专题页练习 30 秒 / 3 分钟答题。
4. 遇到需要推导或代码解释的点，再回到 `coding/` 看最小实现。

## 快速入口

- 基础原理：
  - [Activation Functions](foundations/architectures/activation-functions.md)
  - [Attention Basics](foundations/architectures/attention-basics.md)
  - [Normalization](foundations/architectures/normalization.md)
  - [Position Encoding](foundations/architectures/position-encoding.md)
  - [Embedding](foundations/architectures/embedding.md)
  - [Loss Functions](foundations/training/loss-functions.md)
- 论文精读：
  - [Attention Is All You Need](papers/transformers/attention.md)
  - [BERT](papers/transformers/bert.md)
  - [GPT](papers/transformers/gpt.md)
  - [BLIP-2](papers/multimodal/blip2.md)
  - [LoRA](papers/training/lora.md)
- 面试专题：
  - [DL Basics](interview/deep-learning/basics.md)
  - [Transformer](interview/deep-learning/transformer.md)
  - [RAG](interview/systems/rag.md)
  - [Agent](interview/systems/agent.md)
  - [Python 并发与服务](interview/python/concurrency-and-service.md)
  - [场景题](interview/scenarios/case-studies.md)
- 代码实现：
  - [coding/README.md](coding/README.md)

## 当前 V1 覆盖范围

V1 先交付：

- 一套可导航的目录骨架与索引页
- 6 篇基础笔记
- 5 篇论文笔记
- 6 个面试专题页
- 3 个最小可运行代码实现

后续扩展时，优先把问题挂到 `interview/index.md` 和对应专题页；只有当内容足够稳定、会被多处复用时，再拆成独立知识文档。
