# MoE

## 1. Dense、MMOE、MoE 这类结构各自是什么？Dense 和 MoE 的区别在哪里？

先区分三个概念：

- `Dense`：每个 token 都走同一套参数，所有层都全量计算
- `MoE`：每个 token 只激活少数几个专家，属于稀疏激活
- `MMoE`：多任务场景下的多门控专家结构，不同任务有不同 gate

Dense 和 MoE 的核心区别：

| 维度 | Dense | MoE |
| --- | --- | --- |
| 每个 token 用多少参数 | 全部共享参数 | 只走部分专家 |
| 计算量 | 随总参数一起涨 | 可在较低 FLOPs 下提升总容量 |
| 表达方式 | 单一路径 | 条件路由、多路径 |
| 工程难点 | 相对简单 | 路由、负载均衡、并行通信更复杂 |

一句话：`MoE 用“稀疏激活”换“更大模型容量”。`

---

## 2. 介绍一下基于 MoE 的模型架构：router、shared expert / routed expert、top-k 路由、capacity factor 分别起什么作用？

典型 MoE 层一般是：

`输入 token -> router 打分 -> 选 top-k expert -> expert 计算 -> 加权聚合输出`

几个核心组件：

- `router`：给每个 token 计算该送到哪些 expert
- `routed expert`：真正参与路由竞争的专家
- `shared expert`：所有 token 都会经过的共享专家，补充通用能力
- `top-k 路由`：每个 token 只选分数最高的 `k` 个专家
- `capacity factor`：限制每个 expert 最多接多少 token，防止某些 expert 爆满

各自作用：

- router 决定“谁擅长处理当前 token”
- shared expert 负责稳定的公共知识
- routed expert 负责稀疏、专门化能力
- top-k 决定稀疏程度和计算量
- capacity factor 决定负载上限和丢 token 风险

---

## 3. 标准 MoE 和 MMoE 有什么区别？分别适用于什么场景？

标准 MoE 和 MMoE 最大区别在“有没有多任务视角”。

| 维度 | 标准 MoE | MMoE |
| --- | --- | --- |
| 使用目标 | 提升单任务或通用模型容量 | 多任务学习 |
| gate | 通常一套路由 | 每个任务一个 gate |
| expert 共享方式 | token 级共享 | 多任务共享 + 任务特定选择 |
| 常见场景 | LLM、稀疏 Transformer | 推荐、广告、CTR/CVR 多任务 |

适用场景：

- `标准 MoE`：更适合大模型、通用建模、语言模型扩容
- `MMoE`：更适合多个相关但不完全相同的任务一起学

一句话：`MoE 更偏模型扩容，MMoE 更偏多任务知识共享。`

---

## 4. MoE 的专家数量通常如何选择？应该从任务相关性、模型容量、路由负载和过拟合风险等角度如何分析？

专家数量没有固定答案，通常要同时看四件事：

- 任务是否天然存在多子分布
- 当前 dense 模型是不是已经容量不足
- 集群通信和路由是否承受得住
- 数据量是否足够支撑专家专门化

分析框架可以这样答：

- 任务相关性高、模式差异明显：可以多一些 expert
- 数据量不大：expert 太多容易每个专家都吃不到足够样本
- 路由负载不均：expert 太多会放大空转和通信问题
- 过拟合风险高：过多专家可能学成“记忆分工”而不是泛化分工

实际选择时常做：

- 固定总 FLOPs，比几组 expert 数量
- 看验证集效果
- 看各 expert token 占比是否均衡
- 看是否出现大量 token dropping

所以不是专家越多越好，而是要看**容量收益是否大于路由代价**。

---

## 5. 如果基于 MoE 的模型在训练时负载均衡不好怎么办？通常看哪些监控指标，辅助 loss、capacity factor、token dropping、路由温度等手段分别解决什么问题？

先看监控，再谈优化。

常见监控指标：

- 每个 expert 接收到的 token 占比
- router 概率分布熵
- token dropping 比例
- 各 expert 的实际利用率
- 通信耗时和 step time

常见问题和对应手段：

| 手段 | 主要解决什么问题 |
| --- | --- |
| 辅助负载均衡 loss | 防止少数 expert 吃掉大多数 token |
| 调整 capacity factor | 缓解 expert 爆满或浪费容量 |
| token dropping 策略 | 当 expert 超载时控制计算和稳定性 |
| 路由温度调节 | 控制 router 分布过尖或过平 |
| top-k 调整 | 在表达力、通信量和均衡性之间折中 |

排查顺序通常是：

1. 先看是不是 router 过于尖锐，导致头部 expert 垄断
2. 再看 capacity 是否太小，造成大量 token 被挤掉
3. 再看辅助 loss 权重是不是太弱或太强
4. 最后看数据分布本身是否极不均衡

一句话：`MoE 训练里最怕的不是专家不够多，而是路由只会用少数几个专家。`
