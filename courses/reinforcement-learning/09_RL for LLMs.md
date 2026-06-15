# 09 大语言模型中的强化学习

>  在 LLM 中，一般经常让 LLM 完成句子的续写，其中隐含了大模型有一个对“世界知识” 的认知，不论是代码撰写还是文学创作等。

## Instruction Finetuning

> 指令微调，在 LLaVA 论文中就有所涉及

在 LLM 在 pretrain 完毕知乎，他只会进行句子的续写，在语料库中找到对应的 token 的分布然后选择最高的那个。尽管这个句子可能有一点逻辑，但是作为一个助手来讲，这个回答的方式并不是很好，因此需要指令微调额工作。

实验数据表明：对于同一个指令微调的库，模型本身的参数越大，则模型经过微调后的效果会越好。

指令微调具体的做法就是，收集一些 `(instruction, output)` 的回答，然后对模型进行进一步的微调。

这个方法也有几个问题：

1. 这个数据非常的“昂贵”，也就是网络或者文学中比较难找到对应的数据。
2. 有的开放问题是没有正确答案的，例如文学创作等
3. LLM 对于每一个 token 都是平等地对待的，但是有的答案是更加错误的
4. 如果完全使用 人类的回答作为数据，那么正如 BC 一样，LLM 的能力将无法超过人类。

正因为这几个问题，所以我们引出了 RLHF，从人类的偏好中学习。

## RLHF

### 基础

整体流程大致为，首先训练一个模型，模拟人类对于语言的偏好，然后再对模型最大化这个 reward。

这里使用 RL 的主要原因是，人类的评价不是一个可微的结果，所以只能用 RL 来进行优化。

模型参数是 $\theta$，它会按照自己的概率分布 $p_\theta(s)$ 生成一个输出 $\hat{s}$，然后奖励函数 $R4 会给这个输出打分，而 RL 的目标就是最大化这个函数值
$$
\mathbb{E}_{\hat{s} \sim p_\theta(s)}[R(\hat{s})]
$$
考虑到为了让奖励最大化，使用 Policy gradient  算法。

注意到这里的 $R$ 函数实际上是人类给出的，可能是不可微的。因此不对奖励函数 $R$ 求导，只需要对模型生成该样本的 log probability 求导（详细推导参见梯度策略章节），最后可以得到：
$$
\nabla_\theta \mathbb{E}_{\hat{s}\sim p_\theta(s)}[R(\hat{s})]
=
\mathbb{E}_{\hat{s}\sim p_\theta(s)}
\left[
R(\hat{s})\nabla_\theta \log p_\theta(\hat{s})
\right]
$$
总结来说，如果一个样本的 reward 很高就提高对应的概率。

回顾一下几个问题：

1. 人类数据很昂贵：训练一个模型学习人类的偏好，然后用模型来执行 RL。
2. 人类的评判是有噪声和校准的基准：询问人类的对比上的偏好，而不是直接要求打分。

### 算法

已有一个已经预训练好的语言模型 $p^{PT}(y \mid x)$ 以及一个奖励模型 $RM_\phi(x,y)$ ，面向 $\mathbb{E}_{\hat{y} \sim p_\theta^{RL}(\hat{y}\mid x)}\left[RM_\phi(x,\hat{y})\right]$ 目标进行优化。

回忆一下之前 RL 的内容，reward model 的模型本来就是有缺陷的，所以为了避免模型走火入魔需要限制更新的距离：
$$
\mathbb{E}_{\hat{y}\sim p_\theta^{RL}(\hat{y}|x)}
\left[
RM_\phi(x,\hat{y})
-
\beta
\log
\left(
\frac{
p_\theta^{RL}(\hat{y}|x)
}{
p^{PT}(\hat{y}|x)
}
\right)
\right]
$$

> **现在我们有一个 可以微分的 reward model 了，为什么我们不直接进行 反向传播，而是还有进行这种 RL 呢？**

原因是语言模型生成 token 得到行为是离散采样的，我们无法对 token 来进行反向传播，也就是 
$$
\theta
\rightarrow p_\theta(\hat y|x)
\rightarrow \hat y
\rightarrow RM_\phi(x,\hat y)
$$
过程中的 $p_\theta(\hat y|x) \rightarrow \hat y$ 是不可微分的，所以我们要使用 RL 来绕过这个步骤。

## DPO