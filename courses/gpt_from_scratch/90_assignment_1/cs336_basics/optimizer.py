from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """优化器初始化步骤

        Args:
            params: 模型本身的所有参数
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 设置默认值, 为每一个未设置的 group 绑定自动的参数
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()   # 外部传入 hook
        for group in self.param_groups: # 遍历所有 group
            lr = group["lr"]
            for p in group["params"]:   # 遍历所有 param
                if p.grad is None:
                    continue
                state = self.state[p]   # state 是优化器的相关状态, 例如步数
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad  # 无梯度更新参数本身, 直接修改 data 属性
                state["t"] = t + 1  # 更新 state
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.99), eps=1e-8):
        # 忽略输入参数的范围检查
        defaults = {"lr": lr, "weight_decay": weight_decay, 
                    "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, weight_decay, eps = group["lr"], group["weight_decay"], group["eps"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0: # 初始化 state 参数
                    state["t"] = 1  # 从 1 开始
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                grad = p.grad.data
                t = state["t"]
                lr_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data -= weight_decay * lr * p.data
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                state["t"] = t + 1

        return loss
