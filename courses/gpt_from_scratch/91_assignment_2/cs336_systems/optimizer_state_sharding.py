from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """按参数分配优化器状态，更新后广播参数（ZeRO stage 1）。

    所有进程须以相同顺序提供、添加参数组，并在 step 前完成梯度同步。
    param_groups 保留全部参数，因此继承的 zero_grad 会清理整个模型。
    """

    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._owners: dict[torch.Tensor, int] = {}
        self._shard_sizes = [0] * self.world_size
        self.optimizer: Optimizer | None = None
        super().__init__(params, defaults=kwargs)

        # 即使当前进程没有参数，也传入空参数组，避免空参数列表报错。
        self.optimizer = optimizer_cls([self._local_group(group) for group in self.param_groups], **kwargs)
        self.defaults = self.optimizer.defaults
        for group, local_group in zip(self.param_groups, self.optimizer.param_groups):
            group.update({key: value for key, value in local_group.items() if key != "params"})
        self.state = self.optimizer.state

    def _local_group(self, group: dict[str, Any]) -> dict[str, Any]:
        return {
            **group,
            "params": [param for param in group["params"] if self._owners[param] == self.rank],
        }

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for param in group["params"]:
            if param not in self._owners:
                # 按元素数量贪心分配；已分配的参数不迁移，以保留已有状态。
                owner = min(range(self.world_size), key=self._shard_sizes.__getitem__)
                self._owners[param] = owner
                self._shard_sizes[owner] += param.numel()
        if self.optimizer is not None:
            self.optimizer.add_param_group(self._local_group(group))

    def step(self, closure=None, **kwargs):
        # 同步外部对学习率等选项的修改，例如学习率调度器的更新。
        for group, local_group in zip(self.param_groups, self.optimizer.param_groups):
            local_group.update({key: value for key, value in group.items() if key != "params"})
        loss = self.optimizer.step(closure=closure, **kwargs)
        if self.world_size > 1:
            with torch.no_grad():
                for param, owner in self._owners.items():
                    dist.broadcast(param, src=owner)
        return loss
