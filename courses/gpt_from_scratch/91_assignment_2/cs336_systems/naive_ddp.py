import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


class NaiveDDPModule(torch.nn.Module):
    """包装模型，在反向传播完成后逐参数同步梯度。"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        with torch.no_grad():
            for tensor in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(tensor, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # 假设各进程使用相同的参数，且所有可训练参数都有梯度。
        for param in self.module.parameters():
            if param.requires_grad:
                if param.grad is None:
                    raise RuntimeError("所有可训练参数都必须有梯度")
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(dist.get_world_size())


class NaiveDDP:
    """单进程对应一张 GPU，反向传播结束后逐个同步参数梯度。

    假设每个进程每步的 batch size 相同，loss 是本地样本的平均值，
    并且所有需要训练的参数在每一步都参与反向传播。
    """

    def __init__(self, global_rank, local_rank, world_size, backend="nccl"):
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}" if backend == "nccl" else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        dist.init_process_group(backend, rank=global_rank, world_size=world_size)
        self.model = None
        self.opti = None
        self.loss = None
        self.sampler = None
        self.dataloader = None

    def setup(self, model, loss, lr=0.001):
        self.model = model.to(self.device)
        # 所有进程都参与广播，rank 0 提供初始参数和缓冲区。
        with torch.no_grad():
            for tensor in list(self.model.parameters()) + list(self.model.buffers()):
                dist.broadcast(tensor, src=0)
        self.opti = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss = loss

    def get_data(self, dataset, batch_size):
        # 丢弃不足一个全局 batch 的样本，保证各进程步数和 batch size 一致。
        self.sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.global_rank,
            shuffle=True, drop_last=True,
        )
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=self.sampler, drop_last=True,
        )
        return self.dataloader

    def train_step(self, batch):
        self.model.train()
        x, y = (tensor.to(self.device) for tensor in batch)
        self.opti.zero_grad()
        loss = self.loss(self.model(x), y)
        loss.backward()

        # 各进程必须按相同顺序调用 collective；平均后再更新参数。
        for param in self.model.parameters():
            if param.requires_grad:
                if param.grad is None:
                    raise RuntimeError("假设所有可训练参数每步都有梯度")
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)
        self.opti.step()
        return loss.detach()

    def train_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        total_loss = torch.zeros(2, device=self.device)
        for batch in self.dataloader:
            loss = self.train_step(batch)
            batch_size = batch[0].size(0)
            total_loss[0] += loss * batch_size
            total_loss[1] += batch_size
        # 所有进程参与统计，返回按样本加权的全局平均 loss。
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if total_loss[1].item() == 0:
            raise ValueError("数据集不足一个全局 batch")
        return (total_loss[0] / total_loss[1]).item()

    def save_checkpoint(self, path, epoch):
        if self.global_rank == 0:
            torch.save({"model": self.model.state_dict(),
                        "optimizer": self.opti.state_dict(), "epoch": epoch}, path)

    def load_checkpoint(self, path):
        # 每个进程读取同一份检查点；path 需要对所有进程可见。
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.opti.load_state_dict(checkpoint["optimizer"])
        return checkpoint["epoch"]

    def cleanup(self):
        dist.destroy_process_group()
