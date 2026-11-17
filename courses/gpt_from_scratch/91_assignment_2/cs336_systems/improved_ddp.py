import torch
import torch.distributed as dist

class ImprovedDDPModule(torch.nn.Module):
    """将梯度合并为一个张量，在反向传播后同步。"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        with torch.no_grad():
            for tensor in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(tensor, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # 假设所有进程使用相同的参数，所有可训练参数每步都有梯度。
        grads = []
        for param in self.module.parameters():
            if param.requires_grad:
                if param.grad is None:
                    raise RuntimeError("所有可训练参数都必须有梯度")
                grads.append(param.grad)
        if not grads:
            return
        if any(g.is_sparse or g.dtype != grads[0].dtype or g.device != grads[0].device for g in grads):
            raise ValueError("合并梯度要求所有梯度为同设备、同 dtype 的稠密张量")
        # 合并后只通信一次，再按原形状写回各参数的梯度。
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(dist.get_world_size())
        for grad, reduced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(reduced)


class AsyncDDPModule(torch.nn.Module):
    """梯度计算完成后立即异步通信，与其余反向计算重叠。"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        with torch.no_grad():
            for tensor in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(tensor, src=0)
        self._pending = []
        # 注册在梯度计算完的 hook
        self._hooks = [
            param.register_post_accumulate_grad_hook(self._sync_gradient)
            for param in self.module.parameters() if param.requires_grad
        ]

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _sync_gradient(self, param):
        # 每个参数的梯度累积完成后立即启动通信，结果保留到 finish 中处理。
        grad = param.grad
        if grad.is_sparse:
            raise ValueError("异步梯度同步仅支持稠密梯度")
        work = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
        self._pending.append((work, grad))

    def finish_gradient_synchronization(self):
        # 应当在 optimizer.step() 之前调用
        for work, grad in self._pending:
            work.wait()
            grad.div_(dist.get_world_size())
        self._pending.clear()
