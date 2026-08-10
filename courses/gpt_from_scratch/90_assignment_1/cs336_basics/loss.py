import torch
from torch import Tensor

def cross_entropy(inputs: Tensor, targets: Tensor):
    """
    Args:
        inputs: "batch_size vocab_size"
        targets: "batch_size"
    """
    max_logits = inputs.max(dim=-1, keepdim=True).values
    inputs = inputs - max_logits
    # 构造高级索引, batch 维度要手动构造一个递增序列
    batch_idx = torch.arange(inputs.shape[0])
    target_logits = inputs[batch_idx, targets]
    # 第二项不可以 keepdim
    loss =- target_logits + torch.log(torch.exp(inputs).sum(dim=-1))
    return loss.mean()
