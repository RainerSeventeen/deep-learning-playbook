import torch
from torch import Tensor

from .functions import Functions as F


def sample_next_token(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    """
    Sample the next token from model logits with temperature and top-p sampling.
    实际作用等同于作业中的 decode

    Args:
        logits: 末维为词表维度的原始模型输出
        temperature: 大于 0；小于 1 会放大差距, 大于 1 会缩小差距
        top_p: 位于 (0, 1]；保留累计概率首次达到该值的 token 及其之前的 token
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")

    probs = F.softmax(logits / temperature, dim=-1)

    # 排序后计算累加概率
    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True,
    )
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 注意到需要保留第一次超过 top_p 的 token, 所以整体再右移一位
    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    # 归一化, 让概率和为 1
    norm_sum = torch.sum(sorted_probs, dim=-1, keepdim=True)
    sorted_probs = sorted_probs / norm_sum
    # torch.multinomial 仅接受一维或二维输入；展平前导维度后再恢复原始形状。
    vocab_size = sorted_probs.shape[-1]
    flat_probs = sorted_probs.reshape(-1, vocab_size)
    sampled_sorted_idx = torch.multinomial(flat_probs, num_samples=1).reshape(
        *logits.shape[:-1], 1
    )
    # gather 固定某一个维度, 按照 idx 去取元素
    token_id = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)
    
    return token_id
