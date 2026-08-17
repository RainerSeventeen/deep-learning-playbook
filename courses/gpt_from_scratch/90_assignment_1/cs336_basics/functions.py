import einops
import torch
import math

from torch import Tensor
from jaxtyping import Bool, Float

class Functions:
    @staticmethod
    def linear(x: Tensor, weight: Tensor):
        """
        没有 bias 的 linear 计算函数
        y = Wx
        """
        return einops.einsum(weight, x, "out in, ... in -> ... out")
    
    @staticmethod
    def sigmoid(x: Tensor):
        """pytorch 底层有个更加复杂的优化, 这里只作为演示"""
        if x >= 0:
            return 1 / (1 + torch.exp(-x))
        else:
            # 为了数值稳定性, 在 x 负数时需要上下同时乘 e^(x)
            exp_x = torch.exp(x)
            return exp_x / (1 + exp_x)
    
    @staticmethod
    def silu(x):
        return x * torch.sigmoid(x)

    @staticmethod
    def glu(x, w1, w2):
        # 实际上 SwiGLU 用不到 GLU
        # 逐元素相乘也可以用 einsum(a, b, "i, i -> i")
        return torch.sigmoid(w1 @ x) * (w2 @ x)
    
    @staticmethod
    def softmax(x: Tensor, dim: int):
        """
        softmax(v_i) = exp(v_i) / sum(exp(v_j))
        指数很容易导致数值溢出, 为了数值稳定性会执行归一化
        利用常数加减不变的性质: softmax(v_i - c) == softmax(v_i)
        """
        # 后续需要逐元素进行广播运算, 需要 keepdim
        max_value = x.max(dim=dim, keepdim=True)
        x = x - max_value.values # 注意取 values 而不是 indices
        x = torch.exp(x)
        return x / x.sum(dim=dim, keepdim=True)
    
    @staticmethod
    def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ):
        """缩放点积注意力"""
        qk = einops.einsum(Q, K, " ... q d_k, ... k d_k -> ... q k")
        d_k = Q.shape[-1]
        qk_norm = qk / math.sqrt(d_k)
        if mask is not None:
            # mask 应用, True 不操作, False 全部换成 -inf
            qk_norm.masked_fill_(~mask, float('-inf'))
        attn_score = Functions.softmax(qk_norm, dim=-1)
        return attn_score @ V
