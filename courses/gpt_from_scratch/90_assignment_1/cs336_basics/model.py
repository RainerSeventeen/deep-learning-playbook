
import torch
import math
import einops
from torch import nn
from .functions import Functions as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """没有 bias 的线性层"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # y = Wx, 第二个维度是 in_feature
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), 
                        device=device, dtype=dtype),)

    def set_weights(self, weight=None):
        if weight is None:
            # 截断初始化
            sigma = math.sqrt(2 / (self.in_features + self.out_features))
            nn.init.trunc_normal_(self.weight, mean=0, std=sigma,
                                a=(-3 * sigma), b=(3 * sigma))
        else:
            with torch.no_grad():
                self.weight.copy_(weight)

    def forward(self, input):
        return F.linear(input, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """ Embedding 对于给定的词表 index 取对应的行
        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        
        # 词表矩阵
        self.matrix = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim),
                        device=device, dtype=dtype))

    def set_weights(self, weight):
        with torch.no_grad():
            self.matrix.copy_(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # pytorch 支持列表作为索引
        return self.matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gain = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def set_weights(self, weight):
        with torch.no_grad():
            self.gain.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32) # 计算 sqrt 为了防止溢出, 需要缩放数值
        # RMS 求均值, (batch_size, sequence_length)
        rms = torch.sqrt(self.eps + x.square().mean(dim=-1, keepdim=True))
        result = x / rms * self.gain
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """
    SwiGLU 是 Feed Forward Network 的一种实现, 由 SiLU 和 GLU 实现
    SiLU (x) = x * sigmoid(x)
    GLU(x, W1, W2) = sigmoid(W1x) * W2x, * 是逐元素相乘
    SwiGLU = W2(SiLU(W1x) * W3x), * 是逐元素相乘
    """
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff    # d_ff 一般是 d_model 的 3/8
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

    def set_weights(self, w1_weight, w2_weight, w3_weight):
        with torch.no_grad():
            self.w1.copy_(w1_weight)
            self.w2.copy_(w2_weight)
            self.w3.copy_(w3_weight)

    def forward(self, x):
        # 不使用矩阵乘法, 需要对最后维度进行投影
        hidden = F.silu(F.linear(x, self.w1)) * F.linear(x, self.w3)
        return F.linear(hidden, self.w2)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be an even number")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算 cos 和 sin 的 ik 序列
        # theta_i_k = i * (theta ^ (-2k/d)), i 是 position, k 是 二分之一序号(从 0 开始)
        k = torch.arange(0, d_k // 2, device=device)  # 构造 k 序号序列, 0 ~ d_k // 2 - 1
        freq = theta ** (-2 * k / d_k)  # 构造频率序列, 注意 k 序列是从 0 开始的
        position = torch.arange(max_seq_len, device=device)
        theta_i_k = einops.einsum(position, freq, "i, k -> i k")
        # 等价于利用广播 theta_i_k = position[:, None] * freq[None, :]

        self.register_buffer("cos", torch.cos(theta_i_k), persistent=False)
        self.register_buffer("sin", torch.sin(theta_i_k), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)
        """
        # 1. 对 d_k 两两切分 -> (..., seq_len, d_k // 2, 2)
        x = einops.rearrange(x, "... l (d two) -> ... l d two", two=2)
        # 拆分向量
        x0 = x[..., 0]
        x1 = x[..., 1]
        # 抽取
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # 实际上运算不构造矩阵, 而是单独计算元素
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        out = torch.stack([out0, out1], dim=-1) # 沿着最后一个维度拼接, 并生成新的维度
        return einops.rearrange(out, "... l d two -> ... l (d two)")


class MulitiHeadAttention(nn.Module):
    """多头注意力, 这里假设 QK 的 d_k 和 V 的 d_v 维度是相同的"""
    def __init__(self, d_model, num_heads, apply_rope=False, token_positions=None,
                theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_model % num_heads:
            raise ValueError("d_model cannot be devided by num_heads")
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.apply_rope = apply_rope
        self.token_positions = token_positions
        if self.apply_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

    def set_weights(self, Wq, Wk, Wv, Wo):
        self.q_proj.set_weights(Wq)
        self.k_proj.set_weights(Wk)
        self.v_proj.set_weights(Wv)
        self.o_proj.set_weights(Wo)

    def forward(self, Q, K, V):
        """ casual masked attention
        Args:
            Q : [" ... queries d_model"]
            K : [" ... keys d_model"]
            V : [" ... queries d_model"]
        """
        Q = self.q_proj(Q)
        K = self.k_proj(K)
        V = self.v_proj(V)
        # 注意这里要将 head 的维度移动到前面去, 为了符合缩放点积的约定
        Q = einops.rearrange(Q, "... q (n d) -> ... n q d", n=self.num_heads)
        K = einops.rearrange(K, "... q (n d) -> ... n q d", n=self.num_heads)
        V = einops.rearrange(V, "... q (n d) -> ... n q d", n=self.num_heads)

        if self.apply_rope:
            # RoPE 不需要应用到 V 上
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)

        # 下三角矩阵设置为 1, 不偏移 (也就是包含对角线)
        seq_len = Q.shape[-2]
        mask = torch.ones((seq_len, seq_len))
        mask = torch.tril(mask, diagonal=0)
        mask = mask.to(torch.bool)

        out = F.scaled_dot_product_attention(Q, K, V, mask)
        out = einops.rearrange(out, "... n q d -> ... q (n d)")
        return self.o_proj(out)

