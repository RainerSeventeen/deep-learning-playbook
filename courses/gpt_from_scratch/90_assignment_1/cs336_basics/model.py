
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