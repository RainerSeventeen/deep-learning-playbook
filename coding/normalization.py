"""归一化算法

常见的 Normalization 算法：
- Batch Norm (BN)
- Layer Norm (LN)
- RMS Norm
- Instance Norm
- Group Norm
- Weight Norm
"""

import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    """对 mini-batch 的每个特征维度做归一化。

    训练时：均值/方差来自当前 batch，并用 momentum 更新 running 统计量。
    推理时：使用积累的 running_mean / running_var。

    输入形状：(N, C) 或 (N, C, H, W)
    统计维度：N（以及 H、W），即对每个通道 C 独立归一化。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # 可学习的仿射参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # 推理时使用的 running 统计量（不参与梯度）
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 把通道维移到最后，方便广播
        is_4d = x.dim() == 4  # (N, C, H, W)
        if is_4d:
            N, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)

        if self.training:
            mean = x.mean(dim=0)                           # (C,)
            var  = x.var(dim=0, unbiased=False)            # (C,)
            # 更新 running 统计量（PyTorch 官方用无偏方差更新 running_var）
            var_unbiased = x.var(dim=0, unbiased=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var_unbiased
        else:
            mean = self.running_mean
            var  = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta

        if is_4d:
            out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return out


class LayerNorm(nn.Module):
    """对单个样本的最后 len(normalized_shape) 个维度做归一化。

    不依赖 batch 大小，适合 NLP / Transformer。

    输入形状：任意，通常 (N, seq_len, d_model)
    统计维度：normalized_shape 对应的最后几个维度。
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在最后 len(normalized_shape) 个维度上求均值和方差
        dims = tuple(range(-len(self.normalized_shape), 0)) # 例: (-3, -2, -1, )
        mean = x.mean(dim=dims, keepdim=True)
        var  = x.var(dim=dims, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class RMSNorm(nn.Module):
    """RMS Norm

    输入张量 (B, T, C)
    """
    def __init__(self, feat_count, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feat_count))

    def forward(self, input):
        rms = torch.sqrt((input ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return input / rms * self.gamma
