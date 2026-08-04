import einops
import torch

class Functions:
    @staticmethod
    def linear(input, weight):
        """
        没有 bias 的 linear 计算函数
        y = Wx
        """
        return einops.einsum(weight, input, "out in, ... in -> ... out")
    
    @staticmethod
    def sigmoid(x):
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