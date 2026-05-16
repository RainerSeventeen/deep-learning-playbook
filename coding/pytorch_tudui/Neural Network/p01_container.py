# 1.torch.nn.Module 最基本的神经网络容器

import torch
import torch.nn as nn

# 重写一个神经网络模块
class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        output = x + 1
        return output


if __name__ == '__main__':
    model = MyModel()
    x = torch.tensor(1.0)
    output = model(x)
    print(output)
