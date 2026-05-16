# 非线性激活，详细参见 https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

# 以ReLU作为例子，截断变换
# 实际包含很多种变换，详细参见nn指导文档

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.nonlinear = torch.nn.ReLU(inplace=False) # 创建个新数组，并非原地处理

    def forward(self, x):
        return self.nonlinear(x)

if __name__ == '__main__':
    input = torch.tensor([[-1, 2, 0, -3, 1],
                          [1, 0, 3, 2, 1],
                          [0, 1, -3, 3, 2],
                          [0, 1, 3, 1, -2],
                          [0, -2, 0, 2, 1]])
    input = torch.reshape(input, (-1, 1, 5, 5))
    model = Model()
    output = model(input)
    print(output)
