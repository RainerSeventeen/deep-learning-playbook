# 损失方程，计算预期输出和实际输出之间的损失，为更新输出提供依据

import torch

inputs = torch.tensor([0, 1, 2],dtype=torch.float32)
targets = torch.tensor([0, 1, 5],dtype=torch.float32)

loss = torch.nn.MSELoss(reduction='mean')
result = loss(inputs, targets)
print(result)

# 1.L1Loss() 相减
# 2.MSELoss() 差的平方，方差MSE
# 3.CrossEntropyLoss() 交叉熵损失，对于多个类别有用 https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

# x_output = [a, b, c] 分别对应class类中对应目标的可能性,例如[0.1, 0.8, 0.1]，更有可能是class = 1

x_output = torch.tensor([0.1, 0.8, 0.1])
# 需要变型为 (N,C) ，其中N是batch_size, C是class数目
x_output = torch.reshape(x_output, (1, 3))
targets = torch.tensor([1]) # 第二个class

entropy_loss = torch.nn.CrossEntropyLoss()
entropy_result = entropy_loss(x_output, targets)
print(entropy_result)
