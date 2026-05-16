# 2. (2D)卷积函数参数讲解 torch.nn.functional.conv2d()
    # input: 输入tensor参数，(minibatch, in_channel, H, W)
    # weight: 卷积核，被卷积的一个矩阵，会反复的移动 (需要有channel参数)
    # bias: 偏置值
    # stride: 卷积移动的步长（默认为1），也可以是一个元组(a,b)分别控制横向和纵向步长
    # padding: 填充数值为0的行列，可以用元组指定，也可以是string的"same"表示控制输出同一个大小，只能在stride = 1时使用

import torch
if __name__ == '__main__':
    # 输入参数
    input = torch.tensor([[1,2,0,3,1],
                          [1,0,3,2,1],
                          [0,1,3,3,2],
                          [0,1,3,1,2],
                          [0,2,0,2,1]])

    # 卷积核
    kernel = torch.tensor([[1,2,1],
                           [0,2,1],
                           [2,1,0]])

    # print(input.shape)
    input = torch.reshape(input,(1, 1, 5, 5)) # 维度变换，更改为符合条件的输入
    # print(input.shape)

    output = torch.nn.functional.conv2d(input, kernel,stride=(1,1))
    print(output)

