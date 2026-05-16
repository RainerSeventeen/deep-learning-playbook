# 最大池化层 max_pool
# 作用：在保留数据特征值的情况下，减少数据的数值

# 池化操作(最大池化)：取池化范围内最大值

# torch.nn.MaxPool2d()
# input 注意应该是四维array
# kernel_size : 池化核大小
# stride : 窗口的步长,默认是池化核的大小
# padding : 周边填充
# dilation : 卷积中插入的数值，一般不用
# ceil_mode(默认False) : ceil模式的开关，向上取整，如果True则保留外围一圈;floor是向下取整
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.pool1(x)
        return x


if __name__ == '__main__':
    input = torch.tensor([[1,2,0,3,1],
                          [1,0,3,2,1],
                          [0,1,3,3,2],
                          [0,1,3,1,2],
                          [0,2,0,2,1]], dtype=torch.float32)
    input = torch.reshape(input,(-1, 1, 5, 5))
    print(input.shape)

    model = Model()
    output = model(input)
    print(output)
