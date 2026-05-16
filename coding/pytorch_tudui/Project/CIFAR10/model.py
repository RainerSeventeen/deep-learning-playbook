import torch
import torch.utils.tensorboard as tb

from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


class NetworkDemo(nn.Module):
    """
    自行搭建的网络，面向 CIFAR10 数据集
    """    
    def __init__(self):
        super(NetworkDemo, self).__init__()
        self.all = Sequential(
        Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2, stride=1),
        MaxPool2d(kernel_size=2),
        Conv2d(32, 32, 5, padding=2),
        MaxPool2d(kernel_size=2),
        Conv2d(32, 64, 5, padding=2),
        MaxPool2d(2),
        # 展开数据
        Flatten(),
        # 线性
        Linear(in_features=1024, out_features=64),
        Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.all(x)
        return x

if __name__ == "__main__":
    net = NetworkDemo()
    print(net)
    input = torch.ones((64, 3, 32, 32))
    output = net(input)
    print(output.shape)

    # 使用 tb 来查看网络
    # writter = tb.SummaryWriter()
    # writter.add_graph(net, input)
    # writter.close()