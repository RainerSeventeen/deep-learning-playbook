# 卷积层的应用
from codecs import namereplace_errors

# torch.nn.Conv2d()
# 如果设置out_channel = 2,会生成2个卷积核（in_channel = 1）
# in_channels/out_channels: int 通道数
# kernel_size: int/tuple 卷积核大小，可以是元组
# stride: int/tuple
# padding: 填充的格式，可以用元组指定 H，W
# padding_mode: 填充的方式

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True) # 装载数据

    model = Model()
    # print(model)

    writer = SummaryWriter("../logs")
    step = 0


    # 开始卷积层
    for data in dataloader:
        imgs,labels = data
        # print(imgs.shape)   # torch.Size([64, 3, 32, 32])

        output = model(imgs) # 应用卷积层

        print(imgs.shape)
        writer.add_images('input', imgs, global_step=step)

        output = torch.reshape(output,(-1, 3, 30, 30))
        # 注意：多个channel压缩到少量channel，多余的信息会存储与 batch size 中，也就是第一个值需要自行计算，设置为-1
        writer.add_images('output', output, global_step=step) # 不可以显示6通道的图像
        step += 1
