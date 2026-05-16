# 优化器 官方文档参见：https://pytorch.org/docs/stable/optim.html

# torch.optim
# 主要有两个参数
# para: 模型的参数
# lr: learning rate 学习速率
# 其他，参见各自优化器的算法


import torch
from torch import nn
import  torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    data_path = '/Users/rainer/MyFiles/Code/Learning/Pytorch/dataset/cifar-10-batches-py'
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = Model()
    loss = nn.CrossEntropyLoss() # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # 设置优化器

    for epoch in range(20):
        # 假设进行20轮训练，则可以观察每一轮后的loss值为多少
        running_loss = 0.0
        for data in dataloader:
            inputs, labels = data
            output = model(inputs)
            result_loss = loss(output, labels)
            optimizer.zero_grad() # 首先清空梯度，避免之前的循环造成影响
            result_loss.backward() # 反向传播，注意是对结果值进行反向传播，是优化器的使用基础，改变了梯度值
            optimizer.step() # 利用梯度进行反向传播
            running_loss += result_loss.item() # 计算这一轮中所有loss的总和
        print(epoch, running_loss)
