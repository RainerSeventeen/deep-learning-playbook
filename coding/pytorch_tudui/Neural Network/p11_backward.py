# 反向传播的示例，是loss function在 神经网路中的实际应用

# 训练 CIFAR10 数据集模型
# 训练流程图可以参考
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
    dataset = torchvision.datasets.CIFAR10(root='/root/TestPytorch/dataset', train=True, download=True, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = Model()
    loss = nn.CrossEntropyLoss()

    for data in dataloader:
        inputs, labels = data
        output = model(inputs)
        result_loss = loss(output, labels)
        result_loss.backward() # 反向传播，注意是对结果值进行反向传播，是优化器的使用基础
        break





