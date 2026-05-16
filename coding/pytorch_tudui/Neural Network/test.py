import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = Model().to(device)  # 将模型移动到 GPU
    loss = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到 GPU
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 设置优化器

    for epoch in range(20):
        running_loss = 0.0
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到 GPU

            output = model(inputs)
            result_loss = loss(output, labels)
            optimizer.zero_grad()  # 清空梯度
            result_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += result_loss.item()  # 累加损失值

        print(f"Epoch {epoch}, Loss: {running_loss / len(dataloader)}")  # 打印平均损失