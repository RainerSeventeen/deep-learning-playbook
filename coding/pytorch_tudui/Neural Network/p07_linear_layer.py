# 线性层

# in_features (int) – 输入样本的大小
# out_features (int) – 输入样本大小
# bias (bool) – 默认True, 表示是否有偏置值
# y = Ax + b, b 就是偏置

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=196608, out_features=10)

    def forward(self, x):
        x = self.linear1(x)
        return x

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Model()
    for data in dataloader:
        imgs,targets = data
        print(imgs.shape)
        # output = torch.reshape(imgs, (1, 1, 1, -1)) # 让最后一个wide值自行计算
        output = torch.flatten(imgs)  # 将输入展开成为一维向量和上一行作用是一样的

        print(output.shape)
        output = model(output)
        print(output.shape)
        break

