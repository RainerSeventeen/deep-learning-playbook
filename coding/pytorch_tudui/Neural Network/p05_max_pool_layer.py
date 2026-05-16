# 最大池化的应用

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x):
        x = self.pool1(x)
        return x

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Model()
    writer = SummaryWriter("../logs")
    step = 0

    for data in dataloader:
        imgs, targets = data
        output = model(imgs)
        writer.add_images("input", imgs, global_step=step)      # 尤其注意是images
        writer.add_images("output", output, global_step=step)
        step += 1

    writer.close()
