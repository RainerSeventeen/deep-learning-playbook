# 1.(torchvision)从网络上获取一个已经按照规范标定完成的Dataset
# 2.(dataloader) 加载数据并放置于tensorboard上

from sympy.abc import q
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# DataLoader 参数设置
# dataset (Dataset): 数据库参数
# batch_size: int 表示每次读取数据数量
# shuffle: bool 表示是否打乱顺序
# num_workers：int 子进程个数，用于读取数据
# drop_last: int 是否舍去余下来的数据



if __name__ == '__main__':
    # 准备测试数据集
    test_data = torchvision.datasets.CIFAR10("./dataset", train=False,
                                             download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

    # 测试数据集中 第1张图片的target
    img, target = test_data[0] # 是 __getitem__ 的返回值
    print(img.shape)


    # dataloader使用方法,(结合TensorBoard)
    writer = SummaryWriter("logs") # 指定路径logs
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)  # 2张图片，3通道，32 * 32 尺寸
        # print(targets)
        writer.add_images("test_loader", imgs, step)
        step += 1

    writer.close()

