import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

# 选择训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import NetworkDemo

if __name__  == "__main__":
    train_data = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=transforms.ToTensor(), download=True)
    
    test_data = torchvision.datasets.CIFAR10(root='../dataset', train=False, transform=transforms.ToTensor(), download=True)

    # 长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)

    print(f"Train: {train_data_size}, Test: {test_data_size}")

    # 加载数据集
    train_data_loader = DataLoader(train_data, batch_size=64)
    test_data_loader = DataLoader(test_data, batch_size=64)

    # 构建模型
    model = NetworkDemo()
    model.to(device)

    # Loss 和 优化器
    loss_func = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 设置训练参数
    train_step = 0
    test_step = 0
    epoch = 10
    
    # TensorBoard
    writer = SummaryWriter("../train/logs")

    for i in range(epoch):
        print(f"------ Train Epoch [{i + 1:2}] ------")
        
        # 训练步骤
        model.train()
        for data in train_data_loader:
            imgs, target = data
            imgs = imgs.to(device)
            target = target.to(device)
            outputs = model(imgs)
            loss = loss_func(outputs, target)
            # 优化器
            optimizer.zero_grad() # 务必清空梯度
            loss.backward()
            optimizer.step()

            train_step += 1
            if train_step % 100 == 0:
                print(f"Iteration [{train_step}], loss {loss.item()}")
                writer.add_scalar("train_loss", loss.item(), train_step)

        # 测试功能
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_data_loader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_func(outputs, targets)
                test_loss += loss

            test_loss = test_loss / len(test_data_loader)
            
        print(f"Test loss average: {test_loss}")
        writer.add_scalar("test_loss_epoch", test_loss.item(), i)

        torch.save(model.state_dict(), f"../train/model/model_{i}.pth")
