import torch
import torchvision

if __name__ == "__main__":
    
    # 1 保存整个模型，结构+参数
    vgg16 =torchvision.models.vgg16()
    # torch.save(vgg16, "./model/vgg16.pth")

    # 读取文件
    model = torch.load("./model/vgg16.pth", weights_only=False)
    print(model)

    # 2 保存为字典，只有参数
    torch.save(vgg16.state_dict(), "./model/vgg16_dict.pth")
    model_dict = torch.load("./model/vgg16_dict.pth")
    vgg16.load_state_dict(model_dict)
    print(model_dict)
    print(vgg16)