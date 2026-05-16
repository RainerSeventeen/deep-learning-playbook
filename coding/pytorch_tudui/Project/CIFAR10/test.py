import cv2
import torchvision
import torch
from PIL import Image

from model import NetworkDemo

image_path = "../dataset/test/plane.jpeg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用 PIL 读取
img = Image.fromarray(img)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
])

img = transform(img)
print(img.shape)

# 加载模型
model = NetworkDemo()
state_dict = torch.load("../train/model/model_8.pth")
model.load_state_dict(state_dict)
model.eval()


# 构建 batch size
img = torch.reshape(img, (1, 3, 32, 32))
with torch.no_grad():
    output = model(img)
print(output)

# 可以打印数据集中的那个类别
from torchvision.datasets import CIFAR10
dataset = CIFAR10(root="../dataset", train=False, download=False)
print(dataset.classes[output.argmax(1)])