# transform 库常用于数据（一般是图像）类型的转化

from torchvision import transforms
from PIL import Image
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter

# 一般而言是用来对图片进行转换格式的

# tensor 数据类型 (Class ToTensor)
# Convert a PIL Image or ndarray to tensor and scale the values accordingly
if __name__ == '__main__':
    img_path = '/root/TestPytorch/dataset/test_data/train/ants/0013035.jpg'
    img = Image.open(img_path)
    print(img)

# 1.将 PIL 文件转为 Tensor格式
    tensor_trans = transforms.ToTensor()
    tensor_img = tensor_trans(img) #转化格式为tensor
    # print(tensor_img)

# 2.使用 opencv
    img_cv = cv.imread(img_path)
    tensor_cv = tensor_trans(img_cv)
    # print(tensor_cv)
    writer = SummaryWriter('/root/TestPytorch/logs')
    writer.add_image('img_cv', tensor_cv, global_step=10)

# 3.使用 Normalize 归一化库
    # Normalize 参数包括 mean 代表均值，RGB通道则为3个值的数组，std代表标准差，公式如下
    # output[channel] = (input[channel] - mean[channel]) / std[channel]
    print(tensor_cv[0][0][0])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std = [0.5, 0.5, 0.5])
    img_normalize = normalize(tensor_cv)
    writer.add_image('img_normalize', img_normalize, global_step=10)
    print(img_normalize[0][0][0])

# 4. 使用 Resize
    # 默认使用 (H, W) 参数进行输入，如果输入单一值，将会按照最小边匹配进行缩放
    resize = transforms.Resize((512, 512))
    print(tensor_cv.size())
    img_resize = resize(tensor_cv)
    print(img_resize.size())
    writer.add_image('img_resize', img_resize, global_step=10)

# 5.使用 Compose 整合操作流程
    trans_compose = transforms.Compose([
    transforms.Resize((512, 512)),transforms.ToTensor(),
    ])
    img_compose = trans_compose(img) # 先进行resize 再进行totensor
    writer.add_image('img_compose', img_compose, global_step=10)





