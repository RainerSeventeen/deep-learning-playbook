# TensorBoard 一个常用于训练结果可视化的工具
import time
from torch.utils.tensorboard import SummaryWriter
from dataset import Hymenoptera
import cv2 as cv

if __name__ == '__main__':

    writer = SummaryWriter(log_dir='logs') # 将事件存储到目标logs文件夹
    img_path = 'dataset/hymenoptera_data/train/ants/0013035.jpg'
    img_cv = cv.imread(img_path)
    print(img_cv.shape) # HWC

    # 1. 添加单个图片演示， 数据类型，通道数等参数
    writer.add_image('test', img_cv, 0, dataformats='HWC')

    # 2. 添加某个标量测试
    writer.add_scalar(tag='loss', scalar_value=0.5)

    # 3. 添加一连串的数据，类似于绘制函数图像
    for i in range(100):
        writer.add_scalar(tag='loss2', scalar_value=2 * i, global_step=i)

    writer.close()


# 启动命令 --logdir xxx --port 1234 