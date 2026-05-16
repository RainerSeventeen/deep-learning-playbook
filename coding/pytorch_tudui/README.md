# PyTorch 学习笔记

这个仓库整理了个人学习 PyTorch 过程中的代码实验，包括基础算子演示、数据预处理示例、TensorBoard 使用方法，以及一个基于 CIFAR-10 的简单卷积网络训练与推理流程。

## 仓库结构
- `Neural Network/`：围绕 `torch.nn` 的小实验，涵盖容器、卷积/池化层、激活函数、优化器和反向传播等基础概念。
- `dataset.py`：自定义 `Dataset` 示例（以 `hymenoptera_data` 为例），演示如何加载图片并返回 `(image, label)`。
- `transform.py`：常用 `torchvision.transforms` 变换示例（ToTensor、Normalize、Resize、Compose 等）。
- `dataloader.py`：使用 `DataLoader` 读取 CIFAR-10 并把批次写入 TensorBoard 的示例。
- `tensorboard_demo.py`：TensorBoard 基础用法演示。
- `Project/CIFAR10/`：核心示例。包含 `model.py`（简单 CNN）、`train.py`（训练与验证记录）、`test.py`（加载已训好的权重并对单张图片推理）。
- `Project/PretrainedModel/model_save.py`：演示如何保存与加载 `torchvision` 预训练模型（完整模型 & `state_dict`）。
- `Project/dataset/`：训练脚本自动下载的 CIFAR-10 数据与推理用测试图片。

## 环境准备
建议使用 Python 3.9+，并安装以下依赖：
```
pip install torch torchvision torchaudio
pip install opencv-python pillow tensorboard
```
如需 GPU，请安装与本机 CUDA 对应的 PyTorch 版本。

## 运行训练
- 进入仓库根目录运行：
```
python Project/CIFAR10/train.py
```
- 训练脚本会自动下载 CIFAR-10 到 `Project/dataset/`，默认使用可用的 GPU，否则回退到 CPU。
- 日志默认写入 `Project/train/logs`，模型权重按 epoch 保存在 `Project/train/model/model_*.pth`。

### 查看训练日志
在新的终端运行：
```
tensorboard --logdir Project/train/logs --port 6006
```
然后在浏览器访问输出的地址即可查看损失曲线等指标。

## 推理测试
确保 `Project/train/model/` 下已有训练好的权重（示例默认读取 `model_8.pth`），然后运行：
```
python Project/CIFAR10/test.py
```
脚本会对 `Project/dataset/test/plane.jpeg` 进行预处理，加载权重并输出预测的 CIFAR-10 类别。

## 其他示例
- 想了解自定义数据集：阅读 `dataset.py` 并将 `hymenoptera_data` 路径替换为自己的数据目录。
- 想快速记录/可视化：参考 `dataloader.py` 与 `tensorboard_demo.py` 的 `SummaryWriter` 用法。
- 想学习模型保存：运行 `Project/PretrainedModel/model_save.py`，对比保存完整模型与 `state_dict` 的差异。

## 常见问题
- 下载数据较慢：可提前手动将 CIFAR-10 解压到 `Project/dataset/cifar-10-batches-py`。
- 权重文件不存在：先运行训练脚本生成 `model_*.pth`，或在 `test.py` 中将加载路径改为已有的权重。

欢迎在现有脚本基础上继续实验或扩展更复杂的模型。祝学习愉快！
