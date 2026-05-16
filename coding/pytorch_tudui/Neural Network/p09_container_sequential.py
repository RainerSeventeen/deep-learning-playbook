# container下的 sequential 函数 https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential
# 可以组合网络中的多个层次，简化代码逻辑

# 例如
# model = nn.Sequential(
#           nn.Conv2d(1,20,5),
#           nn.ReLU(),
#           nn.Conv2d(20,64,5),
#           nn.ReLU()
#         )