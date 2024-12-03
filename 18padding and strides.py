import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # 4 * 4
X = torch.rand(size=(8, 8))  # 算上padding之后是 10 * 10 ，然后运行10-3+1 = 8
print(comp_conv2d(conv2d, X).shape)  # output shape ：8 * 8
# torch.Size([8, 8])

# 如下示例中，我们使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))  # 理解为算上padding之后是12* 10 ，所以输出size是12-5+1 = 8，10-3+1 = 8
print(comp_conv2d(conv2d, X).shape)
# torch.Size([8, 8])

# 我们将高度和宽度的步幅设置为2，从而将输入的高度和宽度减半
# 在输入图像的边界填充元素称为填充（padding） 每次滑动元素的数量称为步幅（stride）
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)  # 理解为算上padding之后是10 * 10 ，所以输出size是（10-3+1）/2=4
print(comp_conv2d(conv2d, X).shape)
# torch.Size([4, 4])

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4)) # 理解为算上padding之后是8 * 10 ，所以输出size是（8-3+1）/3=2，floor(（10-5+1）/4)+1=2
print(comp_conv2d(conv2d, X).shape)
# torch.Size([2, 2])

# 在实践中，我们很少使用不一致的步幅或填充
