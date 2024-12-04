import torch
from torch import nn
from d2l import torch as d2l


# 1. 多输入通道
def corr2d_multi_in(X, K):
    """
    对多通道输入 X 和多通道卷积核 K 执行二维互相关运算。

    参数:
    X -- 输入张量，形状为 (num_channels, height, width)
    K -- 卷积核张量，形状为 (num_channels, kernel_height, kernel_width)

    返回:
    输出张量，形状为 (output_height, output_width)
    """
    # 遍历“X”和“K”的第0个维度（通道维度），对每个通道应用corr2d函数，
    # 然后将所有通道的结果相加得到最终输出。
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


# 定义一个多通道输入张量 X
# X 包含两个通道，每个通道是一个 3x3 的矩阵
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],  # 第一个通道
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

# 定义一个多通道卷积核 K
# K 也包含两个通道，每个通道是一个 2x2 的矩阵
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))


# 逐通道计算二维互相关运算的结果，然后将结果相加。
# tensor([[ 56.,  72.],
#         [104., 120.]])

# 2. 多输出通道
def corr2d_multi_in_out(X, K):
    """
    对多通道输入 X 和多输出通道卷积核 K 执行二维互相关运算。

    参数:
    X -- 输入张量，形状为 (num_input_channels, height, width)
    K -- 卷积核张量，形状为 (num_output_channels, num_input_channels, kernel_height, kernel_width)

    返回:
    输出张量，形状为 (num_output_channels, output_height, output_width)
    """
    # 迭代“K”的第0个维度（即每个输出通道对应的卷积核），
    # 每次都对输入“X”执行多输入通道的二维互相关运算。
    # 最后将所有结果堆叠在一起，形成一个三维张量。
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
# torch.Size([3, 2, 2, 2])

print(corr2d_multi_in_out(X, K))


# tensor([[[ 56.,  72.],
#          [104., 120.]],
#
#         [[ 76., 100.],
#          [148., 172.]],
#
#         [[ 96., 128.],
#          [192., 224.]]])

# 3. 1X1卷积层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    print('c_i: ', c_i)  # 输入的通道数
    print('h: ', h)  # 输入的高
    print('w: ', w)  # 输入的宽
    # c_i: 3
    # h: 3
    # w: 3
    c_o = K.shape[0]  # 卷积核的通道数
    X = X.view(c_i, h * w)  # 3 * 9
    K = K.view(c_o, c_i)  # 2 * 3
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)


X = torch.normal(0, 1, (3, 3, 3))   # 形状为 (3, 3, 3) 的输入张量
K = torch.normal(0, 1, (2, 3, 1, 1))  # 形状为 (2, 3, 1, 1) 的卷积核张量

Y1 = corr2d_multi_in_out_1x1(X, K)  # 计算1x1卷积层的结果
Y2 = corr2d_multi_in_out(X, K)  # 计算常规多输出通道二维互相关运算的结果

# 验证两个结果是否一致
# 使用绝对差值的总和小于一个小阈值来判断一致性
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6

# (Y1 - Y2).norm().item() < 1e-6
