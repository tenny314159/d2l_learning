

import torch
from torch import nn
from torch.nn import functional as F


# 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


print('1.不带参数的层')
layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
# 1.不带参数的层
# tensor([-2., -1.,  0.,  1.,  2.])
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
print(net)
# Sequential(
#   (0): Linear(in_features=8, out_features=128, bias=True)
#   (1): CenteredLayer()
# )
Y = net(torch.rand(4, 8))  # Y是4*128维的
print(Y.mean())
# tensor(3.7253e-09, grad_fn=<MeanBackward0>)


# 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


print('2.带参数的层')
dense = MyLinear(5, 3)
print(dense.weight)

# 2.带参数的层
# Parameter containing:
# tensor([[ 0.3980, -1.0751, -0.6859],
#         [ 1.1354,  0.2650, -0.3968],
#         [ 0.4099,  1.2767,  1.3754],
#         [-0.8939, -1.7529, -1.1131],
#         [-1.0037,  0.9632,  0.8082]], requires_grad=True)
Y = dense(torch.rand(2, 5))  # 均匀分布
print(Y)
# tensor([[0.1152, 0.0000, 0.3852],
#         [0.0000, 1.2351, 2.2998]])
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
# tensor([[7.9339],
#         [4.2747]])
