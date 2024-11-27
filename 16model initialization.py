
import torch
from torch import nn
from torch.nn import functional as F


# 1.实例化nn.Sequential来构建我们的模型
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(),
                    nn.Linear(256, 10))
X = torch.rand(2, 20)  # 输出是[2,10] #均匀分布（Uniform Distribution）采样的随机数，其范围在0到1之间（不包括1）。
#这意味着每个生成的随机数都是独立且均匀地从0到1的区间中选择的
print('1.实例化nn.Sequential来构建我们的模型')
print(net(X))
# 1.实例化nn.Sequential来构建我们的模型
# tensor([[ 0.1696, -0.3822,  0.0887,  0.0499, -0.1803, -0.1282,  0.0064, -0.0611,
#           0.0761,  0.1372],
#         [ 0.1040, -0.1660,  0.0215,  0.1977, -0.1217, -0.2048, -0.0985, -0.3243,
#           0.1085,  0.1488]], grad_fn=<AddmmBackward0>)


# 2.自定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print('2.自定义模型')
print(net(X))
# 2.自定义模型
# tensor([[-0.2959,  0.1920,  0.0789,  0.0355, -0.2098, -0.0056, -0.2797, -0.1670,
#          -0.0367,  0.1369],
#         [-0.1255,  0.2599, -0.0171,  0.1679, -0.1155, -0.0154, -0.3407, -0.1537,
#          -0.1061,  0.2045]], grad_fn=<AddmmBackward0>)


# 3.自定义顺序模型
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        print(args)  # 一个tuple, tuple和list非常类似，但是，tuple一旦创建完毕，就不能修改了。
        for block in args:
            # 这里，`block`是`Module`子类的一个实例。我们把它保存在'Module'类的成员变量
            # `_children` 中。`block`的类型是OrderedDict。
            self._modules[block] = block  # 每个Module都有一个_modules属性

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        print(self._modules.values())
        for block in self._modules.values():
            X = block(X)
        return X


print('3.自定义顺序模型')
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))
# 3.自定义顺序模型
# (Linear(in_features=20, out_features=256, bias=True), ReLU(), Linear(in_features=256, out_features=10, bias=True))
# odict_values([Linear(in_features=20, out_features=256, bias=True), ReLU(), Linear(in_features=256, out_features=10, bias=True)])
# tensor([[-0.0152, -0.1536, -0.1348, -0.1252,  0.1296, -0.2319, -0.2193, -0.1306,
#           0.1117,  0.0637],
#         [-0.1633, -0.1088, -0.3489, -0.1735,  0.0126, -0.2765, -0.1912, -0.1871,
#           0.1286,  0.1573]], grad_fn=<AddmmBackward0>)


# 4.如何将任意代码集成到神经网络计算的流程中
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=True)
        self.liner = nn.Linear(20, 20)

    def forward(self, X):
        X = self.liner(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.liner(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


print('4.如何将任意代码集成到神经网络计算的流程中')
net = FixedHiddenMLP()
print(net(X))
# 4.如何将任意代码集成到神经网络计算的流程中
# tensor(-0.0018, grad_fn=<SumBackward0>)


# 5.组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


print('5.组合块')
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
# 5.组合块
# tensor(0.2274, grad_fn=<SumBackward0>)
