import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
print(net(X))


# tensor([[-0.0298, -0.1373, -0.2311, -0.0839, -0.0065,  0.0261,  0.2116, -0.2484,
#           0.0595, -0.0885],
#         [-0.0720, -0.1561, -0.1912, -0.0773, -0.0314,  0.1616,  0.3479, -0.0609,
#          -0.1726,  0.0106]], grad_fn=<AddmmBackward0>)


class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print(net(X))


# tensor([[ 0.1747,  0.0481, -0.0752,  0.1011, -0.1006,  0.0258,  0.0003,  0.0299,
#          -0.0067, -0.0516],
#         [ 0.0255, -0.1205, -0.0470,  0.1755, -0.0599,  0.0272, -0.2284,  0.2040,
#          -0.0421, -0.0279]], grad_fn=<AddmmBackward0>)


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))


# tensor([[-0.0386, -0.1071,  0.0487, -0.0208,  0.1066, -0.2901, -0.1569, -0.0851,
#           0.1470, -0.1247],
#         [ 0.0102, -0.0442,  0.1535,  0.0505,  0.0823, -0.3253,  0.0058, -0.0710,
#           0.1613, -0.1169]], grad_fn=<AddmmBackward0>)


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
print(net(X))


# tensor(0.0357, grad_fn=<SumBackward0>)


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print('chimera(X)', chimera(X))
# chimera(X) tensor(-0.2073, grad_fn=<SumBackward0>)


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))  # size :[2,1]
# tensor([[-0.3232],
#         [-0.2250]], grad_fn=<AddmmBackward0>)
print(net[2].state_dict())
# OrderedDict([('weight', tensor([[ 0.2194, -0.3201, -0.1590,  0.1407, -0.0254,  0.0687, -0.3110, -0.0936]])), ('bias', tensor([-0.2946]))])
print(type(net[2].bias))
# <class 'torch.nn.parameter.Parameter'>
print(net[2].bias)  # net中的第三个层的偏置参数。bias是层的一个属性，它是一个PyTorch张量，包含了该层所有神经元的偏置值
print(net[
          2].bias.data)  # 这行代码与第一行类似，但是它打印的是bias张量的数据。在PyTorch中，.data属性用于访问张量中存储的原始数据。这通常用于访问张量的值，而不考虑任何与自动梯度计算（autograd）相关的上下文
# Parameter containing:
# tensor([-0.2946], requires_grad=True)
# tensor([-0.2946])
print(net[2].weight.grad == None)
# True
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 只打印模型中第一个层的参数名称和形状
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 打印整个模型的所有参数名称和形状
# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
print(net.state_dict()['2.bias'].data)


# tensor([-0.2946])


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
# tensor([[0.1066],
#         [0.1066]], grad_fn=<AddmmBackward0>)
print(rgnet[0][1][0].bias.data)


# tensor([-0.3819, -0.3415,  0.0064, -0.0621,  0.1932,  0.1594,  0.0555, -0.3223])


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


print(net.apply(init_normal))
print(net[0].weight.data[0], net[0].bias.data[0])


# (tensor([ 0.0230,  0.0008, -0.0044,  0.0025]), tensor(0.))


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


print(net.apply(init_constant))
print(net[0].weight.data[0], net[0].bias.data[0])


# (tensor([1., 1., 1., 1.]), tensor(0.))

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)  # Xavier


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


print(net[0].apply(init_xavier))
print(net[2].apply(init_42))
# Linear(in_features=4, out_features=8, bias=True)
# Linear(in_features=8, out_features=1, bias=True)
print(net[0].weight.data[0])
print(net[2].weight.data)
# tensor([-0.4789, -0.6774,  0.0066, -0.3185])
# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])


