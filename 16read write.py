

import torch
from torch import nn
from torch.nn import functional as F


# 加载和保存张量
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load("x-file")
print(x2)
# tensor([0, 1, 2, 3])

y = torch.zeros(4)
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
#{'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}


# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
print(Y)
# tensor([[-0.1904,  0.0208,  0.0846, -0.0905, -0.1365,  0.1554, -0.4984, -0.2112,
#           0.3347, -0.0897],
#         [-0.1827,  0.2142,  0.2662, -0.1672, -0.2022,  0.2926, -0.7227, -0.4742,
#           0.7152,  0.2727]], grad_fn=<AddmmBackward0>)

torch.save(net.state_dict(), 'mlp.params')

clone_net = MLP()
clone_net.load_state_dict(torch.load("mlp.params"))

Y_clone = clone_net(X)
print(Y_clone)
# tensor([[-0.1904,  0.0208,  0.0846, -0.0905, -0.1365,  0.1554, -0.4984, -0.2112,
#           0.3347, -0.0897],
#         [-0.1827,  0.2142,  0.2662, -0.1672, -0.2022,  0.2926, -0.7227, -0.4742,
#           0.7152,  0.2727]], grad_fn=<AddmmBackward0>)
