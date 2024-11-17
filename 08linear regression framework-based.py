import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))
# [tensor([[ 0.6522, -1.0055],
#         [ 1.0068,  0.1994],
#         [-0.1423, -1.1701],
#         [-0.1479,  0.0294],
#         [-0.6823,  1.0106],
#         [-0.2858, -1.1352],
#         [ 0.5891, -0.6093],
#         [ 0.6882,  0.2980],
#         [ 0.7594,  1.9317],
#         [ 0.1814, -0.6548]]), tensor([[ 8.9183],
#         [ 5.5330],
#         [ 7.9157],
#         [ 3.8140],
#         [-0.5967],
#         [ 7.4841],
#         [ 7.4543],
#         [ 4.5585],
#         [-0.8613],
#         [ 6.7910]])]
'''
object 可迭代对象或者可调用对象
sentinel 当传入sentinel实参是，
object必须是可调用对象，iter会一直调用object直到返回sentinel
iter返回的是一个迭代器，使用next函数可以从迭代器中取出具体的值。
'''

# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()  # MSE 均方误差
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
# epoch 1, loss 0.000306
# epoch 2, loss 0.000101
# epoch 3, loss 0.000101
# epoch 4, loss 0.000101
# epoch 5, loss 0.000101
# epoch 6, loss 0.000101
# epoch 7, loss 0.000101
# epoch 8, loss 0.000102
# epoch 9, loss 0.000101
# epoch 10, loss 0.000101

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
# w的估计误差： tensor([-5.6028e-05, -6.1750e-04])
# b的估计误差： tensor([-8.7261e-05])
