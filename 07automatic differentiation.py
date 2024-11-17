import torch

x = torch.arange(4.0)
print(x)
# tensor([0., 1., 2., 3.])

x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)  # 默认值是None
# None

y = 2 * torch.dot(x, x)
print(y)
# tensor(28., grad_fn=<MulBackward0>)

y.backward()
print(x.grad)
# tensor([ 0.,  4.,  8., 12.])

print(x.grad == 4 * x)
# tensor([True, True, True, True])

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()  # 梯度清0
y = x.sum()
y.backward()  # 求sum梯度是1
print(x.grad)
# tensor([1., 1., 1., 1.])

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)
# tensor([0., 2., 4., 6.])

x.grad.zero_()
y = x * x
u = y.detach()  # 返回一个与当前 graph 分离的、不再需要梯度的新张量
z = u * x
z.sum().backward()
print(x.grad == u)
# tensor([True, True, True, True])

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)
# tensor([True, True, True, True])


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()  # 反向传播计算
print(a.grad == d / a)
# tensor(True)
