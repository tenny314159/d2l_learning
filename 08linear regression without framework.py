import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    '''
    torch.normal(means, std, out=None)返回一个张量，包含从给定参数means,std的离散正态分布中抽取随机数。 
    均值means是一个张量，包含每个输出元素相关的正态分布的均值。 
    std是一个张量，包含每个输出元素相关的正态分布的标准差。 
    均值和标准差的形状不须匹配，但每个张量的元素个数须相同。 
    参数: means (Tensor) – 均值 std (Tensor) – 标准差 out (Tensor) – 可选的输出张量
    '''
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))  # reshape中的-1的作用就在此: 自动计算


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])
# features: tensor([-0.3146,  1.2558])
# label: tensor([-0.7006])

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)

    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # shuffle(x) 方法可以将序列 x 随机打乱位置
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        '''
        yield语句提取函数，并将值返回给函数调用方，
        然后从停止的地方重新启动。yield语句可以多次调用。
        yield 输出的是一个对象，相当于是一个容器，
        想取什么数据就取出什么，而return 只会返回一个值，
        且return后面的代码不会执行。
        '''


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
# tensor([[ 1.7864, -1.1456],
#         [ 0.0944,  0.6660],
#         [ 1.2795, -1.1822],
#         [ 1.2692,  0.3109],
#         [ 0.7440, -1.0682],
#         [-0.2291, -0.4523],
#         [-0.0399,  0.5042],
#         [ 1.4626,  0.0429],
#         [ 1.5917, -0.0891],
#         [ 0.2985,  0.8326]])
#  tensor([[11.6908],
#         [ 2.1237],
#         [10.7781],
#         [ 5.6756],
#         [ 9.3133],
#         [ 5.2706],
#         [ 2.4010],
#         [ 6.9851],
#         [ 7.6800],
#         [ 1.9478]])


w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):  # sgd小批量随机梯度下降
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            '''
            使用with管理上下文不仅可以在执行with语句体后自动执行退出操作（即__exit__方法定义语句），
            更重要的是能够在发生异常时，确保始终能执行退出操作、释放相应的资源。
            '''


# lr = 0.00001
lr = 0.001
num_epochs = 50
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


# epoch 1, loss 12.605930
# epoch 2, loss 10.501246
# epoch 3, loss 8.748064
# epoch 4, loss 7.287671
# epoch 5, loss 6.071124
# epoch 6, loss 5.057709
# epoch 7, loss 4.213507
# epoch 8, loss 3.510236
# epoch 9, loss 2.924393
# epoch 10, loss 2.436336
# epoch 11, loss 2.029762
# epoch 12, loss 1.691054
# epoch 13, loss 1.408876
# epoch 14, loss 1.173799
# epoch 15, loss 0.977956
# epoch 16, loss 0.814798
# epoch 17, loss 0.678867
# epoch 18, loss 0.565621
# epoch 19, loss 0.471270
# epoch 20, loss 0.392663
# epoch 21, loss 0.327172
# epoch 22, loss 0.272608
# epoch 23, loss 0.227147
# epoch 24, loss 0.189271
# epoch 25, loss 0.157713
# epoch 26, loss 0.131419
# epoch 27, loss 0.109511
# epoch 28, loss 0.091257
# epoch 29, loss 0.076047
# epoch 30, loss 0.063375
# epoch 31, loss 0.052816
# epoch 32, loss 0.044018
# epoch 33, loss 0.036688
# epoch 34, loss 0.030580
# epoch 35, loss 0.025490
# epoch 36, loss 0.021250
# epoch 37, loss 0.017715
# epoch 38, loss 0.014771
# epoch 39, loss 0.012317
# epoch 40, loss 0.010273
# epoch 41, loss 0.008569
# epoch 42, loss 0.007149
# epoch 43, loss 0.005966
# epoch 44, loss 0.004980
# epoch 45, loss 0.004159
# epoch 46, loss 0.003474
# epoch 47, loss 0.002904
# epoch 48, loss 0.002428
# epoch 49, loss 0.002032
# epoch 50, loss 0.001702


print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
# w的估计误差: tensor([ 0.0138, -0.0217], grad_fn=<SubBackward0>)
# b的估计误差: tensor([0.0243], grad_fn=<RsubBackward1>)