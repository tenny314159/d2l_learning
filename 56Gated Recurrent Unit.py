import torch
from torch import nn
from d2l import torch as d2l

import math
from rnn_utils import load_data_time_machine, RNNModelScratch, RNNModel

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


def grad_clipping(net, theta):
    """裁剪梯度。"""
    # 如果 net 是 nn.Module 的实例（即使用 PyTorch 构建的模型）
    if isinstance(net, nn.Module):
        # 获取所有需要计算梯度的参数列表
        params = [p for p in net.parameters() if p.requires_grad]
    # 如果 net 是自定义的模型（例如上述的 RNNModelScratch）
    else:
        # 获取自定义模型的参数列表
        params = net.params
    # 计算参数梯度的范数，即所有参数梯度平方和的平方根
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # 如果梯度范数超过指定阈值 theta
    if norm > theta:
        # 对于每个参数
        for param in params:
            # 将参数的梯度值裁剪至指定范围内，保持梯度范数不超过 theta
            param.grad[:] *= theta / norm


# 定义一个函数来训练只有一个迭代周期的模型
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期"""
    # 初始化隐藏状态和计时器
    state, timer = None, d2l.Timer()
    # 初始化度量指标的累加器，用于计算损失和样本数量
    metric = d2l.Accumulator(2)
    # 遍历训练迭代器中的每个批次数据
    for X, Y in train_iter:
        # 如果隐藏状态为空或使用随机迭代器
        if state is None or use_random_iter:
            # 初始化隐藏状态，批量大小为 X 的行数，设备为指定的设备
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 如果 net 是 nn.Module 的实例且隐藏状态不是元组类型
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 分离隐藏状态的计算图
                state.detach_()
            else:
                # 对于隐藏状态中的每个元素
                for s in state:
                    # 分离隐藏状态的计算图，用于减少内存占用和加速计算
                    s.detach_()
        # 将目标序列 Y 转置并展平为一维张量
        y = Y.T.reshape(-1)
        # 将输入序列和目标序列移动到指定的设备上
        X, y = X.to(device), y.to(device)
        # 使用输入序列和隐藏状态进行前向传播计算，得到预测值和更新后的隐藏状态
        y_hat, state = net(X, state)
        # 计算预测值与目标值之间的损失
        l = loss(y_hat, y.long()).mean()
        # 如果使用 PyTorch 内置的优化器
        if isinstance(updater, torch.optim.Optimizer):
            # 清空优化器中的梯度
            updater.zero_grad()
            # 反向传播计算梯度
            l.backward()
            # 裁剪梯度
            grad_clipping(net, 1)
            # 执行一步参数更新
            updater.step()
        else:
            # 反向传播计算梯度
            l.backward()
            # 裁剪梯度
            grad_clipping(net, 1)
            # 执行自定义的参数更新函数
            updater(batch_size=1)
        # 累加损失和样本数量
        metric.add(l * y.numel(), y.numel())
    # 计算平均损失和每秒处理的样本数，返回平均损失的指数形式（以 e 为底）和每秒样本处理速度
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# 首先定义预测函数来生成用户提供的prefix之后的新字符
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在 'prefix' 后面生成新字符。"""
    # 获取模型的初始隐藏状态，批量大小为 1，设备为指定的设备
    state = net.begin_state(batch_size=1, device=device)
    # 将 prefix 的第一个字符索引添加到输出列表中
    outputs = [vocab[prefix[0]]]
    # 定义一个函数 get_input，用于获取输入序列的张量表示
    # 输入序列只包含一个字符，将该字符的索引转换为张量，并进行形状调整为 (1, 1)
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    # 对于 prefix 中除第一个字符之外的每个字符 y
    for y in prefix[1:]:
        # 使用当前输入字符和隐藏状态进行前向传播计算，得到输出和更新后的隐藏状态
        _, state = net(get_input(), state)
        # 将字符 y 的索引添加到输出列表中
        outputs.append(vocab[y])
    # 生成指定数量的新字符
    for _ in range(num_preds):
        # 使用当前输入字符和隐藏状态进行前向传播计算，得到输出和更新后的隐藏状态
        y, state = net(get_input(), state)
        # 将输出张量中概率最大的字符索引添加到输出列表中
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # 将输出列表中的字符索引转换为对应的字符，并拼接成一个字符串返回
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 训练函数支持从零开始或使用高级API实现的循环神经网络模型
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型"""
    # 定义损失函数为交叉熵损失
    loss = nn.CrossEntropyLoss()
    # 创建动画对象，用于可视化训练过程的损失变化
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # 如果模型是 nn.Module 的实例
    if isinstance(net, nn.Module):
        # 使用 PyTorch 的优化器 SGD 进行参数更新
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # # 否则，使用自定义的梯度下降函数进行参数更新
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # 定义一个预测函数，用于生成给定前缀之后的新字符序列
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 遍历每个迭代周期
    for epoch in range(num_epochs):
        # 训练一个迭代周期，并返回困惑度和每秒样本处理速度
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        # 每隔 10 个迭代周期生成
        if (epoch + 1) % 10 == 0:
            # 打印以 'time traveller' 为前缀的新字符序列
            print(predict('time traveller'))
            # 将当前迭代周期的困惑度添加到动画中进行可视化
            animator.add(epoch + 1, [ppl])
    # 打印最终的困惑度和每秒样本处理速度
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 标记/秒 {str(device)}')
    # 生成并打印以 'time traveller' 为前缀的新字符序列
    print(predict('time traveller'))
    # 生成并打印以 'traveller' 为前缀的新字符序列
    print(predict('traveller'))


vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1

# model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
#                             init_gru_state, gru)
# train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# d2l.plt.show()
# 困惑度 1.1, 24415.8 标记/秒 cuda:0
# time traveller for so it will be convenient to speak of himwas e
# travelleryou can show black is white by argument said filby


num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = RNNModel(gru_layer, len(vocab))
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()
# 困惑度 1.1, 316481.9 标记/秒 cuda:0
# time traveller with a slight accession ofcheerfulness really thi
# travelleryou can show black is white by argument said filby
