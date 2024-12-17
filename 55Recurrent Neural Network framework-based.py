import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import collections
import random
import re
import math


# 下载并存储 'time_machine' 数据集的 URL 和哈希值
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


def count_corpus(tokens):
    """
    统计标记的频率。

    Parameters:
        tokens (list): 标记列表。

    Returns:
        collections.Counter: 包含标记频率的 Counter 对象。

    Raises:
        None
    """
    # 检查 tokens 是否是一个列表的列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 如果 tokens 是一个列表的列表，则将其展平为一维列表
        tokens = [token for line in tokens for token in line]
    # 使用 collections.Counter 统计标记的频率
    return collections.Counter(tokens)


def tokenize(lines, token='word'):
    """
    将文本行列表进行分词处理。

    Parameters:
        lines (list): 文本行列表。
        token (str): 令牌类型，可选值为 'word'（默认）或 'char'。

    Returns:
        list: 分词后的结果列表。

    Raises:
        None
    """
    # 如果令牌类型为 'word'
    if token == 'word':
        # 以空格为分隔符将每行字符串拆分为单词列表
        return [line.split() for line in lines]
    # 如果令牌类型为 'char'
    elif token == 'char':
        # 将每行字符串拆分为字符列表
        return [list(line) for line in lines]
    else:
        # 若指定的令牌类型无效，则打印错误信息
        print('错位：未知令牌类型：' + token)


class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        初始化词表对象。

        Parameters:
            tokens (list): 标记列表（默认为 None）。
            min_freq (int): 最小频率阈值，低于该频率的标记将被过滤掉（默认为 0）。
            reserved_tokens (list): 保留的特殊标记列表（默认为 None）。

        Returns:
            None

        Raises:
            None
        """
        # 如果输入的 tokens 为 None，则将其设置为空列表
        if tokens is None:
            tokens = []
        # 如果保留的特殊标记列表 reserved_tokens 为 None，则将其设置为空列表
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计 tokens 中标记的频率，并返回一个包含标记频率的 Counter 对象
        counter = count_corpus(tokens)  # 遍历得到每一个独一无二token出现的次数
        # 根据标记的频率进行排序，并将结果存储在 self.token_freqs 中
        # sorted() 函数使用 counter.items() 作为排序对象，使用标记频率 x[1] 作为排序依据，降序排序
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 设置未知标记索引为 0，构建包含未知标记和保留特殊标记的列表 uniq_tokens
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        # 将频率大于等于 min_freq 且不在 uniq_tokens 中的标记添加到 uniq_tokens 列表中
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        # 初始化索引到标记和标记到索引的空列表和字典
        self.idx_to_token, self.token_to_idx = [], dict()
        # 遍历 uniq_tokens 中的每个标记，将其添加到索引到标记的列表中，并将标记和对应索引存储到标记到索引的字典中
        # 索引值从 0 开始递增，对应于标记在列表中的位置
        for token in uniq_tokens:
            # 将当前标记 `token` 添加到索引到标记的列表 `self.idx_to_token` 的末尾
            self.idx_to_token.append(token)
            # 将当前标记 `token` 和其对应的索引值存储到标记到索引的字典 `self.token_to_idx` 中
            # 索引值是 `self.idx_to_token` 列表的长度减去 1，即标记在列表中的位置索引
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """
        获取词表的长度。

        Parameters:
            None

        Returns:
            int: 词表的长度。

        Raises:
            None
        """
        # 获取词表的长度
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        根据标记获取其对应的索引或索引列表。

        Parameters:
            tokens (str or list): 标记字符串或标记列表。

        Returns:
            int or list: 标记的索引或索引列表。

        Raises:
            None
        """
        # 如果 tokens 不是列表或元组，则返回对应的索引或默认的未知标记索引
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        # 对于输入的标记列表 tokens，逐个调用 self.__getitem__() 方法获取每个标记对应的索引值，并返回索引值的列表
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        根据索引获取对应的标记或标记列表。

        Parameters:
            indices (int or list): 索引或索引列表。

        Returns:
            str or list: 索引对应的标记或标记列表。

        Raises:
            None
        """
        # 如果输入的 indices 不是列表或元组类型，则返回对应索引值处的标记
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # 对于输入的索引列表 indices，逐个取出每个索引值 index，并通过 self.idx_to_token[index] 获取对应的标记值，最后返回标记值组成的列表
        return [self.idx_to_token[index] for index in indices]


def read_time_machine():
    """Load the time machine dataset into a list of text lines. """
    """将时间机器数据集加载为文本行的列表。"""
    # 打开 'time_machine' 数据集文件，并使用文件对象 f 进行操作
    with open(d2l.download('time_machine'), 'r') as f:
        # 读取文件的所有行，并将每行存储在列表 lines 中
        lines = f.readlines()
        # 把不是大写字母、小写字母的东西，全部变成空格
        # 去除非字母字符，并转换为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的标记索引列表和词汇表"""
    # 加载时光机器数据集的文本行
    lines = read_time_machine()
    # 将文本行转换为字符标记列表
    tokens = tokenize(lines, 'char')
    # 构建词汇表
    vocab = Vocab(tokens)
    # 将文本转换为标记索引列表
    corpus = [vocab[token] for line in tokens for token in line]
    # 截断文本长度（若有限制）
    if max_tokens > 0:
        # 如果设置了最大标记数 max_tokens，对标记索引列表 corpus 进行截断，只保留前 max_tokens 个标记
        corpus = corpus[:max_tokens]
    # 返回截断后的标记索引列表 corpus 和词汇表 vocab
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随即抽样生成一个小批量子序列"""
    # 从原始序列中随机选择一个起始位置
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 计算能够生成的子序列数量
    num_subseqs = (len(corpus) - 1) // num_steps
    # 创建初始索引列表
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 进行随机打乱
    random.shuffle(initial_indices)

    # 返回从指定位置开始的长度为num_steps的子序列
    def data(pos):
        return corpus[pos:pos + num_steps]

    # 计算批次的数量
    num_batches = num_subseqs // batch_size
    # 对每个批次进行迭代
    for i in range(0, batch_size * num_batches, batch_size):
        # 获取当前批次的初始索引列表
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        # 根据初始索引列表获取对应的特征序列X
        X = [data(j) for j in initial_indices_per_batch]
        # 根据初始索引列表获取对应的标签序列Y
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # 使用torch.tensor将X和Y转换为张量，并通过yield语句返回
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 随机选择一个偏移量作为起始位置
    offset = random.randint(0, num_steps)
    # 计算可以生成的子序列的总长度
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    # 创建特征序列X的张量
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    # 创建标签序列Y的张量
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    # 重新调整Xs和Ys的形状，使其成为(batch_size, -1)的二维张量
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    # 计算可以生成的批次数量
    num_batches = Xs.shape[1] // num_steps
    # 对每个批次进行迭代
    for i in range(0, num_steps * num_batches, num_steps):
        # 获取当前批次的特征序列X
        X = Xs[:, i:i + num_steps]
        # 获取当前批次的标签序列Y
        Y = Ys[:, i:i + num_steps]
        # 使用yield语句返回X和Y作为生成器的输出
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # 根据use_random_iter选择数据迭代函数
        if use_random_iter:
            # 使用随机分区迭代器
            self.data_iter_fn = seq_data_iter_random
        else:
            # 使用顺序分区迭代器
            self.data_iter_fn = seq_data_iter_sequential
        # 加载数据集和词汇表
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        # 设置批量大小和步长
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        # 返回数据迭代器
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词汇表"""
    # 这个对象将作为数据的迭代器，用于产生小批量的样本和标签。
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    # 返回数据迭代器和对应的词汇表
    return data_iter, data_iter.vocab


# 设置批量大小和时间步数
batch_size, num_steps = 32, 35
# 调用 load_data_time_machine 函数加载时间机器数据集，返回训练数据迭代器和词汇表
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 定义模型
# 设置隐藏单元的数量为 256
num_hiddens = 256
# 使用 nn.RNN 类定义一个循环神经网络层
# 输入大小为词汇表的大小，隐藏单元数量为 num_hiddens
# 将该循环神经网络层赋值给 rnn_layer
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 使用张量来初始化隐藏状态
# 创建一个形状为 (1, batch_size, num_hiddens) 的张量，用于初始化隐藏状态
# 全部元素初始化为 0
# 将该张量赋值给变量 state
state = torch.zeros((1, batch_size, num_hiddens))
# 打印隐藏状态张量的形状
print(state.shape)
# torch.Size([1, 32, 256])

# 通过一个隐藏状态和一个输入，我们可以用更新后的隐藏状态计算输出
# 创建一个形状为 (num_steps, batch_size, len(vocab)) 的随机张量 X
# 用于表示输入的序列，每个时间步的输入为一个词汇表大小的独热编码向量
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
# 将输入 X 和初始隐藏状态 state 作为输入传递给循环神经网络层 rnn_layer 进行前向计算
# 返回输出张量 Y 和更新后的隐藏状态 state_new
Y, state_new = rnn_layer(X, state)
# 打印输出张量 Y 和更新后的隐藏状态 state_new 的形状
print(Y.shape, state_new.shape)
# torch.Size([35, 32, 256]) torch.Size([1, 32, 256])


# 我们为一个完整的循环神经网络模型定义一个RNNModel类
class RNNModel(nn.Module):
    """循环神经网络模型"""

    # 初始化函数
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        # 调用父类的构造函数，初始化继承的属性
        super(RNNModel, self).__init__(**kwargs)
        # 循环神经网络层
        self.rnn = rnn_layer
        # 词汇表大小
        self.vocab_size = vocab_size
        # 隐藏状态的大小
        self.num_hiddens = self.rnn.hidden_size
        # 如果循环神经网络不是双向的
        if not self.rnn.bidirectional:
            # 方向数量为1
            self.num_directions = 1
            # 线性层的输入大小为隐藏状态大小，输出大小为词汇表大小
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        # 如果循环神经网络是双向的
        else:
            # 方向数量为2
            self.num_directions = 2
            # 线性层的输入大小为隐藏状态大小的两倍，输出大小为词汇表大小
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

            # 前项传播函数

    def forward(self, inputs, state):
        # 将输入的索引序列转换为独热编码张量 X
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        # 将 X 转换为 float32 类型
        X = X.to(torch.float32)
        # 使用循环神经网络层 rnn 进行前向计算，返回输出张量 Y 和更新后的隐藏状态 state
        Y, state = self.rnn(X, state)
        # 将输出张量 Y 展平并通过线性层 linear 进行变换得到最终的输出
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        # 返回最终输出和更新后的隐藏状态
        return output, state

    # 创建循环神经网络的初始隐藏状态
    def begin_state(self, device, batch_size=1):
        # 如果循环神经网络不是LSTM类型
        if not isinstance(self.rnn, nn.LSTM):
            # 创建全零的隐藏状态张量
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        # 如果循环神经网络是LSTM类型
        else:
            # 创建全零的隐藏状态张量和记忆单元张量
            # 第一个张量是全零的隐藏状态张量，第二个张量是全零的记忆单元张量
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))

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
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    # 如果梯度范数超过指定阈值 theta
    if norm > theta:
        # 对于每个参数
        for param in params:
            # 将参数的梯度值裁剪至指定范围内，保持梯度范数不超过 theta
            param.grad[:] *= theta / norm


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
            state = net.begin_state(batch_size=X.shape[0],device=device)
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
            grad_clipping(net,1)
            # 执行一步参数更新
            updater.step()
        else:
            # 反向传播计算梯度
            l.backward()
            # 裁剪梯度
            grad_clipping(net,1)
            # 执行自定义的参数更新函数
            updater(batch_size=1)
        # 累加损失和样本数量
        metric.add(l * y.numel(), y.numel())
    # 计算平均损失和每秒处理的样本数，返回平均损失的指数形式（以 e 为底）和每秒样本处理速度
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型"""
    # 定义损失函数为交叉熵损失
    loss = nn.CrossEntropyLoss()
    # 创建动画对象，用于可视化训练过程的损失变化
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'],xlim=[10,num_epochs])
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


# 用一个具有随即权重的模型进行预测
# 尝试使用GPU设备，如果不可用则使用CPU
device = d2l.try_gpu()
# 创建RNN模型实例
net = RNNModel(rnn_layer, vocab_size=len(vocab))
# 将模型移动到指定设备上
net = net.to(device)
# 对模型进行预测，生成文本
predict_ch8('time traveller', 10, net, vocab, device)


# 使用高级API训练模型
# 设置训练的迭代周期数和学习率
num_epochs, lr = 500, 1
# 使用高级API训练模型，传入模型、训练数据迭代器、词汇表、学习率和迭代周期数进行训练
train_ch8(net, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()
# 困惑度 1.3, 382281.7 标记/秒 cuda:0
# time traveller brt sofor an un ind anoc ahe thiek ous cheve in h
# traveller crover fo so merg yenccl astinge to te yor ce ss
