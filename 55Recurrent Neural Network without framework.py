import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import collections
import re
import random

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


# 定义批量大小和时间步数
batch_size, num_steps = 32, 35
# 加载时间机器数据并创建词汇表
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 独热编码
# 打印词汇表的大小
print(len(vocab))
# 28
# 使用独热编码将 [0, 2] 表示的物体下标转换为独热向量，其中0表示第一个元素，2表示第3个元素
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))  # [0,2] 表示物体下标，0表示第一个元素，2表示第3个元素
# tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0]])


# 小批量形状是(批量大小，时间步数)
# 创建一个张量，形状为(2, 5)，表示批量大小为2，时间步数为5
X = torch.arange(10).reshape((2, 5))
# 对X的转置进行独热编码，其中28表示编码长度，返回独热编码后的形状
print(F.one_hot(X.T, 28).shape)


# torch.Size([5, 2, 28])


# 初始化循环神经网络模型的模型参数
def get_params(vocab_size, num_hiddens, device):
    # 设置输入和输出的维度为词汇表大小
    num_inputs = num_outputs = vocab_size

    # 定义normal函数用于生成服从正态分布的随机张量，并乘以0.01进行缩放
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 初始化模型参数
    # 输入到隐藏层的权重矩阵，形状为(词汇表大小, 隐藏单元个数)
    W_xh = normal((num_inputs, num_hiddens))
    # 隐藏层到隐藏层的权重矩阵，形状为(隐藏单元个数, 隐藏单元个数)
    W_hh = normal((num_hiddens, num_hiddens))
    # 隐藏层的偏置向量，形状为(隐藏单元个数,)
    b_h = torch.zeros(num_hiddens, device=device)
    # 隐藏层到输出层的权重矩阵，形状为(隐藏单元个数, 词汇表大小)
    W_hq = normal((num_hiddens, num_outputs))
    # 输出层的偏置向量，形状为(词汇表大小,)
    b_q = torch.zeros(num_outputs, device=device)
    # 将所有参数放入列表中
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    # 遍历所有参数
    for param in params:
        # 设置参数的requires_grad为True，用于梯度计算
        param.requires_grad_(True)
    # 返回模型的参数
    return params


# 一个init_rnn_state函数在初始化时返回隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    # 返回一个包含隐藏状态的元组，元组中的唯一元素是一个形状为(批量大小, 隐藏单元个数)的全零张量
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# 下面的rnn函数定义了如何在一个时间步计算隐藏状态和输出
def rnn(inputs, state, params):
    # 从参数元组中解包获取输入到隐藏层的权重矩阵 W_xh，
    # 隐藏层到隐藏层的权重矩阵 W_hh，
    # 隐藏层的偏置向量 b_h，
    # 隐藏层到输出层的权重矩阵 W_hq，
    # 输出层的偏置向量 b_q
    W_xh, W_hh, b_h, W_hq, b_q = params
    # 从状态元组中解包获取隐藏状态 H
    # 注意这里使用逗号是为了确保 H 为一个元组
    H, = state
    # 创建一个空列表用于存储输出
    outputs = []
    # 对于输入序列中的每个输入 X
    # 输入序列通常是一个时间步的数据，可以是单个时间步的特征向量或者是嵌入向量
    for X in inputs:
        # 计算新的隐藏状态 H，使用双曲正切函数作为激活函数
        # 根据当前输入 X、上一时间步的隐藏状态 H、以及权重矩阵和偏置向量来计算
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 计算输出 Y，通过隐藏状态 H 与权重矩阵 W_hq 相乘并加上偏置向量 b_q 得到
        Y = torch.mm(H, W_hq) + b_q
        # 将输出 Y 添加到输出列表中
        outputs.append(Y)
    # 将输出列表中的输出张量沿着行维度进行拼接，得到一个形状为 (时间步数 * 批量大小, 输出维度) 的张量
    # 返回拼接后的输出张量和最后一个时间步的隐藏状态 H
    return torch.cat(outputs, dim=0), (H,)


# 创建一个类来包装这些函数
class RNNModelScratch:
    # 初始化模型参数
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        # 保存词汇表大小和隐藏单元个数作为类的属性
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 调用 get_params 函数初始化模型的参数，并保存为类的属性
        # 参数包括输入到隐藏层的权重矩阵、隐藏层到隐藏层的权重矩阵、隐藏层的偏置向量、隐藏层到输出层的权重矩阵、输出层的偏置向量
        self.params = get_params(vocab_size, num_hiddens, device)
        # 初始化隐藏状态的函数和前向传播函数
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # 将输入序列 X 进行独热编码，形状为 (时间步数, 批量大小, 词汇表大小)
        # 并将数据类型转换为浮点型
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # 调用前向传播函数进行模型计算，并返回输出
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        # 返回初始化的隐藏状态，用于模型的初始时间步
        return self.init_state(batch_size, self.num_hiddens, device)


# 检查输出是否具有正确的形状
# 设置隐藏单元个数为 512
num_hiddens = 512
# 创建一个 RNNModelScratch 的实例 net，指定词汇表大小、隐藏单元个数、设备、获取参数函数、初始化隐藏状态函数和前向传播函数
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
# 获取模型的初始隐藏状态，输入的批量大小为 X 的行数，设备使用与 X 相同的设备
state = net.begin_state(X.shape[0], d2l.try_gpu())
# 使用输入 X 和初始隐藏状态进行前向传播计算，得到输出张量 Y 和更新后的隐藏状态 new_state
# 将输入和状态都移动到与 X 相同的设备上进行计算
Y, new_state = net(X.to(d2l.try_gpu()), state)
# 输出 Y 的形状，new_state 的长度（即元素个数）和 new_state 中第一个元素的形状
print(Y.shape, len(new_state), new_state[0].shape)


# torch.Size([10, 28]) 1 torch.Size([2, 512])

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


# 生成以 'time traveller ' 为前缀的 10 个新字符
# 注意：由于模型尚未训练，这里的预测结果是随机初始化后的预测
print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))
# time traveller vu vu vu v


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


# 训练函数支持从零开始或使用高级API实现的循环神经网络模型
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


# # 现在我们可以训练循环神经网络模型
# # 设置迭代周期数和学习率
# num_epochs, lr = 500, 1
# # 调用训练函数进行模型训练，使用训练数据迭代器、词汇表、学习率、迭代周期数和设备信息作为输入
# train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
# d2l.plt.show()
# # 困惑度 1.0, 63966.3 标记/秒 cuda:0
# # time traveller for so it will be convenient to speak of himwas e
# # traveller with a slight accession ofcheerfulness really thi


# 最后，让我们检查一下使用随即抽样方法的结果
# 调用训练函数进行模型训练，使用训练数据迭代器、词汇表、学习率、迭代周期数、设备信息和随机抽样标志位作为输入
# 设置 use_random_iter 参数为 True，表示使用随机抽样方法进行训练
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)
d2l.plt.show()
# 困惑度 1.4, 66251.4 标记/秒 cuda:0
# time travellerit s against reason said filbycin aid io butay on
# travellerit s against reason said filbycin aid io butay on

