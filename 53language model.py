import random
import torch
from d2l import torch as d2l
import re
import collections

# 下载并存储 'time_machine' 数据集的 URL 和哈希值
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


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


# 使用d2l库中的函数读取时间机器文本，并进行分词处理
tokens = d2l.tokenize(read_time_machine())
# 将tokens中的所有单词连接成一个列表，生成语料库corpus
corpus = [token for line in tokens for token in line]
# 使用语料库corpus构建词汇表vocab
vocab = d2l.Vocab(corpus)
# 输出词汇表中出现频率最高的前10个单词和对应的频率
print(vocab.token_freqs[:10])
# [('the', 2261), ('i', 1267), ('and', 1245), ('of', 1155), ('a', 816), ('to', 695), ('was', 552), ('in', 541), ('that', 443), ('my', 440)]

# 最流行的词被称为停用词画出的词频图
# 从词汇表的token_freqs中提取频率信息，存储在列表freqs中
freqs = [freq for token, freq in vocab.token_freqs]
# 使用d2l库中的plot函数绘制词频图
# 设置横轴为token，纵轴为对应的频率，横轴使用对数刻度，纵轴也使用对数刻度
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
d2l.plt.show()

# 其他的词元组合，比如二元语法、三元语法等等，又会如何呢？
# 使用列表推导式将corpus中的相邻两个词组成二元语法的词元组合，存储在bigram_tokens列表中
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]  # 二元语法
# 使用bigram_tokens构建二元语法的词汇表bigram_vocab
# bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab = Vocab(bigram_tokens)
# 输出二元语法词汇表中出现频率最高的前10个词元组合和对应的频率
print(bigram_vocab.token_freqs[:10])
# [(('of', 'the'), 309), (('in', 'the'), 169), (('i', 'had'), 130), (('i', 'was'), 112), (('and', 'the'), 109), (('the', 'time'), 102), (('it', 'was'), 99), (('to', 'the'), 85), (('as', 'i'), 78), (('of', 'a'), 73)]

# 三元语法
# 使用列表推导式将corpus中的相邻三个词组成三元语法的词元组合，存储在trigram_tokens列表中
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
# 使用trigram_tokens构建三元语法的词汇表trigram_vocab
trigram_vocab = Vocab(trigram_tokens)
# 输出三元语法词汇表中出现频率最高的前10个词元组合和对应的频率
print(trigram_vocab.token_freqs[:10])
# [(('the', 'time', 'traveller'), 59), (('the', 'time', 'machine'), 30), (('the', 'medical', 'man'), 24), (('it', 'seemed', 'to'), 16), (('it', 'was', 'a'), 15), (('here', 'and', 'there'), 15), (('seemed', 'to', 'me'), 14), (('i', 'did', 'not'), 14), (('i', 'saw', 'the'), 13), (('i', 'began', 'to'), 13)]

# 直观地对比三种模型中的标记频率
# 从bigram词汇表的token_freqs中提取频率信息，存储在列表bigram_freqs中
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
# 从trigram词汇表的token_freqs中提取频率信息，存储在列表trigram_freqs中
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
# 使用d2l库中的plot函数绘制标记频率对比图
# 将bigram和trigram的频率信息传入plot函数，横轴为token，纵轴为对应的频率
# 横轴和纵轴都使用对数刻度，同时在图例中标明每个模型的名称
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
d2l.plt.show()


# 随即生成一个小批量数据的特征和标签以供读取
# 在随即采样中，每个样本都是在原始的长序列上任意捕获的子序列

# 给一段很长的序列，连续切成很多段长为T的子序列
# 一开始加了一点随机，使得每次切的都不一样
# 取随即批量的时候，再随即把它们取出来
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


# 生成一个从0到34的序列
my_seq = list(range(35))
# 使用seq_data_iter_random函数生成随机抽样的小批量特征和标签数据
# batch_size=2表示每个批次的样本数量为2
# num_steps=5表示每个样本的长度为5
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY: ', Y)  # Y是X长度的后一个，X里面的两个不一定是连续的


# X:  tensor([[25, 26, 27, 28, 29],
#         [15, 16, 17, 18, 19]])
# Y:  tensor([[26, 27, 28, 29, 30],
#         [16, 17, 18, 19, 20]])
# X:  tensor([[20, 21, 22, 23, 24],
#         [ 5,  6,  7,  8,  9]])
# Y:  tensor([[21, 22, 23, 24, 25],
#         [ 6,  7,  8,  9, 10]])
# X:  tensor([[ 0,  1,  2,  3,  4],
#         [10, 11, 12, 13, 14]])
# Y:  tensor([[ 1,  2,  3,  4,  5],
#         [11, 12, 13, 14, 15]])

# 保证两个相邻的小批量中的子序列在原始序列上也是相邻的
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


# 读取每个小批量的子序列的特征X和标签Y
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    # 第二个小批量的起始位置是接着第一个小批量的结束位置后面的
    print('X: ', X, '\nY: ', Y)  # 第二个mini-batch[9-13]是接着第一个mini-batch[3-7]后面


# X:  tensor([[ 4,  5,  6,  7,  8],
#         [19, 20, 21, 22, 23]])
# Y:  tensor([[ 5,  6,  7,  8,  9],
#         [20, 21, 22, 23, 24]])
# X:  tensor([[ 9, 10, 11, 12, 13],
#         [24, 25, 26, 27, 28]])
# Y:  tensor([[10, 11, 12, 13, 14],
#         [25, 26, 27, 28, 29]])
# X:  tensor([[14, 15, 16, 17, 18],
#         [29, 30, 31, 32, 33]])
# Y:  tensor([[15, 16, 17, 18, 19],
#         [30, 31, 32, 33, 34]])

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


# 最后，定义一个函数 load_data_time_machine，它同时返回数据迭代器和词汇表
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词汇表"""
    # 这个对象将作为数据的迭代器，用于产生小批量的样本和标签。
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    # 返回数据迭代器和对应的词汇表
    return data_iter, data_iter.vocab
