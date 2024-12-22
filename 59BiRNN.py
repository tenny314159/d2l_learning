
import torch
from torch import nn
from d2l import torch as d2l
from rnn_utils import load_data_time_machine, RNNModel, train_ch8


# 定义批量大小、时间步数和设备（如果可用的话，使用GPU）
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
# 加载时间机器数据集并生成训练迭代器和词汇表
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
# 获取词汇表大小、隐藏单元数量和层数
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
# 输入大小等于词汇表大小
num_inputs = vocab_size
# 创建一个双向LSTM层
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)  # bidirectional是双向参数
# 使用双向LSTM层构建RNN模型
model = RNNModel(lstm_layer, len(vocab))
# 将模型移动到指定设备（GPU）
model = model.to(device)
# 定义训练的轮数和学习率
num_epochs, lr = 500, 1
# 使用d2l库中的训练函数进行模型训练（双向循环神经网络无法预测未来信息）
train_ch8(model, train_iter, vocab, lr, num_epochs, device) # 双向循环神经网络无法预测未来信息
d2l.plt.show()
# 困惑度 1.1, 92218.2 标记/秒 cuda:0
# time travellerererererererererererererererererererererererererer
# travellerererererererererererererererererererererererererer

