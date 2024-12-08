import torch
from torch import nn
from d2l import torch as d2l
from torchvision import transforms
import torchvision
from torch.utils import data

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
# Conv2d output shape:	 torch.Size([1, 96, 54, 54])
# ReLU output shape:	 torch.Size([1, 96, 54, 54])
# MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
# Conv2d output shape:	 torch.Size([1, 256, 26, 26])
# ReLU output shape:	 torch.Size([1, 256, 26, 26])
# MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
# Conv2d output shape:	 torch.Size([1, 384, 12, 12])
# ReLU output shape:	 torch.Size([1, 384, 12, 12])
# Conv2d output shape:	 torch.Size([1, 384, 12, 12])
# ReLU output shape:	 torch.Size([1, 384, 12, 12])
# Conv2d output shape:	 torch.Size([1, 256, 12, 12])
# ReLU output shape:	 torch.Size([1, 256, 12, 12])
# MaxPool2d output shape:	 torch.Size([1, 256, 5, 5])
# Flatten output shape:	 torch.Size([1, 6400])
# Linear output shape:	 torch.Size([1, 4096])
# ReLU output shape:	 torch.Size([1, 4096])
# Dropout output shape:	 torch.Size([1, 4096])
# Linear output shape:	 torch.Size([1, 4096])
# ReLU output shape:	 torch.Size([1, 4096])
# Dropout output shape:	 torch.Size([1, 4096])
# Linear output shape:	 torch.Size([1, 10])


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=trans, download=True)
    # return (data.DataLoader(mnist_train, batch_size, shuffle=True,
    #                         num_workers=get_dataloader_workers()),
    #         data.DataLoader(mnist_test, batch_size, shuffle=False,
    #                         num_workers=get_dataloader_workers()))
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))


batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# training on cuda:0

# loss 0.331, train acc 0.879, test acc 0.880
# 1131.7 examples/sec on cuda:0
