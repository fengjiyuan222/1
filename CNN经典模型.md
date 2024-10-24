**CNN经典模型**





**LetNet模型**

![img](https://i-blog.csdnimg.cn/blog_migrate/55864490f64bc43a20a9e42451467fbb.png)

```python
import torch
# 导入torch.nn模块
from torch import nn


# 定义LeNet网络模型
# MyLeNet5（子类）继承nn.Module（父类）
class MyLeNet5(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数
    # init()函数：进行初始化，申明模型中各层的定义
    def __init__(self):
        # super：引入父类的初始化方法给子类进行初始化
        super(MyLeNet5, self).__init__()
        
        
        # 卷积层，输入大小为28*28，输出大小为28*28，输入通道为1，输出为6，卷积核为5，扩充边缘为2
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 使用sigmoid作为激活函数
        self.Sigmoid = nn.Sigmoid()
        # AvgPool2d：二维平均池化操作
        # 池化层，输入大小为28*28，输出大小为14*14，输入通道为6，输出为6，卷积核为2，步长为2
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层，输入大小为14*14，输出大小为10*10，输入通道为6，输出为16，卷积核为5
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 池化层，输入大小为10*10，输出大小为5*5，输入通道为16，输出为16，卷积核为2，步长为2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层，输入大小为5*5，输出大小为1*1，输入通道为16，输出为120，卷积核为5
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # Flatten()：将张量（多维数组）平坦化处理，张量的第0维表示的是batch_size（数量），所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # 全连接层
        # Linear（in_features，out_features）
        # in_features指的是[batch_size, size]中的size,即样本的大小
        # out_features指的是[batch_size，output_size]中的output_size，样本输出的维度大小，也代表了该全连接层的神经元个数
        self.f6 = nn.Linear(120, 84)
        # 全连接层&输出层
        self.output = nn.Linear(84, 10)

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # x输入为28*28*1， 输出为28*28*6
        x = self.Sigmoid(self.c1(x))
        # x输入为28*28*6，输出为14*14*6
        x = self.s2(x)
        # x输入为14*14*6，输出为10*10*16
        x = self.Sigmoid(self.c3(x))
        # x输入为10*10*16，输出为5*5*16
        x = self.s4(x)
        # x输入为5*5*16，输出为1*1*120
        x = self.c5(x)
        x = self.flatten(x)
        # x输入为120，输出为84
        x = self.f6(x)
        # x输入为84，输出为10
        x = self.output(x)
        return x


# 测试代码
# 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
# 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
# “__main__” 始终指当前执行模块的名称（包含后缀.py）
# if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行
if __name__ == "__main__":
    # rand:返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，此处为四维张量
    x = torch.rand([1, 1, 28, 28])
    # 模型实例化
    model = MyLeNet5()
    y = model(x)
```

**训练**

```python
import torch
from torch import nn
from model import MyLeNet5


# lr_scheduler：提供一些根据epoch训练次数来调整学习率的方法
from torch.optim import lr_scheduler


# torchvision：PyTorch的一个图形库，服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型
# transforms：主要是用于常见的一些图形变换
# datasets：包含加载数据的函数及常用的数据集接口
from torchvision import datasets, transforms
# os：operating system（操作系统），os模块封装了常见的文件和目录操作
import os

# 数据转化为Tensor格式
# Compose()：将多个transforms的操作整合在一起
# ToTensor(): 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且归一化到[0,1.0]之间
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
# MNIST数据集来自美国国家标准与技术研究所, 训练集 (training set)、测试集(test set)由分别由来自250个不同人手写的数字构成
# MNIST数据集包含：Training set images、Training set images、Test set images、Test set labels
# train = true是训练集，false为测试集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)


# DataLoader：将读取的数据按照batch size大小封装并行训练
# dataset (Dataset)：加载的数据集
# batch_size (int, optional)：每个batch加载多少个样本(默认: 1)
# shuffle (bool, optional)：设置为True时会在每个epoch重新打乱数据(默认: False)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有NVIDA显卡，转到GPU训练，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化，将模型转到device
model = MyLeNet5().to(device)

# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义优化器(随机梯度下降法)
# params(iterable)：要训练的参数，一般传入的是model.parameters()
# lr(float)：learning_rate学习率，也就是步长
# momentum(float, 可选)：动量因子（默认：0），矫正优化率
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率，每隔10轮变为原来的0.1
# StepLR：用于调整学习率，一般情况下会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果
# optimizer （Optimizer）：需要更改学习率的优化器
# step_size（int）：每训练step_size个epoch，更新一次参数
# gamma（float）：更新lr的乘法因子
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    
    # dataloader: 传入数据（数据包括：训练数据和标签）
    # enumerate()：用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，一般用在for循环当中
    # enumerate返回值有两个：一个是序号，一个是数据（包含训练数据和标签）
    # x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型的）
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播
        x, y = x.to(device), y.to(device)
        # 计算训练值
        output = model(x)
        # 计算观测值（label）与训练值的损失函数
        cur_loss = loss_fn(output, y)
        # torch.max(input, dim)函数
        # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        _, pred = torch.max(output, axis=1)
        # 计算每批次的准确率
        # output.shape[0]一维长度为该批次的数量
        # torch.sum()对输入的tensor数据的某一维度求和
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        # 清空过往梯度
        optimizer.zero_grad()
        # 反向传播，计算当前梯度
        cur_loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        # .item()：得到元素张量的元素值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    # 计算训练的错误率
    print('train_loss' + str(train_loss))
    # 计算训练的准确率
    print('train_acc' + str(train_acc))


# 定义验证函数
def val(dataloader, model, loss_fn):
    # model.eval()：设置为验证模式，如果模型中有Batch Normalization或Dropout，则不启用，以防改变权值
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # with torch.no_grad()：将with语句包裹起来的部分停止梯度的更新，从而节省了GPU算力和显存，但是并不会影响dropout和BN层的行为
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        # 计算验证的错误率
        print("val_loss：" + str(loss / n))
        # 计算验证的准确率
        print("val_acc：" + str(current / n))
        # 返回模型准确率
        return current / n


# 开始训练
# 训练次数
epoch = 10
# 用于判断最佳模型
min_acc = 0
for t in range(epoch):
    print(f'epoch {t + 1}\n---------------')
    # 训练模型
    train(train_dataloader, model, loss_fn, optimizer)
    # 验证模型
    a = val(test_dataloader, model, loss_fn)
    # 保存最好的模型权重
    if a > min_acc:
        folder = 'save_model'
        # path.exists：判断括号里的文件是否存在，存在为True，括号内可以是文件路径
        if not os.path.exists(folder):
            # os.mkdir() ：用于以数字权限模式创建目录
            os.mkdir('save_model')
        min_acc = a
        print('save best model')
        # torch.save(state, dir)保存模型等相关参数，dir表示保存文件的路径+保存文件名
        # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
        torch.save(model.state_dict(), 'save_model/best_model.pth')
print('Done!')
```

**预测**

```python
import torch
from model import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# Compose()：将多个transforms的操作整合在一起
data_transform = transforms.Compose([
    # ToTensor()：数据转化为Tensor格式
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有NVIDA显卡，转到GPU训练，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化，将模型转到device
model = MyLeNet5().to(device)

# 加载train.py里训练好的模型
model.load_state_dict(torch.load("D:/pycharm/file/save_model/best_model.pth"))

# 结果类型
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 把Tensor转化为图片，方便可视化
show = ToPILImage()

# 进入验证阶段
for i in range(10):
    x, y = test_dataset[i][0], test_dataset[i][1]
    # show()：显示图片
    show(x).show()
    # unsqueeze(input, dim)，input(Tensor)：输入张量，dim (int)：插入维度的索引，最终将张量维度扩展为4维
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x)
        # argmax(input)：返回指定维度最大值的序号
        # 得到验证类别中数值最高的那一类，再对应classes中的那一类
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        # 输出预测值与真实值
        print(f'predicted: "{predicted}", actual:"{actual}"')
```

最早的使用卷积和池化思想的模型







**AlexNet**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/dbcbcc632ea823bc331e7fae8d2790ff.png#pic_center)

```python
import time
import torch
from torch import nn, optim
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```

**性能提升**：AlexNet 的出现使得在 ImageNet 大规模图像分类任务中的错误率大幅下降，标志着深度学习方法在计算机视觉中的成功。

**深度学习的推广**：AlexNet 的成功展示了深度学习在图像识别中的潜力，推动了后续许多卷积神经网络（如 VGG、GoogLeNet、ResNet）的发展。

**引入新技术**：ReLU、Dropout、数据增强等创新技术在 AlexNet 中首次得到广泛应用，并成为后来神经网络中的标准技术。



**VGG**

![img](https://i-blog.csdnimg.cn/blog_migrate/d19533e472e6972e1b772a7ba79fb13c.png)

```python
import time
import torch
from torch import nn, optim

import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)

conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net

net = vgg(conv_arch, fc_features, fc_hidden_units)

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
```

![{C0782F18-8BAC-44EA-8BF5-ECDB3699215C}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{C0782F18-8BAC-44EA-8BF5-ECDB3699215C}.png)

![{75E8ED84-E241-460E-ADAC-264D55C3E152}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{75E8ED84-E241-460E-ADAC-264D55C3E152}.png)

VGGNet 是第一个**显著加深卷积网络**的模型，成功证明了增加网络深度可以显著提升模型的性能

引入小卷积核的堆叠

建立卷积神经网络设计的模块化原则



**NiN**

![img](https://i-blog.csdnimg.cn/blog_migrate/29506419c0286bef936ccb6004feac7d.png)



**googlenet**



![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7af198c87db70bd143752678c117b267.png)

利用不同大小的卷积核实现不同尺度的感知，最后进行融合，可以得到图像更好的表征。

![img](https://pica.zhimg.com/80/v2-39e361ad7cdbbb521f6be1bdb57452e0_720w.webp)

加了并行线路





![img](https://pica.zhimg.com/80/v2-766c3f59d3791da39ad805606d6445f6_720w.webp)



```python
import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    # 传入的参数中aux_logits=True表示训练过程用到辅助分类器，aux_logits=False表示验证过程不用辅助分类器
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        
#self.aux_logits = aux_logits 这一行的作用是将传入的参数 aux_logits 存储为当前 GoogLeNet 类实例的属性。这有几个重要的目的：
#控制辅助分类器的启用：通过设置 aux_logits 参数，用户可以决定在训练过程中是否使用辅助分类器。这对于在不同的训练和验证模式下灵活使用模型非常重要。
#灵活性：在定义网络时，可能不希望在评估模型时使用辅助分类器（例如，在推理阶段）。通过这个属性，可以在 forward 方法中轻松地检查是否需要返回辅助分类器的输出。
#简化代码逻辑：通过将参数存储为实例属性，后续的方法（如 forward 方法）可以直接使用 self.aux_logits 进行条件判断，而不需要每次都检查原始参数值。这提高了代码的可读性和可维护性
        self.aux_logits = aux_logits
    
    
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        
        
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

#InceptionAux 是一个定义了辅助分类器的类，通常用于中间层的特征进行分类。
#self.aux1 和 self.aux2 分别用于处理不同深度的特征输出：512和528个通道的特征图。
#num_classes 是最终分类的类别数（例如，1000类），用于确保输出的维度正确。

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
#这段代码通过条件性地调用权重初始化方法，确保在需要时对模型的权重进行适当的初始化，从而为训练打下良好的基础。这样的设计使得模型在创建时可以灵活地选择是否初始化权重。
        if init_weights:
            self._initialize_weights()

            
            
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        
        
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

            
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        
        
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

            
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x
    
    
    
#对于每个卷积层（nn.Conv2d），使用 Kaiming 正态分布初始化其权重，这种方法适合 ReLU 激活函数，能够帮助缓解梯度消失问题。如果卷积层有偏置（bias），则将偏置初始化为零。
#对于全连接层（nn.Linear），使用正态分布（均值为0，标准差为0.01）初始化权重。偏置也被初始化为零。

    def _initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

                
                
# Inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

#nn.Sequential 是 PyTorch 中的一个容器，用于将多个神经网络层按顺序组合在一起。它
        
        
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        
        return torch.cat(outputs, 1)  # 按 channel 对四个分支拼接 
    
# torch.cat(outputs, 1)
#多尺度特征提取：每个分支可能使用不同的卷积操作（如1x1、3x3、5x5卷积和池化），因此它们能够提取不同尺度的特征。拼接后，模型可以综合这些特征，从而提高分类性能。
#维度扩展：拼接后，输出张量的通道数会增加，允许后续层处理更多的特征信息。例如，如果每个分支输出的通道数为64，拼接后，输出的通道数将是256（64x4）。
#保持空间信息：通过沿通道维度拼接，空间维度（高度和宽度）保持不变，使得后续层可以继续处理相同大小的特征图。




# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        
        
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        
    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


# 基础卷积层（卷积+ReLU）
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
```

`def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):`

**区分实例变量**：通过`self`，你可以定义实例属性，例如`self.aux_logits`、`self.conv1`等。这样，每个实例都可以有独立的属性值。

**访问其他方法和属性**：你可以在类的其他方法中使用`self`来调用同一实例的其他方法或访问其他属性。例如，你可以在`forward`方法中使用`self.conv1`来调用在`__init__`中定义的卷积层。

**保持状态**：`self`使得对象能够保持状态，实例的属性可以在对象的生命周期内存储数据，这对于构建面向对象的程序非常重要。



`super(GoogLeNet, self).__init__()` 的作用是调用父类的构造函数。在面向对象编程中，特别是在使用继承时，这样做有几个重要的用途：

1. **初始化父类属性**：通过调用父类的构造函数，`GoogLeNet`类可以继承并初始化其父类（`nn.Module`）中的属性。这确保了父类的一些基本设置（如内部状态和必要的初始化）得以完成。
2. **继承功能**：`GoogLeNet`类是从`nn.Module`继承的，调用`super()`使得子类可以使用父类中定义的方法和属性，比如`forward`方法和模型的注册机制。
3. **多重继承支持**：在使用多重继承时，`super()`能确保调用正确的父类构造函数，遵循方法解析顺序（MRO），保证代码的可维护性和一致性。



`local_response_normalization`（LRN）是一种归一化技术，主要用于深度学习中的卷积神经网络。它在一些早期的卷积神经网络架构（如AlexNet）中被使用，目的是增强模型的泛化能力。LRN 通过对局部区域的激活进行归一化，来增加神经元的竞争性。





辅助分类器

### 1. **缓解梯度消失问题**

- 在深层网络中，随着网络深度的增加，梯度在反向传播过程中可能会变得非常小，导致模型难以训练。辅助分类器通过在中间层引入额外的损失信号，可以有效地缓解这个问题。

### 2. **提供额外的监督信号**

- 辅助分类器能够在网络的中间层对特征进行分类，这意味着在训练期间，网络不仅根据最终输出进行优化，还会利用中间层的输出进行优化。这种额外的监督信号有助于网络更好地学习特征。

### 3. **增强模型的泛化能力**

- 通过训练辅助分类器，模型能够学习到更丰富的特征表示，从而提升对未见数据的泛化能力。这样在测试阶段，主分类器的性能通常会有所提升。

### 4. **提高收敛速度**

- 辅助分类器可以帮助模型更快地收敛，尤其是在深度网络的初期训练阶段。由于有额外的损失信号，模型能够更有效地调整权重。

### 5. **多任务学习**

- 辅助分类器允许模型在多个任务上同时进行训练，从而提高整体性能。在某些应用中，可能希望模型不仅进行主任务的分类，还能进行其他相关任务的学习。



**resnet**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d2a92a16242e41eb4128fae07d49b8d1.png)



ResNet 的核心思想是通过**残差块（Residual Block）\**来解决随着网络深度增加，训练过程中出现的梯度消失和梯度爆炸问题。传统的深层神经网络往往很难训练，因为深度增加后，梯度可能会逐渐消失，导致模型表现变差。ResNet 通过引入\**跳跃连接（skip connection）** 或称为**短路连接（shortcut connection）**，让输入直接绕过若干层并加到输出上，缓解了这个问题。

```python
#model.py

import torch.nn as nn
import torch

#18/34
class BasicBlock(nn.Module):
    expansion = 1 #每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):#downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)#BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x #捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

#50,101,152
class Bottleneck(nn.Module):
    expansion = 4#4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,#输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):#block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
```





**densenet**



DenseNet 相较于传统的卷积网络（如 ResNet）有一个显著的不同点，就是其**密集连接**的结构。在 DenseNet 中，**每一层的输出都被传递给后面所有的层**，也就是说，第 lll 层的输出不仅会被传递给第 l+1l+1l+1 层，还会传递给第 l+2l+2l+2、l+3l+3l+3 等层。这种设计大幅度增加了特征的复用，并且强化了梯度的传递

这种设计大幅度增加了特征的复用，并且强化了梯度的传递。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义Dense Block
#目的：DenseBlock的目的是通过密集连接（每一层都与前面的所有层相连）来提高特征的利用率。这样可以减轻梯度消失的问题，并提高信息流动。
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

            
            
            
    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

        
        
        
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

    
    
# 定义Transition Layer
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        
        
def forward(self, x):
    return self.layer(x)

# 定义DenseNet
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_blocks=4, num_layers_per_block=6, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # Initial Convolution layer
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense Blocks and Transition Layers
        num_channels = 2 * growth_rate
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(DenseBlock(num_channels, growth_rate, num_layers_per_block))
            num_channels = num_channels + num_layers_per_block * growth_rate
            if i != num_blocks - 1:  # Skip transition layer after the last block
                self.blocks.append(TransitionLayer(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        # Global average pooling and classification layer
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        for block in self.blocks:
            x = block(x)

        x = F.relu(self.bn2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 初始化DenseNet模型
model = DenseNet(growth_rate=32, num_blocks=4, num_layers_per_block=6, num_classes=10)

# 测试模型输出
x = torch.randn(1, 3, 224, 224)  # 假设输入图像为 224x224
output = model(x)
print(output.shape)  # 输出的形状为 (1, 10)，对应10个分类
```