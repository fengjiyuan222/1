**RNN**

RNN之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。

![img](https://i-blog.csdnimg.cn/blog_migrate/9bde293942231a0eeecea9e863106106.png)

**one-to-one**

![img](https://i-blog.csdnimg.cn/blog_migrate/c41844735cc06b24e8852e37446b2683.png)

**one-to-n**

![img](https://i-blog.csdnimg.cn/blog_migrate/bbfe3f21ce47babd4edb2dcda20aa18f.png)

**n-to-n**

最经典的RNN结构，输入、输出都是等长的序列数据。



RNN引入了隐状态h（hidden state）的概念，h可以对序列形的数据提取特征，接着再转换为输出。先从h1的计算开始看：

![img](https://i-blog.csdnimg.cn/blog_migrate/22f24be09e7a5a8894bfdef3ac2737b8.png)

**每一步使用的参数U、W、b都是一样的，也就是说每个步骤的参数都是共享的，这是RNN的重要特点，一定要牢记。**

![img](https://i-blog.csdnimg.cn/blog_migrate/2872defe9c0a18a8b0da16d3e38b8581.png)

依次计算剩下来的

![img](https://i-blog.csdnimg.cn/blog_migrate/95bee7ff3717d7b8b9d004b9fb462176.png)

得到输出值的方法就是直接通过h进行计算

**一个箭头就表示对对应的向量做一次类似于f(Wx+b)的变换，这里的这个箭头就表示对h1进行一次变换，得到输出y1**

![img](https://i-blog.csdnimg.cn/blog_migrate/068791e9e997447c03627567169c6d2b.png)

剩下的输出类似进行（使用和y1同样的参数V和c）

![img](https://i-blog.csdnimg.cn/blog_migrate/55860d012b9ba4b249104918d1c8abd5.png)

这就是最经典的RNN结构，它的输入是x1, x2, …..xn，输出为y1, y2, …yn，也就是说，**输入和输出序列必须要是等长的**。



**n-to-one**

输出是一个单独的值而不是序列，这种结构通常用来处理序列分类问题。

![img](https://i-blog.csdnimg.cn/blog_migrate/798c3ecb13e2e2ae1e0e489fd6720978.png)





**Encoder-Decoder**

是 **n-to-m**，输入、输出为不等长的序列。

**Encoder-Decoder结构先将输入数据编码成一个上下文语义向量c**

![img](https://i-blog.csdnimg.cn/blog_migrate/416e8b297fa25fc65a4b213da11b013d.png)

**语义向量c**可以有多种表达方式，最简单的方法就是把Encoder的最后一个隐状态赋值给c，还可以对最后的隐状态做一个变换得到c，也可以对所有的隐状态做变换。

**拿到c之后，就用另一个RNN网络对其进行解码**，这部分RNN网络被称为Decoder。Decoder的RNN可以与Encoder的一样，也可以不一样。具体做法就是将c当做之前的初始状态h0输入到Decoder中
![img](https://i-blog.csdnimg.cn/blog_migrate/6dfe4c27429d9392147faf90874d048a.png)

- Encoder：将 input序列 →转成→ 固定长度的向量
- Decoder：将 固定长度的向量 →转成→ output序列
- Encoder 与 Decoder 可以彼此独立使用，实际上经常一起使用



**Encoder-Decoder 缺点**

1.最大的局限性：编码和解码之间的唯一联系是固定长度的语义向量c
2.编码要把整个序列的信息压缩进一个固定长度的语义向量c
3.语义向量c无法完全表达整个序列的信息
4.先输入的内容携带的信息，会被后输入的信息稀释掉，或者被覆盖掉
5.输入序列越长，这样的现象越严重，这样使得在Decoder解码时一开始就没有获得足够的输入序列信息，解码效果会打折扣

**Attention Mechanism**

Attention机制通过在每个时间输入不同的c来解决问题

![img](https://i-blog.csdnimg.cn/blog_migrate/d8871b55950efec7b184d3d1fa08bcdc.png)

每一个c会自动去选取与当前所要输出的y最合适的上下文信息。具体来说，我们用aij衡量Encoder中第j阶段的hj和解码时第i阶段的相关性，最终Decoder中第i阶段的输入的上下文信息 ci就来自于所有 hj 对 aij 的加权和。

![img](https://i-blog.csdnimg.cn/blog_migrate/79dbb5f5ab854d56974f7ea37230ed5c.png)



**这些权重 aij 是怎么来的？**

aij 同样是从模型中学出的，它实际和Decoder的第i-1阶段的隐状态、Encoder第j个阶段的隐状态有关。

**a1j 的计算**

![img](https://i-blog.csdnimg.cn/blog_migrate/d67289aaa613906092e8d663e4072181.png)

**a2j 的计算**

![img](https://i-blog.csdnimg.cn/blog_migrate/bf31fcec5d3db03f14d5092c84e6c574.png)

**a3j 的计算**

![img](https://i-blog.csdnimg.cn/blog_migrate/2baab74bc1fcb201801eae97d280505d.png)



Attention 的优点：
1.在机器翻译时，让生词不只是关注全局的语义向量c，增加了“注意力范围”。表示接下来输出的词要重点关注输入序列种的哪些部分。根据关注的区域来产生下一个输出。
2.不要求编码器将所有信息全输入在一个固定长度的向量中。
3.将输入编码成一个向量的序列，解码时，每一步选择性的从序列中挑一个子集进行处理。
4.在每一个输出时，能够充分利用输入携带的信息，每个语义向量Ci不一样，注意力焦点不一样。
Attention 的缺点
1.需要为每个输入输出组合分别计算attention。50个单词的输出输出序列需要计算2500个attention。
2.attention在决定专注于某个方面之前需要遍历一遍记忆再决定下一个输出是以什么。

**Multilayer RNNs**

![img](https://i-blog.csdnimg.cn/blog_migrate/b2be45c84d9f24dae59922a2cf962a1a.png)



#### RNN数学原理

![è¿éåå¾çæè¿°](https://i-blog.csdnimg.cn/blog_migrate/8f72b8b85a5b93cce05dfdb2a06b67df.png)

有一条单向流动的信息流是从输入单元到达隐藏单元的，与此同时另一条单向流动的信息流从隐藏单元到达输出单元。在某些情况下，RNNs会打破后者的限制，引导信息从输出单元返回隐藏单元，这些被称为“Back Projections”，并且隐藏层的输入还包括上一隐藏层的状态，即隐藏层内的节点可以自连也可以互连。

**其中计算第t次的隐含层状态时为**

![s_{t} = f(U*x_t + W*s_{t-1})](https://i-blog.csdnimg.cn/blog_migrate/8375f59f8bc4d156bda549a48affb407.gif)

![è¿éåå¾çæè¿°](https://i-blog.csdnimg.cn/blog_migrate/67bf3c010343fbe1b196f23d4c6392ca.png)





**RNN网络前向传播过程中满足下面的公式**

![è¿éåå¾çæè¿°](https://i-blog.csdnimg.cn/blog_migrate/aad0dc33d2d0f672449f0fff580c62da.png)



#### LSTM

RNN模型如果需要实现长期记忆的话需要将当前的隐含态的计算与前n次的计算挂钩

![s_t=f(U*x_t+W_1*s_{t-1}+W_2*s_{t-2}+...W_n*s_{t-n})](https://i-blog.csdnimg.cn/blog_migrate/82555b9c85297d9f972650c1d3eb3544.gif)





**标准的RNN**

![è¿éåå¾çæè¿°](https://i-blog.csdnimg.cn/blog_migrate/e913138606d7c003b57e62952595a216.png)

**LSTM**

![img](https://i-blog.csdnimg.cn/blog_migrate/a5f771bede44cfbee9cd9c694bd0a825.png)

![è¿éåå¾çæè¿°](https://i-blog.csdnimg.cn/blog_migrate/586795bd058f8e6225a0d8c330fdc875.png)

LSTM 的关键就是**细胞状态（cell）**，水平线在图上方贯穿运行。细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。 

![è¿éåå¾çæè¿°](https://i-blog.csdnimg.cn/blog_migrate/6886f249fef5bcbf0588deb3055bf20b.png)

LSTM 有通过精心设计的称作为“**门”的结构来去除或者增加信息到细胞状态的能力**。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。

![è¿éåå¾çæè¿°](https://i-blog.csdnimg.cn/blog_migrate/d5c1471a1d6f2834cc082e3b9d3adfdd.png)

LSTM中有3个控制门：输入门，输出门，记忆门。

**forget gate：选择忘记过去某些信息：**

![img](https://i-blog.csdnimg.cn/blog_migrate/8e2043034a720edee2a3859d0e27ef39.png)

**input gate：记忆现在的某些信息：**

![img](https://i-blog.csdnimg.cn/blog_migrate/889bffa0695e0364c066616eed112526.png)

将过去与现在的记忆进行合并：

![img](https://i-blog.csdnimg.cn/blog_migrate/372b980795161a1d1c31e49c3f3bd1cd.png)

**output gate：输出**

![img](https://i-blog.csdnimg.cn/blog_migrate/e8253c61452f84a946520a05af926984.png)

**公式总结**

![\begin{pmatrix} i\\ f\\ o\\ g \end{pmatrix}=\begin{pmatrix} \sigma \\ \sigma\\ \sigma\\ tanh \end{pmatrix}W\begin{pmatrix} h_{t-1}\\ x_t \end{pmatrix}](https://i-blog.csdnimg.cn/blog_migrate/cff6339ffb2c955b7ff8ab1a750c349d.gif)

![h_t=o\odot tanh(c_t)](https://i-blog.csdnimg.cn/blog_migrate/9e05449afad08dba38ff06837044b3ff.gif)

![h_t=o\odot tanh(c_t)](https://i-blog.csdnimg.cn/blog_migrate/9e05449afad08dba38ff06837044b3ff.gif)

![img](https://i-blog.csdnimg.cn/blog_migrate/476d99bb79c51f9a29b00c2041c710d5.png)



