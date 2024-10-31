以下是Transformer的一些重要组成部分和特点：

**1.自注意力机制（Self-Attention）**：这是Transformer的核心概念之一，它使模型能够同时考虑输入序列中的所有位置，而不是像循环神经网络（RNN）或卷积神经网络（CNN）一样逐步处理。自注意力机制允许模型根据输入序列中的不同部分来赋予不同的注意权重，从而更好地捕捉语义关系。
**2.多头注意力（Multi-Head Attention）**：Transformer中的自注意力机制被扩展为多个注意力头，每个头可以学习不同的注意权重，以更好地捕捉不同类型的关系。多头注意力允许模型并行处理不同的信息子空间。
**3.堆叠层（Stacked Layers）**：Transformer通常由多个相同的编码器和解码器层堆叠而成。这些堆叠的层有助于模型学习复杂的特征表示和语义。
**4.位置编码（Positional Encoding）**：由于Transformer没有内置的序列位置信息，它需要额外的位置编码来表达输入序列中单词的位置顺序。
**5.残差连接和层归一化（Residual Connections and Layer Normalization）**：这些技术有助于减轻训练过程中的梯度消失和爆炸问题，使模型更容易训练。
编码器和解码器：Transformer通常包括一个编码器用于处理输入序列和一个解码器用于生成输出序列，这使其适用于序列到序列的任务，如机器翻译。





结构

**Encoder block由6个encoder堆叠而成，图中的一个框代表的是一个encoder的内部结构，一个Encoder是由Multi-Head Attention和全连接神经网络Feed Forward Network构成。如下图所示**：



其中，编码器负责特征编码，即从原始的、比较底层的输入序列信号（以音频为例，比如原始音频中单个采样点数据构成的序列，或者音频的人工设计的[MFCC](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Mel-frequency_cepstrum)、[Fbank](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Filter_bank)特征等）提取出抽象的、具有明显语义特征的特征信息；解码器负责从编码器得到的原始序列的特征信息中"破译"出目标序列的内容（比如从音频序列的特征中"破译"其对应的文本信息）。

![{CC96D2C2-300A-41A6-AD3C-FF6D3555BF0C}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{CC96D2C2-300A-41A6-AD3C-FF6D3555BF0C}.png)

*Transformer的编码组件是由6个编码器叠加在一起组成的，解码器同样如此。所有的编码器在结构上是相同的，但是它们之间并没有共享参数。编码器的简略结构如下：*

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/630deb7da181d99eb9dd7d70f6b4da98.png#pic_center)

**从编码器输入的句子首先会经过一个自注意力层，这一层帮助编码器在对每个单词编码的时候时刻关注句子的其它单词。解码器中的解码注意力层的作用是关注输入句子的相关部分，类似于seq2seq的注意力。**

![{3CEAAB22-F6B8-4021-9523-9CED4C594A65}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{3CEAAB22-F6B8-4021-9523-9CED4C594A65}.png)



**编码器（Encoder）**：

- 通常由多个编码器层堆叠而成，每层包含：
  - **多头自注意力机制（Multi-Head Self-Attention）**：通过自注意力机制，模型能够聚焦于输入序列中不同位置的信息。
  - **前馈神经网络（Feedforward Neural Network）**：对每个位置的表示进行进一步处理。
  - **残差连接和层归一化（Residual Connection and Layer Normalization）**：用于稳定训练，提高性能。

**解码器（Decoder）**：

比编码器多一个

**Masked Multi-Head Self-Attention**：防止在生成时看到后续的单词。



### 自注意力机制

**序列建模**：自注意力可以用于序列数据（例如文本、时间序列、音频等）的建模。它可以捕捉序列中不同位置的依赖关系，从而更好地理解上下文。这对于机器翻译、文本生成、情感分析等任务非常有用。

**并行计算**：自注意力可以并行计算，这意味着可以有效地在现代硬件上进行加速。相比于RNN和CNN等序列模型，它更容易在GPU和TPU等硬件上进行高效的训练和推理。（因为在自注意力中可以并行的计算得分）

**长距离依赖捕捉**：传统的循环神经网络（RNN）在处理长序列时可能面临梯度消失或梯度爆炸的问题。自注意力可以更好地处理长距离依赖关系，因为它不需要按顺序处理输入序列。

![img](https://pic2.zhimg.com/v2-b197149ad3bc4f14897a93fc61802ce3_b.jpg)

Transformer中用的注意力机制包括**Query (** Q **)，Key (** K **)和Value (** V **)**三个组成部分（学习过数据库的同学对这三个名词应该比较熟悉）。可以这样理解， V 是我们手头已经有的所有资料，可以作为一个知识库； Q 是我们待查询的东西，我们希望把 V 中和 Q 有关的信息都找出来；而 K 是 V 这个知识库的钥匙 ，V中每个位置的信息对应于一个 K 。对于 V 中的每个位置的信息而言，如果 Q 和对应钥匙 的匹配程度越高，那么就可以从该条信息中找到和 Q 更多的内容。

**1.计算 Q 和 K 的相似度**

 **计算Q和K的相似度**：既然要计算相似度，我们应当首先确定两个向量之间的相似度的度量标准。简单起见，我们直接以内积作为衡量两个向量之间相似度的方式

向量之间的相似度衡量并没有特定标准，比如也可以用两个向量之间的余弦相似度，不过现在主流的还是直接两个向量做内积。

![{00ED228B-B0DB-4757-8195-749570CE2BEF}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{00ED228B-B0DB-4757-8195-749570CE2BEF}.png)

**2.根据计算得到的相似度，取出 V 每条信息中和 Q 有关的内容**

得到 Q 和 K 之间的相似度 attention 之后，我们就可以计算 Q 中的每一个位置的元素和 V 中所有位置元素的注意力运算结果。

![{1CF3BC00-B0FA-4115-A103-8B481F16AB64}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{1CF3BC00-B0FA-4115-A103-8B481F16AB64}.png)



注意力机制本质上可以认为是**求一个离散概率分布的数学期望**。定义一个离散的随机变量 X ，随机变量 X 的所有可能取值为 [X1,X2,X3,X4] ，X∼P，离散分布 P=[p1,p2,p3,p4] ，于是注意力的计算结果就是随机变量 X 的数学希望 EX=∑i=i4Xipi ，而离散分布 P 就是上文通过 Q 和 K 计算得到的 softmax 归一化之后的attention 。通过这里的解释，我们也可以更好地理解为什么计算 attention 的时候需要使用 softmax 函数进行归一化，归一化之后的每一个 attention 行向量都是一个离散的概率分布，具有更好的数学意义。







**自注意力的计算**

第一步

生成三个向量，即**查询向量、键向量和一个值向量**。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cc00beb97c344a486d07e3d9e8a58f06.png#pic_center)

查询向量、键向量、值向量组合起来就可以得到三个向量矩阵Query、Keys、Values。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d027e1a13de1965169178cb2e5eb9ec1.png#pic_center)

**第二步**是计算得分

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/59ee8dec83e584b860869ef701a3a4e3.png)

**第三步和第四步**是将分数除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值，这样做是为了防止内积过大。)，然后通过softmax传递结果。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bc1a451b1bb9c568a382f9d08fa28341.png)

**第五步**是将每个值向量乘以softmax分数(这是为了准备之后将它们求和)。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/00994ceb6bf9e66db19611c496463364.png#pic_center)

**第六步**是对加权值向量求和，然后即得到自注意力层在该位置的输出

整体的计算图如图所示：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e976d386a1aad85c2efb7fc965099c27.png#pic_center)

相比于原始的attention计算公式，上述公式多了一个系数 dk 。这一做法有点类似一种正则化，避免 QKT 的数值计算结果过大，导致 softmax 向着梯度很小的区域偏移。

最终得到了自注意力，并将得到的向量传递给前馈神经网络。





**自注意力层的完善——“多头”注意力机制：**

**Multi——Head Attention**

1、扩展了模型专注于不同位置的能力。
2、有多个查询/键/值权重矩阵集合，（Transformer使用八个注意力头）并且每一个都是随机初始化的。和上边一样，用矩阵X乘以WQ、WK、WV来产生查询、键、值矩阵。
3、self-attention只是使用了一组WQ、WK、WV来进行变换得到查询、键、值矩阵，而Multi-Head Attention使用多组WQ，WK，WV得到多组查询、键、值矩阵，然后每组分别计算得到一个Z矩阵。



Transformer中还对上述注意力机制进行了改进，使用了“多头注意力机制”(Multi-Head Attention)。多头注意力机制假设输入的特征空间可以分为互不相交的几个子空间，然后我们只需要在几个子空间内单独计算注意力，最后将多个子空间的计算结果拼接即可。举个例子，假设 Q,K,V 的维度都是512，长度都是 L ，现将维度512的特征空间分成8个64维的子空间，每个子空间内单独计算注意力，于是每个子维度的计算结果的尺寸为 (L,64) ，然后再将8个子维度的计算结果沿着特征维度拼起来，得到最终的计算结果，尺寸为 (L,512) ，和输入的 Q 的尺寸保持一致。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a923f7bb907110650448d4b773bf0671.png#pic_center)

前馈层只需要一个矩阵，则把得到的8个矩阵拼接在一起，然后用一个附加的权重矩阵WO与它们相乘。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fd6af04ca65df88d3f13f4aaf987b0f3.png#pic_center)

**总结整个流程：**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0b8dacfc201e24ef7dc0e690b41b998c.png#pic_center)





### Add&Normalize

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/29a24a78b70aa77ffd41b5ae2bfdc5e7.png#pic_center)

**Add**
Add，就是在z的基础上加了一个残差块X，加入残差块的目的是为了防止在深度神经网络的训练过程中发生退化的问题，退化的意思就是深度神经网络通过增加网络的层数，Loss逐渐减小，然后趋于稳定达到饱和，然后再继续增加网络层数，Loss反而增大。

**Normalize**
归一化目的：
1、加快训练速度
2、提高训练的稳定性
使用到的归一化方法是Layer Normalization。

### 全连接层Feed Forward

全连接层是一个两层的神经网络，先线性变换，然后ReLU非线性，再线性变换。

这两层网络就是为了将输入的Z映射到更加高维的空间中然后通过非线性函数ReLU进行筛选，筛选完后再变回原来的维度。

经过6个encoder后输入到decoder中。



### 注意力机制完整代码

```python
class MultiHeadAttention(nn.Module):
    
#d_k: 每个注意力头的键（Key）和查询（Query）的维度。
#d_v: 每个注意力头的值（Value）的维度。
#d_model: 输入特征的总维度。
#num_heads: 注意力头的数量。
#p: dropout 比例，用于防止过拟合。
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        super(MultiHeadAttention, self).__init__()
        #存储参数
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)
        
        #线性变化 用于将输入的查询、键和值映射到相应的维度。
        #nn.Linear(d_model, d_k * num_heads)
        #d_model输入维度  d_k * num_heads转变成的输出维度
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        # Normalization
        #nn.init.normal_ 函数会将指定张量（这里是线性层的权重）用均值为 0 和指定标准差的正态分布进行填充。
        #初始化查询权重（W_Q）的权重，维度为 d_model 到 d_k * num_heads。
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        
    def forward(self, Q, K, V, attn_mask, **kwargs):
#Q: 查询（Query）向量。
#K: 键（Key）向量。
#V: 值（Value）向量。
#attn_mask: 注意力掩码，用于屏蔽掉不需要关注的部分
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        #多头分割
        #通过线性层将 Q 转换为 (N, seq_len, d_k * num_heads) 的形状。
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)
        
        # pre-process mask
#如果提供了 attn_mask，将其形状调整为 (N, 1, q_len, k_len)，并进行广播，以便能在后续的注意力得分计算中使用。
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)    # broadcast
            attn_mask = attn_mask.bool()
            
        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)        # attention weights
        attns = self.dropout(attns)

        # 计算输出
        output = torch.matmul(attns, V)

        # 多头合并
        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)
        output = self.W_out(output)

        return output
```

**位置编码**

![{DDDF0097-6D64-451E-9A57-D8ED81843A21}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{DDDF0097-6D64-451E-9A57-D8ED81843A21}.png)

```python
def pos_sinusoid_embedding(seq_len, d_model):
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embeddings.float()
```

**逐位置前馈网络**

```python
class PoswiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, p=0.):
        super(PoswiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, 1, 0)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        out = self.conv1(X.transpose(1, 2))     # (N, d_model, seq_len) -> (N, d_ff, seq_len)
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)   # (N, d_ff, seq_len) -> (N, d_model, seq_len)
        out = self.dropout(out)
        return out
```



### Decoder整体结构

和Encoder Block一样，Decoder也是由6个decoder堆叠而成的，Nx=6。包含两个 Multi-Head Attention 层。第一个 Multi-Head Attention 层采用了 Masked 操作。第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算。

**Masked Multi-Head Attention**

与Encoder的Multi-Head Attention计算原理一样，只是多加了一个mask码。mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。


**padding mask**
什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。
具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！

**sequence mask**
sequence mask 是为了使得 decoder 不能看见未来的信息。对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。这在训练的时候有效，因为训练的时候每次我们是将target数据完整输入进decoder中地，预测时不需要，预测的时候我们只能得到前一时刻预测出的输出。



### 掩码机制

- encoder的self attention的长度mask
- decoder的self attention的causal mask
- encoder和decoder的cross-attention的mask

**encoder的self attention的长度mask**

具体来说，既然我们的目的是为了让attention的计算结果不受padding的影响，那么一个比较简单的方法是，直接将padding对应位置的attention权重置为0即可。

也就是说，除了当 Q 和 K 都为前4个token以外，其余情形的attention权重均为0，因为该情形下 Q 和 K 总有一个为padding的部分。如下图所示，其中白色代表attention权重不为0，灰色代表attention权重为0，即被mask掉的部分。

![img](https://pic3.zhimg.com/v2-a0f302d4ac364f54c6114b19d5fc96aa_b.jpg)

子图 (a) 表示只要 Q 和 K 其一为padding，那么我们就将其attention权重置为0；而子图 (b) 表示当 K 为padding时将对应attention权重为0。实际模型训练过程中使用(b)而不使用(a)，使用 (b) 不会出错是因为 Q 为padding的部分最终计算loss时会被过滤掉，所以 Q 是否mask无影响。

而使用(a)时，由于有些行的所有位置都被mask掉了，这部分计算attention时容易出现NaN。举个例子，我们可以将"早上好！"后面的4个位置的文本字符都用一个特殊的token "<ignore>"来填充，然后再计算交叉熵损失的过程中利用`torch.nn.functional.cross_entropy`的`ignore_idx`参数设成 "<ignore>" 将这部分的loss去掉，不纳入计算。

**decoder的self attention的causal mask**

![{7BBD5A82-AC25-4C52-A939-2A4FEB097015}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{7BBD5A82-AC25-4C52-A939-2A4FEB097015}.png)

![img](https://pic3.zhimg.com/v2-cabd330eddb252ee017536b4cd4c20f0_b.jpg)

encoder中只是起到提取特征的作用，不需要像decoder那样计算自回归的交叉熵损失函数，所以不需要额外的causal约束。

理论上来说，给encoder加上causal mask也可以，但是其意义和非causal存在一定差异。对于encoder中的第 i 个位置而言，不加入causal mask时，该位置可看到整个序列的所有信息；加入causal mask时，该位置只能看到第 1∼(i−1) 个位置的信息，这一约束没有必要。



**encoder和decoder的cross-attention的mask**

![img](https://pic3.zhimg.com/v2-15b9d97638e6e61279cf0e3afed71b46_b.jpg)

掩码机制补充：

1. 要尤其注意mask是True还是False，这一点非常容易错。通常的用法是将应该被mask掉的部分置为True，但是也有一些代码是将应该被保留的部分置为True，比如Pytorch 2.0中官方实现的[scaled_dot_product_attention](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)；
2. mask有很多等价写法，比如可以将被mask掉的部分的值设为-inf，被保留部分的值设为0，然后直接`attn+=mask`即可（参考[LLaMA](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py%23L299-L300)），这和masked_fill的写法等价，用一种即可；
3. 注意mask的数据类型，最好固定为bool；
4. 总体来说，实际使用过程中的mask就只有上面提到的三种，但是mask的设计本身非常灵活，可能会根据某些情形来设置特定的mask，比如预训练中常用的掩码预测任务（参考Bert[[14\]](https://zhuanlan.zhihu.com/p/648127076#ref_14)）。又比如，我们限定第 i 个位置的 Q 只能attend到 [i−l,i−1] 范围内的 K ，那么此时mask需要做相应的更改，此处不再赘述。





**输出**
Output如图中所示，首先经过一次线性变换（线性变换层是一个简单的全连接神经网络，它可以把解码组件产生的向量投射到一个比它大得多的，被称为对数几率的向量里），然后Softmax得到输出的概率分布（softmax层会把向量变成概率），然后通过词典，输出概率最大的对应的单词作为我们的预测输出。



**Self-Attention的实现过程**

- 准备输入
- 初始化参数
- 获取key，query和value
- 给input1计算attention score
- 计算softmax
- 给value乘上score
- 给value加权求和获取output1
- 重复步骤4-7，获取output2，output3

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b021027fb8b935d6f320a54b5ae19abb.png)











完整的encoder代码如下所示：

encoder layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        assert dim % n == 0
        hdim = dim // n     # dimension of each attention head
        super(EncoderLayer, self).__init__()
        # LayerNorm
        #nn.LayerNorm 是 PyTorch 中的一个模块，用于实现层归一化（Layer Normalization）。
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        
        
        # MultiHeadAttention
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
#embed_dim：输入特征的总维度（即嵌入维度）。
#num_heads：注意力头的数量。注意力头数必须能整除 embed_dim。
#kdim（可选）：键的维度，默认为 embed_dim。
#vdim（可选）：值的维度，默认为 embed_dim。
#dropout（可选）：用于注意力权重的 dropout 比例，防止过拟合。
#bias（可选）：是否使用偏置，默认为 True。
#add_zero_attn（可选）：是否在注意力中添加一个零的注意力项，默认为 False。



        # Position-wise Feedforward Neural Network
#通常由两个线性层和一个激活函数（通常是 ReLU）组成，其结构可以表示为：
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)

    
    
    # enc_in：输入特征，通常是编码器的输入序列
    # 形状(batch_size,seq_len,d_model)
    
    
    # attn_mask：注意力掩码，用于屏蔽掉不需要关注的位置
    # 形状(batch_size,seq_len,seq_len)
    
    def forward(self, enc_in, attn_mask):
        # 保留原始输入
        residual = enc_in
        
        # MultiHeadAttention forward
        context = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        
        # 残差连接和归一化
        out = self.norm1(residual + context)
        
        # 更新残差
        residual = out
        
        
        # 位置-wise 前馈网络
        out = self.poswise_ffn(out)
        
        # 再次进行残差连接和归一化
        out = self.norm2(residual + out)

        return out
```

![{FE5700F7-9FCE-45E4-9E42-8761C9F19BA3}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{FE5700F7-9FCE-45E4-9E42-8761C9F19BA3}.png)



![{18CEC4A7-3471-443E-8FFA-7B6A8AD29736}](C:\Users\Administrator\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{18CEC4A7-3471-443E-8FFA-7B6A8AD29736}.png)



完整的encoder代码

```python
class Encoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, enc_dim, num_heads, dff, tgt_len,
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            enc_dim: input dimension of encoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the maximum length of sequences
        """
        super(Encoder, self).__init__()
        # 设置最大序列长度
        self.tgt_len = tgt_len
        # 位置嵌入
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, enc_dim), freeze=True)
        
        self.emb_dropout = nn.Dropout(dropout_emb)
        
        #encoder层 可以创建指定数量的 EncoderLayer 实例。
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )
    
    def forward(self, X, X_lens, mask=None):
        # X：输入特征 , 形状为(batch_size,seq_len,d_model)
        #  X_lens 输入特征长度  输入序列的长度（通常用于掩码）
        # mask 可选的掩码，用于屏蔽掉不需要关注的位置。
        
        
        # 添加位置嵌入
        batch_size, seq_len, d_model = X.shape
        out = X + self.pos_emb(torch.arange(seq_len, device=X.device)) 
        
        # 应用 dropout
        out = self.emb_dropout(out)
        
        
        # 通过编码器层
        for layer in self.layers:
            out = layer(out, mask)
        return out
```

### 解码器

首先从decoder layer开始

```python
class DecoderLayer(nn.Module):
    def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        super(DecoderLayer, self).__init__()
        assert dim % n == 0
        hdim = dim // n
        # LayerNorms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        # Position-wise Feed-Forward Networks
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)
        # MultiHeadAttention, both self-attention and encoder-decoder cross attention)
        self.dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.enc_dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)

    def forward(self, dec_in, enc_out, dec_mask, dec_enc_mask, cache=None, freqs_cis=None):
        # decoder's self-attention
        residual = dec_in
        context = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        dec_out = self.norm1(residual + context)
        # encoder-decoder cross attention
        residual = dec_out
        context = self.enc_dec_attn(dec_out, enc_out, enc_out, dec_enc_mask)
        dec_out = self.norm2(residual + context)
        # position-wise feed-forward networks
        residual = dec_out
        out = self.poswise_ffn(dec_out)
        dec_out = self.norm3(residual + out)
        return dec_out
```

decoder的完整代码

```python
class Decoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, dec_dim, num_heads, dff, tgt_len, tgt_vocab_size,
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            dec_dim: input dimension of decoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the target length to be embedded.
            tgt_vocab_size: the target vocabulary size.
        """
        super(Decoder, self).__init__()

        # output embedding
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)                            # embedding dropout
        # position embedding
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, dec_dim), freeze=True)
        # decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dec_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in
                range(num_layers)
            ]
        )

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask, cache=None):
        # output embedding and position embedding
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)
        # decoder layers
        for layer in self.layers:
                dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        return dec_out
```





完整

```python
class Transformer(nn.Module):
    def __init__(
            self, frontend: nn.Module, encoder: nn.Module, decoder: nn.Module,
            dec_out_dim: int, vocab: int,
    ) -> None:
        super().__init__()
        self.frontend = frontend     # feature extractor
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab)

    def forward(self, X: torch.Tensor, X_lens: torch.Tensor, labels: torch.Tensor):
        X_lens, labels = X_lens.long(), labels.long()
        b = X.size(0)
        device = X.device
        # frontend
        out = self.frontend(X)
        max_feat_len = out.size(1)                            # compute after frontend because of optional subsampling
        max_label_len = labels.size(1)
        # encoder
        enc_mask = get_len_mask(b, max_feat_len, X_lens, device)
        enc_out = self.encoder(out, X_lens, enc_mask)
        # decoder
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)
        logits = self.linear(dec_out)

        return logits
```



验证

```python
if __name__ == "__main__":
    # constants
    batch_size = 16                 # batch size
    max_feat_len = 100              # the maximum length of input sequence
    max_lable_len = 50              # the maximum length of output sequence
    fbank_dim = 80                  # the dimension of input feature
    hidden_dim = 512                # the dimension of hidden layer
    vocab_size = 26                 # the size of vocabulary

    # dummy data
    fbank_feature = torch.randn(batch_size, max_feat_len, fbank_dim)        # input sequence
    feat_lens = torch.randint(1, max_feat_len, (batch_size,))               # the length of each input sequence in the batch
    labels = torch.randint(0, vocab_size, (batch_size, max_lable_len))      # output sequence
    label_lens = torch.randint(1, max_label_len, (batch_size,))             # the length of each output sequence in the batch

    # model
    feature_extractor = nn.Linear(fbank_dim, hidden_dim)                    # alinear layer to simulate the audio feature extractor
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, enc_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, dec_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048, tgt_vocab_size=vocab_size
    )
    transformer = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size)

    # forward check
    logits = transformer(fbank_feature, feat_lens, labels)
    print(f"logits: {logits.shape}")     # (batch_size, max_label_len, vocab_size)

    # output msg
    # logits: torch.Size([16, 100, 26])
```