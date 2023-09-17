# 注意力机制

#### 1. 网络架构设计的动机
由神经网络的万能逼近定理，只要参数数量足够多，一个具有单个隐藏层的MLP理论上可以拟合所有函数，
将单层的超大隐藏层拆分成多层MLP，主要是为了加入更多非线性层来从更少参数中获得同等的非线性能力。

卷积层同样可以展开为一个稀疏的全连接层，可见卷积层实质上是利用先验知识（在图像上相近的像素相关性更强，更可能被融合为特征）
对全连接层中的大矩阵参数进行约束，从而用更少参数表达了带有先验偏好的全连接层。

综上，我们可以将网络架构设计的动机归结为以下三种：
1. 基于先验知识设计网络结构，等价于给出一种全连接层中参数约束方法，从而用少量参数表达一个具有先验知识的全连接层，
   让网络拥有先验知识，更易学到目标知识。此类设计的代表有卷积层、GRU中的门结构、RepVGG中的分支结构等。
2. 改变非线性函数的类型与插入位置，使模型获得更灵活的非线性能力。此类设计的代表有ResNet中的shortcut（第二版中的BN-Relu-conv-merge
   融合了relu前与relu后的特征），自注意力机制等）
3. 基于性能的考量，如提高网络计算的并行性、降低计算所需的缓存等来进行网络设计，一个典型例子是RNN的不可并行性导致的衰落。
<br><br>
#### 2. 注意力机制
![三种架构](https://zh-v2.d2l.ai/_images/cnn-rnn-self-attention.svg)
卷积网络的优点在于提供了局部性这一先验知识，使网络更容易学到抽取特征的能力。在图像数据上，这种局部性是符合直觉与常识的；在自然语言数据上，这种局部性却未必成立。要想理解一个词表达的含义，往往需要用到它的上下文，相关的上下文有时在词的附近，有时却可能相距甚远。

因此，我们希望我们的网络层有较宽广乃至全局化的视野，能够看到所有的上下文（或仅上文），又能像人一样具有注意力，仅仅关注与当前词语相关的部分。前者可以靠全连接来实现，后者理论上也可以由全连接层习得，但这可能过于困难。**我们希望引入一个类似于mask的机制，来遮蔽输入中与当前需要编码的部分不相关的内容，突出相关的内容，让后续网络可以专心做特征的融合。所谓的注意力机制，就是指计算这个mask的方法。**
<br><br>
##### 2.1 qkv架构
注意力机制常用的实现方式是使用qkv架构，即$query,key,value$。
$query$与$key$分别表示当前信息希望用于匹配其他信息的特征与希望被其他信息匹配的特征，通过计算所有的$(query, key)$对的匹配度，可以每个最终编码每个$value$时需要关注哪些位置的$value$（即高匹配度的$value$）。

<br><br>
##### 2.2 非参数注意力与自注意力
非参数注意力指获得q,k以及计算q,k匹配度时没有需要学习的参数的情况。
自注意力则值q,k均从**自身信息**中编码得到（一种极端的情况为用自身信息直接作为q,k,v），而非作为某种附加结构存在。

###### 2.2.1 Nadaraya-Watson核回归
Nadaraya-Watson核回归就是一种典型的非参数回归模型，考虑一元函数的回归问题，假设需要回归的数据形式为$(x_i,y_i)$，令$q_i=k_i=x_i, v_i=y_i$，用高斯核$K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2})$来计算q,k的匹配度并作为最终融合v的权重，最终可得:
$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$
这可以被直观理解为越近的数据应当相关性越强，而更需要被注意，因而拥有更高的权重
<br><br>
###### 2.2.2 内积注意力
考虑一个更为复杂的例子，假设需要自回归的数据为$(\vec{x_i})$，令$q_i=k_i=v_i=\vec{x_i}$，如果这些$\vec{x_i}$均由word2vec等方法编码得到并已对模长做正则化，即可以用$cos<\vec{x_i}, \vec{x_j}>$来表示$\vec{x_i}, \vec{x_j}$之间的相关性，则我们可以通过$QK^T$来获得所有$(q_i,v_i)$的相关性，再通过$softmax(QK^T)$来获得最终的注意力mask
![内积注意力](https://pic2.zhimg.com/80/v2-179fd393b3aac244ec338767ef5d8d3d_1440w.webp)
<br><br>
##### 2.3 参数注意力
参数注意力指编码q,k或计算q,k匹配度时有需要学习的参数的情况，参数一般在编码过程中体现。

2.2.2中提到的内积注意力的一个明显缺点是，自身对自身的注意力永远是最大的，很容易陷入“自恋”中去。从现实经验来看，我们关注的与其说是与自身相似的，不如说是与自身的某种特质关联的；我们希望展示给他人的也未必是真实的自己，而是经过修饰的。因此，我们不直接使用自身信息来作为q,k，甚至v，而是用三个全连接层对原始$input, shape_{input}=(batch,seqlen,dim)$分别做线性变换后，把结果作为q,k,v，再通过softmax与内积进行mask的计算。这里的$W^Q,W^K,W^V$可以理解为新特征空间的一组基
![q,k,v](https://pic4.zhimg.com/80/v2-28903ee6a9c01d4895af7836b1e5997f_1440w.webp)
![内积](https://pic4.zhimg.com/80/v2-3173490f0b8fb89b22a22b65d2851c7f_1440w.webp)
![mask计算](https://pic3.zhimg.com/80/v2-a574d12396e1e2006716eb58f9fa5806_1440w.webp)
<br><br>

#### 3. 多头注意力
2.3中实现的注意力计算方式一般被称为单头注意力，其特征是会在dim维度上做全局的softmax，由于softmax的特性，很容易造成过于关注单个位置或过于关注自身的局面。我们希望由原始输入生成多组q,k,v,获得多个小型mask并作用于多个v，再拼接为最终输出。

![多头注意力](https://pic4.zhimg.com/80/v2-382a68f2a5543f00b7a4a1fd84e29b83_1440w.webp)

图中实现多头注意力的方式是对$W^Q,W^K,W^V$在最后一维上做分割，2.3中提到$W^Q,W^K,W^V$可以理解为新特征空间的一组基，则分割可以认为是将**原来向高维特征空间的单次映射转换为向多个低维子空间的多次映射**。最后在每个头内部进行注意力权重的计算，**这种类似分组卷积的处理减少了计算量。**

在代码实现中，我们一般不分割$W^Q,W^K,W^V$，而是直接分割最终得到的$Q,K,V$矩阵，容易证明这是等价的。我们将$ Q,K,V$矩阵从$(batch,seqlen,dim)$分割为$(batch, seqlen, head,dim // head)$，再转为$(batch, head, seqlen, dim // head)$， 计算完毕后再转换并concat为$(batch, seqlen, dim)$。
```
def forward(self, X, attention_mask = None, head_mask = None):
        q, k, v = self.W_q(X), self.W_k(X), self.W_v(X)
        q, k, v = self._split_m_head(q), self._split_m_head(k), self._split_m_head(v)
        
        weight = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.dim / self.m_head)
        if attention_mask is not None:
            weight += attention_mask
        weight = self.dropout(torch.softmax(weight, dim=-1))
        
        if head_mask is not None:
            weight *= head_mask
        
        v = torch.matmul(weight, v)
        batch_size, num_heads, seq_len, head_dims = v.shape
        v = v.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dims)
        
        return v
```

#### 4. 其他相关组件
###### 4.1 attention_mask & head_mask
* attention_mask用于遮蔽对某些token的注意力，如当前时间步本不该被看到的内容，其使用时机在计算$QK^T$与计算softmax之间，故使用加性处理即可，只需给希望遮蔽的部分加上一个较大的负常数即可。
```
weight = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.dim / self.m_head)
if attention_mask is not None:
    weight += attention_mask
weight = self.dropout(torch.softmax(weight, dim=-1))
```
* head_mask用来遮蔽某些头的注意力，如当前任务不需要用到的头。其使用时机在softmax后计算注意力权重与v的乘积前，使用哈达玛积计算。
```
if head_mask is not None:
    weight *= head_mask
v = torch.matmul(weight, v)
```
<br>

###### 4.3 正则化层
不同于卷积网络中常用的BN层在Batch维度做正则化，Transformer架构中的正则化是在样本内部完成的，更具体来说是在dim维度完成的，好处是对样本量依赖小，对序列长度依赖小等。

基于使用正则化时机的不同，可把正则化分为$post-norm$与$pre-norm$，即在进入注意力层后做正则化还是在进入注意力后做正则化。目前以第二种更为主流。

目前使用的正则化方法主要有以下两种：
* LayerNorm： 对dim维计算均值$\mu$，并用$\gamma\mu + \beta$重新映射
* RMSNorm：将原始向量除以各分量均方根，即将模长缩放为$\sqrt{dim}$,再用$\gamma$做缩放
<br>
###### 4.2 位置编码



#### 5. Transformer的实现

#### 6. Decoder-Encoder架构

#### 7. 生成任务推理方案

#### 8. Llama2的实现