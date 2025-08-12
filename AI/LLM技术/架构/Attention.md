# 注意力机制


```python
'''注意力计算函数'''
def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1) 
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        # 采样
     # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

```

## 完整流程

我们可以结合矩阵维度来更精确地描述注意力机制的计算过程。假设：
- 查询矩阵$Q$的维度为$[B, N, d_k]$，其中$B$是批量大小，$N$是序列长度，$d_k$是每个查询向量的维度
- 键矩阵$K$的维度为$[B, N, d_k]$，与查询矩阵维度一致
- 值矩阵$V$的维度为$[B, N, d_v]$，其中$d_v$是每个值向量的维度（通常$d_v = d_k$）
注意力机制的完整计算过程（含维度变化）如下：
1. 计算注意力分数：$$\text{scores} = \frac{Q \cdot K^T}{\sqrt{d_k}}$$
   其中$K^T$是键矩阵的转置，维度为$[B, d_k, N]$
   运算后$\text{scores}$的维度为$[B, N, N]$

2. 对注意力分数进行 Softmax 归一化：
  $$\text{p\_attn} = \text{Softmax}(\text{scores})$$
   归一化后$\text{p\_attn}$的维度保持不变，仍为$[B, N, N]$

3. （可选）应用 Dropout：
  $$\text{p\_attn} = \text{Dropout}(\text{p\_attn})$$
   维度仍为$[B, N, N]$

4. 计算最终注意力输出：
  $$\text{output} = \text{p\_attn} \cdot V$$
   矩阵相乘后$\text{output}$的维度为$[B, N, d_v]$

完整的注意力机制公式（含维度信息）可表示为：
$$\text{Attention}(Q_{[B,N,d_k]}, K_{[B,N,d_k]}, V_{[B,N,d_v]}) = \text{Softmax}\left(\frac{Q_{[B,N,d_k]} \cdot K^T_{[B,d_k,N]}}{\sqrt{d_k}}\right)_{[B,N,N]} \cdot V_{[B,N,d_v]}$$

这个维度描述清晰地展示了注意力机制如何将查询、键、值矩阵通过矩阵运算转换为最终的注意力输出。

## 除以$\sqrt{d_k}$的作用

在注意力机制中，对查询（Q）和键（K）的内积结果除以$\sqrt{d_k}$（$d_k$为键/查询向量的维度），是为了**缓解内积结果过大导致的梯度问题和 Softmax 饱和问题**，具体原因可从以下两方面分析：
### 1. 避免内积结果随维度增长而过大
查询向量$q$和键向量$k$的内积计算公式为：  
$$q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i$$ 

假设$q$和$k$的各分量是均值为 0、方差为 1 的独立随机变量，那么内积的期望和方差为：  
- 期望：$E[q \cdot k] = \sum_{i=1}^{d_k} E[q_i \cdot k_i] = 0$（因$q_i$和$k_i$独立，$E[q_i \cdot k_i] = E[q_i] \cdot E[k_i] = 0$）。  
- 方差：$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i \cdot k_i) = d_k$（因$\text{Var}(q_i \cdot k_i) = \text{Var}(q_i) \cdot \text{Var}(k_i) = 1 \cdot 1 = 1$）。  

可见，内积结果的方差随$d_k$线性增长，即**维度越高，内积结果的数值可能越大**（例如当$d_k=512$时，内积结果可能轻松达到几十甚至上百）。


### 2. 防止 Softmax 函数进入饱和区
注意力分数经过内积计算后，会输入 Softmax 函数进行归一化：  
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$ 

当$x_i$数值过大时，$e^{x_i}$会因指数函数的特性快速趋近于无穷大，导致：  
- 数值计算上的溢出（如浮点数超出表示范围）；  
- Softmax 的输出会过于“陡峭”：即最大的$x_i$对应的概率接近 1，其他概率接近 0，导致梯度消失（因为 Softmax 在饱和区的导数接近 0）。  

而除以$\sqrt{d_k}$后，内积结果的方差被缩放为：  
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$$ 

这样一来，内积结果被限制在一个合理的数值范围内（均值 0，方差 1），避免了 Softmax 进入饱和区，保证了梯度的稳定传播和注意力权重的有效学习。


### 总结
$\sqrt{d_k}$的作用是**对 Q 和 K 的内积结果进行“标准化”缩放**，通过控制内积的方差（使其不随维度增长），**避免 Softmax 函数因输入值过大而失效，最终保证注意力机制的稳定训练和有效计算**。

# 自注意力机制

就是$Q, K, V$都相同

```python
# attention 为上文定义的注意力计算函数
attention(x, x, x)
```

# 掩码注意力机制

核心就是自注意力机制由于是矩阵运算，于是可以并行训练
对于一个序列，每个 token 都可以作为一个训练语料，即以这个 token 为分界线，前面的 token 可以看到，后面的 token 倍替换为遮盖的 mask
于是对于一个序列形成一个上三角都是 mask 的矩阵的训练语料，右上角都被遮住：

```python
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】 【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BOS>    I     like    you   </EOS>
```

具体实现来生成 mask 矩阵，-inf 会导致分数几乎为 0，所以注意力也变成了 0：

```python
# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# triu 函数的功能是创建一个上三角矩阵
mask = torch.triu(mask, diagonal=1)

#------

# 此处的 scores 为计算得到的注意力分数，mask 为上文生成的掩码矩阵
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

# 多头注意力机制

对于 view 操作熟练
对于普通注意力机制熟练
对于获取 mask 矩阵熟悉
然后看维度变化

```python
import torch.nn as nn
import torch

'''多头自注意力计算模块'''
class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x n_embd（head_dim = n_embeds / n_heads）
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
         
        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
           mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
           mask = torch.triu(mask, diagonal=1)
           # 注册为模型的缓冲区
           self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C // n_head)，然后交换维度，变成 (B, n_head, T, C // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)


        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


```

# Transformer

## FNN

```python
class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
    
```

## Layer Norm

Layer Norm（层归一化）是深度学习中一种重要的归一化技术，旨在通过标准化输入数据的分布来加速模型训练、提升稳定性。它与 Batch Norm、Instance Norm 等归一化方法的核心区别在于**归一化的维度**，这使得它在处理序列数据（如文本）或小批量场景时表现更优。

### **一、Layer Norm 的核心思想**
在深度学习中，随着网络层数加深，每一层的输入分布会因前层参数更新而不断变化（即“内部协变量偏移”），导致训练困难（如收敛慢、梯度不稳定）。归一化技术通过将输入数据标准化到均值为 0、方差为 1 的分布，缓解这一问题。

Layer Norm 的核心是：**在“特征维度”上对单个样本的所有特征进行归一化**，与样本所在的批次（batch）无关。


### **二、Layer Norm 的计算步骤**
假设输入张量为$x \in \mathbb{R}^{N \times D}$，其中$N$是样本数（batch size），$D$是每个样本的特征维度（如词向量维度、图像通道数等）。Layer Norm 的计算过程如下：


1. **计算特征维度的均值**  
   对每个样本的所有特征计算均值：  
   $$
   \mu_i = \frac{1}{D} \sum_{j=1}^{D} x_{i, j}
   $$  
   其中$x_{i, j}$是第$i$个样本的第$j$个特征，$\mu_i$是第$i$个样本的特征均值。


2. **计算特征维度的方差**  
   对每个样本的所有特征计算方差（加$\epsilon$避免分母为 0）：  
   $$
   \sigma_i^2 = \frac{1}{D} \sum_{j=1}^{D} (x_{i, j} - \mu_i)^2 + \epsilon
   $$  
   其中$\epsilon$是一个极小值（通常取$10^{-5}$）。


3. **标准化（归一化）**  
   将每个特征减去均值、除以标准差，得到标准化后的特征：  
   $$
   \hat{x}_{i, j} = \frac{x_{i, j} - \mu_i}{\sqrt{\sigma_i^2}}
   $$  


4. **缩放与偏移（可学习参数）**  
   为了保留数据的表达能力，引入可学习参数$\gamma$（缩放）和$\beta$（偏移），对标准化结果进行调整：  
   $$
   Y_{i, j} = \gamma \cdot \hat{x}_{i, j} + \beta
   $$  
   其中$y_{i, j}$是 Layer Norm 的输出。

### **三、与其他归一化方法的对比**
不同归一化方法的核心差异在于**归一化的维度**，以下是常见对比：

| 方法         | 归一化维度                          | 适用场景                     | 缺点                          |
|--------------|-------------------------------------|------------------------------|-------------------------------|
| Batch Norm   | 批次维度（对整个 batch 的同一特征）   | 计算机视觉（图像）、大 batch  | 依赖 batch size，小 batch 效果差 |
| Layer Norm   | 特征维度（对单个样本的所有特征）    | NLP（文本）、小 batch、RNN    | 对特征维度分布敏感            |
| Instance Norm | 单个样本的通道内（如单张图像的通道） | 风格迁移、图像生成           | 忽略样本间关联                |
| Group Norm   | 将特征分组，组内归一化              | 小 batch 场景（如医学影像）    | 分组方式需人工设计            |


### **四、Layer Norm 的优点**


1. **不依赖 batch size**  
   归一化仅基于单个样本的特征，不受批次大小影响，适合小批量训练（如 NLP 中 batch size 较小时）或在线学习。

2. **适合序列数据**  
   在循环神经网络（RNN）或 Transformer 等处理序列的模型中，序列长度可能动态变化（如文本 padding），Layer Norm 可稳定处理每个时间步的特征。

3. **训练与推理一致**  
   Batch Norm 在推理时需使用训练阶段的移动均值/方差，而 Layer Norm 在训练和推理时计算逻辑完全一致，无需额外存储参数。

4. **缓解梯度问题**  
   标准化后的输入分布更稳定，有助于缓解梯度消失/爆炸，加速模型收敛。


### **五、典型应用场景**

1. **自然语言处理（NLP）**  
   - Transformer 模型（如 BERT、GPT）的核心组件“Add & Norm”中，Layer Norm 是标配。例如，每个多头注意力层或前馈网络层后，都会通过 Layer Norm+残差连接稳定训练。  
   - 原因：文本样本的特征分布差异大（如不同句子的语义差异），Batch Norm 在小 batch 下效果差，而 Layer Norm 更鲁棒。

2. **循环神经网络（RNN/LSTM/GRU）**  
   序列数据的每个时间步输入分布可能波动，Layer Norm 可在每个时间步独立归一化，提升模型稳定性。

3. **小批量或动态 batch 场景**  
   如医学影像（样本量少）、在线推理（实时生成 batch）等，Layer Norm 可避免 Batch Norm 的性能下降。

### **六、注意事项**

- **特征维度需足够大**：若特征维度$D$过小（如小于 10），均值和方差的估计可能不稳定，影响归一化效果。  
- **可学习参数的作用**：$\gamma$和$\beta$允许模型“还原”部分原始分布，避免过度归一化导致的信息丢失。  
- **与残差连接配合**：在 Transformer 等模型中，Layer Norm 常与残差连接（$x + \text{SubLayer}(x)$）结合，进一步增强训练稳定性。

### **七、代码实现**

```python
class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
    super().__init__()
    # 线性矩阵做映射
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
    
    def forward(self, x):
    # 在统计每个样本所有维度的值，求均值和方差
    mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
    std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
    # 注意这里也在最后一个维度发生了广播
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

## 残差

由于 Transformer 模型结构较复杂、层数较深，​为了避免模型退化，Transformer 采用了残差连接的思想来连接每一个子层。残差连接，即下一层的输入不仅是上一层的输出，还包括上一层的输入。残差连接允许最底层信息直接传到最高层，让高层专注于残差的学习。

```python
# 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络
out = h + self.feed_forward.forward(self.fnn_norm(h))
```