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
- 查询矩阵 $Q$ 的维度为 $[B, N, d_k]$，其中 $B$ 是批量大小，$N$ 是序列长度，$d_k$ 是每个查询向量的维度
- 键矩阵 $K$ 的维度为 $[B, N, d_k]$，与查询矩阵维度一致
- 值矩阵 $V$ 的维度为 $[B, N, d_v]$，其中 $d_v$ 是每个值向量的维度（通常 $d_v = d_k$）
注意力机制的完整计算过程（含维度变化）如下：
1. 计算注意力分数：$$\text{scores} = \frac{Q \cdot K^T}{\sqrt{d_k}}$$
   其中 $K^T$ 是键矩阵的转置，维度为 $[B, d_k, N]$
   运算后 $\text{scores}$ 的维度为 $[B, N, N]$

2. 对注意力分数进行 Softmax 归一化：
   $$\text{p\_attn} = \text{Softmax}(\text{scores})$$
   归一化后 $\text{p\_attn}$ 的维度保持不变，仍为 $[B, N, N]$

3. （可选）应用 Dropout：
   $$\text{p\_attn} = \text{Dropout}(\text{p\_attn})$$
   维度仍为 $[B, N, N]$

4. 计算最终注意力输出：
   $$\text{output} = \text{p\_attn} \cdot V$$
   矩阵相乘后 $\text{output}$ 的维度为 $[B, N, d_v]$

完整的注意力机制公式（含维度信息）可表示为：
$$\text{Attention}(Q_{[B,N,d_k]}, K_{[B,N,d_k]}, V_{[B,N,d_v]}) = \text{Softmax}\left(\frac{Q_{[B,N,d_k]} \cdot K^T_{[B,d_k,N]}}{\sqrt{d_k}}\right)_{[B,N,N]} \cdot V_{[B,N,d_v]}$$

这个维度描述清晰地展示了注意力机制如何将查询、键、值矩阵通过矩阵运算转换为最终的注意力输出。

## 除以 $\sqrt{d_k}$ 的作用

在注意力机制中，对查询（Q）和键（K）的内积结果除以 $\sqrt{d_k}$（$d_k$ 为键/查询向量的维度），是为了**缓解内积结果过大导致的梯度问题和 Softmax 饱和问题**，具体原因可从以下两方面分析：
### 1. 避免内积结果随维度增长而过大
查询向量 $q$ 和键向量 $k$ 的内积计算公式为：  
$$q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i$$  

假设 $q$ 和 $k$ 的各分量是均值为 0、方差为 1 的独立随机变量，那么内积的期望和方差为：  
- 期望：$E[q \cdot k] = \sum_{i=1}^{d_k} E[q_i \cdot k_i] = 0$（因 $q_i$ 和 $k_i$ 独立，$E[q_i \cdot k_i] = E[q_i] \cdot E[k_i] = 0$）。  
- 方差：$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i \cdot k_i) = d_k$（因 $\text{Var}(q_i \cdot k_i) = \text{Var}(q_i) \cdot \text{Var}(k_i) = 1 \cdot 1 = 1$）。  

可见，内积结果的方差随 $d_k$ 线性增长，即**维度越高，内积结果的数值可能越大**（例如当 $d_k=512$ 时，内积结果可能轻松达到几十甚至上百）。


### 2. 防止 Softmax 函数进入饱和区
注意力分数经过内积计算后，会输入 Softmax 函数进行归一化：  
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$  

当 $x_i$ 数值过大时，$e^{x_i}$ 会因指数函数的特性快速趋近于无穷大，导致：  
- 数值计算上的溢出（如浮点数超出表示范围）；  
- Softmax 的输出会过于“陡峭”：即最大的 $x_i$ 对应的概率接近 1，其他概率接近 0，导致梯度消失（因为 Softmax 在饱和区的导数接近 0）。  

而除以 $\sqrt{d_k}$ 后，内积结果的方差被缩放为：  
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$$  

这样一来，内积结果被限制在一个合理的数值范围内（均值 0，方差 1），避免了 Softmax 进入饱和区，保证了梯度的稳定传播和注意力权重的有效学习。


### 总结
$\sqrt{d_k}$ 的作用是**对 Q 和 K 的内积结果进行“标准化”缩放**，通过控制内积的方差（使其不随维度增长），**避免 Softmax 函数因输入值过大而失效，最终保证注意力机制的稳定训练和有效计算**。

# 自注意力机制

就是 $Q, K, V$ 都相同

```python
# attention 为上文定义的注意力计算函数
attention(x, x, x)
```

# 掩码自注意力

