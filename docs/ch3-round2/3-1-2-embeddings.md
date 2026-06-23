# 1.2 嵌入表示

> **难度** ⭐⭐⭐☆☆ · **前置**：[4.2 NumPy 与张量](../ch1-intro/1-4-2-numpy-tensors.md)、[3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md)

!!! abstract "读完这一节，你会"
    - 说清"嵌入（embedding）"到底解决了什么问题，以及它为什么是稠密向量而不是稀疏编码
    - 看懂独热编码（one-hot）与嵌入的本质区别，并会算出二者在内存和语义上的差距
    - 用 NumPy 从零实现一次"查表"，再用 PyTorch 的 `nn.Embedding` 把它写成一行可训练的代码
    - 明白这张"嵌入表"为什么能在训练中自己学出语义，从而为后面的词嵌入、CLIP 打好底子

## 先想清楚：模型其实不认识"词"

在进入正题之前，我们得先承认一个有点扫兴的事实：**神经网络只会算数，它根本不认识文字。** 你给它一个单词 "cat"，它无从下手——它能乘的是矩阵，能加的是向量，唯独不能直接对一串字母做线性代数。

所以摆在我们面前的第一个问题永远是：**怎么把"猫""狗""国王"这样的离散符号，变成模型能吃的数字向量？** 这件事看起来不起眼，却是整个自然语言处理（NLP）乃至多模态模型的地基。地基没打好，后面盖什么楼都会塌。

最朴素的想法你可能马上能想到：给每个词编个号。假设词表里一共有一万个词，那 "cat" 是第 3 号，"dog" 是第 7 号，就用数字 3、7 来代表它们。听起来很自然，但这里藏着一个致命的坑——**编号是有大小关系的**。模型会"以为" 7 比 3 大、比 3 重要，甚至以为 "dog"（7）约等于两个 "cat"（3）。可现实里这两个词之间根本没有这种数量关系。直接用编号，等于硬塞给模型一堆它会误读的假信息。

## 第一版补救：独热编码，以及它的两个大麻烦

为了消除"编号有大小"这个误导，人们想出了一个干净利落的办法——**独热编码（one-hot encoding）**。

它的思路是：既然一万个词，那就给每个词分配一个长度为一万的向量，这个向量里**只有它自己那一位是 1，其余全是 0**。比如 "cat" 是第 3 号，它的向量就是 $(0,0,1,0,\dots,0)$，只有第 3 位亮着。这样一来，任意两个不同的词，它们的向量都是相互垂直的，谁也不比谁"大"，编号的误导一下子就被铲除了。

这一步很重要，但独热编码立刻暴露出两个让人头疼的麻烦。

**第一个麻烦是太占地方。** 词表有多大，向量就有多长。一万个词，每个词就是一个一万维的向量；真实的语言模型词表动辄五万、十万，那每个词都成了十万维的"巨型稀疏向量"——其中只有一个 1，剩下九万九千九百九十九个全是 0。这种"绝大多数位置都是 0"的表示，我们叫它**稀疏表示（sparse representation）**。它把内存浪费得一塌糊涂。

**第二个麻烦更要命：它丢掉了所有语义。** 在独热编码的世界里，"cat" 和 "dog" 的距离，跟 "cat" 和 "数据库" 的距离一模一样——因为任意两个独热向量都互相垂直，点积都是 0。可我们明明知道，猫和狗的关系，远比猫和数据库亲近。**独热编码把每个词都关进了一个孤岛，词与词之间再没有任何"远近亲疏"。** 这对想理解语言的模型来说，是无法接受的。

## 真正的主角：用一张可学习的"查找表"

既然稀疏又长又没语义，那我们干脆反过来要求：能不能用一个**又短、又稠密、还能体现语义**的向量来代表每个词？这正是**嵌入（embedding）**登场的地方。

所谓嵌入，说白了就是：**给每个词分配一个不长的实数向量**（比如 64 维或 300 维），让意思相近的词，向量也靠得近。这种"绝大多数位置都是有意义的非零数"的表示，与稀疏表示相对，叫**稠密表示（dense representation）**。

那这些向量从哪来？关键的一招是——**别去手工设计它们，把它们当成模型的参数，让模型在训练里自己学。** 我们准备一张大表：表有 $V$ 行（$V$ 是词表大小），每行是一个 $d$ 维向量（$d$ 是我们选定的嵌入维度，比如 64）。这张 $V\times d$ 的表，就叫**嵌入矩阵（embedding matrix）**，记作 $E$。

用的时候极其简单：想要第 $i$ 个词的向量，**就去表里把第 $i$ 行取出来**。是的，本质上就是"查表"（lookup），没有任何复杂运算。下面这张图把这件事画了出来：

<div class="diagram">
<svg viewBox="0 0 420 230" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ea" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 左侧：输入索引 -->
  <text x="40" y="28" font-size="12" fill="var(--dia-stroke-soft)">词的编号</text>
  <rect x="36" y="98" width="56" height="30" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5" rx="3"/>
  <text x="64" y="118" text-anchor="middle" font-size="14" fill="var(--dia-accent-deep)">i = 2</text>
  <!-- 中间：嵌入矩阵 E（V 行 d 列） -->
  <text x="225" y="28" font-size="12" fill="var(--dia-stroke-soft)">嵌入矩阵 E（V 行 × d 列）</text>
  <rect x="160" y="40" width="130" height="150" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <!-- 行分隔 -->
  <line x1="160" y1="78" x2="290" y2="78" stroke="var(--dia-rule)" stroke-width="1"/>
  <line x1="160" y1="116" x2="290" y2="116" stroke="var(--dia-rule)" stroke-width="1"/>
  <line x1="160" y1="152" x2="290" y2="152" stroke="var(--dia-rule)" stroke-width="1"/>
  <text x="166" y="64" font-size="11" fill="var(--dia-stroke-tertiary)">第 0 行</text>
  <text x="166" y="102" font-size="11" fill="var(--dia-stroke-tertiary)">第 1 行</text>
  <!-- 高亮第 2 行 -->
  <rect x="160" y="116" width="130" height="36" fill="var(--dia-accent-soft)"/>
  <text x="166" y="139" font-size="11" fill="var(--dia-accent-deep)">第 2 行</text>
  <text x="240" y="139" font-size="11" fill="var(--dia-accent-deep)">取它</text>
  <text x="166" y="176" font-size="11" fill="var(--dia-stroke-tertiary)">…</text>
  <!-- 箭头：索引 -> 矩阵第2行 -->
  <line x1="94" y1="120" x2="156" y2="130" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#ea)"/>
  <!-- 右侧：取出的稠密向量 -->
  <text x="360" y="28" font-size="12" fill="var(--dia-stroke-soft)">稠密向量</text>
  <rect x="330" y="116" width="78" height="36" fill="var(--dia-green-soft)" stroke="var(--dia-green)" stroke-width="1.5" rx="3"/>
  <text x="369" y="139" text-anchor="middle" font-size="11" fill="var(--dia-stroke)">[0.2, -1.1, …]</text>
  <line x1="290" y1="134" x2="326" y2="134" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#ea)"/>
</svg>
</div>
<p class="figure-caption">图 1.2-1：嵌入层就是一张可学习的查找表——拿词的编号当行号，去表里取出对应的那一行稠密向量。</p>

这里有个值得停下来体会的小细节：**"查表"其实和"独热向量乘以嵌入矩阵"是同一件事。** 你回忆一下，独热向量 $\mathbf{o}_i$ 只有第 $i$ 位是 1，那么 $\mathbf{o}_i^\top E$ 算出来恰好就是 $E$ 的第 $i$ 行。换句话说：

$$\mathbf{o}_i^\top E = E_{i,:}.$$

所以嵌入层在数学上完全可以看成"一次特殊的矩阵乘法"，只不过因为独热向量里全是 0，真去做这个乘法纯属浪费——直接按行号取出来又快又省。这也解释了为什么深度学习框架要专门提供一个嵌入层，而不是让你老老实实做矩阵乘。

## 它凭什么能学出"语义"

到这里你可能会犯嘀咕：这张表一开始是随机初始化的，里面的数字毫无意义，凭什么训练完之后"猫"和"狗"就靠近了？

答案藏在"它是参数"这件事里。既然嵌入矩阵 $E$ 是模型的参数，那它就会跟着损失函数一起被梯度下降优化（如果你忘了梯度下降在干嘛，回头看一眼 [3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md)）。训练时，模型为了把任务做好——比如预测下一个词、或者判断一句话的情感——会不断微调每一行向量。**那些总在相似上下文里出现的词，它们的向量会被一点点推到相近的位置**，因为这样最有利于降低损失。

这背后是语言学里一个很有名的洞见，叫**分布式语义假设（distributional semantics）**，一句话概括就是："一个词的意思，由它周围常出现的词决定。"——你看一个词总和"喵""毛""宠物"作伴，又总和"汪""遛"作伴，模型自然就把"猫"和"狗"放到了语义空间里相邻的区域。

学完之后，这个空间会展现出惊人的结构。最经典的例子是：$\text{vec}(\text{国王}) - \text{vec}(\text{男}) + \text{vec}(\text{女}) \approx \text{vec}(\text{王后})$。向量的加减，居然对应上了语义的类比。下面这张图给你一个直观感受：

<div class="diagram">
<svg viewBox="0 0 380 230" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 坐标 -->
  <line x1="40" y1="200" x2="350" y2="200" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <line x1="40" y1="200" x2="40" y2="24" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <!-- 四个点：国王、王后、男、女，形成平行四边形 -->
  <circle cx="120" cy="70" r="5" fill="var(--dia-accent)"/>
  <text x="92" y="60" font-size="12" fill="var(--dia-accent-deep)">国王</text>
  <circle cx="250" cy="70" r="5" fill="var(--dia-accent)"/>
  <text x="258" y="60" font-size="12" fill="var(--dia-accent-deep)">王后</text>
  <circle cx="120" cy="160" r="5" fill="var(--dia-blue)"/>
  <text x="96" y="180" font-size="12" fill="var(--dia-blue)">男</text>
  <circle cx="250" cy="160" r="5" fill="var(--dia-blue)"/>
  <text x="256" y="180" font-size="12" fill="var(--dia-blue)">女</text>
  <!-- 两条平行的"性别"向量（虚线） -->
  <line x1="120" y1="160" x2="120" y2="78" stroke="var(--dia-green)" stroke-width="1.5" stroke-dasharray="4 3"/>
  <line x1="250" y1="160" x2="250" y2="78" stroke="var(--dia-green)" stroke-width="1.5" stroke-dasharray="4 3"/>
  <!-- 顶部"王室"向量 -->
  <line x1="126" y1="70" x2="244" y2="70" stroke="var(--dia-stroke-soft)" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="300" y="120" font-size="11" fill="var(--dia-green)">同一段位移</text>
  <text x="300" y="136" font-size="11" fill="var(--dia-green)">≈ 性别</text>
</svg>
</div>
<p class="figure-caption">图 1.2-2：在学好的嵌入空间里，"国王→王后"和"男→女"是几乎相同的一段位移——语义关系变成了向量的几何关系。</p>

正是这种结构，让嵌入向量远比独热编码"聪明"：它不只是给每个词一个身份证号，而是把整个词表组织进了一个有意义的几何空间。

## 动手实现：从一行 NumPy 到一层 PyTorch

道理讲透了，我们落到代码上。嵌入的核心动作就是"按行号取行"，所以从零实现简单到出乎你意料。

我们先用 NumPy 造一张随机的嵌入表，再用"花式索引"一次取出好几个词的向量：

```python
import numpy as np

V, d = 10, 4                       # 词表 10 个词，每个词用 4 维向量表示
np.random.seed(0)
E = np.random.randn(V, d)          # 嵌入矩阵：10 行 4 列，先随机初始化

ids = np.array([2, 7, 2])          # 想取第 2、7、2 号词（注意第 2 号出现两次）
vectors = E[ids]                   # 花式索引，一步取出对应的行
print(vectors.shape)               # (3, 4)：3 个词，每个 4 维
print(np.allclose(vectors[0], vectors[2]))   # True：同一个编号取出的行完全相同
```

最后一行的 `True` 值得你注意：**只要是同一个词，无论它在句子里出现几次，查出来的都是表里同一行。** 这说明"语义"是存在表里的，而不是临时算出来的。

但 NumPy 版本有个根本缺陷——它不会自己学。我们手动随机生成的 `E` 永远是随机的，因为 NumPy 不帮我们算梯度。要让这张表在训练中被优化，就得请出 PyTorch 的 `nn.Embedding`。它本质上就是上面那张表，只不过被注册成了**可训练参数**，能自动接入反向传播：

```python
import torch
import torch.nn as nn

emb = nn.Embedding(num_embeddings=10, embedding_dim=4)   # 同样是 10×4 的查找表

ids = torch.tensor([2, 7, 2])      # 输入是整数索引，不是独热向量！
vectors = emb(ids)                 # 前向传播 = 查表
print(vectors.shape)               # torch.Size([3, 4])
print(emb.weight.shape)            # torch.Size([10, 4])：这就是那张可学习的表
```

请特别留意输入：**`nn.Embedding` 吃的是整数编号，而不是独热向量。** 这正是它高效的原因——它在内部直接按行号取，跳过了那个"乘以一堆 0"的浪费。

它和 NumPy 版唯一的、也是最关键的区别在于：`emb.weight` 带着 `requires_grad=True`。这意味着当你把它接进一个网络、算出损失、反向传播时，**梯度会顺着流回这张表，把每一行向量往"更有利于任务"的方向挪一点点**。训练几个回合下来，原本随机的表，就长出了语义。我们用一个最小的例子把这条链路跑通：

```python
ids = torch.tensor([2, 7])
out = emb(ids).sum()               # 随便定义一个标量损失，仅作演示
out.backward()                     # 反向传播

# 只有被取过的第 2、7 行才有非零梯度，其余行的梯度是 0
print(emb.weight.grad[2])          # 一个非零的 4 维向量
print(emb.weight.grad[0])          # 全是 0：第 0 号词这次没被用到
```

输出印证了一件很符合直觉的事：**这一步里没被用到的词，它的那一行不会被更新。** 嵌入表就是这样，一个词一个词地、随着它在数据里出现，被慢慢雕琢成形的。

## 维度 d 该选多大

你可能会问：嵌入维度 $d$ 到底取多少合适？这没有标准答案，但有一条朴素的权衡。

$d$ 太小，向量"装不下"足够的语义，不同的词容易被挤到一起，区分不开；$d$ 太大，参数变多、容易过拟合，还更费显存和算力。实践里，小任务常用 $50$ 到 $300$，大型语言模型的词嵌入则可能上千。**一个好用的起手式是：先从一两百维试起，再根据验证集表现往上或往下调。** 别一开始就纠结这个数字，它远没有"数据够不够、上下文建模得好不好"重要。

## 容易踩的坑

- **索引越界**：`nn.Embedding(num_embeddings=10, ...)` 只认 0 到 9 的编号，喂进去一个 10 就会直接报 `index out of range`。词表大小一定要覆盖所有可能出现的 token id。
- **把独热向量喂进去**：新手常误以为要先做独热再送入嵌入层。不需要——`nn.Embedding` 的输入就是整数索引，多此一举不仅慢，还会因为形状不对而报错。
- **混淆 `*` 与查表**：嵌入不是逐元素相乘，也不是普通的全连接层；它是"按行号取行"。理解了 $\mathbf{o}_i^\top E = E_{i,:}$ 这个等价关系，就不会搞混。
- **忘了它是参数、会被更新**：如果你用了别人预训练好的词向量又不想动它，记得设 `emb.weight.requires_grad = False`，否则训练会悄悄把它改掉。
- **padding 没处理**：变长句子常用 0 来补齐（padding）。可以给 `nn.Embedding(..., padding_idx=0)`，让第 0 行永远是 0 且不参与梯度更新，避免"空白"也被学出语义。

## 它在后面会怎么用到

嵌入是 Round 2 里一块承上启下的地基，后面好几节都建在它之上：

- **词嵌入与文本分类**会把这张表正式用到语言上，结合 Word2Vec、GloVe 等经典方法，把整段文本变成向量再做分类（见 [4.1 文本分类与词嵌入](3-4-1-text-embeddings.md)）。
- **注意力机制与 Transformer** 的输入，第一步就是把每个 token 过一遍嵌入层，再加上位置编码——没有嵌入，Transformer 连"看"都看不懂一个词（见 [2.1 注意力机制](3-2-1-attention.md)、[2.2 Transformer 架构](3-2-2-transformer.md)）。
- **CLIP** 更进一步，把图像和文本**嵌入到同一个空间**里，让"一张猫的照片"和"cat 这个词"靠得很近——它玩的就是"共享嵌入空间"这个思想（见 [3.4 视觉-文本编码器 CLIP](3-3-4-clip.md)）。
- 往回看，这一节的"查表 = 独热乘矩阵"全靠你在 [3.1 线性代数](../ch1-intro/1-3-1-linear-algebra.md) 里打下的矩阵乘法直觉。

## 练习

??? note "基础练习"
    1. 用 NumPy 建一个 $8\times3$ 的随机嵌入矩阵，写一个函数 `lookup(E, ids)`，对一批编号返回它们的向量。再验证：把编号 `[1, 1]` 取出来的两行是否完全相等。
    2. 用 `nn.Embedding(num_embeddings=8, embedding_dim=3)` 复现上一题，打印 `emb.weight.shape`，并说说它和你手写的矩阵 `E` 在"会不会被训练更新"上有什么本质区别。
    3. 手动构造词 "cat"（编号 2）的独热向量，让它乘以嵌入矩阵 $E$，验证结果确实等于 `E[2]`。亲手感受一下"查表 = 独热乘矩阵"。

??? note "进阶练习"
    1. 给嵌入层接一个最简单的下游任务：随机造一些 `(编号, 标签)` 数据，让 `nn.Embedding` 后接一个线性层做二分类，训练几十步。观察 `emb.weight` 在训练前后是否真的变了。
    2. 训练完后，用余弦相似度找出和某个词向量"最近"的几个词。想一想：随机初始化、没怎么训练的表，找出来的近邻有意义吗？为什么？
    3. 设 `padding_idx=0` 重做练习 1 的第 2 题，反向传播一次后打印 `emb.weight.grad[0]`，确认第 0 行的梯度始终为 0，理解 padding 是怎么被"保护"起来的。

## 小结

- 模型只会算数，所以第一步永远是把离散的词变成向量；**独热编码**能去掉编号的误导，但它又长（稀疏）又没语义。
- **嵌入**用一张可学习的查找表，给每个词一个又短又稠密的向量；"查表"在数学上等价于"独热向量乘嵌入矩阵"。
- 因为嵌入矩阵是**参数**，它会跟着梯度下降被优化，从而依据分布式语义自己学出"意思相近的词，向量也相近"。
- 在 PyTorch 里，`nn.Embedding` 就是这张可训练的表，输入整数编号、输出稠密向量，是 NLP 与多模态模型雷打不动的第一层。

想把直觉建得更牢，强烈推荐 Jay Alammar 的图解博客 [*The Illustrated Word2Vec*](https://jalammar.github.io/illustrated-word2vec/)，它用大量动图把"嵌入空间长什么样"讲得清清楚楚；官方接口细节可查 [PyTorch `nn.Embedding` 文档](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)。
