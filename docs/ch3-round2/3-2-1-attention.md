# 2.1 注意力机制

> **难度** ⭐⭐⭐⭐☆ · **前置**：[2.2 梯度下降与反向传播](../ch2-round1/2-2-2-backprop.md)、[1.2 嵌入表示](3-1-2-embeddings.md)

!!! abstract "读完这一节，你会"
    - 用"按相关度去查一张软字典"的眼光，理解 Query、Key、Value 三者各自在做什么
    - 会手推一个 3 个词的小例子，亲手算出注意力权重和输出向量
    - 说清楚缩放点积里那个 $\sqrt{d_k}$ 是干嘛的，以及注意力凭什么能搞定长程依赖
    - 既能用 NumPy 从零写出注意力，也能用 PyTorch 的现成接口算同样的结果

## 先说说它到底想解决什么麻烦

我们在 [1.2 嵌入表示](3-1-2-embeddings.md) 里已经学会了把每个词变成一个向量。可一句话不是一堆孤立的词，词和词之间是有关系的。来看一句很经典的英文：*The animal didn't cross the street because **it** was too tired.* 这里的 *it* 到底指谁？是 *animal* 还是 *street*？人一眼就知道是 *animal*，因为"太累了"只能形容动物。但模型要想答对，就得让 *it* 这个词"回头看一眼"前面所有的词，并且**重点看 *animal***。

在注意力出现之前，主流办法是循环网络（RNN），它像传话游戏一样把信息一个词一个词往后传。问题是，传得越远，前面的信息就丢得越多——等传到 *it* 的时候，*animal* 早就被稀释得快没影了。这就是所谓的**长程依赖（long-range dependency）**难题：离得远的词之间，关系很难建立。

注意力机制换了个思路：**与其费劲地把信息一站站接力，不如让每个词都能一步直达地去看其他任何一个词**，并且自己决定"该多看谁、少看谁"。两个词隔得再远，中间也只有一步距离。这个看似简单的转变，正是后面 [2.2 Transformer 架构](3-2-2-transformer.md) 能横扫 NLP 乃至整个深度学习的根基。

## 一个核心比喻：去图书馆查一张"软字典"

注意力最难的不是公式，而是 Query、Key、Value 这三个名字一上来就把人绕晕。所以我们先不碰数学，用一个查字典的画面把它讲透。

想象你在图书馆找资料。你心里有一个**要查的问题**，这就是 **Query（查询）**，可以简称 Q。书架上每本书都贴着一个**标签**，告诉你它讲的是什么，这就是 **Key（键）**，简称 K。而书里真正的**内容**，就是 **Value（值）**，简称 V。

查资料的过程是这样的：你拿着自己的问题 Q，去和每一本书的标签 K 比一比，**问题和哪个标签越对得上，那本书就越相关**。但你不会只读最相关的那一本、把别的全扔掉——你会按相关程度，给每本书分配一个"该读多少"的权重，然后把所有书的内容 V **按这个权重混合起来**，得到最终答案。

这就是注意力和真实字典最大的不同：真字典是"精确命中"，查 *cat* 只返回 *cat* 那一条；而注意力是一张**软字典（soft dictionary）**——它对每一条都给一点权重，最后返回的是一个加权平均的结果。"软"在这里就是"不二选一、按比例混合"的意思。

回到那句话：*it* 的 Query 去和句中每个词的 Key 做匹配，发现自己和 *animal* 的 Key 最对味，于是给 *animal* 的 Value 分了最大的权重，*it* 的新表示里就主要装进了 *animal* 的信息。它"看见"了正确的词。

## 把比喻翻译成公式：缩放点积注意力

比喻清楚了，我们把它一步步变成数学。注意力里最常用的版本叫**缩放点积注意力（scaled dot-product attention）**，我们顺着"打分 → 归一化 → 加权"这三步来推。

**第一步：用点积给相关度打分。** 怎么衡量一个 Query 和一个 Key "对不对得上"？最简单的办法就是把这两个向量做**点积**。点积越大，说明两个向量方向越接近、越相关。设 Query 矩阵是 $Q$、Key 矩阵是 $K$，所有 Query 对所有 Key 的打分一次就能算完：

$$S = QK^\top.$$

$S_{ij}$ 就是第 $i$ 个 Query 和第 $j$ 个 Key 的相关度分数。

**第二步：缩放一下，别让分数失控。** 这里有个细节很关键。当 Key 的维度 $d_k$ 比较大时，点积是 $d_k$ 个数相加，结果的数值会随维度变得很大。分数一大，下一步的 softmax 就会被推到非常"尖"的区域——几乎把全部权重压给一个词，其余全是接近 0 的梯度，训练就走不动了。解决办法很朴素：**除以 $\sqrt{d_k}$ 把方差拉回来**：

$$S' = \frac{QK^\top}{\sqrt{d_k}}.$$

这就是名字里"缩放（scaled）"二字的来历。

**第三步：用 softmax 变成权重，再加权 Value。** 现在每行是一组分数，我们希望它们变成一组"和为 1 的权重"，这正是 **softmax** 干的活——它把任意一组实数压成一组正的、加起来等于 1 的比例。对 $S'$ 的每一行做 softmax 得到权重矩阵 $A$，再用它去加权 Value 矩阵 $V$，就得到了输出：

$$\text{Attention}(Q,K,V)=\underbrace{\text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)}_{A:\ \text{注意力权重}}\,V.$$

整条公式从右往左读就是那句比喻：拿 Q 比 K 打分（$QK^\top$）、缩放、softmax 成权重、再去混合 V。下面这张图把这三步串了起来：

<div class="diagram">
<svg viewBox="0 0 420 210" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="aa" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <!-- Q K V 三个输入框 -->
  <rect x="20" y="24" width="58" height="30" rx="4" fill="none" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="49" y="44" text-anchor="middle" font-size="14" fill="var(--dia-accent)">Q</text>
  <rect x="20" y="90" width="58" height="30" rx="4" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <text x="49" y="110" text-anchor="middle" font-size="14" fill="var(--dia-blue)">K</text>
  <rect x="20" y="156" width="58" height="30" rx="4" fill="none" stroke="var(--dia-green)" stroke-width="2"/>
  <text x="49" y="176" text-anchor="middle" font-size="14" fill="var(--dia-green)">V</text>
  <!-- 打分框 -->
  <rect x="140" y="57" width="92" height="30" rx="4" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="186" y="77" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">QKᵀ/√dₖ</text>
  <!-- softmax 框 -->
  <rect x="270" y="57" width="78" height="30" rx="4" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="309" y="77" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">softmax</text>
  <!-- 加权输出框 -->
  <rect x="270" y="140" width="78" height="30" rx="4" fill="none" stroke="var(--dia-accent-deep)" stroke-width="2"/>
  <text x="309" y="160" text-anchor="middle" font-size="12" fill="var(--dia-accent-deep)">输出</text>
  <!-- 连线 Q,K -> 打分 -->
  <path d="M78 39 L130 66" stroke="var(--dia-stroke-soft)" stroke-width="1.3" marker-end="url(#aa)"/>
  <path d="M78 105 L130 78" stroke="var(--dia-stroke-soft)" stroke-width="1.3" marker-end="url(#aa)"/>
  <!-- 打分 -> softmax -->
  <path d="M232 72 L262 72" stroke="var(--dia-stroke-soft)" stroke-width="1.3" marker-end="url(#aa)"/>
  <!-- softmax(权重A) -> 输出 -->
  <path d="M309 87 L309 132" stroke="var(--dia-stroke-soft)" stroke-width="1.3" marker-end="url(#aa)"/>
  <text x="318" y="112" font-size="11" fill="var(--dia-stroke-tertiary)">权重 A</text>
  <!-- V -> 输出 -->
  <path d="M78 171 L262 156" stroke="var(--dia-green)" stroke-width="1.3" marker-end="url(#aa)"/>
</svg>
</div>
<p class="figure-caption">图 2.1-1：缩放点积注意力的数据流——Q 与 K 打分并缩放，softmax 成权重 A，再去加权混合 V 得到输出。</p>

顺带说一句，当 Q、K、V 都来自**同一句话自己**（每个词既当查询者，又当被查的标签和内容）时，这种特殊情形就叫**自注意力（self-attention）**，它正是 Transformer 的主力。

## 手把手算一个 3 个词的小例子

公式看十遍，不如自己算一遍。我们把维度压到最小，用 3 个词、每个向量 2 维，手推一次。

假设句子里有 3 个词，为简单起见，我们让它们的 Query、Key、Value 都直接取下面这组向量（真实模型里 Q、K、V 是用三个权重矩阵从词向量乘出来的，这里跳过那步，专注算注意力本身）：

$$
Q=K=V=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.
$$

矩阵的每一行代表一个词。现在我们站在**第 1 个词**的角度，算它的注意力输出。

**第一步，打分。** 第 1 个词的 Query 是 $q_1=(1,0)$，分别和三个 Key 做点积：

$$
q_1\cdot k_1 = 1,\quad q_1\cdot k_2 = 0,\quad q_1\cdot k_3 = 1.
$$

所以第 1 行的原始分数是 $(1,\,0,\,1)$。第 1 个词和第 1、第 3 个词相关，和第 2 个词无关——这和向量摆放是吻合的。

**第二步，缩放。** 这里 $d_k=2$，所以每个分数除以 $\sqrt{2}\approx1.414$，得到 $(0.707,\,0,\,0.707)$。

**第三步，softmax 成权重。** softmax 要先对每个数取指数，再除以总和。三个指数分别是 $e^{0.707}\approx2.028$、$e^{0}=1$、$e^{0.707}\approx2.028$，总和约 $5.056$，于是权重是

$$
\Big(\tfrac{2.028}{5.056},\ \tfrac{1}{5.056},\ \tfrac{2.028}{5.056}\Big)\approx(0.401,\ 0.198,\ 0.401).
$$

三个数加起来正好是 1，且第 1、3 个词权重大、第 2 个词权重小——完全符合预期。

**第四步，加权 Value。** 用这组权重去混合三个 Value（也就是那三行）：

$$
\text{out}_1 = 0.401\!\begin{bmatrix}1\\0\end{bmatrix}+0.198\!\begin{bmatrix}0\\1\end{bmatrix}+0.401\!\begin{bmatrix}1\\1\end{bmatrix}=\begin{bmatrix}0.802\\0.599\end{bmatrix}.
$$

这就是第 1 个词经过注意力后的新表示。你看，它已经不再是原来的 $(1,0)$，而是**融进了它所关注的那些词的信息**。把同样的流程对第 2、第 3 个词各走一遍，就得到整句话的输出。这正是自注意力的全部秘密。

## 从零写一遍，再用 PyTorch 对一遍答案

道理和手算都通了，我们用代码验证。先看**从零实现**——把上面四步原样翻译成 NumPy，你会发现它和公式几乎一一对应：

```python
import numpy as np

def softmax(x, axis=-1):
    # 先减去每行最大值再取指数，防止数值溢出（这个小技巧很重要）
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)   # 第一、二步：打分并缩放
    weights = softmax(scores, axis=-1)  # 第三步：softmax 成权重
    return weights @ V, weights         # 第四步：加权混合 V

# 就用手算例子里那组向量
M = np.array([[1., 0.],
              [0., 1.],
              [1., 1.]])
out, w = attention(M, M, M)
print("注意力权重：\n", w.round(3))
print("输出：\n", out.round(3))
```

跑出来，第一行的权重正是 `[0.401 0.198 0.401]`、第一行输出正是 `[0.802 0.599]`，和我们手推的分毫不差。能对上，说明你真的理解了每一步在干什么。

接着用**库版**——PyTorch 从 2.0 起内置了 `scaled_dot_product_attention`，工程里我们直接调它，因为它在底层做了显存和速度的优化：

```python
import torch
import torch.nn.functional as F

M = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])
# 接口要求带上 batch 维度，所以前面加一维
out = F.scaled_dot_product_attention(M[None], M[None], M[None])
print(out.squeeze(0).round(decimals=3))
```

输出和我们手写的 NumPy 版完全一致。这也说明一件让人安心的事：**注意力听着唬人，核心其实就是那四步**——现成的库只是把它写得更快、更省显存，骨架和你亲手推的一模一样。

## 为什么这套机制能搞定长程依赖

回到一开始那个老大难问题。RNN 处理一句话，信息要一站接一站地传，第 1 个词想影响第 20 个词，中间得经过 19 步——路径长，信息一路衰减。

注意力则完全不同。看回那条公式 $\text{softmax}(QK^\top/\sqrt{d_k})V$：第 20 个词的 Query 是**和包括第 1 个词在内的每一个 Key 直接做点积**的。换句话说，**任意两个词之间的"距离"都只有一步**，不管它们在句子里隔多远。下面这张图把两种连接方式摆在一起，对比一目了然：

<div class="diagram">
<svg viewBox="0 0 420 220" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ra" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0 0 L5 3 L0 6 Z" fill="var(--dia-blue)"/></marker>
  </defs>
  <!-- 上半：RNN 链式 -->
  <text x="20" y="28" font-size="12" fill="var(--dia-stroke-soft)">RNN：信息一站站接力，远了就衰减</text>
  <circle cx="60" cy="60" r="13" fill="none" stroke="var(--dia-blue)" stroke-width="1.6"/>
  <circle cx="150" cy="60" r="13" fill="none" stroke="var(--dia-blue)" stroke-width="1.6"/>
  <circle cx="240" cy="60" r="13" fill="none" stroke="var(--dia-blue)" stroke-width="1.6"/>
  <circle cx="330" cy="60" r="13" fill="none" stroke="var(--dia-blue)" stroke-width="1.6"/>
  <path d="M73 60 L137 60" stroke="var(--dia-blue)" stroke-width="1.4" marker-end="url(#ra)"/>
  <path d="M163 60 L227 60" stroke="var(--dia-blue)" stroke-width="1.4" marker-end="url(#ra)"/>
  <path d="M253 60 L317 60" stroke="var(--dia-blue)" stroke-width="1.4" marker-end="url(#ra)"/>
  <!-- 下半：注意力全连接 -->
  <text x="20" y="138" font-size="12" fill="var(--dia-stroke-soft)">注意力：每个词一步直达任意词</text>
  <circle cx="60" cy="178" r="13" fill="none" stroke="var(--dia-accent)" stroke-width="1.6"/>
  <circle cx="150" cy="178" r="13" fill="none" stroke="var(--dia-accent)" stroke-width="1.6"/>
  <circle cx="240" cy="178" r="13" fill="none" stroke="var(--dia-accent)" stroke-width="1.6"/>
  <circle cx="330" cy="178" r="13" fill="none" stroke="var(--dia-accent)" stroke-width="1.6"/>
  <!-- 全连接弧线 -->
  <path d="M73 172 Q105 140 137 172" fill="none" stroke="var(--dia-accent)" stroke-width="1.1"/>
  <path d="M73 175 Q150 120 227 175" fill="none" stroke="var(--dia-accent)" stroke-width="1.1"/>
  <path d="M73 178 Q195 108 317 178" fill="none" stroke="var(--dia-accent)" stroke-width="1.1"/>
  <path d="M163 172 Q195 140 227 172" fill="none" stroke="var(--dia-accent)" stroke-width="1.1"/>
  <path d="M163 175 Q240 120 317 175" fill="none" stroke="var(--dia-accent)" stroke-width="1.1"/>
  <path d="M253 172 Q285 140 317 172" fill="none" stroke="var(--dia-accent)" stroke-width="1.1"/>
</svg>
</div>
<p class="figure-caption">图 2.1-2：RNN 靠链式接力传信息，距离越远衰减越重；注意力让每个词都能一步直连任意词，长程依赖迎刃而解。</p>

还有一个常被忽略却同样重要的好处：RNN 必须**按顺序**一个词一个词算，没法并行；而注意力的 $QK^\top$ 是一次矩阵乘法，**整句话所有词同时算完**，特别适合用 GPU 加速（见 [1.1 GPU 训练](3-1-1-gpu-training.md)）。又快、又能抓远距离关系，这就是注意力胜出的两张王牌。

## 容易踩的坑

- **softmax 不减最大值就溢出**：指数函数涨得飞快，分数稍大 `np.exp` 就溢出成 `inf`。一定要先减去每行的最大值再取指数，结果不变却稳得多——这一点在 [3.2 微积分与梯度](../ch1-intro/1-3-2-calculus-gradients.md) 配套的 softmax 推导里也强调过。
- **忘了除以 $\sqrt{d_k}$**：维度一大，点积数值飙升，softmax 被推成接近 one-hot，梯度几乎为零，训练原地不动。缩放这一步不是可有可无的装饰。
- **softmax 的轴搞错**：要对"每个 Query 看所有 Key"这一行做归一化，所以是 `axis=-1`（最后一维）。写成对列归一化，权重含义就全乱了。
- **矩阵形状对不上**：$Q$ 是 $(n,d_k)$、$K$ 是 $(m,d_k)$，$QK^\top$ 才合法、结果是 $(n,m)$。报 `shapes not aligned` 时，第一反应是 `print(Q.shape, K.shape)`。
- **把"软字典"当成精确查表**：注意力永远是加权混合，不会只挑出一个词。指望它像哈希表那样精确命中，理解就跑偏了。

## 它在后面会怎么用到

这一节是整个 Round 2 进阶部分的地基，往后到处都要用它：

- **[2.2 Transformer 架构](3-2-2-transformer.md)** 把自注意力叠成多头、再堆很多层，就长成了今天几乎所有大模型的骨架——这是注意力最直接、最重要的去处。
- **[4.3 Encoder-Decoder](3-4-3-encoder-decoder.md)** 里，解码器用注意力去"看"编码器的输出，机器翻译就是这么把源句对齐到译句的。
- **[3.4 视觉-文本编码器 CLIP](3-3-4-clip.md)** 和 **[4.5 音频模型](3-4-5-audio.md)** 同样建立在注意力之上，可见它早已不只属于 NLP。
- 真正训练这些大家伙时，离不开 GPU（见 [1.1 GPU 训练](3-1-1-gpu-training.md)），而注意力天然可并行，正好把 GPU 喂饱。

## 练习

??? note "基础练习"
    1. 接着正文的 3 词例子，把**第 2 个词**（Query 为 $(0,1)$）的注意力权重和输出向量手推一遍，再用文中的 `attention` 函数验证你的答案。
    2. 把缩放因子 $\sqrt{d_k}$ 去掉，重新算第 1 个词的权重，对比加权前后权重分布"尖"了多少。想一想：如果维度是 64 而不是 2，差别会更大还是更小？

??? note "进阶练习"
    1. 给注意力加上**因果掩码（causal mask）**：让每个词只能看见它自己和它前面的词（把还没出现的词的分数设成 $-\infty$ 再做 softmax）。这正是语言模型逐字生成时用的技巧，你会在 [4.2 语言建模](3-4-2-language-modeling.md) 再遇到它。
    2. 实现一个简化版的**多头注意力**：把输入向量切成 2 个"头"，每个头各自独立做一遍注意力，再把结果拼接起来。对比单头，观察不同的头是不是关注了不同的词。

## 小结

- 注意力的一句话本质：**每个词带着自己的 Query 去和所有词的 Key 打分，按分数（softmax 成的权重）混合所有词的 Value**——一张按相关度返回结果的软字典。
- 公式 $\text{softmax}(QK^\top/\sqrt{d_k})V$ 就是"打分 → 缩放 → softmax → 加权"四步，那个 $\sqrt{d_k}$ 是为了不让 softmax 失控。
- 它让任意两个词一步直达，既解决了长程依赖，又能在 GPU 上并行，这正是它取代 RNN 的根本原因。

想看更直观的讲解，强烈推荐 Jay Alammar 的图解博客 [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)，它用大量动图把 Q、K、V 讲得明明白白；想读原始出处，就翻那篇开创性的论文 [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)。
