# 2.2 Transformer 架构

> **难度** ⭐⭐⭐⭐☆ · **前置**：[2.1 注意力机制](3-2-1-attention.md)、[2.4 PyTorch 基础](../ch2-round1/2-2-4-pytorch-basics.md)

!!! abstract "读完这一节，你会"
    - 说清一个 Transformer block 里那四件套——多头注意力、前馈网络、残差、LayerNorm——各自在干什么、为什么缺一不可
    - 理解位置编码的用处，并分得清正弦式和可学习式两种做法
    - 讲明白 Encoder 和 Decoder 的区别，以及"掩码注意力"为什么是 Decoder 的命根子
    - 用 PyTorch 亲手搭出一个能跑的 Transformer block，再用官方 `nn.TransformerEncoderLayer` 对照验证

## 从注意力，到一整座架构

上一节我们把**注意力（attention）**这个核心零件拆透了：给定一批向量，每个向量都能"环顾四周"，按相关程度从别的向量那里取信息。但光有这一个零件，还称不上是一个能用的模型——它就像一台只有发动机、没有底盘和变速箱的车。

2017 年那篇标题很狂的论文《Attention Is All You Need》，干的事情就是把注意力这台"发动机"装进一个完整的车架里，这个车架就叫 **Transformer**。它一出来就横扫了机器翻译，几年之内又长成了 BERT、GPT 这些你天天听说的大模型的骨架。所以这一节的目标很明确：**搞清楚那个车架到底由哪些部件拼成，以及为什么要这么拼。**

好消息是，Transformer 看着唬人，其实是高度重复的——它由一个叫 **block（块）** 的小单元一层层堆起来，堆 6 层、12 层、96 层，结构都一样。所以你只要彻底吃透**一个 block**，整座架构就拿下了大半。我们就从这个 block 画起。

<div class="diagram">
<svg viewBox="0 0 420 300" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ta" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <!-- 主干箭头（最左侧竖线，残差从这里分叉再汇合） -->
  <line x1="60" y1="280" x2="60" y2="20" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5" marker-end="url(#ta)"/>
  <text x="36" y="285" font-size="11" fill="var(--dia-stroke-soft)">输入 x</text>
  <!-- 多头注意力盒子 -->
  <rect x="120" y="210" width="200" height="48" rx="5" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <text x="220" y="239" text-anchor="middle" font-size="13" fill="var(--dia-accent-deep)">多头注意力 MHA</text>
  <!-- 加号1（残差汇合） -->
  <circle cx="60" cy="190" r="11" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="60" y="195" text-anchor="middle" font-size="14" fill="var(--dia-stroke-soft)">+</text>
  <!-- 进出注意力盒子的支线 -->
  <line x1="60" y1="234" x2="120" y2="234" stroke="var(--dia-stroke-soft)" stroke-width="1.2"/>
  <line x1="320" y1="234" x2="360" y2="234" stroke="var(--dia-stroke-soft)" stroke-width="1.2"/>
  <line x1="360" y1="234" x2="360" y2="190" stroke="var(--dia-stroke-soft)" stroke-width="1.2"/>
  <line x1="360" y1="190" x2="71" y2="190" stroke="var(--dia-stroke-soft)" stroke-width="1.2" marker-end="url(#ta)"/>
  <text x="335" y="172" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">残差直连</text>
  <!-- LayerNorm 标注1 -->
  <text x="92" y="160" font-size="11" fill="var(--dia-blue)">LayerNorm</text>
  <line x1="60" y1="178" x2="60" y2="148" stroke="var(--dia-blue)" stroke-width="2.5"/>
  <!-- 前馈网络盒子 -->
  <rect x="120" y="86" width="200" height="48" rx="5" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <text x="220" y="115" text-anchor="middle" font-size="13" fill="var(--dia-accent-deep)">前馈网络 FFN</text>
  <!-- 加号2 -->
  <circle cx="60" cy="66" r="11" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="60" y="71" text-anchor="middle" font-size="14" fill="var(--dia-stroke-soft)">+</text>
  <line x1="60" y1="110" x2="120" y2="110" stroke="var(--dia-stroke-soft)" stroke-width="1.2"/>
  <line x1="320" y1="110" x2="360" y2="110" stroke="var(--dia-stroke-soft)" stroke-width="1.2"/>
  <line x1="360" y1="110" x2="360" y2="66" stroke="var(--dia-stroke-soft)" stroke-width="1.2"/>
  <line x1="360" y1="66" x2="71" y2="66" stroke="var(--dia-stroke-soft)" stroke-width="1.2" marker-end="url(#ta)"/>
  <text x="335" y="48" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">残差直连</text>
  <text x="92" y="36" font-size="11" fill="var(--dia-blue)">LayerNorm</text>
  <line x1="60" y1="54" x2="60" y2="28" stroke="var(--dia-blue)" stroke-width="2.5"/>
</svg>
</div>
<p class="figure-caption">图 2.2-1：一个 Transformer block。数据从下往上走，先过多头注意力、再过前馈网络；每一块外面都包着一条"残差直连"和一道 LayerNorm。</p>

顺着这张图从下往上走一遍，这一节剩下的内容就是把图里每个部件讲清楚：先是多头注意力，再是前馈网络，最后是那两条不起眼却至关重要的残差和归一化。

## 多头注意力：与其看一遍，不如多看几遍

先说图里最下面那个最大的盒子——**多头注意力（multi-head attention，MHA）**。

上一节我们算的是**单头**注意力：把输入投影成查询 $Q$、键 $K$、值 $V$ 三组向量，再用一个公式算出每个位置该从别处取多少信息：

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

这套机制本身没问题，但它有个局限：**一次只能关注"一种"关系。** 可一句话里，词与词的关系往往是多重的——既有语法上的主谓宾搭配，又有语义上的指代关系，还有上下文的修饰关系。指望一组 $Q,K,V$ 把这些全捕捉到，太勉强了。

于是研究者想了个很朴素的办法：**与其用一个注意力看一遍，不如同时用好几个注意力，各看各的，最后把结果拼起来。** 每一个独立的注意力叫一个**头（head）**，多个头并行就是"多头"。打个比方，这就像读一段文字时，让语文老师看修辞、数学老师看逻辑、历史老师看背景，三个人各读一遍再汇总——比一个人读三遍更全面。

具体怎么做？设一共有 $h$ 个头，我们就准备 $h$ 套独立的投影矩阵，把输入分别投影成 $h$ 组 $Q_i,K_i,V_i$（每组维度是总维度的 $1/h$，这样总计算量不变），各自算一遍注意力，再把 $h$ 个输出在维度上拼接，最后用一个矩阵 $W^O$ 把它们融合回原来的维度：

$$\text{MHA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O,\quad \text{head}_i=\text{Attention}(Q_i,K_i,V_i).$$

记住这个画面就够了：**多头 = 多个视角并行看 + 最后融合。** 它几乎不增加计算量，却显著增强了模型的表达力，这也是为什么实际模型里头数动辄 8 个、16 个。

## 前馈网络：给每个位置单独"想一想"

注意力解决的是"位置之间怎么交流信息"，但交流完之后，每个位置还需要对自己手里的信息做一番加工。这件事交给图里上面那个盒子——**前馈网络（feed-forward network, FFN）**。

它的结构简单得出奇，就是两层全连接夹一个激活函数：

$$\text{FFN}(x)=\max(0,\,xW_1+b_1)\,W_2+b_2.$$

这里有两个关键细节值得讲。**第一，它是"逐位置"独立作用的**——同一个 FFN 被复制到每个位置上分别运算，位置之间互不干扰。注意力管"横向沟通"，FFN 管"纵向深加工"，分工明确。**第二，它中间那层会先把维度撑大再压回来**，通常放大到 4 倍（比如 512 维先升到 2048 维，过完激活再降回 512）。这个"先胖后瘦"的设计，给了模型足够的空间去做非线性变换，是 Transformer 参数量的大头。

所以一个 block 的节奏其实很好记：**先用注意力让大家互通有无，再用 FFN 让每个位置各自消化。** 一横一纵，配合得天衣无缝。

## 残差与 LayerNorm：让深网络稳得住

图里还剩两个看着不起眼的部件：每个盒子外面那条绕过去的**残差直连**，和盒子之后那道 **LayerNorm**。别小看它们——没有这两样，Transformer 根本堆不深，也就训不动。

先说**残差连接（residual connection）**。它的做法简单到让人怀疑：把某一层的输入 $x$ 直接加到这一层的输出上，写成 $y=x+f(x)$。为什么要这么干？因为网络一深，梯度从后往前传时会越乘越小、最后消失（这就是上一章提过的梯度消失）。而那个"+ $x$"相当于给梯度修了一条**高速公路**——反向传播时梯度可以原封不动地直接抄近路传回去，不必每层都被衰减。正是这条高速公路，让人们能放心地把网络堆到几十上百层。

再说 **层归一化（layer normalization, LayerNorm）**。它对**每一个样本的特征向量**做归一化：把这个向量里的数减去自己的均值、除以自己的标准差，拉到一个标准的分布上，再用两个可学习的参数缩放和平移回来。

$$\text{LayerNorm}(x)=\gamma\,\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.$$

你可能会问，这跟卷积网络里常用的 BatchNorm 有什么不一样？区别很关键：**BatchNorm 是跨一批样本、对每个特征算统计量，LayerNorm 是对单个样本、跨它的所有特征算统计量。** 在文本里，每个句子长短不一、批次大小也常变，跨样本统计很不稳；而 LayerNorm 只看样本自己，不受这些影响——这正是 Transformer 偏爱它的原因。把数据每过一道运算就重新拉回稳定分布，整个训练就平顺多了。

## 位置编码：让模型知道谁先谁后

到这里 block 的四件套讲完了，但有个隐藏的麻烦还没解决。回头看注意力的公式，它是对所有位置"一视同仁"地加权求和的——这意味着**它根本不知道词的先后顺序**。把"狗咬人"打乱成"人咬狗"，注意力算出来竟然是一样的！这显然不行，语言里顺序太重要了。

怎么补救？办法是在把词向量送进网络之前，**给每个位置额外加上一个能体现"我排第几"的向量**，这个向量就叫**位置编码（positional encoding）**。主流有两种做法。

**第一种是正弦位置编码**，也是原论文的做法。它用不同频率的正弦、余弦函数，按位置 $pos$ 和维度 $i$ 算出一串固定的数：

$$PE_{(pos,2i)}=\sin\!\Big(\frac{pos}{10000^{2i/d}}\Big),\qquad PE_{(pos,2i+1)}=\cos\!\Big(\frac{pos}{10000^{2i/d}}\Big).$$

它的妙处在于：不同维度用不同的"波长"，就像时钟上的秒针、分针、时针配合起来能唯一标定时刻一样，这一串正余弦也能给每个位置一个独一无二的"指纹"。而且因为是按公式算的、不需要训练，**遇到比训练时更长的句子也能直接外推**。

**第二种是可学习位置编码**，做法更省事：干脆为每个位置准备一个向量，当成普通参数让模型自己在训练中学。BERT、GPT 用的都是这种。它更灵活，但有个硬伤——**位置数量是写死的**，超过训练长度的位置就没有对应编码了，没法外推。两种各有取舍：要外推能力选正弦式，图省事、句长固定就用可学习式。

## Encoder 还是 Decoder：同一套零件，两种搭法

最后一个大问题：你可能在不同地方见过 Transformer 有 **Encoder（编码器）** 和 **Decoder（解码器）** 两半，它们到底差在哪？其实零件完全一样，差别只在**注意力能"看"到哪些位置**。

**Encoder 是双向的。** 它的每个位置在算注意力时，可以同时看到左边和右边的所有词——因为它的任务是"理解"一整句已经摆在眼前的话，前后文当然都该看。BERT 就是纯 Encoder，适合做分类、抽取这类理解类任务。

**Decoder 是单向的。** 它的任务是"一个词一个词地往外生成"，所以在预测第 $t$ 个词时，**绝不能偷看第 $t$ 个词及它之后的内容**——否则就成了抄答案。为此 Decoder 用一种叫**掩码注意力（masked attention）** 的技巧：在 softmax 之前，把所有"未来位置"的注意力分数强行设成负无穷，这样它们的权重就变成 0，等于被遮住了。GPT 就是纯 Decoder，专门做生成。

<div class="diagram">
<svg viewBox="0 0 420 180" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- Encoder：双向，全亮 -->
  <text x="100" y="24" text-anchor="middle" font-size="13" fill="var(--dia-blue)">Encoder（双向：都能看）</text>
  <!-- 4x4 网格 -->
  <g>
    <rect x="40" y="40" width="120" height="120" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
    <!-- 全部填充表示可见 -->
    <rect x="40" y="40" width="120" height="120" fill="var(--dia-accent-soft)"/>
    <line x1="70" y1="40" x2="70" y2="160" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="100" y1="40" x2="100" y2="160" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="130" y1="40" x2="130" y2="160" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="40" y1="70" x2="160" y2="70" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="40" y1="100" x2="160" y2="100" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="40" y1="130" x2="160" y2="130" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
  </g>
  <text x="100" y="176" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">每个位置看到全部</text>
  <!-- Decoder：单向，下三角 -->
  <text x="320" y="24" text-anchor="middle" font-size="13" fill="var(--dia-accent-deep)">Decoder（单向：只看过去）</text>
  <g>
    <rect x="260" y="40" width="120" height="120" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
    <!-- 下三角填充：第 r 行只填到第 r 列 -->
    <rect x="260" y="40"  width="30" height="30" fill="var(--dia-accent-soft)"/>
    <rect x="260" y="70"  width="60" height="30" fill="var(--dia-accent-soft)"/>
    <rect x="260" y="100" width="90" height="30" fill="var(--dia-accent-soft)"/>
    <rect x="260" y="130" width="120" height="30" fill="var(--dia-accent-soft)"/>
    <line x1="290" y1="40" x2="290" y2="160" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="320" y1="40" x2="320" y2="160" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="350" y1="40" x2="350" y2="160" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="260" y1="70" x2="380" y2="70" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="260" y1="100" x2="380" y2="100" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
    <line x1="260" y1="130" x2="380" y2="130" stroke="var(--dia-stroke-tertiary)" stroke-width="0.6"/>
  </g>
  <text x="320" y="176" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">未来被掩码遮住</text>
</svg>
</div>
<p class="figure-caption">图 2.2-2：同样是注意力矩阵，Encoder 全部可见（左，方块全亮），Decoder 只能看自己和左边（右，下三角）。这就是掩码注意力。</p>

至于把两半拼在一起的 **Encoder-Decoder** 结构（原版翻译模型就是这样），适合"输入一段、输出另一段"的任务，比如机器翻译、摘要。这部分我们留到 [4.3 Encoder-Decoder](3-4-3-encoder-decoder.md) 专门讲。

## 动手搭一个 Transformer block

道理讲完了，我们动手把一个 block 拼出来。先从零搭——这样你能看清每个部件是怎么接起来的。下面这段代码就是把图 2.2-1 翻译成 PyTorch：

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # 多头注意力：PyTorch 自带一个现成的，batch_first 让形状是 (批, 序列长, 维度)
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         dropout=dropout, batch_first=True)
        # 前馈网络：先升维到 d_ff，过 ReLU，再降回 d_model（先胖后瘦）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        # 两道 LayerNorm，分别守在注意力和 FFN 之后
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 第一步：注意力 + 残差 + 归一化
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)  # Q=K=V=x，叫自注意力
        x = self.norm1(x + self.drop(attn_out))          # 注意这个 "x +" 就是残差
        # 第二步：前馈 + 残差 + 归一化
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x

# 造一批假数据试跑：2 个句子，每句 10 个词，每个词 512 维
x = torch.randn(2, 10, 512)
block = TransformerBlock()
out = block(x)
print(out.shape)        # torch.Size([2, 10, 512])，进去什么形状，出来还是什么形状
```

跑完你会看到输出形状和输入**一模一样**，都是 `(2, 10, 512)`。这正是 block 的一个重要性质：**它不改变张量的形状，只改变里面的内容**。也正因为形状不变，我们才能像搭积木一样把一个个 block 直接摞起来，堆多少层都行。

留意 `forward` 里那两行 `x = self.norm(x + ...)`——那个不起眼的 `x +`，就是我们前面反复强调的残差连接。少了它，这段代码照样能跑，但网络一深就训不动了。

## 库版：直接用 PyTorch 官方实现

自己搭一遍是为了理解，真正写竞赛代码时，没必要每次重造轮子。PyTorch 已经把上面这套封装成了 `nn.TransformerEncoderLayer`，还能用 `nn.TransformerEncoder` 一次性堆叠多层。我们用它来对照，验证我们对结构的理解没跑偏：

```python
import torch
import torch.nn as nn

# 一层 Encoder block，参数含义和我们手写的完全对应
layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048,
    dropout=0.1, batch_first=True,
)
# 把 6 层一样的 block 堆起来，就是一个完整的 Encoder
encoder = nn.TransformerEncoder(layer, num_layers=6)

x = torch.randn(2, 10, 512)
out = encoder(x)
print(out.shape)        # torch.Size([2, 10, 512])，同样形状不变

# 想要 Decoder 的"只看过去"效果，生成一个上三角掩码即可
mask = nn.Transformer.generate_square_subsequent_mask(10)
print(mask[:3, :3])     # 上三角是 -inf，对应图 2.2-2 里被遮住的未来位置
```

对照着看，你会发现官方实现和我们手写的逻辑严丝合缝：`d_model`、`nhead`、`dim_feedforward` 一一对应，输出形状也都不变。区别只在于官方版考虑了更多工程细节（比如各种掩码、初始化、效率优化）。**理解用手写版，干活用官方版**——这是学这类核心模块的标准姿势。

## 容易踩的坑

- **`d_model` 必须能被 `n_heads` 整除**。因为多头要把维度均分给每个头，512 配 8 个头没问题（每头 64 维），但配 7 个头就会直接报错。
- **忘了加位置编码**。光把词向量丢进 Encoder，模型是分不清词序的，效果会莫名其妙地差。位置编码要在进 block 之前就加到输入上。
- **掩码用错或忘了用**。做生成任务（Decoder）时必须加上三角掩码，否则模型训练时就偷看了未来的答案，一到推理就崩——这种 bug 训练指标好看、实测稀烂，最难查。
- **Pre-LN 还是 Post-LN 别混**。原论文是"先运算再归一化"（Post-LN），现在很多大模型改成"先归一化再运算"（Pre-LN），后者更好训。两者位置不同，照着某份代码抄时别张冠李戴。
- **`batch_first` 不设的坑**。PyTorch 的 Transformer 模块默认形状是 `(序列长, 批, 维度)`，和大家习惯的 `(批, 序列长, 维度)` 正好头两维相反。不设 `batch_first=True`，形状对不上又不报错，结果全错。

## 它在后面会怎么用到

这一节是整个 Round 2 后半程的地基，几乎所有现代模型都建在它上面：

- **语言建模**直接用 Decoder 堆叠出 GPT 式的生成模型（见 [4.2 语言建模](3-4-2-language-modeling.md)）。
- **预训练大模型** BERT、GPT 全是 Transformer 的不同搭法，懂了这一节就懂了它们的骨架（见 [4.4 预训练大模型](3-4-4-pretrained-llms.md)）。
- **视觉-文本编码器 CLIP** 里，处理文本的那一半就是个 Transformer Encoder（见 [3.4 视觉-文本编码器 CLIP](3-3-4-clip.md)）。
- 连**图像分割、目标检测**的前沿模型（ViT、DETR）也把图片切成小块当"词"，套上 Transformer 来做（见 [3.1 目标检测](3-3-1-object-detection.md)）。
- 而它最依赖的那个零件——注意力，请务必先在 [2.1 注意力机制](3-2-1-attention.md) 里吃透。

## 练习

??? note "基础练习"
    1. 把上面手写的 `TransformerBlock` 跑起来，故意把 `n_heads` 改成 7，看看会报什么错，再想一想为什么 512 不能被 7 整除就不行。
    2. 用 `nn.Transformer.generate_square_subsequent_mask(5)` 打印一个 5×5 的掩码，对照图 2.2-2，确认它确实是"上三角为 -inf、下三角为 0"。
    3. 给一段长度为 20、维度为 64 的输入，分别用正弦公式手算前两个位置的位置编码向量，验证它们确实不相等。

??? note "进阶练习"
    1. 自己实现一个不调用 `nn.MultiheadAttention` 的多头注意力：手动把输入投影成 $h$ 组 $Q,K,V$、各自算注意力、再拼接融合，和官方实现对比输出是否接近。
    2. 把手写 block 里的 LayerNorm 全部删掉，再训练同一个小任务，观察损失曲线变得多不稳定——亲眼感受归一化的作用。
    3. 实现一个完整的正弦位置编码模块，把它加到输入上，再喂进 Encoder；对比"加了位置编码"和"没加"在一个简单的序列排序任务上的差距。

## 小结

- 一个 Transformer block 就四件套：**多头注意力**（多视角横向沟通）、**FFN**（逐位置纵向加工）、**残差**（给梯度修高速公路）、**LayerNorm**（把数据拉回稳定分布）。形状不变，所以能无限堆叠。
- **位置编码**给模型补上"谁先谁后"的信息，正弦式能外推、可学习式更灵活。
- **Encoder 双向看全部，Decoder 单向只看过去**，区别全在掩码注意力上。
- 理解用手写版、干活用 `nn.TransformerEncoderLayer`，两者逻辑一一对应。

想把直觉建得更牢，强烈推荐看 Jay Alammar 的图解博客 [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)，它用大量动图把数据流讲得无比清楚；想抠原始细节，就去读那篇开山论文 [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)。
