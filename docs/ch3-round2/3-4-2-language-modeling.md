# 4.2 语言建模

> **难度** ⭐⭐⭐⭐☆ · **前置**：[2.2 Transformer 架构](3-2-2-transformer.md)、[4.1 文本分类与词嵌入](3-4-1-text-embeddings.md)

!!! abstract "读完这一节，你会"
    - 说清楚"语言建模"到底在建什么模——也就是自回归地预测下一个 token
    - 理解字符、词、BPE 三种分词方式各自的取舍，知道为什么大模型几乎都用 BPE
    - 会算困惑度（perplexity），并解释它为什么等于"模型每一步在几个词里犹豫"
    - 看懂 decoder-only（GPT 式）模型为什么必须用因果掩码，并能用 PyTorch 从零训一个小语言模型

## 语言模型到底在"建"什么

你每天用的输入法联想、手机里的自动补全、还有 ChatGPT，它们背后是同一件事：**给定前面已经出现的文字，猜下一个字最可能是什么。** 这就是语言建模（language modeling）的全部出发点——听起来简单到有点不可思议，但正是这个朴素的目标，撑起了今天所有大语言模型。

我们把它说得更精确一点。一句话其实是一串符号，比如"我 爱 吃 苹果"。语言模型想做的，是估计这一整串话出现的概率 $P(\text{我爱吃苹果})$。可一整句话的组合千千万万，没法直接数出来。于是我们用一个数学上恒等、却好算得多的方式，把它拆成"一个接一个"的条件概率：

$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \dots, w_{t-1}).$$

这条公式叫概率的**链式法则**（注意，它和[微积分里的链式法则](../ch1-intro/1-3-2-calculus-gradients.md)同名但是两回事）。它说的是：整句话的概率，等于"在已知前面所有词的前提下、第一个词的概率，乘以第二个词的概率，再乘以第三个……"一路乘到底。换句话说，**只要我们能预测好每一步的"下一个词"，就等于学会了整门语言。** 这种"从左往右、一个一个往外吐"的建模方式，就叫**自回归（autoregressive）**。

<div class="diagram">
<svg viewBox="0 0 420 170" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="lm-a" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 已知前文方块 -->
  <rect x="20" y="60" width="48" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)" stroke-width="1.4"/>
  <text x="44" y="82" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">我</text>
  <rect x="78" y="60" width="48" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)" stroke-width="1.4"/>
  <text x="102" y="82" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">爱</text>
  <rect x="136" y="60" width="48" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)" stroke-width="1.4"/>
  <text x="160" y="82" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">吃</text>
  <!-- 模型 -->
  <rect x="210" y="50" width="78" height="54" rx="6" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.6"/>
  <text x="249" y="82" text-anchor="middle" font-size="13" fill="var(--dia-accent-deep)">语言模型</text>
  <!-- 预测分布 -->
  <rect x="330" y="60" width="70" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-green)" stroke-width="1.6"/>
  <text x="365" y="82" text-anchor="middle" font-size="13" fill="var(--dia-green)">苹果?</text>
  <!-- 箭头 -->
  <line x1="184" y1="77" x2="206" y2="77" stroke="var(--dia-stroke-soft)" stroke-width="1.4" marker-end="url(#lm-a)"/>
  <line x1="288" y1="77" x2="326" y2="77" stroke="var(--dia-accent)" stroke-width="1.6" marker-end="url(#lm-a)"/>
  <!-- 标注：放在方块上下方空白处 -->
  <text x="102" y="40" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">已知前文（context）</text>
  <text x="365" y="128" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">下一个 token 的概率分布</text>
</svg>
</div>
<p class="figure-caption">图 4.2-1：语言建模的核心动作——把已读到的前文喂进模型，输出"下一个 token 该是谁"的概率分布。</p>

把这张图记住：后面无论是 GPT 还是更大的模型，剥到最里层，干的都是这一件事。

## 喂进模型的最小单位：token 与三种分词

上一节我们一直在说"词""字""token"，现在得把它们分清楚。模型并不直接读字符串，它读的是一串整数——每个整数对应词表里的一个**符号单元（token）**。那么问题来了：一段文本到底该切成什么样的单元？这一步叫**分词（tokenization）**，常见有三种切法，各有各的脾气。

**第一种，按字符切（character-level）。** 把"hello"切成 `h e l l o` 五个字符。好处是词表极小（英文也就几十个字符），永远不会遇到"不认识的词"。坏处是序列变得很长，而且模型要从零学起"哪些字符连在一起才算一个词"，学习负担重。

**第二种，按词切（word-level）。** 把"I love cats"切成 `I` `love` `cats` 三个词。每个 token 自带语义，序列也短。但麻烦在于词表会爆炸——光英语就有几十万词，更别说还会冒出训练时没见过的新词（叫 OOV，out-of-vocabulary），模型只能两手一摊标成"未知"。

**第三种，也是今天主流的折中方案——子词切分，最常用的是 BPE（Byte Pair Encoding，字节对编码）。** 它的想法很聪明：先把所有文本拆成单个字符，然后反复统计"哪两个相邻单元最常一起出现"，把出现最频繁的那一对**合并**成一个新单元，如此循环若干次。常见的词（如 `the`、`ing`）会被合并成一整块，而生僻词则保留为几个子词的拼接。这样既控制住了词表大小，又几乎不会出现 OOV——再奇怪的词，大不了拆成一串字符拼出来。

下面这张图把 BPE 的"逐步合并"画了出来，你顺着箭头走一遍就懂了：

<div class="diagram">
<svg viewBox="0 0 420 180" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="bpe-a" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 第一行：单字符 -->
  <text x="30" y="44" font-size="12" fill="var(--dia-stroke-soft)">起点（按字符）</text>
  <g font-size="14" fill="var(--dia-stroke)">
    <rect x="160" y="28" width="22" height="26" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="171" y="46" text-anchor="middle">l</text>
    <rect x="186" y="28" width="22" height="26" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="197" y="46" text-anchor="middle">o</text>
    <rect x="212" y="28" width="22" height="26" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="223" y="46" text-anchor="middle">w</text>
    <rect x="238" y="28" width="22" height="26" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="249" y="46" text-anchor="middle">e</text>
    <rect x="264" y="28" width="22" height="26" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="275" y="46" text-anchor="middle">r</text>
  </g>
  <!-- 箭头1 -->
  <line x1="223" y1="60" x2="223" y2="84" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#bpe-a)"/>
  <text x="300" y="78" font-size="11" fill="var(--dia-accent-deep)">合并最常见对 l+o → lo</text>
  <!-- 第二行 -->
  <g font-size="14" fill="var(--dia-stroke)">
    <rect x="170" y="92" width="34" height="26" rx="3" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)"/><text x="187" y="110" text-anchor="middle">lo</text>
    <rect x="208" y="92" width="22" height="26" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="219" y="110" text-anchor="middle">w</text>
    <rect x="234" y="92" width="22" height="26" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="245" y="110" text-anchor="middle">e</text>
    <rect x="260" y="92" width="22" height="26" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="271" y="110" text-anchor="middle">r</text>
  </g>
  <!-- 箭头2 -->
  <line x1="223" y1="124" x2="223" y2="146" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#bpe-a)"/>
  <text x="300" y="140" font-size="11" fill="var(--dia-accent-deep)">再合并 lo+w → low</text>
  <!-- 第三行 -->
  <g font-size="14" fill="var(--dia-stroke)">
    <rect x="174" y="152" width="42" height="24" rx="3" fill="var(--dia-green-soft)" stroke="var(--dia-green)"/><text x="195" y="169" text-anchor="middle">low</text>
    <rect x="220" y="152" width="22" height="24" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="231" y="169" text-anchor="middle">e</text>
    <rect x="246" y="152" width="22" height="24" rx="3" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-soft)"/><text x="257" y="169" text-anchor="middle">r</text>
  </g>
</svg>
</div>
<p class="figure-caption">图 4.2-2：BPE 从单字符出发，每一步把"最常一起出现的一对"合并成新单元，常见词逐渐被合成整块。</p>

我们也可以直接用现成的分词库感受一下 BPE 的输出。下面这几行加载 GPT-2 用的那套 BPE 分词器，把一句话切成 token 看看：

```python
from transformers import GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")
text = "Tokenization is unbelievable!"
ids = tok.encode(text)              # 文本 → 一串整数 id
print(ids)                          # 例如 [30642, 1634, 318, 21368, 0]
print(tok.convert_ids_to_tokens(ids))
# ['Token', 'ization', 'Ġis', 'Ġunbeliev', 'able', '!']
```

你会注意到两件事：`unbelievable` 这个长词被拆成了 `unbeliev` + `able` 两块（因为整词不够常见），而 `Ġ` 这个怪符号其实代表"前面有个空格"——BPE 连空格都编码进去了，这样模型才能准确还原原文。看懂这一步，你就理解了"模型眼里的文本"长什么样。

## 困惑度：模型每一步在几个词里犹豫

训好一个语言模型，怎么判断它好不好？我们当然希望它把真实出现的句子判得"概率高"。所以最自然的损失，就是让模型对真实下一个 token 的预测概率尽量大——这其实就是你在[激活与损失函数](../ch2-round1/2-2-3-activations-losses.md)那节学过的**交叉熵损失**，只不过这里是在整个词表上做分类，类别数等于词表大小。

不过在 NLP 里，大家更喜欢报告一个叫**困惑度（perplexity, PPL）**的指标，它就是交叉熵的指数形式：

$$\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t \mid w_{<t})\right).$$

公式看着唬人，但它有一个极其直观的解读：**困惑度大约等于"模型在每一步平均要在多少个词之间犹豫"。** 如果一个模型的困惑度是 1，说明它每一步都笃定地只选一个词、完全不纠结，那是完美预测；如果困惑度是 100，说明它每一步像是在 100 个候选里随机猜——非常糊涂。所以**困惑度越低越好**，它是衡量语言模型的头号指标。

举个对照你就有数了：一个完全乱猜、对 5 万词表均匀分布的模型，困惑度就是 50000；而一个不错的英文模型，困惑度能压到 20 上下。代码里算它也很省事，对损失取指数即可：

```python
import torch
import torch.nn.functional as F

# logits: (T, V) 模型对每一步在词表上的打分；targets: (T,) 真实的下一个 token
loss = F.cross_entropy(logits, targets)   # 平均交叉熵（以自然对数 e 为底）
ppl = torch.exp(loss)                      # 困惑度就是交叉熵的指数
print(f"loss={loss.item():.3f}, ppl={ppl.item():.1f}")
```

注意一个常见的小坑：`cross_entropy` 默认用自然对数，所以这里配套的是 `exp`。如果你看到别人用以 2 为底的对数算 PPL，那就得换成 `2**loss`——两种写法都对，但底数必须前后一致。

## GPT 式模型：decoder-only 与因果掩码

铺垫到这里，我们终于可以拼出今天最主流的语言模型骨架了。它叫 **decoder-only（仅解码器）架构**，GPT 系列就是它的代表。名字里的"decoder-only"是相对于完整 Transformer 而言的——[Transformer 架构](3-2-2-transformer.md)原本有编码器和解码器两半，而做纯语言建模时，我们只留下解码器这一半，把它一层层叠起来。

这里有一个**绝对不能搞错**的细节。语言模型是自回归的，预测第 $t$ 个词时，**只能看见它前面的词，绝不能偷看后面的**——否则就成了"抄答案"，训练出的模型一到真实生成时就废了。可是注意力机制天生是"每个位置都能看到所有位置"的。怎么办？答案是给注意力加一个**因果掩码（causal mask）**：在算注意力分数时，把每个位置"看向未来"的那些权重统统设成负无穷，经过 softmax 后它们就变成 0，相当于把未来挡住了。

<div class="diagram">
<svg viewBox="0 0 300 230" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 4x4 注意力掩码网格，行=query 位置，列=key 位置 -->
  <!-- 列标题 -->
  <text x="150" y="20" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">能看到的位置 (key) →</text>
  <!-- 行标题 -->
  <text x="22" y="130" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)" transform="rotate(-90 22 130)">当前位置 (query) ↓</text>
  <!-- 网格：可见=accent 实心，被挡=空心 -->
  <!-- row 1 -->
  <rect x="60" y="40" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="100" y="40" width="40" height="40" fill="var(--dia-bg-deep)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="140" y="40" width="40" height="40" fill="var(--dia-bg-deep)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="180" y="40" width="40" height="40" fill="var(--dia-bg-deep)" stroke="var(--dia-bg)" stroke-width="2"/>
  <!-- row 2 -->
  <rect x="60" y="80" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="100" y="80" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="140" y="80" width="40" height="40" fill="var(--dia-bg-deep)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="180" y="80" width="40" height="40" fill="var(--dia-bg-deep)" stroke="var(--dia-bg)" stroke-width="2"/>
  <!-- row 3 -->
  <rect x="60" y="120" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="100" y="120" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="140" y="120" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="180" y="120" width="40" height="40" fill="var(--dia-bg-deep)" stroke="var(--dia-bg)" stroke-width="2"/>
  <!-- row 4 -->
  <rect x="60" y="160" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="100" y="160" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="140" y="160" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <rect x="180" y="160" width="40" height="40" fill="var(--dia-accent)" stroke="var(--dia-bg)" stroke-width="2"/>
  <!-- 图例放在右侧空白处 -->
  <rect x="240" y="86" width="18" height="18" fill="var(--dia-accent)"/>
  <text x="240" y="120" font-size="11" fill="var(--dia-stroke-soft)">可见</text>
  <rect x="240" y="132" width="18" height="18" fill="var(--dia-bg-deep)" stroke="var(--dia-stroke-tertiary)"/>
  <text x="240" y="166" font-size="11" fill="var(--dia-stroke-soft)">被挡</text>
</svg>
</div>
<p class="figure-caption">图 4.2-3：因果掩码就是一个下三角——第 t 行只允许看前 t 列（自己和更早的位置），右上角的"未来"全部挡掉。</p>

你看图就明白了：这个掩码本质是一个**下三角矩阵**，第一行只能看第一个位置，第二行能看前两个……越往后能看到的越多，但永远看不到自己右边。理解了这个三角，你就抓住了 decoder-only 和上一节那种 encoder（BERT 式、能双向看全文）最根本的区别。

## 动手训一个迷你语言模型

光说不练假把式。我们用 PyTorch 搭一个最小可跑的字符级语言模型，把上面所有概念串起来。为了聚焦语言建模本身，注意力部分我们直接用 PyTorch 内置的 `TransformerEncoderLayer`，但**关键是给它传一个因果掩码**，让它变成自回归的解码器。

第一步，准备数据。我们拿一小段文本，按字符建词表，把整段文本变成一串整数 id：

```python
import torch
import torch.nn as nn

text = "hello world. " * 200          # 玩具语料，真实任务请换成大文本
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}   # 字符 → id
itos = {i: c for c, i in stoi.items()}       # id → 字符
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
V = len(chars)                                # 词表大小
```

第二步，定义模型。它由三块拼成：一个把 token id 变成向量的**嵌入层**（这正是[词嵌入](3-4-1-text-embeddings.md)那节学的东西），一个带因果掩码的 **Transformer 解码层**，最后一个把每个位置映射回"词表上打分"的**线性输出头**：

```python
class MiniLM(nn.Module):
    def __init__(self, V, d=64, nhead=4, T=32):
        super().__init__()
        self.tok_emb = nn.Embedding(V, d)          # token 嵌入
        self.pos_emb = nn.Embedding(T, d)          # 位置嵌入（让模型知道顺序）
        layer = nn.TransformerEncoderLayer(d, nhead, batch_first=True)
        self.block = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d, V)                 # 输出头：每个位置 → 词表打分
        self.T = T

    def forward(self, x):                           # x: (B, T) 一批 token id
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)     # 词义 + 位置
        # 关键一步：构造下三角因果掩码，挡住未来
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.block(h, mask=mask)
        return self.head(h)                         # (B, T, V)
```

请特别留意 `forward` 里造掩码那一行——它就是图 4.2-3 画的那个下三角。少了它，模型就能在训练时偷看答案，看似 loss 降得飞快，一拿去生成却胡言乱语。

第三步，训练。语言建模的标签有个漂亮的特点：**输入序列往后挪一位，就是它自己的标签**（要预测的"下一个词"嘛）。所以我们切一段 token 当输入 `x`，把它整体右移一位当目标 `y`：

```python
T = 32
model = MiniLM(V, T=T)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(300):
    i = torch.randint(0, len(data) - T - 1, (16,))         # 随机取 16 个起点
    x = torch.stack([data[j:j+T] for j in i])              # 输入
    y = torch.stack([data[j+1:j+T+1] for j in i])          # 标签 = 输入右移一位
    logits = model(x)                                       # (B, T, V)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, V), y.reshape(-1))               # 摊平成普通分类
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step}: loss={loss.item():.3f}, ppl={loss.exp().item():.1f}")
```

你会看到 loss 和困惑度一路往下掉。困惑度从一开始接近"词表大小"（瞎猜），慢慢降到很小的个位数——这说明模型已经学会了"hello world"这种简单规律，预测下一个字符时不再犹豫。

最后，自回归地**生成**文本：喂一个起始字符，让模型预测下一个，把预测拼回去再喂进去，循环往复。这正是图 4.2-1 那个动作在反复执行：

```python
@torch.no_grad()
def generate(model, start="h", n=40):
    model.eval()
    ids = [stoi[start]]
    for _ in range(n):
        x = torch.tensor(ids[-T:]).unsqueeze(0)    # 只保留最近 T 个 token
        logits = model(x)[0, -1]                    # 只要最后一个位置的预测
        nxt = torch.softmax(logits, -1).argmax().item()
        ids.append(nxt)
    return "".join(itos[i] for i in ids)

print(generate(model))      # 类似 'hello world. hello world. ...'
```

跑通这一段，你就亲手实现了 GPT 的最小内核——所谓大语言模型，无非是把这个迷你版的层数、维度、数据量同时放大几万倍而已。

## 容易踩的坑

- **忘了加因果掩码**：这是 decoder-only 模型头号大坑。漏了它，模型训练时能看见未来，loss 假性骤降，生成时却完全失灵。看到"训练完美、生成崩坏"，先回去检查掩码。
- **输入和标签没对齐**：标签必须是输入**右移一位**。如果错位成"原地预测自己"，模型只要照抄输入就能让 loss 归零，等于什么都没学到。
- **困惑度的底数搞混**：用 `cross_entropy`（自然对数）算的 loss，PPL 要用 `exp(loss)`；用以 2 为底的 loss 则要 `2**loss`。两套别混用，否则数字差出一截、没法和论文对照。
- **拿字符级困惑度去和词级比**：困惑度强烈依赖分词粒度，字符级的 PPL 和词级、BPE 级的根本不在一个尺度上，不能直接比大小。
- **生成时序列越长越慢**：每生成一个 token 都要把前文重算一遍注意力，朴素实现会越来越慢。真实系统用 KV 缓存来加速，这一点[预训练大模型](3-4-4-pretrained-llms.md)那节会细讲。

## 它在后面会怎么用到

这一节是通往大模型的正门——你已经掌握了 GPT 的全部核心思想，剩下的都是"放大"。

- **预训练大模型**就是把这里的 decoder-only 架构在海量文本上做自回归预训练，再加上指令微调、RLHF 等手段（见 [4.4 预训练大模型](3-4-4-pretrained-llms.md)）。这是本节最直接的下游。
- **Encoder-Decoder** 模型（如翻译用的 T5）会用到这里的解码器，但搭配一个能双向看全文的编码器，处理"输入一段、输出另一段"的任务（见 [4.3 Encoder-Decoder](3-4-3-encoder-decoder.md)）。
- 嵌入层、注意力、交叉熵这些零件，分别来自 [4.1 文本分类与词嵌入](3-4-1-text-embeddings.md)、[2.1 注意力机制](3-2-1-attention.md) 和 [2.3 激活函数与损失函数](../ch2-round1/2-2-3-activations-losses.md)——本节是把它们组装起来的"总装车间"。

## 练习

??? note "基础练习"
    1. 用 GPT-2 的分词器把三句话编码成 token，数一数：同样一句中文和一句等长的英文，谁切出的 token 更多？想一想为什么（提示：BPE 的训练语料主要是英文）。
    2. 给定一个语言模型对某句话每一步的预测概率 `[0.5, 0.25, 0.5, 0.1]`，手算这句话的困惑度。再想想：如果其中某一步概率是 0，困惑度会变成多少？这提示了什么数值问题？
    3. 把本节迷你模型里"造因果掩码"那一行注释掉，重新训练，观察 loss 是不是降得更快、但 `generate` 出来的文本反而更糟。亲手验证一下"偷看未来"的后果。

??? note "进阶练习"
    1. 给迷你模型的 `generate` 加上**温度采样**：把 logits 除以温度 $T_{\text{temp}}$ 再做 softmax，然后用 `torch.multinomial` 随机采样而不是 `argmax`。对比温度取 0.5、1.0、1.5 时生成文本的"保守 vs 放飞"程度。
    2. 自己实现一个最小版 BPE：从字符开始，反复统计相邻对的频次、合并最高频的那一对，做 50 次合并，打印出每一步合并了哪两个单元。对照图 4.2-2 检查你的实现。
    3. 换一段更长的真实文本（比如一篇英文短文）训练迷你模型，在留出的验证文本上计算困惑度，画出它随训练步数下降的曲线——这就是评估语言模型最标准的做法。

## 小结

- 语言建模的本质是**自回归地预测下一个 token**，靠链式法则把整句概率拆成一串条件概率，"会预测下一个词"就等于"学会了语言"。
- 文本要先**分词**才能喂给模型：字符级词表小但序列长，词级语义强但词表爆炸，**BPE 子词**是今天的主流折中。
- **困惑度**衡量模型每一步"在几个词里犹豫"，越低越好，它就是交叉熵取指数。
- **GPT 式 decoder-only** 模型的命门是**因果掩码**（下三角），它保证预测时只看前文、不偷看未来。

想再往深里走，强烈推荐 Andrej Karpathy 的视频教程 [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY)，他从零手敲出一个能写莎士比亚的语言模型，和本节的迷你实现一脉相承、可以无缝衔接。
