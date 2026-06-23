# 4.3 Encoder-Decoder

> **难度** ⭐⭐⭐⭐☆ · **前置**：[2.2 Transformer 架构](3-2-2-transformer.md)、[2.1 注意力机制](3-2-1-attention.md)、[4.2 语言建模](3-4-2-language-modeling.md)

!!! abstract "读完这一节，你会"
    - 说清"编码器-解码器（encoder-decoder）"这套架构在解决什么问题，以及它和单塔语言模型的区别
    - 理解**交叉注意力（cross-attention）**是怎么把源句子的信息喂给解码器的
    - 会区分**贪心解码（greedy）**和**集束搜索（beam search）**，知道各自适合什么场景
    - 能用 PyTorch 搭出一个最小可跑的 seq2seq 翻译/摘要骨架，并跑通自回归生成

## 一句话先说清：它是"读懂再重写"的机器

到目前为止，你见过的语言模型大多在做一件事：给一段文字，往后接着写。可现实里有一大类任务，长得不太一样——**输入是一段文字，输出是另一段文字，而且这两段还不一样长、不一样语言、甚至不一样形式。** 把英文翻成中文、把一篇长文章压成三句话摘要、把一个问题转写成 SQL 查询，都是这个套路。

这类任务有个统一的名字，叫**序列到序列（sequence-to-sequence，简称 seq2seq）**：输进去一个序列，吐出来另一个序列。它和"接着往下写"最大的不同在于，**模型必须先把整段输入完整地读懂、消化成一种内部理解，然后再据此从头组织出一段新的输出**——而不是简单地顺着原文续写。

那怎么让一个网络同时具备"读懂"和"重写"两种能力？最自然的办法，就是把它拆成两半：一半专门负责读，叫**编码器（encoder）**；另一半专门负责写，叫**解码器（decoder）**。编码器把输入嚼碎、提炼成一组富含语义的向量；解码器拿着这组向量当"参考资料"，一个词一个词地把答案生成出来。这就是**编码器-解码器架构**的全部直觉。

<div class="diagram">
<svg viewBox="0 0 480 240" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ed-a" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
    <marker id="ed-b" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-blue)"/></marker>
  </defs>
  <!-- 编码器框 -->
  <rect x="30" y="70" width="150" height="100" rx="6" fill="var(--dia-bg-card)" stroke="var(--dia-blue)" stroke-width="2"/>
  <text x="105" y="56" text-anchor="middle" font-size="13" fill="var(--dia-blue)">编码器 Encoder</text>
  <text x="105" y="126" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">读懂源句</text>
  <!-- 源输入 -->
  <text x="105" y="200" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">I love cats</text>
  <line x1="105" y1="186" x2="105" y2="172" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5" marker-end="url(#ed-b)"/>
  <!-- 上下文向量箭头 -->
  <line x1="182" y1="120" x2="298" y2="120" stroke="var(--dia-accent)" stroke-width="2" marker-end="url(#ed-a)"/>
  <text x="240" y="108" text-anchor="middle" font-size="11" fill="var(--dia-accent)">语义表示</text>
  <text x="240" y="138" text-anchor="middle" font-size="11" fill="var(--dia-accent)">（交叉注意力）</text>
  <!-- 解码器框 -->
  <rect x="300" y="70" width="150" height="100" rx="6" fill="var(--dia-bg-card)" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="375" y="56" text-anchor="middle" font-size="13" fill="var(--dia-accent)">解码器 Decoder</text>
  <text x="375" y="126" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">逐词生成</text>
  <!-- 目标输出 -->
  <text x="375" y="200" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">我 爱 猫</text>
  <line x1="375" y1="172" x2="375" y2="186" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5" marker-end="url(#ed-a)"/>
</svg>
</div>
<p class="figure-caption">图 4.3-1：编码器把源句"读"成一组语义向量，解码器一边看着这组向量、一边把目标句一个词一个词地"写"出来。</p>

把这张图记牢，下面所有内容都只是在给它填细节。

## 编码器和解码器，到底各干了什么

我们把两半拆开，一个一个说。

**编码器**接收整段源序列，比如 "I love cats" 这三个词。它会把这三个词同时看一遍——注意是"同时"，编码器对源句用的是**双向注意力**，每个词都能看到左边和右边的所有词，因为读理解嘛，前后文都得参考。读完之后，它输出的不是一个数，而是**一组向量**：源句有几个词，就有几个向量，每个向量都浓缩了"这个词在整句话里的含义"。这组向量就是解码器要反复查阅的"参考资料"。

**解码器**的任务是把答案写出来，但它写得很"克制"：**一次只写一个词，而且写每个词时，只能看见自己已经写出来的部分，看不见还没写的未来。** 这一点至关重要——因为生成的时候，未来本来就还不存在，模型不能作弊偷看。这种"只看左边"的注意力，我们在语言建模那一节叫过它**带掩码的自注意力（masked self-attention）**。

所以解码器其实同时背着两副担子：一副是**对已生成内容的自注意力**（保证自己写出来的话前后连贯），另一副是**对编码器输出的交叉注意力**（保证写出来的内容忠于原文）。后面这个"交叉注意力"是 encoder-decoder 区别于普通语言模型的灵魂，我们专门拿一节讲它。

顺便澄清一个常见误解：很多人以为 encoder-decoder 就是"两个 Transformer 拼起来"。其实更准确地说，**它们共用同一套 Transformer 积木（多头注意力 + 前馈层），只是在注意力的"看谁"上做了不同安排**——编码器看源句全文，解码器看自己的左侧 + 编码器全文。理解了这一点，你就不会被各种花哨的架构图吓到了。

## 交叉注意力：把源句子的信息"递"给解码器

这是本节最该吃透的机制，我们慢慢来。

先回忆一下注意力的三件套：查询 $Q$（query，"我想找什么"）、键 $K$（key，"我有什么"）、值 $V$（value，"找到了给你什么"）。普通自注意力里，$Q$、$K$、$V$ 全都来自同一句话。而**交叉注意力的巧妙之处在于：它让 $Q$ 来自解码器，$K$ 和 $V$ 来自编码器。**

换句话说，解码器在写第 $t$ 个词时，会拿当前的状态当作"查询"，去问编码器输出的那组源句向量："为了写好这个词，源句里哪几个部分最该参考？"——这正是注意力打分要回答的。打完分、加权求和，源句里相关的信息就被"递"到了解码器手里。它的数学形式和你熟悉的注意力一模一样：

$$
\text{CrossAttn}(Q_{\text{dec}},\,K_{\text{enc}},\,V_{\text{enc}})
=\text{softmax}\!\left(\frac{Q_{\text{dec}}K_{\text{enc}}^\top}{\sqrt{d_k}}\right)V_{\text{enc}}.
$$

公式里下标 $\text{dec}$ 和 $\text{enc}$ 标得很清楚：查询 $Q_{\text{dec}}$ 来自解码器当前位置，键 $K_{\text{enc}}$ 和值 $V_{\text{enc}}$ 来自编码器的全部输出。分母那个 $\sqrt{d_k}$ 还是老规矩，用来把点积缩放到一个稳定的量级，防止 softmax 过早饱和。

这件事在翻译里有个特别漂亮的副产品：如果你把交叉注意力的权重画成一张热力图，常常能看见**目标词和源词之间形成了清晰的对齐**——生成"猫"时，注意力会自动聚焦到源句的 "cats" 上。模型没人教它对齐，它是为了把活儿干好，自己学出来的。

<div class="diagram">
<svg viewBox="0 0 460 210" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ca-a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 编码器输出（源词向量），上方一排 -->
  <text x="230" y="28" text-anchor="middle" font-size="12" fill="var(--dia-blue)">编码器输出（K, V）</text>
  <rect x="70"  y="42" width="60" height="30" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-blue)" stroke-width="1.5"/>
  <text x="100" y="62" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">I</text>
  <rect x="200" y="42" width="60" height="30" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-blue)" stroke-width="1.5"/>
  <text x="230" y="62" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">love</text>
  <rect x="330" y="42" width="60" height="30" rx="4" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="360" y="62" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">cats</text>
  <!-- 解码器当前查询，下方 -->
  <rect x="200" y="150" width="60" height="30" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <text x="230" y="170" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">猫 (Q)</text>
  <text x="230" y="200" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">解码器正在写的位置</text>
  <!-- 注意力连线：粗细表示权重，最粗的连到 cats -->
  <line x1="225" y1="150" x2="105" y2="74" stroke="var(--dia-stroke-tertiary)" stroke-width="1" marker-end="url(#ca-a)"/>
  <line x1="230" y1="150" x2="230" y2="74" stroke="var(--dia-stroke-tertiary)" stroke-width="1" marker-end="url(#ca-a)"/>
  <line x1="235" y1="150" x2="355" y2="74" stroke="var(--dia-accent)" stroke-width="3" marker-end="url(#ca-a)"/>
  <text x="408" y="110" text-anchor="middle" font-size="11" fill="var(--dia-accent)">权重最大</text>
</svg>
</div>
<p class="figure-caption">图 4.3-2：写"猫"时，解码器的查询去问源句各词，注意力自动把最大权重落在 "cats" 上——这就是模型自学出来的对齐。</p>

## 解码怎么解：贪心一步到底，还是集束搜索多看几步

到这里，模型已经能对每个位置输出"下一个词的概率分布"了。可问题是，**一句话有几十个词，每个位置都有几万种候选，组合起来是天文数字，我们不可能把所有可能的句子都生成出来再挑最好的。** 那到底该怎么把这条最优路径"走"出来？这就引出了**解码策略（decoding strategy）**。

最简单的办法叫**贪心解码（greedy decoding）**：每一步都只挑当前概率最高的那个词，挑完接着往下走，绝不回头。它快、好实现，但有个明显的毛病——**当下最优不等于全局最优**。第一步贪心选了个看着不错的词，可能把后面的路全堵死了，最后整句话反而别扭。这就好比下棋只看眼前一步，容易走进死胡同。

为了不那么短视，人们用**集束搜索（beam search）**：每一步不只保留 1 个候选，而是同时保留概率最高的 $k$ 个"半成品句子"（$k$ 叫**束宽**，beam width），让它们各自往下生长；下一步再从所有延伸结果里，重新挑出总分最高的 $k$ 条留着。这样模型相当于"多留了几手棋"，不会因为一步贪心而满盘皆输。

这里要点一句关键：beam search 比较的不是单步概率，而是**整句话的对数概率之和**。我们之所以取对数再相加，而不是把概率直接连乘，是因为几十个小于 1 的概率连乘会迅速下溢到 0，计算机扛不住；取对数把"连乘"变成"连加"，既稳定又等价：

$$
\text{score}(y_{1:T})=\sum_{t=1}^{T}\log P(y_t \mid y_{<t},\,x).
$$

式子里 $x$ 是源句，$y_{<t}$ 是已经生成的部分，整条句子的得分就是每一步对数概率累加起来。束宽 $k$ 越大，搜得越广、质量通常越好，但算得也越慢——$k=1$ 时，集束搜索就退化成了贪心。实践中翻译常取 $k=4$ 到 $10$ 左右，是质量和速度的一个折中。

!!! example "例：束宽为 2 的一步搜索"
    假设要生成两个词，词表只有 {我, 你, 爱}。第一步模型给 "我" 0.6、"你" 0.3、"爱" 0.1。束宽 $k=2$，于是保留 "我"（$\log 0.6$）和 "你"（$\log 0.3$）两条。第二步分别从这两条往下延伸、算各自的累计得分，再从所有候选里挑总分最高的 2 条继续。你会发现，即便 "你" 第一步分低，只要它第二步接的词足够好，仍有机会笑到最后——这正是贪心做不到的"翻盘"。

顺带提一句，如果你想要的不是"最可能"而是"有创意、有多样性"的输出（比如写故事、闲聊），那 beam search 反而会让结果变得保守、重复，这时人们会改用**采样类策略**（如 top-k、top-p 采样）。这部分在 [4.4 预训练大模型](3-4-4-pretrained-llms.md) 里会再展开，这里先记住一句话：**追求"对"用 beam，追求"活"用采样。**

## 动手搭一个最小的 seq2seq

道理讲完了，我们动手把它跑起来。好消息是，PyTorch 内置了 `nn.Transformer`，编码器、解码器、交叉注意力、掩码它都帮你封装好了，我们只需把"词嵌入 + 位置 + 输出投影"这层壳套上去。先把模型骨架写出来：

```python
import torch
import torch.nn as nn

class Seq2SeqTransformer(nn.Module):
    """一个最小可跑的 encoder-decoder 翻译/摘要骨架。"""
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8,
                 num_enc=3, num_dec=3, max_len=128):
        super().__init__()
        # 源、目标各自的词嵌入：把词 id 变成向量
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        # 位置编码：Transformer 本身不知道词序，得显式告诉它每个词排第几
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        # 核心：内部已含编码器、解码器、交叉注意力和各种掩码逻辑
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            batch_first=True,
        )
        # 把解码器输出投影回词表大小，得到每个词的打分（logits）
        self.out = nn.Linear(d_model, tgt_vocab)

    def encode_inputs(self, ids, emb):
        x = emb(ids) + self.pos[:, :ids.size(1)]   # 词义 + 位置
        return x

    def forward(self, src_ids, tgt_ids):
        src = self.encode_inputs(src_ids, self.src_emb)
        tgt = self.encode_inputs(tgt_ids, self.tgt_emb)
        # 给解码器自注意力生成"只能看左边"的因果掩码，防止偷看未来
        T = tgt_ids.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        h = self.transformer(src, tgt, tgt_mask=causal)   # 交叉注意力在内部完成
        return self.out(h)                                 # (batch, T, tgt_vocab)
```

这里有两处最该留意。**第一，那个 `causal` 掩码**：它是解码器自注意力"只看左边、不偷看未来"的关键，少了它，训练时模型就会作弊（直接抄到答案），一上线生成就崩。**第二，交叉注意力被藏在了 `self.transformer(src, tgt, ...)` 这一步里**——你把源和目标都传进去，PyTorch 内部就自动让解码器去对编码器输出做交叉注意力，不需要我们手写。

模型有了，训练目标其实就是上一节的语言建模换了个条件：**给定源句，逐位置预测目标句的下一个词**，用的还是交叉熵损失。注意一个经典做法叫**教师强制（teacher forcing）**——训练时我们不拿模型自己上一步的预测当输入，而是直接喂真实的目标词，这样训练更稳更快：

```python
import torch.nn.functional as F

model = Seq2SeqTransformer(src_vocab=10000, tgt_vocab=10000)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

src = torch.randint(0, 10000, (16, 20))     # 16 句源文，各 20 词（演示用随机数）
tgt = torch.randint(0, 10000, (16, 21))     # 目标句多一位，含起始符 <bos>

# 教师强制：用 tgt[:, :-1] 当输入，预测 tgt[:, 1:]（整体右移一位）
logits = model(src, tgt[:, :-1])            # (16, 20, 10000)
loss = F.cross_entropy(logits.reshape(-1, 10000), tgt[:, 1:].reshape(-1))
loss.backward(); opt.step()
print(loss.item())   # 随机权重下约等于 ln(10000) ≈ 9.2，训练后会一路降下来
```

那个 `tgt[:, :-1]` 当输入、`tgt[:, 1:]` 当标签的"错位一格"，正是把整段目标句变成了一连串"预测下一个词"的小任务。最后的损失值在随机初始化时约等于 $\ln(\text{词表大小})\approx 9.2$，这是个很好的"健全性检查"——如果一开始就远小于它，多半是掩码漏了、模型偷看了答案。

真正生成时没有"标准答案"可喂，得改成**自回归（autoregressive）**：从起始符 `<bos>` 出发，每生成一个词就把它接回输入，再生成下一个，直到吐出结束符 `<eos>`。下面是最朴素的贪心版：

```python
@torch.no_grad()
def greedy_decode(model, src, bos_id, eos_id, max_len=50):
    model.eval()
    ys = torch.tensor([[bos_id]])                 # 从 <bos> 开始
    for _ in range(max_len):
        logits = model(src, ys)                   # 看着源句 + 已生成的部分
        next_id = logits[:, -1].argmax(-1, keepdim=True)  # 取最后一位概率最大的词
        ys = torch.cat([ys, next_id], dim=1)      # 接回输入，继续往下生成
        if next_id.item() == eos_id:              # 撞到结束符就收工
            break
    return ys
```

把这段里的 `argmax`（贪心）换成"维护 $k$ 条候选、按累计对数概率剪枝"，就是集束搜索了。原理我们上面已经讲透，工程上 HuggingFace 的 `model.generate(..., num_beams=k)` 一行就能调用，竞赛里直接用现成的即可。

## 容易踩的坑

- **忘了给解码器加因果掩码**：这是 seq2seq 最致命的 bug。训练时一切正常、损失降得飞快，可一上线生成就胡言乱语——因为训练时模型偷看了未来的答案，根本没学会"靠左边预测右边"。
- **训练用教师强制、推理用自回归，两者不一致**：训练时每一步都喂真实词，推理时却只能喂自己上一步的预测。一旦中途预测错一个词，错误会顺着自回归滚雪球般放大，这叫**曝光偏差（exposure bias）**。短文本影响小，长文本要警惕。
- **把交叉注意力也加了因果掩码**：因为源句是"已知全文"，编码器输出就该被解码器**完整**看到。只有解码器的**自**注意力才需要因果掩码，交叉注意力千万别也给挡住。
- **beam search 不做长度归一化**：句子越长、对数概率累加得越负，beam search 会天然偏爱短句子，导致翻译"少词漏译"。常见补救是把总分除以长度（length normalization）。
- **位置编码长度不够**：测试句子比训练时见过的最长句还长，位置编码会越界。设 `max_len` 时留足余量。

## 它在后面会怎么用到

- 这套架构是 **T5、BART、mBART** 等一大批"编码器-解码器型"预训练模型的骨架，也是机器翻译、文本摘要的主流方案，详见 [4.4 预训练大模型](3-4-4-pretrained-llms.md)。那里还会对比"纯解码器"的 GPT 路线，帮你想清楚什么任务该用哪种。
- 交叉注意力是它的灵魂，而注意力的底层机制在 [2.1 注意力机制](3-2-1-attention.md) 和 [2.2 Transformer 架构](3-2-2-transformer.md) 里有完整推导，没吃透的同学务必回去补。
- 解码器逐词预测、用交叉熵训练，本质上就是带条件的语言建模，和 [4.2 语言建模](3-4-2-language-modeling.md) 一脉相承。
- 把"图像"当作源、用编码器读图、用解码器写文字，就成了**图像描述（image captioning）**；语音识别（语音转文字）同样是 seq2seq 的一个实例，可参考 [4.5 音频模型](3-4-5-audio.md)。
- 综合性的 NLP 真题（翻译、摘要、生成）会反复用到本节内容，刷题见 [NLP 类真题](../ch4-camp/4-2-2-nlp-problems.md)。

## 练习

??? note "基础练习"
    1. 用自己的话说清：编码器对源句用的是"双向注意力"，而解码器对自己生成的内容用的是"带掩码的因果注意力"。为什么这两处的注意力非得不一样？（提示：从"读理解"和"边写边生成"两种场景的差别去想。）
    2. 在交叉注意力的公式里，$Q$、$K$、$V$ 分别来自编码器还是解码器？把它和普通自注意力做个对照，写清两者的区别。
    3. 束宽 $k=1$ 时，集束搜索等价于哪种解码？为什么 $k$ 越大通常质量越好、却越慢？

??? note "进阶练习"
    1. 把上面的 `greedy_decode` 改写成 `beam_search(model, src, k, ...)`：维护 $k$ 条候选序列及其累计对数概率，每步扩展后只保留总分最高的 $k$ 条，遇到 `<eos>` 的候选移入完成列表。最后比较 $k=1$ 和 $k=5$ 输出有什么不同。
    2. 给你的 beam search 加上**长度归一化**：把每条候选的总分除以其长度的 $\alpha$ 次方（$\alpha$ 取 $0.6$ 左右）。观察它是否缓解了"偏爱短句"的问题。
    3. 取一个训练好的翻译模型，把某次生成时交叉注意力的权重矩阵取出来、画成热力图，看看目标词和源词之间是否真的形成了对角线状的对齐。

## 小结

- **一句话**：encoder-decoder 就是"一半专门读懂源句、一半专门重写答案"的 seq2seq 架构，二者靠**交叉注意力**连通——解码器用自己的查询去问编码器要相关信息。
- 解码器靠**因果掩码的自注意力**保证不偷看未来，靠**对编码器输出的交叉注意力**保证忠于原文；训练用教师强制 + 交叉熵，推理用自回归。
- 把答案"解"出来有两条路：**贪心**快而短视，**集束搜索**多留几手、比的是整句的对数概率之和；追求"对"用 beam，追求"活"改用采样。

想把直觉和工程都打牢，强烈推荐 Jay Alammar 的图解博客 [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)，它用动画把编码器、解码器和交叉注意力的数据流讲得一清二楚；想动手，HuggingFace 的 [translation 教程](https://huggingface.co/docs/transformers/tasks/translation) 用十几行就能微调出一个能用的翻译模型。原始论文则是那篇开创一切的 [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)。
