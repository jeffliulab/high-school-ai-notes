# 4.4 预训练大模型

> **难度** ⭐⭐⭐⭐☆ · **前置**：[4.2 语言建模](3-4-2-language-modeling.md)、[2.2 Transformer 架构](3-2-2-transformer.md)、[1.3 模型微调](3-1-3-finetuning.md)

!!! abstract "读完这一节，你会"
    - 说清 BERT 和 GPT 的根本区别：一个靠"完形填空"双向看上下文，一个靠"猜下一个词"只能从左往右看
    - 知道"预训练 + 微调"这套打法为什么这么能打，并能用几行代码把 BERT 微调成一个文本分类器或 NER 模型
    - 理解提示学习（prompting）是怎么回事，会区分零样本、少样本、指令微调三种用法
    - 在赛题里看到"用 BERT/GPT 做某任务"时，能立刻判断该选哪种模型、套哪种范式

## 为什么"预训练"会成为大模型时代的分水岭

在前面几节里，我们已经能从零训练一个语言模型，也搭过 Transformer。但你可能隐隐有个疑问：每来一个新任务，比如判断影评是好评还是差评、从句子里抽出人名地名，难道都要从头训练一个新模型吗？如果每个任务都要几万条标注数据，那大多数任务根本玩不起。

预训练（pre-training）就是为了解决这件事而生的。它的思路朴素得惊人：**先让模型在海量没人标注过的文本上做一个"自己跟自己玩"的任务，把语言的规律学个透；然后再拿这个学好底子的模型，针对你的具体任务用少量数据稍微调一调。** 前一步叫预训练，后一步叫微调（fine-tuning）。

为什么这么做有效？打个比方。预训练就像让一个孩子先把母语读写练熟——读了几百万本书，他对词与词怎么搭配、句子怎么连贯、世界大致是什么样，已经有了扎实的语感。这时候你再教他"判断一句话是不是讽刺"，他几十个例子就学会了，因为底层的语言能力早就具备，只差临门一脚。**预训练学的是通用的"语言常识"，微调补的是任务专属的那一点点知识。**

而这套打法之所以能在 2018 年之后彻底改变 NLP，关键在于一个特别巧妙的设计：**预训练任务不需要任何人工标注。** 文本本身就自带标签——把一个词盖住让模型猜，答案就是被盖住的那个词。这意味着互联网上海量的免费文本，全都能拿来当训练数据。下面这张图把整个流程画了出来：

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="pa" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 预训练阶段 -->
  <rect x="20" y="60" width="140" height="62" rx="6" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <text x="90" y="86" text-anchor="middle" font-size="13" fill="var(--dia-blue)">预训练</text>
  <text x="90" y="105" text-anchor="middle" font-size="10.5" fill="var(--dia-stroke-soft)">海量无标注文本</text>
  <text x="90" y="44" text-anchor="middle" font-size="10.5" fill="var(--dia-stroke-tertiary)">慢 · 贵 · 一次</text>
  <!-- 箭头 -->
  <line x1="160" y1="91" x2="255" y2="91" stroke="var(--dia-accent)" stroke-width="2" marker-end="url(#pa)"/>
  <text x="207" y="82" text-anchor="middle" font-size="10.5" fill="var(--dia-accent)">通用语言能力</text>
  <!-- 微调阶段 -->
  <rect x="260" y="60" width="140" height="62" rx="6" fill="none" stroke="var(--dia-green)" stroke-width="2"/>
  <text x="330" y="86" text-anchor="middle" font-size="13" fill="var(--dia-green)">微调</text>
  <text x="330" y="105" text-anchor="middle" font-size="10.5" fill="var(--dia-stroke-soft)">少量标注数据</text>
  <text x="330" y="44" text-anchor="middle" font-size="10.5" fill="var(--dia-stroke-tertiary)">快 · 便宜 · 每个任务</text>
</svg>
</div>
<p class="figure-caption">图 4.4-1：预训练只做一次（昂贵），微调可以针对无数下游任务反复进行（廉价）。这就是"先打底子，再补专长"。</p>

接下来我们看两条最有代表性的技术路线：BERT 和 GPT。它们都是 Transformer，但因为预训练任务不同，性格也截然不同。

## BERT：靠"完形填空"学会双向理解

我们先从 BERT 说起。它的全名是 Bidirectional Encoder Representations from Transformers，听着唬人，但拆开看其实就两个关键词：**双向（bidirectional）** 和 **编码器（encoder）**。

先说它在玩什么游戏。BERT 的预训练任务叫**掩码语言模型（Masked Language Model，MLM）**，说白了就是**完形填空**：把句子里随机大约 15% 的词用一个特殊标记 `[MASK]` 盖住，让模型根据剩下的上下文猜出被盖住的是什么。比如：

> 输入：今天 天气 真 `[MASK]`，我们 去 公园 散步。
> 模型要猜：好

这里有个关键点值得停下来想一想。为了猜出 `[MASK]`，模型既要看它**左边**的"今天天气真"，也要看它**右边**的"我们去公园散步"——左右两边的信息它都能用上。这就是"双向"的含义。**正因为完形填空允许模型同时看到一个词的左右两侧，BERT 学到的每个词的表示，都融合了完整的上下文。** 这种双向理解能力，让它在"读懂一句话"这类任务上特别强。

!!! note "为什么 GPT 做不到双向？"
    你可能会问，那为什么不让所有模型都双向？因为 GPT 玩的是另一个游戏——"猜下一个词"。如果允许它看到右边，那就等于把答案直接透题给它了，它会作弊。所以"能否双向"不是谁更先进，而是由预训练任务决定的，我们马上就讲。

BERT 还有个常被提到的小设计：每个输入句子前面会加一个特殊的 `[CLS]` 标记。你可以把它理解成一个"句子摘要位"——经过多层 Transformer 之后，`[CLS]` 这个位置的输出向量，被训练成能代表整句话的含义。后面做分类时，我们就专门拿它来用。

## GPT：靠"猜下一个词"学会生成

再看 GPT，它代表的是另一条路线。GPT 是 Generative Pre-trained Transformer 的缩写，核心是**因果语言模型（Causal Language Model，CLM）**，也叫自回归语言模型。

它的游戏更简单：**给定前面所有的词，预测下一个词。** 这正是我们在 [4.2 语言建模](3-4-2-language-modeling.md) 里讲过的语言建模任务。形式化地写出来，对一句话 $w_1, w_2, \dots, w_n$，GPT 要最大化的是：

$$
P(w_1,\dots,w_n)=\prod_{t=1}^{n} P(w_t \mid w_1,\dots,w_{t-1}).
$$

注意每一项 $P(w_t \mid w_1,\dots,w_{t-1})$ 里，条件**只有 $w_t$ 之前的词**——它永远只能往左看，看不到右边。这就是"因果"或"单向"的意思：未来还没发生，不能偷看。在 Transformer 实现上，这是靠一个**因果掩码（causal mask）** 做到的，它把注意力矩阵里"看向未来"的那些位置全部屏蔽掉。

那为什么 GPT 这种"只能往左看"的模型反而成了今天 ChatGPT 这类产品的基础？因为**它天生就会生成**。猜下一个词、把猜出来的词接到句子末尾、再猜下一个……这样一个词一个词地往后接，就能写出一整段连贯的文字。BERT 擅长"读懂"，GPT 擅长"写出"，这是它们最根本的分工。

下面这张表把两者放在一起对比，方便你一眼分清：

| 对比项 | BERT（编码器） | GPT（解码器） |
|---|---|---|
| 预训练任务 | 掩码语言模型（完形填空） | 因果语言模型（猜下一个词） |
| 能看到的上下文 | 双向（左右都看） | 单向（只看左边） |
| 最擅长 | 理解类任务（分类、NER、问答） | 生成类任务（写作、对话、续写） |
| 怎么用在下游任务 | 接个小分类头微调 | 微调，或直接用提示（prompting） |
| 典型代表 | BERT、RoBERTa、DeBERTa | GPT 系列、LLaMA、Qwen |

## 把 BERT 微调成分类器：原理与代码

讲了这么多，我们动手把 BERT 用起来。最经典的用法，是把它微调成一个文本分类器，比如判断一句话的情感是正面还是负面。

原理很直接，分三步想清楚：**第一步**，把句子喂进预训练好的 BERT，它会输出每个位置的向量，其中 `[CLS]` 位置的向量代表整句话；**第二步**，在这个 `[CLS]` 向量上面接一个很小的全连接层（叫"分类头"），把它映射成几个类别的分数；**第三步**，用你的标注数据训练，让梯度同时微调 BERT 的参数和这个分类头。因为 BERT 底子已经很好，这一步通常几个 epoch、几千条数据就够了。

我们先看用 Hugging Face 的 `transformers` 库怎么写，这是竞赛和工程里的标准做法：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. 加载预训练模型和它配套的分词器（tokenizer）
#    num_labels=2 表示二分类，库会自动帮我们在 [CLS] 上接一个分类头
name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

# 2. 把一句话变成模型能吃的数字（token id），并自动加上 [CLS]、[SEP]
text = "this movie is absolutely wonderful"
inputs = tokenizer(text, return_tensors="pt")   # 返回 PyTorch 张量

# 3. 前向计算，拿到两个类别的原始分数（logits）
with torch.no_grad():
    logits = model(**inputs).logits
print(logits)                       # 形如 tensor([[-0.3, 0.5]])
print(logits.argmax(dim=-1))        # 取分数最大的那个类别作为预测
```

上面只是"用"，还没"训"。真正微调时，我们要把多条数据组成 batch，算损失再反向传播。下面这段把训练的核心循环写出来，你会发现它和你之前训练任何 PyTorch 模型几乎一模一样——这正是 `transformers` 的好处，预训练模型用起来就像一个普通的 `nn.Module`：

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)   # 微调用很小的学习率！
model.train()

for batch_texts, batch_labels in dataloader:     # dataloader 你自己按数据集构造
    enc = tokenizer(batch_texts, padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
    labels = torch.tensor(batch_labels)

    outputs = model(**enc, labels=labels)         # 传了 labels，库会自动算交叉熵损失
    loss = outputs.loss

    loss.backward()                               # 反向传播
    optimizer.step()                              # 更新参数（含 BERT 和分类头）
    optimizer.zero_grad()
```

这里有个细节特别值得你记住：**微调用的学习率要比从零训练小得多**，常见的是 $2\times10^{-5}$ 这个量级。为什么？因为 BERT 的参数已经在预训练里调到了一个很好的状态，你只是想轻轻地"挪一挪"让它适应新任务，而不是把它的语言知识全推翻重学。学习率一旦太大，预训练学到的好底子很容易被一次大步长的更新"冲垮"，这种现象叫灾难性遗忘（catastrophic forgetting）。

## 用 BERT 做 NER：从"整句一个标签"到"每个词一个标签"

分类是给**整句话**一个标签。但有一类任务不一样——**命名实体识别（Named Entity Recognition，NER）**，它要给句子里**每一个词**都打一个标签，标出哪些词是人名、地名、机构名。比如：

> 句子：李 华 在 北京 大学 读书
> 标签：B-PER I-PER O B-ORG I-ORG O

这里的标签用了一套叫 **BIO** 的约定：`B-` 表示某类实体的开头（Begin），`I-` 表示实体的中间或延续（Inside），`O` 表示这个词不属于任何实体（Outside）。所以"李华"是一个人名（B-PER 接 I-PER），"北京大学"是一个机构名。

那 BERT 怎么做这件事？道理和分类几乎一样，只差一点：**分类是只用 `[CLS]` 一个向量，NER 则是用每个词各自的输出向量。** 每个词的向量都接同一个分类头，独立地预测它的 BIO 标签。换句话说，分类是"一句话 → 一个标签"，NER 是"每个词 → 一个标签"，模型主体没变，只是输出的粒度从整句变成了逐词。

代码上，只要把任务头从"序列分类"换成"逐 token 分类"即可：

```python
from transformers import AutoModelForTokenClassification

# 假设有 5 类标签：O, B-PER, I-PER, B-ORG, I-ORG
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=5)

enc = tokenizer("Li Hua studies at Peking University", return_tensors="pt")
logits = model(**enc).logits          # 形状 [1, 序列长度, 5]
preds = logits.argmax(dim=-1)         # 每个 token 各自取一个标签
print(preds)
```

注意输出 `logits` 的形状是 `[1, 序列长度, 5]`——比分类多了一个"序列长度"维度，正对应"每个词一个预测"。

## 提示学习：不微调也能让大模型干活

到这里，BERT 那套"接个头、微调一下"的范式你已经懂了。但 GPT 这类生成模型还带来了一种全新的、更省事的用法——**提示学习（prompting）**。它的颠覆之处在于：**很多时候，你根本不用训练，只要把任务用自然语言"说"给模型听，它就能照做。**

为什么生成模型能这样？因为它本质上是"根据前文猜后文"。那我们只要把前文设计成一个"提问"，模型续写出来的"后文"自然就是答案。比如你想做情感分类，可以这样写提示词（prompt）：

```text
判断下面这句影评的情感是"正面"还是"负面"。
影评：这部电影节奏拖沓，看得我昏昏欲睡。
情感：
```

模型读到这里，会顺着续写出"负面"。它并没有为这个任务专门训练过，纯靠预训练学到的语言理解能力就完成了。这种"一个例子都不给、直接让它做"的方式，叫**零样本（zero-shot）**。

如果任务比较难，模型一上来做不好，我们可以在提示里**先给它几个做好的例子**当示范，再问真正的问题。这叫**少样本（few-shot）**，也叫上下文学习（in-context learning）——神奇的是模型并没有更新任何参数，只是"看了几个例子"就学会了照葫芦画瓢：

```text
影评：演员演技炸裂，剧情扣人心弦。 情感：正面
影评：剧情老套，特效廉价。 情感：负面
影评：这部电影节奏拖沓，看得我昏昏欲睡。 情感：
```

那为什么今天的 ChatGPT 能那么听话地理解你各种花式指令？这背后还有一步叫**指令微调（instruction tuning）**：在预训练之后，再用大量"指令—回答"配对的数据微调模型，专门教它"听懂人话、按要求办事"。所以现代大模型其实是把三种东西叠在一起——预训练打底、指令微调学会服从、提示学习在使用时灵活调度。

!!! tip "微调还是提示，怎么选？"
    一个实用的判断：**数据多、任务固定、要追求极致精度**，就微调一个 BERT；**数据少、任务多变、想快速验证想法**，就用大模型 + 提示。竞赛里两种你都可能用到——分类/NER 这类有标注数据的题目，微调 BERT 往往更稳更准；而一些开放式、灵活的题，提示一个大模型反而又快又省事。

## 容易踩的坑

- **分词器和模型必须配套**：每个预训练模型都有它自己的分词器，`bert-base-uncased` 的分词器不能配 `roberta` 的模型。永远用 `AutoTokenizer.from_pretrained` 加载和模型**同名**的那个，否则 token id 对不上，结果全是乱的。
- **微调学习率别照搬从零训练的**：从零训练常用 $10^{-3}$，但微调 BERT 要用 $2\times10^{-5}$ 这种小值。学习率太大会触发灾难性遗忘，把预训练的好底子毁掉，表现为训练几步后效果不升反降。
- **`uncased` 与 `cased` 别混用**：`uncased` 模型会把文本全转小写，做 NER（人名地名常靠大写区分）时这会丢信息，应该用 `cased` 版本。
- **NER 里子词对齐容易错**：BERT 会把一个词拆成多个子词（subword），比如 "playing" → "play" + "##ing"。标签是按"词"给的，预测是按"子词"出的，对齐没处理好，准确率会莫名其妙地低。
- **别忘了 padding 和 truncation**：一个 batch 里句子长短不一，必须 `padding=True` 补齐、`truncation=True` 截断到 `max_length`，否则张量拼不成一个 batch。
- **`[CLS]` 不是天生就代表句意**：它能代表整句，是**微调过程**训出来的。直接拿没微调过的 `[CLS]` 向量当句向量，效果通常很一般。

## 它在后面会怎么用到

- 这一节的"预训练 + 微调"思想，正是 [1.3 模型微调](3-1-3-finetuning.md) 的核心；那里会讲 LoRA 等更省显存的高效微调方法，让你在单张 GPU 上也能调动大模型。
- BERT 输出的词向量，本质上是一种带上下文的嵌入表示，和 [1.2 嵌入表示](3-1-2-embeddings.md)、[4.1 文本分类与词嵌入](3-4-1-text-embeddings.md) 是一脉相承的——区别在于 BERT 的嵌入会随上下文变化。
- BERT 和 GPT 都建立在 [2.2 Transformer 架构](3-2-2-transformer.md) 和 [2.1 注意力机制](3-2-1-attention.md) 之上，BERT 用编码器、GPT 用解码器，掩码机制的不同正源于此。
- "用图像 + 文本一起预训练"的思路在 [3.4 视觉-文本编码器 CLIP](3-3-4-clip.md) 里会再次出现，那是把预训练范式推广到了多模态。
- NLP 类真题里会大量出现"微调 BERT 做分类/NER""用大模型做提示推理"的题型，到 [2.2 NLP 类真题](../ch4-camp/4-2-2-nlp-problems.md) 时我们会专门练。

## 练习

??? note "基础练习"
    1. 不查资料，用自己的话说清：为什么 BERT 能双向看上下文，而 GPT 只能从左往右看？把答案落到"预训练任务"上，而不是"模型更先进"。
    2. 给定句子"周 杰 伦 在 台北 开 演唱会"，按 BIO 约定，为每个词写出 NER 标签（人名 PER、地名 LOC）。
    3. 用 `transformers` 加载一个预训练情感分类模型（如 `distilbert-base-uncased-finetuned-sst-2-english`），对三句你自己写的影评做预测，看它判断得对不对。

??? note "进阶练习"
    1. 找一个小的情感分类数据集（如 SST-2 的一个子集），完整地把 `bert-base-uncased` 微调一遍，记录训练前（零样本）和训练后的准确率，亲眼看看微调带来了多大提升。
    2. 同一个情感分类任务，分别用两种方式做：①微调 BERT；②给一个生成式大模型写零样本/少样本提示。比较两者的准确率、所需数据量和耗时，写一段你对"何时该微调、何时该提示"的体会。
    3. 在 NER 任务里，亲手处理一次"子词与标签对齐"：把被拆成多个子词的词，只让第一个子词参与损失计算，其余子词标为 -100（忽略）。跑通后对比"不做对齐"的版本，看 F1 差多少。

## 小结

- 一句话记住核心：**预训练学通用语感（不用标注），微调补任务专长（少量标注）**——这套打法是大模型时代的根基。
- **BERT 靠完形填空学会双向理解，擅长读；GPT 靠猜下一个词学会单向生成，擅长写。** 能否双向，由预训练任务决定，不分高下。
- 微调 BERT 就是"接个小头 + 用很小的学习率训几轮"：分类用 `[CLS]` 向量，NER 用每个词的向量。
- 提示学习让你不训练也能调用大模型，分零样本、少样本，叠上指令微调就成了今天会"听话"的 ChatGPT。

想把这套范式的来龙去脉看得更透，强烈推荐读 Jay Alammar 的图解博客 [*The Illustrated BERT*](https://jalammar.github.io/illustrated-bert/) 和 [*The Illustrated GPT-2*](https://jalammar.github.io/illustrated-gpt2/)，它们用大量动图把"掩码""因果掩码"讲得无比直观。想动手，Hugging Face 官方的 [NLP Course](https://huggingface.co/learn/nlp-course) 是免费且极好的实战教程。
