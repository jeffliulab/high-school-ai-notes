# 2.2 NLP 类真题

> **难度** ⭐⭐⭐⭐☆ · **前置**：[4.1 文本分类与词嵌入](../ch3-round2/3-4-1-text-embeddings.md)、[4.4 预训练大模型](../ch3-round2/3-4-4-pretrained-llms.md)、[1.3 限时解题策略](4-1-3-time-strategy.md)

!!! abstract "读完这一节，你会"
    - 拿到一道陌生的 NLP 题，能在五分钟内判断它属于哪一类（分类 / 生成 / 跨模态），并选对配套套路
    - 掌握一套"读题 → 搭基线 → 看错例 → 迭代"的标准流水线，把时间花在刀刃上
    - 通过三个自编练习示例，把文本分类、文本生成、图文跨模态各走一遍完整流程

到了集训营这一关，知识点你已经在 Round 1、Round 2 里学得差不多了。这一节不再教新模型，而是教你**怎么把已经会的东西，在限时、陌生数据集的真题压力下稳稳地用出来**。我们先把 NLP 题的常见类型理清楚，再给你一套可复用的解题流水线，最后用三个自编例子演练一遍。

!!! warning "关于真题的重要说明"
    本节出现的所有题目都是**自编练习示例，并非 IOAI / IAIO 官方真题**。我手上并不掌握官方真题原文，所以**绝不会**把任何题目伪装成"某年官方真题"给你。真正的官方题目，请以下面这两个官方来源为准（链接待核实补全）：

    - IOAI 官方：<https://ioai-official.org>
    - IAIO 官方：<https://iaio-official.org>

    历年真题的整理与索引，另见 [1.3 历年真题库](../ch5-champion/5-1-3-ioai-past.md) 与 [A2 真题与数据集](../appendix/a2-datasets.md)。

## 先学会"认题"：NLP 真题就这么几类

很多同学一拿到 NLP 题就慌，其实大可不必。因为不管题面怎么包装，竞赛里的文本任务翻来覆去就是三大类，认出类型，你就知道该掏出哪套工具了。

第一类是**文本分类（text classification）**：给你一段文字，让你判断它属于哪一类——是正面还是负面评论、是垃圾邮件还是正常邮件、是哪种语言。这类题的输出是一个标签，评分一般看准确率（accuracy）或 F1 分数。

第二类是**文本生成（text generation）**：让模型自己写出一段文字——翻译、摘要、问答、续写都算。它的输出是一串字，评分要用专门的指标，比如机器翻译常用的 BLEU、摘要常用的 ROUGE。

第三类是**跨模态（multimodal）**：文字和别的东西（最常见是图像）混在一起，比如"看图说话"、"根据问题在图里找答案"。这类题往往要请出 [3.4 视觉-文本编码器 CLIP](../ch3-round2/3-3-4-clip.md) 这样能同时理解图和文的模型。

下面这张表帮你拿到题就对号入座：

| 题型 | 典型任务 | 输出形式 | 常用评分 | 趁手的模型 |
|---|---|---|---|---|
| 文本分类 | 情感分析、垃圾邮件识别 | 一个标签 | Accuracy / F1 | 词嵌入 + 线性层，或微调 BERT |
| 文本生成 | 翻译、摘要、问答 | 一串文字 | BLEU / ROUGE | Encoder-Decoder、预训练 LLM |
| 跨模态 | 看图问答、图文检索 | 标签或文字 | Accuracy / 检索命中率 | CLIP、多模态大模型 |

## 一套通用流水线：别一上来就调大模型

认出题型之后，先别急着把最大的模型搬出来。竞赛是限时的，最稳的打法是**先用最简单的方法跑通一个能交卷的版本，再逐步加码**。这套流水线我建议你背下来：

1. **读懂数据**：先 `head` 看几条样本，数一数有几类、每类多少条、文本多长、有没有缺失和乱码。一上来就动手训练，往往是错例满天飞的开始。
2. **搭一个最笨的基线（baseline）**：分类题就上"词袋 + 逻辑回归"，生成题就上一个小的预训练模型零样本跑一遍。基线的意义是给你一个分数底线——后面任何改动，都要和它比一比是不是真的变好了。
3. **看错例，别只看分数**：把模型做错的样本挑出来读一读。你常会发现错误是有规律的（比如全错在长句上、或全错在某一类上），这比盯着一个准确率数字干瞪眼有用得多（详见 [1.1 误差分析与调试](4-1-1-error-analysis.md)）。
4. **针对性迭代**：根据错例去改——是数据不均衡？那就加权重；是模型太弱？那就换微调；是过拟合？那就加正则。每改一处，记一笔实验（详见 [1.2 调参与实验管理](4-1-2-experiment-management.md)）。

记住一句话：**baseline first（先有基线）**。竞赛里最痛的不是分数不够高，而是折腾两小时大模型却连一次有效提交都没产出。

## 自编示例一：英文影评情感分类

我们用第一个例子把"分类流水线"走一遍。**再强调一次，这是自编练习，不是官方真题。**

!!! example "练习示例 1（自编，非官方真题）"
    给你两万条英文影评，每条带一个标签：`1` 表示好评、`0` 表示差评。请训练一个模型，在测试集上把好评差评分开，评分用准确率。

按流水线，第一步先搭最笨的基线。文本分类里最经典的基线是**词频—逆文档频率（TF-IDF）**把文字转成向量，再接一个逻辑回归。它简单、快、还经常出乎意料地强：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# texts_train 是影评列表，y_train 是 0/1 标签
clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=20000),  # 取一元词+二元词组
    LogisticRegression(max_iter=1000),
)
clf.fit(texts_train, y_train)
print("baseline acc:", clf.score(texts_val, y_val))   # 先拿到一个分数底线
```

这个基线往往就能到 85% 上下。**它的价值在于：现在你有一个能交卷的版本了，剩下的时间都是在锦上添花，而不是裸奔。**

如果分数还不够，再上微调（fine-tuning）一个预训练模型。这一步前面 [1.3 模型微调](../ch3-round2/3-1-3-finetuning.md) 讲过原理，这里只看怎么用：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
# 把文本切成 token，截断到 256 长度，再喂给 Trainer 训练（此处略去训练循环）
enc = tok(texts_train, truncation=True, max_length=256, padding=True)
```

微调一般能把准确率再抬几个点。但注意：**它慢得多、还容易过拟合**。所以正确顺序永远是基线在前、微调在后——确认基线确实顶不住了，再花这份时间。

## 自编示例二：让模型写一句话摘要

第二个例子换成生成题，体会一下它和分类的不同。

!!! example "练习示例 2（自编，非官方真题）"
    给你一批新闻段落，请为每段生成一句话摘要，评分用 ROUGE-L（衡量生成摘要和参考摘要的重合程度）。

生成题的基线，最省事的做法是直接请一个预训练的 Encoder-Decoder 模型来零样本（zero-shot）跑一遍——不训练，先看看现成模型能到几分：

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
text = "..."   # 一段新闻
out = summarizer(text, max_length=40, min_length=10, do_sample=False)
print(out[0]["summary_text"])
```

评分这一步是生成题最容易丢分的地方，因为你不能像分类那样"对一个数就算对"。生成的句子和参考答案字面不同、意思相近也算好。所以要用 ROUGE 这类**看重合度**的指标：

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
score = scorer.score(reference, generated)        # reference 是参考摘要
print(score["rougeL"].fmeasure)                   # 0~1，越高越好
```

拿到零样本分数后，再决定要不要微调。生成题的迭代方向和分类不太一样：除了换更强的模型，**解码策略（decoding）**也很关键——比如把贪心解码换成束搜索（beam search），常常能白捡几分。这背后的原理见 [4.3 Encoder-Decoder](../ch3-round2/3-4-3-encoder-decoder.md)。

## 自编示例三：图文跨模态的看图问答

第三个例子把文字和图像搅到一起，演练跨模态。

!!! example "练习示例 3（自编，非官方真题）"
    给你一批"图片 + 一个问题"，问题形如"图里的猫是什么颜色？"，答案是若干候选词中的一个。请判断每张图对应的正确答案，评分用准确率。

跨模态题的关键，是找一个**能把图和文放进同一个空间比较**的模型。CLIP 就是为此而生的：它能分别给图片和文字算出向量，然后看哪段文字的向量和图片最贴近。于是这道题可以巧妙地转成"检索"——把每个候选答案拼成一句话，看哪句和图片最配：

```python
import torch, clip
from PIL import Image

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("cat.jpg")).unsqueeze(0)

# 把每个候选答案套进问题模板，变成一句完整的话
candidates = ["a photo of a black cat", "a photo of a white cat", "a photo of an orange cat"]
text = clip.tokenize(candidates)

with torch.no_grad():
    logits_per_image, _ = model(image, text)        # 图片和每句话的相似度
    probs = logits_per_image.softmax(dim=-1)        # 转成概率
print("最可能的答案：", candidates[probs.argmax()])
```

你看，**我们没有训练任何东西**，只是借 CLIP 已经学好的"图文对齐"能力，把问答硬生生改造成了相似度比较。这正是跨模态题最值钱的解题直觉：**把陌生任务，翻译成一个你已经有现成模型能解的任务。** CLIP 的原理详见 [3.4 视觉-文本编码器 CLIP](../ch3-round2/3-3-4-clip.md)。

## 容易踩的坑

- **拿到题直接上最大模型**：限时赛里这是头号失误。先用 TF-IDF 或零样本基线拿到能交卷的分数，再加码——别让自己裸奔到交卷前一刻。
- **分类指标看错**：数据不均衡时（比如 95% 是好评），光看准确率会骗人——一个全猜好评的模型也有 95%。这种时候要看 F1（详见 [1.4 分类评估指标](../ch2-round1/2-1-4-classification-metrics.md)）。
- **生成题忘了解码策略**：同一个模型，贪心解码和束搜索能差出好几分。改完模型分数不动时，先回头看看解码参数。
- **token 截断悄悄丢信息**：`max_length` 设太短，长文本的后半截直接没了，模型自然学不好。先统计一下文本长度分布再定这个值。
- **数据泄漏**：TF-IDF 的词表、标准化的统计量，都只能用训练集来拟合，再套到验证/测试集上，否则就是偷看答案。
- **把自编题当官方真题背**：本节所有题都是自编的，做来练手法即可；真要刷官方题，认准上面给的官方链接。

## 它在后面会怎么用到

这一节的解题套路，是为后面几场硬仗做准备的：

- **团队赛**里 NLP 子任务往往要你在几小时内从零搭出可用方案，这套"基线先行 + 看错例迭代"的流水线就是你的主心骨（见 [3. 团队挑战 Team Challenge](4-3-team-challenge.md)）。
- **选拔测试**会限时考综合能力，认题快、基线稳，能帮你把分数下限焊死（见 [4. 选拔测试准备](4-4-selection-tests.md)）。
- 到了国际赛，全真模拟时你会反复用到这里的判断流程（见 [3.1 全真模拟](../ch5-champion/5-3-1-full-mock.md)）。
- 与之并列的另外两类真题，可对照着看 [2.1 CV 类真题](4-2-1-cv-problems.md) 和 [2.3 信号·优化类真题](4-2-3-signal-opt-problems.md)，三类题的流水线骨架其实是相通的。

## 练习

??? note "基础练习"
    1. 找一份带标签的英文情感数据集（自编或公开均可），照示例一搭出 TF-IDF + 逻辑回归基线，记下准确率。再把 `ngram_range` 从 `(1,1)` 改成 `(1,2)`，看看分数有没有变化。
    2. 用 `pipeline("summarization")` 对三段你自己挑的文字做零样本摘要，再人工读一读：哪段摘得好、哪段摘得离谱？尝试用一句话说清它在什么情况下容易出错。

??? note "进阶练习"
    1. 把示例一的逻辑回归基线和微调 DistilBERT 都跑出来，列一张表对比两者的准确率、训练耗时和过拟合程度，写一句话总结"什么时候值得花时间微调"。
    2. 照示例三的思路，自己设计一个把"分类任务改造成 CLIP 检索"的方案：选一个小图像分类集，把每个类别名套进 `"a photo of a {label}"` 模板，零样本跑一遍准确率，并和一个普通的图像分类基线比一比。

## 小结

NLP 真题万变不离三类——分类、生成、跨模态；认对类型、选对指标、**先搭基线再迭代**，就赢了一大半。三个自编示例分别演示了"TF-IDF/微调做分类"、"零样本+ROUGE 做生成"、"借 CLIP 把问答改成检索"这三种最实用的手法，但它们都只是练手用的，**真正的官方真题以 [IOAI 官方](https://ioai-official.org) 与 [IAIO 官方](https://iaio-official.org)（链接待核实补全）为准**。想系统补 NLP 基础，回头看 [4.2 语言建模](../ch3-round2/3-4-2-language-modeling.md) 和 [4.4 预训练大模型](../ch3-round2/3-4-4-pretrained-llms.md) 这两节最划算。
