# 4.1 文本分类与词嵌入

> **难度** ⭐⭐⭐☆☆ · **前置**：[1.2 嵌入表示](3-1-2-embeddings.md)、[1.3 逻辑回归与分类](../ch2-round1/2-1-3-logistic-regression.md)

!!! abstract "读完这一节，你会"
    - 说清楚为什么文字必须先变成数字，并理解词袋、TF-IDF 各自在解决什么问题
    - 用大白话讲明白 Word2Vec（skip-gram）、GloVe、FastText 三种词向量的核心直觉与区别
    - 从零搭出一条"文本 → 向量 → 分类器"的完整管道，并知道库里对应的一行调用

## 计算机不认字，只认数

在做任何文本任务之前，我们先得面对一个最朴素的事实：**模型只会算数，它根本不认识文字。** 你给它一句"这部电影太好看了"，它看到的不是中文，而是一串它完全无法直接相乘相加的符号。所以无论后面要做情感分类、垃圾邮件识别还是话题归类，第一步永远是同一件事——**想办法把文字变成一串数字（向量）**。

这件事看似简单，做起来却大有讲究。因为我们不只是要"变成数字"，更希望这串数字能**保留住文字里的意义**：意思相近的词，向量也该靠得近；一句话里哪些词重要、哪些词是没营养的"的、了、是"，最好也能体现出来。这一节，我们就沿着"从笨办法到聪明办法"的顺序，把文本向量化的几种主流做法讲透。

## 词袋：最朴素的"数数法"

最直接的想法是什么呢？既然要把一句话变成向量，那我们干脆**数一数每个词出现了几次**。这就是**词袋模型（Bag of Words，简称 BoW）**——之所以叫"袋"，是因为它把一句话里的词一股脑儿倒进袋子里，只关心"出现了哪些词、各几次"，**完全不管词的先后顺序**。

举个例子，假设我们的词表只有 `["电影", "好看", "难看"]` 这三个词，那么：

- "电影好看" → `[1, 1, 0]`
- "电影难看" → `[1, 0, 1]`
- "电影电影好看" → `[2, 1, 0]`

每句话都变成了一个长度等于词表大小的向量，向量的第几个位置就记着第几个词出现了几次。简单粗暴，但它真的能用——很多情感分类任务，光靠词袋加一个逻辑回归就能做到八九不离十。

不过你应该已经嗅到它的两个毛病了。第一，它**丢掉了语序**，"狗咬人"和"人咬狗"在它眼里一模一样。第二，也是更要命的：像"的、是、了"这种词几乎每句话都出现，词频高得吓人，可它们对判断句子含义几乎毫无帮助。我们需要一个办法，**把这种"常见但没营养"的词压下去**——这就引出了 TF-IDF。

## TF-IDF：给词的"重要性"打分

TF-IDF 的核心思想，可以用一句大白话概括：**一个词如果在这篇文档里出现得多，但在所有文档里又不那么常见，那它多半就是这篇文档的关键词。** 这个想法非常符合直觉——"梯度"这个词在一篇深度学习文章里反复出现，而在大多数文章里压根不出现，那它显然是这篇文章的主题词；反观"的"字，哪篇文章里都一大堆，反而说明它没什么区分度。

它的名字就是两部分相乘。**词频（Term Frequency，TF）**衡量"这个词在当前文档里出现得多不多"：

$$\text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 的总词数}}.$$

**逆文档频率（Inverse Document Frequency，IDF）**衡量"这个词在整个语料里稀不稀有"：

$$\text{IDF}(t) = \log\frac{N}{1 + n_t},$$

其中 $N$ 是文档总数，$n_t$ 是包含词 $t$ 的文档数；分母加 $1$ 是为了防止某个词一篇都没出现时除以零。你可以看出，一个词越常见（$n_t$ 越大），IDF 就越小，它的权重就被压得越低。

最后把两者一乘，就得到这个词在这篇文档里的 TF-IDF 权重：

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t).$$

于是"的、是、了"这种到处都有的词，IDF 几乎为零，权重被狠狠压下去；而真正有区分度的主题词则被凸显出来。**TF-IDF 本质上还是词袋，只是把"数次数"换成了"打权重"**，所以它依然丢语序、依然不懂词与词之间的含义关系。要解决这个更深的问题，就得请出真正的"词嵌入"了。

## 词嵌入：让"意思"住进向量里

到这里我们换个思路。前面的词袋和 TF-IDF，每个词占向量里的一个坑，词与词之间彼此孤立——在它们眼里，"国王"和"女王"的距离，跟"国王"和"香蕉"的距离没有任何区别，因为每个词都只是一个独立的位置。这显然不对劲。

**词嵌入（word embedding）**要做的，就是给每个词学一个稠密的、低维的向量（比如 100 维或 300 维），并且让这些向量满足一个美妙的性质：**意思相近的词，向量也靠得近。** 更神奇的是，向量之间的方向还能编码语义关系，最著名的例子就是：

$$\text{vec}(\text{国王}) - \text{vec}(\text{男人}) + \text{vec}(\text{女人}) \approx \text{vec}(\text{女王}).$$

也就是说，"从男人到国王"这个方向，和"从女人到女王"这个方向几乎平行——向量的几何里，竟然藏下了"性别"和"王权"这样的抽象概念。下面这张图把这个画面画了出来：

<div class="diagram">
<svg viewBox="0 0 380 220" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ea" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 男人 -> 国王 -->
  <circle cx="70" cy="170" r="4" fill="var(--dia-blue)"/>
  <circle cx="170" cy="90" r="4" fill="var(--dia-blue)"/>
  <line x1="74" y1="167" x2="166" y2="93" stroke="var(--dia-accent)" stroke-width="1.8" marker-end="url(#ea)"/>
  <!-- 女人 -> 女王 -->
  <circle cx="210" cy="170" r="4" fill="var(--dia-green)"/>
  <circle cx="310" cy="90" r="4" fill="var(--dia-green)"/>
  <line x1="214" y1="167" x2="306" y2="93" stroke="var(--dia-accent)" stroke-width="1.8" marker-end="url(#ea)"/>
  <!-- 标注：全部放在点的外侧空白处 -->
  <text x="50" y="190" font-size="12" fill="var(--dia-stroke-soft)">男人</text>
  <text x="150" y="80" font-size="12" fill="var(--dia-stroke-soft)">国王</text>
  <text x="195" y="190" font-size="12" fill="var(--dia-stroke-soft)">女人</text>
  <text x="295" y="80" font-size="12" fill="var(--dia-stroke-soft)">女王</text>
  <text x="110" y="125" font-size="11" fill="var(--dia-accent)">+王权</text>
  <text x="250" y="125" font-size="11" fill="var(--dia-accent)">+王权</text>
</svg>
</div>
<p class="figure-caption">图 4.1-1：词向量里，"加上王权"是一个固定的方向；两条平行箭头说明语义关系被编码进了向量的几何之中。</p>

那么问题来了——这些神奇的向量是怎么学出来的？最经典的答案，就是 Word2Vec。

## Word2Vec：靠"猜上下文"学词义

Word2Vec 背后有一个朴素到近乎哲学的假设，语言学里叫**分布假说**：**一个词的含义，是由它周围经常出现的词决定的。** 换句话说，"你是什么样的词，看你跟谁在一起就知道了"。如果两个词总是出现在相似的上下文里（比如"猫"和"狗"周围常常都是"喂、宠物、可爱"），那它们的意思大概率也相近。

Word2Vec 把这个假设变成了一个可训练的任务。它最常用的版本叫 **skip-gram**，做的事情是：**给定中心词，去预测它周围的上下文词。** 比如句子"小猫喜欢吃鱼"，拿"喜欢"当中心词，模型就要努力预测出它两边的"小猫、吃"等词。下面这张图画出了这个滑动的窗口：

<div class="diagram">
<svg viewBox="0 0 400 170" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="sg" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 词序列方框 -->
  <rect x="20"  y="60" width="64" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-tertiary)"/>
  <rect x="96"  y="60" width="64" height="34" rx="4" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="2"/>
  <rect x="172" y="60" width="64" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-tertiary)"/>
  <rect x="248" y="60" width="64" height="34" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-tertiary)"/>
  <text x="52"  y="82" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">小猫</text>
  <text x="128" y="82" text-anchor="middle" font-size="12" fill="var(--dia-accent-deep)">喜欢</text>
  <text x="204" y="82" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">吃</text>
  <text x="280" y="82" text-anchor="middle" font-size="12" fill="var(--dia-stroke)">鱼</text>
  <!-- 中心词标注 -->
  <text x="128" y="48" text-anchor="middle" font-size="11" fill="var(--dia-accent)">中心词</text>
  <!-- 预测箭头 -->
  <path d="M120 96 Q90 130 56 100" fill="none" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#sg)"/>
  <path d="M136 96 Q166 130 200 100" fill="none" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#sg)"/>
  <text x="128" y="150" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">用中心词预测左右的上下文词</text>
</svg>
</div>
<p class="figure-caption">图 4.1-2：skip-gram 把"用中心词猜上下文"当作训练任务，副产品就是每个词的嵌入向量。</p>

关键在于：我们**并不真的关心"预测上下文"这个任务做得多准**，我们要的是它的副产品——为了把这个预测任务做好，模型必须给每个词学一组合适的向量参数，而这组参数恰好就成了我们想要的词嵌入。这是机器学习里一个非常典型的套路：**设一个假的目标任务，真正想要的是训练过程中顺带学到的中间表示。**

数学上，skip-gram 想最大化"给定中心词 $c$、出现上下文词 $o$"的概率，这个概率用 softmax 写出来是：

$$P(o \mid c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{w} \exp(\mathbf{u}_w^\top \mathbf{v}_c)},$$

其中 $\mathbf{v}_c$ 是中心词的向量，$\mathbf{u}_o$ 是上下文词的向量。你不用记这个式子的细节，只需抓住一点：**两个词的向量内积越大，它们越可能共同出现**——训练就是在把"经常做邻居"的词的向量往一起拽。

至于 **GloVe**，你可以把它看成 Word2Vec 的"近亲"。它换了个角度：不再一个窗口一个窗口地滑，而是先在整个语料上统计出**任意两个词共同出现的次数表**（共现矩阵），再让词向量去拟合这张全局统计表。**Word2Vec 是局部预测，GloVe 是全局统计**，二者目标不同、殊途同归，效果也旗鼓相当，实际用时挑顺手的那个即可。

## FastText：把词拆成"零件"来理解

Word2Vec 和 GloVe 都有一个共同的软肋：它们以**整个词**为最小单位。这带来两个麻烦。第一，遇到训练时**没见过的新词**（专业上叫 out-of-vocabulary，词表外词），它们直接傻眼，给不出任何向量。第二，对于像英语这种靠词缀变形的语言，`play`、`playing`、`played` 在它们眼里是三个毫不相干的词，明明意思紧密相关，却各学各的向量，很浪费。

FastText 的解法非常聪明：**它不把词看成不可分割的整体，而是把词拆成一串字符级的小片段（子词，subword）。** 比如英文单词 `where`，会被拆成 `<wh`、`whe`、`her`、`ere`、`re>` 这样的若干个三字符片段（n-gram，这里的尖括号标记词的开头和结尾）。一个词的向量，就由它所有子词向量加起来得到。

这一拆，前面两个麻烦就迎刃而解了。**碰到新词**，哪怕整词没见过，只要它的子词见过，就能拼出一个合理的向量；**词形变化**的词因为共享了大量子词，向量自然就靠得近。所以处理英语、德语这类形态丰富的语言，或者语料里新词、错别字特别多的场景，FastText 往往比 Word2Vec 更稳。

把这三者放一起对比一下，你就清楚该在什么时候用谁了：

| 方法 | 学习方式 | 最小单位 | 能处理新词吗 |
| --- | --- | --- | --- |
| Word2Vec | 局部窗口预测上下文 | 整个词 | 不能 |
| GloVe | 拟合全局共现统计 | 整个词 | 不能 |
| FastText | 局部预测 + 子词拼接 | 子词片段 | 能 |

## 动手搭一条文本分类管道

道理讲完了，我们动手把一整条**文本分类管道**跑通。所谓管道，就是"原始文字 → 向量化 → 喂给分类器 → 出预测"这一条流水线。我们先用最经典、最不容易出错的 **TF-IDF + 逻辑回归** 组合，因为它在中小数据集上又快又强，是文本分类雷打不动的基线。

为了让你看清"向量化"到底在干什么，我们**先从零用 NumPy 手算一遍词袋**，建立直观：

```python
import numpy as np

corpus = ["电影 好看", "电影 难看", "电影 电影 好看"]

# 第一步：扫一遍语料，建一个词表（每个词分配一个固定的列号）
vocab = sorted({w for doc in corpus for w in doc.split()})
word2idx = {w: i for i, w in enumerate(vocab)}   # {'好看':0, '电影':1, '难看':2}

# 第二步：每篇文档变成一个"数次数"的向量
X = np.zeros((len(corpus), len(vocab)))
for r, doc in enumerate(corpus):
    for w in doc.split():
        X[r, word2idx[w]] += 1

print(vocab)   # ['好看', '电影', '难看']
print(X)       # [[1 1 0]  "电影 好看"
               #  [0 1 1]  "电影 难看"
               #  [1 2 0]] "电影 电影 好看"
```

你看，输出和我们手算的完全一致：第三行里"电影"出现两次，对应列就是 `2`。**这就是词袋的全部秘密——数数、填表，没有任何魔法。**

理解了底层，实际工程里我们当然不会自己写这些。scikit-learn 把"TF-IDF 向量化 + 分类器"打包成了一条 `Pipeline`，几行就能搞定：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

texts  = ["电影 好看 推荐", "剧情 难看 失望", "演技 在线 好看", "浪费 时间 难看"]
labels = [1, 0, 1, 0]          # 1=好评, 0=差评

# 把"向量化"和"分类器"串成一条管道，fit 时会自动按顺序处理
clf = make_pipeline(
    TfidfVectorizer(),          # 文字 -> TF-IDF 向量
    LogisticRegression(),       # 向量 -> 类别
)
clf.fit(texts, labels)

print(clf.predict(["这部 电影 好看"]))   # -> [1]，判为好评
```

注意这里一个很省心的设计：**用 `Pipeline` 把向量化和分类器绑在一起后，预测新句子时，它会自动用训练时学到的同一套词表和 IDF 去转换**，你完全不用手动再做一遍向量化。这也顺手避开了一个新手常踩的坑——拿测试数据重新算 IDF 导致的数据泄漏。

如果你想用前面讲的词嵌入而不是 TF-IDF，最省事的办法是直接加载别人在海量语料上训练好的现成词向量（比如 `gensim` 库里的预训练 GloVe），把一句话里每个词的向量取出来求平均，得到一个"句子向量"，再喂给分类器即可。这种"预训练向量 + 简单分类器"的思路，正是 [1.2 嵌入表示](3-1-2-embeddings.md) 里讲的迁移学习在 NLP 上的最初形态。

## 容易踩的坑

- **维度爆炸**：词袋和 TF-IDF 的向量长度等于词表大小，语料一大就是几万维，绝大多数是 0。记得用稀疏矩阵存储（sklearn 默认就是），别 `.toarray()` 把它强行变成稠密矩阵，否则内存会瞬间爆掉。
- **IDF 必须只在训练集上 `fit`**：对验证集和测试集只能用 `transform`，绝不能重新 `fit_transform`。用 `Pipeline` 能自动帮你守住这条线，手动写就很容易疏忽——这本质上和 [4.3 Pandas 与数据处理](../ch1-intro/1-4-3-pandas-data.md) 里讲的标准化数据泄漏是同一个坑。
- **中文要先分词**：英文天然用空格断词，中文得先用 jieba 之类的工具把"我爱自然语言处理"切成"我 / 爱 / 自然语言处理"，否则整句会被当成一个超长的"词"。
- **词嵌入别拿来当查字典**：词向量编码的是"上下文里的统计共现"，不是词典定义。所以"good"和"bad"的向量反而可能很近——因为它俩出现的上下文极其相似（都跟在"这电影真"后面）。情感任务里这点要特别小心。

## 它在后面会怎么用到

这一节是你叩开 NLP 大门的第一步，它打下的地基会一路用到后面：

- 词嵌入把"离散的词"变成"连续的向量"，这正是所有现代语言模型的输入起点，下一节 [4.2 语言建模](3-4-2-language-modeling.md) 会接着讲模型如何预测下一个词。
- 词袋/词嵌入这种"对一句话求平均"的做法丢掉了语序，而真正解决语序问题、让模型学会"该看哪个词"的，是 [2.1 注意力机制](3-2-1-attention.md) 和 [2.2 Transformer 架构](3-2-2-transformer.md)。
- 把文本嵌入和图像嵌入对齐到同一个空间，就能让模型"看图说话"，这是 [3.4 视觉-文本编码器 CLIP](3-3-4-clip.md) 的核心思想。

## 练习

??? note "基础练习"
    1. 手算下面三句话的词袋向量（词表自己建）：「天气 真好」「天气 真差」「真好 真好」。然后说说为什么词袋分不清「狗咬人」和「人咬狗」。
    2. 用 `TfidfVectorizer` 对一小批句子做向量化，打印出 `get_feature_names_out()`（词表）和向量矩阵，找出 IDF 最低的那个词，验证它是不是出现得最普遍的那个。

??? note "进阶练习"
    1. 在 Word2Vec 或 GloVe 的预训练向量上，亲手验证 `国王 - 男人 + 女人 ≈ 女王`：取出这四个词的向量做加减，再用余弦相似度找出最接近的词，看看是不是"女王"。
    2. 同一份情感分类数据，分别用「TF-IDF + 逻辑回归」和「预训练词向量取平均 + 逻辑回归」两条管道训练，比较准确率。想一想：在数据量很小的时候，哪种更可能赢，为什么？

## 小结

- 文本任务的第一步永远是**把文字变成向量**：词袋数次数、TF-IDF 打权重，但两者都丢语序、不懂词义。
- **词嵌入**让意思相近的词向量靠得近；Word2Vec 靠局部窗口"猜上下文"来学，GloVe 靠拟合全局共现统计，FastText 进一步拆成子词、能应对新词。
- 一条最实用的基线管道是 **TF-IDF + 逻辑回归**，用 sklearn 的 `Pipeline` 串起来既简洁又能自动防数据泄漏。

想把词向量的直觉建得更牢，强烈推荐 Jay Alammar 的图解博客 [*The Illustrated Word2Vec*](https://jalammar.github.io/illustrated-word2vec/)，它用大量动图把 skip-gram 讲得一目了然；想系统了解，可读 [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)（Jurafsky & Martin）第 6 章。
