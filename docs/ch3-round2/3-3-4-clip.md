# 3.4 视觉-文本编码器 CLIP

> **难度** ⭐⭐⭐⭐☆ · **前置**：[2.2 Transformer 架构](3-2-2-transformer.md)、[1.2 嵌入表示](3-1-2-embeddings.md)

!!! abstract "读完这一节，你会"
    - 说清 CLIP 的双塔（图像编码器 + 文本编码器）架构，以及它把图和文映射到同一个向量空间这件事
    - 推导对比损失 InfoNCE，理解它为什么本质上是一个"在一批样本里做分类"的交叉熵
    - 解释 CLIP 为什么能做零样本（zero-shot）分类，不微调就能认出训练时没专门标过的类别
    - 会用一段 PyTorch 代码算出图文相似度矩阵和对比损失，并知道下游怎么微调

## 先问一句：图片和文字能放进同一个空间吗

我们之前训练图像分类器，套路都是固定的：先把每张图标上一个类别号（猫=0、狗=1……），再让模型去学"图 → 类别号"的映射。这套办法很有效，但它有个天花板——**模型只认识你标过的那几个类**。你想让它多认一个"水豚"，就得重新收集水豚图、重新标注、重新训练。标注是昂贵的人力活，这条路注定走不远。

那有没有别的监督信号，又多又便宜、还自带语义？有——**互联网上海量的"图片 + 配文"**。一张照片下面的那句话（caption），其实就是人类顺手给图片写的"标签"，只不过它不是一个数字，而是一整句自然语言。CLIP（Contrastive Language–Image Pre-training，对比式语言-图像预训练）的核心想法就是：**别再硬背类别号了，直接让模型去学"哪张图配哪句话"。**

要做到这件事，关键的一步是：把图片和文字这两种完全不同的东西，各自编码成一个向量，并且想办法让它们落在**同一个向量空间**里。只有在同一个空间里，我们才能用"距离近不近"来判断一张图和一句话配不配。下面这张图就是 CLIP 最核心的画面：

<div class="diagram">
<svg viewBox="0 0 420 230" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ca" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <!-- 图像侧输入 -->
  <rect x="20" y="30" width="64" height="46" rx="4" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <text x="52" y="58" text-anchor="middle" font-size="12" fill="var(--dia-accent)">图片</text>
  <!-- 文本侧输入 -->
  <rect x="20" y="154" width="64" height="46" rx="4" fill="var(--dia-accent-soft)" stroke="var(--dia-blue)" stroke-width="1.5"/>
  <text x="52" y="182" text-anchor="middle" font-size="12" fill="var(--dia-blue)">配文</text>
  <!-- 图像编码器 -->
  <rect x="128" y="28" width="92" height="50" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <text x="174" y="49" text-anchor="middle" font-size="11" fill="var(--dia-stroke)">图像编码器</text>
  <text x="174" y="65" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">ViT</text>
  <!-- 文本编码器 -->
  <rect x="128" y="152" width="92" height="50" rx="4" fill="var(--dia-bg-card)" stroke="var(--dia-blue)" stroke-width="1.5"/>
  <text x="174" y="173" text-anchor="middle" font-size="11" fill="var(--dia-stroke)">文本编码器</text>
  <text x="174" y="189" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">Transformer</text>
  <!-- 箭头 输入->编码器 -->
  <line x1="84" y1="53" x2="126" y2="53" stroke="var(--dia-stroke-soft)" stroke-width="1.2" marker-end="url(#ca)"/>
  <line x1="84" y1="177" x2="126" y2="177" stroke="var(--dia-stroke-soft)" stroke-width="1.2" marker-end="url(#ca)"/>
  <!-- 箭头 编码器->共享空间 -->
  <line x1="220" y1="53" x2="296" y2="100" stroke="var(--dia-stroke-soft)" stroke-width="1.2" marker-end="url(#ca)"/>
  <line x1="220" y1="177" x2="296" y2="130" stroke="var(--dia-stroke-soft)" stroke-width="1.2" marker-end="url(#ca)"/>
  <!-- 共享向量空间 -->
  <rect x="300" y="78" width="104" height="74" rx="6" fill="var(--dia-bg-deep)" stroke="var(--dia-stroke-tertiary)" stroke-width="1.2" stroke-dasharray="4 3"/>
  <circle cx="332" cy="108" r="4" fill="var(--dia-accent)"/>
  <circle cx="340" cy="118" r="4" fill="var(--dia-blue)"/>
  <circle cx="378" cy="128" r="4" fill="var(--dia-accent)"/>
  <circle cx="370" cy="100" r="4" fill="var(--dia-blue)"/>
  <text x="352" y="170" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">共享向量空间</text>
</svg>
</div>
<p class="figure-caption">图 3.4-1：CLIP 的双塔结构。图片走图像编码器、配文走文本编码器，两条路各自输出一个向量，落进同一个空间——配对的图文应当靠得很近。</p>

把这张图记牢，后面所有的公式和代码，都只是在把"配对的图文靠近、不配对的推远"这件事说得更精确而已。

## 双塔架构：两个编码器，一个共享空间

CLIP 之所以叫"双塔"（two-tower）或"双编码器"（dual-encoder），是因为它有两个互相独立的编码器，像两座并排的塔。

**图像这一侧**，用一个图像编码器把整张图压成一个向量。原始论文用过 ResNet，但效果更好、现在更常用的是 **视觉 Transformer（Vision Transformer，ViT）**——它把图片切成一个个小方块（patch），当作"视觉单词"喂进 Transformer。如果你对 Transformer 怎么处理序列还不熟，建议回头看 [2.2 Transformer 架构](3-2-2-transformer.md)。

**文本这一侧**，用一个文本编码器（一个标准的 Transformer）把一句话压成一个向量。它的输入是分好词的 token 序列，输出取某个特殊位置（比如句末标记）的表示，作为整句话的"嵌入"。关于把词变成向量这件事，[1.2 嵌入表示](3-1-2-embeddings.md) 已经讲过。

两座塔各自的输出维度可能不一样，所以最后还各接一个线性投影层，把它们映射到**同一个维度** $d$ 的空间里。于是对一张图 $I$，我们得到一个向量 $\mathbf{f}_I\in\mathbb{R}^d$；对一句话 $T$，得到 $\mathbf{f}_T\in\mathbb{R}^d$。

这里有个不起眼但很关键的细节：我们会把每个向量都**归一化成单位长度**，即 $\hat{\mathbf{f}}=\mathbf{f}/\lVert\mathbf{f}\rVert$。为什么要这么做？因为归一化之后，两个向量的点积就正好等于它们夹角的余弦——也就是**余弦相似度（cosine similarity）**。这样我们衡量"图和文配不配"，就只看方向、不受向量长短的干扰，干净又稳定。

## 对比损失 InfoNCE：把"配对"变成一道分类题

架构搭好了，真正的灵魂在于：**怎么训练这两座塔，才能让配对的图文靠近、不配对的推远？** 答案是对比损失，CLIP 用的是其中最经典的一种，叫 **InfoNCE**。我们一步步把它推出来。

训练时，我们不是一张一张地喂，而是一次喂进 $N$ 对图文 $(I_1,T_1),\dots,(I_N,T_N)$。把这 $N$ 张图都编码、归一化，得到 $\hat{\mathbf{f}}_{I_1},\dots,\hat{\mathbf{f}}_{I_N}$；把 $N$ 句话也都编码、归一化，得到 $\hat{\mathbf{f}}_{T_1},\dots,\hat{\mathbf{f}}_{T_N}$。

现在两两算相似度，我们就得到一个 $N\times N$ 的相似度矩阵 $S$，其中

$$S_{ij}=\hat{\mathbf{f}}_{I_i}\cdot\hat{\mathbf{f}}_{T_j}.$$

关键的观察是：在这个矩阵里，**只有对角线上的 $N$ 个元素是"真配对"**（第 $i$ 张图本来就配第 $i$ 句话），其余 $N^2-N$ 个都是"不配对"。我们的训练目标就变成一句话：**让对角线最大、其余最小。** 下面这张图把这个矩阵画了出来：

<div class="diagram">
<svg viewBox="0 0 320 240" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 列标题（文本 T1..T4） -->
  <text x="92"  y="34" text-anchor="middle" font-size="11" fill="var(--dia-blue)">T₁</text>
  <text x="142" y="34" text-anchor="middle" font-size="11" fill="var(--dia-blue)">T₂</text>
  <text x="192" y="34" text-anchor="middle" font-size="11" fill="var(--dia-blue)">T₃</text>
  <text x="242" y="34" text-anchor="middle" font-size="11" fill="var(--dia-blue)">T₄</text>
  <!-- 行标题（图像 I1..I4） -->
  <text x="48" y="68"  text-anchor="middle" font-size="11" fill="var(--dia-accent)">I₁</text>
  <text x="48" y="118" text-anchor="middle" font-size="11" fill="var(--dia-accent)">I₂</text>
  <text x="48" y="168" text-anchor="middle" font-size="11" fill="var(--dia-accent)">I₃</text>
  <text x="48" y="218" text-anchor="middle" font-size="11" fill="var(--dia-accent)">I₄</text>
  <!-- 4x4 网格，对角线高亮 -->
  <g stroke="var(--dia-stroke-tertiary)" stroke-width="1">
    <!-- row1 -->
    <rect x="68"  y="48"  width="48" height="40" fill="var(--dia-accent)"/>
    <rect x="118" y="48"  width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="168" y="48"  width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="218" y="48"  width="48" height="40" fill="var(--dia-bg-card)"/>
    <!-- row2 -->
    <rect x="68"  y="98"  width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="118" y="98"  width="48" height="40" fill="var(--dia-accent)"/>
    <rect x="168" y="98"  width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="218" y="98"  width="48" height="40" fill="var(--dia-bg-card)"/>
    <!-- row3 -->
    <rect x="68"  y="148" width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="118" y="148" width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="168" y="148" width="48" height="40" fill="var(--dia-accent)"/>
    <rect x="218" y="148" width="48" height="40" fill="var(--dia-bg-card)"/>
    <!-- row4 -->
    <rect x="68"  y="198" width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="118" y="198" width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="168" y="198" width="48" height="40" fill="var(--dia-bg-card)"/>
    <rect x="218" y="198" width="48" height="40" fill="var(--dia-accent)"/>
  </g>
</svg>
</div>
<p class="figure-caption">图 3.4-2：一个批次里 4 对图文的相似度矩阵。深色对角线是"真配对"，要把它们的相似度拉到最高；其余浅色格子是"负样本"，要压下去。</p>

怎么用一个损失函数表达"让对角线最大"？这里有个很漂亮的转化：**把每一行看成一道分类题。** 对第 $i$ 张图，它面对 $N$ 句话，正确答案就是第 $i$ 句——这不就是一个 $N$ 选 1 的分类吗？于是我们直接套用 softmax 交叉熵。先对相似度除以一个**温度系数（temperature）** $\tau$（它控制分布的"尖锐"程度），再做 softmax：

$$
\mathcal{L}_{I\to T}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N}\exp(S_{ij}/\tau)}.
$$

你仔细看这个式子：分子是"图 $i$ 配它的正确文本"的相似度，分母是它配批次里所有文本的相似度之和。要让这个 log 概率变大，就得让分子（对角线）相对于分母（整行）尽量突出——正好就是我们要的"对角线最大"。这种"在一批样本里，把正样本从一堆负样本中挑出来"的损失，就叫 **InfoNCE**。

对称地，我们也可以反过来"用每一句话去认它的图"，得到 $\mathcal{L}_{T\to I}$（这次是按列做 softmax）。CLIP 把两个方向加起来取平均，让训练更稳：

$$
\mathcal{L}=\tfrac{1}{2}\big(\mathcal{L}_{I\to T}+\mathcal{L}_{T\to I}\big).
$$

!!! example "例：温度系数 τ 在调什么"
    $\tau$ 越小，$S/\tau$ 的差距被放得越大，softmax 越"尖"——模型会非常自信地只把概率压在最像的那一个上，对负样本的惩罚也更狠。$\tau$ 太大则分布太平，正负样本拉不开。CLIP 干脆把 $\tau$ 也当成一个可学习参数，让模型自己在训练中找到合适的"尖锐度"。

## 动手：算一遍相似度矩阵和对比损失

道理讲完了，我们用 PyTorch 把上面的损失从零实现一遍。为了聚焦在损失本身，这里用随机向量假装是编码器的输出，重点看**归一化 → 相似度矩阵 → 对称交叉熵**这三步：

```python
import torch
import torch.nn.functional as F

N, d = 4, 16                       # 一批 4 对图文，每个向量 16 维
img_feat = torch.randn(N, d)       # 假装是图像编码器的输出
txt_feat = torch.randn(N, d)       # 假装是文本编码器的输出

# 第一步：把每个向量归一化成单位长度，点积就等于余弦相似度
img_feat = F.normalize(img_feat, dim=1)
txt_feat = F.normalize(txt_feat, dim=1)

# 第二步：算 N×N 相似度矩阵，并除以温度系数
temperature = 0.07
logits = img_feat @ txt_feat.t() / temperature   # logits[i, j] = 图i 配 文j 的相似度

# 第三步：对角线是正确答案，做对称交叉熵
labels = torch.arange(N)                          # 正确标签就是 0,1,2,...,N-1
loss_i2t = F.cross_entropy(logits, labels)        # 每行：用图找文
loss_t2i = F.cross_entropy(logits.t(), labels)    # 每列：用文找图
loss = (loss_i2t + loss_t2i) / 2
print(loss.item())
```

这段代码的精髓全在最后三行：`labels = arange(N)` 这一步特别巧妙——因为第 $i$ 行的正确答案恰好是第 $i$ 列，标签序列正好就是 $0,1,2,\dots$。所以我们根本不需要任何人工标注，**"谁配谁"这个监督信号，是从"图文成对出现"这件事里自动生出来的**。这正是 CLIP 便宜又能规模化的根本原因。

真实项目里，你不会自己搭这两座塔，而是直接用 OpenAI 开源的预训练 CLIP。下面是库版的对照写法，几行就能算出"一张图和几句候选描述各自有多配"：

```python
import clip          # pip install git+https://github.com/openai/CLIP.git
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)
texts = clip.tokenize(["a photo of a cat",
                       "a photo of a dog",
                       "a photo of a car"]).to(device)

with torch.no_grad():
    logits_per_image, _ = model(image, texts)        # 图 对 三句话 的相似度
    probs = logits_per_image.softmax(dim=-1)         # 转成概率
print(probs)    # 比如 [[0.96, 0.03, 0.01]] —— 模型认为这张图最像 "a cat"
```

注意最后这步：我们给了三句话当候选，模型输出"这张图分别有多像每句话"。它从没被专门训练成一个"猫/狗/车三分类器"，却能直接做这道分类题——这就引出了 CLIP 最迷人的能力。

## 零样本分类：不微调就能认新类别

CLIP 真正震撼业界的，是它的**零样本（zero-shot）**能力：面对一批训练时从没专门标注过的类别，它不需要任何额外训练、不看一张示例图，就能直接分类。这是怎么做到的？

诀窍在上一段的代码里其实已经露出来了。**做分类，不再是"图 → 类别号"，而是"图 → 哪句描述最配"。** 假设你要在 ImageNet 的 1000 个类上分类，你只需把每个类别名套进一个模板，造出 1000 句话——"a photo of a {类别名}"，比如 "a photo of a goldfish"、"a photo of a tractor"。然后：

1. 用文本编码器把这 1000 句话各编码成一个向量，作为 1000 个"类别原型"；
2. 把待分类的图片编码成一个向量；
3. 看这张图的向量和哪个类别原型的余弦相似度最高，就判成那一类。

你可以把它理解成：CLIP 在训练中已经学会了"语义对齐"这件通用本事，于是任何能用一句话描述出来的类别，它都能现场临时拼出一个分类器。**想加一个新类别？写一句话就行，不用一行训练代码。** 这种灵活性是传统分类器做梦都想不到的——也是为什么 CLIP 成了今天几乎所有多模态大模型（图文对话、文生图）的视觉地基。

## 下游怎么用：从零样本到微调

零样本虽好，但当你手上确实有一批标注数据、又追求最高精度时，可以在 CLIP 的基础上做**微调（fine-tuning）**。常见有三种用法，按"动多少参数"从轻到重排：

**第一种，当冻结的特征提取器。** 把 CLIP 的图像编码器参数全部冻住，只把它当成一个"把图变成好向量"的黑盒，在它输出的向量上面接一个小小的线性分类头来训练。这叫**线性探测（linear probing）**，又快又省，数据少时尤其稳。关于"冻住预训练编码器、只训练顶部"的通用思路，[3.3 迁移学习与预训练编码器](../ch2-round1/2-3-3-transfer-learning.md) 讲得更细。

**第二种，提示工程（prompt engineering）。** 连参数都不动，只优化那句模板话术。把 "a photo of a {类}" 换成更贴合任务的描述（比如卫星图任务用 "a satellite photo of {类}"），零样本精度往往就能涨好几个点。

**第三种，端到端微调。** 用你的数据，连同对比损失或分类损失一起，把整个编码器再训练几轮。精度最高，但要的数据和算力也最多，还容易把预训练学到的通用能力"训没了"（灾难性遗忘），需要小心调小学习率。关于微调的火候把控，见 [1.3 模型微调](3-1-3-finetuning.md)。

一句话总结这三种的取舍：**数据越少、越想省事，就越往"只动一点点"那头靠；数据越多、越追求极致精度，才考虑端到端微调。**

## 容易踩的坑

- **忘了归一化就算点积**：不做 `F.normalize`，点积会受向量长度干扰，相似度失真，训练直接学歪。InfoNCE 的前提就是用余弦相似度。
- **批次太小，负样本不够**：InfoNCE 的负样本来自同一个批次，批次越大、负样本越多，对比学到的空间越好。这也是为什么 CLIP 原版要用上万的超大批次（靠多卡）。小卡上复现时这点最容易被忽略。
- **温度系数当成普通常数**：$\tau$ 对结果影响很大，且 CLIP 里它是**可学习**的。自己实现时若把它写死成一个随手填的数，效果可能差很多。
- **零样本模板太随意**：直接用单个类别名（"cat"）当文本，往往不如套上 "a photo of a cat" 的模板——因为训练时的配文基本都是完整句子，模板让分布对得上。
- **以为 CLIP 能精确定位**：CLIP 学的是"整图配整句"的全局对齐，它不擅长说清"物体在图的哪个位置"。要定位请用检测/分割模型（见 [3.1 目标检测](3-3-1-object-detection.md)、[3.2 图像分割](3-3-2-segmentation.md)）。

## 它在后面会怎么用到

CLIP 这种"图文对齐 + 对比学习"的思路，是 Round 2 视觉-语言这一块的枢纽，往后会反复照面：

- 它的对比学习思想和 [3.3 数据增强与自监督](3-3-3-augmentation-ssl.md) 里的自监督是一脉相承的——都是"不用人工标签，靠样本之间的关系来学表示"。
- 它依赖的两座塔，分别是 [2.2 Transformer 架构](3-2-2-transformer.md) 在图像（ViT）和文本上的应用，而向量空间这套语言来自 [1.2 嵌入表示](3-1-2-embeddings.md)。
- CLIP 的文本编码器是许多**文生图扩散模型**的"听懂人话"的入口，会在 [3.5 生成模型 GAN·扩散](3-3-5-generative.md) 里再次出现。
- 把 CLIP 用于具体任务时的冻结/微调权衡，串起了 [3.3 迁移学习与预训练编码器](../ch2-round1/2-3-3-transfer-learning.md) 和 [1.3 模型微调](3-1-3-finetuning.md)。

## 练习

??? note "基础练习"
    1. 用上面的"从零"代码，把 `temperature` 分别设成 `0.01`、`0.07`、`1.0`，观察同一批随机向量算出的 loss 怎么变，并解释温度为什么会有这种影响。
    2. 给定 3 句候选描述和 1 张图，用库版 CLIP 跑出 `softmax` 概率，再换一张语义完全不同的图，确认概率分布也跟着变了。体会"零样本分类"是怎么发生的。

??? note "进阶练习"
    1. 把"从零"代码里的对称交叉熵拆开：只用 $\mathcal{L}_{I\to T}$ 训练，和用对称的 $\frac{1}{2}(\mathcal{L}_{I\to T}+\mathcal{L}_{T\to I})$ 训练，在一个小玩具数据上对比。想一想为什么对称版通常更稳。
    2. 做一个迷你零样本实验：取 CIFAR-10 的几个类，为每类写一句 "a photo of a {类}" 模板，用预训练 CLIP 在测试集上算零样本准确率；再把模板换成更贴切的说法，看精度涨了多少。这就是提示工程的威力。

## 小结

- CLIP 是**双塔**：图像编码器（ViT）和文本编码器（Transformer）各自出一个向量，归一化后落进**同一个向量空间**。
- 训练用**对比损失 InfoNCE**：把一批图文的相似度排成 $N\times N$ 矩阵，让对角线（真配对）最大、其余最小，本质是"在批内做 $N$ 选 1 的对称交叉熵"。
- 监督信号来自互联网海量的"图 + 配文"，不需要人工类别标注，所以又便宜又能规模化——这就是它为什么行。
- 它能**零样本分类**：把类别名写成一句话当"原型"，看图和哪句最配；下游还能按数据多少选择线性探测、提示工程或端到端微调。

想深入，强烈推荐读 OpenAI 的原论文 [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020)，以及它的[官方开源实现](https://github.com/openai/CLIP)——配着上面的代码读，会非常通透。
