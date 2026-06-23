# 1.3 模型微调

> **难度** ⭐⭐⭐☆☆ · **前置**：[3.3 迁移学习与预训练编码器](../ch2-round1/2-3-3-transfer-learning.md)、[2.5 优化与正则化](../ch2-round1/2-2-5-optimization-regularization.md)

!!! abstract "读完这一节，你会"
    - 说清"微调（fine-tuning）"和"迁移学习（transfer learning）"到底是什么关系、差在哪
    - 知道为什么微调要配 warm-up 学习率，以及为什么要给不同层用不同的学习率（分层学习率）
    - 判断什么时候该冻结、什么时候该逐层解冻预训练权重
    - 用 PyTorch 写出一套"分组学习率 + warm-up + 逐层解冻"的标准微调代码

## 先把上一关的故事接上：从"借特征"到"轻轻改特征"

在 Round 1 里我们已经见过[迁移学习](../ch2-round1/2-3-3-transfer-learning.md)：拿一个在海量数据上训练好的大模型（比如 ImageNet 上的 ResNet），把它当成一个现成的"特征提取器"，只在它后面接一个新的分类头，训练这个小头。那种做法里，预训练模型本身是**冻住不动**的，我们借的是它已经学好的特征。

这一节要往前走一步。你会发现，只借特征有时候不够用——如果你的任务和原任务差得有点远（比如原模型学的是日常照片，你要分类的是医学影像），那些现成特征就不那么贴切了。这时我们希望让预训练模型自己也**跟着新任务微微调整一下**，而不是死死冻住。这个"让预训练权重也参与训练、但只小幅改动"的过程，就叫**微调（fine-tuning）**。

所以先把这层关系理清楚，免得后面一直绕：

- **迁移学习**是个大思路——"把在 A 任务上学到的知识搬到 B 任务上用"。它是目标，是策略。
- **微调**是实现迁移学习的一种**具体手段**——"不仅搬过来，还允许预训练权重以很小的步子继续学"。

换句话说，"冻结主干 + 只训新头"和"微调整个网络"都属于迁移学习，区别只在于：**预训练那部分参数到底动不动、动多少**。这也是这一节真正的主题——怎么让它"动得恰到好处"。

## 为什么不能拿原来的学习率猛冲：warm-up 的来历

想象你接手了一个已经被前人精心打磨好的模型，它的权重正稳稳地待在一个不错的位置。现在你要用新数据继续训练它。第一反应可能是：直接套用平时训练的学习率，开干。

可问题来了——刚开始的几步，最危险。为什么？因为你新接的那个分类头是**随机初始化**的，它一开始什么都不懂，会产生很大的、方向混乱的梯度。如果这时学习率给得太猛，这些"乱梯度"会顺着反向传播一路冲进预训练主干，**把好不容易学好的权重一下子搅乱**，相当于把前人的成果毁了一大半。这种现象有个形象的说法，叫"灾难性遗忘（catastrophic forgetting）"。

解决办法很朴素：**一开始把学习率压得很小，让它在最初几百步里慢慢、线性地升上去，到了目标值再开始正常衰减。** 这个"先小步热身、再放开"的过程，就叫 **warm-up（学习率预热）**。它给了随机的新头一点时间先"稳下来"，等梯度不那么狂躁了，再让整个网络一起大步学。

下面这张图把 warm-up 的学习率曲线画了出来，你可以顺着它走一遍。

<div class="diagram">
<svg viewBox="0 0 380 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 坐标轴 -->
  <line x1="40" y1="160" x2="350" y2="160" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <line x1="40" y1="160" x2="40" y2="30" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <!-- warm-up 上升段 -->
  <path d="M40 158 L120 50" fill="none" stroke="var(--dia-accent)" stroke-width="2.5"/>
  <!-- 衰减段 -->
  <path d="M120 50 Q230 60 340 130" fill="none" stroke="var(--dia-blue)" stroke-width="2.5"/>
  <!-- 峰值虚线 -->
  <line x1="120" y1="50" x2="120" y2="160" stroke="var(--dia-stroke-tertiary)" stroke-width="1" stroke-dasharray="3 3"/>
  <!-- 文字标注：放空白处 -->
  <text x="44" y="24" font-size="11" fill="var(--dia-stroke-soft)">学习率 ↑</text>
  <text x="300" y="178" font-size="11" fill="var(--dia-stroke-soft)">训练步数 →</text>
  <text x="62" y="110" font-size="11" fill="var(--dia-accent-deep)" transform="rotate(-52 62 110)">预热上升</text>
  <text x="225" y="48" font-size="11" fill="var(--dia-blue)">余弦/线性衰减</text>
  <text x="92" y="174" font-size="10" fill="var(--dia-stroke-tertiary)">warm-up 结束</text>
</svg>
</div>
<p class="figure-caption">图 1.3-1：微调常用的学习率曲线——先在最初若干步里线性预热到峰值，再缓缓衰减到接近 0。</p>

## 不同层别一刀切：分层学习率

接着上面的思路再想一层。一个深层网络，**底层和顶层学的东西其实很不一样**。

底层（靠近输入那几层）学的是非常通用的东西——边缘、纹理、颜色块，这些对几乎所有视觉任务都管用，本来就接近正确，没必要大改。顶层（靠近输出那几层）学的是和具体任务高度相关的高级语义，换了任务最需要重新适配的就是它们。

既然如此，**给所有层用同一个学习率就很浪费、甚至有害**：底层被大学习率一冲，通用特征就被破坏了；顶层用太小的学习率又学不动。合理的做法是**分层学习率（layer-wise / discriminative learning rate）**——越靠近输入的层，学习率越小（甚至接近 0）；越靠近输出的层，学习率越大。

一个常用的设定方式是让学习率沿层"逐层打折"：设最顶层的学习率为 $\eta$，往下每一层乘一个折扣因子 $\gamma\in(0,1)$，那么第 $k$ 层（从顶往下数）的学习率就是

$$\eta_k=\eta\cdot\gamma^{\,k}.$$

比如取 $\eta=10^{-3}$、$\gamma=0.7$，顶层是 $10^{-3}$，往下一层变 $7\times10^{-4}$，再往下 $4.9\times10^{-4}$……越深越温柔。这样既保住了底层宝贵的通用特征，又让顶层能充分适应新任务。

## 冻结还是解冻：到底什么时候放开

"冻结（freeze）"就是把某些层的参数设成不参与梯度更新，"解冻（unfreeze）"反之。该冻多少、什么时候解冻，没有唯一答案，但有一条很好用的经验判断，关键看**你的数据有多少、和原任务有多像**：

- **数据少 + 任务相似**：主干几乎全冻，只训新头。数据太少，放开主干极易过拟合，而任务相似意味着现成特征本来就够用。
- **数据少 + 任务差别大**：冻住底层，只解冻顶部少数几层。底层通用、可留；顶层需要适配，但又不敢全放开（怕过拟合）。
- **数据多 + 任务差别大**：可以解冻整个网络做全面微调，因为数据足够撑得起这么多参数的更新。

实践里还有一个很稳的套路，叫**逐层解冻（gradual unfreezing）**：先只训新头几轮，让它稳定下来；再解冻最顶层、训几轮；然后一层层往下解冻。这样做的好处是，每次只让一小部分预训练权重"动起来"，把灾难性遗忘的风险摊薄到整个训练过程中，而不是一上来就全盘震荡。

## 还有第四条路：只加一点点新参数（adapter）

到这里你可能会想：现在的大模型动辄上亿参数，全量微调一遍既费显存又费时间，难道每换一个任务都得把所有权重重训、再各存一整份吗？这正是 **adapter（适配器）** 这类方法想解决的问题。

它的核心想法很巧：**把原模型的权重整个冻死，一个都不动；只在网络里插入一些很小的、新的可训练模块，训练时只更新这些小模块。** 因为新增的参数量可能只占原模型的百分之一甚至更少，所以这类做法统称**参数高效微调（PEFT, parameter-efficient fine-tuning）**。

最有代表性的一个是 **LoRA（Low-Rank Adaptation，低秩适配）**。它注意到：微调时权重的"改动量" $\Delta W$ 往往是低秩的——也就是说，这个大矩阵其实可以用两个又瘦又长的小矩阵相乘来近似。于是它不直接训练 $\Delta W$，而是把它拆成 $\Delta W = BA$，其中 $A$、$B$ 都很"瘦"：

$$W_{\text{new}} = W_{\text{冻结}} + \Delta W = W_{\text{冻结}} + BA,\qquad A\in\mathbb{R}^{r\times d},\; B\in\mathbb{R}^{d\times r},\; r\ll d.$$

这里的 $r$ 叫"秩（rank）"，取得很小（比如 8 或 16），所以 $A$、$B$ 加起来的参数量远小于原来的 $W$。训练时只更新 $A$、$B$，原始 $W$ 纹丝不动。换一个任务，只要换一套小小的 $A$、$B$ 就行——这就是为什么现在大模型可以"一个底座、千百套小适配器"地灵活复用。

## 动手：一套标准的微调代码

道理讲完，我们用 PyTorch 把"分组学习率 + warm-up + 逐层解冻"这套组合拳写出来。为了聚焦微调本身，这里以一个预训练的 ResNet18 接新分类头为例。

第一步，加载预训练模型，先把整个主干**冻住**，只换上一个适配新任务的新分类头：

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 先把所有预训练参数冻住：不参与梯度更新
for p in model.parameters():
    p.requires_grad = False

# 换上一个新的分类头（随机初始化），假设新任务是 10 类
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)   # 新头默认 requires_grad=True
```

第二步，搭一个**分组学习率**的优化器。我们把网络粗分成几组，越靠近输出给越大的学习率；注意这里先只让新头有学习率，主干暂时是冻的：

```python
# 把参数按"离输出的远近"分组，给不同的学习率
optimizer = torch.optim.AdamW([
    {"params": model.fc.parameters(),     "lr": 1e-3},   # 新头：学得最快
    {"params": model.layer4.parameters(), "lr": 5e-4},   # 顶部主干：稍慢（待会解冻后生效）
    {"params": model.layer3.parameters(), "lr": 2e-4},   # 更深：更慢
], weight_decay=1e-4)
```

第三步，给学习率配上 **warm-up + 衰减**的调度。PyTorch 里可以用 `LinearLR`（线性预热）接 `CosineAnnealingLR`（余弦衰减），再用 `SequentialLR` 把两段拼起来：

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.01, total_iters=500)   # 前 500 步从 1% 线性升到 100%
decay  = CosineAnnealingLR(optimizer, T_max=5000)                  # 之后余弦衰减到接近 0
scheduler = SequentialLR(optimizer, [warmup, decay], milestones=[500])
```

第四步，**逐层解冻**：训练循环里，先训新头若干轮，到了预定的轮次再把更深的层放开。下面用一个辅助函数表达这个意思：

```python
def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

for epoch in range(num_epochs):
    if epoch == 2:
        unfreeze(model.layer4)   # 第 2 轮起，解冻顶部主干
    if epoch == 4:
        unfreeze(model.layer3)   # 第 4 轮起，再往下解冻一层

    for x, y in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()         # 每个 step 走一下学习率调度
```

把这四步连起来你会发现，整套微调的精髓就是**"控制节奏"**：先稳住新头（冻结 + warm-up），再分层、分批地、用各自合适的小学习率慢慢放开主干。节奏对了，预训练知识就既被保住、又被悄悄改进了。

如果你用的是 Hugging Face 生态，这套东西大多已经被封装好了——`transformers` 的 `Trainer` 自带 warm-up 调度，而 PEFT/LoRA 微调只要几行 `peft` 配置即可，原理却完全是上面这一套。

## 容易踩的坑

- **忘了 warm-up 就放大学习率**：随机初始化的新头会喷出巨大梯度，第一步就可能把预训练主干冲毁。微调里 warm-up 几乎是默认配置，别省。
- **解冻了却忘了把它加进优化器**：`requires_grad=True` 只是允许它求梯度，如果这组参数当初没被放进 `optimizer` 的参数组，它依然不会被更新。两件事要对上。
- **学习率整体偏大**：微调的学习率通常要比从头训练小一个数量级（常见 $10^{-5}\sim10^{-4}$）。沿用从头训练的大学习率，是新手最常见的"越训越差"。
- **数据太少还全量解冻**：参数远多于样本，几乎必然过拟合。数据少时老老实实多冻几层。
- **BatchNorm 的统计量没处理好**：冻结主干时，BatchNorm 层的均值方差是否更新会影响结果。任务分布和原数据差异大时，常需要让 BN 在新数据上重新统计（`model.train()` 与 `eval()` 切换要想清楚）。

## 它在后面会怎么用到

- 微调是用好[预训练大模型（LLM）](3-4-4-pretrained-llms.md)的核心手段，LoRA/PEFT 正是当下让一块普通显卡也能"调"大模型的关键技术。
- [视觉-文本编码器 CLIP](3-3-4-clip.md) 同样常以微调或加适配器的方式迁移到下游分类、检索任务。
- warm-up、分层学习率属于训练调参的一部分，更系统的实验与调参管理见 [1.2 调参与实验管理](../ch4-camp/4-1-2-experiment-management.md)；它和 [2.5 优化与正则化](../ch2-round1/2-2-5-optimization-regularization.md) 里的学习率调度、权重衰减一脉相承。

## 练习

??? note "基础练习"
    1. 用一句话向同学解释"迁移学习"和"微调"的区别，并各举一个你冻结多少层的具体场景。
    2. 设顶层学习率 $\eta=2\times10^{-3}$、逐层折扣 $\gamma=0.8$，算出从顶往下第 0、1、2、3 层各自的学习率。
    3. 把上面的微调代码改成"只训练新头、主干全程冻结"的最简版，并说明这等价于哪一种迁移学习做法。

??? note "进阶练习"
    1. 自己写一个 warm-up 调度：前 $T_w$ 步学习率从 0 线性升到 $\eta$，之后按 $\eta\cdot\cos$ 衰减；用一个简单循环把每一步的学习率打印或画出来，对照图 1.3-1 验证形状。
    2. 查一查 LoRA 里"秩 $r$"取大取小分别意味着什么：$r$ 增大对可训练参数量、对拟合能力、对过拟合风险各有什么影响？
    3. 设计一个对照实验：同一个预训练模型、同一份小数据集，分别用"全冻只训头""逐层解冻""LoRA"三种方式微调，比较它们的准确率、显存占用和训练时间，并解释结果。

## 小结

一句话总结：**微调就是迁移学习里"让预训练权重也小步参与训练"的那条路，而它能不能成功，全看你会不会控制节奏——warm-up 稳住开局，分层学习率因层施教，逐层解冻分批放开，数据不够时还能用 adapter/LoRA 只动一丁点参数。**

想深入，原始的 warm-up 与微调实践可读 ULMFiT 论文（Howard & Ruder, 2018，提出了分层学习率与逐层解冻）；参数高效微调推荐看 [LoRA 论文](https://arxiv.org/abs/2106.09685) 与 Hugging Face 的 [PEFT 文档](https://huggingface.co/docs/peft)，里面有大量可直接套用的微调范例。
