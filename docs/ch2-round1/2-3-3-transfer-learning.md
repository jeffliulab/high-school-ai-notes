# 3.3 迁移学习与预训练编码器

> **难度** ⭐⭐⭐☆☆ · **前置**：[3.2 图像分类实战](2-3-2-image-classification.md)、[3.1 卷积与池化](2-3-1-conv-pooling.md)

!!! abstract "读完这一节，你会"
    - 说清"预训练 + 微调"为什么有效，以及它和从零训练的差别在哪
    - 用 PyTorch 加载现成的 ResNet / VGG，把最后一层换成适配你自己任务的分类头
    - 区分"冻结特征提取"和"整体微调"两种做法，知道各自在什么时候用
    - 明白 Round 1 资格赛里碰到小数据集图像题时，为什么迁移学习几乎是默认选择

## 为什么不从零训练，而要"借"一个别人练好的模型

上一节我们从零搭了一个 CNN 来做图像分类，你大概也体会到了：要想让网络真正学出点东西，得有**很多**带标签的图片，还得在上面训练**很久**。可竞赛和现实里，你拿到的往往只有几百张、几千张图，CPU 上还跑不动太大的网络。这时候从头训练一个深网，几乎注定过拟合——参数太多，数据太少，网络只会把训练集"背"下来。

那怎么办？这里有一个非常关键、也非常省事的思路：**别从零开始，去借一个别人已经在海量数据上练好的模型，把它学到的本事拿过来用。** 这就是**迁移学习（transfer learning）**——把在一个任务上学到的知识，"迁移"到另一个相关任务上。

它为什么能成立？关键在于一个观察：**视觉特征是通用的。** 一个在 ImageNet（一个有一百多万张图、一千个类别的大数据集）上训练好的网络，它前面那些卷积层学会的，并不是"怎么认猫"这种很具体的东西，而是更底层、更通用的视觉零件——边缘、纹理、颜色块，再往上是角点、圆环、网格、毛发的质感……这些零件不管你最后要认猫、认狗、认 X 光片还是认手写棋子，**都用得上**。既然这些底层零件别人已经在海量数据上替你练好了，你又何必拿自己那点可怜的数据重头再练一遍呢？

## 网络的两半：通用的"特征提取器"和专用的"分类头"

要理解迁移学习怎么操作，先得把一个分类网络拆成两半来看。

你可以把一个像 ResNet 这样的 CNN 想成两段流水线。**前面一长串卷积层**负责把一张原始图片，一层层地榨成一个浓缩的特征向量——它读懂了图里"有什么纹理、有什么形状、有什么结构"。这一段我们叫它**特征提取器**，更专业的叫法是**编码器（encoder）**，因为它把图像"编码"成了一串数字。**最后那一两层全连接层**，才是真正下判断的地方：它拿着这个特征向量，输出"这是第几类"。这一段叫**分类头（classification head）**。

关键就在这儿：编码器学到的是通用视觉知识，**换个任务还能用**；而分类头是和具体类别绑死的——人家在 ImageNet 上训练出来的分类头输出 1000 个类，你的任务可能只有 5 个类，对不上。所以迁移学习的标准动作就一句话：**留下编码器，换掉分类头。**

<div class="diagram">
<svg viewBox="0 0 420 170" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 输入图 -->
  <rect x="14" y="64" width="40" height="40" rx="3" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="34" y="122" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">输入图片</text>
  <!-- 编码器（保留，蓝色） -->
  <rect x="78" y="48" width="190" height="72" rx="4" fill="var(--dia-accent-soft)" stroke="var(--dia-blue)" stroke-width="2"/>
  <text x="173" y="80" text-anchor="middle" font-size="13" fill="var(--dia-blue)">预训练编码器（保留 · 冻结）</text>
  <text x="173" y="100" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">边缘 → 纹理 → 形状 → 结构</text>
  <!-- 特征向量 -->
  <line x1="268" y1="84" x2="300" y2="84" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="300" y="42" text-anchor="middle" font-size="10" fill="var(--dia-stroke-soft)">特征向量</text>
  <!-- 分类头（替换，橙色） -->
  <rect x="300" y="56" width="100" height="56" rx="4" fill="none" stroke="var(--dia-accent)" stroke-width="2" stroke-dasharray="5 3"/>
  <text x="350" y="80" text-anchor="middle" font-size="12" fill="var(--dia-accent)">新分类头</text>
  <text x="350" y="98" text-anchor="middle" font-size="11" fill="var(--dia-accent)">（替换 · 训练）</text>
</svg>
</div>
<p class="figure-caption">图 3.3：迁移学习的核心动作——保留预训练编码器，只把末端的分类头换成适配自己任务的新层。</p>

## 冻结还是解冻：两种微调策略

换掉分类头之后，还有一个要紧的选择：**前面那个借来的编码器，训练时到底动不动它？** 这就分出了两种常见做法，它们适合的场景不一样，我们一个个说。

第一种叫**特征提取（feature extraction）**，也就是把编码器**冻结（freeze）**起来。冻结的意思是，训练时不让这部分参数更新——它就纯粹当一个"现成的、固定的"特征转换器用，反向传播的梯度只流到新的分类头里。这样做的好处是**快、稳、省**：要训练的参数极少，CPU 上也跑得动，数据再少也不容易过拟合。当你的数据集很小、而且和 ImageNet 里的自然图像比较像时，优先选它。

第二种叫**整体微调（fine-tuning）**，也就是把编码器**解冻**，让它和分类头一起继续训练，只不过用一个**很小的学习率**。为什么学习率要小？因为编码器里那些权重是别人辛辛苦苦练好的"宝贝"，你只想轻轻地、小幅地调整它去适应你的新数据，而不是用大步子把它原有的知识一脚踹乱。当你的数据相对充裕、或者你的图片和自然图像差得比较远（比如医学影像、卫星图）时，解冻微调通常能再榨出几个点的准确率。

!!! tip "一个实用的折中"
    很多人会先**冻着编码器训练几轮**，让新分类头先"热身"到一个合理状态；然后再**解冻最后一两个卷积块**，配一个小学习率一起微调。这样既稳又能涨点，是竞赛里很常见的套路。记住一个原则：**离输入越近的层越通用、越该冻住；离输出越近的层越专用、越值得调。**

## 动手：用 PyTorch 加载并改造一个 ResNet

道理讲完，我们来真刀真枪地写一遍。任务设定成：你有一个只分 **5 类** 的小图像数据集，想借 ImageNet 上预训练好的 **ResNet-18** 来做。

第一步，把预训练模型加载进来。`torchvision` 已经帮我们打包好了模型结构和训练好的权重，一行就能拿到：

```python
import torch
import torch.nn as nn
from torchvision import models

# weights=... 会自动下载并加载在 ImageNet 上预训练好的参数
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
```

第二步，**冻结整个编码器**。做法是遍历模型的所有参数，把它们的 `requires_grad` 关掉——这等于告诉 PyTorch："这些参数别算梯度、别更新。"

```python
for param in model.parameters():
    param.requires_grad = False        # 全部冻结，先当固定特征提取器用
```

第三步，**换掉分类头**。ResNet 最后那层全连接层叫 `model.fc`，原本输出 1000 个 ImageNet 类别。我们先读出它的输入维度，再造一个新的、输出 5 类的全连接层把它替换掉。注意：新建的层默认 `requires_grad=True`，所以**只有它会参与训练**，正合我们意。

```python
num_features = model.fc.in_features    # 读出编码器吐出的特征维度（ResNet-18 是 512）
model.fc = nn.Linear(num_features, 5)  # 换成输出 5 类的新分类头
```

第四步，**训练时只把"需要更新的参数"交给优化器**。这是个容易忽略的细节：如果你把 `model.parameters()` 全丢给优化器，被冻结的参数虽然不更新，但有些优化器仍会做无谓的动作。更干净的写法是只筛出 `requires_grad` 为真的那些：

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 之后就是标准训练循环（和 3.2 图像分类实战里完全一样）
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)            # 前向：图片 → 特征 → 5 类得分
    loss = criterion(outputs, labels)
    loss.backward()                    # 梯度只会流进新分类头
    optimizer.step()
```

你会发现，即便在 CPU 上、即便只有几百张图，这个模型也能很快收敛到一个相当不错的准确率——**因为真正"难学"的视觉特征，早就由别人在 ImageNet 上替我们学好了，我们只是在最后一层上"贴了个标签"而已。**

如果想从"冻结"切换到"解冻微调"，改动也很小：把编码器的 `requires_grad` 重新打开，并换一个**小得多**的学习率。

```python
for param in model.parameters():
    param.requires_grad = True         # 解冻：让编码器也一起微调
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 学习率调到很小，轻轻地调
```

VGG 系列也是一样的思路，只是分类头的名字不同——VGG 的最后一层是 `model.classifier[6]`，把它换成 `nn.Linear(4096, 5)` 即可。换个模型，换个分类头的位置，套路完全一致。

## 容易踩的坑

- **预处理必须和预训练时一致。** 模型在 ImageNet 上训练时，输入图片被缩放到 224×224、并按特定的均值方差做了标准化。你喂自己的图时如果不照着做同样的归一化，编码器拿到的就是"它没见过的分布"，效果会莫名其妙地差。`torchvision` 的 `weights.transforms()` 直接给了配套的预处理，照搬即可。
- **冻结了却忘了换优化器。** 如果你先全冻结、又把 `model.parameters()` 整个丢给优化器，再后来解冻，优化器里记录的还是旧参数列表，新解冻的层不会被训练。改了 `requires_grad` 后，记得重新建一次优化器。
- **微调时学习率太大。** 解冻后还用 `1e-3` 这种"从零训练"的学习率，会把预训练权重冲垮，准确率不升反降。微调请用 `1e-5` 量级。
- **类别数没改对。** 新分类头的输出维度必须等于你自己任务的类别数，否则交叉熵会直接报维度错误。
- **小数据还硬要解冻。** 数据只有几百张时贸然解冻整个大网络，等于又把过拟合的风险请了回来。这种情况老老实实冻着用。

## 它在后面会怎么用到

- 这里讲的"借编码器 + 换头"是一切**微调**的雏形。到了 Round 2，[1.3 模型微调](../ch3-round2/3-1-3-finetuning.md) 会把它推广到更大的模型，并讲 GPU 上的实战技巧。
- 编码器吐出的那个"特征向量"，其实就是一种**嵌入表示**，[1.2 嵌入表示](../ch3-round2/3-1-2-embeddings.md) 会专门讲它能拿来做什么。
- 当编码器不只学图像、还学会把"图像和文字"对齐时，就成了威力更大的多模态编码器，那就是 [3.4 视觉-文本编码器 CLIP](../ch3-round2/3-3-4-clip.md) 的故事。
- 想回顾"为什么大网络在小数据上容易过拟合、怎么缓解"，可看 [2.5 优化与正则化](2-2-5-optimization-regularization.md)。

## 练习

??? note "基础练习"
    1. 用 `torchvision` 加载预训练的 ResNet-18，打印 `model.fc`，确认它的输入维度是 512、输出是 1000。然后把它替换成输出 3 类的新层，再打印一次验证。
    2. 写两行代码，统计这个模型里"可训练参数"占总参数的比例（提示：`sum(p.numel() for p in ... if p.requires_grad)`）。先冻结编码器统计一次，再解冻统计一次，对比两个数字，体会冻结到底省下了多少。

??? note "进阶练习"
    1. 在同一个小数据集上做对照实验：(a) 从零训练一个 ResNet-18；(b) 冻结编码器只训分类头；(c) 解冻微调。记录三者在相同轮数下的验证准确率和单轮耗时，并解释为什么会出现这样的差距。
    2. 实现"先冻结热身、再解冻微调"的两阶段训练：前 3 轮冻着编码器、用 `1e-3`，之后解冻最后一个卷积块、换 `1e-5`。观察解冻那一刻准确率曲线的变化。

## 小结

一句话记住迁移学习：**别从零造轮子，留下别人练好的编码器，只换掉并训练那个适配你任务的分类头；数据少就冻着用，数据够就用小学习率轻轻微调。** 这是 Round 1 处理小数据图像题最划算的一招。

想看更系统的官方教程，PyTorch 的 [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 用一个"蚂蚁 vs 蜜蜂"的小数据集把这一整套流程完整跑了一遍，强烈建议照着敲一遍。
