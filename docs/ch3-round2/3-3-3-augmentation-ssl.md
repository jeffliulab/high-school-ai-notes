# 3.3 数据增强与自监督

> **难度** ⭐⭐⭐☆☆ · **前置**：[3.2 图像分类实战](../ch2-round1/2-3-2-image-classification.md)、[3.1 卷积与池化](../ch2-round1/2-3-1-conv-pooling.md)

!!! abstract "读完这一节，你会"
    - 说清楚为什么"标签太少、数据太贵"是深度学习的常态，以及数据增强和自监督各自怎么对症下药
    - 会用 torchvision 写出翻转、旋转、裁剪、色彩抖动这几样常用增强，并知道哪些能用、哪些会把标签弄坏
    - 理解 Mixup 与 CutMix 是怎么"把两张图混在一起"的，以及它们为什么能让模型更稳
    - 用大白话讲明白对比学习（contrastive learning）和 SimCLR 的核心思路：靠"猜两张图是不是同一张"来学表示

## 模型很贪吃，可标签很贵

我们先聊一个绕不开的现实问题，它正是这一整节的动机。

深度神经网络是出了名的"数据贪吃鬼"——参数动辄上千万，你喂得越多，它学得越好。可问题在于，有用的数据往往不是图片本身，而是图片上那个**标签**（label）。一张猫的照片到处都是，但"这张图里是一只猫"这句标注，得有人一张张地看、一张张地标，又慢又贵。在医学影像这种场景里，标一张图甚至需要请专科医生，成本高到离谱。

于是我们就被夹在中间：模型想要海量带标签的数据，现实却只给得起一小撮。怎么办？这一节给出两条互补的思路。

**第一条叫数据增强（data augmentation）**：手上的带标签图片就这么多，那我们能不能把每一张"变出好几张"？把一张猫的照片左右翻转、稍微旋转、裁掉一个角——它**还是一只猫**，标签没变，但对模型来说已经是一张"没见过"的新图了。这等于免费扩充了训练集。

**第二条叫自监督学习（self-supervised learning，常简称 SSL）**：既然标签贵，那我们干脆**先不用标签**。海量的无标签图片不要钱，我们想办法让模型从这些图片自己跟自己玩的"小游戏"里，学到一套好用的特征表示；等真要做分类时，再用那一小撮带标签数据轻轻一调就行。

下面这张图把这两条路并排画了出来，你先有个整体印象：

<div class="diagram">
<svg viewBox="0 0 420 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 左：数据增强 -->
  <rect x="20" y="70" width="46" height="46" rx="3" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="43" y="98" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">原图</text>
  <rect x="92" y="34" width="40" height="40" rx="3" fill="none" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <rect x="92" y="82" width="40" height="40" rx="3" fill="none" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <rect x="92" y="130" width="40" height="40" rx="3" fill="none" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <line x1="68" y1="86" x2="90" y2="54" stroke="var(--dia-accent)" stroke-width="1.2"/>
  <line x1="68" y1="93" x2="90" y2="102" stroke="var(--dia-accent)" stroke-width="1.2"/>
  <line x1="68" y1="100" x2="90" y2="150" stroke="var(--dia-accent)" stroke-width="1.2"/>
  <text x="80" y="190" text-anchor="middle" font-size="12" fill="var(--dia-accent-deep)">数据增强：1 张变多张（标签不变）</text>
  <!-- 分隔 -->
  <line x1="215" y1="20" x2="215" y2="170" stroke="var(--dia-rule)" stroke-width="1"/>
  <!-- 右：自监督 -->
  <text x="320" y="40" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">海量无标签图片</text>
  <rect x="260" y="58" width="120" height="44" rx="4" fill="var(--dia-accent-soft)" stroke="var(--dia-blue)" stroke-width="1.5"/>
  <text x="320" y="85" text-anchor="middle" font-size="11" fill="var(--dia-blue)">自己出题、自己学</text>
  <rect x="270" y="120" width="100" height="36" rx="4" fill="none" stroke="var(--dia-green)" stroke-width="1.5"/>
  <text x="320" y="142" text-anchor="middle" font-size="11" fill="var(--dia-green)">学到好特征</text>
  <line x1="320" y1="102" x2="320" y2="120" stroke="var(--dia-stroke-tertiary)" stroke-width="1.2"/>
  <text x="320" y="190" text-anchor="middle" font-size="12" fill="var(--dia-blue)">自监督：无标签也能学表示</text>
</svg>
</div>
<p class="figure-caption">图 3.3-1：两条对付"标签太贵"的路。左边把每张带标签图变出多张；右边干脆先用无标签图学特征。</p>

## 数据增强：给模型"换个角度看同一只猫"

我们先从更简单、也更常用的数据增强讲起。

它背后的道理特别朴素：你想让模型认得出猫，可现实中的猫会出现在各种角度、各种光线、画面里的各个位置。如果训练时只给它看正襟危坐、居中、光线良好的猫，那它一遇到歪着脑袋、偏在角落、光线昏暗的猫就傻眼了——这就是**过拟合**：模型把训练集背得太死，一换情况就翻车。

数据增强做的，就是在训练时**故意给图片加上这些"无关紧要的变化"**，逼着模型去抓住"猫之所以是猫"的本质特征，而不是死记某张图的像素。常见的几招是这样的：

- **水平翻转（horizontal flip）**：左右镜像一下。一只朝左的猫翻过来朝右，还是猫，标签不变——这几乎是图像任务里最安全、最有效的一招。
- **随机旋转（rotation）**：把图转个小角度，比如正负 15 度，模拟拍照时手抖、歪头。
- **随机裁剪（random crop）**：从图里抠出一块再放大，模拟物体出现在画面不同位置、不同大小。
- **色彩抖动（color jitter）**：随机改一改亮度、对比度、饱和度、色调，模拟不同光照和相机。

不过这里有个**你必须当心的前提**：增强不能把标签弄坏。水平翻转猫没问题，可如果你做的是手写数字识别，把数字 "6" 上下翻转就成了 "9"，标签彻底错了；旋转字母 "b" 太多也会变成 "p"。所以**增强的种类得看任务来挑**——这是个常考点，下面写代码时我们还会强调一次。

道理讲清楚了，来看 torchvision 里怎么写。它把这些增强都做成了可以"串起来"的流水线：

```python
import torch
from torchvision import transforms

# 训练时用的增强流水线：图片每次被读取，都会随机经历这一串变换
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),          # 随机裁一块再缩放到 224×224（裁剪+缩放二合一）
    transforms.RandomHorizontalFlip(p=0.5),     # 50% 概率左右翻转
    transforms.RandomRotation(degrees=15),      # 在 ±15° 内随机旋转
    transforms.ColorJitter(                     # 色彩抖动
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),                      # 转成张量，像素值缩到 [0,1]
    transforms.Normalize(                       # 按 ImageNet 统计量做标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# 验证/测试时：不要随机增强！只做确定性的缩放和标准化
test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

这段代码里有两个细节值得你停下来想一想。

第一，**增强是"在线"的、随机的**：同一张图片，每个 epoch 被读到时经历的翻转角度、裁剪位置都不一样，所以模型几乎每次看到的都是"新图"。第二，**验证和测试时一定要关掉随机增强**——你评估模型表现时，希望每次输入都一样、结果可复现；如果测试时还在随机翻转旋转，那分数就成了"看运气"。这是新手特别容易犯的错。

## Mixup 与 CutMix：把两张图揉在一起

上面那些增强都还停留在"改一张图"。接下来这两招更大胆，它们直接**把两张图、连同两个标签一起混合**，效果出奇地好。

先说 **Mixup**。它的想法简单到有点离谱：随便取两张图 $x_1$、$x_2$ 和它们的标签 $y_1$、$y_2$，按一个随机比例 $\lambda\in[0,1]$ 把它们**线性地加权平均**，得到一张"半透明叠加"的新图，标签也按同样的比例混合：

$$
\tilde{x}=\lambda x_1+(1-\lambda)x_2,\qquad
\tilde{y}=\lambda y_1+(1-\lambda)y_2.
$$

你可能会皱眉：把一只猫和一条狗按 0.7 和 0.3 叠在一起，这图看着像鬼影，能学到什么？关键就在标签上——我们告诉模型"这张图 70% 是猫、30% 是狗"。这逼着模型的输出变得**平滑而不绝对**：它不会再对任何一张图都狂妄地喊"100% 是猫"，从而不容易过拟合，对没见过的数据也更稳。

**CutMix** 是 Mixup 的"剪贴版"。它不做半透明叠加，而是从图 $x_2$ 上**抠下一个矩形块，贴到图 $x_1$ 上**，盖住的那块对应的图。标签则按"被贴上的块占了多大面积"来混合：如果贴上去的块占了整张图的 30%，那标签就是 70% 的 $y_1$ 加 30% 的 $y_2$。比起 Mixup 糊成一团，CutMix 生成的图更"清晰"，还顺带逼着模型别只盯着物体最显眼的那一处，而要学会看局部线索。

这两招在 Round 2 的图像分类题里非常实用——很多刷榜的 baseline 都默认开着它们。下面是 Mixup 的核心实现，短短几行，你能看清它到底在干嘛：

```python
import numpy as np
import torch

def mixup_batch(x, y, alpha=0.2):
    """对一个 batch 做 Mixup。x: 图像张量 (N,C,H,W)，y: 标签 (N,)。"""
    lam = np.random.beta(alpha, alpha)     # 从 Beta 分布采一个混合比例 λ
    idx = torch.randperm(x.size(0))        # 把这一批图随机打乱成"另一组"
    mixed_x = lam * x + (1 - lam) * x[idx] # 两张图按 λ 线性叠加
    y_a, y_b = y, y[idx]                   # 对应的两套标签
    return mixed_x, y_a, y_b, lam

# 训练时，损失也要对两套标签按 λ 加权：
def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

注意看 `mixed_x` 那一行：我们没有真的去配对图片，而是把整批图复制一份、打乱顺序，再和原批整体相加——这是个很常见的向量化技巧，省掉了写循环。损失函数那行同样体现了"标签也要混合"：我们对两套标签分别算损失，再按 $\lambda$ 加权，等价于用混合标签 $\tilde y$ 去算。

## 自监督：没有标签，模型怎么自学？

现在进入这一节最有意思、也是 Round 2 的"高阶味道"所在的部分——自监督学习。

我们前面说了，自监督的目标是**不用人工标签，也能学到一套好用的特征表示**。这听起来像变魔术，关键在于一句话：**让模型自己出题、自己批改。** 既然图片里本来就藏着结构，我们就设计一个"小游戏"，游戏的答案能从图片本身自动得到，不需要人来标。模型为了把这个游戏玩好，就被迫学会理解图像内容。

历史上人们试过各种小游戏，比如"把图片打乱成拼图块让模型还原"、"把图片转个角度让模型猜转了多少度"。但真正让自监督一战成名的，是一类叫**对比学习（contrastive learning）**的方法。它的核心思想，用一句大白话就能说清：

> **同一张图的两个"变体"，应该长得像；不同图，应该长得不像。**

怎么理解？拿一张猫的照片，用上一节学的数据增强，对它做两次**不同的随机增强**——一次裁左上角加翻转，一次裁右下角加色彩抖动。这两张图看起来不一样，但它们来自同一只猫，所以我们希望模型把它们编码成的特征向量**靠得很近**。而对于另一张完全不同的图（比如一辆车），我们希望它的特征**离这只猫远远的**。模型为了同时满足"自己的两个变体要近、跟别人要远"，就不得不去抓住图像真正的语义内容。

这正是 **SimCLR** 这套方法的骨架。我们把它的流程拆成四步看：

1. **造正样本对**：取一张图，做两次不同的随机增强，得到 $x_i$ 和 $x_j$。它俩是一对"正样本（positive pair）"——本该相似。
2. **编码**：用同一个卷积网络（比如 ResNet）把它们各自压成一个特征向量。
3. **投影**：再过一个小的全连接网络，把特征映射到一个专门用来比较的空间，得到 $z_i$、$z_j$。
4. **对比**：在一个 batch 里，让 $z_i$ 和它的正搭档 $z_j$ 尽量靠近，同时和 batch 里**所有其他图**（它们都算"负样本"）尽量远离。

第四步用的损失叫 **InfoNCE**（也叫对比损失），它本质上是一个"$2N$ 选 1"的分类问题：给定 $z_i$，要从一大堆候选里把它真正的搭档 $z_j$ 挑出来。写成公式是

$$
\ell_{i}=-\log
\frac{\exp\!\big(\operatorname{sim}(z_i,z_j)/\tau\big)}
{\displaystyle\sum_{k\neq i}\exp\!\big(\operatorname{sim}(z_i,z_k)/\tau\big)}.
$$

别被它吓到，拆开看其实很亲切。分子是"$z_i$ 和它正搭档 $z_j$ 的相似度"，分母是"$z_i$ 和 batch 里所有别人相似度的总和"——这不就是个 softmax 嘛！我们希望分子在分母里占的比重越大越好，也就是"真搭档"在所有候选里最突出。其中 $\operatorname{sim}$ 一般用余弦相似度，$\tau$ 是个叫**温度（temperature）**的小参数，用来调节模型对"区分难易"的敏感度。

下面这张图把"两个变体要近、跟负样本要远"这件事画了出来：

<div class="diagram">
<svg viewBox="0 0 420 190" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ssa" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 原图 -->
  <rect x="24" y="78" width="40" height="40" rx="3" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="44" y="135" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">一张图</text>
  <!-- 两次增强 -->
  <rect x="100" y="44" width="34" height="34" rx="3" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <rect x="100" y="112" width="34" height="34" rx="3" fill="var(--dia-accent-soft)" stroke="var(--dia-accent)" stroke-width="1.5"/>
  <line x1="64" y1="92" x2="98" y2="64" stroke="var(--dia-stroke-tertiary)" stroke-width="1.2"/>
  <line x1="64" y1="104" x2="98" y2="128" stroke="var(--dia-stroke-tertiary)" stroke-width="1.2"/>
  <text x="117" y="98" text-anchor="middle" font-size="9" fill="var(--dia-accent-deep)">两次增强</text>
  <!-- 特征空间 -->
  <circle cx="300" cy="95" r="64" fill="none" stroke="var(--dia-rule)" stroke-width="1"/>
  <circle cx="290" cy="78" r="5" fill="var(--dia-accent)"/>
  <circle cx="304" cy="86" r="5" fill="var(--dia-accent)"/>
  <text x="297" y="68" text-anchor="middle" font-size="9" fill="var(--dia-accent-deep)">拉近</text>
  <circle cx="340" cy="130" r="5" fill="var(--dia-blue)"/>
  <circle cx="262" cy="128" r="5" fill="var(--dia-blue)"/>
  <text x="300" y="155" text-anchor="middle" font-size="9" fill="var(--dia-blue)">负样本：推远</text>
  <line x1="160" y1="80" x2="232" y2="88" stroke="var(--dia-accent)" stroke-width="1.3" marker-end="url(#ssa)"/>
  <text x="44" y="178" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">图像空间</text>
  <text x="300" y="178" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">特征空间</text>
</svg>
</div>
<p class="figure-caption">图 3.3-2：对比学习把同一张图的两个增强变体在特征空间里拉近（橙点），把别的图推远（蓝点）。</p>

学完之后，那个用来比较的投影头就被丢掉了，我们保留前面的编码网络——它现在已经学会了把图像变成有意义的特征。接下来只要拿一小撮带标签数据，在它后面接一个简单分类器轻轻一训（这一步和 [1.3 模型微调](3-1-3-finetuning.md) 是一回事），就能得到一个很强的分类器。**这正是自监督的价值：把"贵的标注"用在了刀刃上。**

## 容易踩的坑

- **测试时还开着随机增强**：随机增强只属于训练阶段。验证和测试时必须换成确定性的 `Resize + CenterCrop`，否则分数会随机抖动、无法复现。
- **增强破坏了标签**：旋转、翻转要看任务。手写数字里 6 翻成 9、字母 b 转成 p，标签就错了；做这类任务前先想清楚哪些增强是"安全"的。
- **增强强度拉满**：不是越狠越好。色彩抖动、裁剪太激进会把图弄得连人都认不出，模型反而学不到东西。增强是为了"合理的变化"，不是"毁图"。
- **Mixup/CutMix 忘了混标签**：只混了图、标签还用原来的硬标签，等于喂了错答案。损失必须对两套标签按比例加权。
- **对比学习的 batch 太小**：SimCLR 的负样本来自同一个 batch，batch 越大、负样本越多、学得越好。batch 太小时效果会明显变差——这也是它对算力要求高的原因。
- **把 Normalize 的均值方差写错**：用预训练模型时，要用它当初训练所用的那套 `mean/std`（如 ImageNet 的），自己乱填会让输入分布对不上。

## 它在后面会怎么用到

- 数据增强是几乎所有视觉任务的标配，[目标检测](3-3-1-object-detection.md) 和 [图像分割](3-3-2-segmentation.md) 里也大量使用，只是要注意框和掩码得跟着图一起变换。
- 自监督学到的强编码器，正是 [1.3 模型微调](3-1-3-finetuning.md) 的理想起点；它产出的特征也就是 [1.2 嵌入表示](3-1-2-embeddings.md) 所说的 embedding。
- "用增强造正样本对、用对比损失对齐"这套思路，被 [视觉-文本编码器 CLIP](3-3-4-clip.md) 推广到了"图像配文字"上——CLIP 本质就是图文之间的对比学习，学完这一节再去看它会顺很多。
- 增强本质是一种正则化手段，背后的动机和 [2.5 优化与正则化](../ch2-round1/2-2-5-optimization-regularization.md) 里讲的"对抗过拟合"一脉相承。

## 练习

??? note "基础练习"
    1. 用 torchvision 搭一条增强流水线，对同一张图片连续读取 4 次并可视化，亲眼确认每次得到的图都不一样；再想一想：如果这是手写数字任务，上面四种增强里哪一种你绝不能用，为什么？
    2. 在 CIFAR-10 上训练一个小 CNN，分别在"不加增强"和"加翻转+随机裁剪"两种设置下跑相同的轮数，比较测试准确率。你应该会看到加了增强后验证集表现更稳、过拟合更轻。

??? note "进阶练习"
    1. 把上面的 `mixup_batch` 接进你的训练循环，在 CIFAR-10 上对比开 / 关 Mixup 的效果。再自己动手实现 CutMix（提示：随机生成一个矩形框，用切片把另一张图的对应区域贴过来，标签按框的面积比例混合）。
    2. （挑战）用一个小型 ResNet 在无标签的 CIFAR-10 上跑一个迷你版 SimCLR：每张图做两次增强、过编码器和投影头、用 InfoNCE 损失训练。训完后冻住编码器，只用 10% 的带标签数据训练一个线性分类器，看看它的准确率能不能逼近"全监督从头训"的水平。这会让你切身体会到自监督"省标签"的威力。

## 小结

- 一句话记住这一节：**标签贵，所以我们要么把现有数据"变多"（增强），要么干脆"先不用标签"自学（自监督）。**
- 数据增强用翻转、旋转、裁剪、色彩抖动等"无关变化"对抗过拟合，但要挑不会破坏标签的招数，且只在训练时开启。
- Mixup 和 CutMix 把两张图连标签一起混合，让模型输出更平滑、更稳健，是图像分类刷分的常用手段。
- 自监督靠"自己出题"学表示；对比学习（SimCLR）的核心是"同图的两个变体拉近、和别人推远"，学完的编码器再微调一下就很能打。

想把自监督的来龙去脉看得更透，推荐读 SimCLR 原论文 [*A Simple Framework for Contrastive Learning*](https://arxiv.org/abs/2002.05709)，以及 Lilian Weng 的博客 [*Self-Supervised Representation Learning*](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)——它把各路方法的脉络梳理得非常清楚。
