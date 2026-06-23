# 3.2 图像分割

> **难度** ⭐⭐⭐⭐☆ · **前置**：[3.1 目标检测](3-3-1-object-detection.md)、[3.1 卷积与池化](../ch2-round1/2-3-1-conv-pooling.md)

!!! abstract "读完这一节，你会"
    - 说清"语义分割"和"实例分割"到底有什么不同，以及各自适合什么任务
    - 看懂 U-Net 的"编码—解码 + 跳连"结构，明白每个部件解决的是什么问题
    - 会写 Dice 损失、会算 IoU 指标，并知道它们为什么比逐像素准确率更靠谱
    - 能用 PyTorch 搭一个最小可跑的 U-Net，并在分割任务上训练起来

## 从"框住物体"到"描出物体的轮廓"

上一节我们学了目标检测，它做的事情是给每个物体画一个**框**，告诉你"这里有一只猫，大概在这片矩形区域里"。但框是个很粗糙的回答——框里除了猫，还混着一堆背景。如果你想做的是把猫从背景里**精确抠出来**，比如自动驾驶里要分清"哪些像素是路、哪些是行人"，或者医学影像里要圈出肿瘤的**确切边界**，那一个矩形框就远远不够了。

这时我们需要的是更精细的回答：**对图像里的每一个像素，单独判断它属于哪一类。** 这件事就叫**图像分割（image segmentation）**。换句话说，分类是"给整张图一个标签"，检测是"给每个框一个标签"，而分割是"给每个像素一个标签"——精度一路往上走，难度也一路往上走。

你可以把分割想成"按区域涂色"：拿到一张街景照，模型把所有属于"马路"的像素涂成灰色、"行人"涂成红色、"天空"涂成蓝色。涂完之后，每个物体的轮廓自然就出来了，连弯弯曲曲的边界都能贴合。这正是分割比检测强大的地方。

## 语义分割 vs 实例分割：同样涂色，区别在"分不分得清两只猫"

刚才说的"按区域涂色"，其实还藏着一个细分。我们用一个例子来体会。

假设一张图里有两只猫并排坐着。**语义分割（semantic segmentation）**只关心"每个像素是什么类别"，所以它会把两只猫的像素**全涂成同一种"猫色"**——它知道这片区域是猫，但不区分这是第几只猫。而**实例分割（instance segmentation）**更进一步，它不光要知道"这是猫"，还要把**第一只猫和第二只猫分别涂成不同的颜色**，告诉你这是两个独立的个体。

一句话记住这个区别：**语义分割回答"是什么"，实例分割还要回答"是哪一个"。** 下面这张图把两者画在一起，你对照着看就清楚了：

<div class="diagram">
<svg viewBox="0 0 460 180" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 左：原图 -->
  <rect x="20" y="40" width="100" height="90" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <circle cx="50" cy="90" r="18" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <circle cx="92" cy="90" r="18" fill="none" stroke="var(--dia-stroke-soft)" stroke-width="1.5"/>
  <text x="70" y="158" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">原图：两只猫</text>
  <!-- 箭头 -->
  <line x1="128" y1="85" x2="158" y2="85" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <!-- 中：语义分割 -->
  <rect x="168" y="40" width="100" height="90" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <circle cx="198" cy="90" r="18" fill="var(--dia-accent)"/>
  <circle cx="240" cy="90" r="18" fill="var(--dia-accent)"/>
  <text x="218" y="158" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">语义分割：同一种"猫色"</text>
  <!-- 箭头 -->
  <line x1="276" y1="85" x2="306" y2="85" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <!-- 右：实例分割 -->
  <rect x="316" y="40" width="100" height="90" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <circle cx="346" cy="90" r="18" fill="var(--dia-accent)"/>
  <circle cx="388" cy="90" r="18" fill="var(--dia-blue)"/>
  <text x="366" y="158" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">实例分割：两只分开上色</text>
</svg>
</div>
<p class="figure-caption">图 3.2-1：同一张"两只猫"，语义分割把它们涂成同一类，实例分割则把每只单独区分开。</p>

那竞赛里会考哪个？两者都可能，但**语义分割是基础、也是更常考的入门形态**，因为它本质上就是"逐像素分类"，模型结构最干净。实例分割通常要在检测的基础上再加一个分割头（最有名的就是 Mask R-CNN），实现更复杂。所以本节我们把重点放在语义分割，把它吃透了，实例分割只是再叠一层逻辑而已。

## 难点在哪：分类网络为什么不能直接拿来分割

你可能会想：既然语义分割就是"逐像素分类"，那我把普通的图像分类网络（比如学过的 CNN）搬过来，对每个像素跑一遍分类不就行了？

这个想法方向对，但有个绕不过去的坎。回想一下分类网络长什么样：它靠**池化和带步长的卷积**不断把图像缩小，从 $224\times224$ 一路缩到 $7\times7$，最后接一个全连接层吐出一个类别。这个"缩小"的过程叫**下采样（downsampling）**，它的好处是能看到越来越大的范围、提取越来越抽象的语义——所以网络深处"知道"图里有只猫。**但坏处是，空间分辨率被压没了**：到了 $7\times7$，你已经说不清猫的边界具体在哪个像素了。

而分割恰恰要求输出和输入**一样大**：输入是 $224\times224$ 的图，输出也得是 $224\times224$ 的"类别图"，每个位置对应一个像素的标签。于是矛盾就来了——**我们既需要深层的、抽象的语义（知道这是猫），又需要浅层的、精细的位置（知道猫的边在哪）。** 怎么把这两样都拿到手，正是分割网络要解决的核心问题。

历史上第一个漂亮地回答这个问题的，是**全卷积网络（FCN，Fully Convolutional Network）**。它的思路简单又关键：**把分类网络末尾那个全连接层换成卷积层**，这样网络从头到尾全是卷积，就不再要求固定输入尺寸，输出也变回了一张二维的图。再把缩小后的特征图通过**上采样（upsampling）**放大回原尺寸，就得到了逐像素的预测。FCN 第一次让"端到端的逐像素分割"跑通了，后面所有分割网络都站在它的肩膀上。

不过 FCN 直接把很小的特征图一口气放大回去，边界会糊成一团。怎么把边界也修得锐利？这就轮到本节的主角 U-Net 登场了。

## U-Net：编码器抓语义，解码器还原细节，跳连补回边界

U-Net 这个名字来自它的形状——把结构图画出来，左边一路向下、右边一路向上，整体像个字母 **U**。它最早是为医学图像分割设计的，但因为又简单又好用，如今几乎是分割任务的"默认起手式"。我们顺着这个 U 字走一遍。

**左半边叫编码器（encoder），负责"看懂"。** 它就是我们熟悉的那套：卷积 + 池化，一层层把图像缩小、把通道（特征数）变多。每下一层，分辨率减半、语义更抽象。走到 U 字底部时，网络已经"理解"了整张图的内容，但空间细节也丢得差不多了。

**右半边叫解码器（decoder），负责"还原"。** 它做的事情和编码器相反：通过**上采样**（常用转置卷积或插值）把特征图一步步放大，分辨率翻倍、通道变少，最终恢复到和输入一样的尺寸，输出每个像素的类别。

但光是"缩小再放大"，丢掉的边界细节是补不回来的——这正是 FCN 边界糊的原因。U-Net 的点睛之笔是**跳连（skip connection，跳跃连接）**：它在每一层，把编码器那边**还没被缩小、保留着精细位置信息**的特征图，直接横跨过来，拼接到解码器对应的层上。这样一来，解码器在放大时，既有从底部带上来的"深层语义"，又有从跳连补回来的"浅层细节"——**两样都拿到了，前面那个核心矛盾就解开了。** 下面这张结构图把这条 U 字路线和跳连都画了出来：

<div class="diagram">
<svg viewBox="0 0 460 280" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ua" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
    <marker id="us" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <!-- 编码器（左，逐层变小） -->
  <rect x="40"  y="30"  width="60" height="40" fill="var(--dia-blue)" opacity="0.85"/>
  <rect x="52"  y="90"  width="48" height="34" fill="var(--dia-blue)" opacity="0.85"/>
  <rect x="62"  y="142" width="38" height="28" fill="var(--dia-blue)" opacity="0.85"/>
  <!-- U 底 -->
  <rect x="200" y="200" width="60" height="30" fill="var(--dia-stroke-soft)"/>
  <!-- 解码器（右，逐层变大） -->
  <rect x="360" y="142" width="38" height="28" fill="var(--dia-accent)" opacity="0.85"/>
  <rect x="360" y="90"  width="48" height="34" fill="var(--dia-accent)" opacity="0.85"/>
  <rect x="360" y="30"  width="60" height="40" fill="var(--dia-accent)" opacity="0.85"/>
  <!-- 下行箭头 -->
  <line x1="76" y1="74" x2="76" y2="86"  stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ua)"/>
  <line x1="76" y1="128" x2="80" y2="138" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ua)"/>
  <line x1="80" y1="172" x2="205" y2="200" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ua)"/>
  <!-- 上行箭头 -->
  <line x1="258" y1="200" x2="378" y2="172" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ua)"/>
  <line x1="382" y1="138" x2="382" y2="128" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ua)"/>
  <line x1="386" y1="86" x2="386" y2="74"  stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ua)"/>
  <!-- 跳连（橙色横向虚线） -->
  <line x1="100" y1="50"  x2="360" y2="50"  stroke="var(--dia-accent)" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#us)"/>
  <line x1="100" y1="107" x2="360" y2="107" stroke="var(--dia-accent)" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#us)"/>
  <line x1="100" y1="156" x2="360" y2="156" stroke="var(--dia-accent)" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#us)"/>
  <!-- 文字标注：全部放空白处 -->
  <text x="70"  y="20"  text-anchor="middle" font-size="12" fill="var(--dia-blue)">编码器（下采样）</text>
  <text x="390" y="20"  text-anchor="middle" font-size="12" fill="var(--dia-accent)">解码器（上采样）</text>
  <text x="230" y="252" text-anchor="middle" font-size="12" fill="var(--dia-stroke-soft)">U 字底部：语义最抽象</text>
  <text x="230" y="40"  text-anchor="middle" font-size="11" fill="var(--dia-accent)">跳连：把细节横跨补回来</text>
</svg>
</div>
<p class="figure-caption">图 3.2-2：U-Net 的左下右上像个 U 字；蓝色编码器抓语义，橙色解码器还原尺寸，橙色虚线的跳连把精细边界补回来。</p>

记住这个 U 字和三条横向虚线，你就抓住了 U-Net 的全部精髓。剩下的卷积层数、通道数都只是细节，可以随任务调整。

## 怎么评分割好不好：IoU 这个指标

模型搭好之前，我们得先想清楚一件事——**怎么判断分割结果是好是坏？** 这件事比想象中要小心。

一个最朴素的想法是看"逐像素准确率"：预测对的像素占多少。但这个指标有个致命问题。设想一张医学图，肿瘤只占了 2% 的像素，剩下 98% 全是背景。这时哪怕模型偷懒，**把整张图都判成"背景"、一个肿瘤像素都没找到**，它的逐像素准确率照样高达 98%！这显然是个废物模型，但准确率却给了它高分。所以**像素准确率在类别不平衡时会严重骗人**，分割任务基本不用它当主指标。

我们真正想衡量的是：**预测出来的那块区域，和真实区域"重合得有多好"。** 这就引出了分割的标准指标——**交并比（IoU，Intersection over Union）**。顾名思义，它是"交集除以并集"：把预测区域和真实区域**重叠的部分**（交集）算出来，再除以两者**合起来覆盖的总面积**（并集）：

$$\text{IoU}=\frac{|\,P\cap G\,|}{|\,P\cup G\,|}.$$

这里 $P$ 是预测的那组像素，$G$ 是真实标注（ground truth）的那组像素。IoU 的值落在 0 到 1 之间：完全不重合是 0，完全吻合是 1。它聪明在哪？回到刚才那个偷懒模型——它一个肿瘤像素都没预测，交集为 0，IoU 直接就是 0，骗不过去了。这就是为什么分割竞赛几乎都用 IoU（或它的多类平均版 **mIoU**）来排名。

## Dice 损失：让模型"学着去重合"

IoU 是评分用的指标，但它有个麻烦：交集、并集都涉及"数像素个数"这种不可导的操作，没法直接拿来对网络求梯度、做训练。我们需要一个"长得像 IoU、但可以求导"的**损失函数**来引导训练，这就是 **Dice 损失**。

它基于 **Dice 系数**，和 IoU 是近亲，同样衡量两块区域的重合度，只是公式略有不同——分子是交集的两倍，分母是两者面积之和：

$$\text{Dice}=\frac{2\,|P\cap G|}{|P|+|G|}.$$

为了能求导，我们把"是不是这一类"的硬判断（0 或 1）换成网络输出的**软概率** $p\in[0,1]$，于是交集 $|P\cap G|$ 就写成预测概率和真实标签逐像素相乘再求和 $\sum p_i g_i$。Dice **损失**就定义为"$1$ 减去 Dice 系数"——重合越好，Dice 越接近 1，损失越接近 0，正好是我们想最小化的方向：

$$L_{\text{Dice}}=1-\frac{2\sum_i p_i g_i+\epsilon}{\sum_i p_i+\sum_i g_i+\epsilon}.$$

那个小小的 $\epsilon$（比如 $10^{-6}$）是为了防止分母为零导致除零报错，是工程上的常规操作。**为什么 Dice 损失在分割里这么受欢迎？** 因为它天生对类别不平衡免疫——它只看"重合比例"，不管背景有多大，所以哪怕前景只占 2%，它也会逼着模型去把那 2% 找准。实战里，常见做法是把 Dice 损失和普通的交叉熵损失**加在一起用**，取两者之长。

## 动手：用 PyTorch 搭一个最小 U-Net

道理讲透了，我们来把它写成代码。先实现刚才说的 Dice 损失——把公式翻译成几行 PyTorch，你会发现它和数学式几乎一一对应：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred_logits, target, eps=1e-6):
    """pred_logits: (B,1,H,W) 网络原始输出；target: (B,1,H,W) 取值 0/1 的标注。"""
    pred = torch.sigmoid(pred_logits)          # 先压到 0~1 的软概率
    pred = pred.flatten(1)                      # 拉平成 (B, H*W)，方便逐像素相乘
    target = target.flatten(1)
    inter = (pred * target).sum(dim=1)          # 交集：∑ p_i * g_i
    union = pred.sum(dim=1) + target.sum(dim=1) # |P| + |G|
    dice = (2 * inter + eps) / (union + eps)    # Dice 系数
    return 1 - dice.mean()                       # 损失 = 1 - Dice，越小越好
```

注释里每一行都对得上前面的公式：先 `sigmoid` 把 logits 变成概率，再算交集、算面积和，最后 `1 - dice`。这就是 Dice 损失的全部。

接着搭网络本身。完整 U-Net 层数较多，我们写一个**缩小版**，只下采样两次再上采样两次，但"编码—解码 + 跳连"这套骨架一个不少——把这个看懂，加深它只是复制粘贴的事：

```python
def conv_block(in_ch, out_ch):
    """U-Net 的基本积木：两层 3×3 卷积 + ReLU。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
    )

class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.enc1 = conv_block(in_ch, 32)        # 编码器第 1 层
        self.enc2 = conv_block(32, 64)           # 编码器第 2 层
        self.pool = nn.MaxPool2d(2)              # 下采样：尺寸减半
        self.bottleneck = conv_block(64, 128)    # U 字底部
        # 上采样用转置卷积，尺寸翻倍
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)          # 输入 128 = 64(上采样) + 64(跳连)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)           # 输入 64 = 32(上采样) + 32(跳连)
        self.head = nn.Conv2d(32, num_classes, 1)  # 1×1 卷积出逐像素类别

    def forward(self, x):
        e1 = self.enc1(x)                # 保留 e1、e2 供跳连用
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(self.pool(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # 跳连：拼接编码器特征
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # 跳连：拼接更浅的特征
        return self.head(d1)
```

代码里最该盯住的，是 `forward` 里那两句 `torch.cat([..., e2], dim=1)` 和 `torch.cat([..., e1], dim=1)`——**这就是"跳连"在代码里的样子**：把编码器存下来的 `e1`、`e2`，沿通道维拼接到解码器对应的层上。正因为有这两句，解码器才同时拿到了语义和细节。把这一步去掉，网络就退化成了边界糊糊的 FCN。

最后，我们造一批假数据，确认整个流程能从前向跑到反向、损失能下降：

```python
model = TinyUNet(in_ch=3, num_classes=1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(2, 3, 64, 64)               # 2 张 64×64 的 RGB 图
y = (torch.rand(2, 1, 64, 64) > 0.5).float() # 随机的 0/1 掩码当标注

for step in range(3):
    pred = model(x)                          # 前向：输出和输入同尺寸 (2,1,64,64)
    loss = dice_loss(pred, y) + F.binary_cross_entropy_with_logits(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"step {step}  loss={loss.item():.4f}")
```

你会看到输出形状是 `(2, 1, 64, 64)`——和输入图一样大，这正是分割网络区别于分类网络的标志；损失也在一步步下降，说明"Dice + 交叉熵"这套配方把模型往正确方向推。真实任务里把假数据换成真实图像和标注掩码、把训练步数加大，骨架完全不用改。

至于库版方案：实战中你很少从零手搓 U-Net，社区有现成的轮子，比如 `segmentation-models-pytorch` 库，一行 `smp.Unet(encoder_name="resnet34", classes=1)` 就能拿到一个带预训练编码器的 U-Net——它把编码器换成了我们在 [3.3 迁移学习与预训练编码器](../ch2-round1/2-3-3-transfer-learning.md) 学过的 ResNet，开箱即用、效果还更好。理解了上面的骨架，你才知道这一行背后到底发生了什么。

## 容易踩的坑

- **拿像素准确率当主指标**：前景占比很小时它会严重虚高，一个全判背景的废物模型也能拿 98%。分割请用 **IoU / mIoU** 排名，训练用 **Dice（常配交叉熵）**。
- **忘了 `sigmoid` / `softmax`**：Dice 损失要喂概率，不能直接喂网络的原始 logits。二分类用 `sigmoid`，多类用 `softmax`，漏了这步损失会算错。
- **跳连尺寸对不齐**：要 `torch.cat` 拼接的两个特征图，高宽必须完全相同。如果输入尺寸不是 2 的整数次幂，池化后可能差一个像素，拼接就报错——常用 `padding` 或裁剪对齐。
- **Dice 分母忘加 $\epsilon$**：当某张图里某个类别一个像素都没有时，分母会是 0，导致 `nan`。务必加上一个小 $\epsilon$ 防除零。
- **混淆语义与实例分割**：题目要求"区分每一个个体"时，普通 U-Net 做不到，需要 Mask R-CNN 这类实例分割方法。读题先看清要的是哪一种。

## 它在后面会怎么用到

分割是计算机视觉的核心技能之一，它和你前后学的内容都连得上：

- 它直接建立在 [3.1 目标检测](3-3-1-object-detection.md) 之上——实例分割（如 Mask R-CNN）就是"检测框 + 框内再做一次分割"。
- U-Net 的"编码—解码"结构和 [2.3 自编码器](3-2-3-autoencoders.md) 是同一个家族，区别只在输出的是类别图而非重建图。
- 它的编码器通常用 [3.3 迁移学习与预训练编码器](../ch2-round1/2-3-3-transfer-learning.md) 里的预训练 ResNet，省数据又涨点。
- 整套卷积、池化、上采样的底层运算，回到 [3.1 卷积与池化](../ch2-round1/2-3-1-conv-pooling.md) 复习更踏实。
- 有意思的是，U-Net 还在 [3.5 生成模型 GAN·扩散](3-3-5-generative.md) 里以另一种身份出现——扩散模型的去噪网络几乎都用 U-Net 当骨架。

## 练习

??? note "基础练习"
    1. 手算 IoU：预测区域和真实区域各是一个 $10\times10$ 的方块，它们在水平方向错开了 5 格（即重叠一半）。算出这两块的交集像素数、并集像素数和 IoU。（提示：交集是 $5\times10$。）
    2. 用一句话分别说清"语义分割"和"实例分割"的区别，并各举一个现实里更适合用它的任务。
    3. 把上面 `TinyUNet` 的两句 `torch.cat` 跳连删掉，重新跑训练，对比损失下降的快慢——亲手感受跳连的作用。

??? note "进阶练习"
    1. 把 `TinyUNet` 改成支持**多类分割**（比如 3 类）：输出通道改为 3，损失换成 `F.cross_entropy`，标注掩码改成取值 0/1/2 的整数图。想清楚哪些地方要跟着改。
    2. 自己实现一个 `compute_iou(pred, target)` 函数，对二值掩码算 IoU，并验证：当预测全为背景、真实有前景时，它确实返回 0。
    3. 阅读 `segmentation-models-pytorch` 的文档，用 `smp.Unet` 配 `resnet34` 编码器，在一个公开分割小数据集（如 Oxford-IIIT Pet 的分割标注）上训练几个 epoch，记录 mIoU 随训练的变化。

## 小结

一句话总结：**分割就是"逐像素分类"，U-Net 用"编码器抓语义、解码器还尺寸、跳连补细节"巧妙地同时拿到了语义和位置，训练用 Dice 损失（抗不平衡）、评分用 IoU（不被背景骗）。**

- **语义分割**只分"是什么类"，**实例分割**还要分"是哪一个个体"。
- **U-Net = 编码—解码 + 跳连**，跳连是它锐利边界的关键，在代码里就是几句 `torch.cat`。
- 训练用 **Dice 损失**（常与交叉熵合用），评分用 **IoU / mIoU**，别用会骗人的像素准确率。

想看 U-Net 的原始思想，推荐读它 2015 年的原论文 [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)，篇幅不长、图也清楚；想直接上手现成模型，去看 [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) 的文档，几行代码就能搭出一个带预训练编码器的强 baseline。
