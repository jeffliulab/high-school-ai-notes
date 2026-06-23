# 3.1 目标检测

> **难度** ⭐⭐⭐⭐☆ · **前置**：[3.2 图像分类实战](../ch2-round1/2-3-2-image-classification.md)、[3.1 卷积与池化](../ch2-round1/2-3-1-conv-pooling.md)

!!! abstract "读完这一节，你会"
    - 说清"分类"和"检测"差在哪，理解一个检测框其实是"位置回归 + 类别分类"两件事合在一起
    - 用大白话讲明白锚框（anchor）和区域提议（region proposal）在解决什么问题
    - 手写一遍非极大抑制（NMS），把一堆重叠的框收拾成几个干净的结果
    - 分得清 Faster R-CNN、SSD、YOLO 这三类主流检测器各自的取舍

## 从"这是什么"到"它在哪儿"

在 Round 1 里，我们已经会做**图像分类**了：丢给模型一张图，它告诉你"这是一只猫"。但现实里的问题往往更进一步——一张街景照片里同时有行人、汽车、红绿灯，我们不光想知道"图里有什么"，还想知道"每样东西具体在哪个位置"。这就从分类问题升级成了**目标检测（object detection）**。

那"在哪儿"该怎么表示呢？最常用的办法是给每个物体框一个矩形，叫**边界框（bounding box）**，用四个数字描述它：框中心的横纵坐标 $(x, y)$，再加上框的宽 $w$ 和高 $h$。所以一次检测的输出，本质上是一串这样的东西：**(一个框的四个坐标，这个框里是什么类别，有多大把握)**。

把这件事拆开看，你会发现它其实是两个我们已经熟悉的任务拼在一起：**预测那四个坐标，是一个回归问题；判断框里是什么，是一个分类问题。** 几乎所有检测器，骨子里都在同时干这两件事——这就是常说的"框回归 + 分类"双头结构。先记住这个画面，后面无论看哪种检测器都不会迷路。

<div class="diagram">
<svg viewBox="0 0 380 210" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 图像边框 -->
  <rect x="20" y="20" width="200" height="150" fill="var(--dia-bg-card)" stroke="var(--dia-stroke-tertiary)" stroke-width="1"/>
  <text x="120" y="15" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">一张输入图</text>
  <!-- 物体1：猫 -->
  <rect x="45" y="60" width="80" height="80" fill="none" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="48" y="55" font-size="11" fill="var(--dia-accent)">cat 0.97</text>
  <!-- 物体2：狗 -->
  <rect x="135" y="85" width="70" height="65" fill="none" stroke="var(--dia-blue)" stroke-width="2"/>
  <text x="138" y="80" font-size="11" fill="var(--dia-blue)">dog 0.91</text>
  <!-- 右侧：一个框的四元组说明 -->
  <line x1="240" y1="95" x2="270" y2="95" stroke="var(--dia-stroke-tertiary)" stroke-width="1" stroke-dasharray="3 2"/>
  <text x="278" y="60" font-size="11" fill="var(--dia-stroke-soft)">每个框 = </text>
  <text x="278" y="82" font-size="11" fill="var(--dia-accent)">坐标 (x, y, w, h)</text>
  <text x="278" y="100" font-size="11" fill="var(--dia-blue)">类别 + 置信度</text>
  <text x="278" y="124" font-size="10" fill="var(--dia-stroke-soft)">= 回归 + 分类</text>
</svg>
</div>
<p class="figure-caption">图 3.1-1：目标检测的输出是一串边界框，每个框同时携带"位置坐标"（靠回归得到）和"类别 + 置信度"（靠分类得到）。</p>

## 一个绕不开的难题：物体可能出现在任何地方

分类之所以简单，是因为整张图就对应一个答案。可检测难在哪？难在**物体可能出现在图里任何位置、任何大小**——你事先并不知道该往哪儿看、该框多大。

最朴素的想法是"滑动窗口"：拿一个小框在图上从左到右、从上到下一格一格地滑，每滑到一处就问一次分类器"这里面有东西吗"。这思路没错，但代价大得吓人——光位置就成千上万种，再配上几种不同的框大小、长宽比例，要问的次数会爆炸。所以现代检测器都不再傻滑，而是想办法**聪明地、批量地给出一堆"候选位置"**，再逐一判断。围绕"怎么生成候选位置"，主流方法分成了两大流派。

### 两阶段：先提议，再精修（以 Faster R-CNN 为代表）

第一种思路是"分两步走"。**第一步**，让一个专门的小网络先粗略地圈出图里"看起来像有东西"的若干个区域，这一步生成的候选框叫**区域提议（region proposal）**——你可以理解成它先帮你把注意力收缩到几百个值得看的地方，而不是几万个。**第二步**，再对每个提议区域做精细的分类，并微调框的坐标，让它更贴合物体。

这一派的代表是 **Faster R-CNN**，它里面那个负责生成提议的小网络叫 RPN（Region Proposal Network，区域提议网络）。两阶段的好处是**准**——先粗筛再精修，定位往往很到位；坏处是**慢**，因为要跑两遍。

### 单阶段：一步到位（以 SSD、YOLO 为代表）

第二种思路更激进：**干脆不要"提议"这一步，让网络看一眼图就直接吐出所有的框和类别。** 这就是 **YOLO**（You Only Look Once，"只看一眼"）名字的由来，**SSD**（Single Shot Detector，"单次检测器"）也是同一思路。它们把图切成一个个网格，让每个网格直接负责预测"我这片区域里有没有物体、是什么、框多大"。

单阶段牺牲了一点精度，换来了**快**——快到能实时处理视频。所以自动驾驶、实时监控这类"等不起"的场景，几乎都用单阶段。一句话记住这个取舍：**两阶段重精度，单阶段重速度。**

## 锚框：不让网络从零猜框

无论哪一派，都还有个共同的麻烦：让网络凭空回归出 $(x, y, w, h)$ 太难了——它得在毫无参照的情况下猜出一个物体多大多宽，训练起来很不稳。

工程师想了个巧妙的办法：**预先在图上每个位置都摆好一组形状各异的"参考框"，让网络不去猜绝对坐标，而只去预测"真实框相对这些参考框要怎么微调"。** 这些预设的参考框就叫**锚框（anchor box）**。

打个比方，这就像你不直接报一个陌生地址的经纬度，而是说"在地铁站东边 200 米、再往北 50 米"——有了"地铁站"这个锚点，描述位置一下子就容易多了。锚框就是检测器的"地铁站"：每个位置预先放几个不同长宽比、不同大小的锚框（比如一个偏瘦高的、一个偏扁宽的、一个方正的），网络只需学一个小小的偏移量，去把最接近的那个锚框拉到物体上。

<div class="diagram">
<svg viewBox="0 0 380 200" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 中心锚点 -->
  <circle cx="110" cy="100" r="3" fill="var(--dia-stroke)"/>
  <text x="100" y="92" font-size="10" fill="var(--dia-stroke-soft)">锚点</text>
  <!-- 三个不同形状的锚框，共享中心 -->
  <rect x="80" y="55" width="60" height="90" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <rect x="65" y="78" width="90" height="44" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5" stroke-dasharray="4 3"/>
  <rect x="80" y="70" width="60" height="60" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5" stroke-dasharray="1 3"/>
  <text x="70" y="170" font-size="10" fill="var(--dia-stroke-soft)">预设的几个锚框（瘦高/扁宽/方正）</text>
  <!-- 箭头：微调 -->
  <line x1="165" y1="100" x2="215" y2="100" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#oa)"/>
  <defs>
    <marker id="oa" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <text x="180" y="92" text-anchor="middle" font-size="10" fill="var(--dia-accent)">学微调量</text>
  <!-- 真实框 -->
  <rect x="250" y="62" width="64" height="78" fill="none" stroke="var(--dia-accent)" stroke-width="2"/>
  <text x="250" y="56" font-size="10" fill="var(--dia-accent)">贴合物体的真实框</text>
</svg>
</div>
<p class="figure-caption">图 3.1-2：每个位置预先放好几种形状的锚框，网络只需学一个小偏移量，把最接近的锚框微调到真实物体上——比凭空猜坐标稳得多。</p>

值得一提的是，近几年也出现了一批**无锚框（anchor-free）**的检测器（比如 CenterNet、FCOS），它们直接预测物体中心点和到四条边的距离，省掉了锚框那一堆超参数。不过锚框的思想仍然是你理解检测器的基础，先把它吃透。

## 怎么知道一个框框得准不准：IoU

在讲下一个核心概念之前，我们得先有把"尺子"，来衡量两个框重叠得有多厉害。这把尺子叫 **IoU（Intersection over Union，交并比）**：它等于两个框**相交的面积**除以**它们合起来覆盖的总面积**。

$$\text{IoU} = \frac{\text{交集面积}}{\text{并集面积}}.$$

直觉很简单：两个框完全重合，IoU 就是 1；完全不挨着，就是 0；部分重叠，落在 0 到 1 之间。IoU 有两个大用处：训练时用它判断"某个预测框算不算命中了真实物体"（通常 IoU 超过 0.5 才算数），以及——接下来这个更关键的——用它来去掉重复的框。

## 非极大抑制：把一堆重叠的框收拾干净

这里要解决检测里一个非常实际的麻烦。网络跑完后，**同一个物体周围往往会冒出好几个高度重叠的框**——因为附近好几个锚框都觉得"这儿有只猫"。可我们最终只想要一个框。怎么从这一堆里挑出最好的、扔掉多余的？

办法叫**非极大抑制（Non-Maximum Suppression，NMS）**。名字听着唬人，其实逻辑朴素得像在排队：

1. 把所有框按置信度从高到低排好队；
2. 拿出当前置信度最高的那个框，留下它，认定它是个"胜者"；
3. 看看剩下的框里，谁和这个胜者的 IoU 太高（重叠太多，超过某个阈值，比如 0.5），就判定它们是在重复框同一个物体，**统统抑制掉（删除）**；
4. 在剩下还活着的框里，重复第 2 到第 3 步，直到没有框可处理为止。

说白了就是一句话：**置信度最高的留下，和它撞车太厉害的都让路。** 下面这张图把这个"留一个、删一片"的过程画了出来。

<div class="diagram">
<svg viewBox="0 0 400 190" font-family="-apple-system, 'Segoe UI', sans-serif">
  <!-- 左：NMS 之前，一堆重叠框 -->
  <text x="90" y="20" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">NMS 之前：一堆重叠框</text>
  <rect x="40" y="45" width="90" height="100" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <rect x="52" y="55" width="90" height="100" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <rect x="30" y="38" width="90" height="100" fill="none" stroke="var(--dia-stroke-tertiary)" stroke-width="1.5"/>
  <text x="62" y="100" font-size="10" fill="var(--dia-stroke-soft)">0.9</text>
  <text x="92" y="120" font-size="10" fill="var(--dia-stroke-soft)">0.7</text>
  <!-- 箭头 -->
  <line x1="185" y1="95" x2="225" y2="95" stroke="var(--dia-accent)" stroke-width="1.5" marker-end="url(#na)"/>
  <defs>
    <marker id="na" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-accent)"/></marker>
  </defs>
  <text x="205" y="86" text-anchor="middle" font-size="10" fill="var(--dia-accent)">NMS</text>
  <!-- 右：只剩一个 -->
  <text x="320" y="20" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">之后：只留最强的</text>
  <rect x="280" y="45" width="90" height="100" fill="none" stroke="var(--dia-accent)" stroke-width="2.5"/>
  <text x="312" y="100" font-size="11" fill="var(--dia-accent)">0.9</text>
</svg>
</div>
<p class="figure-caption">图 3.1-3：NMS 在每一簇重叠框里只保留置信度最高的那个，把和它重叠过多的其余框全部抑制，得到干净的检测结果。</p>

### 动手写一个 NMS

道理讲清楚了，我们用 NumPy 把它实现一遍——你会发现代码和上面的四步几乎是一一对应的。先写出整个函数，再逐段解读：

```python
import numpy as np

def nms(boxes, scores, iou_thresh=0.5):
    """
    boxes:  (N, 4) 数组，每行是一个框 [x1, y1, x2, y2]（左上、右下两个角点）
    scores: (N,)   数组，每个框的置信度
    返回：保留下来的框的下标列表
    """
    # 先算出每个框自己的面积，后面算 IoU 要用
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 第 1 步：按置信度从高到低排队（argsort 默认升序，[::-1] 翻成降序）
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 第 2 步：队首就是当前最高分的框，留下它
        i = order[0]
        keep.append(i)

        # 第 3 步：算这个胜者和其余所有框的交集面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)      # 不重叠时宽高会是负数，截到 0
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # 交并比 = 交集 / (各自面积之和 - 交集)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 只保留那些和胜者重叠不算多的框，进入下一轮
        remain = np.where(iou <= iou_thresh)[0]
        order = order[remain + 1]           # +1 是因为上面切片从 order[1:] 算起

        # 第 4 步：循环回去，对剩下的框再挑下一个胜者
    return keep
```

代码最妙的地方在第 3 步：我们没有写循环去逐个算 IoU，而是用 NumPy 的 `maximum` / `minimum` 一次性算出胜者和**所有**剩余框的交集——这正是上一章反复强调的向量化思想，比 `for` 循环快得多。

来跑个小例子验证一下。我们造三个框，其中前两个几乎重叠（都框着同一个物体），第三个在别处：

```python
boxes = np.array([[ 10,  10,  60,  60],   # 框 A，分高，和 B 重叠
                  [ 12,  12,  62,  62],   # 框 B，和 A 几乎重合
                  [200, 200, 260, 260]])  # 框 C，独自在角落
scores = np.array([0.95, 0.80, 0.90])

print(nms(boxes, scores, iou_thresh=0.5))   # 输出 [0, 2]
```

结果是 `[0, 2]`：框 A（最高分 0.95）和框 C（0.90）被保留，而和 A 严重重叠的框 B 被抑制掉了——和我们的预期完全一致。

### 库版：现成的轮子

竞赛或工程里当然不必每次手写，深度学习框架早就备好了高效实现。比如在 PyTorch 里，一行就能调用经过 GPU 优化的 NMS：

```python
import torch
from torchvision.ops import nms

boxes = torch.tensor([[10., 10., 60., 60.],
                      [12., 12., 62., 62.],
                      [200., 200., 260., 260.]])
scores = torch.tensor([0.95, 0.80, 0.90])

keep = nms(boxes, scores, iou_threshold=0.5)
print(keep)        # tensor([0, 2])，和我们手写版一致
```

手写一遍是为了真懂它在干嘛，实战里则放心用库——这也是我们学每个算法时的一贯态度：先吃透原理，再站在巨人的肩膀上。

## 容易踩的坑

- **坐标格式别搞混**：边界框有两种常见写法，一种是"中心 + 宽高" $(x, y, w, h)$，另一种是"两个角点" $(x_1, y_1, x_2, y_2)$。算 IoU 和做 NMS 通常用角点格式，喂错格式会算出一堆离谱的重叠面积。
- **NMS 是按类别分别做的**：一只猫的框和一辆车的框就算重叠，也不该互相抑制——它们是不同物体。正确做法是对每个类别单独跑一次 NMS。
- **IoU 阈值要权衡**：阈值设太低，会把挨得近的两个真实物体误删成一个；设太高，又会留下一堆重复框。0.5 是常用的折中起点，但要看场景调。
- **锚框尺寸要贴合数据**：如果你的数据集里全是细长的物体（比如电线杆），却用了一堆方正的锚框，网络会学得很费劲。锚框的长宽比最好参照数据集里真实框的分布来设。
- **类别不平衡**：一张图里背景区域远多于物体，负样本（没物体的框）会淹没正样本。单阶段检测器常用 Focal Loss 等手段来缓解这个问题。

## 它在后面会怎么用到

目标检测是计算机视觉里承上启下的一环，它用到的很多想法你之前学过、之后还会反复见到：

- 检测器的"骨干网络"（backbone）几乎都是一个预训练好的卷积网络，这正是 [3.3 迁移学习与预训练编码器](../ch2-round1/2-3-3-transfer-learning.md) 讲的复用思路。
- 把检测的"框"细化到"每个像素属于哪个物体"，就走到了下一节 [3.2 图像分割](3-3-2-segmentation.md)——很多分割模型（如 Mask R-CNN）干脆就是在检测器上加一个分支。
- 近年最前沿的检测器（如 DETR）已经把锚框和 NMS 都扔掉了，改用 [2.1 注意力机制](3-2-1-attention.md) 和 [2.2 Transformer 架构](3-2-2-transformer.md) 直接"集合预测"出所有框。学完本节，你就有基础去理解它们为什么是一次范式升级。
- 真正上手训练和调检测模型时，会用到 [1.1 GPU 训练](3-1-1-gpu-training.md) 的知识；典型的检测数据集（COCO、Pascal VOC）整理在 [A2 真题与数据集](../appendix/a2-datasets.md)。

## 练习

??? note "基础练习"
    1. 手算 IoU：框 A 是 $[0, 0, 4, 4]$，框 B 是 $[2, 2, 6, 6]$（都是角点格式）。先算交集面积，再算并集，最后写出 IoU。（答案应是 $4/28$。）
    2. 用一句话向同桌解释清楚：YOLO 和 Faster R-CNN 最根本的区别是什么，各自适合什么场景。
    3. 把上面手写的 `nms` 函数跑一跑，故意把 `iou_thresh` 从 0.5 调到 0.9，观察被保留的框数量怎么变，并解释为什么。

??? note "进阶练习"
    1. 给 `nms` 加上"按类别分别抑制"的逻辑：输入再多一个 `labels` 数组，只在同类别的框之间做抑制。验证两个不同类别但位置重叠的框都能被保留。
    2. 实现一个简化版的 **Soft-NMS**：不再把重叠过多的框直接删掉，而是按重叠程度给它的置信度打个折扣，分数低到一定程度才淘汰。对比它和普通 NMS 在密集物体场景下的差别。
    3. 用 `torchvision` 加载一个预训练的 Faster R-CNN，对一张你自己的照片做检测，把返回的框、类别、分数画到图上。注意先用置信度阈值过滤掉低分框。

## 小结

- 目标检测 = **定位（框回归）+ 识别（分类）**，输出是一串带类别和置信度的边界框。
- 为了不"傻滑窗口"，检测器靠**锚框**或**区域提议**批量给出候选位置，再逐一精修。
- 主流分两派：**两阶段（Faster R-CNN）重精度，单阶段（YOLO、SSD）重速度。**
- **NMS** 用 IoU 把同一物体周围的重叠框收拾成一个，是几乎所有经典检测器的标准收尾步骤。

想把直觉建得更牢，可以读 YOLO 原论文 [*You Only Look Once*](https://arxiv.org/abs/1506.02640) 和 [*Faster R-CNN*](https://arxiv.org/abs/1506.01497)，它们写得相当好读；想直接上手，[torchvision 的检测教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) 提供了开箱即用的预训练模型。
