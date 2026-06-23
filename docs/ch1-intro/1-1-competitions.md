# 1.1 赛事与赛程介绍

> **难度** ⭐☆☆☆☆ · **前置**：无 · **类型**：导览

!!! abstract "读完这一页，你会知道"
    - USAAIO、IOAI、IAIO 三者分别是什么，又是怎么联系在一起的
    - 想进国家队，要一步步闯过哪几关
    - 自己是否符合参赛资格，以及一年里的关键时间点

## USAAIO 是通往国际赛的入口

如果你的目标是站上国际人工智能奥林匹克的赛场，那么对美国和加拿大的中学生来说，USAAIO 是绕不开的第一步。它本身并不是终点，而是一场层层筛选的选拔赛：你需要先在北美赛中连闯三轮，成绩最拔尖的几位才会被选拔出来，组成国家队，去参加两大国际赛事——夏天在国外举办的 **IOAI**，以及冬天举办的 **IAIO**。

让人安心的是，这三个比赛考的内容其实大同小异，都围绕着机器学习、深度学习、计算机视觉和自然语言处理这四大板块展开。正因如此，本教程直接对照覆盖面最广的 IOAI 官方考纲来组织所有内容——你只要踏实跟着学完一遍，三个比赛也就一起准备好了。

下面这张表，帮你快速分清这三个名字：

| | **USAAIO（USA–NA–AIO）** | **IOAI** | **IAIO** |
|---|---|---|---|
| 全称 | 美国/北美人工智能奥林匹克 | 国际人工智能奥林匹克 | 国际人工智能奥林匹克（冬季） |
| 它的角色 | **北美选拔赛**，是入口 | 夏季举办的国际主赛 | 冬季举办的国际赛 |
| 国家队名额 | 负责选拔出代表队 | 每个国家最多 8 人 | 每个国家 4 人 |
| 举办时间 | 冬 → 春 → 夏，分三轮 | 每年 8 月 | 每年 2 月 |

简单记：**USAAIO 是国内选拔，IOAI 和 IAIO 是它送你去的国际舞台。**

## 三轮选拔：从报名到国家队

USAAIO 的选拔像闯关游戏，一共三关，一关比一关难，能用的算力也一关比一关强：

<div class="diagram">
<svg viewBox="0 0 740 120" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ah" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/></marker>
  </defs>
  <rect x="8" y="38" width="150" height="52" rx="6" fill="var(--dia-bg-card)" stroke="var(--dia-stroke)"/>
  <text x="83" y="62" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">Round 1 资格赛</text>
  <text x="83" y="79" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">在线 · 仅 CPU</text>
  <rect x="194" y="38" width="150" height="52" rx="6" fill="var(--dia-bg-card)" stroke="var(--dia-stroke)"/>
  <text x="269" y="62" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">Round 2 邀请赛</text>
  <text x="269" y="79" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">MIT · 允许 GPU</text>
  <rect x="380" y="38" width="150" height="52" rx="6" fill="var(--dia-bg-card)" stroke="var(--dia-stroke)"/>
  <text x="455" y="62" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">Camp 集训营</text>
  <text x="455" y="79" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">Harvard · 选拔</text>
  <rect x="566" y="38" width="166" height="52" rx="6" fill="var(--dia-accent)" stroke="var(--dia-accent-deep)"/>
  <text x="649" y="62" text-anchor="middle" font-size="14" fill="#fff">国家队</text>
  <text x="649" y="79" text-anchor="middle" font-size="11" fill="#ffe8dc">IOAI 8 · IAIO 4</text>
  <path d="M158 64 H190" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ah)"/>
  <path d="M344 64 H376" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ah)"/>
  <path d="M530 64 H562" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ah)"/>
</svg>
</div>
<p class="figure-caption">从 Round 1 到国家队的完整晋级路线。</p>

**第一关，Round 1（资格赛）** 面向所有人开放，是报名后人人都能参加的线上考试。你会在学校或官方授权的考点、在全程摄像头监考下，登录 Google Colab 答题。这一轮**只能用 CPU**，所以题目不会要求你训练特别大的模型，考的是基本功：数学基础、Python、经典机器学习，以及最基础的神经网络和卷积网络。题目分两类，一类是要你在文本框里用文字和公式写出推导的"非编码题"，另一类是真正写代码跑结果的"编码题"。

**第二关，Round 2（邀请赛）** 只邀请 Round 1 的高分选手参加，在 MIT 现场进行。和第一关最大的不同是，**这一轮允许使用 GPU**，于是题目可以涉及更大的模型、更全的考纲，难度明显抬升。这一关同时还会评出 Gold、Silver、Bronze 等奖项。

**第三关，Camp（集训营）** 在 6 月于哈佛大学举办，是真正决定谁能进国家队的环节。在营里，组织方会通过一系列"选拔测试"，从中挑出最终代表美国出战 IOAI 和 IAIO 的选手。

把三关并排放在一起看会更清楚：

| 轮次 | 在哪考、怎么考 | 算力 | 主要考查 | 通过后 |
|---|---|---|---|---|
| Round 1 资格赛 | 线上，学校/授权考点，全程监考 | 仅 CPU | 数学 + Python + 经典 ML + 基础神经网络/CNN | 高分者受邀进 Round 2 |
| Round 2 邀请赛 | MIT 现场，监考 | 允许 GPU | 全考纲，难度更高 | 顶尖者进 Camp，并评奖 |
| Camp 集训营 | 哈佛，6 月 | —— | 选拔测试 | 入选国家队 → IOAI / IAIO |

## 2026 赛季的时间线

为了让你对节奏有个直观感受，下面列出 2026 赛季的实际安排。要注意的是，**每一年的具体日期都会变，请以官方届时公布为准**，这里只是给你一个参照。

| 时间 | 事件 | 地点 |
|---|---|---|
| 2026-01-30 | USAAIO Round 1 | 线上 |
| 2026-02-23 ~ 27 | IAIO 2026（冬季国际赛） | 斯洛文尼亚·卢布尔雅那 |
| 2026 年 3–4 月 | USAAIO Round 2 | MIT |
| 2026 年 6 月 | USAAIO Camp | 哈佛大学 |
| 2026-08-02 ~ 08 | IOAI 2026（夏季国际赛） | 哈萨克斯坦·阿斯塔纳 |

你会发现，冬季的 IAIO 时间其实排在 USAAIO Round 2 之前——这是因为 IAIO 选的是上一个赛季选拔出来的选手。换句话说，一整套"选拔 → 出战"的周期会跨越一年多，不必被日期顺序绕晕，记住"先国内选拔、再国际出战"这条主线就好。

## 你有参赛资格吗

USAAIO 对选手有三条基本要求，同时满足才能代表美国/北美出战国际赛：

- 在**国际比赛首日时还未满 20 岁**；
- 当年是**全日制中学生**，也就是说，已经成为全日制大学生的同学不符合条件；
- 是美国或加拿大的公民、永久居民，或者正在美国/加拿大全日制就读的中学生。

如果你刚上高中、对 AI 感兴趣，那么时间完全来得及——越早开始打基础，后面越从容。

## 晋级与奖项

整条晋级链路可以这样理解：在 Round 1 拿到足够高的分数，你就会收到 Round 2 的邀请；在 Round 2 表现出色，不仅能晋级 Camp，还可能拿到 Gold、Silver、Bronze 或 Honorable Mention 的奖项；最后在 Camp 的选拔测试中胜出，你就正式成为国家队的一员，前往 IOAI（8 人）或 IAIO（4 人）的赛场。

## 官方信息源

竞赛的规则和日期偶尔会调整，**最权威的永远是官网**。建议你收藏下面几个地址，随时核对最新信息：

- USAAIO 官网：<https://www.usaaio.org/> —— 赛程、考纲、规则都在这里
- IOAI 官网：<https://ioai-official.org/>
- IAIO 官网：<https://iaio-official.org/>

至于历年真题和数据集，我们整理在了 [附录 A2](../appendix/a2-datasets.md)，等你学完基础、想实战练手时可以回来取用。

## 小结

- USAAIO 是北美学生通往 IOAI / IAIO 的**选拔入口**，本身不是终点。
- 选拔分三关：Round 1（线上、CPU）→ Round 2（MIT、GPU）→ Camp（哈佛、定国家队）。
- 三个比赛考纲相通，跟着本教程按 IOAI 考纲学一遍即可全覆盖；资格、日期以官网为准。
