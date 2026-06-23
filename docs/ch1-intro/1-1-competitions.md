# 1.1 赛事与赛程介绍

> **难度** ⭐☆☆☆☆ · **前置**：无 · **类型**：导览

USAAIO 不是一场孤立的考试，而是一条**选拔通道**：先在北美赛（USA–NA–AIO）里闯过三关，才能代表美国/北美出战两大国际 AI 奥林匹克 —— **IOAI**（夏季）与 **IAIO**（冬季）。先搞清"打的是什么、怎么晋级"，学习时才不迷路。

!!! abstract "本页目标"
    - 说清 USAAIO / IOAI / IAIO 三者的关系与区别
    - 看懂 USAAIO 三轮选拔流程与晋级规则
    - 知道自己是否有资格、关键时间节点在哪

## 一、三大赛事是什么

| | **USAAIO（USA–NA–AIO）** | **IOAI** | **IAIO** |
|---|---|---|---|
| 全称 | USA / North America AI Olympiad | Int'l Olympiad in AI | Int'l AI Olympiad（冬季） |
| 角色 | **北美选拔赛**（入口） | 夏季国际主赛 | 冬季国际赛 |
| 国家队名额 | 选出代表队 | 每国最多 **8 人**（2 队 × 4） | 每国 **4 人** |
| 时间 | 冬 → 春 → 夏（三轮） | 夏季（8 月） | 冬季（2 月） |

!!! tip "一句话"
    **先过 USAAIO，才能去 IOAI / IAIO。** 三者考纲高度重合（经典 ML / 深度学习 / CV / NLP），按 IOAI 官方考纲准备即可全覆盖 —— 详见 [1.2 考纲总览与学习路线](1-2-syllabus-roadmap.md)。

## 二、USAAIO 三轮选拔

<div class="diagram">
<svg viewBox="0 0 740 120" font-family="-apple-system, 'Segoe UI', sans-serif">
  <defs>
    <marker id="ah" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
      <path d="M0 0 L6 3 L0 6 Z" fill="var(--dia-stroke-soft)"/>
    </marker>
  </defs>
  <!-- Round 1 -->
  <rect x="8" y="38" width="150" height="52" rx="6" fill="var(--dia-bg-card)" stroke="var(--dia-stroke)"/>
  <text x="83" y="62" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">Round 1 资格赛</text>
  <text x="83" y="79" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">在线 · 仅 CPU</text>
  <!-- Round 2 -->
  <rect x="194" y="38" width="150" height="52" rx="6" fill="var(--dia-bg-card)" stroke="var(--dia-stroke)"/>
  <text x="269" y="62" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">Round 2 邀请赛</text>
  <text x="269" y="79" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">MIT · 允许 GPU</text>
  <!-- Camp -->
  <rect x="380" y="38" width="150" height="52" rx="6" fill="var(--dia-bg-card)" stroke="var(--dia-stroke)"/>
  <text x="455" y="62" text-anchor="middle" font-size="14" fill="var(--dia-stroke)">Camp 集训营</text>
  <text x="455" y="79" text-anchor="middle" font-size="11" fill="var(--dia-stroke-soft)">Harvard · 选拔</text>
  <!-- National team -->
  <rect x="566" y="38" width="166" height="52" rx="6" fill="var(--dia-accent)" stroke="var(--dia-accent-deep)"/>
  <text x="649" y="62" text-anchor="middle" font-size="14" fill="#fff">国家队</text>
  <text x="649" y="79" text-anchor="middle" font-size="11" fill="#ffe8dc">IOAI 8 · IAIO 4</text>
  <!-- arrows -->
  <path d="M158 64 H190" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ah)"/>
  <path d="M344 64 H376" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ah)"/>
  <path d="M530 64 H562" stroke="var(--dia-stroke-soft)" stroke-width="1.5" marker-end="url(#ah)"/>
</svg>
</div>
<p class="figure-caption">图 1.1：USAAIO 三轮选拔 → 国家队 → IOAI / IAIO 的晋级流水线。</p>

| 轮次 | 形式 | 算力 | 考查内容 | 晋级 |
|---|---|---|---|---|
| **Round 1 资格赛** | 在线（学校或授权考点），全程屏幕+摄像头监考 | **仅 CPU**（Google Colab） | 数学基础 + Python + 经典 ML + 基础 PyTorch + CNN；含"非编码题（Markdown 写数学解答）"与"编码题" | 高分者受邀进 Round 2 |
| **Round 2 邀请赛** | 现场（MIT），监考 | **允许 GPU**（Colab L4） | 全考纲，难度抬升 | 顶尖者进 Camp，并评奖 |
| **Camp 集训营** | 现场（Harvard，6 月） | — | 选拔测试（Team Selection Tests） | 选出代表队 → IOAI / IAIO |

## 三、2026 赛季时间线（示例）

!!! note "日期每年变动"
    下表为 **2026 赛季**的实际安排，仅供感受节奏；**每年具体日期以官方公布为准**。

| 时间 | 事件 | 地点 |
|---|---|---|
| 2026-01-30 | USAAIO Round 1 | 在线 |
| 2026-02-23 ~ 27 | IAIO 2026（冬季国际赛） | 斯洛文尼亚·卢布尔雅那 |
| 2026 年 3–4 月 | USAAIO Round 2 | MIT |
| 2026 年 6 月 | USAAIO Camp | Harvard |
| 2026-08-02 ~ 08 | IOAI 2026（夏季国际赛） | 哈萨克斯坦·阿斯塔纳 |

## 四、参赛资格

- 国际赛**首日时未满 20 岁**；
- 当年为**全日制中学生**，**不能是全日制大学生**；
- 美国/加拿大公民、永久居民，或在美/加全日制就读的中学生。

## 五、奖项与晋级

- **Round 1** 高分 → 受邀 **Round 2**；
- **Round 2** → 评 **USAAIO Camper / Gold / Silver / Bronze / Honorable Mention**；
- **Camp** 选拔测试 → 入选**国家队**，赴 IOAI（8 人）/ IAIO（4 人）。

## 六、官方链接与信息源

- USAAIO 官网：<https://www.usaaio.org/>（赛程、考纲、规则）
- IOAI 官网：<https://ioai-official.org/>
- IAIO 官网：<https://iaio-official.org/>
- 历年真题与数据集：见 [附录 A2](../appendix/a2-datasets.md)

> 信息源：usaaio.org/2026-usa-na-aio、ioai-official.org、iaio-official.org、competitionsciences.org。
