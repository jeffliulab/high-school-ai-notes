# 高中人工智能笔记 · High School AI Notes

面向 **USAAIO（USA / North America AI Olympiad）→ IOAI / IAIO** 的中文备赛教程。
框架：MkDocs Material；外观参照 soma-zero e-Manual 风格。

🔗 在线站点：<https://jeffliulab.github.io/high-school-ai-notes/>

## 结构（四关卡难度主轴 + 基础章）
- **第1章 入门与基础** — 赛事介绍 + 数学与编程基础
- **第2章 Round 1** — 资格赛（CPU·基础）：经典 ML、DL 入门、CNN 入门
- **第3章 Round 2** — 邀请赛（GPU·全考纲）：DL 进阶、Transformer、CV、NLP/音频
- **第4章 Camp** — 集训营：竞赛工程、真题精讲、团队赛、选拔
- **第5章 Champion** — 国际赛：IOAI / IAIO 实战与冲刺
- **附录** — 资料中心 / 真题与数据集 / 速查 / 工具

每篇知识点笔记按「直觉 → 数学 → 代码 → 真题 → 练习」循序展开。

## 本地预览
```bash
pip install -r requirements.txt
mkdocs serve
```

## 部署
推送到 `main` 触发 GitHub Actions（`.github/workflows/deploy.yml`），构建并发布到 `gh-pages`。
