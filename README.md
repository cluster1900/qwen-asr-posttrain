# Qwen3-ASR Robust ASR & 大模型底层与后训练实战指南

这是一个基于官方 `Qwen/Qwen3-ASR-0.6B`（`qwen-asr` / Transformers API）的独立中文鲁棒 ASR 后训练工程（单张 A100，Colab + Google Drive 持久化，完整流水线 `Base → SFT → DPO → RL → 封存 Test / 真实 OOD 评测`），并配套出版了一本面向普通开发者与高中学历学习者的全栈实战电子书（**HTML / PDF / EPUB3**）。

---

## 📖 配套电子书：《从零到一：大模型底层原理、GPU 算子优化与任意模型后训练实战》

> 从高中数学与 PyTorch 内存/计算图底层出发，系统讲透 GPU 架构与 Roofline、GEMM / FlashAttention / RMSNorm / SwiGLU / RoPE / Triton 算子优化、显存精算与选卡、`Qwen3-ASR-0.6B` 全部 `782,426,112` 参数逐层解剖，以及 `LoRA → SFT → DPO → RL/GRPO` 与配对簇 Bootstrap 统计门禁的端到端实战（含 18 张原创高清矢量原理图与任意模型 0-to-1 通用脚手架）。

- **🌐 在线阅读（GitHub Pages，推荐）**：https://cluster1900.github.io/qwen-asr-posttrain/
  - 支持侧边目录实时搜索、亮色 / 护眼 / 暗黑主题切换、字号缩放、高清矢量图与 MathML 公式，内置 PDF / EPUB3 快捷下载。
- **📦 三格式成品下载（GitHub Release `book-v1.0.0`）**：https://github.com/cluster1900/qwen-asr-posttrain/releases/tag/book-v1.0.0
  - `zero-to-one-llm-posttraining-v1.0.0.html`（离线单文件 HTML，664 KB）
  - `zero-to-one-llm-posttraining-v1.0.0.pdf`（83 页 A4 高清印刷版 PDF，10 MB）
  - `zero-to-one-llm-posttraining-v1.0.0.epub`（标准 EPUB3 电子书，864 KB）
  - `book-v1.0.0_SHA256SUMS.txt`（全部成品 SHA-256 校验清单）
- **📁 源文件与构建说明**：见 [`book/README.md`](book/README.md)（每次发布 `book-v*` Release，[`.github/workflows/book-pages.yml`](.github/workflows/book-pages.yml) 自动拉取验签并更新 GitHub Pages）。

---

## 🧪 后训练工程总览与核心实测结论

项目遵循严格的文档先行、不可变哈希溯源、固定 Greedy 评测解码、独立进程 100% 读回验签（Readback）与冻结门禁纪律，已完成 **Phase 0 至 Phase 6 Track 3B** 全流程闭环：

| 实验阶段 / 评测基准 | Base | SFT | DPO | RL | 关键结论 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 5 封存 `test.jsonl`**（AISHELL-1 合成五场景，2500 条） | `2220` 字 (`6.05%`) | `2204` (`6.00%`) | **`2200`** (`5.99%`) | `2213` (`6.03%`) | DPO 封存集总体最优（`-20` 字 / `-21` 词编辑），验证 `mismatched_only` 直通差异掩码 + Parent Anchor 根治静音幻觉 |
| **Track 3B 封存 `test.jsonl`**（OpenSLR 28 物理增强，2500 条 / 500 簇） | `2053` 字 (`5.59%`) | **`1985`** (`5.41%`, `-68`) | **`1986`** (`5.41%`, `-67`) | `gate_failed`（未发布） | `sft/dpo vs base` 500 簇配对 Bootstrap 95% CI 严格负值（$P > 99\%$）；纯 A100 400 簇子集单调提升 `1586 → 1551 → 1548` |
| **Track 3B 真实会议 `AliMeeting Eval` OOD**（1000 条，0 重叠近/远场，全量 A100） | `1038` 字 (`6.25%`) | `1001` (`6.03%`, `-37`) | **`998`** (`6.01%`, **`-40` 字 / `-34` 词**) | — | **Sim-to-Real 突破**：近讲 `-19`、真实 8 麦阵列远场 `-21`、标点 `100%` 保持，`dpo vs base` 95% CI `[-0.4746, -0.0243] pp` 严格为负 |

- **最终晋级发布谱系**：`Base → sft_06b_zh_20261124T194600_step000240 → dpo_06b_zh_20261125T005003_step000040`（Stage 3 RL 跑满 3 轮共 23 组实验后严格遵守不放宽冻结门禁铁律，诚实闭环为 `gate_failed` 不发布）。
- 完整实验进度、门禁配置、风险排错复盘与 GitHub Releases 资产索引见：
  - [`docs/qwen3-asr/00_progress.md`](docs/qwen3-asr/00_progress.md)
  - [`docs/qwen3-asr/04_training_plan.md`](docs/qwen3-asr/04_training_plan.md)
  - [`docs/qwen3-asr/06_risks_and_decisions.md`](docs/qwen3-asr/06_risks_and_decisions.md)
  - [`docs/qwen3-asr/13_github_full_archive.md`](docs/qwen3-asr/13_github_full_archive.md)
  - [`conclusions/README.md`](conclusions/README.md)

---

## 📂 目录结构

```text
book/                配套实战书籍源码、18 张原创高清矢量图、三格式构建/审校与 Pages 生成脚本
conclusions/         结构化终局评测报告与配对簇 Bootstrap 数学期望（JSON/CSV/Markdown）
configs/             数据、推理与 SFT/DPO/RL 训练 YAML 配置及冻结基线快照
docs/qwen3-asr/      架构、开发、数据、训练、测试、风险决策、运行日志与归档文档
evaluation/          zh-cer-v1 CER、锁定 jieba WER、场景聚合与配对 Bootstrap 统计门禁
inference/           Qwen3-ASR 官方 API 统一 Greedy 推理与续跑封装
notebooks/           Colab 编排入口（复用逻辑全部下沉至 Python 模块）
scripts/             manifest 构建、OpenSLR 28 物理混响增强、标点自对齐、Drive 同步与基准脚本
train/               SFT、DPO、RL 训练 runner、离线 reference logprob 缓存、门禁与发布读回脚本
tests/               数据、合同、恢复、训练、评测与目录 README 合同测试
.github/workflows/   GitHub Actions（自动从 book-v* Release 拉取验签并发布 GitHub Pages）
```

---

## 🛠️ 常用命令

```bash
# 1. 运行仓库全部合同单元测试
.venv/bin/python -m unittest discover -s tests -v

# 2. 本地重建书籍三格式（HTML / PDF / EPUB3）并执行自动化审校
python3 book/scripts/build_book.py && python3 book/scripts/audit_book.py

# 3. 本地直接从 GitHub Release 拉取 HTML 电子书并启动预览服务
python3 book/scripts/build_pages.py --from-release book-v1.0.0 --serve 8080
```

## 历史参考

- 重置前仓库历史状态保存在本地 Git 标签 `pre-reset-20261012`，不作为当前实现或实验结果来源。
- 全部归档资产（权重、运行日志、评测结论与书籍成品）均已在 [GitHub Releases](https://github.com/cluster1900/qwen-asr-posttrain/releases) 中附带 SHA-256 清单发布。
