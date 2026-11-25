# 项目评测与实验结论汇总（`conclusions/`）

## 目录职责

存放本项目已闭环各阶段（Phase 5 封存测试集、Phase 6 Track 1 锁定 `jieba` 中文 WER 重算、Phase 6 Track 2 真实会议 AliMeeting OOD、Phase 6 Track 3B Stage 3 RL Wave 3 门禁日志与 Stage 4 封存测试集 / AliMeeting OOD 终局评测）的结构化核心结论文件（`benchmark_summary.json`、`math_expectation_*.json`、`metrics.json`、`by_scenario.csv`、`run_report.md`、`gate.json`），便于在 GitHub 代码库中直接审阅与复核。

原始音频、全量 83 次运行的 checkpoint、预测 JSONL 与全量发布权重均通过本仓库 GitHub Releases 全量归档（详见 [13_github_full_archive.md](../docs/qwen3-asr/13_github_full_archive.md)）。

## 目录清单

- `phase5_test_benchmark_20261026/`：Phase 5 封存 `test.jsonl` 四阶段终局评测摘要与报告。
- `phase6_track1_wer_recompute_20261030/`：Phase 6 Track 1 锁定 `jieba==0.42.1` 中文 WER 四阶段离线重算总表。
- `phase6_track2_alimeeting_ood_20261030/`：Phase 6 Track 2 合成增强版在 AliMeeting Eval OOD 上的评测摘要与报告。
- `phase6_track3b_stage3_rl_wave3_20261125/`：Phase 6 Track 3B Stage 3 RL Wave 3 三组位移校准实验的 `gate.json`、`status.json`、`eval_decisions.jsonl` 与 `metrics.json`。
- `phase6_track3b_stage4_alimeeting_ood_20261125/`：Phase 6 Track 3B Stage 4 只读 AliMeeting Eval OOD 终局评测全量 `benchmark_summary.json`、`math_expectation_*.json` 与分阶段 `metrics.json`/`by_scenario.csv`。
- `phase6_track3b_stage4_sealed_test_20261125/`：Phase 6 Track 3B Stage 4 封存 `test.jsonl` 终局评测全量 `benchmark_summary.json`、`math_expectation_*.json`（含纯 A100 400 簇审计）与分阶段 `metrics.json`/`by_scenario.csv`。

## 维护要求

- 本目录仅存放轻量文本结论（JSON/CSV/Markdown），严禁写入原始/派生音频、权重文件（`*.safetensors`/`*.pt`）或凭证。
- 任何新增或更新的阶段终局结论须同步更新 `docs/qwen3-asr/00_progress.md` 与本目录说明。
