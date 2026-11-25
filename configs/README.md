# configs

## 目录职责

配置是数据、推理和训练的唯一声明来源。运行时必须保存解析后的配置副本；禁止在 notebook 或 runner 中隐藏关键超参数。学习率、LoRA 结构、`decode.eval`、`decode.rollout` 和门禁阈值都在开跑前写入配置，开跑后保持不可变。

## 当前文件清单

- `README.md`：本说明文档。
- `inference/base_eval.yaml`：Phase 0 greedy 评测配置，schema `qwen3-asr-infer-v1`，seed 42，`wer_tokenizer` 为 null。
- `inference/base_eval_wer.yaml`：锁定 `jieba==0.42.1` 及内置 `dict.txt` SHA-256（`7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8`、`wer_hmm: false`）的 greedy 评测配置，schema `qwen3-asr-infer-v1`，与 `base_eval.yaml` 共享同一 `model` 与 `decode_eval_sha256`（`4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`），用于输出中文 WER 辅助指标。
- `data/manifest_mvp.yaml`：Phase 1 正式数据配置，schema `qwen3-asr-data-v1`，seed 42，val 每个 scenario 至少 50 条。`source_version` 是 `aishell1-openslr33-pilot-20261015`，`audited` 为 true。行数不够时仍是 `insufficient_data`。OpenSLR 33 按交集最严原则限定于非商业学术研究与评测，不公开再分发原始或派生音频（见 `docs/qwen3-asr/03_data_plan.md`）。
- `data/alimeeting_ood_eval.yaml`：Phase 6 Track 2 AliMeeting Eval（OpenSLR 119，`Eval_Ali.tar.gz`，许可证 `CC BY-SA 4.0`）只读真实远场会议 OOD 数据配置，schema `qwen3-asr-ood-data-v1`，seed 42。固定 `usage_policy: read_only_ood_evaluation`、`far_field_channel: 0`、时长范围 `[1.0, 20.0]` 秒、跨全会议说话人轨零重叠过滤（`overlap_eps_sec: 0.001`）、纯中文过滤以及按会议轮转抽取最多 `500` 个近场/远场 1:1 配对源语句（最多 `1000` 条 WAV）。严禁混入训练集。
- `data/manifest_slr28_track3.yaml`：Phase 6 Track 3 OpenSLR 28（`rirs_noises.zip`，Apache-2.0，SHA-256 `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7`）物理房间脉冲响应与真实噪声增强数据配置，schema `qwen3-asr-data-v2`，`audio_processor_version: phase6-slr28-phys-v1`，seed 42，按房间家族与噪声录音 ID 零泄漏切分 `train`/`val`/`test`。
- `train/sft_smoke.yaml`：SFT smoke 配置，schema `qwen3-asr-train-v1`，6 step，第 5 step 中断。`wer_tokenizer` 为 null。`allow_dirty` 为 true，因为当前工作区还有未提交改动。它不能发布。
- `train/sft_pilot.yaml`：已执行的 100 step SFT pilot，学习率 `1e-4`。每 10 step 保存并评固定 2500 条 val。`interrupt_after_step` 为 null。`keep_last` 为 10。`publishable` 在 release 读回之前保持 false。
- `train/sft_pilot_lr1e5.yaml`：下一轮 100 step SFT，学习率 `1e-5`。解码、数据和 gate 与 `sft_pilot.yaml` 相同。不恢复 run `20261015T104034_sft_06b_zh_seed42`。当前 run 是 `20261015T234939_sft_06b_zh_seed42`，step 30 的评测未完成。
- `train/aishell_val_base_20261015.json`：pilot 的父 baseline，schema `qwen3-asr-baseline-v1`。它钉住 val 文件 SHA-256 和五个 scenario 的 base micro CER。
- `train/aishell_val_sft_step30_20261016.json`：DPO 的 SFT 父指标，schema `qwen3-asr-parent-metrics-v1`。数字来自 step 30 的 2500 条 val。
- `train/dpo_pilot.yaml`：DPO pilot。学习率 `5e-6`，LoRA dropout `0.0`，累积 4，beta `0.1`，父 release 是 `sft_06b_zh_20261015T234939_step000030`。`audit_disposition` 保持 `unlistened`。
- `train/dpo_dropout_probe.yaml`：dropout probe。学习率 `5e-5`，warmup 0，最多 10 step，`train_scenarios` 只有 dropout。父 release 与 pilot 相同。`selected_step` 保持 null，`publishable` 保持 false。
- `train/dpo_gap_strip_probe.yaml`：gap-strip dropout probe。预算与 `dpo_dropout_probe.yaml` 相同，pair 换成 `pairs_gap_strip` 和 `gap_strip_ref_logprobs`。`launch_utterance` 是「开始吧」。`listen_waiver` 仍是 `user_20261017_skip_listen`，`audit_disposition` 仍是 `unlistened`。`selected_step` 保持 null，`publishable` 保持 false。不恢复 `20261018T052928` 或 `20261017T131802`。
- `train/dpo_pilot_nondropout.yaml`：非静音四场景 DPO pilot。预算与 `dpo_pilot.yaml` 相同（`profile: pilot`、学习率 `5e-6`、warmup 2、最多 100 step、beta `0.1`），`train_scenarios` 设为 `[clean, noise, reverb, far_field]`，直接复用 2026-10-17 冻结的 `ref_logprobs/` 和 `pairs/` 哈希（3640 train / 533 val pairs），全量 2500 条 val 门禁评测。`audit_disposition` 保持 `unlistened`，`publishable` 保持 false。不恢复 `20261018T132840`、`20261018T052928` 或 `20261017T131802`。
- `train/dpo_pilot_aligned.yaml`：无标点单字原位替换、去 Clean 同音拷贝与差异 token 梯度掩码 DPO pilot。保持 `profile: pilot`、学习率 `5e-6`、最多 100 step、beta `0.1`，开启 `train_scenarios: [noise, reverb, far_field]`、`val_scenarios: [clean, noise, reverb, far_field]`、`aligned_substitution_only: true`、`exclude_clean_identical_rejected: true`、`max_edit_count: 1`、`loss_token_scope: mismatched_only`、`ref_margin_min: -4.0`、`ref_margin_max: 0.0`（内存过滤后 164 train / 118 val pairs），并将优化器阻尼校准为 `warmup_steps: 10`、`eps: 0.000001`、`weight_decay: 0.0`。复用 2026-10-17 冻结的 `ref_logprobs/` 和 `pairs/` 哈希，全量 2500 条 val 门禁不变。`audit_disposition` 保持 `unlistened`，`publishable` 保持 false。不恢复 `20261019T130250`、`20261019T012838` 及前三次 DPO run。已执行的 run `20261020T014427_dpo_06b_zh_seed42` 在 step 10 风险停止，不恢复。
- `train/dpo_pilot_edit1_seq.yaml`：同一 164 条单字过滤和同一套冻结门禁，`loss_token_scope` 为 `sequence`，`warmup_steps` 为 2。step 1 的学习率是 `2.5e-6`，step 2 起是 `5e-6`。`eps` 仍是 `1e-6`，`weight_decay` 仍是 `0.0`。不恢复 `20261020T014427`。`publishable` 保持 false。run `20261020T024637_dpo_06b_zh_seed42` 已在 step 10 风险停止，不恢复。
- `train/dpo_pilot_edit1_anchor.yaml`：同一 164 条单字过滤、`mismatched_only`、`warmup_steps: 10` 和同一套冻结门禁。每个 optimizer step 另加 `dropout_anchor_weight: 0.05` 的 token 平均 NLL，教师是 train dropout 行的 `rejected_text`（SFT step 30 贪心），不是金标。不恢复 `20261020T014427` 或 `20261020T024637`。`publishable` 保持 false。run `20261020T041218_dpo_06b_zh_seed42` 的 step 40 通过，step 41 风险停止，不恢复。
- `train/dpo_pilot_edit1_anchor_stop40.yaml`：与 `dpo_pilot_edit1_anchor.yaml` 相同，只把 `max_steps` 改成 40，停在评测间隔上，不再走第 41 步。其他 pilot 的 `max_steps` 仍是 100。不恢复 `20261020T041218`。run `20261020T114647_dpo_06b_zh_seed42` 在 step 30 风险停止，`selected_step` 是 10。不恢复。`publishable` 保持 false。
- `train/dpo_pilot_edit1_anchor_step40_h100.yaml`：纯方案 1。在 `dpo_pilot_edit1_anchor_stop40.yaml` 基础上设置 `optim.scheduler_max_steps: 100`，使采样器在 `train.max_steps: 40` 准时停步，同时余弦学习率保持 100 步周期（step 20 `4.8643084e-6`、step 30 `4.4736e-6`、step 40 `3.875e-6`），每步仍取 1 条 `dropout` 锚（`dropout_anchor_samples_per_step: 1` 缺省），已在 run `20261022T133007_dpo_06b_zh_seed42` 上 100% 确定性复现 `20261020T041218` Step 10–40 过关结果并在 Step 40 无风险完赛（`selected_step = 40`）。`publishable` 保持 false。
- `train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`：方案 1 + 方案 3 对比配置。在 `dpo_pilot_edit1_anchor_step40_h100.yaml` 基础上开启 `dpo.dropout_anchor_samples_per_step: 4`（与 `gradient_accumulation_steps: 4` 对齐），每个 optimizer step 对 4 条确定性 `dropout` 锚样本做等权平均（每条权重 `0.05 / 4 = 0.0125`，逐条反传），保持 `max_grad_norm: 1.0` 与门禁阈值不变。run `20261022T161911_dpo_06b_zh_seed42` 在 step 10 因 `preference_accuracy=0.5` 风险停止，不恢复。`publishable` 保持 false。
- `train/aishell_val_dpo_step40_20261022.json`：RL 的 DPO 父指标，schema `qwen3-asr-parent-metrics-v1`。数字来自 `20261022T133007_dpo_06b_zh_seed42`（`dpo_06b_zh_20261022T133007_step000040`）step 40 的 2500 条 val（总计 micro CER `0.06297105406881485`）。
- `train/rl_smoke.yaml`：Phase 4 RL 6 step smoke 配置，schema `qwen3-asr-train-v1`，`stage: rl`，`profile: smoke`，学习率 `1e-5`，1 group / 累积 2，第 5 step 中断，父 release 为 `dpo_06b_zh_20261022T133007_step000040`。`publishable` 为 false。
- `train/rl_pilot.yaml`：Phase 4 RL 50 step pilot 配置，schema `qwen3-asr-train-v1`，`stage: rl`，`profile: pilot`，学习率 `1e-5`，LoRA dropout `0.0`，1 group（`G=4`：1 greedy + 3 sampled）/ 累积 2，`kl_beta: 0.04`，每 5 step 保存并评 2500 条 val（含固定 step 0 前缀词表条件 `raw_kl <= 0.001` 与相对 step 0 的 `min_reward_improvement: 0.002`），父 release 为 `dpo_06b_zh_20261022T133007_step000040`。已执行的 run `20261023T020835_rl_06b_zh_seed42` 在 step 5 风险停止，不恢复。
- `train/rl_pilot_aligned.yaml`：Phase 4 对齐版 RL pilot 配置，schema `qwen3-asr-train-v1`，`stage: rl`，`profile: pilot`。开启 `rl.train_scenarios: [noise, reverb, far_field]`、组内无标点等长替换对齐与差异 token 梯度掩码（`rl.loss_token_scope: mismatched_only`、`rl.max_edit_count: 2`）、仅非零信号 group 计入累积窗口（`rl.accumulate_signal_groups_only: true`、`gradient_accumulation_steps: 4`）、`rl.dropout_anchor_weight: 0.05`（以冻结 DPO step 40 贪心预测为教师锚定 train `dropout`），以及阻尼优化器与日程（`learning_rate: 1e-6`、`warmup_steps: 5`、`scheduler_max_steps: 50`、`max_steps: 20`、`save_steps: 5`、`eval_steps: 5`、`kl_beta: 0.10`、`min_reward_improvement: 0.00002`）。父 release 为 `dpo_06b_zh_20261022T133007_step000040`。已执行的 run `20261023T042034_rl_06b_zh_seed42` 在 step 5 一次性通过全部门禁（`selected_step = 5`），并封包发布为 `rl_06b_zh_20261023T042034_step000005`。`publishable` 保持 false。
- `train/aishell_val_rl_step5_20261023.json`：Phase 4 RL 选出 release（`rl_06b_zh_20261023T042034_step000005`，step 5）在 2500 条固定 `val` 上的固化指标，schema `qwen3-asr-parent-metrics-v1`（总计 micro CER `0.06275259421081376`）。
- `train/sft_track3_smoke.yaml`：Phase 6 Track 3 SLR28 物理增强 6-step SFT 中断恢复 smoke 配置（同时锁定 Nov-2026 `requirements.track3.lock` 哈希与 `jieba==0.42.1` WER 评分，供 Base 分片评测与 SFT smoke 共用，不可发布）。
- `train/aishell_slr28_val_base_20261124.json`：Phase 6 Track 3 OpenSLR 28 物理增强 2500 条 `val.jsonl`（SHA-256 `c768edcefbbb377670b7263df9bf2c0250126ba63370632dc31cd27b4b862383`）的官方 Base 基线指标，schema `qwen3-asr-baseline-v1`（`predictions_sha256=6c5a383c6150f28660d382c0fc5aae8ada58643933c8a30579e5323fe607e1ab`，总体 `micro_cer=6.4091%` / `micro_wer=9.9879%`，`far_field=4.7788%` 为 Phase 1 合成数据的 `2.01×`）。
- `train/sft_track3_scale_lr2e5_ga16.yaml`：Phase 6 Track 3 SFT `scale` 主配置（仅四类物理声学场景 `[clean, noise, reverb, far_field]`、`lr=2e-5`、`ga=16`、`rank=16`、`max_steps=120`、每 30 步全量 2500 条 val 门禁与锁定 WER 评测）。
- `train/sft_track3_scale_lr1e5_ga16.yaml`：Phase 6 Track 3 SFT `scale` 低学习率对照配置（四类物理场景、`lr=1e-5`、`ga=16`、`rank=16`、`max_steps=120`）。
- `train/sft_track3_scale_lr3e5_ga16.yaml`：Phase 6 Track 3 SFT `scale` 高学习率探索配置（四类物理场景、`lr=3e-5`、`ga=16`、`rank=16`、`max_steps=120`）。
- `train/sft_track3_scale_r32_lr2e5_ga16.yaml`：Phase 6 Track 3 SFT `scale` 高秩容量配置（四类物理场景、`rank=32`/`alpha=64`、`lr=2e-5`、`ga=16`、`max_steps=120`）。
- `train/sft_track3_scale_all5_lr2e5_ga16.yaml`：Phase 6 Track 3 SFT `scale` 五场景对照消融（包含 `dropout`，`lr=2e-5`、`ga=16`、`rank=16`、`max_steps=120`），验证排除 `dropout` 对防止静音幻觉与提升真实远场的作用。
- `train/sft_track3_scale_lr15e6_ga16_160.yaml`：Phase 6 Track 3 SFT `scale` 长步数配置（四类物理场景、`lr=1.5e-5`、`ga=16`、`rank=16`、`max_steps=160`、每 40 步评测，覆盖 2560 条物理样本）。
- `train/sft_track3_scale_lr5e6_ga8.yaml`：Phase 6 Track 3 SFT `scale` Wave 2 温和学习率配置（四类物理场景、`lr=5e-6`、`ga=8`、`warmup=5`、`max_steps=60`、每 10 步全量 2500 条 val 门禁，避免过快抹除 Qwen3-ASR 预训练中文标点先验）。
- `train/sft_track3_scale_lr3e6_ga8.yaml`：Phase 6 Track 3 SFT `scale` Wave 2 超温和保标点配置（四类物理场景、`lr=3e-6`、`ga=8`、`warmup=5`、`max_steps=60`、每 10 步全量 2500 条 val 门禁）。
- `train/sft_track3_scale_lr1e5_ga8.yaml`：Phase 6 Track 3 SFT `scale` Wave 2 Phase 2 对齐学习率配置（四类物理场景、`lr=1e-5`、`ga=8`、`warmup=5`、`max_steps=60`、每 10 步全量 2500 条 val 门禁）。
- `train/sft_track3_scale_phys_only_lr3e6_ga8.yaml`：Phase 6 Track 3 SFT `scale` Wave 2 纯物理退化三场景配置（仅 `[noise, reverb, far_field]`、`lr=3e-6`、`ga=8`、`warmup=5`、`max_steps=60`、每 10 步全量 2500 条 val 门禁，验证剔除 `clean` 稀释后对远场与混响的提升）。
- `train/sft_track3_scale_all5_lr3e6_ga8.yaml`：Phase 6 Track 3 SFT `scale` Wave 2 温和五场景对照消融（含 `dropout`、`lr=3e-6`、`ga=8`、`warmup=5`、`max_steps=60`、每 10 步全量 2500 条 val 门禁）。
- `train/sft_track3_scale_phys_lr2e6_ga32.yaml`：Phase 6 Track 3 SFT `scale` Wave 2B 高声学多样性低漂移配置（纯物理退化三场景 `[noise, reverb, far_field]`、`ga=32` 每 10 步覆盖 320 条实测房间样本、`lr=2e-6` 超温和保标点、`warmup=3`、`max_steps=40`）。
- `train/sft_track3b_punct_lr2e5_ga16.yaml`：Phase 6 Track 3B 保标点深物理 SFT 主配置（`train_punct.jsonl` + `target_text_field: text_punct`，四物理场景、`lr=2e-5`、`ga=16`、`r=16`、`max_steps=160`、每 40 步评测）。
- `train/sft_track3b_punct_lr1e5_ga16.yaml`：Phase 6 Track 3B 保标点深物理 SFT 温和对照（`target_text_field: text_punct`，四物理场景、`lr=1e-5`、`ga=16`、`r=16`、`max_steps=160`）。
- `train/sft_track3b_punct_phys_lr2e5_ga16.yaml`：Phase 6 Track 3B 保标点纯退化三场景配置（`target_text_field: text_punct`，仅 `[noise, reverb, far_field]`、`lr=2e-5`、`ga=16`、`r=16`、`max_steps=160`）。
- `train/sft_track3b_punct_phys_lr5e5_ga32.yaml`：Phase 6 Track 3B 保标点大批量高学习率去混响配置（`target_text_field: text_punct`，仅 `[noise, reverb, far_field]`、`lr=5e-5`、`ga=32` 覆盖 5120 条实测退化样本、`max_steps=160`）。
- `train/sft_track3b_punct_r32_lr2e5_ga16.yaml`：Phase 6 Track 3B 保标点高秩容量配置（`target_text_field: text_punct`，四物理场景、`rank=32`/`alpha=64`、`lr=2e-5`、`ga=16`、`max_steps=160`）。
- `train/sft_track3b_punct_lr1e5_ga8_step20.yaml`：Phase 6 Track 3B 保标点细粒度早停对照（`target_text_field: text_punct`，四物理场景、`lr=1e-5`、`ga=8`、`max_steps=100`、每 20 步评测）。
- `train/sft_track3b_punct_phys_r32_lr5e5_ga32_320.yaml`：Phase 6 Track 3B 保标点全 epoch 级高秩去混响配置（`target_text_field: text_punct`，纯物理退化三场景 `[noise, reverb, far_field]`、`rank=32`/`alpha=64`、`lr=5e-5`、`ga=32`、`max_steps=320` 覆盖 10,240 条实测房间退化样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_all4_lr5e5_ga32_320.yaml`：Phase 6 Track 3B 保标点全 epoch 级四场景均衡配置（`target_text_field: text_punct`，四物理场景 `[clean, noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=5e-5`、`ga=32`、`max_steps=320` 覆盖 10,240 条样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_lr1e4_ga32.yaml`：Phase 6 Track 3B 保标点大批量高学习率上界探索配置（`target_text_field: text_punct`，纯物理退化三场景 `[noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=1e-4`、`ga=32`、`max_steps=160` 覆盖 5,120 条样本、每 40 步全量 val 门禁）。
- `train/sft_track3b_punct_all4_r32_lr1e4_ga32_320.yaml`：Phase 6 Track 3B 保标点全 epoch 级高秩高学习率四场景配置（`target_text_field: text_punct`，四物理场景 `[clean, noise, reverb, far_field]`、`rank=32`/`alpha=64`、`lr=1e-4`、`ga=32`、`max_steps=320` 覆盖 10,240 条样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_lr2e5_ga32_320.yaml`：Phase 6 Track 3B 保标点低方差大批量全 epoch 级甜蜜点配置（`target_text_field: text_punct`，纯物理退化三场景 `[noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=2e-5`、`ga=32`、`max_steps=320` 覆盖 10,240 条退化样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_all4_r32_lr3e5_ga32.yaml`：Phase 6 Track 3B 保标点黄金中点高秩四场景配置（`target_text_field: text_punct`，四物理场景 `[clean, noise, reverb, far_field]`、`rank=32`/`alpha=64`、`lr=3e-5`、`ga=32`、`max_steps=160` 覆盖 5,120 条样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_lr3e5_ga32.yaml`：Phase 6 Track 3B 保标点黄金中点纯物理退化配置（`target_text_field: text_punct`，纯物理退化三场景 `[noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=3e-5`、`ga=32`、`max_steps=160` 覆盖 5,120 条退化样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_all4_lr2e5_ga32_320.yaml`：Phase 6 Track 3B 甜蜜点孪生 A 配置（四物理场景 `[clean, noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=2e-5`、`ga=32`、`max_steps=320` 覆盖 10,240 条样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_r32_lr2e5_ga32_320.yaml`：Phase 6 Track 3B 甜蜜点孪生 B 高秩配置（纯物理退化三场景 `[noise, reverb, far_field]`、`rank=32`/`alpha=64`、`lr=2e-5`、`ga=32`、`max_steps=320` 覆盖 10,240 条退化样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_lr15e6_ga32_320.yaml`：Phase 6 Track 3B 甜蜜点超低方差近邻配置（纯物理退化三场景 `[noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=1.5e-5`、`ga=32`、`max_steps=320` 覆盖 10,240 条退化样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_all4_r32_lr2e5_ga32_320.yaml`：Phase 6 Track 3B 甜蜜点 2x2 网格补齐配置（四物理场景 `[clean, noise, reverb, far_field]`、`rank=32`/`alpha=64`、`lr=2e-5`、`ga=32`、`max_steps=320` 覆盖 10,240 条样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_lr15e6_ga32_160.yaml`：Phase 6 Track 3B 超低方差 160 步余弦收官配置（纯物理退化三场景 `[noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=1.5e-5`、`ga=32`、`max_steps=160` 覆盖 5,120 条退化样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_lr1e5_ga32_160.yaml`：Phase 6 Track 3B 低学习率下界 160 步余弦收官配置（纯物理退化三场景 `[noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=1.0e-5`、`ga=32`、`max_steps=160` 覆盖 5,120 条退化样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_lr2e5_ga32_160.yaml`：Phase 6 Track 3B 冠军区 160 步余弦退火收官配置（纯物理退化三场景 `[noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=2.0e-5`、`ga=32`、`max_steps=160` 覆盖 5,120 条退化样本、每 80 步全量 val 门禁）。
- `train/sft_track3b_punct_phys_lr25e6_ga32_160.yaml`：Phase 6 Track 3B 冠军区 160 步余弦中点配置（纯物理退化三场景 `[noise, reverb, far_field]`、`rank=16`/`alpha=32`、`lr=2.5e-5`、`ga=32`、`max_steps=160` 覆盖 5,120 条退化样本、每 80 步全量 val 门禁）。
- `train/aishell_slr28_val_sft_w1s240_20261124.json`：Phase 6 Track 3B Stage 1 最终冠军不可变 SFT release（`sft_06b_zh_20261124T194600_step000240`、`phys r=16 lr=1.5e-5 ga=32 Step 240`，2,500 行 readback `mismatch_count=0`，`micro_cer=6.090%`/`micro_wer=9.644%`、`-117` 字错误、`-58` 物理退化、`P_CER(phys3)=99.65%`）在 `aishell1_slr28` `val.jsonl` 上的固定父指标基线，供 Stage 2 Track 3B DPO 门禁引用。
- `train/dpo_track3b_smoke.yaml`：Phase 6 Track 3B Stage 2 标点匹配 DPO 6-step 中断恢复 smoke 配置（`profile: smoke`、`interrupt_after_step: 5`、`save_steps: 5`、`eval_steps: 6`、`max_steps: 6`，使用 10 条 `smoke_manifest/val.jsonl`，父 release 为 `sft_06b_zh_20261124T194600_step000240`，不可发布）。
- `train/dpo_track3b_e1_excl_lr5e6_b01_ga4.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 主配置（`me=1`、去 clean 同音拷贝、`margin=[-4,0]`、`243` train / `160` val、`lr=5e-6`、`beta=0.1`、`ga=4`、`max_steps=60`、每 20 步全量 val 门禁、`dropout_anchor_weight=0.05`）。
- `train/dpo_track3b_e1_excl_lr2e6_b01_ga4.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 低学习率配置（`me=1`、去 clean 同音、`margin=[-4,0]`、`lr=2e-6`、`beta=0.1`、`ga=4`、`max_steps=60`、每 20 步评测）。
- `train/dpo_track3b_e1_excl_lr1e5_b01_ga4.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 高学习率配置（`me=1`、去 clean 同音、`margin=[-4,0]`、`lr=1e-5`、`beta=0.1`、`ga=4`、`max_steps=60`、每 20 步评测）。
- `train/dpo_track3b_e1_excl_lr5e6_b02_ga4.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 高 `beta=0.2` 正则配置（`me=1`、去 clean 同音、`margin=[-4,0]`、`lr=5e-6`、`beta=0.2`、`ga=4`、`max_steps=60`、每 20 步评测）。
- `train/dpo_track3b_e2_excl_lr5e6_b01_ga8.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 双字替换主配置（`me=2`、去 clean 同音、`margin=[-4,0]`、`388` train / `160` val、`lr=5e-6`、`beta=0.1`、`ga=8`、`max_steps=48`、每 16 步评测）。
- `train/dpo_track3b_e2_excl_lr2e6_b01_ga8.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 双字替换低学习率配置（`me=2`、去 clean 同音、`margin=[-4,0]`、`lr=2e-6`、`beta=0.1`、`ga=8`、`max_steps=48`、每 16 步评测）。
- `train/dpo_track3b_e2_excl_m8_lr5e6_b005_ga8.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 宽 margin + 低 beta 配置（`me=2`、去 clean 同音、`margin=[-8,0]`、`476` train / `242` val、`lr=5e-6`、`beta=0.05`、`ga=8`、`max_steps=54`、每 18 步评测、`dropout_anchor_weight=0.02`）。
- `train/dpo_track3b_e1_all_lr3e6_b01_ga8.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 单字全覆盖配置（`me=1`、保留全部退化对 `exclude_clean_identical_rejected=false`、`margin=[-4,0]`、`676` train / `160` val、`lr=3e-6`、`beta=0.1`、`ga=8`、`max_steps=75`、每 25 步评测）。
- `train/dpo_track3b_e2_all_lr3e6_b01_ga16.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 双字全覆盖大批量配置（`me=2`、`exclude_clean_identical_rejected=false`、`margin=[-4,0]`、`920` train / `160` val、`lr=3e-6`、`beta=0.1`、`ga=16`、`max_steps=54`、每 18 步评测）。
- `train/dpo_track3b_e2_excl_lr1e6_b01_ga4.yaml`：Phase 6 Track 3B Stage 2 DPO `scale` 超温和长步数配置（`me=2`、去 clean 同音、`margin=[-4,0]`、`388` train / `160` val、`lr=1e-6`、`beta=0.1`、`ga=4`、`max_steps=90`、每 30 步评测）。
- `train/dpo_track3b_w2_e1_lr5e6_b02_a08_e15.yaml`：Phase 6 Track 3B Stage 2 DPO Wave 2 甜区配置（`w3` 强锚孪生：`me=1`、去 clean 同音、`lr=5e-6`、`beta=0.2`、`dropout_anchor_weight=0.08`、`ga=4`、`max_steps=45`、每 15 步全量评测）。
- `train/dpo_track3b_w2_e1_lr75e7_b02_a05_e15.yaml`：Phase 6 Track 3B Stage 2 DPO Wave 2 甜区配置（`lr=7.5e-6`、`beta=0.2`、`anchor=0.05`、`ga=4`、`max_steps=45`、每 15 步评测）。
- `train/dpo_track3b_w2_e1_lr5e6_b03_a08_e15.yaml`：Phase 6 Track 3B Stage 2 DPO Wave 2 甜区配置（`lr=5e-6`、`beta=0.3`、`anchor=0.08`、`ga=4`、`max_steps=45`、每 15 步评测）。
- `train/dpo_track3b_w2_e1_lr1e5_b01_a08_e15.yaml`：Phase 6 Track 3B Stage 2 DPO Wave 2 甜区配置（`w2` 强锚细评孪生：`lr=1e-5`、`beta=0.1`、`anchor=0.08`、`ga=4`、`max_steps=45`、每 15 步评测）。
- `train/dpo_track3b_w2_e1_lr75e7_b015_a08_e15.yaml`：Phase 6 Track 3B Stage 2 DPO Wave 2 几何中点配置（`lr=7.5e-6`、`beta=0.15`、`anchor=0.08`、`ga=4`、`max_steps=45`、每 15 步评测）。
- `train/dpo_track3b_w2_e1_m8_lr5e6_b02_a08_e15.yaml`：Phase 6 Track 3B Stage 2 DPO Wave 2 宽 margin 甜区配置（`margin=[-8,0]`、`279` train / `242` val、`lr=5e-6`、`beta=0.2`、`anchor=0.08`、`max_steps=45`、每 15 步评测）。
- `train/dpo_track3b_w2_e1_lr5e6_b02_h100_a08_e15.yaml`：Phase 6 Track 3B Stage 2 DPO Wave 2 方案 1 余弦配置（`scheduler_max_steps=100`、`lr=5e-6`、`beta=0.2`、`anchor=0.08`、`max_steps=45`、每 15 步评测）。
- `train/dpo_track3b_w2_e1_lr5e6_b02_a10_s2_e15.yaml`：Phase 6 Track 3B Stage 2 DPO Wave 2 双样本强静音锚配置（`anchor=0.10`、`dropout_anchor_samples_per_step=2`、`lr=5e-6`、`beta=0.2`、`max_steps=45`、每 15 步评测）。
- `train/aishell_slr28_val_dpo_w3s40_20261125.json`：Phase 6 Track 3B Stage 2 冠军不可变 DPO release（`dpo_06b_zh_20261125T005003_step000040`、`lr=5e-6 beta=0.2 Step 40`，2500 条读回 `mismatch_count=0`，`micro_cer=6.068%`、`-125` 字编辑、`-76` 词编辑、`-67` 物理退化）在 `aishell1_slr28` `val.jsonl` 上的固定父指标基线，供 Stage 3 Track 3B RL 门禁引用。
- `train/rl_track3b_smoke.yaml`：Phase 6 Track 3B Stage 3 标点匹配 RL 6-step 中断恢复 smoke 配置（`profile: smoke`、`interrupt_after_step: 5`、`save_steps: 5`、`eval_steps: 6`、`max_steps: 6`，使用 10 条 `smoke_manifest/val.jsonl`，父 release 为 `dpo_06b_zh_20261125T005003_step000040`，不可发布）。
- `train/rl_track3b_lr1e6_kl010_e2_ga4.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 主配置（`allow_matched_punctuation=true`、`lr=1e-6`、`kl_beta=0.10`、`max_edit_count=2`、`ga=4`、`dropout_anchor_weight=0.05`、`max_steps=15`、每 5 步全量 val 与 `raw_kl` 门禁）。
- `train/rl_track3b_lr5e7_kl010_e2_ga4.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 超温和学习率配置（`lr=5e-7`、`kl_beta=0.10`、`me=2`、`ga=4`、`anchor=0.05`、`max_steps=15`、每 5 步评测）。
- `train/rl_track3b_lr2e6_kl010_e2_ga4.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 强学习率配置（`lr=2e-6`、`kl_beta=0.10`、`me=2`、`ga=4`、`anchor=0.05`、`max_steps=15`、每 5 步评测）。
- `train/rl_track3b_lr1e6_kl005_e2_ga4.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 轻 KL 约束配置（`lr=1e-6`、`kl_beta=0.05`、`me=2`、`ga=4`、`anchor=0.05`、`max_steps=15`、每 5 步评测）。
- `train/rl_track3b_lr1e6_kl020_e2_ga4.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 强 KL 锚定配置（`lr=1e-6`、`kl_beta=0.20`、`me=2`、`ga=4`、`anchor=0.05`、`max_steps=15`、每 5 步评测）。
- `train/rl_track3b_lr1e6_kl010_e1_ga4.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 严格单字替换配置（`lr=1e-6`、`kl_beta=0.10`、`me=1`、`ga=4`、`anchor=0.05`、`max_steps=15`、每 5 步评测）。
- `train/rl_track3b_lr2e6_kl010_e1_ga4.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 单字替换强学习率配置（`lr=2e-6`、`kl_beta=0.10`、`me=1`、`ga=4`、`anchor=0.05`、`max_steps=15`、每 5 步评测）。
- `train/rl_track3b_lr1e6_kl010_e2_ga8.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 大批量低方差配置（`lr=1e-6`、`kl_beta=0.10`、`me=2`、`ga=8`、`anchor=0.05`、`max_steps=12`、每 4 步评测）。
- `train/rl_track3b_lr1e6_kl010_e2_ga4_ffrv.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 远场+混响专精配置（`train_scenarios=[far_field, reverb]`、`lr=1e-6`、`kl_beta=0.10`、`me=2`、`ga=4`、`anchor=0.05`、`max_steps=15`、每 5 步评测）。
- `train/rl_track3b_lr15e7_kl015_e2_ga4_a03.yaml`：Phase 6 Track 3B Stage 3 RL `scale` 均衡中点配置（`lr=1.5e-6`、`kl_beta=0.15`、`me=2`、`ga=4`、`anchor=0.03`、`max_steps=15`、每 5 步评测）。
- `train/rl_track3b_w2_fp32_lr5e7_kl010_a01_e8.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 `float32` LoRA 温和配置（`trainable_dtype=float32`、`lr=5e-7`、`kl_beta=0.10`、`anchor=0.01`、`me=2`、`ga=4`、`max_steps=24`、每 8 步评测）。
- `train/rl_track3b_w2_fp32_lr1e6_kl010_a01_e8.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 `float32` LoRA 核心配置（`trainable_dtype=float32`、`lr=1e-6`、`kl_beta=0.10`、`anchor=0.01`、`me=2`、`ga=4`、`max_steps=24`、每 8 步评测）。
- `train/rl_track3b_w2_fp32_lr2e6_kl010_a01_e8.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 `float32` LoRA 强信号配置（`trainable_dtype=float32`、`lr=2e-6`、`kl_beta=0.10`、`anchor=0.01`、`me=2`、`ga=4`、`max_steps=24`、每 8 步评测）。
- `train/rl_track3b_w2_fp32_lr3e6_kl010_a01_e8.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 `float32` LoRA 高学习率配置（`trainable_dtype=float32`、`lr=3e-6`、`kl_beta=0.10`、`anchor=0.01`、`me=2`、`ga=4`、`max_steps=24`、每 8 步评测）。
- `train/rl_track3b_w2_fp32_lr5e6_kl010_a01_e8.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 `float32` LoRA 上界配置（`trainable_dtype=float32`、`lr=5e-6`、`kl_beta=0.10`、`anchor=0.01`、`me=2`、`ga=4`、`max_steps=24`、每 8 步评测）。
- `train/rl_track3b_w2_fp32_lr2e6_kl005_a00_e10.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 纯 KL 约束无静音锚配置（`trainable_dtype=float32`、`lr=2e-6`、`kl_beta=0.05`、`anchor=0.0`、`me=2`、`ga=4`、`max_steps=30`、每 10 步评测）。
- `train/rl_track3b_w2_fp32_lr1e6_kl002_a005_e10.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 轻 KL + 极轻静音锚配置（`trainable_dtype=float32`、`lr=1e-6`、`kl_beta=0.02`、`anchor=0.005`、`me=2`、`ga=4`、`max_steps=30`、每 10 步评测）。
- `train/rl_track3b_w2_fp32_lr2e6_kl010_e1_a01_e8.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 `float32` 单字替换配置（`trainable_dtype=float32`、`lr=2e-6`、`kl_beta=0.10`、`me=1`、`anchor=0.01`、`ga=4`、`max_steps=24`、每 8 步评测）。
- `train/rl_track3b_w2_fp32_lr2e6_kl010_ffrv_a01_e8.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 `float32` 远场+混响专精配置（`trainable_dtype=float32`、`train_scenarios=[far_field, reverb]`、`lr=2e-6`、`kl_beta=0.10`、`anchor=0.01`、`ga=4`、`max_steps=24`、每 8 步评测）。
- `train/rl_track3b_w2_fp32_lr1e6_kl010_a02_ga8_e6.yaml`：Phase 6 Track 3B Stage 3 RL Wave 2 `float32` 大批量低方差配置（`trainable_dtype=float32`、`ga=8`、`lr=1e-6`、`kl_beta=0.10`、`anchor=0.02`、`me=2`、`max_steps=18`、每 6 步评测）。

训练字段以 [`docs/qwen3-asr/04_training_plan.md`](../docs/qwen3-asr/04_training_plan.md) 为准。数据字段以 [`docs/qwen3-asr/03_data_plan.md`](../docs/qwen3-asr/03_data_plan.md) 为准。

## 主要输入输出

- **输入**：开发者或实验调度器指定的超参数与路径。
- **输出**：在运行启动时解析并固化至 `runs/<run_id>/resolved_config.yaml`，同时写语义 `config_sha256` 和 YAML 字节 hash；训练 checkpoint 必须携带该副本。

## 使用入口

由各 CLI 脚本通过 `--config` 参数读取，例如：
```bash
python3 train/train_sft.py --config configs/train/sft_pilot.yaml ...
```

开发阶段先用文档模板生成配置，配置提交前运行未知字段、有限数值、stage/parent release、`decode.eval.do_sample=false`、`world_size=1` 和必填 gate 的 preflight。恢复只能使用原 run 的 resolved config；改变学习率、预算、阈值或 LoRA 目标必须新建 run。

## 维护要求

新增配置模板或修改配置 schema 时，必须在同一提交同步更新对应目录的 `README.md` 和训练/数据方案文档。
