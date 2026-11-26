# DPO v2 大规模中文数据扩容、抗锁死门禁与 LR Warmup 训练方案

最后更新：2026-11-26（分支 `dpo-v2-scale`，基于 `main`）

## 背景

Track 3B 的历史发布 DPO 在封存测试集上与 SFT 接近，在 AliMeeting OOD 上少 3 个字编辑、8 个词编辑。首轮实验检验有效偏好对数量与学习率对这一差距的影响。历史记录可以回放门禁行为，数据量和学习率的因果贡献仍待四组实验测量。[历史结果诊断](14_dpo_rl_diagnosis.md)

1. **有效训练偏好对数量**。
   - Track 3B 使用 5 小时训练子集，共 3,937 条源句。`max_edit_count=1`、同源 clean 排除及参考分差 `[-4,0]` 使 w3 保留 243 对、200 条独立源句。w3 选中第 40 步，累积 4，已处理 160 对次。
   - 另有固定留出的 160 对用来评估偏好，w3 把正确转写排在错误转写之前的比例为 4/160（2.5%）。w2 的峰值学习率为 `1e-5`，其相对 SFT 的物理三场景编辑差从第 20 步的 −11 转为第 40 步的 +6。
2. **旧门禁把未受训 `dropout` 微小噪声和早期低 preference 直接等同于硬熔断（过早锁死）**：
   - 旧门禁将 `degraded_macro_cer` 固定为含 `dropout` 的四场景均值，且只要 `macro_delta > 0.0` 或 `preference_accuracy < 0.55` 就立刻 `risk_stop=True`。
   - 对 Track 3B DPO Wave 1 离线重放显示：`t3-w8`（920 对大批量配置 `me=2, all, ga=16, lr=3e-6`）在 Step 18 实际已取得 `phys3 = -3` 编辑（`far_field -3, noise -2, reverb +2`）、`clean = -2`、`pref = 0.575`、`gap_insertions = 0`，却仅因未参与训练的 `dropout` 场景波动 `+5` 字（`+0.000683`）把四场景均值拉高 `+0.000068` 而被立即处死；`t3-w5`（388 对，`me=2, excl, ga=8, lr=2e-6`）在 Step 16 也已取得 `phys3 = -2`、`clean = -1`、`gap_ins = 0`，同样因 `dropout +7` 与早期 `pref = 0.4625` 过早中止。
   - 与之形成鲜明对比的是：Track 3B Wave 1 全部 12 个评测步的 `locate_gap_insertions == 0`、`dropout_delta ∈ [+0.000137, +0.001229]`；而 Phase 3 所有真实的静音幻觉事故（如 `20261018T132840`、`20261019T012838`、`20261020T041218` Step 41 吐出 `Human Rights Watch`）均表现为 `locate_gap_insertions >= 10` 且 `dropout_delta >= +0.001775`（最高 `+0.006417`）。两者在指标空间中完全可分。
3. **Warmup 期 `bfloat16` 低学习率 ULP 下溢风险**：
   - Track 3B DPO 虽已有线性 warmup（`warmup_steps = 5..9`，约占总步数 10%），但 LoRA 主权重仍为 `bfloat16`（7 位尾数）；在 240 步 SFT 后 `lora_A` 中位幅值已达 `~1.45e-2`（`bf16` ULP `~1.22e-4`），warmup 初期的低有效 LR 步长在 `bf16` 下会被严重舍入截断，必须与 `lora.trainable_dtype: float32` 配合使用，使完整 linear warmup + cosine 退火曲线产生真实平滑的参数更新。

## 范围

### 本次做什么

1. 基于 `main` 切出独立分支 `dpo-v2-scale`，冻结并保留全部历史 Phase 1–6 配置、哈希与结论不变。
2. 制定面向 **`5,000–15,000+` 高信号偏好对** 的中文开源语音语料分层扩容方案、许可合规边界与零说话人/录音泄漏切分合同。
3. 在 `train/sft_schema.py`、`train/sft_contract.py`、`train/dpo_contract.py`、`train/train_dpo.py` 中实现**向后完全兼容的 DPO v2 解耦抗锁死门禁**（7 个可选 `gate.*` 字段），在严格守住 `clean`、健康度、未受训 `dropout` 上限与静音幻觉（`max_gap_insertions`）底线的前提下，消除过早锁死。
4. 新增本地离线门禁重放、合成边界压测与 LR Warmup 曲线核查脚本 `scripts/replay_dpo_gate.py`，先在本机用真实历史评测产物验证新旧门禁差异与幻觉拦截能力，再进入后续 Colab 大数据物化与 GPU 训练。
5. 新增完整单元测试 `tests/test_dpo_v2_gate.py`，同步更新 `docs/qwen3-asr/` 与各受管目录 `README.md`。

### 本次不做什么

- 不在本地 CPU 环境下载多 GB 公开语料或启动 Colab A100 GPU 训练（本阶段先完成方案冻结、代码实现与本地门禁推理/压测验证）。
- 不修改任何历史 `profile: pilot` 或 Track 3B 既有 YAML 配置文件及其语义配置 SHA-256。
- 不将 ShareAlike 协议语料（`AliMeeting`、`AISHELL-4`）混入任何训练集或 DPO 偏好对。
- 不提前将任何 DPO v2 GPU 训练或效果提升标记为已完成。

## 设计

### 一、中文开源语音语料分层扩容方案（目标：有效 DPO 对放大 20×–60×）

为从根本上打破 `243` 对单字瓶颈，同时补齐朗读体之外的真实会议/日常噪声与自然口语分布，语料按**许可洁净度、获取成本与声学价值**分三层接入，并维持 ShareAlike 只读隔离区：

| 层级 | 语料名称 | 开源地址 / 标识 | 总时长 / 规模 | 许可证与使用边界 | 声学与文本价值 | 拟抽样配额与泄漏隔离键 |
| --- | --- | --- | --- | --- | --- | --- |
| **Tier 1（立即 10×–30× 扩容）** | **AISHELL-1 Full Train** | OpenSLR 33 (`data_aishell.tgz`) | ~150h 训练（`120,098` 句，340 说话人） | Apache-2.0 / 学术非商（沿用 `03_data_plan.md` 交集最严原则） | 与现有 SLR28 物理管线 100% 兼容，当前仅用了 5h（3.2%），直接释放剩余 145h | 首批抽取 **50h–100h**（~40k–80k 句，覆盖全部 340 名官方 `train` 说话人，与 40 `dev` / 20 `test` 说话人严格不交），叠加 SLR28 物理 RIR/真实噪声 |
| **Tier 1（纯商可用洁净补充）** | **THCHS-30** | OpenSLR 18 (`data_thchs30.tgz` 6.4GB + `test-noise.tgz` 1.9GB) | ~30–35h（新闻朗读 + 真实车/咖啡馆/白噪） | **Apache-2.0**（无学术附加限制） | 独立说话人池与真实环境噪声源，低门槛直接解包 | 全量 `train` split 按 `speaker_id` 隔离进入训练集（不跨入 val/test） |
| **Tier 1（公有领域多口音）** | **Common Voice zh-CN** | HuggingFace (`mozilla-foundation/common_voice_*` / 镜像) | 数十小时高质量验证集（持续增长） | **CC0-1.0**（公有领域） | 众包多口音、真实消费级麦克风录音与房间底噪 | 仅取 `validated` 且 `up_votes > down_votes`、按 `client_id` 严格单 split 隔离 |
| **Tier 2（真实噪声与自然口语破局）** | **WenetSpeech S / M** | OpenSLR 121 / HF `wenet-e2e/wenetspeech` | S=100h / M=1,000h / L=10,000h（高置信度 `confidence >= 0.95`） | CC BY 4.0（原始媒体版权属原作者，仅限学术非商研究） | YouTube / 播客 / 访谈 / 有声书的真实背景音乐、混响、口语填充词与专有名词，直接弥补朗读体局限 | 首批抽取 **WenetSpeech-S（100h）** 中高置信 `confidence >= 0.95`、`duration ∈ [1.0, 20.0]s`、无拉丁混写子集，按 `recording_id`（`aid`）整段归属 `train` |
| **Tier 2（真实多房间远场）** | **RealMAN** | HF `AISHELL/RealMAN` | 83.7h 真实语音（48.3h 静态 + 35.4h 移动）+ 144.5h 真实 32 场景噪声 | **CC BY 4.0**（无 ShareAlike 传染） | 32 种真实室内外环境（会议室/教室/食堂/地铁/街道）真实远场阵列表现，无合成 Sim-to-Real 鸿沟 | 确定性抽取固定单通道（如 `ch0`）与近场同步标注，按 `scene_id + speaker_id` 隔离，仅 `train` 进入训练 |
| **Tier 2（自然对话补充）** | **Emilia-YODAS ZH** | HF `amphion/Emilia-Dataset`（YODAS 子集） | 数千小时级 WebDataset `.tar` | **CC BY 4.0**（仅限 YODAS 子集；原 Emilia 主集为 CC BY-NC 4.0 须区分） | 自然对话语速与真实声学背景，流式 tar 易分片采样 | 抽样 30h–50h 高 DNSMOS 语音，按源视频 ID 隔离并过滤低置信伪标 |
| **Tier 3（学术非商高容量备选）** | **MagicData-RAMC** / **Aidatatang_200zh** / **ST-CMDS** / **KeSpeech** | OpenSLR 123 (180h) / OpenSLR 62 (200h) / OpenSLR 38 (100h) / KeSpeech (1,542h) | 合计 >2,000h | **CC BY-NC-ND 4.0** / 学术非商 | MagicData-RAMC（180h 15 领域手机自由对话）尤其贴合真实会议口语；KeSpeech 覆盖 8 大官话方言区 | 仅在 Tier 1 + Tier 2 仍需扩大对数时按学术非商边界分批接入，严禁用于商业权重 |
| **严格只读隔离区（禁止训练）** | **AliMeeting** / **AISHELL-4** | OpenSLR 119 / OpenSLR 111 | 4h Eval / 120h 8 麦会议 | **CC BY-SA 4.0**（ShareAlike） | 仅作真实多人会议远场 OOD 基准 | `trainable: false`，**严禁**进入任何训练 manifest、DPO pair 或 RL rollout |

#### DPO v2 语料量纲与 Pair 产能估算

- **基准参照（Track 3B v1）**：`3,937` 条 AISHELL-1 源语句 × 3 类物理退化（`noise`/`reverb`/`far_field`）= `11,811` 条退化音频 → SFT 产生 `2,557` 条物理错句 → `me=1, excl, margin=[-4,0]` 过滤剩 `243` 对，`me=2, excl, margin=[-8,0]` 过滤剩 `476` 对。
- **DPO v2 第一阶段（Phase A：Tier 1 扩容，AISHELL-1 50h + THCHS-30 25h ≈ `60,000` 源语句）**：
  - 叠加 SLR28 物理退化（为每个新源句保留 clean 音频和父模型预测，生成 noise、reverb、far_field 三类物理增强；复用同一训练切分的历史 dropout 音频与父模型预测作为锚池）→ `~180,000` 条物理退化音频；
  - 用已发布 SFT 冠军（`sft_06b_zh_20261124T194600_step000240`）分片 greedy 推理 → 预计产生 `~35,000–45,000` 条物理退化错句；
  - 经标点投射（`--align-punctuation-from-rejected`）+ 原位替换过滤（`max_edit_count: 2`、`ref_margin_min: -8.0`、`exclude_clean_identical_rejected: true`）→ 预计稳定产出 **`5,000–8,000` 条高信号纯声学 DPO 偏好对**（约 v1 的 **20×–33×**）。
- **DPO v2 第二阶段（Phase B：叠加 Tier 2，WenetSpeech-S 50h + RealMAN 40h ≈ `+50,000` 真实噪声/远场语句）**：
  - 真实声学错误率更高（约 6%–10% CER），无需人工加噪即可直接产出真实错题对 → 追加 **`5,000–8,000` 条真实声学偏好对**，总有效训练集达到 **`10,000–15,000+` 对**（约 v1 的 **40×–60×**），同时直接补足真实会议室远场与口语分布。
- **验证集与测试集零污染铁律**：
  - `val.jsonl`（2,500 条 AISHELL-1 SLR28 官方 `dev`）、封存 `test.jsonl`（2,500 条 AISHELL-1 SLR28 官方 `test`）和只读 `AliMeeting Eval`（1,000 条）保持 **100% 不动**，所有 DPO v2 均与 Track 3B 基线同口径直接可比；
  - 全部新增语料必须通过 `speaker_id` / `recording_id` / 归一化文本 SHA-256 三重去重，确保与 `val.jsonl`、`test.jsonl`、`AliMeeting Eval` 的说话人、音频字节与文本交集严格为空。

---

### 二、DPO v2 解耦抗锁死门禁设计（100% 向后兼容）

核心设计原则：**绝不放宽最终选中模型的质量红线（`passed=True` 仍要求物理退化不恶化、clean 安全、健康度达标、preference 过线、无静音幻觉），但把“禁止入选（软风险）”与“立即处死训练（硬安全熔断）”解耦**。

#### 1. 配置 Schema 扩展（`train/sft_schema.py`）

在 `qwen3-asr-train-v1` 的 `gate` 块新增 7 个**全部可选**字段；**省略或取历史默认值时，不写入 `resolved["gate"]`**，从而保证全部历史 SFT / DPO / RL 配置文件的 `semantic_config_sha256`、`semantic_dpo_config_sha256`、`semantic_rl_config_sha256` **逐位零变动**：

| 字段名 | 类型与合法边界 | 历史缺省值（省略行为） | DPO v2 推荐值 | 语义说明 |
| --- | --- | --- | --- | --- |
| `gate.degraded_macro_scenarios` | 可选非空不重复列表，元素须为 `("dropout", "far_field", "noise", "reverb")` 子集；长度 `>= min_improved_degraded_scenarios` | `None`（全部 4 类退化） | `["far_field", "noise", "reverb"]` | 指定计算 `degraded_macro_cer`、`macro_delta` 与 `improved_degraded_scenarios` 的目标退化场景集合；将未受训的 `dropout` 从主选点宏平均解耦 |
| `gate.degraded_macro_stop_increase_max` | 可选有限浮点，须 `>= degraded_macro_increase_max` | `None`（等于 `degraded_macro_increase_max`） | `0.002` | 主退化宏平均的**硬熔断线**（`+0.20 pp`）。`macro_delta > stop_max` 立即硬熔断；`degraded_macro_increase_max < macro_delta <= stop_max` 记为软风险（`passed=False` 不可入选，但不立即中断训练） |
| `gate.dropout_cer_increase_max` | 可选有限浮点 `>= 0.0` | `None`（仅受通用 `scenario_cer_increase_max` 约束） | `0.002` | 未受训 `dropout` 场景的独立**硬安全上限**（约 `+14` 字 / 7,324 字）；超过即硬熔断，防止静音段大面积漂移 |
| `gate.max_gap_insertions` | 可选整数 `>= 0` | `None`（不启用） | `5` | 校验 `locate["locate_gap_insertions"]` 的**硬安全上限**；直接精准捕获 `Human Rights Watch` 等静音段幻觉（历史事故均 `>= 10`，健康运行均为 `0`），任何步数均立即硬熔断 |
| `gate.stop_on_low_preference` | 可选布尔值 | 缺省 `True`（不写入规范 dict） | `false` | 设为 `false` 时，`preference_accuracy < min_preference_accuracy` 或 `<= 0.5` 仍计入软风险（严禁 `passed=True` 或发布），但不会因单次早期评测低 preference 立即强杀训练 |
| `gate.min_stop_step` | 可选整数 `>= 0` | 缺省 `0`（不写入规范 dict） | `40`（`>= warmup_steps`） | 预热观察窗终点步数。在 `step <= min_stop_step` 期间，仅**硬安全风险**触发 `risk_stop=True`；观察窗内软失败的耐心计数为 0，失败的评测点不能入选 |
| `gate.max_consecutive_non_passing_evals` | 可选整数 `>= 1` | `None`（不启用） | `2` | 耐心早停上限。仅累计 `step > min_stop_step` 的评测，若连续未通过（`passed=False`）的评测次数达到该值，触发 `risk_stop=True` 优雅收敛止损，避免空烧 A100 算力 |

#### 2. 硬风险 vs 软风险裁决合同（`train/sft_contract.py` & `train/dpo_contract.py`）

- **硬风险（`hard_risks`，无论是否处于 `min_stop_step` 均立即 `passed=False, risk_stop=True`）**：
  1. 任意场景样本数 `< min_rows_per_scenario` 或验证对数 `< min_val_pairs`；
  2. 整体或任一场景的 `empty_rate`、`repeated_rate`、`long_rate`、`truncation_rate` 越界或 `valid_output_rate < valid_output_rate_min`；
  3. `clean_delta_vs_parent > clean_cer_increase_max` 或 `clean_delta_vs_base > clean_cer_increase_vs_base_max`；
  4. 任一退化场景 `delta > scenario_cer_increase_max`；
  5. 若配置 `dropout_cer_increase_max`：`dropout_delta > dropout_cer_increase_max`；
  6. 启用静音插入上限时，计数缺失、非法或超过上限均触发硬停；
  7. `macro_delta > degraded_macro_stop_increase_max`（未设该字段时等于 `degraded_macro_increase_max`）；
  8. 当 `stop_on_low_preference=True` 且不在 `min_stop_step` 观察窗内时，`preference_accuracy` 未过线；
  9. 当配置 `max_consecutive_non_passing_evals` 且不在 `min_stop_step` 观察窗内时，连续未通过次数 `>= max_consecutive_non_passing_evals`。
- **软风险（`soft_risks`，严格阻止 `passed=True` 与发布，但单次不触发 `risk_stop=True`）**：
  1. `degraded_macro_increase_max < macro_delta <= degraded_macro_stop_increase_max`；
  2. `preference_accuracy` 未过线且（`stop_on_low_preference=False` 或 `step <= min_stop_step`）。
- **通过与选点合同**：
  - `passed = (not hard_risks) and (not soft_risks) and (improved_degraded_scenarios >= min_improved_degraded_scenarios)`。
  - 当 `passed=True` 时，`risk_stop` 恒为 `False`、`consecutive_non_passing_evals` 归零，与 `train/dpo_release.py` 的发布校验 100% 兼容。
  - `train/train_dpo.py` 调用 `judge_dpo_checkpoint(..., step=step, locate=locate, prior_non_passing_evals=trailing_non_passing_evals(decisions, min_stop_step))`，跨 `--resume` 从 `eval_decisions.jsonl` 确定性恢复连续未通过计数。

---

### 三、DPO v2 LR Warmup + `float32` LoRA 充分步数配方与每步音频数（`ga`）预演

基于本机 Track 3B DPO Wave 1 真实 A100 日志（`results/phase6_track3_slr28_20261124/dpo_wave1/t3-w*_dpo_*/metrics.jsonl` 与评测时间戳）标定：
- **单步实测耗时**：`ga=4` `3.13s`、`ga=8` `5.89s`、`ga=16` `11.41s`，满足 $\text{step\_time}(ga) \approx 0.69 \times ga + 0.37\text{ s}$；
- **单步梯度范数标度（`beta=0.1`）**：`ga=4` 中位 `2.30`（100% 触发 `max_grad_norm=1.0` 强截断）、`ga=8` 中位 `1.71`、`ga=16` 中位 `1.20`，严格遵循 $\approx 2.30 \times (\beta/0.1) \times \sqrt{4/ga}$；
- **单次全量 2,500 条 `val` 评测耗时**：`~28.8m` 解码 + `~0.7m` 偏好打分 ≈ **`29.5 分钟/次`**；
- **步数修复动因**：旧配方 `240` 步（`eval_steps=40`，6 次评测）不仅优化步数偏少，而且训练仅 `46m` 却把 `177m`（`79%`）算力花在评测上；升级至 **`500` 步（1 epoch，`ga=16`，`min_train_pairs=8000`，`warmup=50`，`eval_steps=100` 共 5 次评测）** 后，单次 360 分钟 A100 会话内总耗时仅 `~243 分钟`（训练 `95m` + 评测 `148m`），步数翻倍且零闲置对、零短尾步。

| 维度 | Track 3B DPO v1（现状） | DPO v2 修复版（充分步数 + 可调 `ga` + Warmup） | 变更动因 |
| --- | --- | --- | --- |
| **有效训练偏好对** | `243`（单字）/ `920`（最大） | **`8,000–15,000+`**（Tier 1 + Tier 2，默认门槛 `8,000`） | 固定验证对和充分更新次数，用不重复样本把 1 epoch 自然撑到 `500+` 步 |
| **过滤规则** | `me=1`, `margin=[-4,0]` | `me=2`, `margin=[-8,0]`, `allow_matched_punctuation=true`, `exclude_clean_identical_rejected=true` | 在大规模语料上保留高信噪比 1–2 字声学替换对，同时屏蔽纯 clean 同音污染 |
| **LoRA 主权重精度** | BF16 | FP32 LoRA，基座仍为 BF16 | 保存小学习率累积的参数变化，使用同精度零步测量实际更新效果 |
| **每步音频数（`ga`）与总步数** | 累积 4，最多 60 步（选中 40 步） | 默认 **累积 16、500 步（8,000 对次 = 1.00 epoch）**；开跑前可通过预演工具与 `--ga` 灵活调整（如 `8/10/12/16/24/32`） | 既避免 `ga=4` 强梯度截断，又避免 `240` 步过短导致 AdamW 与余弦退火未展开 |
| **学习率与预热** | 峰值 `5e-6`，预热 6 步 | 峰值 `5e-6` 或 `1e-5`，**预热 50 步（10%）**，余弦降至峰值的 10% | 两种学习率使用同样的 500 步调度时长，充分覆盖 AdamW 动量预热 |
| **Beta 与静音锚** | `beta=0.2`，锚权重 0.05 | `beta=0.1`，锚权重 0.02 | 四组使用同一批父模型 dropout 假设作锚 |
| **评测间隔与门禁** | 每 20 步 | **每 100 步（1,600 对次，全程 5 次）**，观察窗 **100 步**，耐心 2 次 | 覆盖 warmup 与首评，clean、dropout、静音插入和输出健康风险每次评测均生效 |

#### 首轮受控实验与灵活 `ga` 调整入口

同一扩容池默认生成至少 `8,000` 对有效偏好对（可通过 `--ga` 灵活适配实际物化池规模，如 `5,000` 对池选用 `--ga 10 --max-steps 500` 或 `--ga 8 --max-steps 600`）。用固定种子取其中 `256` 对作为小池，大池使用整个合格池。各自比较 `5e-6` 和 `1e-5`，默认统一 FP32 LoRA、`ga=16`、`beta=0.1`、`anchor=0.02`、warmup `50`、`500` 步、每 `100` 步评测（共 `8,000` 对次；小池循环 `32` 轮与大池保持严格等更新预算）。观察窗 `100` 步内软失败计数为零，第 `200`、`300` 步连续软失败时停在第 `300` 步；硬风险在任一次评测立即停止。

四组配置由 `scripts/prepare_dpo_v2_sweep.py --manifest-dir <manifest> --pair-metadata <pairs_null/pair_metadata.json> --pairs-dir <pairs_logprob> --output-dir <新目录>` 在 CPU 上生成，默认配方见 `configs/experiments/dpo_v2_sweep.yaml`。工具支持可选 `--ga`、`--max-steps`、`--warmup-steps`、`--eval-steps`、`--min-stop-step` 与 `--rehearse-only` 灵活调整每步音频数和步数（改 `--ga` 时自动联动重算一致的 `warmup_steps`/`eval_steps`/`save_steps`/`keep_last`/`min_stop_step` 与小池 `max_epochs`，并拒绝 `max_steps × ga > 大池实际对数` 的静默短跑配置）。开跑前务必先用 `scripts/rehearse_dpo_schedule.py sweep` 对比各档 `ga` 的总步数、尾步大小、梯度截断风险、门禁最早早停步与实测标定 A100 耗时。

#### 输入、归因和发布

- 用 `scripts/build_dpo_pairs.py --align-punctuation-from-rejected --require-clean-controls` 构建偏好对。每条物理训练对携带同源 clean 贪心结果（归一化），包括 clean 转写完全正确的源句；clean 预测必须来自同一父 adapter、以 EOS 结束且归一化后非空。与金标不符的重复输出、重复源记录或缺少对照时，构建失败。验证预测来自同一父 adapter 或已发布 SFT 中 SHA-256 匹配的评分文件。带锚组将同一父模型、训练切分的 dropout 样本与预测并入建对输入。
- `train/reference_logprob.py --lora-trainable-dtype float32` 使用父 SFT 配置打分。缓存身份绑定两个原始 pair 文件、父 adapter、精度和设备型号，断点续算需匹配。v2 在首个参数更新前对固定留出对核查两侧 logprob（绝对误差 ≤1e-4），保存每对分差、绝对排序准确率及同精度父模型全量贪心 CER。
- 主验证组和补充组按照 Track 3B 的历史候选身份固定为 160 对和 82 对。配方固定验证名单的 SHA-256，以 pair ID、源句、场景及两侧文本核对身份，只用主组作训练门禁。训练池的参考分差范围由训练配置独立控制。
- v2 正式发布在 `train/dpo_release.py prepare --parent-release-dir <SFT发布目录>` 对发布 SFT 与候选的逐条评分进行源句簇配对 bootstrap。三类物理退化净减至少 20 次字编辑、95% CER 差值区间上界严格小于零才可封包。封包时核验 checkpoint 绑定的零步三份文件、所选步的偏好与评分文件以及两组候选顺序，重算相对零步的偏好命中率和绝对排序率。发布证据分别列出相对已发布 SFT 与同精度零步的变化；原有独立新进程读回仍是最后一道发布检查。
- 正式回放要求历史 9 组、12 次评测及 Phase 3 事故齐全。各门禁按自身首个硬停截断选点。缺失计数优先从原始评分或偏好桶恢复，无法还原时报告失败。训练恢复沿用已写入历史的硬停决定。

---

### 四、本地离线门禁推理与压测工具设计（`scripts/replay_dpo_gate.py`）

在触碰任何 GPU 之前，通过纯本地 CPU 脚本 `scripts/replay_dpo_gate.py` 复用真实生产函数 `judge_dpo_checkpoint`、`select_sft_checkpoint` 和 `lr_multiplier` 完成三类离线验证：

1. **真实历史运行双门禁对照重放**：
   - 自动加载本地 `results/phase6_track3_slr28_20261124/dpo_wave1/` 全部 9 组运行（12 个评测步）与 `results/phase3_dpo_gap_strip_probe_20261018/runs/20261018T132840_dpo_06b_zh_seed42/`（真实 `Human Rights Watch` 静音幻觉事故），逐一对比**旧门禁** vs **DPO v2 解耦门禁**的 `passed`、`risk_stop`、触发原因及最终选点。
2. **7 项合成边界压测（Stress Suite）**：
   - 静音幻觉（`gap_ins=10`）在 warmup 期内必须立即硬熔断；
   - `dropout` CER 飙升（`+0.005 > 0.002`）在 warmup 期内必须立即硬熔断；
   - `clean` 回退（`+0.015 > 0.01`）必须立即硬熔断；
   - 健康度越界（`empty_rate=0.02 > 0.01`）必须立即硬熔断；
   - Warmup 早期低 preference（`step=20, pref=0.48, phys3=-3`）必须保持 `passed=False, risk_stop=False`（既不误选也不锁死）；
   - `phys3` 轻微抖动后恢复（Step 40 `+0.0005` 软失误 → Step 80 `-0.0012` 通过）必须顺利跑通并选中 Step 80；
   - 持续无效训练（Step 80 软失误 → Step 120 软失误，`consecutive=2`）必须在 Step 120 触发耐心早停。
3. **LR Warmup + Cosine 调度曲线离线核验**：
   - 调用 `train.sft_contract.lr_multiplier` 输出 `configs/experiments/dpo_v2_sweep.yaml` 中配置的 DPO v2 配方（默认 `max_steps=500, warmup_steps=50, min_lr_ratio=0.1`）在关键步的精确学习率倍率与绝对 LR，确认线性爬坡与余弦退火逐点准确。

### 五、每步音频数（`ga`）与步数/耗时本地预演工具设计（`scripts/rehearse_dpo_schedule.py`）

- **背景与目标**：DPO 阶段中 `micro_batch_size` 固定为 1 对，因此 `batch.gradient_accumulation_steps`（`ga`）直接决定每步处理的音频/偏好对数量、单 epoch 总步数、梯度裁剪幅度以及单次 Colab 360 分钟 A100 会话内的训练/评测总耗时。通过本地纯 CPU 预演，先暴露步数偏少、重截断、短尾步和超时风险，再由 `scripts/prepare_dpo_v2_sweep.py --ga <K>` 灵活生成不可变 YAML。
- **输入与输出**：
  - `sweep` 子命令接受 `--train-pairs <N>`（或直接读取 `--pair-train <train_pairs.jsonl>` 按对齐过滤规则计数）、`--ga-list`（如 `4,8,10,12,16,20,24,32,64`）、`--max-steps`（可选固定步数或按 1 epoch 自动计算）、`--drop-tiny-tail`、`--base-lr`、`--beta`、`--warmup-ratio`、`--target-evals`、`--small-pool-pairs`、实测标定 A100 耗时参数（`--sec-per-pair 0.69`、`--step-overhead-sec 0.37`、`--eval-minutes 29.5`、`--max-runtime-minutes 360`）以及可选 `--output-json`。
  - 逐档复用生产 `iterate_updates`、`lr_multiplier`、`judge_dpo_checkpoint` 与 `trailing_non_passing_evals` 输出精确步数、大/小池尾步对数、估算 `grad_norm`、有效差异 token、各评测步学习率、最早耐心早停步、训练/评测/总耗时及风险标签（`TOO_FEW_STEPS`、`HEAVY_GRAD_CLIP`、`TINY_TAIL_STEP`、`RUNTIME_OVERFLOW`、`HIGH_EVAL_OVERHEAD`、`EARLY_STOP_BEFORE_WARMUP`），自动标出推荐 `ga`。

## 测试

1. **单元测试与历史哈希零漂移测试（`tests/test_dpo_v2_gate.py`、`tests/test_dpo_v2_inputs.py`）**：
   - 单测覆盖三阶段代表性配置及显式历史默认值。另对 `main@9ef7551` 和当前树的全部 95 份 YAML 用相同 manifest 哈希逐份运行语义哈希函数；
   - 验证 7 个新 `gate.*` 可选字段的正常解析、非法值/越界值拒绝；
   - 验证 `judge_sft_checkpoint`、`judge_dpo_checkpoint`、`judge_rl_checkpoint` 的向后兼容与硬/软风险解耦、预热窗与耐心早停；
   - 验证 `scripts/replay_dpo_gate.py`、`scripts/rehearse_dpo_schedule.py`（步数偏少/重截断/短尾步/超时诊断与推荐档位）以及 `scripts/prepare_dpo_v2_sweep.py` 的灵活 `--ga`/`--max-steps`/`--rehearse-only` 联动生成与不足池拒收。
2. **全量合同测试与目录 README 合同测试**：
   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   .venv/bin/python -m py_compile train/*.py scripts/*.py evaluation/*.py inference/*.py
   git diff --check
   ```
3. **本地离线门禁推理 + 每步音频数预演执行**：
   ```bash
   .venv/bin/python scripts/replay_dpo_gate.py
   .venv/bin/python scripts/rehearse_dpo_schedule.py sweep --train-pairs 8000 --ga-list 4,8,10,12,16,20,24,32,64 --drop-tiny-tail
   ```
   通过标准：`replay_dpo_gate.py` 退出码 `0`（全部 7 项压测 `PASS`、静音幻觉拦截、Wave 1 误杀解除）；`rehearse_dpo_schedule.py` 清晰标出各档 `ga` 的步数/梯度裁剪/耗时权衡并给出单会话可完赛的推荐 `ga`。

## 验收

### 阶段 A（本机方案、代码与离线门禁验证，本次交付范围）

- `docs/qwen3-asr/` 专项方案、数据、训练、测试、风险、进度文档与受管目录 `README.md` 全部同步完成。
- `.venv/bin/python -m unittest discover -s tests -v`（含 `test_dpo_v2_gate` 与 `test_directory_readmes`）100% 通过。
- `.venv/bin/python scripts/replay_dpo_gate.py` 在本机全部真实历史产物与 7 项压测场景上 100% 符合预期并验证 LR Warmup 曲线。

### 阶段 B（后续 Colab A100 语料物化与 DPO v2 正式训练，尚未执行）

- 完成 Tier 1（首批 AISHELL-1 50h + THCHS-30）及 Tier 2 语料的零泄漏 manifest 构建、SFT 分片推理、标点匹配 DPO pair 构建与离线 reference logprob 校验（`max_abs <= 1e-4`），有效训练对达到 **`>= 5,000` 对**。
- 先在 A100 上通过 6-step DPO v2 中断恢复 smoke（验证 `fp32` LoRA、新门禁字段与 `--resume` 连续计数恢复）。
- 正式 DPO v2 训练至少有 1 组候选同时满足：（1）通过 DPO v2 解耦门禁（`phys3` 严降、`clean` 安全、`dropout` 增量 `<= 0.002`、`gap_insertions <= 5`、`preference >= 0.55`）；（2）在 `val.jsonl` 上相对 SFT 冠军的 `physical_degraded_3` 改善幅度显著超越 v1（要求三场景字编辑净减至少 20 个，配对源句簇的 95% CER 差值区间上界小于零，另报 `raw_policy_accuracy` 相对同精度零步的变化）；（3）完成独立新进程 2,500 条读回发布并在 `test.jsonl` 与只读 `AliMeeting Eval` 上复验。

## 影响

- **对已有数据与评测的影响**：零改动；冻结的 `val.jsonl`、`test.jsonl`、`AliMeeting Eval` 与历史 Track 3B 数据集保持只读。
- **对已有训练配置与哈希的影响**：零漂移；全部历史 SFT / DPO / RL YAML 因未设置新增可选 `gate.*` 字段，`resolved["gate"]` 与语义 SHA-256 保持逐位一致，历史已发布 release 及其 `dpo_release.py` / `rl_release.py` 校验链不受任何影响。
- 每组实际见过的样本对次和独立源句数随评测保存。数据量和学习率的效果须由四组实验给出。
