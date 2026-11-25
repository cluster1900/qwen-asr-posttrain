# 附录：随手可查的工程师口袋手册

---

## 附录 A：全书核心数学公式速查表

| 主题 | 核心公式 | 关键变量与速记含义 |
| :--- | :--- | :--- |
| **GEMM 计算量** | $\text{FLOPs} = 2 M N K$ | $[M,K] \times [K,N]$ 矩阵乘总乘加次数 |
| **Roofline 性能上限** | $P_{\text{实测}} \le \min\!\big(P_{\text{peak}},\; \text{BW} \times \text{AI}\big)$ | $\text{AI} = \text{FLOPs}/\text{Bytes}$；A100 拐点 $\approx 156$ |
| **Safe Softmax** | $p_i = \dfrac{e^{z_i - m}}{\sum_j e^{z_j - m}},\; m = \max_j z_j$ | 减最大值防止 $e^{z_i}$ 上溢成 `inf` |
| **交叉熵梯度** | $\dfrac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = p_i - \mathbb{I}(i = y^*)$ | 正确类拉高，错误类按自信度压低 |
| **缩放点积注意力** | $\text{Attn}(Q,K,V) = \text{Softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}} + M\right)V$ | 除 $\sqrt{d_k}$ 防止点积方差过大导致梯度消失 |
| **RMSNorm** | $\text{RMSNorm}(\mathbf{x}) = \dfrac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$ | 省略减均值与 $\beta$，融合提速 ~15% |
| **SwiGLU MLP** | $\big(\text{SiLU}(\mathbf{x}\mathbf{W}_g) \odot \mathbf{x}\mathbf{W}_u\big)\mathbf{W}_d$ | 3 矩阵动态门控，$\text{SiLU}(z) = z \cdot \sigma(z)$ |
| **RoPE 相对性** | $(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$ | 点积仅依赖相对距离 $n-m$，外推稳定 |
| **全参训练静态显存** | $M_{\text{静}} \approx 16 P \text{ Bytes}$ ($2P$ 权 + $2P$ 梯 + $12P$ AdamW) | 含激活约 $18\sim 20 P\text{ GB}$（$P$ 单位十亿） |
| **推理 KV Cache** | $2 \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot S \cdot B \cdot \text{sizeof}(\text{dtype})$ | Qwen3-0.6B (GQA 8KV, BF16) = `112 KiB/token` |
| **LoRA 低秩更新** | $\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \dfrac{\alpha}{r} \mathbf{B}\mathbf{A}\mathbf{x}$ | $\mathbf{A}$ 随机初始化，$\mathbf{B} = \mathbf{0}$，FP32 主权重防下溢 |
| **DPO 损失函数** | $-\log \sigma\!\left(\beta\left[(\log \pi_\theta(y_w)-\log\pi_\theta(y_l)) - (\log\pi_{\text{ref}}(y_w)-\log\pi_{\text{ref}}(y_l))\right]\right)$ | 第 0 步必为 $\ln 2 \approx 0.693147$ |
| **STE 差异梯度掩码** | `diff_nll + (full_nll - diff_nll).detach()` | 前向全句对齐，反向仅在差异位点求导 |
| **GRPO 组内优势** | $A_i = R_i - \frac{1}{G}\sum_{j=1}^G R_j$ | 全组同分（$A \equiv 0$）整组含 KL 全归零 |
| **Schulman `k3` KL** | $k_3(d) = e^d - 1 - d,\; d = \log\pi_{\text{ref}} - \log\pi_\theta$ | 恒 $\ge 0$、低方差、期望严格无偏等于 KL |

---

## 附录 B：OOM、NaN、幻觉与指标不涨“全场景排错诊断表”

| 故障现象 | 最可能的 3 个根因 | 一步到位的排查与修复动作 |
| :--- | :--- | :--- |
| **`CUDA OutOfMemoryError`** | ① `micro_batch` 或单条超长样本爆激活；② 15 万词表 FP32 Logits 占满显存；③ 碎片过多 | 降 `micro_batch=1` 增 `grad_accum` → 开 `Gradient Checkpointing` → 硬截断超长样本 → 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → 换 Chunked CE / QLoRA |
| **Loss 瞬间变 `NaN` / `inf`** | ① 在老卡用了 FP16 未开 `GradScaler`；② 学习率过大引发梯度爆炸；③ 数据里混入全 0/损坏样本或全部 `-100` 标签（除以 0！） | 换 BF16（若卡支持）；检查 `max_grad_norm=1.0`；把 `lr` 降 5–10 倍并加 Warmup；断言每批有效监督 Token 数 `> 0` 且输入全有限（`torch.isfinite`） |
| **Loss 一动不动 / `grad` 全为 `None`** | ① LoRA + Gradient Checkpointing 忘了给 Embedding 输出加 `requires_grad_(True)` hook；② `lr=1e-6` 在 BF16 LoRA 遭遇 ULP 吞噬；③ 优化器漏传可训练参数 | 跑 1 步后打印所有可训练参数的 `param.grad.abs().sum()`；将 LoRA 权重转为 **`float32`** 主权重；确认 `use_reentrant=False` |
| **模型学会“空输出”或疯狂复读不停止** | ① 标签掩码把句末 EOS (`<\|im_end\|>`) 误涂成了 `-100`；② Padding Token 与 EOS 共用同一个 ID 导致真实 EOS 被掩掉；③ 学习率太大训坏停机概率 | 打印 1 条样本的 `labels` 尾部 ID，确保真实 EOS 仍在监督范围内；区分 `pad_id` 与 `eos_id` 掩码逻辑 |
| **SFT 训完原能力暴跌 / 标点消失** | ① 训练金标格式（如无标点）与基座预训练先验冲突；② 学习率过大或 epoch 过多导致灾难性遗忘 | 肉眼对比 20 条 Base 输出与训练金标；对齐标点/格式先验；调低学习率（`1e-5` 档）或只挂 Attention LoRA |
| **DPO 第 1 步 Loss 不等于 `0.693`** | 离线 `ref_logp` 与在线 `policy_logp` 的 Prompt 模板、精度、Token 归一化方式（sum vs mean）或父权重不一致 | 强制加 Step 0 校验：抽 20 条核对 `abs(policy_logp - ref_logp) < 1e-4` 才允许开训 |
| **DPO / RL 训出奇怪幻觉（如静音吐英文）** | ① 偏好对不等长或含无声学支撑的伪对；② 共享后缀 $t > t_0$ 上 $\log Z$ 不抵消产生有毒负梯度 | 过滤伪对与不等长对；启用 `mismatched_only` 直通梯度掩码；加轻权重父版本 Anchor 正则 |
| **验证集涨了 0.1% 但上线反而更差** | ① 数据按单条随机切分导致说话人/文档泄漏；② 反复按验证集挑参导致验证集过拟合；③ 样本量太小属随机抖动 | 严格按源实体分组零泄漏切分；跑配对簇 Bootstrap 95% CI（要求区间不跨 0）；设立永不参与调参的封存 Test + 真实 OOD 评测集 |

---

## 附录 C：中英核心术语辞典与延伸阅读

### 常用中英术语对照
- **Autograd**：自动求导引擎，基于动态计算图反向应用链式法则。
- **Arithmetic Intensity (AI)**：算术强度，浮点运算次数除以显存搬运字节数，单位 FLOPs/Byte。
- **BF16 / FP16 / TF32 / NF4**：半精度脑浮点（8 指数 7 尾数）/ 标准半精度（5 指数 10 尾数）/ 张量浮点 32 / 4-bit 正态浮点量化。
- **CER / WER**：字错误率（Character Error Rate，中文主指标）/ 词错误率（Word Error Rate）。
- **Compute-Bound / Memory-Bound**：算力受限（吃满 Tensor Core）/ 显存带宽受限（算力在等搬运）。
- **DPO (Direct Preference Optimization)**：直接偏好优化，无需独立 Reward Model 的离线对齐方法。
- **FlashAttention**：IO 感知的分块融合注意力算法，利用 SRAM 与 Online Softmax 将显存从 $O(N^2)$ 降至 $O(N)$。
- **GQA (Grouped-Query Attention)**：分组查询注意力，多个 Q 头共享 1 组 K/V 头，大幅削减 KV Cache 显存。
- **Gradient Checkpointing**：梯度检查点 / 激活重计算，以约 +30% 时间换取 60%–80% 激活显存节省。
- **GRPO (Group Relative Policy Optimization)**：组相对策略优化，用同题多候选组内均值替代 Critic 的轻量在线 RL。
- **HBM / SRAM**：高带宽显存（容量大、较慢）/ 片上静态随机存储器（容量小、极快）。
- **KL Divergence / `k3` Estimator**：散度，衡量策略偏离参考模型的程度；`k3` 为非负低方差无偏估计器。
- **KV Cache**：自回归解码阶段缓存历史 Token 的 Key/Value 矩阵，避免重复计算。
- **LoRA / QLoRA**：低秩适配（冻结基座 + 训练低秩旁路 $BA$）/ 4-bit 量化基座 + LoRA。
- **OOD (Out-of-Distribution)**：分布外数据，指模型训练时从未见过的真实新环境/新领域数据。
- **Prefill / Decode**：推理预填充阶段（并行吃 Prompt，算力受限）/ 逐字解码阶段（自回归吐词，带宽受限）。
- **QK-Norm**：Qwen3 采用的逐头 Q/K RMSNorm，彻底锁死深层长序列注意力数值稳定。
- **Roofline Model**：屋顶线性能模型，可视化诊断算子是卡带宽还是卡算力。
- **RoPE (Rotary Position Embedding)**：旋转位置编码，通过二维复数旋转将相对距离自然注入点积。
- **SFT (Supervised Fine-Tuning)**：有监督微调，用标准输入-输出对计算交叉熵损失。
- **Sim-to-Real Gap**：仿真到现实鸿沟，理想合成训练数据与真实复杂物理世界分布之间的差距。
- **SM / Warp**：流多处理器（GPU 独立计算车间）/ 线程束（32 线程绑定的 SIMT 最小调度单元）。
- **STE (Straight-Through Estimator)**：直通估计器，通过 `.detach()` 让前向值与反向梯度路径解耦的技巧。
- **ULP (Unit in the Last Place)**：浮点数在当前量级下相邻两个可表示值之间的最小间隔。

### 延伸学习推荐
1. **论文原典**：*Attention Is All You Need* (Vaswani et al., 2017)、*LoRA* (Hu et al., 2021)、*FlashAttention & FlashAttention-2* (Tri Dao, 2022–2023)、*Direct Preference Optimization* (Rafailov et al., 2023)、*DeepSeekMath / GRPO* (Shao et al., 2024)、*Qwen3 Technical Report & Qwen3-ASR* (Alibaba Qwen Team, 2025–2026)。
2. **动手实践工具**：PyTorch Profiler + Nsight Systems（实测 GPU Kernel 耗时）、OpenAI Triton Tutorials（手写融合算子）、HuggingFace `transformers` + `peft` + `trl`、vLLM（工业推理与 PagedAttention）。
3. **本仓库实证代码参考**：欢迎结合本项目的 `train/train_sft.py`、`train/train_dpo.py`、`train/train_rl.py`、`evaluation/math_expectation.py` 以及 `docs/qwen3-asr/06_risks_and_decisions.md` 对照阅读，亲眼见证理论如何落地为可复现的工程系统。
