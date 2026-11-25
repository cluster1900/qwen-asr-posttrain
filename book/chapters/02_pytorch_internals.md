# 第 2 章 PyTorch 底层深度解剖：内存、计算图与浮点精度陷阱

> **本章情绪定心丸**：很多开发者写了一两年 PyTorch，其实一直在“凭体感编程”——`.view()` 报错就乱加 `.contiguous()`，显存爆了就到处塞 `torch.cuda.empty_cache()`，遇到 `bf16` 和 `fp16` 全靠复制粘贴。读完这一章，你会真正看见 PyTorch 在显存里到底干了什么，从“调包侠”蜕变成能一眼看穿报错根因的底层玩家。

---

## 2.1 Tensor 的真面目：一维连续内存 + 视图说明书

在 Python 里，你创建一个形状为 `(2, 3)` 的矩阵 `A`，直觉上以为显存里真的画了一个 2 行 3 列的格子。**错！** 计算机的内存（RAM）和显卡的显存（HBM）物理上永远只是**一根长长的一维数组**。

PyTorch 的一个 `Tensor` 对象在底层由两部分组成：

1. **数据区（`UntypedStorage`）**：一块真正躺在内存或显存里的一维连续字节。
2. **元信息头（Metadata，类似“阅读说明书”）**：包含 `shape`（形状）、`stride`（步长）、`storage_offset`（起始偏移量）、`dtype`（数据类型）和 `device`（CPU 还是 CUDA）。

![图 2-1 PyTorch Tensor 底层内存真相：同一块一维 Storage 上的两个零拷贝视图](assets/diagrams/02_tensor_memory_stride.svg)

看懂图 2-1，你就瞬间解开了 PyTorch 面试和工程里最高频的三个谜题：

1. **为什么 `transpose()`、`permute()`、`slice` 瞬间完成且不增加显存？**
   因为它们压根没搬动显存里的任何数据！只是新建了一个轻量的元信息头，把 `stride` 从 `(3, 1)` 改成了 `(1, 3)`。这种操作叫**零拷贝视图（Zero-Copy View）**。

2. **为什么 `A.transpose(0, 1).view(-1)` 会直接报错 `RuntimeError: view size is not compatible with input tensor's size and stride`？**
   因为 `.view()` 严格要求张量在物理内存上必须是**连续排列（contiguous）**的（即行优先满足 `stride[i] = stride[i+1] * shape[i+1]`）。转置后的 `B` 逻辑顺序是 `[0, 3, 1, 4, 2, 5]`，但物理显存里躺着的还是 `[0, 1, 2, 3, 4, 5]`，没法只改说明书就拉成一维！
   - **解决办法**：调用 `B.contiguous().view(-1)`（真正开辟新显存按新顺序拷贝一遍），或者直接用 `B.reshape(-1)`（连续时零拷贝，不连续时自动拷贝）。
3. **为什么原地修改（In-place）切片有时会悄悄把原张量改坏？**
   因为 `B` 和 `A` 指向同一块 `Storage`！执行 `B[0, 1] = 99` 时，`A[1, 0]` 也会变成 `99`。需要真正独立副本时，务必显式写 `.clone()`。

```python
import torch

a = torch.arange(6).view(2, 3)       # shape=(2,3), stride=(3,1), contiguous=True
b = a.transpose(0, 1)                # shape=(3,2), stride=(1,3), contiguous=False
print("共享底层内存:", a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr())  # True!

# 广播机制（Broadcasting）也是靠 stride=0 实现的零拷贝魔法！
bias = torch.tensor([10, 20, 30])    # shape=(3,), stride=(1,)
expanded = bias.expand(2, 3)         # shape=(2,3), stride=(0, 1) —— 行步长为 0，完全不复制内存！
print("expand stride:", expanded.stride())
```

---

## 2.2 Autograd 自动求导引擎：动态计算图、Detach 与原地修改雷区

第 1 章我们手算了反向传播，PyTorch 的 **Autograd** 就是帮你全自动维护这张“导数追责图”的引擎。你需要掌握它的四个关键机制：

### 1. 叶子张量（Leaf Tensor）与 `grad_fn`
- 用户直接创建且 `requires_grad=True` 的张量（比如模型的所有 `nn.Parameter` 权重）叫**叶子节点**。只有叶子节点在 `.backward()` 后默认保留 `.grad`（中间节点的梯度算完就立即释放以省显存，除非显式调用 `t.retain_grad()`）。
- 每一个由运算产生的新张量，身上都挂着一个 `grad_fn` 指针（例如 `<AddBackward0>`、`<MmBackward0>`），记录着“我是谁算出来的，反向该怎么求导”，层层回指构成**有向无环计算图（DAG）**。

### 2. 三大“切断求导”工具的区别（新手必考）
| 写法 | 是否记录计算图 | 是否省显存 | 典型用途 |
| :--- | :---: | :---: | :--- |
| `with torch.no_grad():` | 否 | **极省**（不存中间激活） | 验证集评估、推理生成、参考模型前向、更新权重 |
| `with torch.inference_mode():` | 否（且关闭版本计数） | **最省最快** | 纯推理部署（但产出的张量绝不能再送回训练图！） |
| `y = x.detach()` | 切断 `y` 往回传梯度 | 共享数据不拷贝 | **直通估计器（STE）**、把 Loss/Logprob 当常数记日志 |

::: tip 高级技巧：一行 `.detach()` 实现“前向保留完整值，反向只传局部梯度”
你在第 10 章和真实代码 `train/train_dpo.py` 中会见到这句堪称艺术的代码：
```python
return diff_nll + (full_nll - diff_nll).detach()
```
算算看：

- **前向求值时**：`diff_nll + full_nll - diff_nll = full_nll`（一分不多，一分不少！）；
- **反向求导时**：括号里的 `.detach()` 导数为 `0`，所以整个表达式的梯度 **100% 等于 `∇diff_nll`**！
这项技巧叫**直通估计器（Straight-Through Estimator, STE）**，在量化感知训练（QAT）、VQ-VAE、以及精准屏蔽 DPO 共享后缀毒性梯度中威力无穷。
:::

### 3. 为什么反向传播常报 `one of the variables needed for gradient computation has been modified by an inplace operation`？
记住第 1 章的发现：算 $\frac{\partial L}{\partial w_2}$ 时需要前向存下来的中间激活 $h$。PyTorch 为每个张量维护了一个隐形的版本计数器 `_version`。如果在前向计算之后、`.backward()` 之前，你用原地操作（名字带下划线如 `x.add_(1)`、`param.copy_(...)` 或 `x += 1`）改掉了前向用过的张量，`_version` 就会加 1，Autograd 发现“物证被篡改了”，立刻报错拒绝求导！

> **真实项目踩坑案例（见 `train/train_rl.py`）**：在在线强化学习中，为了节省一份 1.56GB 的参考模型显存，我们把冻结的参考 LoRA 权重临时 `param.copy_()` 换进当前模型里算 `log π_ref`，算完再换回策略权重 `π_θ`。早期如果**先算策略 `π_θ` 前向（建好计算图）、再原地 `copy_()` 换入参考权重、最后 `backward()`**，就会立刻触发 `_version` 冲突报错！**正确顺序**是：先在 `torch.no_grad()` 下换入参考权重算出 `ref_logp` 并恢复策略权重，**然后**再对策略权重做前向和反传（参见第 11 章图 11-1）。

---

## 2.3 Gradient Checkpointing（梯度检查点）与 LoRA 断链钩子

训练长序列大模型时，28 层甚至 80 层的中间激活值往往比模型权重本身大好几倍。PyTorch 提供了神器 **Gradient Checkpointing（激活重计算）**——核心哲学是**“用时间换空间”**：

![图 2-2 标准反向传播 vs Gradient Checkpointing 显存与时间权衡](assets/diagrams/03_autograd_graph_ckpt.svg)

- **原理**：前向跑过每一层 Transformer Block 时，只保留进入该层的“入口输入张量”，层内部算出的 Q/K/V、Attention 矩阵、SwiGLU 3072 维中间结果**统统扔掉**！等到反向传播一路退回到第 $k$ 层时，再临时从入口张量重跑一次第 $k$ 层的前向，拿到局部激活立刻求导，求完立刻释放。
- **代价与收益**：计算时间增加约 **25%–33%**（相当于多跑一次前向，因为反向本身计算量≈2×前向，所以总耗时从 $3\times$ 变成 $4\times$），但激活显存直接暴降 **60%–80%**（从 $O(L)$ 降为 $O(\sqrt{L})$ 甚至按层边界计的极小常数）。

::: warning 实战必踩大坑：LoRA + Gradient Checkpointing 为什么会出现 `None` 梯度或无梯度警告？
看懂这个坑，你能少熬三个通宵！
当你用 LoRA 训练模型时，词表嵌入层 `embed_tokens` 是**冻结的**（`requires_grad=False`）。于是第 0 层 Transformer Block 收到的输入 `hidden_states` 的 `requires_grad` 也是 `False`！
而 PyTorch 的 `checkpoint(..., use_reentrant=False)` 在判断是否需要把该段纳入反向重计算时，会检查输入张量是否需要梯度。如果输入 `requires_grad=False`，旧版 `use_reentrant=True` 会直接把整条链路跳过，导致后面挂在 Attention 上的 LoRA 参数**压根收不到梯度**！

**标准工业解法（直接摘自本项目 `train/train_sft.py` 第 313–328 行）**：
```python
def enable_training_mode(model) -> None:
    model.thinker.config.use_cache = False  # 训练必须关 KV Cache，否则与 Checkpointing 冲突
    embeddings = model.thinker.get_input_embeddings()

    def mark_grad(_module, _inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)     # 强制给入口激活打上“需要梯度”标记！
        return output

    embeddings.register_forward_hook(mark_grad)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```
顺便记住：现代 PyTorch（2.0+）永远优先传 `use_reentrant=False`，它对多输入、LoRA 和分布式 DDP/FSDP 兼容性远好于旧版 `use_reentrant=True`。
:::

---

## 2.4 混合精度与浮点二进制真相：为什么 BF16 会“吃掉”你的小学习率？

很多同学只背过口诀：“A100/H100/4090 用 BF16，T4/V100 用 FP16”。但如果不看懂浮点数在计算机里的二进制长相，迟早会栽在精度陷阱里。

计算机用科学计数法存小数：$(-1)^{\text{符号}} \times (1.\text{尾数})_2 \times 2^{\text{指数} - \text{偏移}}$。

- **指数位（Exponent）**决定能表示的**最大/最小范围**（动态范围）；
- **尾数位（Mantissa）**决定数字有多**精细**（相邻两个可表示数的间隔，叫 **1 ULP**, Unit in the Last Place）。

![图 2-3 现代大模型浮点格式位结构对比与 BF16 小步长 ULP 吞噬陷阱](assets/diagrams/04_float_formats_ulp.svg)

对比图 2-3，四件事豁然开朗：

1. **FP16 为什么爱报 `NaN`？** 它只有 5 位指数，最大只能表示 **`65,504`**！反向传播时梯度稍大就上溢成 `inf`，稍小（小于 $6 \times 10^{-5}$）就下溢变成 `0`。所以用 FP16 必须配一个 `torch.cuda.amp.GradScaler`（先放大 Loss 几千倍算完梯度再缩回来）。
2. **BF16（Brain Float 16）为什么是现代训练之王？** Google Brain 把 FP32 粗暴地“拦腰砍掉最后 16 位尾数”，完整保留了 **8 位指数**！因此它的表示范围和 FP32 一模一样（最大 $\pm 3.4 \times 10^{38}$），**彻底告别上溢，不需要 `GradScaler`**！
3. **TF32（TensorFloat-32）是什么？** A100 起引入的硬件偷懒加速模式：张量存显存时依然是 FP32，送进 Tensor Core 做矩阵乘时自动截断成 10 位尾数（像 FP16 一样快，又不溢出）。注意：在需要逐位可复现的严谨对比实验（如本项目的 `train_sft.py` / `train_dpo.py`）中，通常会显式设置 `allow_tf32 = False`，防止不同显卡间出现微小舍入漂移。
4. **BF16 的阿喀琉斯之踵——7 位尾数的 ULP 吞噬陷阱！**
   7 位二进制尾数意味着精度只有 $2^{-7} \approx \frac{1}{128} \approx 0.78\%$！举个震撼的例子：在 BF16 里，`1.0` 下一个能表示的数不是 `1.000001`，而是 **`1.0078125`**（间隔 `0.0078125`）！如果你对 `1.0` 执行 `1.0 + 0.001`，**舍入后还是 `1.0`，你的更新被吞得一干二净！**

::: warning 本项目 2026-11-25 真实 A100 事故复盘：9 MB 显存救活整个 RL 实验
在 `Qwen3-ASR-0.6B` 经历完 SFT 和 DPO 后，我们进入在线 RL 阶段，学习率设为保守的 `lr = 1e-6`。结果跑完一看日志，损失和指标几乎像卡死一样诡异抖动！
把 LoRA 权重抽出来做统计，真相令人脊背发凉：

- `lora_A` 权重中位幅值是 `1.45e-2`，在这个量级下 BF16 的相邻间距 **ULP = `1.22e-4`**；
- 优化器单步更新量 $\Delta w \approx 1 \times 10^{-6}$，远小于半个 ULP（`6.1e-5`）！
- 实测结果：**99.7% 的 `lora_A` 参数与 78% 的 `lora_B` 参数在更新后被四舍五入回原值，完全零更新**，剩下的 0.3% 偶尔跳过一格 ULP 反而制造了巨大的 28 层离散噪声！

**修复方案只需 3 行代码**：基座 7.82 亿参数照常保持 `bfloat16`（省显存、跑 Tensor Core），**仅把 458 万个可训练 LoRA 权重保持为 `float32` 主权重（Master Weights）**！多占的显存只有 $4,587,520 \times 2\text{ Bytes} \approx \mathbf{9.17\text{ MB}}$，却 100% 根治了小学习率下溢！
```python
# 见本项目 train/train_sft.py 第 298-301 行：混合精度 LoRA 黄金法则
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.float()  # LoRA 参数永远用 FP32 主权重保存与更新！
```
:::

---

## 2.5 优化器 AdamW 与余弦学习率调度：别踩 `max_steps` 坑！

为什么现在几乎所有大模型都用 **AdamW**，而不是第 1 章里的朴素 SGD？
想象你在漆黑的峡谷里下山：峡谷两边峭壁很陡（梯度极大），谷底小路很缓（梯度极小）。

- **朴素 SGD** 用同一个步长：步长大了就在两侧峭壁间来回撞墙震荡；步长小了在谷底走得像蜗牛。
- **AdamW** 给**每一个参数**都配了两个私家仪表盘：
  1. **一阶动量 $m_t$**（历史梯度的指数滑动平均，$\beta_1 = 0.9$）：类似下坡惯性，滤掉单步小批次噪声；
  2. **二阶动量 $v_t$**（历史梯度平方的滑动平均，$\beta_2 = 0.95$ 或 $0.999$）：测量这个方向平时“抖得有多剧烈”。

更新公式简洁而有力：

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
\end{aligned}$$

你看 $\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$：梯度大的峭壁方向除以一个大数（刹车），梯度小的平缓方向除以一个小数（加速），**让每个参数自动拥有最适合自己的有效步长**！而末尾单独剥离出来的 $-\eta_t \lambda \theta_{t-1}$ 就是 **Decoupled Weight Decay（解耦权重衰减）**，轻轻把权重往 0 拉防过拟合（这就是 Adam**W** 名字里 `W` 的由来）。

### 为什么需要 Warmup + Cosine Decay？以及“改 `max_steps` 毁掉曲线”的经典坑
- **Warmup（预热）**：刚开始训练时，$v_t$ 还没攒够统计信息、LoRA 刚起步方向很抖，直接上最大学习率很容易一脚把预训练基座踹歪。所以前 `warmup_steps` 步让学习率从 `0` 线性爬升到峰值 $\eta_{\max}$。
- **Cosine Decay（余弦退火）**：随后像余弦曲线从 $\cos(0)=1$ 平滑落到 $\cos(\pi)=-1$，把学习率优雅地降到 $\eta_{\min} = \text{min\_lr\_ratio} \cdot \eta_{\max}$（通常是峰值的 10%），让参数在终点稳稳落进平坦的极小值盆地。

::: warning 真实踩坑：`max_steps` 与 `scheduler_max_steps` 必须解耦！
在本项目 DPO 实验中，100 步余弦计划在第 10、20、30、40 步全部通过门禁，却在尾步第 41 步翻车。最初我们图省事直接把 `max_steps: 100` 改成 `max_steps: 40` 想“早停在 40 步”，结果第 30 步居然失败了！
原因一算就懂：把 `max_steps` 缩成 40 后，余弦退火的分母从 `90` 被压成了 `30`，导致第 30 步学习率提前暴跌到 `1.625e-6`（仅为原计划同期 `4.47e-6` 的 36%），优化轨迹完全变了！
**工程最佳实践**（见 `train/train_sft.py` 的 `HorizonScheduler`）：永远把**“停步步数 `train.max_steps`”**和**“余弦衰减总周期 `optim.scheduler_max_steps`”**拆成两个独立参数！这样你想在第 40 步早停，前 40 步的学习率曲线和原 100 步实验可以做到**逐位一致**。
:::

::: note 本章小结
1. `Tensor` = 底层一维 `Storage` + 视图元信息（`shape/stride/offset`），懂 `stride` 就能分清零拷贝与显存搬运。
2. 反向传播靠 `grad_fn` 链条，切忌在反传前对已入图张量做原地修改；开启 Gradient Checkpointing + LoRA 时，别忘了给 Embedding 输出加 `requires_grad_(True)` 的 forward hook。
3. BF16 虽不会溢出，但尾数仅 7 位；小学习率后训练时，务必保持可训练 LoRA 权重为 FP32 主权重（仅多几 MB 显存，收益巨大）。
4. 永远把“训练停步”与“余弦周期”解耦，并且断点续训时要连同 `optimizer`、`scheduler` 和 `RNG` 状态一起完整保存恢复。
:::
