# 第二篇：硬件与算子

# 第 3 章 GPU 硬件架构与性能优化全景

> **本章情绪定心丸**：你也许听过显卡参数表里一连串外星词——SM、Warp、Tensor Core、HBM2e、SRAM、TFLOPS、Roofline、CUDA Graph……别晕！这一章我们用**“米其林大厨 vs 万人工厂流水线”**的生活比喻，把一块价值十万元的 NVIDIA A100 拆得透明如水晶。懂了这一章，你以后看到模型跑得慢，30 秒就能判断到底该调 Batch Size、开算子融合、做量化，还是换显卡！

---

## 3.1 CPU 与 GPU 的本质区别：跑车 vs 一百节满载的高铁

先打破一个常见误解：**GPU 的单个核心其实比 CPU 慢得多、傻得多！**

- **CPU（比如你的笔记本芯片）**：像 **8 位顶级全能教授**。配了巨大的缓存、极其聪明的分支预测器，专门擅长处理复杂逻辑（`if/else` 嵌套、数据库索引、操作系统调度），追求**单件任务延迟极低（Low Latency）**。
- **GPU（比如 NVIDIA A100）**：像 **10,000 名动作整齐划一的小学生**。每个孩子逻辑能力简单，但把一万道乘法题分给一万个孩子**同时**写，一秒钟就能交卷！GPU 追求的是**整体吞吐量极大（High Throughput）**。

大模型前向和反向里 95% 以上的活儿是什么？就是把几百万个数字并行相乘再相加（矩阵乘法）。这种“无脑但海量重复”的工作，简直是为 GPU 量身定做的天堂。

---

## 3.2 钻进一块 A100 内部：SM、Warp、Tensor Core 与四级存储金字塔

拿当前最经典的训练卡 **NVIDIA A100 80GB SXM** 做解剖标本，打开它的金属散热盖，里面有两大子系统：**算力工厂（Compute）**与**存储金字塔（Memory Hierarchy）**。

![图 3-1 现代 GPU（以 NVIDIA A100 为例）内部架构与四级存储金字塔](assets/diagrams/05_gpu_architecture_mem.svg)

### 1. 计算侧三大核心名词（一听就懂）
- **SM（Streaming Multiprocessor，流多处理器）**：你可以把 GPU 想象成一栋拥有 **108 个独立车间**的大工厂（A100 有 108 个 SM；H100 有 132 个 SM；RTX 4090 有 128 个 SM）。每个车间自带工人、小黑板和工具箱，彼此并行干活。
- **Warp（线程束 = 32 个线程）**：GPU 调度干活的最小单位不是 1 个线程，而是 **32 个线程绑成一捆**（一个 Warp），它们奉行 **SIMT（单指令多线程）**：车间主任喊一声“加！”，32 个工人同时对各自手里的数据做加法。
  - **为什么代码里忌讳 `if/else`（Warp Divergence，线程束分化）？** 如果同一捆 32 个工人里，前 16 个要走 `if`、后 16 个要走 `else`，因为他们必须听同一个口令，只能前 16 人先干、后 16 人干瞪眼，接着再换后 16 人干、前 16 人干瞪眼——**瞬间浪费一半算力！**
- **CUDA Core vs Tensor Core**：
  - **CUDA Core** 是普通计算器：1 个时钟周期算 1 次标量乘加（$a \times b + c$）。A100 全卡 FP32 算力只有 **19.5 TFLOPS**。
  - **Tensor Core** 是专为 AI 打造的“矩阵外挂”：1 条硬件指令直接吃进两块小矩阵（如 $16 \times 8$ 乘 $8 \times 16$），一个周期吐出整块矩阵乘加结果！A100 开启 BF16/FP16 后，Tensor Core 算力直接飙升至 **312 TFLOPS（暴涨 16 倍！）**；H100 的 FP8 更是突破 **1,979 TFLOPS**！

### 2. 存储侧四级金字塔：算力免费，搬运天价！
再看图 3-1 右半边的存储金字塔，这是理解**一切 GPU 优化（包括 FlashAttention、算子融合、量化）的第一把钥匙**：

| 存储层级 | 相当于厨房里的 | A100 80GB 容量 | 读写带宽 | 延迟（周期） |
| :--- | :--- | :---: | :---: | :---: |
| **寄存器 (Registers)** | 厨师双手正握着的勺子 | ~27 MB（每 SM 256 KB） | > 100 TB/s | **1 cycle** |
| **SRAM / Shared Memory (L1)** | 灶台旁边的切菜案板 | ~20 MB（每 SM 192 KB） | **~19 TB/s** | ~20–30 cycles |
| **L2 Cache** | 厨房里的备菜冷柜 | 40 MB（全卡共享） | ~4–5 TB/s | ~200 cycles |
| **显存 HBM2e（Global Memory）** | 离厨房 100 米远的大冷库 | **80 GB** | **2.0 TB/s** | **~400–600 cycles** |

看到那个刺眼的差距了吗？

- **SRAM（片上共享内存）比 HBM（显存）快了近 10 倍，延迟低了 20 倍，但容量小了 4,000 倍！**
- 如果你的代码傻乎乎地算一步就把大矩阵写回 HBM（大冷库），下一步再重新从 HBM 搬回来算，那么 312 TFLOPS 的超级 Tensor Core 厨师有 **95% 的时间都在翘着二郎腿等搬运工**！

---

## 3.3 Roofline（屋顶线）模型：用一道初中除法诊断任何性能瓶颈

怎么定量判断一个算子到底是在“等搬运工（**Memory-Bound 访存受限**）”还是在“厨师忙冒烟（**Compute-Bound 算力受限**）”？

计算机体系结构大师 Samuel Williams 提出了一个极其优雅的模型——**Roofline（屋顶线模型）**。你只需要算一个比值，叫做**算术强度（Arithmetic Intensity, AI）**：

$$\text{AI} = \frac{\text{总浮点运算量（FLOPs）}}{\text{总 HBM 显存读写字节数（Bytes）}} \quad \left(\text{单位：FLOPs / Byte}\right)$$

再算一下显卡自身的**硬件算存比拐点（Ridge Point）**——拿 A100 80GB (BF16) 算：

$$\text{Ridge Point}_{\text{A100}} = \frac{\text{峰值算力}}{\text{显存带宽}} = \frac{312 \times 10^{12}\text{ FLOPs/s}}{2.0 \times 10^{12}\text{ Bytes/s}} = \mathbf{156\text{ FLOPs / Byte}}$$

![图 3-2 Roofline 性能屋顶线模型：一眼判别算子是卡带宽还是卡算力](assets/diagrams/06_roofline_model.svg)

这句话请背下来，它值十万年薪的调优直觉：
> **在 A100 上，每从 HBM 显存搬运 1 个字节的数据进来，如果你对它做的计算少于 156 次浮点运算，你的 Tensor Core 就一定在摸鱼（Memory-Bound）；超过 156 次，才真正开始吃满算力（Compute-Bound）！**

### 两个立刻让你拍案叫绝的实战手算！

::: example 手算 1：为什么逐元素操作（ReLU / 加法 / RMSNorm）永远卡带宽？
假设我们要对一个包含 $N$ 个 BF16 元素的向量做 `y = x + 1`：

- 计算量：$N$ 次加法（$N$ FLOPs）；
- 显存读写：读 `x` 占 $2N$ 字节 + 写 `y` 占 $2N$ 字节 = $4N$ 字节；
- 算术强度：$\text{AI} = \frac{N}{4N} = \mathbf{0.25\text{ FLOPs/Byte}}$！

对比 A100 的拐点 `156`：**差了 624 倍！** 这意味着你跑 `ReLU` 时，A100 实际只能发挥出 $2.0\text{ TB/s} \times 0.25 = 0.5\text{ TFLOPS}$，99.8% 的算力全闲着！
**怎么优化？** **算子融合（Kernel Fusion）**！把连续的 `Residual Add + RMSNorm + Dropout` 合并成 1 个 GPU Kernel，只从 HBM 读 1 次、在寄存器里一口气全算完、只写回 1 次，直接提速 3–5 倍！
:::

::: example 手算 2：为什么大模型推理“单条生成（Batch=1）”永远吃不满算力，而“加 Batch”就能飞起来？
考虑一个常见的线性层矩阵乘：输入 $\mathbf{X}$ 形状 `[B, K]`，权重 $\mathbf{W}$ 形状 `[K, N]`（设 $K = N = 4096$，BF16 每元素 2 字节）：

- 计算量：$2 B K N$ FLOPs；
- 显存读写（主要是把巨大的权重 $\mathbf{W}$ 从 HBM 搬出来）：$(BK + KN + BN) \times 2$ 字节。

代入 **Batch = 1（推理逐字生成 Decode）**：
$$\text{AI}_{B=1} = \frac{2 \times 1 \times 4096 \times 4096}{2 \times (4096 + 4096^2 + 4096)} \approx \mathbf{1.0\text{ FLOPs/Byte}} \ll 156$$
震撼吧！每吐出 1 个汉字，GPU 都得把几十 GB 的全部模型权重从 HBM 完完整整搬运一遍，却只做了 1 个向量乘积！所以**推理 Decode 阶段的速度上限 ≈ `显存带宽 ÷ 模型权重总字节数`**！

- 例如 7B 模型 BF16 占 14 GB，在带宽 2.0 TB/s 的 A100 上，单条极限吐字速度理论天花板就是 $2000 \div 14 \approx 142\text{ tokens/s}$；如果把权重做 **INT4 量化**（缩到 3.5 GB），搬运量缩减 4 倍，吐字速度立刻提速近 3–4 倍！

再代入 **Batch = 256（训练或并发 Prefill）**：
$$\text{AI}_{B=256} = \frac{2 \times 256 \times 4096^2}{2 \times (256 \times 4096 + 4096^2 + 256 \times 4096)} = \frac{256}{1 + \frac{512}{4096}} \approx \mathbf{227.6\text{ FLOPs/Byte}} > 156$$
同一份权重搬进芯片一次，同时喂给 256 个 Token 复用，瞬间越过 Roofline 拐点，**312 TFLOPS 满血咆哮！** 这就是为什么推理服务（如 vLLM）拼命做 Continuous Batching（连续批处理），而训练里增大有效 Token 吞吐能大幅提升 MFU（模型算力利用率）。
:::

---

## 3.4 工程师必知的六大 GPU 性能优化武器

理解了硬件和 Roofline，下面这六项工业级优化技术对你来说就不再是黑箱，而是一套清晰的组合拳：

1. **合并访存（Memory Coalescing）与 8/16/64 维度对齐**
   - HBM 每次取货不是按 1 字节零买，而是整箱搬运（128 字节 Cache Line）。如果一个 Warp 里的 32 个线程恰好读取连续地址（如 `a[0..31]`），1 次内存事务搞定；如果跨步乱读（非连续张量），就要发 32 次事务，带宽暴跌 32 倍！
   - Tensor Core 的硬件矩阵指令要求维度是 **8（FP16/BF16）或 16（INT8/FP8）的倍数**（最好是 64/128 的倍数）。这就是为什么你会看到 Qwen3 的词表大小不是随意的 151,851，而是特意补齐到 **`151,936 = 1187 × 128`**！
2. **算子融合（Operator Fusion）与 `torch.compile`**
   - PyTorch 2.0 引入的 `torch.compile(model)`（底层是 TorchInductor 自动生成 Triton 代码），会自动扫描计算图里相邻的逐元素算子（如 `SiLU + 乘法 + RMSNorm`），把它们融合成单个 GPU Kernel，省掉中间写回 HBM 的开销，**一行代码白嫖 15%–40% 训练加速**！
3. **减少 Kernel Launch Overhead 与 CUDA Graph**
   - CPU 每给 GPU 下达一次小算子发射命令，大概需要 5–10 微秒。像 `Qwen3-ASR-0.6B` 这样的小模型，在 `batch_size=1` 做自回归解码时，GPU 算完一层只需几微秒，结果 **CPU 派活的速度还没 GPU 干活快**！
   - **CUDA Graph** 把一整串几百个小算子的发射轨迹录成一张“回放磁带”，下次只需 CPU 喊 1 声 `replay()`，GPU 自己一口气跑完 28 层，彻底消灭派活延迟！
4. **DataLoader 异步预取与 Pin Memory**
   - 别让几万块的 GPU 等 CPU 读硬盘解码 WAV 音频！开启 `pin_memory=True`（锁页内存，让 GPU 通过 DMA 直接高速搬运）、`num_workers > 0`、`non_blocking=True` 异步传输，把数据加载时间彻底藏进 GPU 计算时间背后。
   - *注意*：在要求 100% 确定性复现或单线程 Colab 小内存环境下（如本项目 `num_workers: 0`），可以先将音频预解码或打包到本地高速 SSD，再按需开启多进程。
5. **激活重计算（Gradient Checkpointing）与混合精度（BF16 + FP32 Master）**
   - 第 2 章已详述：既省显存又能放大 `micro_batch_size` 提高算术强度 AI，一举两得。
6. **多卡分布式：DDP vs ZeRO / FSDP 速览**
   - **DDP（数据并行）**：每张卡存一份完整模型、梯度、优化器，各自算不同数据，反向传播时边算边用 All-Reduce 同步梯度（通信与计算重叠）。缺点：单卡放不下模型时无能为力。
   - **DeepSpeed ZeRO / PyTorch FSDP（完全分片数据并行）**：把第 5 章要讲的显存“四大吞金兽”按卡数 $N$ 切分到多张卡上！
     - **ZeRO-1**：切分优化器状态（省 4× 显存，通信零额外开销）；
     - **ZeRO-2**：再切分梯度（省 8× 显存，几乎零额外通信）；
     - **ZeRO-3 / FSDP**：连模型权重也切分，算到哪一层再临时 All-Gather 拼回来，算完立刻释放（$N$ 张卡显存近似降为 $1/N$，但吃卡间互联带宽，**强烈依赖 NVLink**）。

```python
# 5 行代码亲手测一下你手里 GPU 的真实 BF16 TFLOPS 算力（Benchmark 脚本）
import time, torch

def benchmark_gemm(n: int = 4096, iters: int = 50):
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    a = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    for _ in range(10):  # 预热，避免把编译/初始化算进去
        _ = a @ b
    torch.cuda.synchronize()  # 极其重要！GPU 是异步执行的，测时间必须先同步！
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tflops = (2 * n ** 3) / dt / 1e12
    print(f"GPU={torch.cuda.get_device_name(0)} | {n}x{n} BF16 GEMM | {dt*1e3:.2f} ms | {tflops:.1f} TFLOPS")
```

::: note 本章小结
1. GPU 优化的第一性原理：**算力极便宜，HBM 搬运极昂贵**。
2. Roofline 公式：$\text{实测性能} \le \min(P_{\text{peak}},\ \text{BW} \times \text{AI})$。A100 拐点约 `156 FLOPs/Byte`。
3. 记住两大黄金场景：**训练/Prefill** 通常是算力受限（靠 BF16/FP8 Tensor Core 加速）；**单条推理 Decode 与逐元素算子** 永远是显存带宽受限（靠算子融合、Batch 合并、权重 INT4/INT8 量化加速）。
4. 测 GPU 耗时永远不要忘记 `torch.cuda.synchronize()`，否则你测到的只是 CPU 派发任务的时间！
:::
