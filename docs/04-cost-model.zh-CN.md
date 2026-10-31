> English: [04-cost-model.md](04-cost-model.md)

# 成本模型

在本仓库里跑 GPart 的实测成本，以及为什么这个数字比读者从方法描述里预期的大。

**先读这段：** 下面所有内容描述的都是**本仓库的 `stateless` backend**，写它是为了
塞进 8 GB 显存。它不是关于已发表 GPart 的主张。上游实现会把 assignment 表实体化，
那份我们放不下；那边的数字会不一样。

## 1. 为什么要 `stateless` backend

GPart 的 ΔW 是 `P · diag(s) · θ`，其中 `P ∈ {0,1}^{N×d}` 把 `N` 个被覆盖的权重
各自分到 `d` 个组之一。为一个 700 module、2184 M 权重的 target set 实体化 `P`：

```
2184e6 × 23224320 × 1 bit ≈ 6.3 TB    (dense)
2184e6            × 4 bytes ≈ 8.7 GB  （每权重一个 int32 group id）
```

两者在 8 GB 里都放不下，还要和 SDXL UNet 共处。所以本仓库在运行时从一个 hash
重算组归属：`P` 是*隐式的*，由一个作用在扁平参数索引上的、确定性带种子的 mixer
定义，并在每一次 forward 和 backward 里重新生成。

这个选择就是训练慢的原因。

## 2. 实测每步成本

一个 optimizer step = forward + backward + optimizer，effective batch 2，
832×1216 等效 buckets，bf16，RTX 级 8 GB 卡。

| adapter | 覆盖权重 | s/step | 相对 LoRA |
|---|---|---|---|
| LoRA（all-linear, r8） | 2184.0 M | **4.85** | 1.0× |
| GPart attn2 | 545.6 M | **8.11** | 1.7× |
| GPart all-linear | 2184.0 M | **20.5** | **4.2×** |

两个结构性事实，都与设计直觉相反：

**(a) 成本跟着被覆盖的权重走，不跟 `d` 走。** 把 `d` 从 23.2 M 提到 50 M ——
可训练参数翻了一倍多，按理这是你最该预期主导成本的东西 —— 只贵 **+13%**。
贵的地方不是 `θ`，而是碰遍 target set 里的每个权重、重算它的 group id 再
scatter/accumulate。

**(b) 成本在 ΔW 的实体化，不在 hash。** 完整的 forward 必须造出一个与 target
矩阵同尺寸的 dense ΔW（2184 M fp32 ≈ 8.7 GB 的访存量），linear 层才能把它加进去。
hash 函数只产出 group id；dense 重建才是瓶颈。

## 3. int32 partition backend

起点是 int64 的 `splitmix64`，在这块硬件上它比必要的更慢，因为多条 torch 路径里
64 位整数运算要么是模拟的、要么没被向量化。改用 **int32** 上的 mixer
（murmur3 风格的 finalizer，`0x85EBCA6B` / `0xC2B2AE35`，gamma 用 `0x9E3779B1`）
能产出同类的均匀 partition，并跑在原生的 32 位整数通道上。

微基准：对全部 700 个 module 做一整趟 partition。

| backend | 时间 | |
|---|---|---|
| `splitmix64`（int64） | 2.16 s | |
| `int32`（murmur finalizer） | 0.48 s | −78% |

每步两趟（forward + backward）→ **约省 3.4 s/step**。

端到端，在 20.5 s/step 的 all-linear run 上：

| | s/step |
|---|---|
| `splitmix64` | 20.5 |
| `int32` | **~17.0** |
| 加速 | **~1.15–1.2×** |

这比我们工作笔记里早先的估计小得多 —— 那个估计假设 hash 是主要成本，并预测
20.5 → 5–8 s/step。并不是：去掉 hash 的 78% 之后，每一步还剩约 83%，因为 ΔW
的实体化一点没动。

**正确性。** 每个权重的 group id 必须由它的扁平索引唯一确定，这样 save/load 和
多 GPU 切分才能一致。int32 在 `2^32` 以内是精确的；本仓库最大的 target set 是
2184 M 权重 < 4.29 G，所以在这个仓库的每一种配置下，截断都是无损的。如果
`start_offset + numel > 2^32`，代码会直接报错，而不是静默回绕。

两个 backend 的 round-trip 都验证过逐位一致（save → load → 输出相同）。
int32 backend 的 `θ` 语义与 `splitmix64` 不同（不同的 partition 意味着不同的基），
所以 **checkpoint 不能在两个 backend 之间互换**。

## 4. 这个成本模型说明了什么，没说明什么

✅ 4.2× 这个数字**在这个 backend、这块硬件上**是真实的。如果你被限制在 8 GB 显存，
而 target set 是 all-linear，那么一步就是这个成本。

❌ 它**不**意味着「GPart 作为一种方法比 LoRA 慢 4 倍」。一个实体化 `P` 的实现是
拿内存换速度；上游代码就是那么做的，我们没有 benchmark 它。

❌ 它也不意味着 GPart 在*采样*时慢。加载一个训好的 adapter 和加载 LoRA 成本相同：
`θ` 向量是 23.2 M 参数，比 r16 LoRA 的 42.7 M 还小。

### 对想复现的人的后果

- 如果你在小卡上跑 all-linear GPart，按 **LoRA 墙钟时间的约 4×** 做预算。那个
  1200 步的 all-linear 角色 run 是几小时量级的任务，不是几分钟。
- 能接受 0.03 的分数差时，优先选 **attn2-only**（8.11 s/step）—— 它是本仓库里
  成本/收益更好的那个点。
- `int32` backend 值得打开，但它是补丁不是解药；它换来约 1.2×，代价是与
  `splitmix64` 的 checkpoint 不兼容。

## 5. 校准

`src/bench/` 可以在你自己的硬件上复现这些数字：

| | |
|---|---|
| `bench_d.py` | 每步时间随 `d` 与 target set 的变化 |
| `profile_train_step.py` | 一个真实训练步的分阶段拆解 |
| `param_budget.py` | 某个 target set 的 module 数与覆盖权重数 |

在引用这里的任何数字之前，先把它们跑一遍 —— 整数路径与浮点路径之间的比值非常
依赖平台，而 8 GB 这个约束决定了我们对 backend 的选择。
