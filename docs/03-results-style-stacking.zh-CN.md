> English: [03-results-style-stacking.md](03-results-style-stacking.md)

# 画风迁移与 adapter 叠加

又做了两个任务，因为单凭角色结果可以被「LoRA 恰好适合某一张脸」这种说法解释掉。
两者都从同一个底座模型、同一个 sampler、同样的 3 个 seed × 2 个 prompt 出发，
并报告相对于同一 seed、无 adapter 那张图的图像统计量。

```
dRGB  逐通道平均 |I − I_base|   （图像变了多少）
S     平均饱和度                （base 0.297）
edge  亮度通道平均 |∇|           （base 0.028）
```

`dRGB` 是控制变量。如果两个 adapter 让图像变化了同样的量，其中一个保住了细节而
另一个没有，那么这个差异是关于 adapter 的，不是关于强度的。

## 1. 画风迁移

数据：一个画风（`reserve1999`，76 张图）。与角色任务相同的两套配方
（matched lr 5e-4/1200、kohya 1e-4/cosine）。

| config | dRGB | S | V | edge |
|---|---|---|---|---|
| base | 0.000 | 0.297 | 0.501 | 0.028 |
| LoRA style（kohya） | 0.131 | 0.200 | 0.488 | 0.027 |
| **LoRA style（matched）** | **0.168** | **0.240** | 0.435 | **0.028** |
| GPart style（all-linear） | 0.146 | 0.233 | 0.483 | 0.020 |
| **GPart style（attn2）** | **0.172** | **0.155** | 0.444 | **0.017** |

这张表里掉出两件事。

**(a)「GPart 把画风推得更狠」同样是配方假象。** 在相同位移下 —— matched LoRA
0.168，GPart attn2 0.172 —— 图像变化是一样的。你在开箱 run 里看到的更大数字，
还是那个更大的 lr 带来的更大 ΔW。

**(b) 在相同位移下，GPart 把它花在了一层全局的「洗白」（global wash）上。**
饱和度掉到 0.155，边缘能量掉到 0.017，**相对 base 是高频细节损失 39%**。matched
LoRA 用同样的 0.168 dRGB，落在 S 0.240、edge 0.028 —— 边缘能量与 base 完全一致，
细节完整保留。

显而易见的反驳是：这个画风本身就平、就糊，GPart 只是把它复现得更好。并不是：
用同样 76 张图训的 kohya LoRA 打出 edge 0.027。细节损失是 GPart 这个 run 的属性，
不是这个画风的属性。

还要注意，GPart 自己的 all-linear run（edge 0.020）在**更少**的位移
（0.146 vs 0.172）下，比它的 attn2 run（0.017）保留了更多 —— 把同样的参数预算
摊到只有 attention 的 module 上，产生的是一种更全局、更少结构的改变。

## 2. 把画风 adapter 和角色 adapter 叠起来

组合正是满秩 partition 被寄予厚望的地方：GPart 的 `ΔW = P·diag(s)·θ` 对 `θ`
是线性的，所以两个在共享 partition 坐标里组合的 adapter 会精确相加；而两个 LoRA
在 factor space 里组合会产生交叉项（`B₁A₁ + B₂A₂` 不等于任何一个单独的 `BA`）。

我们测的是朴素加法那一种 —— 两个 adapter 都加载，ΔW 相加 —— 因为用户就是这么做的：

| config | 角色 score | dRGB | S | edge |
|---|---|---|---|---|
| base | 0.08 | 0.000 | 0.297 | 0.028 |
| LoRA 叠加（kohya 那一对） | **0.00** | — | — | — |
| **LoRA 叠加（matched 那一对）** | **0.57** | 0.206 | 0.249 | **0.029** |
| GPart attn2 画风 + attn2 角色 | 0.40 | 0.206 | 0.161 | **0.021** |
| GPart attn2 画风 + all-linear 角色 | 0.27 | 0.177 | 0.129 | 0.021 |

**(a)「LoRA 叠加会把 adapter 平均掉」是 kohya 的假象。** 开箱那一对塌到 0.00 ——
角色完全消失 —— 因为每个组成成分本身就已经处在约 1/33 的位移上。matched 那一对
得 0.57，配饰完好。

**(b)「GPart 是更好的组合底座」也不成立。** 在完全相同的 dRGB（0.206）下，叠加
结果精确复现了单 adapter 的模式：LoRA 保住 S 0.249 / edge 0.029，GPart 掉到
S 0.161 / edge 0.021。组合既没有挽救也没有放大这个差异。

叠加还跑了**位移对齐**（displacement-matched）的变体（`gp_stack_attn_m`、
`gp_stack_attn_fm`、`gp_stack_all_m`，在工作树里），这样对比就不必押在
「dRGB 就是那个正确的控制变量」上。那里的排序是一样的。它们的原始图像和日志没有
打包进 `results/`，因为被主张的只是排序 —— 如果你要逐行数字，可以从
`scripts/history/stacktest_2026-10-30.sh` 重新生成。

## 3. 叠加**没有**测到什么

GPart 公开发表的主张并不是「加法合并很容易」—— 而是**在共享坐标里**做组合，在
那些对矩阵空间非线性的运算下仍然保持精确：TIES 合并、与其他 adapter 的正交化、
符号冲突。在 factor space 里，这些运算会破坏或错误归属 LoRA 的分量。

上面这些在本仓库里一个都没实现。上面的表对两个方法都只是朴素的 `ΔW₁ + ΔW₂`，
而这恰好是唯一一个 LoRA 的交叉项不起作用的组合运算。所以：

- ✅ 我们可以说：朴素的加法叠加在这些任务上没有显示出 GPart 的优势。
- ❌ 我们不能说：GPart 不是更好的组合底座。那个假设在这里没有被检验，而它仍然
  是 GPart 的结构性优势最可能出现的地方。

见 `00-claims.zh-CN.md`「我们不主张」第 4 行。

## 4. 图像

| 图 | 内容 |
|---|---|
| `results/figures/horizontal_full_prompt_style.png` | 完整 prompt 下的画风各组 |
| `results/figures/horizontal_stacking_p0.png` | base / kohya 叠加 / matched 叠加 / 两个 GPart 叠加 |
