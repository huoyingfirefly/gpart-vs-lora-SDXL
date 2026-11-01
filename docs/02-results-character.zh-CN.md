> English: [02-results-character.md](02-results-character.md)

# 角色任务

一个角色（`aima`，21 张原图 × 3 个 caption 变体），NoobAI-XL v1.1，只训 UNet，
prompt 用 `p0`（干净场景，只有 trigger token），WD14 tag 分数取 10 个 tag、
阈值 0.35。除特别注明外，每一行都是 `n = 6` 个 seed。

`p0` 是有区分度的 prompt。在 `p1`（完整描述，发色、瞳色、每件配饰都写明了）下，
底座模型的得分几乎和任何 adapter 一样好 —— 因为已经没有东西留给 adapter 去提供了。
在 `p1` 上测到的任何东西，都是关于「跟不跟文本」的陈述，而不是关于 adapter 的。

## 1. 开箱结果，以及为什么它不算数

| config | 配方 | modules | α | ‖ΔW‖ | 逐权重 RMS | score |
|---|---|---|---|---|---|---|
| base | — | — | — | 0 | 0 | 0.08 |
| LoRA（kohya，随仓库发布） | lr 1e-4, cosine-restarts, 10 ep | all-linear r16 | 1.0 | 8.20 | 0.000103 | **0.05** |
| GPart（随仓库发布） | lr 5e-4, constant, 1200 st | attn2 | 1.0 | 78.29 | 0.003355 | **0.43** |

按字面读，这是 33× 的位移差和 0.43 vs 0.05 的得分差，也就是「GPart 学会了这个
角色，LoRA 什么都没学到」。它并不是看上去那样：

- kohya adapter 的逐权重位移只有 GPart 的 **1/33**。那是一个训练不足的 ΔW，不是
  训练充分、强度 1.0 的 ΔW。
- 它**低于 base**（0.05 vs 0.08）。一个输出与「没有 adapter」无法区分的 adapter，
  不能当作关于 LoRA 的证据。
- 把 kohya LoRA 的强度调高能找回一部分（最优点在 scale 2.0，到 0.27 之后饱和），
  这是从另一个方向得到的同一个结论：ΔW 就是太小了。

随仓库发布的 GPart 与随仓库发布的 LoRA，在 lr（5×）、schedule、步数、batch size
和 rank 上都不同 —— 即 `01-protocol.zh-CN.md` §0 里五项混淆因子中的四项。

## 2. 覆盖不是解释

下一个显而易见的假设是：GPart 只碰 `attn2`（280 个 module，545.6 M 权重），而
LoRA 碰全部 700 个 linear module（2184 M 权重）。把 GPart 提到同样的覆盖 —— 同一
配方、同一个 `d`、只改 `--targets`：

| config | modules | 覆盖 | α | score | Δ |
|---|---|---|---|---|---|
| GPart attn2 | 280 | 545.6 M | 1.0 | 0.43 | — |
| GPart all-linear | 700 | 2184.0 M | 1.0 | **0.40** | −0.03 |

「覆盖」作为解释被证伪了：它反而略微**有害**。逐 tag 看是一次重新洗牌，不是增益 ——

```
pink_hair  0.17 → 0.50     beret         0.67 → 0.17
red_eyes   0.67 → 1.00     black_ribbon  0.50 → 0.17
```

发色和瞳色变好；配饰退步。同样的位移预算，摊到了 4× 的权重上。

## 3. 对齐配方、对齐覆盖，各自取各自的最优

matched LoRA 是 r8/α4、all-linear（同样那 700 个 module）、lr 5e-4 constant、
1200 步、同样的 latent、同样的 caption 变体。20.9 M 参数对 GPart 的 23.2 M ——
这里 GPart 的容量**更多**。

| config | modules | α | ‖ΔW‖ | score (p0) | score (p2 留出) |
|---|---|---|---|---|---|
| base | — | — | 0 | 0.08 | — |
| **LoRA matched** | all-linear r8 | 1.0 | 68.77 | **0.70** | **0.73** |
| GPart attn2 | 280 | 1.0 | 78.29 | 0.43 | 0.53 |
| GPart all-linear | 700 | 1.0 | — | 0.40 | — |
| GPart all-linear | 700 | 1.3 | — | 0.55 | — |
| **GPart all-linear（峰值）** | 700 | 1.5 | — | **0.57** | — |
| GPart all-linear | 700 | 1.8 | — | 0.40 | — |

两边都在各自的最优点上。matched LoRA 在 1.0 饱和，在 1.5 和 2.0 变差；GPart 在
1.3–1.5 见峰，到 1.8 崩掉。这张表里没有谁是被压低了强度的。

### α 扫描是一个真实的峰，而这个峰是用图像质量换来的

从 α 1.0 到 1.5 换来 +0.17 分。看 contact sheet 就能清楚看到换来的是什么：配饰
出现了，但服装细节没跟上；到 α 1.5 已出现整体偏色；α 1.8 时背景糊掉、构图变形。
tag 分数在涨，图像在变差 —— 这正是「分数本身不是质量指标」的原因
（`00-claims.zh-CN.md`）。

### 位移效率

为了拿到一个**更低**的分数，GPart 需要大约 2× 于 matched LoRA 的逐权重 RMS
（α 1.3 时 0.0029，对比 LoRA 自己在 1.0 的位移）。‖ΔW‖ 同一个数量级，单位位移里
携带的角色信息更少 —— 这个发现在画风和 stacking 任务里再次出现
（`03-results-style-stacking.zh-CN.md`）。

## 4. 步数预算：训练不足到底拿走了什么

给 1200 步那个 run 的中间 checkpoint 打分（`p0`，`n = 3`）：

| steps | score | beret | hair_ornament | 备注 |
|---|---|---|---|---|
| 400 | 0.30 | **0.00** | 0.33 | 发色和瞳色已经在，配饰没了 |
| 600 | 0.23 | 0.00 | 0.67 | 在 400 的噪声范围内 |
| 800 | 0.33 | 0.67 | 0.67 | 配饰出现 |
| 1200 | 0.43 | 0.67 | 0.83 | 参照（`n = 6`） |

训练不足不是均匀衰减。粗粒度、高对比度的身份线索（发色、瞳色）先到，配饰最后到。
400 步时 `white_hair` 和 `red_eyes` 已经是 1.00，而 `beret` 是 0.00。

实际后果：**800 步是能保住配饰的最省预算。**（400/600/800 三个点之间的差异小于
`n = 3` 的噪声 —— 不要把它们的先后顺序读成一条曲线。）

## 5. 身份差距取决于 prompt，而在等位移量下它是零

上面所有数字都是在 `p0` 一个 prompt 上量的。把 §3 的对比换到四种景别上重跑，
每种 3 个 seed（每臂 12 张，adapter 不变，名义 α 1.0 下 `‖ΔW‖` 68.8 vs 78.4）：

| | 景别 | LoRA @1.0 | GPart @1.0 |
|---|---|---|---|
| h0 | 近景 | 0.77 | 0.70 |
| h1 | 大远景天台 | 0.47 | 0.40 |
| h2 | 中景 / 列车 | 0.67 | 0.53 |
| h3 | 背面 | 0.30 | 0.37 |
| | **均值** | **0.55** | **0.50** |

按 (prompt, seed) 配对重采样 4000 次：**+0.050，95% CI [−0.058, +0.175]** —— 分不出零。
逐景别看，四个景别的置信区间也全部跨 0（`scripts/boot_identity_2026-11-02.py`）。

所以 §3 那个 0.27 的差距是 `p0` 造的。`p0` 是 `full body, standing, from front, …
crowd in background` —— 这组里结构性要求最高的一个。LoRA 覆盖 attn1 + FFN，管空间
结构；GPart 只挂 attn2，管 prompt 注入。**prompt 越约束构图，LoRA 赢得越多。**
拍一张没什么构图可言的近景，两家打平。

### 强度上限是不对称的

两家在 α 上都是严格线性的（`ΔW` 随 LoRA scale 缩放，GPart 则是 `P·diag(s)·(αθ)`）。
最优点不是：

| GPart 角色 α | ‖ΔW‖ | 身份（叠 GPart 画风 @0.469） |
|---|---|---|
| 0.88 | 68.8 | 0.42 |
| 1.0 | 78.4 | 0.54 |
| **1.3** | 101.9 | **0.63** |
| 1.6 | 125.4 | 0.56 |

LoRA 在 1.0 到顶、1.5 和 2.0 退化；GPart 在 1.3–1.6 才到顶。既然两家对自己的缩放
参数都是线性的，这个不对称就是两个 `ΔW` **方向**的性质 —— LoRA 的方向里有些分量被
放大后是破坏性的，GPart 那个稀疏的 attn2-only 支撑集里这种分量少。这意味着
「各自取最优」这个比较口径本身就不中立：

- **等位移量**（GPart @0.88 vs LoRA @1.0，两边 ‖ΔW‖ 都 68.8）：GPart 0.417 vs
  LoRA 0.500，GPart − LoRA = −0.083，CI [−0.225, +0.058] —— **没差别**。
- **各自最优**（GPart @1.3，1.8 倍位移）：GPart 0.633 vs LoRA 0.500，
  GPart − LoRA = +0.133，CI [+0.025, +0.233] —— **GPart 领先**。

哪个才算「对」取决于问什么。问「哪种参数化在每单位权重变化里携带更多身份」，是打平。
问「我该拿哪个 adapter 以它的最佳设置出图」，GPart 能吃下更大位移这件事本身就是资产。

这些行的完整三家横评见 `results/tables/holdout_three_family.csv`（§`03` §5）。

## 6. 图像

| 图 | 内容 |
|---|---|
| `results/figures/horizontal_full_prompt_character.png` | 完整 prompt，6 个 seed：base / kohya LoRA / GPart attn2 / matched LoRA |
| `results/figures/horizontal_alpha_sweep_top.png` | `p0`，3 个 seed：base / matched LoRA / GPart attn2（α 1.0）/ GPart all-linear（α 1.0） |
| `results/figures/horizontal_alpha_sweep_bottom.png` | `p0`，3 个 seed：GPart all-linear 的 α 1.3 / 1.5 / 1.8 |

在接受 0.57 这个峰值之前，先把下面那张图里 α 1.5 和 1.8 两行，拿去和上面那张图里
α 1.0 那一行对比。

## 7. 本任务特有的注意事项

- 一个角色。第二个角色（`juxueli`，同一套采集流程）作为单臂复现进行中 —— 它能
  检验 GPart 到底能不能学会另一个角色，但**不会**承载 GPart-vs-LoRA 的结论。
- **身份分数不是一个稳定的排名。** 换 prompt 集会移动 0.22，再叠一个画风 adapter
  又移动 0.09（§5）。本文件里任何单独引用的数字，都是*某个配置*的数字，不是某种方法的。
- tag 集合是按 caption 词频选出的 10 个 tag，并检查过 base 区分度；换一套 tag
  集合会移动绝对分数。base 那一行是锚。
- `p2`（留出的 caption）只对三个主要行打了分。
