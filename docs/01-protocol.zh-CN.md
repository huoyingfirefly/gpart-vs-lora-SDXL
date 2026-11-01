> English: [01-protocol.md](01-protocol.md)

# 协议

整个仓库的重点就是这个协议，所以它写在数字前面。

## 0. 这套协议要防的那种失败模式

一次 PEFT 对比只有在两个 adapter 收到**同一个训练信号**时才有意义。不碰算法就能
造出 30× 的权重位移差：

| 旋钮 | 影响 |
|---|---|
| lr | 5× 的差距就足以把一个「死掉」的 adapter 变成一个强 adapter |
| optimizer 步数 × grad-accum × batch | exposure 次数 |
| lr schedule（constant vs cosine-with-restarts） | 随时间变化的有效步长 |
| 哪些 module 被 adapt | 容量与覆盖半径 |
| 采样时用的那个**强度** | 只对其中一方做事后调音 |

随仓库发布的 GPart 与 kohya-LoRA 配置在这五项里有四项不同。任何从它们得出的
headline 都是在比较配方。

## 1. 固定采样协议

`results/` 里所有图像都用同一套采样配置。选它是为了对齐 ComfyUI 的 "normal"
schedule（diffusers 里叫 `trailing` —— 这点很重要，默认的 `leading` 会让每张图
都偏移）：

```
scheduler   Euler ancestral
spacing     trailing
steps       30
CFG         5.0
size        832 x 1216
dtype       bf16
```

## 2. 训练配方（所有对齐过的 run）

```
optimizer        AdamW (weight_decay 0)
lr               5e-4, constant, warmup 50 steps
steps            1200  (800 for the shortened character runs, see §5)
grad accumulation 2  (effective batch 2)
noise offset     0.05
min-SNR gamma    5
grad clip        1.0
max resolution   1536 (bucketed)
seed             42
holdout          4 images, val every 200 steps
base model       NoobAI-XL v1.1 (SDXL), UNet only
```

数据：每个角色 21 张原图，每张扩展成 **3 个 caption 变体**（完整 caption / trigger
+ 场景 tag / 只有 trigger），以此强迫 trigger token 去承载身份。共 63 个样本，
1200 步约等于 38 个 epoch。

## 3. Adapter 配置

| 行 | adapter | targets | modules | 覆盖权重 | params |
|---|---|---|---|---|---|
| `gpart_attn2` | GPart, `d = 23,224,320` | `attn2.{to_q,to_k,to_v,to_out.0}` | 280 | 545.6 M | 23.2 M |
| `gpart_alllinear` | GPart，同一个 `d` | + `ff.net.0.proj`, `ff.net.2` | 700 | 2184.0 M | 23.2 M |
| `lora_r8_matched` | LoRA `r=8, α=4` | 同样 700 个 | 700 | 2184.0 M | 20.9 M |
| `lora_kohya` | LoRA `r=16, α=8`, kohya | all linear | 700 | 2184.0 M | 42.7 M |

`gpart_alllinear` 存在的唯一目的是回答「覆盖解释了多少？」—— 相对 `gpart_attn2`，
它**只**改了 `--targets`。

`lora_kohya` 是开箱参照：lr 1e-4、cosine-with-restarts、10 epochs、batch 1×4、
1024 buckets、AdamW8bit。它放在这里是为了量化那个混淆因子，**不是**作为公平的
对手。

## 4. 指标

**角色身份** —— WD14 booru tagger，对 10 个标准 tag：

```
white_hair  pink_hair  gradient_hair  red_eyes  beret
hair_flower hair_ornament choker black_dress black_ribbon
```

tag 概率 ≥ 0.35 记为命中。`score` = 10 个 tag 上的平均命中率。`base`（不带
adapter）是地板；读差值，不要读绝对值。

prompt：`p0`（干净场景，只有 trigger —— 有区分度的那个 prompt）与 `p1`（完整描述
—— 该由 adapter 提供的一切都被写明了，所以它几乎测不到 adapter）。结论报在 `p0` 上。

`p0` 同时也是这组里结构性要求最高的 prompt（全身、站立、背景人群），所以 §`02` §5
补了一个**四景别留出集** —— 近景 / 大远景 / 中景 / 背面，各 3 个 seed：

```
h0  upper body, close-up, looking at viewer, indoors, cafe, night, warm lamp light
h1  wide shot, full body, standing, from side, outdoors, rooftop, sunset
h2  medium shot, sitting, hugging knees, indoors, train interior, dusk light
h3  full body, walking, from behind, outdoors, rainy street, neon reflections, night
```

（每句前面都带 `r1999style, aima, 1girl, solo,`，后面带质量 tag；完整文本见
`configs/prompts/holdout4.jsonl`）

`h0`–`h3` 覆盖了「身份 adapter 几乎无事可做」（`h0`）到「事情最多」（`h3`）这一整段，
后者十个 tag 里有 `red_eyes`、`choker` 两个结构性看不见，天花板是 0.8 不是 1.0。
**不写清楚 prompt 集就引用的身份数字，是「作者碰巧用了哪一套」的数字**；
两套只在 `p0` 这种 prompt 上方向一致。

**画风 / 图像变化** —— 对同一 seed、不带 adapter 的那张图：

```
dRGB   逐通道平均 |I - I_base|
S      平均饱和度
V      平均明度
edge   亮度通道的平均 |梯度|
```

`edge` 是更有用的那个：它把「图像变化很大」和「图像保住了细节」区分开。

**位移** —— GPart 用 `‖ΔW‖` = α‖θ‖，LoRA 用 `‖BA‖`；逐权重 RMS 的分母用
adapter 实际覆盖的权重数。这个约定很重要：一个只 adapt attn2 的 adapter，若按
整个模型归一化，看上去会小约 2×。凡是在乎这点的地方，两种约定都报。

## 5. 强度在两边都要扫

永远不要拿一个方法的最优值去比另一个方法的默认值。实际跑过的扫描：

| | 扫了哪些 |
|---|---|
| kohya LoRA | 1.0 / 1.5 / 2.0 / 3.0 |
| matched LoRA | 1.0 / 1.5 / 2.0 |
| GPart attn2 | 1.0 |
| GPart all-linear | 1.0 / 1.3 / 1.5 / 1.8 |

两次扫描缺失，作为缺口列明：GPart attn2 在 1.0 以上没扫；all-linear 切换之后的
GPart 在强度 1.0 上没扫（这一项是单独做的）。

*缩短版* run（800 步而不是 1200 步）的步数预算，是从 `02-results-character.zh-CN.md`
§4 的步数曲线推出来的，不是随手定的。

## 6. Seed

角色行：6 个 seed（`310977364113511, 12345, 67890, 111111, 222222, 333333`）。
画风与 stacking 行：3 个 seed × 2 个 prompt。所有表都标了 `n`。

## 7. 没有控制住的东西

- GPart 的 `d`（两个点：23.2 M、50 M）和 lr（一个值）。
- LoRA 的 rank（只有 r8 和 r16，而且两者来自不同配方）。
- Text encoder：从未训练。内置的 GPart 路径只训 UNet。
- NoobAI-XL 各 checkpoint 之间的底座差异（只用了 v1.1）。

任何依赖这些被扫描过的结论，都超出本仓库的范围。
