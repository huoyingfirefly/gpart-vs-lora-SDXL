> English: [00-claims.md](00-claims.md)

# 本仓库主张什么、不主张什么

这一页是刻意最先写的。任何对外表述都应该能追溯到下面某一行。如果 README 或论文
草稿里有一句话在这里找不到支撑，那句话就是越界。

## 我们主张

| # | 主张 | 证据 |
|---|---|---|
| C1 | 在这个任务上，「GPart 打赢 LoRA」的开箱结果是训练配方解释的，不是参数化方式解释的。随仓库发布的 LoRA 跑在 lr 1e-4，逐权重位移只到 GPart 的 1/33。 | `02-results-character.zh-CN.md` §1 |
| C2 | 在训练配方、层覆盖、容量、以及各自的最优强度全部对齐之后，LoRA 在角色身份 tag 上得分更高（0.70 vs 0.57）。 | `02-results-character.zh-CN.md` §3 |
| C3 | 层覆盖不是原因：在覆盖对齐之后，all-linear GPart 得 0.40，而 attn2-only 是 0.43。 | `02-results-character.zh-CN.md` §2 |
| C4 | GPart 的强度工作点扫过 α = 1.0/1.3/1.5/1.8；1.3–1.5 是一个真实的峰，而且仍然低于 LoRA 的最优点。 | `02-results-character.zh-CN.md` §3 |
| C5 | 在相同图像位移（dRGB）下，GPart 的饱和度和边缘能量都低于 LoRA —— style 任务、stacking 任务、character 任务上都是如此。 | `03-results-style-stacking.zh-CN.md` |
| C6 | 训练不足对身份 tag 的破坏是不均匀的：400 步时发色和瞳色已经迁移，但配饰得分 0.00；到 800 步它们才出现。 | `02-results-character.zh-CN.md` §4 |
| C7 | 在本仓库的 `stateless` backend 下，每步成本随**被覆盖的权重数**增长，几乎与 `d` 无关：2184 M → 20.5 s/step，545.6 M → 8.11 s/step，而 `d` 从 23.2 M 提到 50 M 只多 **+13%**。 | `04-cost-model.zh-CN.md` |

## 我们明确不主张

| ✗ | 为什么不 |
|---|---|
| 「低秩在通用意义上优于满秩 PEFT。」 | 一个角色、一个画风、一个底座模型、一类指标。 |
| 「GPart 做不到 X。」 | 我们只测了一族配置。`d`、lr、target module 的选择都没有超出协议里标注的那两个点去做扫描。 |
| 「GPart 比 LoRA 慢。」 | 4.2× 这个数字只属于本仓库的 stateless backend。上游 GPart 会把 assignment 表实体化，那份内存在 8 GB 里放不下。这是实现属性，不是方法属性。 |
| 「LoRA 更适合做 adapter 组合的底座。」 | 我们只测了朴素加法合并。GPart 真正的卖点是*在共享坐标下、在非加法运算（TIES、正交化）下保持精确的线性组合*。那个没做。 |
| 「tag 分数衡量图像质量。」 | 它衡量的是 booru tagger 与该角色标准 tag 的一致度。分数升高可能同时伴随图像变差；α 扫描正好显示了这一点。 |

## 一句话版本

> 在一个 SDXL 角色 + 一个画风的受控对比里，把训练配方对齐之后，全秩 partition
> 相对低秩 adapter 所声称的优势消失了；剩下的差异表现为：在相同图像位移下，
> 饱和度更低、细节保留更少。
