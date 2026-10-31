# 04 · 训练

输入：`03-dataset.md` 那个缓存目录。输出：`runs/<名字>/adapter/`。
中间会生成 caption 变体、跑完整训练循环、写一份 `run_manifest.json`。

## 1. 一条命令

```bash
python scripts/train_gpart.py \
    --data     D:/AI/lora/datasets_juxueli/10_juxueli \
    --name     juxueli \
    --trigger  abc \
    --identity "blue hair,orange eyes" \
    --preset   attn2 \
    --steps    1200
```

四个必填参数，没有一个是能猜的：

| | 为什么必填 |
|---|---|
| `--data` | 没有默认值。默认值就意味着某个人的 `D:\` 路径被写进了仓库 |
| `--name` | 决定 `runs/<name>/`，也决定 `run_manifest.json` 里记什么 |
| `--trigger` | 写进 caption；**必须是底模里原本不存在的词** |
| `--identity` | 要剥掉的外观标签。省略会回退到 aima 那套，换角色就是静默错误 |

先跑一次 `--dry-run` 看它要执行什么：

```bash
python scripts/train_gpart.py ... --dry-run
```

会打印两次 `$ ...`（caption 变体和训练）以及将要写入的 manifest，一个字节都不动。

## 2. 配方是固定的，不要随手改

```
lr 5e-4 · constant · warmup 50 · grad_accum 2 · grad_clip 1.0
max_res 1536 · noise_offset 0.05 · min_snr_gamma 5 · weight_decay 0 · seed 42
targets  attn2.to_q/to_k/to_v/to_out.0  （preset=attn2）
scope    transformer_block，pattern transformer_blocks
backend  stateless，partition splitmix64
d        23224320
```

这套是 `docs/01-protocol.md` 里的对齐配方。**改 lr 就脱离了对比的前提** ——
仓库的结论之所以成立，正是因为两边都跑在 lr 5e-4 上。

想加别的开关用 `--extra` 透传：

```bash
--extra '--theta-norm-max 70'
```

## 3. 两个 preset 怎么选

| preset | 覆盖模块 | 参数 | 每步耗时（8 GB） | 1200 步 |
|---|---|---|---|---|
| `attn2` | 交叉注意力 q/k/v/out，280 个模块 | 545.6 M | ~8.1 s | **~2.7 h** |
| `alllinear` | 再加自注意力和 FFN，700 个模块 | 2184 M | ~20.5 s | ~6.8 h |

- 只是想让角色长得像：**`attn2`**。它是这个仓库的性价比点。
- 要和 LoRA 做公平对比：`alllinear`。LoRA 那批实验覆盖的就是 700 个模块，
  只训 `attn2` 的 GPart 在覆盖上就输了。

耗时跟**覆盖的权重数**走，不跟 `d` 走（`docs/04-cost-model.md`）。
所以调 `--d` 不改变速度，只改变容量。

## 4. 步数

| 步数 | 结果 |
|---|---|
| 400 | 脸能学到，配件（发饰、纹样）丢 |
| 800 | 配件开始出现，是最省的下限 |
| **1200** | 对齐配方用的值，`results/` 都是这个 |
| 2000+ | 开始过拟合，α 扫描时会看到偏色和细节退化 |

中途可以停：每 `--save-every`（默认 200）步落一个 `step_XXXXX/` 快照，
另有一个 `checkpoint/` 指针指向最近一个，`--resume` 从它继续。

## 5. 跑起来之后

```
runs/juxueli/
├── adapter/                 ← 最终适配器（出图用这个）
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── step_00200/              ← 每 200 步一个快照，可以直接拿去出图
├── checkpoint/              ← --resume 的落点
├── train.jsonl              ← 每 10 步一行：loss / val / lr / grad_norm / theta_norm / 显存
├── status.json              ← 每步覆盖：step / loss / sec / peak_gib（界面的进度条读它）
├── args.json                ← 这次训练的全部参数
└── run_manifest.json        ← 环境 + 命令 + 数据集指纹
```

`theta_norm` 值得盯：它从 0 附近开始涨，**涨到 70 左右就是上限附近**
（`--theta-norm-max` 默认没开，别指望它自己停）。如果 1200 步跑完 theta_norm
还不到 20，说明这个学习率对当前数据不够 —— 这正是开箱对比里 LoRA 输掉的原因，
`docs/02-results-character.md` §1。

## 6. 用界面训练

```bash
python gui/app.py
```

「① 训练」页填同样几项，点「开始训练」。界面做的事就是拼出上面那条命令，
子进程跑 `scripts/train_gpart.py`，日志实时刷在右边，进度条读 `status.json`。

- 「停止」是 `terminate()`，不是优雅退出。已经落盘的 `step_XXXXX/` 不受影响。
- 训练期间「② 出图」会拒绝生成（显存会打架）。
- 日志框第一行就是完整命令，可以复制到终端复现。

## 7. 训练完的第一步不是调参，是看 base

```bash
python -m src.sample --adapter-dir runs/juxueli/adapter --alpha 1.0 \
    --prompts-file configs/prompts/p0_p1.jsonl --seeds 1 2 3 \
    --steps 30 --cfg 5 --size 832 1216 \
    --sched euler_a --spacing trailing --output runs/juxueli/samples
```

同时出一组 `--adapter-dir` 留空的 base 图做对照。**没有 base 对照的「看起来像了」
是不可信的** —— 提示词写得够细时，底模自己就能画出这个角色（`docs/02` §6）。
