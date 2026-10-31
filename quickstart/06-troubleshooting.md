# 06 · 出错对照

先看一张总表，再看下面按阶段展开的细节。

| 症状 | 最可能的原因 | 去哪 |
|---|---|---|
| `no cached samples found in ...` | 目录里没有 `*_sdxl.npz`，或文件名不符合 `名字_WxH_sdxl.npz` | §1 |
| 训练日志说 63 个样本，实际有 84 个文件 | `*_te_outputs.npz` 缺一半，配对失败被**静默跳过** | §1 |
| 训练能跑，出图完全没有角色 | `--identity` 和 caption 里的写法对不上，三种变体一模一样 | §2 |
| `CUDA out of memory`（训练） | `alllinear` + 1536 分辨率太满 | §3 |
| `CUDA out of memory`（出图） | 分辨率太大，或训练进程还占着显存 | §3 |
| `no gpart_theta_d tensors found` | 选中的是 kohya LoRA，不是 GPart 适配器 | §4 |
| 图和 ComfyUI 差很多，构图整体偏移 | 少了 `--spacing trailing` | §5 |
| 图全灰 / 全平 | 适配器没搬上 GPU 就合并了 | §4 |
| `ModuleNotFoundError: tkinter` | 精简版 Python | §6 |
| tag 分数**低于** base | 适配器太弱（位移不够），不是训练崩了 | §7 |
| loss 一开始就在 0.04 左右不动 | 正常。这个 loss 的量级就是 10⁻² | §7 |

---

## 1. 数据集

**`no cached samples found`** —— `src/data.py` 的正则是
`^(?P<id>.+)_(?P<w>\d+)x(?P<h>\d+)_sdxl\.npz$`。所以要同时满足：

- 文件在 `--data` 指的目录**里**（不递归）
- 名字形如 `img_01_0960x1436_sdxl.npz`
- 有配对的 `img_01_te_outputs.npz`

**数量对不上** —— 这是最容易骗过人的一个：`src/data.py` 遇到没有配对 TE 的 latent
会 `continue`，**不报错**。所以：

```bash
# 三个数应该相等
ls data/10_my_character/*_sdxl.npz       | wc -l
ls data/10_my_character/*_te_outputs.npz | wc -l
ls data/10_my_character/*.txt            | wc -l
```

不等就说明有样本被跳了。`scripts/train_gpart.py --dry-run` 和界面的「检查数据集」
都会把这三个数打出来。

**样本数远少于预期** —— 分辨率超过 `--max-res`（默认 1536）的样本也会被跳过。
`960x2161` 这种竖图就会中招。要么调大 `--max-res`（显存会更紧张），要么先裁图重缓存。

## 2. caption 变体没区分度

判断方法，训练前跑一次：

```bash
python -m src.prep.caption_variants --src data/10_my_character \
    --dst data/10_my_character_var --trigger abc --identity "blue hair,orange eyes"
```

看打印出来的三行。如果 `v0` 和 `v1` 几乎一样，说明 `--identity` 里的词一个都没匹配上。
匹配是**精确的**（只忽略大小写）：

- caption 里有 `blue hair`，`--identity "blue_hair"` ❌
- caption 里有 `hair between eyes`，`--identity "hair"` ❌（是子串，不是分词）

`--identity` 要照抄 caption 里的标签原文。

**触发词不在 caption 里** —— `build_variants` 的 `full` 变体是 caption **原文**，
不会自动加触发词。触发词只在 `scene` / `bare` 两个变体里出现。所以 caption 必须以
触发词开头，否则三分之一的数据在教模型「别提这个角色」。

## 3. 显存

训练峰值约 7 GB（8 GB 卡）。

```bash
--extra '--max-res 1024'
--preset attn2                 # 而不是 alllinear，省 12 秒/步和 1.5 GB
```

`--grad-accum 2` 是把两个 micro-batch 累加，**不省显存**，省的是「等效 batch」的
统计特性。训练器的 batch 固定为 1，没有 `--batch-size` 可调 —— 想省显存就降分辨率。

出图峰值约 6 GB。界面在解码前会看剩余显存，低于 3.5 GiB 就把 UNet 挪到内存
（`gui/app.py` 的 `decode()`）。命令行 `src/sample.py` 是整批采样完再 `unet.to("cpu")`
再解码，所以它天然更省 —— 但批量大时 latents 全存在 CPU 内存里。

**训练和出图同时跑一定 OOM。** 界面会拦，命令行不会。

## 4. 适配器

**`no gpart_theta_d tensors found`** —— 你选的是 kohya LoRA（`adapter_config.json`
里 `peft_type` 是 `LORA`），不是 GPart。LoRA 走 `src/sample.py --lora path --lora-scale`。

**图全灰 / 一片平涂** —— θ 停在 CPU 就合并了。PEFT 从磁盘载入的参数默认在 CPU，
`merge_and_unload()` 会在错误的设备上算，或者每一步都过 PCIe（慢 7 倍）。
正确写法是 **先 `.to(device)` 再合并**：

```python
PeftModel.from_pretrained(unet, path).to(device).eval().merge_and_unload()
```

`src/sample.py:248` 和 `gui/app.py` 的 `set_adapter()` 都是这么写的。老版本
`gui.py` 漏了 `.to(device)`，是最常见的「适配器没生效」原因。

**适配器本身有没有问题**：

```bash
python -m src.eval.check_adapter --adapter runs/juxueli/adapter
```

## 5. 采样器

**图和 ComfyUI 差一截，构图整体偏** —— 99% 是 `--spacing`：

| 你的命令 | 实际效果 |
|---|---|
| `--sched euler_a --spacing trailing` | ✅ 和 `results/` 一致 |
| `--sched euler_a`（默认 `leading`） | ❌ 每级 sigma 偏移 |
| `--sched euler` | ❌ 另一个采样器 |

**Stable Diffusion 的官方 scheduler config 里写的就是 `leading`**，所以直觉上
`leading` 才「标准」。不是 —— SDXL 的官方 pipeline 用的是
`EulerDiscreteScheduler` + `leading`，而产出 `results/` 的 run 用的是
euler **ancestral** + `trailing`，也就是 ComfyUI 的 `normal` 调度。这是个会静默
毁掉复现的坑，所以 `configs/*.yaml` 每一行都显式钉死这两个值。

**同一 seed 出图不一致** —— 除了 CUDA 的 kernel 非确定性，还要检查是不是换了
`--sched`。scheduler 的 `step_index` 存在 scheduler 上，`src/sample.py` 在**每个 seed**
前都会 `set_timesteps()` 重置；自己写采样循环时漏了这步，第二张图起就会偏。

## 6. 界面起不来

**`ModuleNotFoundError: No module named 'tkinter'`**

| 系统 | 修法 |
|---|---|
| Windows | 用 python.org 的官方安装包重装，安装时勾上 `tcl/tk and IDLE` |
| Ubuntu/Debian | `sudo apt install python3-tk` |
| macOS (homebrew) | `brew install python-tk@3.10` |

**窗口打开是空的 / 立刻退出** —— 在仓库根目录跑：

```bash
python gui/app.py
```

路径全部相对仓库根（`Path(__file__).resolve().parents[1]`）。`launch.bat` /
`launch.sh` 会自己 `cd` 过去，直接 `python D:\...\gui\app.py` 也行，因为代码用的是
`__file__` 而不是 `cwd`。

## 7. 数字不对

**tag 分数低于 base** —— `docs/02-results-character.md` §1 里的
kohya LoRA 就是这样：0.05 对 base 0.08。原因是权重位移只有 GPart 的 1/33，
模型几乎没动。看 `theta_norm`：

```bash
tail -1 runs/juxueli/train.jsonl
```

`theta_norm` 涨到 20 以上才算「训进去了」，70 是上限附近。1200 步还不到 10，
说明这个 lr 对当前数据太保守。

**loss 一直在 0.04** —— 这个量级正常（flow/epsilon 目标的 mse 在 10⁻²）。
要看的是**趋势**不是绝对值：

```bash
python -c "
import json
for line in open('runs/juxueli/train.jsonl'):
    r=json.loads(line); print(r['step'], round(r['loss'],4), r.get('val'))
" | head -30
```

**val 比 train 高很多且一直涨** —— 过拟合。数据集太小或步数太多，减步数或加图。

**val 是 NaN** —— lr 太高。5e-4 是这套配方验过的值，别往上加。

---

## 还是不对

把这三样贴出来，基本能定位：

```bash
python scripts/train_gpart.py ... --dry-run        # 实际要跑的命令
cat runs/<名字>/run_manifest.json                  # 环境 + 参数 + 数据集指纹
tail -5 runs/<名字>/train.jsonl                    # 最近几步的 loss / theta_norm / 显存
```

`run_manifest.json` 里连 torch 版本、CUDA 版本、GPU 型号都记了，就是为了让
「我这跑不出来」能被别人复现。
