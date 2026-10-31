# GPart Studio — 训练 + 出图的一体化界面

一个 tkinter 桌面程序，两个页签：**① 训练**（选数据集 → 开始训练 → 看日志），
**② 出图**（选适配器 → 写提示词 → 生成）。

没有新增依赖。`tkinter` 是 CPython 自带的，`torch` / `diffusers` 是仓库本来就要装的。

```bash
python gui/app.py          # 或 gui\launch.bat / ./gui/launch.sh
```

第一次启动只建窗口，**不加载模型** —— 8 GB 卡上 UNet 一加载就是 5 GB，而你多半只想
先看一眼训练日志。模型在第一次点「生成」时才进显存。

---

## 它做的事，以及它不做的事

| 做 | 不做 |
|---|---|
| 把 `scripts/train_gpart.py` 跑起来，实时把日志和进度打出来 | 不重新实现训练。界面和命令行**不可能**跑偏，因为它就是同一个进程 |
| 加载 GPart 适配器、合并进 UNet、用协议采样器出图 | 不训 LoRA 臂（`src/train.py --adapter lora` 走命令行） |
| 扫 α、多张连拍、存 PNG | 不打分（`src.eval.tag_eval` 走命令行） |
| 检查数据集缺不缺 npz / caption | 不替代 kohya 生成 latent 缓存 |
| 列出 `runs/` 和 `outputs/` 下所有适配器 | 不进 ComfyUI / A1111（见 `quickstart/05-run.md`） |

---

## ① 训练页

字段和 `train_gpart.py` 一一对应，界面只是把它们拼成命令行：

```
数据集（kohya 缓存目录）  →  必须是含 *_sdxl.npz 的目录
输出名                    →  runs/<输出名>/
触发词                    →  写进 caption，是唯一的身份锚点
要剥掉的外观标签           →  必填。留空会用 aima 那套标签，换角色就是静默错误
覆盖                      →  attn2（280 模块，~8.1 s/步）/ alllinear（700 模块，~20.5 s/步）
步数 / d / 种子
```

「检查这个数据集」会数三个东西，缺哪个直接标红：

- `*_sdxl.npz` —— latent 缓存（kohya 产出）
- `*_te_outputs.npz` —— 文本编码器输出（kohya 产出）
- `*.txt` —— caption

训练跑的完整命令会先打进日志框第一行（`$ python scripts/train_gpart.py ...`），
你可以原样复制到终端里复现。「停止」是 `terminate()`，**不是**优雅退出 ——
checkpoint 每 `--save-every` 步落盘一次，从最近的一个继续。

日志里每一行 `{...}` JSON 是训练器的逐步记录，界面从里面读 `step` / `loss` / `sec`
来推进度条和算 ETA，原始行照样显示。

## ② 出图页

- **适配器**：扫 `runs/**/adapter_config.json` 和 `outputs/**/adapter_config.json`，
  按修改时间倒序。第一项 `base（原模型）` 是不挂适配器的对照组 —— 想知道适配器到底
  改了什么，就先出一张 base。
- **α（θ 缩放）**：这是**采样强度**，不是训练的 `--gpart-alpha`。切换 α 会重新从磁盘
  载入 θ 再合并，不会污染缓存。`results/` 里 α=1.0 得 0.40、α=1.5 得 0.57 ——
  不扫一遍就不知道你的角色在哪个强度上最好。
- **提示词预设**：从 `configs/prompts/*.jsonl` 和数据集目录的 caption 里收集。

### 采样器和协议一致

```python
EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
    beta_schedule="scaled_linear", prediction_type="epsilon",
    timestep_spacing="trailing", steps_offset=1)
```

`trailing` 这一项是**故意的**。工作树里那份老的 `gui.py` 用的是 `leading`，并注释说
euler ancestral「把每张图都糊成平涂」—— 那是 `leading` 的锅，不是 ancestral 的锅。
本仓库 `results/` 里所有图都是 ancestral + `trailing` 出的，所以界面对齐的是
`docs/01-protocol.md` 里的采样器，不是那份老界面的。

**和 `src/sample.py` 是逐字节一致的。** 同样的提示词/种子/步数/尺寸/α，
`python -m src.sample --merge ...` 和界面对同一张图输出完全相同的像素
（已按 base 和挂适配器两种情况验证）。前提是三件事不能变：

1. `trailing` 不能改；
2. 负面提示词和 `src/sample.py` 的默认值**逐字相同**（`DEFAULT_NEG`，不带 `signature`）；
3. 文本编码**必须在 CUDA 上做**。

第 3 条是最容易踩的：为了省显存把文本塔留在 CPU 很诱人（它们 fp32 就占 3.3 GiB），
但 CUDA 上的 CLIP 矩阵乘走 TF32，hidden state 最多差 1.2e-1，**4 步采样的图会明显不同**。
`Engine.warm()` 因此先把提示词在 CUDA 上编码完、把结果按 float32 缓存在内存，
再让 UNet 进入显存 —— 既省显存又不改变数值。改这段代码前先看它的注释。

要是你拿界面出的图和 `src/sample.py` 出的图对不上，按上面三条依次查。

### 显存

目标卡是 8 GB。解码时先看剩余显存，低于 3.5 GiB 就把 UNet 挪到内存再解码 ——
1216px 的 fp32 VAE 解码峰值在 UNet 之上还要 2–3 GiB，不挪就 OOM。
UNet 用 bf16，文本塔留在 CPU（fp32，约 3.3 GiB，每个提示词只跑一次）。

采样时**不要同时训练**：显存会直接打架。界面在训练进程活着时会拒绝「生成」。

---

## 路径

没有任何 `D:\` 硬编码。默认值都是相对仓库根的：

| | |
|---|---|
| `data/noobai_v11_unet` | diffusers 格式 UNet，见 `data/README.md` |
| `data/noobai_v11_aux` | `text_encoder/`、`text_encoder_2/`、`vae/` |
| `runs/` | 训练输出、采样图（`runs/<名字>/samples/`） |
| `outputs/` | 也会被扫，方便接老的产物目录 |

权重放在仓库外面时，用环境变量给默认值（和 `src/eval/tag_eval.py` 的 `WD14_MODEL`
是同一套约定）：

```bat
set GPART_UNET=D:\AI\models\noobai_v11_unet
set GPART_AUX=D:\AI\models\noobai_v11_aux
python gui\app.py
```

三个路径都能在界面上改，改完不需要重启。

---

## 出问题时

| 现象 | 原因 |
|---|---|
| 训练页点了没反应 | 数据集没选，或外观标签空着（两个都会弹框说清楚） |
| 日志刷到一半停了，退出码非 0 | 通常是 OOM。降到 `attn2`、`--max-res 1024`、或调小 batch |
| 生成报 `no gpart_theta_d tensors found` | 选中的是 kohya LoRA，不是 GPart 适配器。用 `src/sample.py --lora` |
| 生成图和 ComfyUI 差很多 | ComfyUI 的 `normal` 对应 `trailing`；界面已经是 `trailing`，那多半是 CFG/步数/分辨率不同 |
| 载入适配器后画面全灰 | θ 没加载到 GPU 就合并了（老 `gui.py` 的坑，这里已经 `.to(device)`） |

更多在 `quickstart/06-troubleshooting.md`。
