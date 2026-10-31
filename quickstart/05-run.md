# 05 · 出图

训练完的适配器在 `runs/<名字>/adapter/`。出图有两种方式：界面和命令行。

## 1. 命令行

```bash
python -m src.sample \
    --adapter-dir runs/juxueli/adapter --alpha 1.0 \
    --prompts-file configs/prompts/p0_p1.jsonl \
    --seeds 1 2 3 \
    --steps 30 --cfg 5 --size 832 1216 \
    --sched euler_a --spacing trailing \
    --tag gp --output runs/juxueli/samples
```

**`--sched euler_a --spacing trailing` 两句都要写。** `src/sample.py` 的默认值是
`euler` + `leading`，和产出 `results/` 的采样器不同：`leading` 会让每一级的噪声
尺度整体偏移，图会「能看出是那个角色，但构图和 ComfyUI 出的不是一回事」。这不是
复现失败，是 spacing 选错了。

对照表：

| | `--sched` | `--spacing` | 谁在用 |
|---|---|---|---|
| 本仓库全部结果 | `euler_a` | `trailing` | `results/` |
| ComfyUI `normal` | — | `trailing` | ComfyUI 用户 |
| ComfyUI `karras` | `euler_a`（sigma 不同） | `trailing` | — |
| `sample.py` 默认 | `euler` | `leading` | 别用 |

一定要出 base 对照：

```bash
python -m src.sample --prompts-file configs/prompts/p0_p1.jsonl \
    --seeds 1 2 3 --steps 30 --cfg 5 --size 832 1216 \
    --sched euler_a --spacing trailing \
    --tag base --output runs/samples_base
```

## 2. 扫 α

GPart 没有内置强度旋钮，**缩放 θ 就是那个旋钮**（`--alpha`）。一个值会应用到所有
适配器，不是扫一排 —— 想扫就循环：

```bash
for A in 0.8 1.0 1.3 1.5; do
  python -m src.sample --adapter-dir runs/juxueli/adapter --alpha $A \
      --prompts-file configs/prompts/p0_p1.jsonl --seeds 1 2 3 \
      --steps 30 --cfg 5 --size 832 1216 \
      --sched euler_a --spacing trailing \
      --tag "a$A" --output "runs/juxueli/samples_a$A"
done
```

Windows 用 `.bat`：

```bat
for %%A in (0.8 1.0 1.3 1.5) do python -m src.sample --adapter-dir runs\juxueli\adapter --alpha %%A --prompts-file configs\prompts\p0_p1.jsonl --seeds 1 2 3 --steps 30 --cfg 5 --size 832 1216 --sched euler_a --spacing trailing --tag a%%A --output runs\juxueli\samples_a%%A
```

在 aima 上的经验（`docs/02-results-character.md` §3）：

| α | tag 分数 | 观感 |
|---|---|---|
| 0.8 | 0.40 | 偏弱，只有大轮廓像 |
| 1.0 | 0.40 | 稳 |
| 1.3 | 0.55 | 明显更像 |
| **1.5** | **0.57** | 分数的顶点，**但整幅偏色、细节开始糊** |
| 1.8 | 0.40 | 过冲，掉回来 |

所以「分数最高的 α」不等于「最好看的 α」。挑你自己眼睛认可的那个。

## 3. 打分（可选）

需要 WD14 tagger（`02-models.md` §3）。

```bash
python -m src.eval.tag_eval --model models/wd14/model.onnx \
    --dirs base=runs/samples_base gp=runs/juxueli/samples \
    --tags "blue hair,orange eyes,hat,bow" --md
```

- `--tags` 换成你这个角色的标准外观标签。默认值是 aima 那套。
- 输出的是**命中率**：每张图被 tagger 判定为「有这些标签」的比例，阈值 `--thr 0.35`。
- 它衡量的是「booru tagger 认不认这个角色」，**不是画质**。偏色、糊、构图崩都不扣分。

想要色彩维度的客观数：

```bash
python -m src.eval.style_stats --ref D:/path/to/reference_images \
    --dirs base=runs/samples_base gp=runs/juxueli/samples
```

## 4. 用界面出图

```bash
python gui/app.py    # → 「② 出图」
```

- 「刷新列表」扫 `runs/` 和 `outputs/` 下所有 `adapter_config.json`，
  包括训练中途的 `step_00200/` 快照 —— 可以用来判断在哪一步开始过拟合。
- α 是那个 Spinbox，改完直接「生成」就会重新从磁盘载入 θ 再合并，不会污染缓存。
- 多张连拍：种子填 `1`、张数填 `3`，出 seed 1/2/3。
- 第一次点「生成」要 60–90 秒（加载 UNet 5 GB + 合并适配器）；之后换提示词是秒级。

## 5. 接到 ComfyUI / WebUI

**仓库不带导出脚本。** GPart 适配器是 PEFT 格式（`adapter_config.json` +
`adapter_model.safetensors`），ComfyUI 和 A1111 都不认。

要接进去，自己写三行把 ΔW 物化进 UNet，再转回单文件 checkpoint：

```python
import torch
from diffusers import UNet2DConditionModel
from peft import PeftModel

unet = UNet2DConditionModel.from_pretrained("data/noobai_v11_unet", torch_dtype=torch.bfloat16)
with torch.no_grad():
    # theta 默认在 CPU，先搬上 GPU 再合并，否则每一步都要过 PCIe
    merged = PeftModel.from_pretrained(unet, "runs/juxueli/adapter").to("cuda").eval()
    merged.merge_and_unload().save_pretrained("runs/juxueli/unet_merged")
```

然后把 `unet_merged/` 和你的底模其余部分（VAE、文本塔）一起用
**diffusers 官方 `convert_diffusers_to_original_sdxl.py`** 转成单文件
safetensors，放进 ComfyUI 的 `models/checkpoints/`。

`merge_and_unload()` 之后 `unet` 里已经没有 LoRA 层了，得到的就是
`W + ΔW`，所以 ComfyUI 那边不用再挂任何东西 —— 强度只能通过换不同 α 导出的版本来调。
这也是为什么在 aima 上做 ComfyUI 对照时，是拿 α=1.0 的合并结果对标 LoRA 0.70。

**先做一次自检再拿出去比：**

```bash
python -m src.eval.check_adapter --adapter runs/juxueli/adapter
```

它会验证 θ 的正交性（`PᵀP ≈ I`）、ΔW 的尺度、以及合并前后是否一致。
数字和 `results/tables/character.*` 里的同一列对不上，说明适配器本身有问题，
不用往采样器上找原因。
