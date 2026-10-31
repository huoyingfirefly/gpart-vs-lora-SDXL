# 02 · 放底模

训练器和采样器要两样东西，都是 diffusers 格式：

```
data/noobai_v11_unet/     UNet2DConditionModel（约 2.6 GB，bf16）
data/noobai_v11_aux/
├── text_encoder/         CLIPTextModel
├── text_encoder_2/       CLIPTextModelWithProjection
└── vae/                  AutoencoderKL
```

这两个名字是 `scripts/train_gpart.py` 和 `gui/app.py` 的默认值，改名字就多写
`--unet` / `--aux`。

## 1. UNet：单文件 checkpoint → diffusers

如果你手上的底模是单个 `.safetensors`（NoobAI-XL、Illustrious、Pony 这类都一样）：

```bash
python -m src.prep.convert_unet \
    --checkpoint D:/AI/models/checkpoints/NoobAI-XL-v1.1.safetensors \
    --config data/sd_xl_base.yaml \
    --output data/noobai_v11_unet
```

- 只抽 `model.diffusion_model.*` 里的 UNet 权重，VAE 和文本塔跳过。
- `--config` 用 `data/sd_xl_base.yaml`（标准 SDXL UNet 结构）。如果你的底模改过结构，
  换成对应的原版 config。
- 输出两个文件：`config.json` + `diffusion_pytorch_model.safetensors`。
- 峰值内存不高（tensor 是零拷贝视图），但磁盘要留 3 GB。

验证：

```bash
python -c "
from diffusers import UNet2DConditionModel
u = UNet2DConditionModel.from_pretrained('data/noobai_v11_unet', local_files_only=True)
print(u.config.sample_size, sum(p.numel() for p in u.parameters())/1e6, 'M')
"
# 期望约 1024 2,567 M —— SDXL UNet 约 25.7 亿参数
```

## 2. aux：文本塔和 VAE

训练要用文本塔给 caption 编码（`caption_variants`），采样要用 VAE 解码
（`gui/app.py`、`src/sample.py`）。

**最稳的做法是从你的底模发布方拿 diffusers 版。** 很多 SDXL 微调都提供 diffusers
格式仓库，里面 `text_encoder/`、`text_encoder_2/`、`vae/` 都是现成的，拷进
`data/noobai_v11_aux/` 即可。

拿不到的话，从 SDXL base 下载这三块：

```bash
pip install "huggingface_hub[cli]"
hf download stabilityai/stable-diffusion-xl-base-1.0 \
    --include "text_encoder/*" "text_encoder_2/*" "vae/*" \
    --local-dir data/noobai_v11_aux
```

> ⚠️ 这一步有前提：**你的底模没有重训文本塔。** 大部分角色/画风微调只动 UNet，
> 这时用 SDXL base 的文本塔是对的。如果发布方明确说过「重训了 TE」，
> 就必须用他们那份，否则 caption 编码和训练时不一致，出图会跑偏。
> 判断方法：对比 checkpoint 里 `conditioner.embedders.*` 的权重和 SDXL base 是否一致。

`NoobAI-XL` 这类全量微调改的是 UNet；VAE 几乎总是 SDXL 原版。

验证：

```bash
python -c "
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection
AutoencoderKL.from_pretrained('data/noobai_v11_aux/vae', local_files_only=True)
CLIPTextModel.from_pretrained('data/noobai_v11_aux/text_encoder', local_files_only=True)
CLIPTextModelWithProjection.from_pretrained('data/noobai_v11_aux/text_encoder_2', local_files_only=True)
print('aux ok')
"
```

## 3. WD14 tagger（可选，只打分用）

训练和出图**不需要**它。只有 `src/eval/tag_eval.py` 给图打角色标签分数时要，
也就是想复现 `results/tables/character.*` 时才要。

```bash
hf download SmilingWolf/wd-v1-4-moat-tagger-v2 --local-dir models/wd14
# 需要 model.onnx 和 selected_tags.csv 在同一个目录下
```

然后：

```bash
python -m src.eval.tag_eval --model models/wd14/model.onnx \
    --dirs base=runs/samples_base gp=runs/juxueli/samples --md
```

`--model` 是必填的，仓库里不写死任何路径。也可以用环境变量 `WD14_MODEL`
给个默认值。

## 4. 显存预算

| | 训练 | 出图 |
|---|---|---|
| UNet | bf16 ~5.1 GB + 激活 | bf16 ~5.1 GB |
| 文本塔 | 训练时不用（缓存已含） | fp32 留在 CPU，~3.3 GB 内存 |
| VAE | — | 解码时临时进显存 2–3 GB |
| 峰值 | ~7 GB | ~6 GB（界面会自动把 UNet 挪走再解码） |

8 GB 卡够用但不宽裕：训练时别开浏览器，别同时跑别的 GPU 任务。
6 GB 卡把 `--max-res` 降到 1024、batch 保持 1。
