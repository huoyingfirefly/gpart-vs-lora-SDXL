# 03 · 备数据

目标：一个目录，里面有图、caption、和 kohya 生成的两种缓存。

```
data/10_my_character/
├── img_01.jpg
├── img_01.txt                     caption
├── img_01_0960x1436_sdxl.npz      latent（kohya 产出）
├── img_01_te_outputs.npz          文本编码器输出（kohya 产出）
├── img_02.jpg
└── ...
```

## 1. 图片

- **15–60 张**。少于 10 张触发词学不稳（训练器会警告）；超过 60 张收益很快变平。
- 同一个角色，**一个角色**。出现 `multiple girls`、`2girls`、`1boy` 的图要么删掉，
  要么裁到只剩主角 —— 否则触发词会被分配去表示「第二个人」。
- 分辨率不超过 `--max-res`（默认 1536）。超过的样本会被 `src/data.py` 静默跳过，
  所以数量对不上时先查这里。
- 竖图横图混着没关系，`--enable_bucket` 会分组。

## 2. caption 怎么写

每张图一个同名 `.txt`，逗号分隔的 booru 风格标签，**开头写触发词**：

```text
abc, 1girl, solo, looking at viewer, blush, smile, skirt, shirt, blue hair, orange eyes, couch
```

三条规则：

1. **触发词放在 caption 里。** `caption_variants` 造三种变体，其中 `full` 变体就是
   caption 原文 —— 触发词不在原文里，这个变体就白训了。
2. **外观标签照实写，然后交给 `--identity` 去剥。** 这正是协议要的：同一个 latent
   在不同变体里分别配「有外观描述」和「只有触发词」的条件，模型才会把外观沉淀进
   触发词，而不是沉淀进 `blue hair` 这几个字。
3. **别写画质词。** `masterpiece, best quality` 这类会挤占 caption 预算，而且它们
   在每个样本里都出现，等于给模型加了一个常量偏置。`results/` 的 run 一个都没写。

`--identity` 里的写法和 caption 里的写法要对上（大小写无所谓，同义词不行）：

| caption 里写的 | `--identity` 里写 |
|---|---|
| `blue hair` | `blue hair` ✅ |
| `blue hair` | `bluehair` ❌ 剥不掉 |
| `blue_hair` | `blue hair` ❌ 剥不掉 |

剥不掉的后果不报错，只是那个标签没被剥，三种变体区分度下降 —— 静默变差，
所以**训练前先手算一个样本**：

```python
from src.prep.caption_variants import build_variants
print(build_variants(
    "abc, 1girl, solo, looking at viewer, blue hair, orange eyes, couch",
    "abc", {"blue hair", "orange eyes"}))
# ['abc, 1girl, solo, looking at viewer, blue hair, orange eyes, couch'   ← full
#  'abc, 1girl, solo, looking at viewer, couch'                          ← scene
#  'abc, 1girl, solo']                                                    ← bare
```

三条对得上，再去训练。

## 3. kohya 缓存

训练器**不读原图**，读的是 kohya sd-scripts 缓存的 latent 和文本嵌入。
用 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 跑一次带缓存的训练
（跑一步就够，产出缓存后 Ctrl-C 掉）。

`dataset.toml`：

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 1536      # 要和训练时 --max-res 一致，或更小
bucket_reso_steps = 64

[[datasets]]
[[datasets.subsets]]
image_dir = "D:/AI/lora/datasets_juxueli/10_juxueli"
num_repeats = 1
```

```bash
accelerate launch --num_cpu_threads_per_process 4 sdxl_train_network.py \
    --dataset_config dataset.toml \
    --pretrained_model_name_or_path D:/AI/models/checkpoints/NoobAI-XL-v1.1.safetensors \
    --output_dir D:/AI/lora/outputs/_cache_probe --output_name _cache_probe \
    --network_module networks.lora --network_dim 4 \
    --cache_latents --cache_text_encoder_outputs \
    --max_train_steps 1
```

sd-scripts 版本之间 flag 名有出入，以 `sdxl_train_network.py --help` 为准；
关键是 **`--cache_latents` 和 `--cache_text_encoder_outputs`** 两个都开。

产出就落在 `image_dir` 里：

```
img_01_0960x1436_sdxl.npz      文件名里带 WxH，src/data.py 用正则从这拿尺寸
img_01_te_outputs.npz          hidden_state1 / hidden_state2 / pool2
```

**latent 存的是 VAE 原始均值，不要预乘 0.13025。** `--cache_latents` 默认就是这么写的，
别自己再缩放一遍。

## 4. 验证数据集

```bash
python scripts/train_gpart.py --data data/10_my_character \
    --name _probe --trigger abc --identity "blue hair,orange eyes" --dry-run
```

`--dry-run` 只做审计 + 打印命令，不训练。它会检查：

- `*_sdxl.npz` 在不在（不在就说明这不是缓存目录）
- `*_te_outputs.npz` 在不在
- `*.txt` 在不在
- 样本数是不是少于 10（警告）

界面里对应的是「检查这个数据集」按钮，同一个逻辑。

## 5. 生成变体（下一步会做，也可以先手动看看）

```bash
python -m src.prep.caption_variants \
    --src data/10_my_character --dst data/10_my_character_var \
    --trigger abc --identity "blue hair,orange eyes"
```

它会**重新编码**三种 caption（所以需要 `data/noobai_v11_aux` 和 GPU），
把 latent 硬链接/拷贝过去，产出 `img_01v0/v1/v2` 三组。
`train_gpart.py` 会自动做这一步，你不用单独跑 —— 除非想先检查变体文本对不对。
