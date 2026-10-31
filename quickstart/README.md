# Quickstart —— 从 clone 到出图

这个文件夹是**给刚 clone 下来的人**看的：从零装环境、备数据、训出一个自己的 GPart
矩阵、再把它跑出图。按顺序读，每一步都能验证，不用猜自己卡在哪。

```
① clone + 装依赖      02-models.md  放底模
        │                  │
        └──────► 03-dataset.md 备数据（图片 → kohya 缓存）
                          │
                          ▼
                  04-train.md  训练 → runs/<名字>/adapter
                          │
                          ▼
                  05-run.md    出图 / 扫 α / 打分 / 接 ComfyUI
```

`06-troubleshooting.md` 是症状 → 原因的表，出问题先查它。

---

## 三步走（已经有底模和缓存的人）

```bash
# 1. 装依赖
python -m pip install -r requirements.txt

# 2. 训练（约 1.8–2.7 小时，8 GB 卡，attn2 覆盖）
python scripts/train_gpart.py \
    --data D:/AI/lora/datasets_juxueli/10_juxueli \
    --name juxueli --trigger abc \
    --identity "black hair,red eyes" \
    --preset attn2 --steps 1200

# 3. 出图
python -m src.sample --adapter-dir runs/juxueli/adapter --alpha 1.0 \
    --prompts-file configs/prompts/p0_p1.jsonl --seeds 1 2 3 \
    --steps 30 --cfg 5 --size 832 1216 \
    --sched euler_a --spacing trailing --output runs/juxueli/samples
```

不想敲命令就开界面，同样的活两下点完：

```bash
python gui/app.py
```

---

## 你真正需要准备的只有三样

| | 是什么 | 从哪来 |
|---|---|---|
| 底模 | diffusers 格式的 UNet + text encoder + VAE | 你自己的 SDXL 系 checkpoint，`02-models.md` |
| 数据 | 20–60 张图 + 每张一句 caption + kohya 缓存 | 你自己的图，`03-dataset.md` |
| 显存 | 训练 ~7 GB 峰值，出图 ~6 GB | 8 GB 卡够用，不宽裕 |

仓库不含任何权重、图片或缓存 —— 这是故意的（许可证和版权），不是没写完。

---

## 各篇速览

| 文件 | 解决什么问题 | 需要多久 |
|---|---|---|
| `01-install.md` | 装依赖，确认 GPU 能用，确认 tkinter 在 | 10 分钟 |
| `02-models.md` | 单文件 checkpoint → diffusers UNet；补齐 `aux/` | 10 分钟 + 拷贝时间 |
| `03-dataset.md` | caption 怎么写；kohya 怎么吐缓存；缓存怎么命名 | 20 分钟 |
| `04-train.md` | 训练配方、时间估算、输出结构、中断续训 | 2–7 小时（跑的时候不用管） |
| `05-run.md` | 采样、扫 α、打分、怎么接 ComfyUI | 10 分钟 |
| `06-troubleshooting.md` | 报错对照表 | 查的时候看 |

---

## 两个最容易踩的坑，先说在前面

**一、caption 里写了外观。** 协议的三种 caption 变体（完整 / 触发词+场景 / 只留触发词）
就是为了逼触发词自己扛住身份。如果你在 caption 里写 `white hair`，同时又在
`--identity white hair` 里把它剥掉，那是自相矛盾；如果两边都没写，训练就学不到外观。
规则很简单：**外观只写在 `--identity` 里，caption 里不写**。

**二、`--sched euler_a --spacing trailing` 被省略。** `src/sample.py` 的默认值是
`euler` + `leading`，跟产出 `results/` 的采样器不一致；`leading` 会让每张图整体偏移。
界面 `gui/app.py` 已经固定在 `trailing`，命令行请自己写全。这条在
`05-run.md` 和 `06-troubleshooting.md` 里各出现一次，因为它值得。
