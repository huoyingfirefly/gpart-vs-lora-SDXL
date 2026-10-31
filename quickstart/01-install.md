# 01 · 装环境

目标：`python -c "import torch; torch.zeros(1).cuda()"` 不报错，`import tkinter` 不报错。
到这一步为止不需要任何模型文件。

## 1. Python

用 **3.10**（`results/` 的复现环境是 3.10.20）。3.11/3.12 大概率也能跑，但 `torch==2.11.0`
的轮子和 `transformers==5.17.0` 是按 3.10 验的，出问题先怀疑这里。

```bash
python --version
```

Windows 上如果没有 3.10，装一个再把 `py -3.10` 当解释器用；下面的 `python` 都替换成它。

## 2. 虚拟环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS / Git Bash
source .venv/bin/activate
```

`gui/launch.bat` 和 `gui/launch.sh` 都会优先用 `.venv` 里的解释器，所以建在仓库根目录
的 `.venv` 最省事。

## 3. torch 先单独装

`requirements.txt` 里写的是 `torch==2.11.0`，**默认 PyPI 那份不带 CUDA**。
如果你要 CUDA 12.8（这批数字用的构建）：

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

驱动支持的 CUDA 版本不是 12.8 就换对应的 index（`cu121`、`cu124`…），
或者直接装 PyPI 那份先用 CPU 验证流程。

## 4. 其余依赖

```bash
pip install -r requirements.txt
```

`peft` **不用装** —— GPart 的实现（以及它依赖的 PEFT）已经 vendored 在
`third_party/GPart/`，`src/sample.py` 和训练器会自己把它加进 `sys.path`。
装了也不冲突，但仓库跑的不是你装的那份。

`gradio` / `streamlit` / `flask` 都不用装。界面是 tkinter 的。

## 5. 验证

```bash
# GPU 在不在
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.11.0+cu128 True NVIDIA GeForce RTX 4060 (或你的卡)

# 显存多少（8 GB 够用，6 GB 要改配置）
nvidia-smi --query-gpu=memory.total --format=csv

# 依赖版本对得上
python -c "import diffusers, transformers, safetensors; print(diffusers.__version__, transformers.__version__, safetensors.__version__)"
# 期望：0.40.0 5.17.0 0.8.0

# 界面的 GUI 库在
python -c "import tkinter; print('tkinter ok')"
# 报 ModuleNotFoundError 说明是精简版 Python：Linux 上装 python3-tk，
# Windows 上重新安装官方 python.org 的安装包（勾选 tcl/tk）
```

## 6. 仓库能 import 就算装好了

```bash
python -c "import src.train, src.sample, src.data; print('ok')"
python -m src.prep.caption_variants --help
python scripts/train_gpart.py --help
```

三条都不报错，说明代码路径通了。**此时还不能训练** —— 没有底模、没有数据，
那正是 `02-models.md` 和 `03-dataset.md` 要解决的。

## 目录约定（后面几步都在往里放东西）

```
gpart-vs-lora-SDXL/
├── data/
│   ├── noobai_v11_unet/     ← 02：diffusers UNet（config.json + safetensors）
│   ├── noobai_v11_aux/      ← 02：text_encoder/ text_encoder_2/ vae/
│   └── 10_my_character/     ← 03：图片 + caption + kohya 缓存
├── runs/                    ← 04/05：训练产物和出图，可随时删
└── .venv/
```

`data/` 和 `runs/` 都在 `.gitignore` 里，不会被误提交。
