> English: [06-release-checklist.md](06-release-checklist.md)

# 发布检查清单

公开之前要核实的事。这里没有一项是形式主义 —— 第 1 项和第 2 项可以直接阻止发布。

## 1. 角色与画风的 IP

`aima` 和 `reserve1999` 的原图是第三方作品，角色可能属于某个权利人。

> **状态：有意不做。** 维护者的决定是照原样发布这些图。下面的条目保留下来，
> 是让复用者看清考量有哪些、自己去做判断 —— 它们不是本仓库待办的事。

- [ ] 查清 `aima` 这个角色的来源（设计、所属作品）。如果设计来自商业 / gacha
      作品，**不得发布任何**由它衍生的图像，或只发布严重降质的缩略图作示意。
- [ ] 在那一臂发布之前，对 `juxueli` 做同样的事。
- [ ] 确认 `reserve1999` 是一个画风，而不是一位艺术家 —— 否则本仓库就是在再分发
      其风格。把一个在世的艺术家点名为画风目标，本身就是声誉和授权问题 ——
      改成中性的描述词。
- [ ] **不要再分发那 21 + 76 张原图。** 它们不是我们的。仓库应该描述如何构建一个
      可比的数据集，而不是直接发一个。

如果上面有任何一项结不清，缓解办法是：发代码、协议和数字；删掉 `results/figures/`。
对比结论不依赖读者看见图，但它的说服力依赖 —— 所以要有意识地决定。

## 2. 底座模型许可证

- [ ] 读 NoobAI-XL v1.1 的 license card。SDXL 衍生模型的条款各不相同：有的禁止
      某些内容类别，有的要求署名或相同方式共享，有的限制*产出物*的商业使用。
- [ ] 决定衍生出的 adapter 权重能不能再分发。建议：**不要发 LoRA/GPart
      checkpoint。** 每个 20–45 MB，离开确切的那个底座 checkpoint 就没用，而且
      它们继承底座许可证。改发配置和配方。
- [ ] 确认发布由该模型导出的 tag 分数表没有问题（一般没问题 —— 那就是评测输出）。

## 3. 第三方代码

- [ ] `third_party/GPart/` —— 按其 `LICENSE` 是 Apache-2.0；
      `third_party/GPart/peft/LICENSE` 是上游 PEFT 的许可证。两者都与以
      Apache-2.0 发布本仓库兼容。两个许可证文件都要留在原位。
- [ ] 我们对 `third_party/` 的修改（`partition_mode` backend）必须标注出来。
      Apache-2.0 要求标记被修改过的文件。在每个打过补丁的文件上加一行 `NOTICE`
      或一段注释头，列明改动。
- [ ] `SMILINGWOLF WD-v1-4-moat-tagger-v2` —— 在任何脚本自动下载它之前，先查
      model card 的许可证。仓库应当给出下载指引，而不是把 ONNX 一起打包。
- [ ] diffusers / safetensors / PyTorch —— Apache-2.0 或 BSD；没问题，但要在
      `THIRD_PARTY.md` 里列出。

## 4. 主张复核

把 `00-claims.zh-CN.md` 从上到下对着 README 和任何公告文案读一遍。

- [ ] 任何地方都不能出现「低秩更好」而不带「在这个实验里」的限定。
- [ ] 任何地方都不能出现「GPart 慢 4 倍」而不带「在我们的 stateless backend 下」。
      成本主张必须点名实现。
- [ ] 任何地方都不能出现「GPart 不能组合」—— 组合测试只做了加法。
- [ ] α 扫描的注意事项要挂在 0.57 这个数字旁边：这个峰是用偏色和细节损失换来的。
- [ ] 「只有一个角色」的限制要出现在 headline 表旁边，而不是只写在
      `05-limitations.zh-CN.md` 里。

## 5. 可复现性

- [ ] 移除或搬走剩下的硬编码路径：
      - `src/train_sdxl.py:33` —— `--data` 的默认值
      - `src/prep_caption_variants.py:73-74` —— `--src` / `--dst` 的默认值
      - `src/style_stats.py:8` —— docstring 里的示例路径（装饰性）
      - `src/sample.py:4-6` —— docstring 里的示例路径（装饰性）
      - `scripts/tag_eval.py:28` —— WD14 ONNX 路径；改成 `--model` 参数
- [ ] `requirements.txt` 钉住的版本确实能一起工作
      （torch / diffusers / transformers / peft / onnxruntime / pillow）。
- [ ] `configs/` 里每一个 config 都至少从发布树跑过一次。
- [ ] `scripts/reproduce*.sh` 能在干净机器上端到端跑通，或者被明确标注为未验证。
- [ ] 写明计时来自哪块 GPU、多少显存。没有硬件的计时没有意义。

## 6. 仓库整洁

- [ ] `.gitignore` 覆盖 `outputs/`、`runs/`、`*.safetensors`、`*.pt`、`*.onnx`、
      latent 缓存、`__pycache__`、编辑器垃圾。
- [ ] 提交的文件里没有 API key、HF token、ComfyUI 主机名或 Windows 绝对路径。
- [ ] `LICENSE` 存在，且与 `THIRD_PARTY.md` 一致。
- [ ] `README.md` 的快速上手，能在不是写它的那台机器上产出东西。
- [ ] 大文件：没有超过约 10 MB 的文件。有更大的图就降采样。

## 7. 叙事框架

这个仓库值得发布的原因不是对 GPart 的裁决，而是：

1. 一套用于 PEFT 对比的对齐配方协议，容易移植到其他方法，并且把造出 33× 位移差
   的那几个具体混淆因子写明了。
2. 一个能塞进 8 GB 的实现的实测成本模型，包括「隐式 partition 的 hash *不是*
   瓶颈」这个发现。
3. 一个被当作阴性结果来陈述的阴性结果。

用这三条开场。不要用「LoRA 打赢了 GPart」开场。
