"""GPart Studio — a two-tab desktop UI: train an adapter, then sample with it.

    python gui/app.py

Zero extra dependencies: tkinter, torch and diffusers are already required by the
rest of the repo. Everything runs locally; nothing is uploaded anywhere.

Tab 1 · TRAIN wraps scripts/train_gpart.py (audit -> caption variants -> train ->
manifest) and streams its log, so the GUI and the CLI cannot drift apart.
Tab 2 · SAMPLE loads a trained adapter, merges it into the UNet, and generates
images with the exact sampler from docs/01-protocol.md (euler ancestral,
trailing spacing).

The UNet is never loaded at startup: on an 8 GiB card it would sit there
occupying 5 GiB while you are only reading the training log.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

# same convention as WD14_MODEL in src/eval/tag_eval.py: an env var if you keep
# the weights outside the repo, otherwise the documented repo-relative default
DEFAULT_UNET = Path(os.environ.get("GPART_UNET", ROOT / "data" / "noobai_v11_unet"))
DEFAULT_AUX = Path(os.environ.get("GPART_AUX", ROOT / "data" / "noobai_v11_aux"))
RUNS_DIR = ROOT / "runs"
TRAINER = ROOT / "scripts" / "train_gpart.py"

SIZES = ["832x1216", "1216x832", "1024x1024", "1152x896", "896x1152",
         "832x832", "768x1152", "512x512"]
# must stay char-for-char equal to src/sample.py's --negative default: the
# negative prompt is part of the sampling protocol, not a matter of taste
DEFAULT_NEG = "low quality, worst quality, blurry, watermark, text"

BG = "#1e1e1e"
FG = "#e6e6e6"
ACCENT = "#0e639c"
STEP_RE = re.compile(r"^\s*\{")


# --------------------------------------------------------------------- helpers
def human_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def count_samples(path: Path) -> dict:
    """What is actually in a dataset dir. Cheap: globs only, no loading."""
    if not path.is_dir():
        return {"exists": False}
    return {
        "exists": True,
        "latents": len(list(path.glob("*_sdxl.npz"))),
        "te": len(list(path.glob("*_te_outputs.npz"))),
        "captions": len(list(path.glob("*.txt"))),
    }


def discover_adapters() -> list[tuple[str, Path]]:
    """Every PeftModel directory under runs/ and outputs/."""
    found: list[tuple[float, str, Path]] = []
    for base in (RUNS_DIR, ROOT / "outputs"):
        if not base.exists():
            continue
        for cfg in base.rglob("adapter_config.json"):
            path = cfg.parent
            label = str(path.relative_to(ROOT)).replace("\\", "/")
            found.append((path.stat().st_mtime, label, path))
    found.sort(reverse=True)
    return [(label, path) for _, label, path in found]


# ---------------------------------------------------------------------- engine
class Engine:
    """UNet + VAE + text encoders, with adapter swap-in/out.

    Imported lazily so the training tab works on a machine where CUDA or the
    base model is not set up yet.
    """

    def __init__(self) -> None:
        self.device = "cuda"
        self.dtype = None
        self.unet = None
        self.model = None
        self.vae = None
        self.enc = None
        self.cached = None
        self.unet_dir: Path | None = None
        self.aux_dir: Path | None = None
        # sentinel, not None: `None` is a real key meaning "no adapter", and a
        # fresh engine must not answer "already loaded" to set_adapter(None)
        self.adapter_key: str | None = "<unloaded>"
        self.ready = False
        self.cond_cache: dict = {}

    def _torch(self):
        import torch

        return torch

    def load(self, unet_dir: Path, aux_dir: Path, cache_dir: Path | None) -> None:
        """Everything except the UNet — that one is built by set_adapter() so the
        first generation does not load a bare UNet and then immediately throw it
        away to load the same UNet again with an adapter on it."""
        torch = self._torch()
        from diffusers import AutoencoderKL

        from src.sample import CachedEmbeds, Encoder

        self.dtype = torch.bfloat16
        self.unet_dir, self.aux_dir = unet_dir, aux_dir
        self.vae = AutoencoderKL.from_pretrained(
            str(aux_dir / "vae"), torch_dtype=torch.float32).eval()
        # the text towers stay on CPU: fp32, ~3.3 GiB, and they run once per prompt
        self.enc = Encoder(aux_dir, "cpu")
        self.cached = (
            CachedEmbeds(cache_dir)
            if cache_dir and cache_dir.is_dir() else None
        )
        self.adapter_key = "<unloaded>"
        self.ready = True

    def unload(self) -> None:
        torch = self._torch()
        self.model = self.unet = self.vae = self.enc = self.cached = None
        self.adapter_key = "<unloaded>"
        self.ready = False
        self.cond_cache.clear()
        torch.cuda.empty_cache()

    def set_adapter(self, path: Path | None, alpha: float = 1.0) -> None:
        key = f"{path}@{alpha}" if path else None
        if key == self.adapter_key and self.unet is not None:
            return
        torch = self._torch()
        import src.sample  # noqa: F401  -- puts third_party/GPart on sys.path
        from peft import PeftModel

        self.model = self.unet = None
        torch.cuda.empty_cache()
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(
            str(self.unet_dir), torch_dtype=self.dtype, low_cpu_mem_usage=True,
            device_map=self.device)
        unet.eval().requires_grad_(False)
        if path is not None:
            wrapped = PeftModel.from_pretrained(unet, str(path))
            # theta reloads on CPU; without this .to(device) the merge round-trips
            # the whole adapter over PCIe on every denoise step
            wrapped = wrapped.to(self.device).eval()
            if alpha != 1.0:
                n = 0
                for name, param in wrapped.named_parameters():
                    if "gpart_theta_d" in name:
                        param.data.mul_(alpha)
                        n += 1
                if n == 0:
                    raise RuntimeError(
                        "no gpart_theta_d tensors found -- is this a GPart adapter? "
                        "For a kohya LoRA use `python -m src.sample --lora`.")
            unet = wrapped.merge_and_unload().eval()
        self.unet = self.model = unet
        self.adapter_key = key
        # the text embeddings do not depend on the adapter, so keep cond_cache

    def warm(self, *texts: str) -> None:
        """Encode texts on CUDA and cache them on the host.

        Encoding on the CPU instead is tempting (the towers are 3.3 GiB fp32 and
        do not have to share the card with the UNet) but it changes the output:
        the CUDA path runs the CLIP matmuls in TF32, and hidden states differ by
        up to ~1.2e-1 absolute, which visibly reshuffles a 4-step image. Call
        this BEFORE set_adapter() so the UNet is not resident yet.
        """
        for text in texts:
            if text not in self.cond_cache:
                self._encode_now(text)

    def _encode_now(self, text: str):
        torch = self._torch()
        hit = self.cached.get(text) if self.cached else None
        if hit is None:
            park = self.model is not None and torch.cuda.is_available()
            if park:
                self.model.to("cpu")
                torch.cuda.empty_cache()
            prev = self.enc.device
            try:
                self.enc.te1.to(self.device)
                self.enc.te2.to(self.device)
                self.enc.device = self.device
                hit = self.enc.encode(text)
            finally:
                self.enc.to_cpu()
                self.enc.device = prev
                if park:
                    self.model.to(self.device)
        # keep the cache on the host: it is a few MB and survives adapter swaps
        self.cond_cache[text] = (hit[0].cpu(), hit[1].cpu())
        return self.cond_cache[text]

    def encode(self, text: str):
        hit = self.cond_cache.get(text)
        if hit is None:
            hit = self._encode_now(text)
        return (hit[0].to(self.device, self.dtype), hit[1].to(self.device, self.dtype))

    def generate(self, prompt, negative, w, h, steps, cfg, seed, alpha=1.0,
                 on_step=None, cancel=None) -> "object":
        torch = self._torch()
        from diffusers import EulerAncestralDiscreteScheduler

        pos_embeds, pos_pool = self.encode(prompt)
        neg_embeds, neg_pool = self.encode(negative)
        time_ids = torch.tensor([[h, w, 0, 0, h, w]], device=self.device, dtype=self.dtype)

        # identical to src/sample.py --sched euler_a --spacing trailing, which is
        # what produced every number in results/
        scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear", prediction_type="epsilon",
            timestep_spacing="trailing", steps_offset=1)
        scheduler.set_timesteps(steps)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn((1, 4, h // 8, w // 8), generator=generator,
                              device=self.device, dtype=self.dtype)
        latents = latents * scheduler.init_noise_sigma

        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps):
                if cancel is not None and cancel.is_set():
                    raise _Cancelled
                inp = scheduler.scale_model_input(latents, t)
                uncond = self.model(inp, t, encoder_hidden_states=neg_embeds,
                                    added_cond_kwargs={"text_embeds": neg_pool,
                                                       "time_ids": time_ids}).sample
                text = self.model(inp, t, encoder_hidden_states=pos_embeds,
                                  added_cond_kwargs={"text_embeds": pos_pool,
                                                     "time_ids": time_ids}).sample
                noise = uncond + cfg * (text - uncond)
                latents = scheduler.step(noise, t, latents, generator=generator).prev_sample
                if on_step is not None:
                    on_step(i + 1, steps)
        return latents

    def decode(self, latents):
        torch = self._torch()
        import numpy as np
        from PIL import Image

        scaled = latents.to(self.device).float() / self.vae.config.scaling_factor
        # fp32 VAE decode of a 1216px image peaks at 2-3 GiB on top of the 5 GiB
        # UNet, so park the UNet on the host when the card is already tight
        moved = False
        free, _ = torch.cuda.mem_get_info()
        if free < 3.5 * 2**30:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            moved = True
        try:
            self.vae.to(self.device)
            with torch.no_grad():
                image = self.vae.decode(scaled).sample
        finally:
            self.vae.to("cpu")
            del scaled
            torch.cuda.empty_cache()
            if moved:
                self.model.to(self.device)
        arr = (image / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        del image
        return Image.fromarray((arr * 255).round().astype(np.uint8))


class _Cancelled(Exception):
    pass


# ------------------------------------------------------------------------- app
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GPart Studio")
        self.geometry("1240x860")
        self.configure(bg=BG)

        self.events: queue.Queue = queue.Queue()
        self.train_proc: subprocess.Popen | None = None
        self.engine = Engine()
        self.engine_busy = False
        self.cancel = threading.Event()
        self.thumbs: list = []
        self.last_image = None
        self.train_out: Path | None = None
        self.train_total = 1
        self._last_step = 0

        self._style()
        self._build()
        self.after(80, self._drain)

    # ------------------------------------------------------------------ chrome
    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        for name, opts in {
            ".": dict(background=BG, foreground=FG, fieldbackground="#2b2b2b"),
            "TFrame": dict(background=BG),
            "TLabel": dict(background=BG, foreground=FG),
            "TLabelframe": dict(background=BG, foreground="#9cdcfe"),
            "TLabelframe.Label": dict(background=BG, foreground="#9cdcfe"),
            "TNotebook": dict(background="#252526", borderwidth=0),
            "TNotebook.Tab": dict(background="#2d2d2d", foreground=FG, padding=(16, 7)),
            "TButton": dict(background="#333333", foreground=FG),
            "Accent.TButton": dict(background=ACCENT, foreground="white"),
            "TCheckbutton": dict(background=BG, foreground=FG),
        }.items():
            style.configure(name, **opts)
        style.map("TNotebook.Tab", background=[("selected", BG)])
        style.map("Accent.TButton", background=[("active", "#1177bb")])
        style.map("TButton", background=[("active", "#3f3f3f")])

    def _build(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.train_tab = ttk.Frame(nb, padding=10)
        self.sample_tab = ttk.Frame(nb, padding=10)
        nb.add(self.train_tab, text="  ① 训练  ")
        nb.add(self.sample_tab, text="  ② 出图  ")
        self._build_train(self.train_tab)
        self._build_sample(self.sample_tab)
        self.nb = nb
        # a hidden notebook page is never laid out (its canvas stays 1x1), so
        # redraw on switch instead of relying on <Configure>
        nb.bind("<<NotebookTabChanged>>", lambda e: self._redraw())

    # ================================================================ TAB 1
    def _build_train(self, root) -> None:
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        # --- dataset
        box = ttk.Labelframe(left, text="数据集（kohya 缓存目录）", padding=8)
        box.pack(fill="x", pady=(0, 8))
        self.data_var = tk.StringVar(value="")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.data_var, width=34).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3, command=self._pick_data).pack(side="left", padx=(4, 0))
        ttk.Button(box, text="检查这个数据集", command=self._check_dataset).pack(fill="x", pady=(6, 0))
        self.data_status = ttk.Label(box, text="未检查", wraplength=300, foreground="#c8c8c8")
        self.data_status.pack(fill="x", pady=(4, 0))

        # --- models
        box = ttk.Labelframe(left, text="底模", padding=8)
        box.pack(fill="x", pady=(0, 8))
        self.unet_var = tk.StringVar(value=str(DEFAULT_UNET))
        self.aux_var = tk.StringVar(value=str(DEFAULT_AUX))
        for label, var in (("UNet", self.unet_var), ("TE / VAE", self.aux_var)):
            row = ttk.Frame(box)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=9).pack(side="left")
            ttk.Entry(row, textvariable=var, width=26).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="…", width=3,
                       command=lambda v=var: self._pick_dir(v)).pack(side="left", padx=(4, 0))

        # --- recipe
        box = ttk.Labelframe(left, text="训练配置", padding=8)
        box.pack(fill="x", pady=(0, 8))
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        def add(row, label, widget, hint=None):
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", pady=3)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            if hint:
                ttk.Label(grid, text=hint, foreground="#8a8a8a",
                          font=("", 8)).grid(row=row, column=2, sticky="w", padx=(6, 0))

        self.name_var = tk.StringVar(value="my_character")
        add(0, "输出名", ttk.Entry(grid, textvariable=self.name_var, width=22))
        self.trigger_var = tk.StringVar(value="mychar")
        add(1, "触发词", ttk.Entry(grid, textvariable=self.trigger_var, width=22),
            "写进 caption")
        self.preset_var = tk.StringVar(value="attn2")
        add(2, "覆盖", ttk.Combobox(grid, textvariable=self.preset_var, state="readonly",
                                   values=["attn2", "alllinear"], width=20),
            "attn2 快 2.5 倍")
        self.steps_var = tk.IntVar(value=1200)
        add(3, "步数", ttk.Spinbox(grid, from_=50, to=5000, increment=100,
                                  textvariable=self.steps_var, width=20),
            "800 起配件才全")
        self.d_var = tk.IntVar(value=23224320)
        add(4, "d", ttk.Entry(grid, textvariable=self.d_var, width=22),
            "23.2 M")
        self.seed_var = tk.IntVar(value=42)
        add(5, "种子", ttk.Spinbox(grid, from_=0, to=999999, textvariable=self.seed_var,
                                  width=20))

        ttk.Label(box, text="要剥掉的外观标签（逗号分隔）").pack(anchor="w", pady=(8, 2))
        self.identity = tk.Text(box, height=4, width=40, wrap="word", bg="#2b2b2b",
                                fg=FG, insertbackground=FG)
        self.identity.pack(fill="x")
        ttk.Label(box, text="这些标签会从 scene / bare 变体里删掉，强迫触发词自己承载外观。",
                  foreground="#8a8a8a", font=("", 8), wraplength=300).pack(anchor="w", pady=(2, 0))

        self.reuse_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="复用已生成的变体（跳过重新编码）",
                        variable=self.reuse_var).pack(anchor="w", pady=(6, 0))

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(4, 6))
        self.train_btn = ttk.Button(row, text="开始训练", style="Accent.TButton",
                                    command=self.on_train)
        self.train_btn.pack(side="left", fill="x", expand=True, ipady=6)
        self.stop_train_btn = ttk.Button(row, text="停止", state="disabled",
                                         command=self.on_stop_train)
        self.stop_train_btn.pack(side="left", padx=(6, 0), ipady=6)

        # --- right: log
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        bar = ttk.Frame(right)
        bar.pack(fill="x")
        self.train_progress = ttk.Progressbar(bar, mode="determinate", maximum=100)
        self.train_progress.pack(side="left", fill="x", expand=True)
        self.train_status = ttk.Label(bar, text="空闲", width=34, anchor="w")
        self.train_status.pack(side="left", padx=(8, 0))

        self.log = tk.Text(right, bg="#111111", fg="#d4d4d4", insertbackground=FG,
                           wrap="none", font=("Consolas", 9), height=20)
        sb = ttk.Scrollbar(right, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        self.log.tag_configure("red", foreground="#f48771")
        self.log.tag_configure("green", foreground="#89d185")
        self.log.tag_configure("blue", foreground="#9cdcfe")
        self.log.configure(state="disabled")

    def _pick_data(self) -> None:
        got = filedialog.askdirectory(title="选择 kohya 缓存目录（含 *_sdxl.npz）")
        if got:
            self.data_var.set(got)
            self._check_dataset()

    def _pick_dir(self, var) -> None:
        got = filedialog.askdirectory()
        if got:
            var.set(got)

    def _check_dataset(self) -> None:
        path = Path(self.data_var.get().strip())
        info = count_samples(path)
        if not info.get("exists"):
            self.data_status.configure(text="目录不存在", foreground="#f48771")
            return
        ok = info["latents"] and info["te"] and info["captions"]
        color = "#89d185" if ok else "#f48771"
        text = (f"latent {info['latents']} · TE {info['te']} · caption {info['captions']}")
        if not info["latents"]:
            text += "\n缺少 *_sdxl.npz —— 这不是缓存目录"
        elif info["latents"] != info["te"]:
            text += "\nlatent 与 TE 数量不一致，训练会跳过缺 TE 的样本"
        self.data_status.configure(text=text, foreground=color)
        self._load_presets(path)
        return None

    def on_train(self) -> None:
        if self.train_proc is not None:
            return
        data = Path(self.data_var.get().strip())
        info = count_samples(data)
        if not info.get("exists") or not info.get("latents"):
            messagebox.showerror("数据集不对",
                                 "请先选一个含 *_sdxl.npz 的 kohya 缓存目录。")
            return
        identity = self.identity.get("1.0", "end").strip().replace("\n", ",")
        if not identity:
            messagebox.showerror(
                "缺外观标签",
                "必须填要剥掉的外观标签。\n\n"
                "留空时训练脚本会回退到 aima 那套标签，换角色就会静默出错。")
            return
        if not TRAINER.exists():
            messagebox.showerror("找不到训练脚本", str(TRAINER))
            return

        cmd = [sys.executable, str(TRAINER),
               "--data", str(data),
               "--name", self.name_var.get().strip(),
               "--trigger", self.trigger_var.get().strip(),
               "--identity", identity,
               "--preset", self.preset_var.get(),
               "--steps", str(int(self.steps_var.get())),
               "--d", str(int(self.d_var.get())),
               "--seed", str(int(self.seed_var.get())),
               "--unet", self.unet_var.get().strip(),
               "--aux", self.aux_var.get().strip(),
               "--out-root", str(RUNS_DIR)]
        if self.reuse_var.get():
            cmd.append("--reuse-variants")

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.train_total = max(1, int(self.steps_var.get()))
        self.train_out = RUNS_DIR / self.name_var.get().strip()
        self._last_step = 0
        # a status.json left over from an earlier run with the same name would
        # drive the bar before this run has written anything
        stale = self.train_out / "status.json"
        if stale.exists():
            stale.unlink()
        self.train_progress.configure(value=0, maximum=self.train_total)
        self.train_btn["state"] = "disabled"
        self.stop_train_btn["state"] = "normal"
        self.train_status.configure(text="启动中…")
        self._append(f"$ {' '.join(cmd)}\n", color="#9cdcfe")

        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.train_proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace", **kwargs)
        except Exception:
            self.train_proc = None
            self.train_btn["state"] = "normal"
            self.stop_train_btn["state"] = "disabled"
            messagebox.showerror("启动失败", traceback.format_exc())
            return
        threading.Thread(target=self._pump_train, daemon=True).start()

    def _pump_train(self) -> None:
        proc = self.train_proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            self.events.put(("train_log", line.rstrip("\n")))
        code = proc.wait()
        self.events.put(("train_done", code))

    def on_stop_train(self) -> None:
        if self.train_proc is not None:
            self.train_proc.terminate()
            self.train_status.configure(text="正在终止…")

    def _append(self, text: str, color: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", (color,) if color else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    # ================================================================ TAB 2
    def _build_sample(self, root) -> None:
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        box = ttk.Labelframe(left, text="适配器", padding=8)
        box.pack(fill="x", pady=(0, 8))
        self.adapter_var = tk.StringVar(value="base（原模型）")
        self.adapter_box = ttk.Combobox(box, textvariable=self.adapter_var,
                                        state="readonly", width=36)
        self.adapter_box.pack(fill="x")
        self.adapter_hint = ttk.Label(box, text="", foreground="#8a8a8a",
                                      font=("", 8), wraplength=300)
        self.adapter_hint.pack(fill="x", pady=(2, 0))
        ttk.Button(box, text="刷新列表", command=self._refresh_adapters).pack(fill="x", pady=(6, 0))
        self._refresh_adapters()

        box = ttk.Labelframe(left, text="提示词", padding=8)
        box.pack(fill="x", pady=(0, 8))
        self.prompt = tk.Text(box, height=6, width=40, wrap="word", bg="#2b2b2b",
                              fg=FG, insertbackground=FG)
        self.prompt.pack(fill="x")
        self.prompt.insert("1.0", "mychar, 1girl, solo, upper body, looking at viewer")
        self.preset_box = ttk.Combobox(box, state="readonly", width=38)
        self.preset_box.pack(fill="x", pady=(6, 0))
        self.preset_box.bind("<<ComboboxSelected>>", self._apply_preset)

        box = ttk.Labelframe(left, text="负面提示词", padding=8)
        box.pack(fill="x", pady=(0, 8))
        self.negative = tk.StringVar(value=DEFAULT_NEG)
        ttk.Entry(box, textvariable=self.negative, width=42).pack(fill="x")

        box = ttk.Labelframe(left, text="参数", padding=8)
        box.pack(fill="x", pady=(0, 8))
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        self.size_var = tk.StringVar(value=SIZES[0])
        self.ssteps_var = tk.IntVar(value=30)
        self.cfg_var = tk.DoubleVar(value=5.0)
        self.sseed_var = tk.StringVar(value="1")
        self.count_var = tk.IntVar(value=1)
        self.alpha_var = tk.DoubleVar(value=1.0)
        rows = [
            ("尺寸", ttk.Combobox(grid, textvariable=self.size_var, values=SIZES,
                                  state="readonly", width=12)),
            ("步数", ttk.Spinbox(grid, from_=1, to=60, textvariable=self.ssteps_var, width=12)),
            ("CFG", ttk.Spinbox(grid, from_=1.0, to=12.0, increment=0.5,
                                textvariable=self.cfg_var, width=12)),
            ("α（θ 缩放）", ttk.Spinbox(grid, from_=0.3, to=2.5, increment=0.1,
                                      textvariable=self.alpha_var, width=12)),
            ("种子", ttk.Entry(grid, textvariable=self.sseed_var, width=13)),
            ("张数", ttk.Spinbox(grid, from_=1, to=12, textvariable=self.count_var, width=12)),
        ]
        for i, (label, widget) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", pady=3)
            widget.grid(row=i, column=1, sticky="w", pady=3)
        ttk.Label(box, text="α 是采样强度，不是训练的 --gpart-alpha。\n"
                            "在 aima 上 α=1.0 得 0.40，α=1.5 得 0.57 —— 要扫。",
                  foreground="#8a8a8a", font=("", 8), wraplength=300).pack(anchor="w", pady=(4, 0))

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(4, 6))
        self.go_btn = ttk.Button(row, text="生成", style="Accent.TButton",
                                 command=self.on_generate)
        self.go_btn.pack(side="left", fill="x", expand=True, ipady=6)
        self.stop_btn = ttk.Button(row, text="停止", state="disabled", command=self.on_stop)
        self.stop_btn.pack(side="left", padx=(6, 0), ipady=6)

        self.sprogress = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.sprogress.pack(fill="x")
        self.sstatus = ttk.Label(left, text="引擎未加载（首次生成时加载，约 1 分钟）",
                                 wraplength=320)
        self.sstatus.pack(fill="x", pady=(6, 0))

        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        self.canvas = tk.Canvas(right, bg="#111111", highlightthickness=0,
                                width=760, height=700)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.thumb_frame = ttk.Frame(right)
        self.thumb_frame.pack(fill="x", pady=(8, 0))
        self._load_presets(None)

    def _refresh_adapters(self) -> None:
        self.adapters = discover_adapters()
        values = ["base（原模型）"] + [label for label, _ in self.adapters]
        self.adapter_box["values"] = values
        if self.adapter_var.get() not in values:
            self.adapter_var.set(values[0])
        n = len(self.adapters)
        self.adapter_hint.configure(
            text=f"在 runs/ 和 outputs/ 下找到 {n} 个适配器" if n
            else "还没找到适配器 —— 先训一个，或把 adapter 目录放进 runs/",
            foreground="#8a8a8a" if n else "#f48771")

    def _apply_preset(self, _e=None) -> None:
        texts = getattr(self, "_presets", [])
        idx = self.preset_box.current() - 1
        if 0 <= idx < len(texts):
            self.prompt.delete("1.0", "end")
            self.prompt.insert("1.0", texts[idx])

    def _load_presets(self, data: Path | None) -> None:
        texts: list[str] = []
        for f in (ROOT / "configs" / "prompts").glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "prompt" in obj:
                        texts.append(obj["prompt"])
        if data and (data.parent / (data.name + "_var")).is_dir():
            src = data.parent / (data.name + "_var")
        else:
            src = data
        if src and src.is_dir():
            for txt in sorted(src.glob("*.txt"))[:20]:
                texts.append(txt.read_text(encoding="utf-8").strip())
        seen, uniq = set(), []
        for t in texts:
            k = t.strip()
            if k and k not in seen:
                seen.add(k)
                uniq.append(k)
        self._presets = uniq
        self.preset_box["values"] = ["— 选一个现成 prompt —"] + [
            t[:64] + ("…" if len(t) > 64 else "") for t in uniq]
        if uniq:
            self.preset_box.current(0)

    # ------------------------------------------------------------ generation
    def on_generate(self) -> None:
        if self.engine_busy or self.train_proc is not None:
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("提示", "请先输入提示词")
            return
        label = self.adapter_var.get()
        adapter = None if label.startswith("base") else dict(self.adapters)[label]
        try:
            w, h = (int(v) for v in self.size_var.get().split("x"))
            steps = max(1, int(self.ssteps_var.get()))
            cfg = float(self.cfg_var.get())
            count = max(1, int(self.count_var.get()))
            alpha = float(self.alpha_var.get())
            base_seed = int(self.sseed_var.get())
        except Exception:
            messagebox.showerror("参数错误", "步数 / CFG / 种子 需要是数字")
            return

        self.engine_busy = True
        self.cancel.clear()
        self.go_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"
        self.sstatus.configure(text="准备中…")
        # every tkinter read happens here, in the main thread; the worker only
        # gets plain Python values
        unet_dir = Path(self.unet_var.get())
        aux_dir = Path(self.aux_var.get())
        data_dir = Path(self.data_var.get()) if self.data_var.get().strip() else None
        threading.Thread(target=self._gen_worker, daemon=True, args=(
            adapter, label, prompt, self.negative.get(), w, h, steps, cfg,
            [base_seed + i for i in range(count)], alpha,
            unet_dir, aux_dir, data_dir)).start()

    def on_stop(self) -> None:
        self.cancel.set()
        self.sstatus.configure(text="正在停止（当前这一步之后中断）…")

    def _gen_worker(self, adapter, label, prompt, negative, w, h, steps, cfg,
                    seeds, alpha, unet_dir, aux_dir, data_dir) -> None:
        try:
            if not self.engine.ready:
                self.events.put(("sstatus", "首次加载文本编码器 / VAE，约 20 秒…"))
                self.engine.load(unet_dir, aux_dir, data_dir)
            # encode first, while the card is free -- this is also what keeps the
            # result identical to src/sample.py (see Engine.warm)
            self.engine.warm(prompt, negative)
            self.engine.set_adapter(adapter, alpha)
            self.events.put(("sstatus", f"已加载 {label} (α={alpha})"))

            out_dir = RUNS_DIR / (Path(adapter).parent.name if adapter else "base") / "samples"
            out_dir.mkdir(parents=True, exist_ok=True)
            for n, seed in enumerate(seeds, 1):
                started = time.time()
                self.events.put(("sstatus", f"[{n}/{len(seeds)}] 采样 seed={seed}"))
                latents = self.engine.generate(
                    prompt, negative, w, h, steps, cfg, seed, alpha,
                    on_step=lambda i, t: self.events.put(("sprogress", (i, t))),
                    cancel=self.cancel)
                image = self.engine.decode(latents)
                del latents
                # keep the run name too: `runs/*/adapter` would otherwise all be
                # saved as "adapter_<seed>.png"
                stem = "_".join(Path(adapter).parts[-2:]) if adapter else "base"
                path = out_dir / f"{stem}_{seed}_{w}x{h}.png"
                image.save(path)
                self.events.put(("image", (image, path, time.time() - started)))
            self.events.put(("gen_done", None))
        except _Cancelled:
            self.events.put(("gen_done", "已中断"))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    # --------------------------------------------------------------- plumbing
    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        if self.train_proc is not None:
            self._poll_status()
        self.after(80, self._drain)

    def _poll_status(self) -> None:
        """src/train.py rewrites status.json every step; train.jsonl only every
        --log-every. Reading the former keeps the bar moving at step resolution."""
        now = time.monotonic()
        if now - getattr(self, "_last_poll", 0.0) < 1.0:
            return
        self._last_poll = now
        if self.train_out is None:
            return
        path = self.train_out / "status.json"
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        step = int(rec.get("step") or 0)
        if step <= self._last_step:
            return
        self._last_step = step
        self.train_progress.configure(value=min(step, self.train_total),
                                      maximum=self.train_total)
        sec = float(rec.get("sec") or 0)
        eta = f" · ETA {human_time((self.train_total - step) * sec / step)}" if step and sec else ""
        loss = rec.get("loss")
        peak = rec.get("peak_gib")
        head = f"step {step}/{self.train_total}"
        if isinstance(loss, float):
            head += f" · loss {loss:.4f}"
        if isinstance(peak, float):
            head += f" · {peak:.1f}G"
        self.train_status.configure(text=head + eta)

    def _handle(self, kind, payload) -> None:
        if kind == "train_log":
            self._on_train_log(payload)
        elif kind == "train_done":
            self._finish_train(payload)
        elif kind == "sstatus":
            self.sstatus.configure(text=payload)
        elif kind == "sprogress":
            done, total = payload
            self.sprogress.configure(value=100 * done / total)
        elif kind == "image":
            image, path, took = payload
            self.last_image = image
            self._add_thumb(image)
            self._redraw()
            self.sstatus.configure(text=f"完成 {path.name} · {took:.1f}s")
        elif kind == "gen_done":
            self._finish_gen(payload)
        elif kind == "error":
            messagebox.showerror("出错", payload[-1500:])
            self._finish_gen("出错")

    def _on_train_log(self, line: str) -> None:
        color = None
        if line.startswith(("[FAIL]", "[warn]")):
            color = "red"
        elif "saved adapter" in line:
            color = "green"
        elif STEP_RE.match(line):
            color = "blue"
            try:
                rec = json.loads(line)
            except Exception:
                rec = None
            if rec and "step" in rec:
                step = int(rec["step"])
                self.train_progress.configure(value=step)
                # `sec` is cumulative wall clock since step 0, not a per-step delta
                sec = float(rec.get("sec") or 0)
                eta = ""
                if step and sec:
                    eta = f" · ETA {human_time((self.train_total - step) * sec / step)}"
                loss = rec.get("loss")
                self.train_status.configure(
                    text=f"step {step}/{self.train_total} · loss {loss:.4f}{eta}"
                    if isinstance(loss, float)
                    else f"step {step}/{self.train_total}{eta}")
        self._append(line, color)

    def _finish_train(self, code: int) -> None:
        self.train_proc = None
        self.train_btn["state"] = "normal"
        self.stop_train_btn["state"] = "disabled"
        if code == 0:
            self.train_status.configure(text="训练完成")
            self._append("训练完成 —— 到「② 出图」刷新适配器列表。", "green")
            self._refresh_adapters()
        else:
            self.train_status.configure(text=f"训练结束，退出码 {code}")
            self._append(f"进程退出码 {code}。checkpoint 每 save_every 步落盘一次，"
                         f"可以从最近的一个 --resume。", "red")

    def _finish_gen(self, note) -> None:
        self.engine_busy = False
        self.go_btn["state"] = "normal"
        self.stop_btn["state"] = "disabled"
        self.sprogress.configure(value=0)
        if note:
            self.sstatus.configure(text=note)

    def _add_thumb(self, image) -> None:
        from PIL import ImageTk

        thumb = image.copy()
        thumb.thumbnail((110, 110))
        tkimg = ImageTk.PhotoImage(thumb)
        self.thumbs.append(tkimg)
        lbl = ttk.Label(self.thumb_frame, image=tkimg)
        lbl.pack(side="left", padx=2)
        lbl.bind("<Button-1>", lambda e, im=image: self._show(im))
        while len(self.thumbs) > 8:
            self.thumbs.pop(0)
            self.thumb_frame.winfo_children()[0].destroy()

    def _show(self, image) -> None:
        self.last_image = image
        self._redraw()

    def _redraw(self) -> None:
        from PIL import ImageTk

        self.canvas.delete("all")
        if self.last_image is None:
            self.canvas.create_text(380, 320, text="还没有图", fill="#555555",
                                    font=("", 14))
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        w, h = self.last_image.size
        scale = min(cw / w, ch / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        shown = self.last_image.resize(size)
        self._photo = ImageTk.PhotoImage(shown)
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)


def main() -> None:
    if not TRAINER.exists():
        print(f"[warn] {TRAINER} missing -- the train tab will refuse to start")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
