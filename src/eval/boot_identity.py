"""Paired bootstrap for the character-identity score.

`tag_eval.py` reports a mean over however many images are in a directory. A mean
with no interval cannot distinguish "GPart is 0.05 better" from "the two are the
same". This script resamples the (prompt, seed) cells that both arms share, so
the comparison is paired — each resample draws the same cells from both arms —
and reports a 95% interval for the difference plus one per prompt.

  python -m src.eval.boot_identity \
      --a "GPart=outputs/holdout/gpart_stack" \
      --b "LoRA=outputs/holdout/lora_stack" \
      --model /path/to/wd14/model.onnx

Image filenames must carry the prompt index and seed as `_NN_s<seed>`, which is
what `src/sample.py` writes. Lower `--n` for a quick look; 4000 is the number
quoted in `docs/02-results-character.md` §5.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:  # imported as part of the package
    from . import tag_eval as te
except ImportError:  # run from inside src/eval
    import tag_eval as te  # type: ignore[no-redef]

KEY = re.compile(r"_(\d{2})_s(\d+)")


def scores(d: Path, sess, idx: dict[str, int], thr: float) -> dict[tuple[str, str], float]:
    """Per-image identity score, keyed by (prompt index, seed)."""
    out: dict[tuple[str, str], float] = {}
    for p in sorted(d.glob("*.png")):
        m = KEY.search(p.name)
        if not m:
            continue
        probs = sess.run(None, {sess.get_inputs()[0].name: te.prepare(p)})[0][0]
        hits = (probs[list(idx.values())] >= thr).astype(float)
        out[(m.group(1), m.group(2))] = hits.mean()
    return out


def boot(diff: np.ndarray, n: int, rng) -> tuple[float, float, float]:
    sims = np.array([diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(n)])
    return diff.mean(), np.percentile(sims, 2.5), np.percentile(sims, 97.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="label=directory")
    ap.add_argument("--b", required=True, help="label=directory")
    ap.add_argument("--model", default=None, help="WD14 .onnx; defaults to tag_eval's")
    ap.add_argument("--tags", default=",".join(te.CHAR_TAGS))
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = Path(args.model) if args.model else te.DEFAULT_MODEL
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names, _ = te.load_names(model.parent / "selected_tags.csv")
    wanted = [t.strip() for t in args.tags.split(",") if t.strip()]
    idx = {t: names.index(t) for t in wanted if t in names}
    if not idx:
        raise SystemExit("none of the requested tags exist in the tagger's csv")

    (la, _, da), (lb, _, db) = (s.partition("=") for s in (args.a, args.b))
    A = scores(Path(da), sess, idx, args.thr)
    B = scores(Path(db), sess, idx, args.thr)
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit("the two arms share no (prompt, seed) cells — cannot pair")

    rng = np.random.default_rng(args.seed)
    a = np.array([A[k] for k in keys])
    b = np.array([B[k] for k in keys])
    d = a - b

    m, lo, hi = boot(d, args.n, rng)
    verdict = "significant" if lo > 0 or hi < 0 else "not significant (interval crosses 0)"
    print(f"n={len(keys)}  {la}={a.mean():.3f}  {lb}={b.mean():.3f}  "
          f"diff={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")

    for pr in sorted({k[0] for k in keys}):
        sel = [i for i, k in enumerate(keys) if k[0] == pr]
        dm, dlo, dhi = boot(d[sel], args.n, rng)
        v = "significant" if dlo > 0 or dhi < 0 else "-"
        print(f"  prompt {pr}: {la} {a[sel].mean():.2f}  {lb} {b[sel].mean():.2f}  "
              f"diff {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}] {v}")


if __name__ == "__main__":
    main()
