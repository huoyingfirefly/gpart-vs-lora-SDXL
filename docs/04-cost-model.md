# Cost model

The measured cost of running GPart in this repo, and why the number is larger
than a reader might expect from the method's description.

**Read this first:** everything below describes **this repository's `stateless`
backend**, written to fit in 8 GB of VRAM. It is not a claim about GPart as
published. The upstream implementation materializes the assignment table, which
we could not fit; the numbers would be different there.

## 1. Why a `stateless` backend

GPart's ΔW is `P · diag(s) · θ` where `P ∈ {0,1}^{N×d}` assigns each of the `N`
covered weights to one of `d` groups. Materializing `P` for a 700-module,
2184 M-weight target set:

```
2184e6 × 23224320 × 1 bit ≈ 6.3 TB   (dense)
2184e6            × 4 bytes ≈ 8.7 GB  (per-weight int32 group id)
```

Neither fits next to an SDXL UNet in 8 GB. So this repo recomputes the group
assignment from a hash on the fly: `P` is *implicit*, defined by a deterministic
seeded mixer over the flat parameter index, and regenerated in every forward and
backward pass.

That choice is what makes training slow.

## 2. Measured per-step cost

One optimizer step = forward + backward + optimizer, effective batch 2,
832×1216-equivalent buckets, bf16, RTX-class 8 GB card.

| adapter | covered weights | s/step | vs LoRA |
|---|---|---|---|
| LoRA (all-linear, r8) | 2184.0 M | **4.85** | 1.0× |
| GPart attn2 | 545.6 M | **8.11** | 1.7× |
| GPart all-linear | 2184.0 M | **20.5** | **4.2×** |

Two structural facts, both unexpected relative to the design intuition:

**(a) Cost tracks covered weights, not `d`.** Raising `d` from 23.2 M to 50 M —
more than doubling the trainable parameter count, the thing you would expect to
dominate — costs **+13%**. The expensive part is not `θ`; it is touching every
weight in the target set to recompute its group id and scatter/accumulate.

**(b) The cost is in ΔW materialization, not the hash.** The full forward pass
has to build a dense ΔW of the target matrices' size (2184 M fp32 = 8.7 GB worth
of traffic) so that the linear layer can add it. The hash function produces only
the group ids; the dense reconstruction is the bottleneck.

## 3. The int32 partition backend

Starting point: `splitmix64` in int64, which on this hardware is slower than
necessary because 64-bit integer ops are emulated or unvectorized in several
torch paths. An alternative mixer over **int32** (murmur3-style finalizer,
`0x85EBCA6B` / `0xC2B2AE35`, with a `0x9E3779B1` gamma) produces the same kind of
uniform partition and runs on native 32-bit integer lanes.

Micro-benchmark, one full partition pass over all 700 modules:

| backend | time | |
|---|---|---|
| `splitmix64` (int64) | 2.16 s | |
| `int32` (murmur finalizer) | 0.48 s | −78% |

Two passes per step (forward + backward) → **~3.4 s/step saved**.

End-to-end, on the 20.5 s/step all-linear run:

| | s/step |
|---|---|
| `splitmix64` | 20.5 |
| `int32` | **~17.0** |
| speedup | **~1.15–1.2×** |

This is much less than an earlier estimate in our working notes, which assumed
the hash was the dominant cost and projected 20.5 → 5–8 s/step. It is not:
removing 78% of the hash leaves ~83% of the step, because ΔW materialization is
untouched.

**Correctness.** Each weight's group id must be uniquely determined by its flat
index, so that save/load and multi-GPU sharding agree. int32 is exact up to
`2^32`; the largest target set here is 2184 M weights < 4.29 G, so the truncation
is lossless for every configuration in this repo. The code raises rather than
silently wrapping if `start_offset + numel > 2^32`.

Round-trip verified bit-exact in both backends (save → load → identical outputs).
The int32 backend's `θ` semantics differ from `splitmix64` (a different partition
means a different basis), so **checkpoints are not interchangeable between
backends**.

## 4. What this cost model says, and what it does not

✅ The 4.2× figure is real **for this backend on this hardware**. If you are
constrained to 8 GB VRAM and the target set is all-linear, that is what a step
costs.

❌ It does **not** mean "GPart is 4× slower than LoRA" as a method property. A
materialized-`P` implementation trades memory for speed; the upstream code does
that and we did not benchmark it.

❌ It also does not mean GPart is slow in *sampling*. Loading a trained adapter
costs the same as loading a LoRA: the `θ` vector is 23.2 M parameters, smaller
than the r16 LoRA's 42.7 M.

### Consequences for anyone reproducing this

- Budget **~4× the LoRA wall-clock** if you run all-linear GPart on a small card.
  The 1200-step all-linear character run is a multi-hour job, not minutes.
- Prefer **attn2-only** (8.11 s/step) when you can afford the 0.03 score
  difference — it is the better cost/benefit point in this repo.
- The `int32` backend is worth turning on but is not a fix; it buys ~1.2× at the
  cost of checkpoint incompatibility with `splitmix64`.

## 5. Calibration

`src/bench/` reproduces these numbers, on your own hardware:

| | |
|---|---|
| `bench_d.py` | per-step time as a function of `d` and target set |
| `profile_train_step.py` | per-phase breakdown of one real training step |
| `param_budget.py` | module count and covered-weight count for a target set |

Run them before quoting anything here — the ratio between integer and float
paths is very platform-dependent, and the 8 GB figure determined our choice of
backend.
