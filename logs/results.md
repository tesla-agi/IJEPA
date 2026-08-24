# Results

CIFAR-10, ViT-Tiny (d=192, depth=6, heads=3, patch=4 → 64 tokens), 2.69M params.
Batch 256, AdamW 3e-4, MPS. Linear probe: frozen encoder, `Linear(192,10)`,
Adam 1e-3, 15 epochs, batch 256 — matched to the BYOL baseline protocol.

## Baseline pretraining (100 epochs, 19,500 steps)

| step | loss | latent var | effective rank |
|---|---|---|---|
| 0 | 1.0039 | 0.977 | 71.4 |
| 50 | 0.7274 | 0.962 | 13.2 |
| 1,900 | 0.3751 | 0.942 | 33.2 |
| 19,500 | 0.2815 | 0.889 | 133.4 |

Rank dips to 13 in the first 50 steps, then climbs monotonically and flattens
around step 18,800.

## Probe vs pretraining length

| pretrain epochs | probe |
|---|---|
| 40 | 51.3 |
| 70 | 55.9 |
| 100 | 57.6 |
| **BYOL (ResNet-18, same data/protocol)** | **80.07** |

Gains decelerate (+4.6 then +1.7 per 30 epochs). Longer pretraining is not the
lever. Two candidate causes for the gap:

1. **Mask shape degeneracy at 8×8.** Scale 0.15–0.20 of 64 patches gives
   9.6–12.8; intersected with aspect 0.75–1.5 and integer sides, only
   {3×3, 3×4, 4×3} are reachable. Only position varies. The paper's 14×14 grid
   admits far more variety.
2. **ViT data-hunger.** 50k images at 32×32 is the regime where the ViT paper
   found CNNs win. BYOL's ResNet gets locality and translation equivariance free.

## Collapse ablations (40 epochs, matched config, seed 42)

| run | loss | var | rank | probe |
|---|---|---|---|---|
| baseline | ~0.35 | 0.95 | ~50 | 51.3 |
| (a) no stop-grad | 0.0000 | 0.0000 | 1.6 | — (constant fn) |
| (b) no EMA | 0.0252 | 0.0220 | 3.7 | 21.0 |
| (c) no target LayerNorm | 0.2425 | 0.9029 | 81.4 | 48.3 |

**(a)** Total collapse in under 50 steps. Loss reaches exactly 0.0000, as theory
predicts for `f ≡ c`. Stop-grad is load-bearing.

**(b)** Dove to rank 6 by step 200, then *bounced out* at step 600 — loss rose
0.028 → 0.79, variance recovered to 0.985 — before settling back to rank 3.7.
The stop-grad + predictor mechanism (SimSiam) delays and partially reverses
collapse but does not prevent it here. **SimSiam's stop-grad-only result does not
transfer to masked latent prediction at this scale.**

**(c)** Did not collapse. Lower loss, *higher* rank, worse probe. The target
LayerNorm's load-bearing job at this scale is scale equalisation, not DC removal.

## Monitor findings

Two independent cases where a single metric would have misled:

- **(b) at step 750:** `var = 0.985`, *above* the baseline's 0.972, while
  `rank = 3.9`. Variance alone read green on a broken run.
- **(c) overall:** every monitor metric read *better* than baseline — lower loss,
  higher rank — yet the probe was 3 points worse.

> Effective rank measures how many directions the representation spans. It does
> not measure whether those directions are useful. Necessary, not sufficient.

## D5 — RMSNorm + SwiGLU (parameter-matched)

SwiGLU hidden width `8d/3 = 512`, giving 296,128 params vs the GELU MLP's
295,872 — 0.09% apart. Any difference is attributable to the mechanism, not
capacity.

| run (40 epochs) | loss | var | rank | probe |
|---|---|---|---|---|
| LayerNorm + GELU | ~0.35 | 0.95 | ~50 | 51.3 |
| RMSNorm + SwiGLU | 0.245 | 0.872 | 136.8 | **55.8** |

+4.5 points at matched parameters and matched steps. Reaches close to the
100-epoch baseline (57.6) in 40 epochs. The early rank dip is much shallower —
30.4 at step 50 vs the baseline's 13.2.

**Caveats:** two changes flipped together, so the 4.5 points are not attributed
between RMSNorm and SwiGLU. Single seed; run-to-run noise not characterised.

## Notes

- The probe measures *linear accessibility*, not information content. A
  representation can hold class structure nonlinearly (cf. MAE: strong
  fine-tuning, weak probing).
- CIFAR classification is a proxy. The downstream target is a DreamerV3 encoder,
  which needs positions, velocities and occlusion structure rather than ten class
  labels. Probe accuracy is the cheapest available signal, not the objective.

## Skipped, logged

- D2 mask-overlay render. Gated by assertion only. If a downstream result is
  unexplained, check mask geometry (row/col transposition) first.
- Assran et al. §3 and the masking ablation tables — not read.