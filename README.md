# JEPA World Model

I-JEPA implemented from scratch in PyTorch — no `timm`, no reference code — and
evaluated against a matched BYOL baseline. Built toward replacing the CNN encoder
in a DreamerV3 world model.

## What this is

I-JEPA learns image representations by hiding regions of an image and predicting
**their representations**, not their pixels. There is no decoder anywhere in the
loss. That removes the reconstruction tax — an encoder trained on pixels must
represent sensor noise and texture, because MSE has no mechanism to discard
anything — but it also makes collapse the global minimum of the objective, since
a constant encoder scores exactly zero.

Most of the architecture exists to prevent that. This repo measures whether each
part is load-bearing.

## Architecture
 one image, 64 patches
    ┌──────────────────┴──────────────────┐
    19–34 context patches all 64 patches
│ │
f_θ context encoder f_θ̄ target encoder
ViT, trains EMA copy, no gradient
│ │
│ index target block
▼ │
g_φ predictor LayerNorm
narrow ViT (d=96, depth=2) stop-gradient
input: context tokens │
+ mask token + pos_embed[j] │
│ │
└──────── L2 in latent space ─────────┘


The mask token carries **position and nothing else** — the predictor is told
*where* to predict, never *what* is there. Swap that positional query for an
action embedding and the same architecture becomes a world model. That is the
seam this project is built toward.

## Results

Full numbers in [`logs/results.md`](logs/results.md).

| | probe (CIFAR-10 linear) |
|---|---|
| I-JEPA, 100 epochs | 57.6 |
| I-JEPA + RMSNorm/SwiGLU, 40 epochs | 55.8 |
| BYOL (ResNet-18, matched protocol) | 80.07 |

### Collapse ablations — 40 epochs, matched config

| run | loss | var | eff. rank | probe |
|---|---|---|---|---|
| baseline | ~0.35 | 0.95 | ~50 | 51.3 |
| no stop-grad | 0.0000 | 0.0000 | 1.6 | — |
| no EMA | 0.0252 | 0.0220 | 3.7 | 21.0 |
| no target LayerNorm | 0.2425 | 0.9029 | 81.4 | 48.3 |

Three findings worth stating plainly:

**Stop-grad is a wall.** Remove it and the encoder becomes a constant function in
under 50 steps, with loss reaching exactly 0.0000 — the value theory predicts for
`f ≡ c`.

**EMA is not optional at this scale.** SimSiam showed stop-grad plus a predictor
suffices. That mechanism is visible here — the run dives toward collapse by step
200, then *bounces out* at step 600 with variance recovering to 0.985 — but it
does not hold. It settles at rank 3.7 and probes at 21%. SimSiam's result does
not transfer to masked latent prediction at this scale.

**Effective rank is necessary, not sufficient.** Removing the target LayerNorm
produced *lower loss and higher rank* than baseline, and a worse probe. Every
monitor metric read green on a run that was measurably worse.

### RMSNorm + SwiGLU, parameter-matched

SwiGLU hidden width `8d/3 = 512` gives 296,128 params against the GELU MLP's
295,872 — 0.09% apart, so the gain is not capacity.

**+4.5 probe points at matched parameters and matched steps**, reaching close to
the 100-epoch baseline in 40 epochs.

## Why the gap to BYOL

Two identified causes, neither of which is undertraining — the probe-vs-epochs
curve (51.3 / 55.9 / 57.6 at 40 / 70 / 100) decelerates clearly.

**Mask shape degeneracy at 8×8.** Scale 0.15–0.20 of 64 patches gives 9.6–12.8;
intersected with aspect ratio 0.75–1.5 and integer sides, only {3×3, 3×4, 4×3}
are reachable. Only *position* varies. The paper's 14×14 grid admits far more.

**ViT data-hunger.** 50k images at 32×32 is precisely the regime where the ViT
paper found CNNs win. BYOL's ResNet gets locality and translation equivariance
for free.

## Layout

models/
layers.py Attention, MLP, SwiGLU, ViTBlock, norm/ffn dispatchers
vit.py PatchEmbed, VIT
masking.py multi-block sampler
ema.py target encoder build, update, momentum schedule
predictor.py narrow ViT with positional mask tokens
train/
ijepa.py pretraining loop
loss.py L2 in latent space
probe.py frozen linear probe
utils/
monitor.py latent variance + effective rank
logs/
results.md


## Run

```bash
python -m train.ijepa     # pretrain
python -m train.probe     # evaluate
```

Config flags on `VIT` and `Predictor`: `norm_type` ∈ {`layer_norm`, `rms_norm`},
`ffn_type` ∈ {`gelu_mlp`, `swiglu`}.

## Status

- [x] ViT encoder, multi-block masking, EMA target, predictor, latent loss
- [x] Collapse monitor — latent variance + effective rank
- [x] Linear probe vs BYOL, with a convergence curve
- [x] Three collapse ablations
- [x] RMSNorm + SwiGLU, parameter-matched
- [ ] V-JEPA tube masks
- [ ] DreamerV3 integration

## A note on framing

I-JEPA is a **representation learner** in the BYOL family — stop-grad, EMA
target, predictor asymmetry, no negatives. It has no actions and no time axis, so
it is not a world model.

In this project the encoder replaces DreamerV3's per-frame CNN. The RSSM
continues to own state, dynamics, and actions. The question being tested is
whether a *predictability-shaped* latent is a better substrate for planning than
a *pixel-shaped* one.

## References

- Assran et al. 2023 — *Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture*
- Grill et al. 2020 — *Bootstrap Your Own Latent*
- Chen & He 2021 — *Exploring Simple Siamese Representation Learning*
- Dosovitskiy et al. 2021 — *An Image is Worth 16×16 Words*
- Hafner et al. 2023 — *Mastering Diverse Domains through World Models*
