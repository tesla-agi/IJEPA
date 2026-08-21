# JEPA World Model

I-JEPA implemented from scratch in PyTorch, built toward replacing the CNN
encoder in a DreamerV3 world model.

## Status

- [x] ViT encoder from scratch (attention, MLP, blocks — no timm)
- [x] Multi-block masking sampler
- [x] EMA target encoder
- [x] Narrow ViT predictor with positional mask tokens
- [x] L2 loss in latent space, no pixel decoder
- [x] Collapse monitor (latent variance + effective rank)
- [ ] Linear probe vs BYOL baseline
- [ ] Collapse ablations
- [ ] V-JEPA tube masks
- [ ] DreamerV3 integration

## Result so far

100 epochs on CIFAR-10, batch 256, AdamW 3e-4:

| | step 0 | step 19,500 |
|---|---|---|
| loss | 1.0039 | 0.2815 |
| latent variance | 0.977 | 0.889 |
| effective rank | 71.4 | 133.4 / 192 |

Effective rank dips to 13 in the first 50 steps, then climbs monotonically
and flattens around step 18,800. Short runs are not diagnostic for rank —
at 10 epochs it reads 33 and is still rising.

## Layout
models/ layers, vit, masking, ema, predictor
train/ loss, ijepa
utils/ monitor


## Run

```bash
python -m train.ijepa
```

## Notes

I-JEPA is a representation learner in the BYOL family, not a world model.
It has no actions and no time axis. In this project the encoder replaces
DreamerV3's per-frame CNN; the RSSM continues to own state, dynamics, and
actions.
