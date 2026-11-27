# cascadeformer_v2_ce

backbone adapted small, CE-only

[Upstream 847efac9bec015d2d1d8f2104d255a0258ad76d7](https://github.com/Yusen-Peng/CascadeFormer/tree/847efac9bec015d2d1d8f2104d255a0258ad76d7)
[Paper](https://arxiv.org/abs/2509.00692)

License evidence: No LICENSE/NOTICE at assigned tree; independent implementation.

Frame projection; temporal T1 to T2; cross-attention fusion; width 48, 4 heads, depth 1+1; dropout in both encoders, fusion and head; no masked reconstruction pretraining

Full: 48 fits, 12 outer checkpoints.
Real smoke: 4 fits, 1 outer checkpoint.

[Full lock](full/locked_protocol.json) | [Checkpoints](full/models/manifest.json)
[Raw predictions](full/predictions.csv) | [Per subject and seed](full/fold_metrics.csv)
[Full completion](full/completion.json) | [Smoke completion](smoke/completion.json)

Each fold folder includes model weights, train/validation indices, fit histories,
held-out predictions and raw inner OOF scores.

[Reproduction commands](../../../../training/modern_benchmark_tung/README.md)
