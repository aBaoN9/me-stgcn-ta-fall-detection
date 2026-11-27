# A3_motion_attention

unmodified notebook 07 SkeletonModel and fit_model

Full: 48 fits, 12 outer checkpoints.
Real smoke: 4 fits, 1 outer checkpoint.

[Full lock](full/locked_protocol.json) | [Checkpoints](full/models/manifest.json)
[Raw predictions](full/predictions.csv) | [Per subject and seed](full/fold_metrics.csv)
[Full completion](full/completion.json) | [Smoke completion](smoke/completion.json)

Each fold folder includes model weights, train/validation indices, fit histories,
held-out predictions and raw inner OOF scores.

[Reproduction commands](../../../../training/modern_benchmark_tung/README.md)
