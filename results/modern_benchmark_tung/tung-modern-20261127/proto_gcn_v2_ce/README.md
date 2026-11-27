# proto_gcn_v2_ce

backbone adapted small, CE-only

[Upstream ddf7f274f9f5d9e45a2fcfeb299bfb3fd7c2303d](https://github.com/firework8/ProtoGCN/tree/ddf7f274f9f5d9e45a2fcfeb299bfb3fd7c2303d)
[Paper](https://arxiv.org/abs/2411.18941)

License evidence: MIT; independent implementation, attribution retained.

Two width-24 dynamic topology blocks; 16 learned prototypes for final edge reconstruction; linear query and memory; reconstructed topology readout fused with pooled features; dropout in residual, reconstruction and head; no class-specific contrastive loss

Full: 48 fits, 12 outer checkpoints.
Real smoke: 4 fits, 1 outer checkpoint.

[Full lock](full/locked_protocol.json) | [Checkpoints](full/models/manifest.json)
[Raw predictions](full/predictions.csv) | [Per subject and seed](full/fold_metrics.csv)
[Full completion](full/completion.json) | [Smoke completion](smoke/completion.json)

Each fold folder includes model weights, train/validation indices, fit histories,
held-out predictions and raw inner OOF scores.

[Reproduction commands](../../../../training/modern_benchmark_tung/README.md)
