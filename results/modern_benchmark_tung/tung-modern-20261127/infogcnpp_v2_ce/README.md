# infogcnpp_v2_ce

backbone adapted small, CE-only

[Upstream 8c44085ac134f35dc85f683dcc775fdb43172ad1](https://github.com/stnoah1/infogcn2/tree/8c44085ac134f35dc85f683dcc775fdb43172ad1)
[Paper](https://arxiv.org/abs/2310.10547)

License evidence: No LICENSE/NOTICE at assigned tree; independent implementation.

Width 24, dynamic skeleton graph with causal four-head temporal encoder; latent Gaussian scale during training; graph-conditioned RK4 one-step latent flow; concatenate present/future; full-clip pooled classification; no reconstruction, future supervision or KL loss; no Apex/torchdiffeq required

Full: 48 fits, 12 outer checkpoints.
Real smoke: 4 fits, 1 outer checkpoint.

[Full lock](full/locked_protocol.json) | [Checkpoints](full/models/manifest.json)
[Raw predictions](full/predictions.csv) | [Per subject and seed](full/fold_metrics.csv)
[Full completion](full/completion.json) | [Smoke completion](smoke/completion.json)

Each fold folder includes model weights, train/validation indices, fit histories,
held-out predictions and raw inner OOF scores.

[Reproduction commands](../../../../training/modern_benchmark_tung/README.md)
