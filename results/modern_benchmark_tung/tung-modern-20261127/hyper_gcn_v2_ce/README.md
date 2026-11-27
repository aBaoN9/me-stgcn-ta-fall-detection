# hyper_gcn_v2_ce

backbone adapted small, CE-only

[Upstream cefe70b9c62ab825722cc8ece3fa26ecf5c334a1](https://github.com/6UOOON9/Hyper-GCN/tree/cefe70b9c62ab825722cc8ece3fa26ecf5c334a1)
[Paper](https://arxiv.org/abs/2411.14796)

License evidence: No LICENSE/NOTICE at assigned tree; independent implementation.

Two width-24 blocks; multi-head adaptive hypergraph with four heads, soft nearest-nine incidence and six anatomical virtual vertices; temporal dilation 1/2; residual and head dropout

Full: 48 fits, 12 outer checkpoints.
Real smoke: 4 fits, 1 outer checkpoint.

[Full lock](full/locked_protocol.json) | [Checkpoints](full/models/manifest.json)
[Raw predictions](full/predictions.csv) | [Per subject and seed](full/fold_metrics.csv)
[Full completion](full/completion.json) | [Smoke completion](smoke/completion.json)

Each fold folder includes model weights, train/validation indices, fit histories,
held-out predictions and raw inner OOF scores.

[Reproduction commands](../../../../training/modern_benchmark_tung/README.md)
