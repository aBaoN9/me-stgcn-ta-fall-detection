# dha_egcn_v2_ce

backbone adapted small, CE-only

[Upstream 307b5086175964d7781c7e070483c5f4c0e0c7ce](https://github.com/oskar-ika-adi-nugroho/DHA-eGCN/tree/307b5086175964d7781c7e070483c5f4c0e0c7ce)
[Paper](https://doi.org/10.3390/s26123932)

License evidence: Apache-2.0; independent implementation, attribution retained.

Two width-24 blocks; two differential maps with hop bias and five joint-to-part contexts; single fixed masked graph with edge weights; temporal dilation 1/2; dropout in residual branch and head; no RICH or ensemble

Full: 48 fits, 12 outer checkpoints.
Real smoke: 4 fits, 1 outer checkpoint.

[Full lock](full/locked_protocol.json) | [Checkpoints](full/models/manifest.json)
[Raw predictions](full/predictions.csv) | [Per subject and seed](full/fold_metrics.csv)
[Full completion](full/completion.json) | [Smoke completion](smoke/completion.json)

Each fold folder includes model weights, train/validation indices, fit histories,
held-out predictions and raw inner OOF scores.

[Reproduction commands](../../../../training/modern_benchmark_tung/README.md)
