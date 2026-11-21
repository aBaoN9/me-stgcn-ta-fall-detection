# A3 multi-recipe tuning

Six locked recipes on unchanged A3, TCN and GRU architectures. See the [plan](../../docs/research/a3-tuning-plan.md) and [execution registration](../../docs/research/a3-tuning-execution.md). This is an exploratory experiment on four already-observed subjects, not a guaranteed improvement or a deployment replacement.

- Full budget: 648 inner fits, 36 refits, three seeds, fixed threshold 0.5, 1440 primary OOF rows.
- Real smoke: 12 additional one-epoch fits before full; unit-test synthetic fits are separate.
- CUDA full only, two same-type GPUs, one process per GPU; no paid resources or package replacement.
- Resume the identical command/output with unchanged source/input/runtime locks. Checkpoint every five epochs, plus fit completion; no silent incompatible resume.
- `progress.json`, `worker0.log`, `worker1.log`: progress. `metrics_partial.csv`: incomplete observations, not final benchmark scores.
- `completion.json`: emitted only after all 684 fits/36 outer folds/1440 rows. Independent scientific review and new latency measurements remain separate.

```bash
python -m training.a3_tuning.test_runner
python -m training.a3_tuning.runner --mode smoke --device cuda --input-root data/gmdcsa24 --output /kaggle/working/a3-tuning-20261121/smoke
python -m training.a3_tuning.runner --mode full --device cuda --workers 2 --input-root data/gmdcsa24 --smoke-evidence /kaggle/working/a3-tuning-20261121/smoke --output /kaggle/working/a3-tuning-20261121/full
```

CPU is supported for tests/smoke, not for the full GPU registration. Default output directories are never cleared. A changed lock requires a new output directory. Notebook launchers belong here rather than in the nine locked historical notebooks.
