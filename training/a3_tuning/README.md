# A3 multi-recipe tuning

Six locked recipes on unchanged A3, TCN and GRU architectures. See the [plan](../../docs/research/a3-tuning-plan.md) and [execution registration](../../docs/research/a3-tuning-execution.md). This is an exploratory experiment on four already-observed subjects, not a guaranteed improvement or a deployment replacement.

## Kaggle launch

- **FULL RUNNING at 14:51 UTC+7 on 2026-11-21**, observed 101/684 completed fits and 4/36 outer folds in the private [Kaggle editor](https://www.kaggle.com/code/nguyngiabodzaivail/a3-multi-recipe-nested-loso-20261121/edit). Refresh the live progress for a later status; this is a timestamped observation, not a final result.
- [Version 1](https://www.kaggle.com/code/nguyngiabodzaivail/a3-multi-recipe-nested-loso-20261121?scriptVersionId=372102995) was submitted with Save & Run All at about 14:36, remained queued for about nine minutes, then was cancelled before execution (0 seconds). Run All in the editor started successfully around 14:47. No duplicate full job is active; do not mistake cancelled version 1 for the live interactive run.
- Launcher: [kaggle_a3_tuning.ipynb](kaggle_a3_tuning.ipynb), immutable launcher commit `d62c14e41fa443ebdda47d5944f97964d04cb448`, source commit `7543cac24fcb1e5ba05554d4aa7113b29ad3632d` (21 SHA-verified files).
- Two free Tesla T4 GPUs, Python 3.13.16, Torch 2.11.0+cu130. Keep the live editor session available; do not Stop Session, restart, or press Run All again during execution. This fallback is an interactive Kaggle session, not a committed background job.
- `/kaggle/working/a3-tuning-20261121/`: launch manifest, preflight logs, smoke and full evidence. Successful completion also produces `/kaggle/working/a3-tuning-20261121-result.tar.gz`.
- Kaggle verified all 21 source SHA values/four inputs, passed 24 tests (23.432 seconds test time), and passed the real 12-fit GPU smoke (15.4 seconds command time) before full. Smoke fingerprint: `ef225a27fe70d70d7d033876693d8273bc4591fc9eda89688a90250d636c2dba`. Local CPU smoke is not GPU evidence. This snapshot does not promise completion in an hour.

## Execution contract

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
