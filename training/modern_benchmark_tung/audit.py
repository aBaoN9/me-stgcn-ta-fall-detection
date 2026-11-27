"""Audit verified folds without fitting, then compare each model with the fresh A3."""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.backbone_benchmark.models import logits_of
from training.component_ablation.models import make_loader

from . import runner
from .models import A3, MODEL_IDS

KEYS = ["variant", "seed", "policy"]
RATIOS = ["accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity", "f1", "f2",
          "false_alarm_rate"]
COUNTS = ["tp", "tn", "fp", "fn"]


def confusion_metrics(y, predicted):
    tp = int(((y == 1) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())

    def fraction(a, b):
        return a / b if b else np.nan

    sensitivity, specificity = fraction(tp, tp + fn), fraction(tn, tn + fp)
    return dict(accuracy=(tp + tn) / len(y), balanced_accuracy=(sensitivity + specificity) / 2,
                precision=fraction(tp, tp + fp), sensitivity=sensitivity, specificity=specificity,
                f1=fraction(2 * tp, 2 * tp + fp + fn), f2=fraction(5 * tp, 5 * tp + fp + 4 * fn),
                false_alarm_rate=fraction(fp, fp + tn), tp=tp, tn=tn, fp=fp, fn=fn)


def metrics_table(rows, keys):
    return pd.DataFrame([dict(zip(keys, key), **confusion_metrics(g.label, g.prediction))
                         for key, g in rows.groupby(keys)])


def verify_full(root, data, device, model_id):
    lock = runner.protocol_lock(data, "full", device, False, model_id)
    if runner.read_json(root / "locked_protocol.json") != lock:
        raise ValueError(f"Run lock mismatch: {model_id}")
    verified = runner.collect(data, lock, device, root)
    completed = runner.validate_summary(data, lock, root, verified)
    if (completed["actual_fits"], completed["outer_runs"], completed["scientific_result"]) != (48, 12, True):
        raise ValueError(f"Invalid completion: {model_id}")
    predictions = pd.read_csv(root / "predictions.csv", dtype={"video_id": str}, float_precision="round_trip")
    if len(predictions) != 1440 or predictions.duplicated(KEYS + ["sample_index"]).any():
        raise ValueError(f"Invalid prediction inventory: {model_id}")
    for _, group in predictions.groupby(KEYS):
        np.testing.assert_array_equal(np.sort(group.sample_index), np.arange(160))
    for filename, keys in (("pooled_by_seed", KEYS), ("fold_metrics", KEYS + ["subject"])):
        actual = pd.read_csv(root / f"{filename}.csv").set_index(keys).sort_index()
        expected = metrics_table(predictions, keys).set_index(keys).sort_index()
        pd.testing.assert_index_equal(actual.index, expected.index)
        np.testing.assert_allclose(actual[RATIOS + COUNTS], expected[RATIOS + COUNTS], atol=1e-12, rtol=0)
    return predictions, lock


def measure(root, data, device, cfg):
    record = next(r for r in runner.read_json(root / "models/manifest.json")
                  if (r["seed"], r["subject"]) == (42, 1))
    checkpoint = root / record["file"]
    model = runner.load_model(checkpoint, device)
    index = int(np.flatnonzero(data["video_ids"] == "S1_ADL_01")[0])
    x = next(iter(make_loader(data, [index], runner.original.VARIANTS[A3], cfg, False, 0)))[0].to(device)
    samples = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for i in range(230):
            sync()
            start = time.perf_counter()
            logits_of(model, x)
            sync()
            if i >= 30:
                samples.append((time.perf_counter() - start) * 1000)
    return dict(variant=record["variant"], parameters=sum(p.numel() for p in model.parameters()),
                checkpoint_bytes=checkpoint.stat().st_size, median_ms=statistics.median(samples),
                p95_ms=np.percentile(samples, 95), samples=len(samples), warmups=30, batch=1,
                shape=str(tuple(x.shape)), dtype=str(x.dtype), threads=torch.get_num_threads(),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                sample="S1_ADL_01", checkpoint_seed=42, checkpoint_subject=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="Runtime or published artifact directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = runner.load_verified(args.input_root)
    locks, rows, timings = {}, [], []

    def artifact(model, mode):
        published = args.runs / model / mode
        return published if published.is_dir() else args.runs / f"{model}-{mode}"

    for model in MODEL_IDS:
        run = artifact(model, "full")
        runner.verify_smoke(data, device, artifact(model, "smoke"), model)
        predictions, lock = verify_full(run, data, device, model)
        locks[model] = lock
        rows.append(predictions)
    base = locks[A3]
    for model in MODEL_IDS:
        lock = locks[model]
        for field in ("inputs", "environment", "outer_subjects", "device", "gpu", "torch", "cuda",
                      "python", "numpy", "pandas", "sklearn", "primary_policy"):
            if lock[field] != base[field]:
                raise ValueError(f"Noncomparable {field}: {model}")
        if {k: v for k, v in lock["config"].items() if k != "variants"} != {
                k: v for k, v in base["config"].items() if k != "variants"}:
            raise ValueError(f"Noncomparable recipe: {model}")
        timings.append(measure(artifact(model, "full"), data, device, lock["config"]))
    frame = pd.concat(rows, ignore_index=True)
    pooled, folds = metrics_table(frame, KEYS), metrics_table(frame, KEYS + ["subject"])
    fixed = frame[frame.policy == "fixed"]
    mistakes = frame[frame.label != frame.prediction]
    recurring = (mistakes[mistakes.policy == "fixed"].groupby(["variant", "video_id", "subject", "label"])
                 .size().rename("error_seeds").reset_index().sort_values("error_seeds", ascending=False))
    paired = fixed.pivot(index=["seed", "sample_index", "video_id", "subject", "label"],
                         columns="variant", values="prediction").reset_index()
    args.output.mkdir(parents=True, exist_ok=False)
    pooled.to_csv(args.output / "by_seed.csv", index=False)
    folds.to_csv(args.output / "by_subject_seed.csv", index=False)
    pooled.groupby(["variant", "policy"])[RATIOS + COUNTS].agg(["mean", "std", "count"]).to_csv(
        args.output / "mean_sd.csv")
    pooled.groupby(["variant", "policy"])[["tn", "fp", "fn", "tp"]].sum().to_csv(
        args.output / "confusion.csv")
    for table, name, by in ((pooled, "paired_delta_seed", ["seed", "policy"]),
                            (folds, "paired_delta_subject_seed", ["seed", "policy", "subject"])):
        values = table.set_index(["variant", *by])[RATIOS + COUNTS]
        delta = values.subtract(values.loc[A3], level=None)
        delta.drop(index=A3, level="variant").to_csv(args.output / f"{name}.csv")
    paired.to_csv(args.output / "paired_fixed_predictions.csv", index=False)
    mistakes.to_csv(args.output / "errors_all_policies.csv", index=False)
    recurring.to_csv(args.output / "recurring_fixed_errors.csv", index=False)
    pd.DataFrame(timings).to_csv(args.output / "latency.csv", index=False)
    print("PASS: 6 models, 288 fits, 72 full checkpoint reloads, 8,640 verified predictions")


if __name__ == "__main__":
    main()
