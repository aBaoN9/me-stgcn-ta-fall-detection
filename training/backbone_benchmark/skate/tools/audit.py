"""Verify completed supplemental results and compare with the locked references."""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.backbone_benchmark import runner as reference
from training.backbone_benchmark.skate import runner
from training.backbone_benchmark.supplement import runner as supplement
from training.component_ablation.models import make_loader

KEYS = ["variant", "seed", "policy"]
MEASURES = ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
            "precision", "f1", "f2", "false_alarm_rate", "tp", "tn", "fp", "fn"]


def metrics(y, predicted):
    tp = int(((y == 1) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return [(tp + tn) / len(y), (sensitivity + specificity) / 2,
            sensitivity, specificity, tp / (tp + fp) if tp + fp else np.nan,
            2 * tp / (2 * tp + fp + fn), 5 * tp / (5 * tp + fp + 4 * fn),
            1 - specificity, tp, tn, fp, fn]


def verify_metrics(run, data, variants):
    rows = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
    if len(rows) != 1440 * variants or rows.duplicated(KEYS + ["sample_index"]).any():
        raise ValueError("Unexpected prediction inventory")
    for _, group in rows.groupby(KEYS):
        np.testing.assert_array_equal(np.sort(group.sample_index), np.arange(160))
    for column, source in (("video_id", "video_ids"), ("label", "labels"), ("subject", "subjects")):
        np.testing.assert_array_equal(rows[column], data[source][rows.sample_index])
    np.testing.assert_array_equal(rows.prediction, (rows.score >= rows.threshold).astype(int))
    for name, keys in (("pooled_by_seed", KEYS), ("fold_metrics", KEYS + ["subject"])):
        actual = pd.read_csv(run / f"{name}.csv").set_index(keys).sort_index()
        calculated = pd.DataFrame(
            [(*key, *metrics(group.label, group.prediction)) for key, group in rows.groupby(keys)],
            columns=[*keys, *MEASURES]).set_index(keys).sort_index()
        pd.testing.assert_index_equal(actual.index, calculated.index)
        np.testing.assert_allclose(actual[MEASURES], calculated, rtol=0, atol=1e-12)
        if name == "pooled_by_seed":
            pooled = calculated.reset_index()
    saved = pd.read_csv(run / "seed_mean_std.csv", header=[0, 1], index_col=[0, 1]).sort_index()
    aggregate = pooled.groupby(["variant", "policy"])[MEASURES].agg(["mean", "std", "count"])
    pd.testing.assert_index_equal(saved.index, aggregate.index)
    np.testing.assert_allclose(saved[aggregate.columns], aggregate, rtol=0, atol=1e-12)
    return rows, pooled


def latency(run, module, data, device):
    manifest = runner.read_json(run / "models/manifest.json")
    lock = runner.read_json(run / "locked_protocol.json")
    index = int(np.flatnonzero(data["video_ids"] == "S1_ADL_01")[0])
    loader = make_loader(data, [index], runner.original.VARIANTS[reference.A3],
                         lock["config"], False, 0)
    features = next(iter(loader))[0].to(device)
    for item in manifest:
        if (item["subject"], item["seed"]) != (1, 42):
            continue
        path = run / item["file"]
        model = module.load_model(path, device)
        samples = []
        with torch.inference_mode():
            for iteration in range(230):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(features)
                torch.cuda.synchronize()
                if iteration >= 30:
                    samples.append((time.perf_counter() - start) * 1000)
        yield dict(variant=item["variant"], parameters=sum(p.numel() for p in model.parameters()),
                   checkpoint_bytes=path.stat().st_size, median_ms=statistics.median(samples),
                   p95_ms=np.percentile(samples, 95), warmups=30, trials=200,
                   video_id="S1_ADL_01", batch_size=1, seed=42, subject=1,
                   torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(device))


def main():
    parser = argparse.ArgumentParser()
    for key in ("run", "reference", "supplement", "input-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    data = runner.load_verified(args.input_root)
    lock = runner.protocol_lock(data, "full", device, False)
    if runner.read_json(args.run / "locked_protocol.json") != lock:
        raise ValueError("Supplemental run lock mismatch")
    done = runner.validate_summary(data, lock, args.run, runner.collect(data, lock, device, args.run))
    if (done["actual_fits"], done["outer_runs"], done["scientific_result"]) != (48, 12, True):
        raise ValueError("Incomplete supplemental benchmark")
    common = ("inputs", "outer_subjects", "primary_policy", "device", "gpu",
              "python", "torch", "cuda", "numpy", "pandas", "sklearn")
    config = {k: v for k, v in lock["config"].items() if k != "variants"}
    for baseline in (args.reference, args.supplement):
        baseline_lock = runner.read_json(baseline / "locked_protocol.json")
        if any(lock[key] != baseline_lock[key] for key in common):
            raise ValueError(f"Reference environment, split or data mismatch: {baseline}")
        if config != {k: v for k, v in baseline_lock["config"].items() if k != "variants"}:
            raise ValueError(f"Reference recipe mismatch: {baseline}")
    comparisons = [(args.run, 1), (args.reference, 2), (args.supplement, 2)]
    rows, pooled = zip(*(verify_metrics(path, data, variants) for path, variants in comparisons))
    fixed = pd.concat(pooled).query("policy == 'fixed'")
    decisions = pd.concat(rows).query("policy == 'fixed'")
    paired = decisions.pivot(index=["seed", "sample_index", "video_id", "subject", "label"],
                             columns="variant", values="prediction").reset_index()
    mistakes = decisions[decisions.label != decisions.prediction]
    args.output.mkdir(parents=True, exist_ok=False)
    fixed.to_csv(args.output / "comparison_fixed_by_seed.csv", index=False)
    fixed.groupby("variant")[MEASURES].agg(["mean", "std"]).to_csv(
        args.output / "comparison_fixed_mean_std.csv")
    paired.to_csv(args.output / "comparison_fixed_paired.csv", index=False)
    mistakes.to_csv(args.output / "comparison_fixed_errors.csv", index=False)
    confusion = fixed.groupby("variant")[["tn", "fp", "fn", "tp"]].sum()
    confusion.to_csv(args.output / "comparison_fixed_confusion.csv")
    by_subject = pd.concat([pd.read_csv(path / "fold_metrics.csv")
                            for path, _ in comparisons]).query("policy == 'fixed'")
    by_subject.to_csv(args.output / "comparison_fixed_by_subject.csv", index=False)
    pd.DataFrame([*latency(args.run, runner, data, device),
                  *latency(args.reference, reference, data, device),
                  *latency(args.supplement, supplement, data, device)]).to_csv(
                      args.output / "comparison_latency.csv", index=False)
    print("PASS: 12 reloads, 1,440 SkateFormer predictions and independent metrics")


if __name__ == "__main__":
    main()
