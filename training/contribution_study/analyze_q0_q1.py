"""DUC-04: fixed analysis of a completed Q0 (A3 attention) vs Q1 (A3 + quality prior) run.

Written and locked on 2026-11-18 before any real Q0/Q1 outer result existed (M0 addendum in
results/contribution_study/m0_locked.json). Every constant below is fixed in advance; nothing here is tuned
on outer-test results. Outputs are descriptive: no p-values (3 seeds repeat the same 160 clips).

1. primary_paired.csv   policy fixed: pooled metrics per seed for Q0/Q1 and the paired delta Q1 - Q0.
2. confusion.csv        tn/fp/fn/tp per variant x seed x policy.
3. per_subject.csv      fold metrics per subject, delta Q1 - Q0 per seed and its mean.
4. flips.csv            per clip: in how many seeds Q1 fixes / breaks the Q0 decision (policy fixed).
5. mechanism.csv        per clip x seed x variant: corr(attention, q') and attention mass on low-quality steps.
6. known_failures.csv   decisions on the DUC-01 failure clips for both variants.
7. analysis_summary.json  headline numbers + Q0 reproduction sanity flag (environment check only).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.component_ablation.features import load_inputs
from training.component_ablation.models import PoseDataset
from training.contribution_study.quality_pooling import frame_quality, temporal_quality

ROOT = Path(__file__).resolve().parents[2]
BASELINE, CHANGE = "Q0_A3_attention", "Q1_A3_quality_prior"
PRIMARY_POLICY = "fixed"
LOW_QUALITY = 0.5  # attention step counted as low quality when q' < 0.5 (locked 2026-11-18)
KNOWN_FAILURES = ["S4_FALL_05", "S4_ADL_05", "S4_ADL_06"]  # from DUC-01 audit of 07/08
HISTORICAL = ROOT / "results/component_ablation/seed_mean_std.csv"
SANITY_SD = 2.0  # Q0 fixed accuracy outside historical mean +/- 2 sd -> investigate environment, never retune Q1
METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "fn", "fp"]


def load_run(run):
    completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
    if not completion.get("complete"):
        raise ValueError("Run is not complete; analyse only validated runs")
    if set(completion["variants"]) != {BASELINE, CHANGE}:
        raise ValueError(f"Run variants {completion['variants']} are not the locked Q0/Q1 pair")
    return completion, pd.read_csv(run / "predictions.csv"), pd.read_csv(run / "pooled_by_seed.csv"), pd.read_csv(run / "fold_metrics.csv")


def paired(frame, keys):
    base = frame[frame["variant"] == BASELINE].set_index(keys)
    change = frame[frame["variant"] == CHANGE].set_index(keys)
    if set(base.index) != set(change.index):
        raise ValueError(f"Q0/Q1 rows do not pair on {keys}")
    change = change.loc[base.index]
    rows = []
    for key in base.index:
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for metric in METRICS:
            row[f"{metric}_q0"], row[f"{metric}_q1"] = base.at[key, metric], change.at[key, metric]
            row[f"{metric}_delta"] = change.at[key, metric] - base.at[key, metric]
        rows.append(row)
    return pd.DataFrame(rows)


def flips(predictions):
    fixed = predictions[predictions["policy"] == PRIMARY_POLICY]
    table = fixed.pivot_table(index=["sample_index", "video_id", "subject", "label", "seed"], columns="variant",
                              values="prediction").reset_index()
    table["q0_correct"] = table[BASELINE] == table["label"]
    table["q1_correct"] = table[CHANGE] == table["label"]
    table["fixed_by_q1"] = ~table["q0_correct"] & table["q1_correct"]
    table["broken_by_q1"] = table["q0_correct"] & ~table["q1_correct"]
    per_clip = table.groupby(["sample_index", "video_id", "subject", "label"]).agg(
        seeds=("seed", "nunique"), q0_errors=("q0_correct", lambda x: int((~x).sum())),
        q1_errors=("q1_correct", lambda x: int((~x).sum())), fixed_by_q1=("fixed_by_q1", "sum"),
        broken_by_q1=("broken_by_q1", "sum")).reset_index()
    return per_clip[(per_clip["q0_errors"] > 0) | (per_clip["q1_errors"] > 0)].sort_values(
        ["fixed_by_q1", "broken_by_q1", "video_id"], ascending=[False, False, True])


def step_quality(data, indices, steps):
    dataset = PoseDataset(data, np.asarray(indices), 8, False, 0)
    features = torch.stack([torch.as_tensor(dataset[offset][0]) for offset in range(len(dataset))])
    return temporal_quality(frame_quality(features), steps).numpy()


def mechanism(run, data, completion):
    rows = []
    for variant in (BASELINE, CHANGE):
        for seed in completion["seeds"]:
            for path in sorted(run.glob(f"{variant}_seed{seed}_subject*_attention.npz")):
                saved = np.load(path)
                indices, attention = saved["indices"], saved["attention"]
                quality = step_quality(data, indices, attention.shape[1])
                for index, weights, q in zip(indices, attention, quality):
                    if not np.isclose(weights.sum(), 1.0, atol=1e-4):
                        raise ValueError(f"Attention in {path.name} does not sum to 1")
                    constant = np.ptp(q) < 1e-9 or np.ptp(weights) < 1e-12
                    rows.append({"variant": variant, "seed": seed, "sample_index": int(index),
                                 "video_id": str(data["video_ids"][index]), "label": int(data["labels"][index]),
                                 "mean_q": float(q.mean()), "min_q": float(q.min()),
                                 "corr_attention_q": np.nan if constant else float(np.corrcoef(weights, q)[0, 1]),
                                 "attention_on_low_q": float(weights[q < LOW_QUALITY].sum())})
    frame = pd.DataFrame(rows)
    expected = 2 * len(completion["seeds"]) * len(data["labels"])
    if len(frame) != expected or frame.duplicated(["variant", "seed", "sample_index"]).any():
        raise ValueError(f"Mechanism coverage {len(frame)} rows, expected {expected} unique")
    return frame


def mechanism_summary(frame):
    out = {}
    for variant, group in frame.groupby("variant"):
        corr = group["corr_attention_q"].dropna()
        out[variant] = {"mean_corr_attention_q": float(corr.mean()) if len(corr) else None,
                        "clips_with_defined_corr": int(len(corr)), "clips_constant_q_or_attention": int(group["corr_attention_q"].isna().sum()),
                        "mean_attention_on_low_q": float(group["attention_on_low_q"].mean()),
                        "mean_attention_on_low_q_when_present": float(group.loc[group["min_q"] < LOW_QUALITY, "attention_on_low_q"].mean())
                        if (group["min_q"] < LOW_QUALITY).any() else None,
                        "clip_seed_with_low_q_steps": int((group["min_q"] < LOW_QUALITY).sum())}
    return out


def q0_sanity(pooled):
    try:
        history = pd.read_csv(HISTORICAL).set_index(["variant", "policy"]).loc[("A3_motion_attention", PRIMARY_POLICY)]
    except (FileNotFoundError, KeyError):
        return {"status": "NOT_RUN", "reason": "historical A3 fixed row unavailable"}
    q0 = pooled[(pooled["variant"] == BASELINE) & (pooled["policy"] == PRIMARY_POLICY)]["accuracy"]
    low, high = history["accuracy_mean"] - SANITY_SD * history["accuracy_std"], history["accuracy_mean"] + SANITY_SD * history["accuracy_std"]
    inside = bool(low <= q0.mean() <= high)
    return {"status": "PASS" if inside else "INVESTIGATE_ENVIRONMENT", "q0_fixed_accuracy_mean": float(q0.mean()),
            "historical_mean": float(history["accuracy_mean"]), "historical_std": float(history["accuracy_std"]),
            "band": [float(low), float(high)],
            "rule": "Environment check only. Historical numbers are never the control and never change Q1."}


def analyse(run, data, output):
    run, output = Path(run), Path(output)
    completion, predictions, pooled, folds = load_run(run)
    if data.get("synthetic", False) and completion.get("scientific_result"):
        raise ValueError("Synthetic data cannot analyse a scientific run")
    output.mkdir(parents=True, exist_ok=True)
    primary = paired(pooled[pooled["policy"] == PRIMARY_POLICY], ["seed"])
    all_policies = paired(pooled, ["policy", "seed"])
    per_subject = paired(folds, ["policy", "seed", "subject"])
    confusion = pooled[["variant", "seed", "policy", "tn", "fp", "fn", "tp"]].sort_values(["policy", "variant", "seed"])
    flip_table = flips(predictions)
    mech = mechanism(run, data, completion)
    known = predictions[predictions["video_id"].isin(KNOWN_FAILURES) & (predictions["policy"] == PRIMARY_POLICY)][
        ["video_id", "label", "variant", "seed", "score", "prediction"]].sort_values(["video_id", "variant", "seed"])
    for name, frame in [("primary_paired", primary), ("all_policies_paired", all_policies), ("per_subject", per_subject),
                        ("confusion", confusion), ("flips", flip_table), ("mechanism", mech), ("known_failures", known)]:
        frame.to_csv(output / f"{name}.csv", index=False)
    fixed_subject = per_subject[per_subject["policy"] == PRIMARY_POLICY]
    summary = {
        "run": run.name, "scientific_result": bool(completion.get("scientific_result")),
        "primary": {"metric": "pooled accuracy, policy fixed", "q0_mean": float(primary["accuracy_q0"].mean()),
                    "q0_sd": float(primary["accuracy_q0"].std(ddof=1)), "q1_mean": float(primary["accuracy_q1"].mean()),
                    "q1_sd": float(primary["accuracy_q1"].std(ddof=1)), "delta_mean": float(primary["accuracy_delta"].mean()),
                    "delta_sd": float(primary["accuracy_delta"].std(ddof=1)) if len(primary) > 1 else None,
                    "seeds_improved": int((primary["accuracy_delta"] > 0).sum()), "seeds_worse": int((primary["accuracy_delta"] < 0).sum()),
                    "seeds": int(len(primary))},
        "fixed_policy_mean_delta": {metric: float(primary[f"{metric}_delta"].mean()) for metric in METRICS},
        "per_subject_mean_accuracy_delta": {int(k): float(v) for k, v in fixed_subject.groupby("subject")["accuracy_delta"].mean().items()},
        "flips": {"clip_seed_fixed_by_q1": int(flip_table["fixed_by_q1"].sum()), "clip_seed_broken_by_q1": int(flip_table["broken_by_q1"].sum())},
        "mechanism": mechanism_summary(mech), "low_quality_threshold": LOW_QUALITY,
        "q0_reproduction_sanity": q0_sanity(pooled),
        "reporting_rule": "Descriptive paired deltas and seed counts; no p-value over 3 seeds on the same 160 clips. Negative or null results are valid.",
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default <run>/analysis")
    args = parser.parse_args()
    summary = analyse(args.run, load_inputs(args.input_root), args.output or args.run / "analysis")
    print(json.dumps(summary["primary"], indent=2))


if __name__ == "__main__":
    main()
