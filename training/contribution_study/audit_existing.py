"""DUC-01: recompute notebook 07/08 metrics from saved OOF predictions and audit paired contributions.

Only reads results/component_ablation and results/timing_ablation; writes results/contribution_study/audit.
Metrics are reimplemented here (numpy only) so the audit does not reuse the code under test.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEY = ["variant", "seed", "policy"]
RATES = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "f2", "false_alarm_rate"]
COUNTS = ["tp", "tn", "fp", "fn"]
TOLERANCE = 1e-9
PAIRS = [  # (factor, baseline, treatment, source); each pair differs by one factor under the same 07/08 recipe
    ("motion", "A0_pose_average", "A1_motion_average", "component"),
    ("motion", "A2_pose_attention", "A3_motion_attention", "component"),
    ("attention", "A0_pose_average", "A2_pose_attention", "component"),
    ("attention", "A1_motion_average", "A3_motion_attention", "component"),
    ("fall_cost_2.25", "A3_motion_attention", "A3_cost_225", "component"),
    ("timing_interval_dt", "T0_legacy_dt", "T1_interval_dt", "timing"),
]
FOCUS_CLIPS = ["S4_FALL_05", "S3_FALL_06", "S2_FALL_03", "S2_FALL_15",
               "S1_ADL_02", "S1_ADL_03", "S3_ADL_05", "S4_ADL_05", "S4_ADL_06"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(labels, predictions):
    labels, predictions = np.asarray(labels, int), np.asarray(predictions, int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    ratio = lambda top, bottom: top / bottom if bottom else np.nan
    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {"accuracy": ratio(tp + tn, len(labels)), "balanced_accuracy": (sensitivity + specificity) / 2,
            "sensitivity": sensitivity, "specificity": specificity, "precision": ratio(tp, tp + fp),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "f2": ratio(5 * tp, 5 * tp + fp + 4 * fn),
            "false_alarm_rate": ratio(fp, fp + tn), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def grouped_metrics(frame, keys):
    rows = [{**dict(zip(keys, group if isinstance(group, tuple) else (group,))), **metrics(part.label, part.prediction)}
            for group, part in frame.groupby(keys, sort=True)]
    return pd.DataFrame(rows)


def compare(recomputed, stored, keys, columns, name, checks):
    merged = recomputed.merge(stored, on=keys, suffixes=("", "_stored"), how="outer", indicator=True)
    missing = int((merged["_merge"] != "both").sum())
    worst = max(float(np.nanmax(np.abs(merged[c].astype(float) - merged[f"{c}_stored"].astype(float)))) for c in columns)
    checks.append({"check": name, "rows": len(merged), "unmatched_rows": missing, "max_abs_diff": worst,
                   "status": "PASS" if missing == 0 and worst <= TOLERANCE else "FAIL"})


def integrity(frame, name, checks):
    def add(check, ok, detail):
        checks.append({"check": f"{name}: {check}", "status": "PASS" if ok else "FAIL", "detail": detail})
    duplicates = int(frame.duplicated(KEY + ["sample_index"]).sum())
    add("unique (variant,seed,policy,sample_index)", duplicates == 0, f"{duplicates} duplicates")
    sizes = frame.groupby(KEY).agg(n=("sample_index", "nunique"), ids=("video_id", "nunique"),
                                   fall=("label", "sum"), rows=("label", "size"))
    bad = sizes[(sizes.n != 160) | (sizes.ids != 160) | (sizes.fall != 79) | (sizes.rows != 160)]
    add("each group has 160 IDs / 79 Fall / 81 ADL", bad.empty, f"{len(sizes)} groups, {len(bad)} bad")
    per_sample = frame.groupby("sample_index")[["video_id", "label", "subject"]].nunique()
    add("label/subject/video_id constant per sample", bool((per_sample == 1).all().all()), "across all groups")
    implied = (frame.score >= frame.threshold).astype(int)
    add("prediction == (score >= threshold)", bool((implied == frame.prediction).all()),
        f"{int((implied != frame.prediction).sum())} mismatches")
    neural = frame[frame.variant != "SVM_v2"]
    add("neural score in [0,1]", bool(neural.score.between(0, 1).all()), f"{len(neural)} rows")
    scores = frame.groupby(["variant", "seed", "sample_index"]).score.nunique()
    add("policies share the same score", bool((scores == 1).all()), "per (variant,seed,sample)")
    fixed = frame[(frame.policy == "fixed") & (frame.variant != "SVM_v2")]
    add("fixed policy threshold == 0.5", bool(np.allclose(fixed.threshold, 0.5)), f"{len(fixed)} rows")


def seed_summary(pooled):
    neural = pooled[pooled.variant != "SVM_v2"]
    return neural.groupby(["variant", "policy"])[RATES + ["fn", "fp"]].agg(["mean", lambda x: x.std(ddof=1)]) \
        .rename(columns={"<lambda_0>": "std"}, level=1)


def paired(predictions, policy):
    by_seed, by_subject, flips = [], [], []
    for factor, base, treat, source in PAIRS:
        frame = predictions[source]
        frame = frame[frame.policy == policy]
        a = frame[frame.variant == base].set_index(["seed", "sample_index"])
        b = frame[frame.variant == treat].set_index(["seed", "sample_index"])
        assert a.index.sort_values().equals(b.index.sort_values()), (base, treat)
        b = b.loc[a.index]
        for seed in sorted(a.index.get_level_values(0).unique()):
            ma, mb = metrics(a.loc[seed].label, a.loc[seed].prediction), metrics(b.loc[seed].label, b.loc[seed].prediction)
            ca, cb = a.loc[seed].prediction == a.loc[seed].label, b.loc[seed].prediction == b.loc[seed].label
            by_seed.append({"factor": factor, "baseline": base, "treatment": treat, "seed": seed,
                            **{f"delta_{k}": mb[k] - ma[k] for k in RATES + COUNTS},
                            "fn_base": ma["fn"], "fn_treat": mb["fn"], "fp_base": ma["fp"], "fp_treat": mb["fp"],
                            "fixed_by_treatment": int((~ca & cb).sum()), "broken_by_treatment": int((ca & ~cb).sum())})
            for subject in sorted(a.loc[seed].subject.unique()):
                sa, sb = a.loc[seed][a.loc[seed].subject == subject], b.loc[seed][b.loc[seed].subject == subject]
                ma, mb = metrics(sa.label, sa.prediction), metrics(sb.label, sb.prediction)
                by_subject.append({"factor": factor, "baseline": base, "treatment": treat, "seed": seed,
                                   "subject": subject, "delta_accuracy": mb["accuracy"] - ma["accuracy"],
                                   "delta_sensitivity": mb["sensitivity"] - ma["sensitivity"],
                                   "delta_specificity": mb["specificity"] - ma["specificity"],
                                   "fn_base": ma["fn"], "fn_treat": mb["fn"], "fp_base": ma["fp"], "fp_treat": mb["fp"]})
            changed = a.loc[seed][ca != cb]
            for idx, row in changed.iterrows():
                flips.append({"factor": factor, "baseline": base, "treatment": treat, "seed": seed,
                              "video_id": row.video_id, "subject": row.subject, "label": row.label,
                              "score_base": row.score, "score_treat": b.loc[(seed, idx)].score,
                              "change": "fixed" if cb.loc[idx] else "broken"})
    by_seed = pd.DataFrame(by_seed)
    summary_cols = [c for c in by_seed.columns if c.startswith("delta_")] + ["fixed_by_treatment", "broken_by_treatment"]
    summary = by_seed.groupby(["factor", "baseline", "treatment"], sort=False)[summary_cols] \
        .agg(["mean", lambda x: x.std(ddof=1), "min", "max"]).rename(columns={"<lambda_0>": "std"}, level=1)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    signs = by_seed.groupby(["factor", "baseline", "treatment"], sort=False).delta_accuracy \
        .agg(seeds_up=lambda x: int((x > 0).sum()), seeds_down=lambda x: int((x < 0).sum()), seeds_tie=lambda x: int((x == 0).sum()))
    return by_seed, summary.join(signs).reset_index(), pd.DataFrame(by_subject), pd.DataFrame(flips)


def clip_errors(predictions, policy):
    frame = pd.concat(predictions.values())
    frame = frame[(frame.policy == policy) & (frame.variant != "SVM_v2")]
    wrong = frame.assign(error=(frame.prediction != frame.label).astype(int))
    table = wrong.pivot_table(index=["video_id", "subject", "label"], columns="variant", values="error", aggfunc="sum")
    table["total_errors"] = table.sum(axis=1)
    scores = wrong.pivot_table(index="video_id", columns="variant", values="score", aggfunc="mean").add_prefix("mean_score_")
    return table.join(scores, on="video_id").reset_index().sort_values(["total_errors", "video_id"], ascending=[False, True])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/contribution_study/audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comp, timing = ROOT / "results/component_ablation", ROOT / "results/timing_ablation"
    predictions = {"component": pd.read_csv(comp / "predictions.csv"), "timing": pd.read_csv(timing / "predictions.csv")}
    checks = []

    for source, folder in [("component", comp), ("timing", timing)]:
        frame = predictions[source]
        integrity(frame, source, checks)
        pooled = grouped_metrics(frame, KEY)
        compare(pooled, pd.read_csv(folder / "pooled_by_seed.csv"), KEY, RATES + COUNTS,
                f"{source}: pooled_by_seed.csv recomputed", checks)
        fold = grouped_metrics(frame, KEY + ["subject"])
        stored_fold = pd.read_csv(folder / "fold_metrics.csv")
        compare(fold, stored_fold[KEY + ["subject"] + RATES + COUNTS], KEY + ["subject"], COUNTS,
                f"{source}: fold_metrics.csv confusion counts recomputed", checks)
        errors = frame[frame.prediction != frame.label].sort_values(KEY + ["sample_index"]).reset_index(drop=True)
        stored_errors = pd.read_csv(folder / "errors.csv").sort_values(KEY + ["sample_index"]).reset_index(drop=True)
        same = len(errors) == len(stored_errors) and errors[KEY + ["sample_index"]].equals(stored_errors[KEY + ["sample_index"]])
        checks.append({"check": f"{source}: errors.csv == misclassified rows", "status": "PASS" if same else "FAIL",
                       "detail": f"{len(errors)} recomputed vs {len(stored_errors)} stored"})
        summary = seed_summary(pooled)
        if source == "component":
            stored = pd.read_csv(folder / "seed_mean_std.csv").set_index(["variant", "policy"])
            stored = stored[stored.index.get_level_values(0) != "SVM_v2"]
            worst = max(float(np.nanmax(np.abs(summary[(m, s)].reindex(stored.index) - stored[f"{m}_{s}"])))
                        for m in RATES for s in ["mean", "std"])
        else:
            stored = pd.read_csv(folder / "seed_mean_std.csv", header=[0, 1], index_col=[0, 1])
            stored.index = stored.index.set_names(["variant", "policy"])
            worst = max(float(np.nanmax(np.abs(summary[(m, s)].reindex(stored.index) - stored[(m, s)])))
                        for m in RATES + ["fn", "fp"] for s in ["mean", "std"])
        checks.append({"check": f"{source}: seed_mean_std.csv (ddof=1) recomputed", "max_abs_diff": worst,
                       "status": "PASS" if worst <= TOLERANCE else "FAIL"})
        pooled.to_csv(args.output / f"{source}_pooled_by_seed_recomputed.csv", index=False)
        flat = summary.copy()
        flat.columns = [f"{m}_{s}" for m, s in flat.columns]
        flat.reset_index().to_csv(args.output / f"{source}_seed_mean_std_recomputed.csv", index=False)

    # 08 T0 reuses the A3_cost_225 recipe; report whether the scores are identical rather than assume it.
    a = predictions["component"].query("variant == 'A3_cost_225' and policy == 'fixed'").set_index(["seed", "sample_index"]).sort_index()
    t = predictions["timing"].query("variant == 'T0_legacy_dt' and policy == 'fixed'").set_index(["seed", "sample_index"]).sort_index()
    checks.append({"check": "info: T0_legacy_dt vs 07 A3_cost_225 (fixed)", "status": "INFO",
                   "detail": f"max |score diff| = {float(np.abs(a.score - t.score).max()):.3g}; "
                             f"prediction disagreements = {int((a.prediction != t.prediction).sum())}/480"})

    outputs = {}
    for policy in ["fixed", "balanced", "sensitivity"]:
        by_seed, summary, by_subject, flips = paired(predictions, policy)
        suffix = "" if policy == "fixed" else f"_{policy}"
        by_seed.to_csv(args.output / f"paired_delta_by_seed{suffix}.csv", index=False)
        summary.to_csv(args.output / f"paired_delta_summary{suffix}.csv", index=False)
        if policy == "fixed":
            by_subject.to_csv(args.output / "paired_delta_by_subject.csv", index=False)
            flips.to_csv(args.output / "paired_clip_flips.csv", index=False)
            outputs["fixed_summary"] = summary
    errors = clip_errors(predictions, "fixed")
    errors.to_csv(args.output / "clip_error_frequency_fixed.csv", index=False)
    errors[errors.video_id.isin(FOCUS_CLIPS)].to_csv(args.output / "focus_clips_fixed.csv", index=False)

    status = "PASS" if all(c["status"] in {"PASS", "INFO"} for c in checks) else "FAIL"
    inputs = [comp / n for n in ["predictions.csv", "pooled_by_seed.csv", "seed_mean_std.csv", "fold_metrics.csv", "errors.csv"]] + \
             [timing / n for n in ["predictions.csv", "pooled_by_seed.csv", "seed_mean_std.csv", "fold_metrics.csv", "errors.csv"]]
    report = {"status": status, "checks": checks, "tolerance": TOLERANCE,
              "code_sha256": sha256(__file__),
              "inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in inputs},
              "environment": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                              "platform": platform.platform()},
              "notes": ["Seeds are repeated runs on the same 160 clips, not independent samples.",
                        "std uses ddof=1 across 3 seeds; no p-values are computed."]}
    (args.output / "audit_checks.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for c in checks:
        print(f"[{c['status']}] {c['check']}", {k: v for k, v in c.items() if k not in {"check", "status"}})
    print("OVERALL", status)
    cols = ["factor", "baseline", "treatment", "delta_accuracy_mean", "delta_accuracy_std", "delta_sensitivity_mean",
            "delta_specificity_mean", "delta_fn_mean", "delta_fp_mean", "seeds_up", "seeds_down", "seeds_tie"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(outputs["fixed_summary"][cols].round(4).to_string(index=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
