"""Cross-model audit of one run-id (GB-20261124-001, M0 section 7). Fixed before any full result exists.

    python -m training.modern_benchmark_duc.tools.audit --run results/modern_benchmark_duc/<run-id> \
        --input-root data/gmdcsa24 [--no-rescore]

For every model under <run>/full/: re-validates folds and CSVs with the runner (reloads and re-scores every
checkpoint unless --no-rescore), checks that A3 and each competitor share inputs, code, environment and
device, recomputes metrics from predictions.csv, then writes the main fixed-0.5 table, paired deltas against
A3 of the same seed, per-subject deltas, per-clip FN/FP counts, clip flips against A3, secondary policies and
audit_checks.json. Models that are not complete are reported as PARTIAL/NOT_RUN/BLOCKED and left out of tables.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.component_ablation import training as original
from training.modern_benchmark_duc import runner
from training.modern_benchmark_duc.models import A3, ASSIGNED, DS_GCN, MODEL_IDS

MAIN = ["accuracy", "f1", "sensitivity", "specificity", "precision", "balanced_accuracy", "fn", "fp"]
DELTA = ["accuracy", "f1", "sensitivity", "specificity", "precision", "balanced_accuracy"]


def model_status(folder):
    if not folder.exists():
        return "NOT_RUN"
    if not (folder / "completion.json").exists():
        return "PARTIAL"
    done = runner.read_json(folder / "completion.json")
    return "COMPLETE" if done.get("complete") and done.get("mode") == "full" and done.get("scientific_result") else "INVALID"


def shared_lock_fields(lock):
    return dict(inputs=lock["inputs"], code=lock["code"], environment=lock["environment"], config=lock["config"],
                outer_subjects=lock["outer_subjects"], policies=lock["policies"], loader_recipe=lock["loader_recipe"])


def pooled_from_predictions(frame):
    return runner.metric_table(frame, ["variant", "seed", "policy"])


def summarise(pooled, policy):
    rows = []
    for variant, group in pooled[pooled["policy"] == policy].groupby("variant"):
        row = dict(variant=variant, seeds=len(group))
        for column in MAIN:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_sd"] = group[column].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def percent_table(summary):
    lines = ["| Model | Accuracy | F1 (Fall) | Sensitivity | Specificity | Precision | FN | FP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in summary.iterrows():
        cells = [f"{100 * row[f'{c}_mean']:.2f} ± {100 * row[f'{c}_sd']:.2f}" for c in
                 ("accuracy", "f1", "sensitivity", "specificity", "precision")]
        cells += [f"{row[f'{c}_mean']:.2f} ± {row[f'{c}_sd']:.2f}" for c in ("fn", "fp")]
        lines.append(f"| {row['variant']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-rescore", action="store_true", help="skip reloading checkpoints (CSV checks only)")
    args = parser.parse_args()
    output = args.output or args.run / "audit"
    output.mkdir(parents=True, exist_ok=True)
    data = runner.load_verified(args.input_root)
    checks, statuses, frames, locks = [], {}, {}, {}

    def check(name, passed, detail=""):
        checks.append(dict(check=name, status="PASS" if passed else "FAIL", detail=detail))

    for model_id in MODEL_IDS:
        folder = args.run / "full" / model_id
        status = "BLOCKED_LICENSE" if model_id == DS_GCN else model_status(folder)
        statuses[model_id] = status
        if status != "COMPLETE":
            continue
        lock = runner.read_json(folder / "locked_protocol.json")
        locks[model_id] = lock
        if not args.no_rescore:
            device = torch.device(lock["environment"]["device"])
            torch.set_num_threads(lock["environment"]["threads"])
            try:
                runner.validate_summary(data, lock, folder, runner.collect(data, lock, device, folder))
                check(f"{model_id}: folds, checkpoints and CSVs re-validated", True)
            except Exception as error:  # report, never hide
                check(f"{model_id}: folds, checkpoints and CSVs re-validated", False, repr(error))
                statuses[model_id] = "INVALID"
                continue
        frame = pd.read_csv(folder / "predictions.csv", dtype={"video_id": str})
        check(f"{model_id}: 1440 prediction rows, 160 clips per seed/policy",
              len(frame) == 1440 and (frame.groupby(["seed", "policy"]).size() == 160).all())
        index = frame["sample_index"].to_numpy()
        check(f"{model_id}: labels/subjects/video ids match source",
              bool((frame["label"].to_numpy() == data["labels"][index]).all()
                   and (frame["subject"].to_numpy() == data["subjects"][index]).all()
                   and (frame["video_id"].to_numpy() == data["video_ids"][index].astype(str)).all()))
        recomputed = pooled_from_predictions(frame)
        saved = pd.read_csv(folder / "pooled_by_seed.csv")
        merged = saved.merge(recomputed, on=["variant", "seed", "policy"], suffixes=("_saved", "_new"))
        same = all(np.allclose(merged[f"{c}_saved"].astype(float), merged[f"{c}_new"].astype(float), equal_nan=True)
                   for c in runner.RATES + runner.COUNTS)
        check(f"{model_id}: pooled_by_seed recomputed from predictions", same and len(merged) == 9)
        frames[model_id] = frame

    complete = [m for m in MODEL_IDS if m in frames]
    if A3 in locks:
        reference = shared_lock_fields(locks[A3])
        for model_id in complete:
            if model_id != A3:
                check(f"{model_id}: same inputs/code/environment/config as A3",
                      shared_lock_fields(locks[model_id]) == reference)
    predictions = pd.concat([frames[m] for m in complete]) if complete else pd.DataFrame(columns=runner.PREDICTION_COLUMNS)
    pooled = pooled_from_predictions(predictions) if complete else pd.DataFrame()
    if complete:
        pooled.to_csv(output / "pooled_by_seed_all.csv", index=False)
        main_table = summarise(pooled, "fixed")
        main_table.to_csv(output / "main_table_fixed.csv", index=False)
        (output / "main_table_fixed.md").write_text(percent_table(main_table), encoding="utf-8")
        pd.concat([summarise(pooled, p).assign(policy=p) for p in ("balanced", "sensitivity")]).to_csv(
            output / "secondary_policies.csv", index=False)
        fixed = predictions[predictions["policy"] == "fixed"]
        errors = fixed.assign(wrong=(fixed["label"] != fixed["prediction"]).astype(int)).groupby(
            ["variant", "video_id", "subject", "label"]).agg(n_wrong=("wrong", "sum"), seeds=("wrong", "size"),
                                                            mean_score=("score", "mean")).reset_index()
        errors[errors["n_wrong"] > 0].assign(error=lambda f: np.where(f["label"] == 1, "FN", "FP")).sort_values(
            ["variant", "n_wrong", "video_id"], ascending=[True, False, True]).to_csv(output / "clip_errors_fixed.csv", index=False)
        folds = runner.metric_table(fixed, ["variant", "seed", "subject"])
        subject = folds.groupby(["variant", "subject"])[["accuracy", "f1", "sensitivity", "specificity", "fn", "fp"]].mean().reset_index()
        if A3 in frames:
            base = subject[subject["variant"] == A3].set_index("subject")
            for column in ("accuracy", "f1", "sensitivity", "specificity"):
                subject[f"delta_{column}_pp"] = [100 * (row[column] - base.loc[row["subject"], column])
                                                 for _, row in subject.iterrows()]
        subject.to_csv(output / "per_subject_fixed.csv", index=False)
        if A3 in frames:
            a3 = pooled[(pooled["variant"] == A3) & (pooled["policy"] == "fixed")].set_index("seed")
            deltas, summary, flips = [], [], []
            a3_fixed = fixed[fixed["variant"] == A3].set_index(["seed", "sample_index"])
            for model_id in complete:
                if model_id == A3:
                    continue
                mine = pooled[(pooled["variant"] == model_id) & (pooled["policy"] == "fixed")].set_index("seed")
                for seed in mine.index:
                    row = dict(variant=model_id, seed=int(seed))
                    row.update({f"delta_{c}_pp": 100 * (mine.loc[seed, c] - a3.loc[seed, c]) for c in DELTA})
                    row.update(delta_fn=int(mine.loc[seed, "fn"] - a3.loc[seed, "fn"]),
                               delta_fp=int(mine.loc[seed, "fp"] - a3.loc[seed, "fp"]))
                    deltas.append(row)
                frame = pd.DataFrame([d for d in deltas if d["variant"] == model_id])
                item = dict(variant=model_id, seeds=len(frame))
                for column in [f"delta_{c}_pp" for c in DELTA] + ["delta_fn", "delta_fp"]:
                    item[f"{column}_mean"], item[f"{column}_sd"] = frame[column].mean(), frame[column].std(ddof=1)
                item["seeds_accuracy_higher"] = int((frame["delta_accuracy_pp"] > 0).sum())
                item["seeds_accuracy_lower"] = int((frame["delta_accuracy_pp"] < 0).sum())
                item["seeds_f1_higher"] = int((frame["delta_f1_pp"] > 0).sum())
                item["seeds_f1_lower"] = int((frame["delta_f1_pp"] < 0).sum())
                summary.append(item)
                own = fixed[fixed["variant"] == model_id].set_index(["seed", "sample_index"])
                joined = own.join(a3_fixed[["prediction"]], rsuffix="_a3")
                joined["fixed_by_model"] = ((joined["prediction"] == joined["label"]) & (joined["prediction_a3"] != joined["label"])).astype(int)
                joined["broken_by_model"] = ((joined["prediction"] != joined["label"]) & (joined["prediction_a3"] == joined["label"])).astype(int)
                per_clip = joined.groupby(["video_id", "subject", "label"])[["fixed_by_model", "broken_by_model"]].sum().reset_index()
                per_clip = per_clip[(per_clip["fixed_by_model"] > 0) | (per_clip["broken_by_model"] > 0)]
                flips.append(per_clip.assign(variant=model_id))
            pd.DataFrame(deltas).to_csv(output / "paired_delta_by_seed.csv", index=False)
            pd.DataFrame(summary).to_csv(output / "paired_delta_summary.csv", index=False)
            if flips:
                pd.concat(flips).to_csv(output / "clip_flips_vs_a3.csv", index=False)
    result = dict(run=str(args.run.as_posix()), statuses=statuses, complete_models=complete,
                  a3_control_complete=A3 in frames, rescored=not args.no_rescore, checks=checks,
                  overall="PASS" if checks and all(c["status"] == "PASS" for c in checks) else "FAIL",
                  note="Seeds repeat the same 160 clips; no p-values. Outer folds were seen during 06/07/08 development.")
    original.write_json(output / "audit_checks.json", result)
    print(json.dumps(dict(overall=result["overall"], statuses=statuses), indent=2))
    raise SystemExit(0 if result["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
