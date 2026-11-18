"""Checks for analyze_q0_q1 on the synthetic smoke run (software check only, not a result).

Run check_quality_pooling first: it rebuilds .runtime/contribution_study/synthetic_smoke from synthetic seed 11.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.contribution_study.analyze_q0_q1 import BASELINE, CHANGE, analyse, step_quality
from training.contribution_study.check_quality_pooling import synthetic_data
from training.contribution_study.runner import build_model

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / ".runtime/contribution_study/synthetic_smoke"
WORK = ROOT / ".runtime/contribution_study/analysis_checks"


def raises(function):
    try:
        function()
    except (ValueError, KeyError):
        return True
    return False


def mutated(name, edit):
    target = WORK / name
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(SMOKE, target, ignore=shutil.ignore_patterns("analysis"))
    edit(target)
    return target


def main():
    if not (SMOKE / "completion.json").exists():
        raise SystemExit("Run python -m training.contribution_study.check_quality_pooling first")
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    data = synthetic_data(np.random.default_rng(11))
    checks = {}

    summary = analyse(SMOKE, data, WORK / "valid")
    pooled = pd.read_csv(SMOKE / "pooled_by_seed.csv").query("policy == 'fixed'").set_index(["variant", "seed"])
    expected_delta = float((pooled.loc[CHANGE, "accuracy"] - pooled.loc[BASELINE, "accuracy"]).mean())
    checks["primary_delta_matches_pooled"] = np.isclose(summary["primary"]["delta_mean"], expected_delta)
    checks["scientific_flag_false_on_synthetic"] = summary["scientific_result"] is False

    mechanism = pd.read_csv(WORK / "valid/mechanism.csv")
    checks["mechanism_full_coverage"] = len(mechanism) == 2 * 160 and mechanism["attention_on_low_q"].between(0, 1 + 1e-6).all()
    checks["confusion_sums_to_160"] = (pd.read_csv(WORK / "valid/confusion.csv")[["tn", "fp", "fn", "tp"]].sum(axis=1) == 160).all()

    flips = pd.read_csv(WORK / "valid/flips.csv")
    predictions = pd.read_csv(SMOKE / "predictions.csv").query("policy == 'fixed'")
    wide = predictions.pivot_table(index="sample_index", columns="variant", values="prediction")
    labels = predictions.drop_duplicates("sample_index").set_index("sample_index")["label"].loc[wide.index]
    checks["flip_counts_match_predictions"] = int(flips["fixed_by_q1"].sum()) == int(((wide[BASELINE] != labels) & (wide[CHANGE] == labels)).sum()) \
        and int(flips["broken_by_q1"].sum()) == int(((wide[BASELINE] == labels) & (wide[CHANGE] != labels)).sum())

    # Alignment: q' used in the analysis must be the prior the Q1 model actually applied. Reload one Q1
    # checkpoint, recompute attention with and without the prior, and check the saved attention ratio.
    checkpoint = torch.load(SMOKE / f"models/{CHANGE}_seed42_subject2.pt", map_location="cpu", weights_only=False)
    saved = np.load(SMOKE / f"{CHANGE}_seed42_subject2_attention.npz")
    model = build_model(checkpoint["variant"], checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    from training.component_ablation.models import PoseDataset
    dataset = PoseDataset(data, saved["indices"], 8, False, 0)
    features = torch.stack([torch.as_tensor(dataset[i][0]) for i in range(len(dataset))])
    with torch.no_grad():
        _, with_prior = model(features)
        model.quality_prior = False
        _, without_prior = model(features)
    q = step_quality(data, saved["indices"], saved["attention"].shape[1])
    implied = without_prior.numpy() * (checkpoint["config"]["quality_epsilon"] + q)
    implied /= implied.sum(axis=1, keepdims=True)
    checks["saved_attention_reproduced_from_checkpoint"] = np.allclose(with_prior.numpy(), saved["attention"], atol=1e-5)
    checks["analysis_q_matches_model_prior"] = np.allclose(implied, saved["attention"], atol=1e-5)

    def incomplete(path):
        value = json.loads((path / "completion.json").read_text(encoding="utf-8"))
        (path / "completion.json").write_text(json.dumps({**value, "complete": False}), encoding="utf-8")
    checks["blocks_incomplete_run"] = raises(lambda: analyse(mutated("incomplete", incomplete), data, WORK / "x1"))

    def wrong_variants(path):
        value = json.loads((path / "completion.json").read_text(encoding="utf-8"))
        (path / "completion.json").write_text(json.dumps({**value, "variants": [BASELINE, "A3_cost_225"]}), encoding="utf-8")
    checks["blocks_wrong_pair"] = raises(lambda: analyse(mutated("wrong_pair", wrong_variants), data, WORK / "x2"))

    def drop_pooled(path):
        frame = pd.read_csv(path / "pooled_by_seed.csv")
        frame.drop(frame[(frame["variant"] == CHANGE) & (frame["policy"] == "fixed")].index).to_csv(path / "pooled_by_seed.csv", index=False)
    checks["blocks_unpaired_rows"] = raises(lambda: analyse(mutated("unpaired", drop_pooled), data, WORK / "x3"))

    def missing_attention(path):
        next(path.glob(f"{CHANGE}_seed42_subject3_attention.npz")).unlink()
    checks["blocks_missing_attention"] = raises(lambda: analyse(mutated("no_attention", missing_attention), data, WORK / "x4"))

    def scientific(path):
        value = json.loads((path / "completion.json").read_text(encoding="utf-8"))
        (path / "completion.json").write_text(json.dumps({**value, "scientific_result": True}), encoding="utf-8")
    checks["blocks_synthetic_data_on_scientific_run"] = raises(lambda: analyse(mutated("scientific", scientific), data, WORK / "x5"))

    for name, value in checks.items():
        print("PASS" if value else "FAIL", name)
    passed = sum(bool(v) for v in checks.values())
    print(f"OVERALL {'PASS' if passed == len(checks) else 'FAIL'} {passed}/{len(checks)}")
    (WORK / "analysis_checks.json").write_text(json.dumps({k: bool(v) for k, v in checks.items()}, indent=2), encoding="utf-8")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
