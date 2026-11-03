"""Regression checks for review DUC-20261028-001 (DUC-REV-01..04): positive and negative cases.

Uses synthetic data and fixtures only; nothing here is experimental evidence and no real input is needed.
Report: .runtime/contribution_study/review_checks.json
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.contribution_study import benchmark_latency
from training.contribution_study.audit_existing import compare
from training.contribution_study.check_quality_pooling import synthetic_data
from training.contribution_study.runner import (LOCKED_INPUTS, make_config, run_experiment, run_neural, signature_hash,
                                                summarize_run, verify_inputs)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / ".runtime/contribution_study/review"


def raises(function, *args):
    try:
        function(*args)
    except (ValueError, RuntimeError):
        return True
    return False


def check_input_guard(checks):
    approved = json.loads(LOCKED_INPUTS.read_text(encoding="utf-8"))["inputs"]
    real = {"provenance": {key: {"path": key, "sha256": value} for key, value in approved.items()}}
    checks["rev01_approved_hashes_pass"] = verify_inputs(real, make_config("full")) == "locked"
    for key in approved:
        changed = {"provenance": {**real["provenance"], key: {"path": key, "sha256": "0" * 64}}}
        missing = {"provenance": {k: v for k, v in real["provenance"].items() if k != key}}
        checks[f"rev01_wrong_{key}_hash_blocked"] = raises(verify_inputs, changed, make_config("smoke"))
        checks[f"rev01_missing_{key}_blocked"] = raises(verify_inputs, missing, make_config("smoke"))
    extra = {"provenance": {**real["provenance"], "extra": {"path": "x", "sha256": "1" * 64}}}
    checks["rev01_unexpected_input_blocked"] = raises(verify_inputs, extra, make_config("smoke"))
    synthetic = synthetic_data(np.random.default_rng(3))
    checks["rev01_synthetic_blocked_in_full"] = raises(verify_inputs, synthetic, make_config("full"))
    checks["rev01_synthetic_smoke_labelled"] = verify_inputs(synthetic, make_config("smoke")) == "synthetic"
    # Guard must fire before any fit: data with a wrong hash and no synthetic label leaves the output untouched.
    unlabelled = {k: v for k, v in synthetic.items() if k != "synthetic"}
    unlabelled["provenance"] = {**real["provenance"], "pose": {"path": "pose", "sha256": "0" * 64}}
    target = OUTPUT / "guard_blocked"
    shutil.rmtree(target, ignore_errors=True)
    checks["rev01_run_blocked_before_train"] = raises(run_experiment, unlabelled, make_config("smoke"), target) \
        and not target.exists()


def fold_paths(folder):
    return sorted(folder.glob("Q*_seed*_subject*.json"))


def edit_reports(folder, change, first_only=True):
    for path in fold_paths(folder)[:1] if first_only else fold_paths(folder):
        report = json.loads(path.read_text(encoding="utf-8"))
        change(report)
        path.write_text(json.dumps(report), encoding="utf-8")


def only_fixed(report):
    report["choices"] = {"fixed": report["choices"]["fixed"]}
    report["metrics"] = [row for row in report["metrics"] if row["policy"] == "fixed"]
    report["predictions"] = [row for row in report["predictions"] if row["policy"] == "fixed"]


def reviewer_repro(report):  # the case reproduced in review: complete=false, fits=0, only fixed, empty metrics
    only_fixed(report)
    report.update(complete=False, fits=0, metrics=[])


def set_field(field, value):
    return lambda report: report.update({field: value})


def change_prediction(field, update):
    def change(report):
        row = report["predictions"][0]
        row[field] = update(row[field])
    return change


def tamper_metric(report):
    report["metrics"][0]["accuracy"] += 0.1


def check_completion(checks):
    data = synthetic_data(np.random.default_rng(11))
    config = make_config("smoke")
    base = OUTPUT / "smoke"
    shutil.rmtree(base, ignore_errors=True)
    run_experiment(data, config, base)
    signature = signature_hash(json.loads((base / "locked_protocol.json").read_text(encoding="utf-8")))
    completion = json.loads((base / "completion.json").read_text(encoding="utf-8"))
    checks["rev02_valid_run_passes"] = completion["complete"] is True and completion["fits"] == completion["fits_expected"] == 32 \
        and completion["scientific_result"] is False and completion["input_guard"] == "synthetic"

    def mutated(name, mutate):
        folder = OUTPUT / f"mutation_{name}"
        shutil.rmtree(folder, ignore_errors=True)
        shutil.copytree(base, folder)
        mutate(folder)
        blocked = raises(summarize_run, data, config, folder, signature, "synthetic") \
            and not (folder / "completion.json").exists()
        shutil.rmtree(folder, ignore_errors=True)
        return blocked

    first = lambda folder, suffix: fold_paths(folder)[0].with_name(fold_paths(folder)[0].stem + suffix)
    mutations = {
        "reviewer_repro_all_reports": lambda f: edit_reports(f, reviewer_repro, first_only=False),
        "complete_false": lambda f: edit_reports(f, set_field("complete", False)),
        "wrong_fits": lambda f: edit_reports(f, set_field("fits", 0)),
        "only_fixed_policy": lambda f: edit_reports(f, only_fixed),
        "empty_metrics": lambda f: edit_reports(f, set_field("metrics", [])),
        "wrong_signature": lambda f: edit_reports(f, set_field("signature_sha256", "0" * 64)),
        "wrong_label": lambda f: edit_reports(f, change_prediction("label", lambda v: 1 - v)),
        "wrong_sample_index": lambda f: edit_reports(f, change_prediction("sample_index", lambda v: v + 1)),
        "prediction_not_threshold": lambda f: edit_reports(f, change_prediction("prediction", lambda v: 1 - v)),
        "nan_score": lambda f: edit_reports(f, change_prediction("score", lambda v: float("nan"))),
        "tampered_fold_metric": lambda f: edit_reports(f, tamper_metric),
        "missing_report": lambda f: fold_paths(f)[0].unlink(),
        "unexpected_report": lambda f: shutil.copy(fold_paths(f)[0], f / "Q9_extra_seed1_subject1.json"),
        "missing_checkpoint": lambda f: first(f, ".pt").unlink(),
        "corrupt_attention": lambda f: first(f, "_attention.npz").write_bytes(b"broken"),
        "missing_history": lambda f: first(f, "_refit_history.csv").unlink(),
    }
    for name, mutate in mutations.items():
        checks[f"rev02_blocked_{name}"] = mutated(name, mutate)

    # Resume must rerun a fold with a broken artifact instead of trusting its report.
    folder = OUTPUT / "resume"
    shutil.rmtree(folder, ignore_errors=True)
    shutil.copytree(base, folder)
    broken = fold_paths(folder)[0]
    broken.with_name(broken.stem + ".pt").unlink()
    run_neural(data, config, folder, "cpu", signature)
    summarize_run(data, config, folder, signature, "synthetic")
    rerun = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
    checks["rev02_resume_reruns_broken_fold"] = rerun["complete"] is True and broken.with_name(broken.stem + ".pt").exists()
    shutil.rmtree(folder, ignore_errors=True)


def check_latency_sync(checks):
    calls = []
    original = torch.cuda.synchronize
    torch.cuda.synchronize = lambda device=None: calls.append(device)
    try:
        benchmark_latency.make_sync("cpu")()
        checks["rev03_cpu_no_cuda_sync"] = calls == []
        for name in ["cuda", "cuda:0"]:
            calls.clear()
            benchmark_latency.make_sync(name)()
            checks[f"rev03_{name.replace(':', '')}_synchronizes_device"] = calls == [torch.device(name)]
        model = torch.nn.Linear(4, 2).eval()
        calls.clear()
        times = benchmark_latency.measure(model, torch.rand(1, 4), 2, 3, torch.device("cpu"))
        checks["rev03_cpu_measure_runs"] = len(times) == 3 and calls == []
    finally:
        torch.cuda.synchronize = original


def check_audit_compare(checks):
    keys, columns = ["variant", "seed", "policy"], ["accuracy"]
    stored = pd.DataFrame({"variant": ["A", "A"], "seed": [1, 2], "policy": ["fixed", "fixed"], "accuracy": [0.9, 0.8]})

    def status(recomputed, stored_frame):
        result = []
        compare(recomputed, stored_frame, keys, columns, "fixture", result)
        return result[0]

    checks["rev04_identical_pass"] = status(stored, stored.copy())["status"] == "PASS"
    nan_stored = stored.assign(accuracy=[np.nan, 0.8])
    checks["rev04_review_case_nan_vs_finite_fails"] = status(stored, nan_stored)["status"] == "FAIL"
    checks["rev04_finite_vs_nan_fails"] = status(nan_stored, stored)["status"] == "FAIL"
    both = status(nan_stored, nan_stored.copy())
    checks["rev04_nan_matches_only_nan"] = both["status"] == "PASS" and both["both_nan_cells"] == 1
    checks["rev04_inf_fails"] = status(stored.assign(accuracy=[np.inf, 0.8]), stored)["status"] == "FAIL"
    checks["rev04_duplicate_key_fails"] = status(pd.concat([stored, stored.iloc[:1]]), stored)["status"] == "FAIL"
    checks["rev04_missing_row_fails"] = status(stored.iloc[:1], stored)["status"] == "FAIL"
    checks["rev04_missing_column_fails"] = status(stored.drop(columns="accuracy"), stored)["status"] == "FAIL"
    checks["rev04_value_diff_fails"] = status(stored.assign(accuracy=[0.9, 0.7]), stored)["status"] == "FAIL"


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    check_audit_compare(checks)
    check_latency_sync(checks)
    check_input_guard(checks)
    check_completion(checks)
    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {"status": status, "checks": checks, "torch": torch.__version__,
              "note": "Fixtures and synthetic smoke only; software regression checks, not experimental results."}
    (OUTPUT / "review_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, value in checks.items():
        print("PASS" if value else "FAIL", key)
    print("OVERALL", status, f"{sum(checks.values())}/{len(checks)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
