import csv
import hashlib
import json
from pathlib import Path
import re
from collections import defaultdict
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    model = ROOT / "models/deployment"
    manifest = json.loads((model / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        require(hashlib.sha256((model / name).read_bytes()).hexdigest() == expected, f"Model artifact mismatch: {name}")
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    expected_notebooks = {
        "01-prepare-pose-data.ipynb", "02-baseline-motion-svm.ipynb", "03-baseline-stgcn.ipynb",
        "06-final-model-nested-loso.ipynb", "07-component-ablation.ipynb", "08-timing-ablation.ipynb",
        "history/04-proposed-me-stgcn-ta.ipynb", "history/04b-sensitivity-tuning.ipynb",
        "history/05-initial-model-export.ipynb",
    }
    require({path.relative_to(ROOT / "notebooks").as_posix() for path in notebooks} == expected_notebooks, "Unexpected notebook inventory")
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        require(notebook["nbformat"] == 4 and notebook["cells"], f"Invalid notebook: {path.name}")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                require(not cell["outputs"] and cell["execution_count"] is None, f"Unexpected cell output: {path.name}")
    result = ROOT / "results/component_ablation"
    completion = json.loads((result / "completion.json").read_text(encoding="utf-8"))
    require(completion["complete"] and completion["mode"] == "full" and completion["outer_runs"] == 64, "Not a complete full run")
    predictions = rows(result / "predictions.csv")
    errors = rows(result / "errors.csv")
    require(len(predictions) == 7680 and len(rows(result / "fold_metrics.csv")) == 192, "Unexpected result coverage")
    key = lambda row: (row["variant"], row["seed"], row["policy"], row["sample_index"])
    require(len({key(row) for row in predictions}) == len(predictions), "Duplicate predictions")
    require({key(row) for row in predictions if row["label"] != row["prediction"]} == {key(row) for row in errors}, "Errors CSV mismatch")
    require(len(rows(result / "pooled_by_seed.csv")) == 48 and len(rows(result / "seed_mean_std.csv")) == 18, "Unexpected aggregate shape")
    require(len(list((result / "outer_reports").glob("*.json"))) == 64, "Missing outer reports")
    timing = ROOT / "results/timing_ablation"
    timing_completion = json.loads((timing / "completion.json").read_text(encoding="utf-8"))
    require(timing_completion["complete"] and timing_completion["mode"] == "full" and timing_completion["outer_runs"] == 24 and timing_completion["model_fits"] == 96, "Incomplete timing experiment")
    timing_predictions = rows(timing / "predictions.csv")
    require(len(timing_predictions) == 2880 and len({key(row) for row in timing_predictions}) == 2880, "Unexpected timing prediction coverage")
    require({key(row) for row in timing_predictions if row["label"] != row["prediction"]} == {key(row) for row in rows(timing / "errors.csv")}, "Timing errors mismatch")
    groups = defaultdict(list)
    for row in timing_predictions:
        require(int(row["prediction"]) == int(float(row["score"]) >= float(row["threshold"])), "Timing score/decision mismatch")
        groups[(row["variant"], row["policy"], row["seed"])].append(row)
    require(len(groups) == 18, "Missing timing seed/policy groups")
    computed = defaultdict(lambda: defaultdict(list))
    for (variant, policy, seed), group in groups.items():
        require({int(row["sample_index"]) for row in group} == set(range(160)), f"Incomplete samples: {variant}/{policy}/{seed}")
        positives = [row for row in group if row["label"] == "1"]
        negatives = [row for row in group if row["label"] == "0"]
        require(len(positives) == 79 and len(negatives) == 81, "Unexpected timing class counts")
        computed[(variant, policy)]["accuracy"].append(sum(row["label"] == row["prediction"] for row in group) / 160)
        computed[(variant, policy)]["sensitivity"].append(sum(row["prediction"] == "1" for row in positives) / 79)
        computed[(variant, policy)]["specificity"].append(sum(row["prediction"] == "0" for row in negatives) / 81)
    with (timing / "seed_mean_std.csv").open(encoding="utf-8", newline="") as stream:
        aggregate = list(csv.reader(stream))
    for row in aggregate[3:]:
        if not row:
            continue
        for metric, values in computed[(row[0], row[1])].items():
            column = aggregate[0].index(metric)
            require(abs(float(row[column]) - mean(values)) < 1e-10 and abs(float(row[column + 1]) - stdev(values)) < 1e-10, f"Timing aggregate mismatch: {metric}")
    require([path.name for path in (ROOT / "docs").glob("*.docx")] == ["bao-cao.docx"], "Keep one current Word report")
    for path in ROOT.rglob("*.md"):
        if any(part in {".git", ".runtime", ".venv"} for part in path.parts):
            continue
        for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            require((path.parent / target.split("#", 1)[0]).exists(), f"Broken document link: {path.relative_to(ROOT)} → {target}")
    require(not list((ROOT / "notebooks").rglob("*.zip")), "Unexpected input archives")
    print("PASS: unchanged model hashes, 9 clean notebooks, one Word report, component/timing predictions and aggregates, document links")


if __name__ == "__main__":
    main()
