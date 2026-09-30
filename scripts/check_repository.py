import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    model = ROOT / "models/deployment_06"
    manifest = json.loads((model / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        require(hashlib.sha256((model / name).read_bytes()).hexdigest() == expected, f"Model artifact mismatch: {name}")
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    require(len(notebooks) == 8, "Need notebook 01–07 and 04b")
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        require(notebook["nbformat"] == 4 and notebook["cells"], f"Invalid notebook: {path.name}")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                require(not cell["outputs"] and cell["execution_count"] is None, f"Unexpected cell output: {path.name}")
    result = ROOT / "results/notebook_07"
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
    for path in ROOT.rglob("*.md"):
        if any(part in {".git", ".runtime", ".venv"} for part in path.parts):
            continue
        for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            require((path.parent / target.split("#", 1)[0]).exists(), f"Broken document link: {path.relative_to(ROOT)} → {target}")
    require(not list((ROOT / "notebooks").glob("*.zip")), "Unexpected private archives")
    print("PASS: model hashes, 8 clean notebooks, 64 full reports, 7680 unique predictions, matching errors and document links")


if __name__ == "__main__":
    main()
