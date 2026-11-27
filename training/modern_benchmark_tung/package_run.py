"""Package verified folds, execution logs and hashes without altering training outputs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .models import MODEL_IDS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_backups(run, reports):
    backups = [report.parent / Path(read(report)["backup"]).name for report in reports]
    for record in read(run / "models/manifest.json"):
        checkpoint = run / record["file"]
        if digest(checkpoint) != record["sha256"]:
            raise ValueError(f"Checkpoint mismatch: {checkpoint}")
        for backup in backups:
            candidate = backup / record["file"]
            if (candidate.is_file() and digest(candidate) == record["sha256"]
                    and all(digest(candidate.with_name(name)) == digest(checkpoint.with_name(name))
                            for name in ("fold.json", "inner_oof.csv"))):
                break
        else:
            raise ValueError(f"No matching backup: {checkpoint}")


def package(runs, output):
    comparison = runs / "comparison"
    if not (comparison / "latency.csv").is_file():
        raise ValueError("Audit output is missing")
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitattributes").write_bytes(b"* -text whitespace=cr-at-eol\n")
    shutil.copytree(comparison, output / "comparison")
    for model in MODEL_IDS:
        container = output / model
        container.mkdir()
        for mode, fits, folds in (("smoke", 4, 1), ("full", 48, 12)):
            run = runs / f"{model}-{mode}"
            done = read(run / "completion.json")
            if (done["actual_fits"], done["outer_runs"], done["complete"], done["synthetic"],
                    done["scientific_result"]) != (fits, folds, True, False, mode == "full"):
                raise ValueError(f"Unexpected completion: {run}")
            monitoring = sorted(runs.parent.glob(f"{run.name}-monitor-*-measurement.json"))
            monitoring = [p for p in monitoring if Path(read(p)["output"]).resolve() == run.resolve()]
            if not monitoring or read(monitoring[-1])["exit_code"] != 0:
                raise ValueError(f"No completed monitoring record: {run}")
            verify_backups(run, monitoring)
            dest = container / mode
            shutil.copytree(run, dest, ignore=shutil.ignore_patterns(".pending-*"))
            logs = container / "logs"
            logs.mkdir(exist_ok=True)
            for report in monitoring:
                shutil.copy2(report, logs / report.name)
            for telemetry in sorted(runs.parent.glob(f"{run.name}-monitor-*-telemetry.csv")):
                shutil.copy2(telemetry, logs / telemetry.name)
            log = runs / f"{model}-{mode}.log"
            if log.is_file():
                shutil.copy2(log, logs / log.name)
    for name in ("tests.log", "final-tests.log", "audit.log", "input-validation.log",
                 "repository-check.log", "environment-observation.log"):
        if (runs / name).is_file():
            shutil.copy2(runs / name, output / name)


def hashes(root):
    manifest = root / "files_sha256.json"
    return {p.relative_to(root).as_posix(): dict(sha256=digest(p), bytes=p.stat().st_size)
            for p in sorted(root.rglob("*")) if p.is_file() and p != manifest}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--manifest", choices=["write", "verify"])
    args = cli.parse_args()
    if not args.manifest:
        package(args.runs, args.output)
        return
    manifest = args.output / "files_sha256.json"
    inventory = hashes(args.output)
    if args.manifest == "write":
        with manifest.open("x", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(inventory, indent=2) + "\n")
    elif read(manifest) != inventory:
        raise ValueError("Artifact manifest mismatch")
    print(f"PASS: {len(inventory)} artifact hashes")


if __name__ == "__main__":
    main()
