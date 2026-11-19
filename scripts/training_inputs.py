import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request


SOURCE_COMMIT = "afc7370c324aabec1a6d1076b19d8247c818d96c"
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/aBaoN9/me-stgcn-ta-fall-detection/"
    f"{SOURCE_COMMIT}/data/gmdcsa24"
)
FILES = (
    ("gmdcsa24_pose_32.npz", 2432871,
     "598638f33066c4b6487dbaa1ccbfcc98a96143f8deae94354278f635324633b4"),
    ("gmdcsa24_features_32.npz", 12606841,
     "4a255c925adbcfb8364eb413dd492c72d2a0cfcdd5074082f470b0aa836e2848"),
    ("gmdcsa24_pose_manifest.csv", 26558,
     "5c41ac710d580a648d6768c5d7ceb2a5b95767f23069519ed5b98737e59417f2"),
    ("gmdcsa24_cross_subject_folds.csv", 34369,
     "774f0ac28b8342e9c8080af075fcda8227ac6a036124514f6c09f0a375b4fbba"),
)


def verify_file(path, expected_size, expected_hash):
    if path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch: {path}; expected {expected_size} bytes")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch: {path}; file left unchanged")


def download_file(root, filename, expected_size, expected_hash):
    destination = root / filename
    request = urllib.request.Request(
        f"{SOURCE_ROOT}/{filename}", headers={"User-Agent": "gmdcsa24-input-check"}
    )
    with tempfile.TemporaryDirectory(prefix=".input-download-", dir=root) as staging:
        temporary = Path(staging) / filename
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        verify_file(temporary, expected_size, expected_hash)
        created = False
        try:
            with temporary.open("rb") as source:
                with destination.open("xb") as target:
                    created = True
                    shutil.copyfileobj(source, target)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise


def prepare_inputs(root, download=False):
    root = Path(root).resolve()
    missing = []
    for filename, expected_size, expected_hash in FILES:
        path = root / filename
        if path.is_symlink():
            raise ValueError(f"Refusing symbolic link: {path}")
        if path.exists():
            verify_file(path, expected_size, expected_hash)
        else:
            missing.append((filename, expected_size, expected_hash))
    if missing and not download:
        names = ", ".join(item[0] for item in missing)
        raise ValueError(f"Missing inputs: {names}. Use --download or copy the exact files.")
    if missing:
        root.mkdir(parents=True, exist_ok=True)
        for filename, expected_size, expected_hash in missing:
            download_file(root, filename, expected_size, expected_hash)
    for filename, expected_size, expected_hash in FILES:
        verify_file(root / filename, expected_size, expected_hash)
    return root


def main():
    parser = argparse.ArgumentParser(
        description="Verify four locked GMDCSA-24 inputs; optionally download missing files."
    )
    parser.add_argument(
        "--input-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "gmdcsa24",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download missing files from the pinned public data commit; never overwrite.",
    )
    args = parser.parse_args()
    try:
        root = prepare_inputs(args.input_root, download=args.download)
    except (OSError, ValueError, urllib.error.URLError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    for filename, expected_size, expected_hash in FILES:
        print(f"OK {filename} {expected_size} bytes {expected_hash}")
    print(f"PASS: four input files verified in {root}")
    print("This verifies input bytes only, not the training environment or a training run.")


if __name__ == "__main__":
    main()
