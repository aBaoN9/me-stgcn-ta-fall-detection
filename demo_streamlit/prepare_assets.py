import urllib.request

from demo_streamlit.data import ROOT, digest


POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task")
POSE_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
POSE_PATH = ROOT / ".runtime/fallscope-assets/pose_landmarker_full.task"


def verify_pose_asset():
    if not POSE_PATH.is_file():
        raise FileNotFoundError("Run python -m demo_streamlit.prepare_assets first")
    if digest(POSE_PATH) != POSE_SHA256:
        raise ValueError("MediaPipe asset SHA-256 mismatch")
    return POSE_PATH


def prepare():
    if not POSE_PATH.exists():
        POSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = POSE_PATH.with_suffix(".download")
        try:
            with urllib.request.urlopen(POSE_URL, timeout=60) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
                    if target.tell() > 20 * 1024 * 1024:
                        raise ValueError("Unexpected pose asset size")
            if digest(temporary) != POSE_SHA256:
                raise ValueError("Downloaded pose asset SHA-256 mismatch")
            temporary.replace(POSE_PATH)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified MediaPipe full v1: {verify_pose_asset()}")


if __name__ == "__main__":
    prepare()
