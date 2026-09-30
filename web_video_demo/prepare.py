from pathlib import Path
import hashlib
import urllib.request


ROOT = Path(__file__).resolve().parent
POSE_MODEL = ROOT / "assets" / "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"


def prepare():
    POSE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if not POSE_MODEL.exists():
        temporary = POSE_MODEL.with_suffix(".download")
        print("Đang tải MediaPipe Pose Landmarker Full chính thức...", flush=True)
        with urllib.request.urlopen(MODEL_URL, timeout=90) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if temporary.stat().st_size < 1000000:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Tải model không đầy đủ. Hãy chạy lại.")
        temporary.replace(POSE_MODEL)
    checksum = hashlib.sha256(POSE_MODEL.read_bytes()).hexdigest()
    if checksum != MODEL_SHA256:
        raise RuntimeError("Model MediaPipe khác bản Full v1 đã kiểm tra. Không chạy suy luận với asset này.")
    print("MediaPipe asset OK | SHA256:", checksum)


if __name__ == "__main__":
    prepare()
