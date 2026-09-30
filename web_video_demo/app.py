from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
import csv
import io
import json
import logging
import math
import os
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from engine import Engine, ROOT
from features import CHANNELS, EDGES
from batch_evaluation import BatchManager


DEFAULT_RUNTIME = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "PoseLab" / "runtime"
RUNTIME = Path(os.getenv("POSELAB_RUNTIME_DIR", str(DEFAULT_RUNTIME)))
RUNTIME.mkdir(parents=True, exist_ok=True)
EXAMPLE = Path(os.getenv("POSELAB_EXAMPLE_VIDEO", str(ROOT.parent / "data/private/example.mp4")))
MAX_BYTES = 200 * 1024 * 1024
jobs = {}
lock = Lock()
executor = ThreadPoolExecutor(max_workers=1)
engine = None
batch_manager = BatchManager(lambda: engine, executor, lock, RUNTIME,
    lambda: any(job["status"] in {"uploading", "running"} for job in jobs.values()))


@asynccontextmanager
async def lifespan(application):
    global engine
    engine = Engine()
    yield
    executor.shutdown(wait=True, cancel_futures=True)
    for job in jobs.values():
        job["path"].unlink(missing_ok=True)


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.method in {"POST", "DELETE"} and request.headers.get("x-poselab") != "local":
        return JSONResponse({"detail": "Yêu cầu phải xuất phát từ giao diện PoseLab."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/config")
def config():
    return {"model": "ME-STGCN-TA", "version": "06 · corrected angle",
            "threshold": engine.config["decision_threshold"], "channels": CHANNELS,
            "edges": EDGES, "example_available": EXAMPLE.exists(), "max_mb": 200}


@app.get("/api/example")
def example():
    if not EXAMPLE.exists():
        raise HTTPException(404, "Không có video mẫu trên máy.")
    return FileResponse(EXAMPLE, media_type="video/mp4", filename="S1_ADL_04.mp4")


def worker(identifier):
    job = jobs[identifier]
    def progress(value, message):
        with lock:
            job.update(progress=value, message=message)
    def observation(value):
        with lock:
            job["latest_window"] = value
    try:
        result = engine.analyze(job["path"], job["filename"], progress,
                                job["mode"], job["window_seconds"], observation)
        with lock:
            job.update(status="done", progress=100, message="Hoàn tất", result=result)
    except Exception as error:
        logging.exception("Video analysis failed")
        with lock:
            job.update(status="error", message=str(error) if isinstance(error, ValueError) else "Xử lý thất bại. Kiểm tra terminal hoặc thử video khác.")


@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile = File(...), mode: str = Form("timeline"),
                     window_seconds: float = Form(3.0)):
    if mode not in {"clip", "timeline"} or not math.isfinite(window_seconds) or not 2 <= window_seconds <= 6:
        await file.close()
        raise HTTPException(400, "Chọn chế độ clip/timeline và cửa sổ từ 2 đến 6 giây.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".avi"}:
        raise HTTPException(400, "Chọn video MP4, WebM, MOV hoặc AVI.")
    with lock:
        if batch_manager.active() or any(job["status"] in {"uploading", "running"} for job in jobs.values()):
            raise HTTPException(409, "Đang xử lý một video. Hãy chờ hoàn tất.")
        if len(jobs) >= 5:
            raise HTTPException(409, "Đã có 5 phiên. Hãy xóa phiên hiện tại hoặc khởi động lại web.")
        identifier = uuid.uuid4().hex
        path = RUNTIME / (identifier + suffix)
        jobs[identifier] = {"status": "uploading", "progress": 0, "message": "Nhận video", "path": path,
                            "mode": mode, "window_seconds": window_seconds, "latest_window": None,
                            "filename": (file.filename or "video").replace("\\", "/").split("/")[-1]}
    try:
        size = 0
        with path.open("wb") as output:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Video vượt quá 200 MB.")
                output.write(block)
        if size == 0:
            raise HTTPException(400, "File rỗng.")
        with lock:
            jobs[identifier]["status"] = "running"
        executor.submit(worker, identifier)
        return {"id": identifier}
    except BaseException:
        with lock:
            jobs.pop(identifier, None)
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def get_job(identifier):
    with lock:
        if identifier not in jobs:
            raise HTTPException(404, "Phiên không còn tồn tại. Hãy tải lại video.")
        return jobs[identifier].copy()


@app.get("/api/jobs/{identifier}")
def job_status(identifier: str):
    job = get_job(identifier)
    return {key: job[key] for key in ["status", "progress", "message", "latest_window"]}


@app.get("/api/jobs/{identifier}/result")
def result(identifier: str):
    job = get_job(identifier)
    if job["status"] != "done":
        raise HTTPException(409, "Chưa có kết quả.")
    return job["result"]


@app.get("/api/jobs/{identifier}/video")
def job_video(identifier: str):
    job = get_job(identifier)
    return FileResponse(job["path"])


@app.get("/api/jobs/{identifier}/download/{format}")
def download(identifier: str, format: str, onset: float | None = None):
    report = {key: value for key, value in result(identifier).items() if key != "overlay"}
    if onset is not None:
        if not math.isfinite(onset) or not 0 <= onset < report["duration"]:
            raise HTTPException(400, "Mốc ngã phải nằm trong thời lượng video.")
        events = (report.get("timeline") or {}).get("events", [])
        subsequent = [event["alert_at"] for event in events if event["alert_at"] >= onset]
        report["manual_reference"] = {"onset_seconds": onset,
            "first_new_alert_delay_seconds": subsequent[0] - onset if subsequent else None,
            "alerts_before_onset": sum(event["alert_at"] < onset for event in events),
            "note": "User annotation for one event; video-time delay, not wall-clock latency or validated sensitivity."}
    if format == "json":
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        mime = "application/json"
    elif format == "csv":
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        for key, value in report.items():
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            writer.writerow([key, "'" + text if text.startswith(("=", "+", "-", "@")) else text])
        payload = "\ufeff" + stream.getvalue()
        mime = "text/csv"
    elif format == "timeline.csv":
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["start_seconds", "end_seconds", "score", "threshold", "state", "label",
                         "detection_rate", "sampled_detected", "quality_reason", "peak_joint_speed",
                         "peak_abs_torso_angle_degrees", "final_hip_dy", "manual_onset_seconds"])
        for record in (report.get("timeline") or {}).get("records", []):
            evidence = record["evidence"] or {}
            writer.writerow([record[key] for key in ["start_seconds", "end_seconds", "score", "threshold",
                             "state", "label", "detection_rate", "sampled_detected", "quality_reason"]] +
                            [evidence.get(key) for key in ["peak_joint_speed", "peak_abs_torso_angle", "final_hip_dy"]] + [onset])
        payload = "\ufeff" + stream.getvalue()
        mime = "text/csv"
    else:
        raise HTTPException(404, "Định dạng không hỗ trợ.")
    return Response(payload, media_type=mime, headers={"Content-Disposition": f'attachment; filename="poselab-{identifier[:8]}.{format}"'})


@app.delete("/api/jobs/{identifier}")
def delete_job(identifier: str):
    job = get_job(identifier)
    if job["status"] in {"uploading", "running"}:
        raise HTTPException(409, "Chờ xử lý xong trước khi xóa.")
    with lock:
        jobs.pop(identifier, None)
    job["path"].unlink(missing_ok=True)
    return {"deleted": True}


app.include_router(batch_manager.router())
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="web")
