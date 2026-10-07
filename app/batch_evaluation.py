from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import logging
import math
import re
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from motion import CONFIRM_WINDOWS, STEP_SECONDS


MAX_FILES = 20
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_TOTAL_BYTES = 600 * 1024 * 1024
PROTOCOL = (
    "One labeled fall event per FALL clip. Only clips with every evaluated window valid enter the confusion matrix. "
    "FALL also requires a human onset and a decision at/after onset. Match the first NEW alert starting at/after "
    "onset until EOF; earlier alerts do not count as detection. No timeliness deadline is applied. Warmup and "
    "the tail after the last decision are not continuously observed. Metrics describe this subset, not deployment safety."
)


def evaluate_report(report, annotation):
    records = report["timeline"]["records"]
    events = report["timeline"]["events"]
    valid = sum(record["score"] is not None for record in records)
    onset = annotation["onset_seconds"]
    expected = annotation["label"]
    after = [event["alert_at"] for event in events if onset is not None and event["alert_at"] >= onset]
    result = {
        "duration": report["duration"], "whole_clip_label": report["label"], "whole_clip_score": report["score"],
        "pose_rate": report["detection_rate"], "window_count": len(records), "valid_windows": valid,
        "has_alert": bool(events), "alert_count": len(events),
        "first_alert_seconds": events[0]["alert_at"] if events else None,
        "matched_alert_seconds": after[0] if after else None,
        "delay_seconds": after[0] - onset if after else None,
        "early_alerts": sum(event["alert_at"] < onset for event in events) if onset is not None else 0,
        "onset_during_warmup": bool(onset is not None and records and onset < records[0]["end_seconds"]),
        "last_decision_seconds": records[-1]["end_seconds"] if records else None,
        "model_sha256": report["model_sha256"], "warnings": report["warnings"],
        "events": events, "threshold": report["threshold"],
        "timeline_records": [{key: record[key] for key in ["start_seconds", "end_seconds", "score", "state",
            "label", "detection_rate", "sampled_detected", "quality_reason"]} for record in records],
    }
    if onset is not None and onset >= report["duration"]:
        outcome, reason = "annotation_error", "Mốc ngã nằm ngoài thời lượng video."
    elif not records:
        outcome, reason = "too_short", "Chưa tích lũy đủ cửa sổ; không có quyết định theo thời gian."
    elif valid != len(records):
        outcome, reason = "incomplete", "Có cửa sổ không đủ pose; loại khỏi ma trận, vẫn giữ các cảnh báo quan sát được."
    elif expected == "adl":
        outcome = "FP" if events else "TN"
        reason = "Có cảnh báo trên video không ngã." if events else "Không có cảnh báo trong các cửa sổ đã đánh giá; không phải chứng nhận an toàn."
    elif onset is None:
        outcome, reason = "needs_onset", "Chưa có mốc ngã do người dùng xác nhận; không ghép cảnh báo với sự kiện."
    elif not any(record["end_seconds"] >= onset for record in records):
        outcome, reason = "no_post_onset_window", "Không có quyết định tại/sau mốc ngã; chưa đủ cơ hội đánh giá."
    else:
        outcome = "TP" if after else "FN"
        reason = "Có đợt cảnh báo mới tại/sau mốc ngã." if after else "Không có đợt cảnh báo mới tại/sau mốc ngã; cảnh báo sớm không được ghép thành đúng."
    if outcome != "TP":
        result["matched_alert_seconds"] = None
        result["delay_seconds"] = None
    result.update(outcome=outcome, reason=reason, eligible=outcome in {"TP", "FP", "FN", "TN"})
    return result


def summarize(rows):
    counts = Counter(row.get("outcome", "pending") for row in rows)
    confusion = {key: counts[key] for key in ["TP", "FP", "FN", "TN"]}
    eligible = sum(confusion.values())
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    delays = [row["delay_seconds"] for row in rows if row.get("outcome") == "TP"]
    return {
        "total": len(rows), "finished": sum(row["status"] in {"done", "error", "cancelled"} for row in rows),
        "eligible": eligible, "excluded_or_pending": len(rows) - eligible,
        "outcomes": dict(counts), "confusion": confusion,
        "coverage": ratio(eligible, len(rows)),
        "accuracy": ratio(counts["TP"] + counts["TN"], eligible),
        "sensitivity": ratio(counts["TP"], counts["TP"] + counts["FN"]),
        "specificity": ratio(counts["TN"], counts["TN"] + counts["FP"]),
        "precision": ratio(counts["TP"], counts["TP"] + counts["FP"]),
        "mean_detected_delay_seconds": sum(delays) / len(delays) if delays else None,
        "observed_adl_alert_clips": sum(row["label"] == "adl" and row.get("has_alert", False) for row in rows),
        "note": "Rates only cover eligible clips. Mean delay covers TP clips only; it excludes misses. Missing denominators return null, not zero.",
    }


class BatchManager:
    def __init__(self, engine, executor, lock, runtime, single_busy):
        self.engine, self.executor, self.lock = engine, executor, lock
        self.runtime, self.single_busy = runtime, single_busy
        self.reports = runtime.parent / "batch_reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        self.batches = {}

    def active(self):
        return any(batch["report"]["status"] in {"uploading", "running"} for batch in self.batches.values())

    def snapshot(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise HTTPException(404, "Không có lượt đánh giá này.")
        with self.lock:
            if identifier in self.batches:
                report = deepcopy(self.batches[identifier]["report"])
            else:
                path = self.reports / f"{identifier}.json"
                if not path.exists():
                    raise HTTPException(404, "Không có lượt đánh giá này.")
                report = json.loads(path.read_text(encoding="utf-8"))
                if report["status"] in {"uploading", "running"}:
                    report["status"] = "interrupted"
                    for row in report["rows"]:
                        if row["status"] in {"pending", "running"}:
                            row.update(status="error", outcome="error", reason="Phiên server bị gián đoạn; cần chạy lại.")
        report["summary"] = summarize(report["rows"])
        return report

    def persist(self, identifier):
        report = self.snapshot(identifier)
        temporary = self.reports / f"{identifier}.pending"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.reports / f"{identifier}.json")

    def clean_path(self, path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logging.exception("Could not clean batch temporary video: %s", path.name)

    def run(self, identifier):
        batch = self.batches[identifier]
        report = batch["report"]
        try:
            for row, path in zip(report["rows"], batch["paths"]):
                with self.lock:
                    cancelled = report["cancel_requested"]
                    row["status"] = "cancelled" if cancelled else "running"
                    if cancelled:
                        row.update(outcome="cancelled", reason="Đã dừng trước khi xử lý video này.")
                if cancelled:
                    self.clean_path(path)
                    continue
                def progress(value, message):
                    with self.lock:
                        row.update(progress=value, message=message)
                try:
                    analyzed = self.engine().analyze(path, row["filename"], progress, "timeline", report["config"]["window_seconds"])
                    evaluated = evaluate_report(analyzed, row)
                    with self.lock:
                        row.update(evaluated, status="done", progress=100)
                except Exception as error:
                    logging.exception("Batch video failed")
                    with self.lock:
                        row.update(status="error", outcome="error", reason=str(error) if isinstance(error, ValueError) else "Xử lý lỗi; xem terminal.")
                finally:
                    self.clean_path(path)
                self.persist(identifier)
            with self.lock:
                report["status"] = "cancelled" if any(row["status"] == "cancelled" for row in report["rows"]) else "done"
                report["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.persist(identifier)
        except Exception:
            logging.exception("Batch run interrupted")
            with self.lock:
                report["status"] = "error"
                for row in report["rows"]:
                    if row["status"] in {"pending", "running"}:
                        row.update(status="error", outcome="error", reason="Lượt xử lý bị gián đoạn; các kết quả trước vẫn được giữ.")
            try:
                self.persist(identifier)
            except OSError:
                logging.exception("Cannot save batch report")
        finally:
            for path in batch["paths"]:
                self.clean_path(path)

    def router(self):
        router = APIRouter()

        @router.post("/api/batches", status_code=202)
        async def create(files: list[UploadFile] = File(...), annotations: str = Form(...),
                         window_seconds: float = Form(3.0), purpose: str = Form("development"), note: str = Form("")):
            identifier, paths = None, []
            try:
                if not 1 <= len(files) <= MAX_FILES or len(annotations) > 20000:
                    raise HTTPException(400, "Chọn từ 1 đến 20 video.")
                if not math.isfinite(window_seconds) or not 2 <= window_seconds <= 6 or purpose not in {"development", "held_out"} or len(note) > 500:
                    raise HTTPException(400, "Cấu hình không hợp lệ.")
                try:
                    labels = json.loads(annotations)
                except (ValueError, TypeError):
                    raise HTTPException(400, "Danh sách nhãn không hợp lệ.")
                if not isinstance(labels, list) or len(labels) != len(files):
                    raise HTTPException(400, "Nhãn phải khớp từng file theo thứ tự.")
                rows = []
                for index, (file, label) in enumerate(zip(files, labels)):
                    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
                    if Path(filename).suffix.lower() not in {".mp4", ".webm", ".mov", ".avi"}:
                        raise HTTPException(400, "Chỉ nhận MP4, WebM, MOV, AVI.")
                    if not isinstance(label, dict) or not isinstance(label.get("label"), str) or label["label"] not in {"adl", "fall"} or label.get("filename") != filename:
                        raise HTTPException(400, "Chọn nhãn thật cho từng video; không đoán từ tên file.")
                    onset = label.get("onset_seconds")
                    if onset is not None and (type(onset) not in {int, float} or not math.isfinite(onset) or onset < 0 or label["label"] != "fall"):
                        raise HTTPException(400, "Mốc ngã phải là số không âm và chỉ dùng cho nhãn FALL.")
                    rows.append({"index": index, "filename": filename, "label": label["label"], "onset_seconds": onset,
                                 "status": "pending", "progress": 0, "outcome": "pending"})
                with self.lock:
                    if self.active() or self.single_busy():
                        raise HTTPException(409, "Đang có video/lượt đánh giá chạy. Hãy chờ hoàn tất.")
                    for key in list(self.batches):
                        if len(self.batches) < 3:
                            break
                        if self.batches[key]["report"]["status"] not in {"uploading", "running"}:
                            self.batches.pop(key)
                    identifier = uuid.uuid4().hex
                    report = {"id": identifier, "status": "uploading", "created_at": datetime.now(timezone.utc).isoformat(),
                        "purpose": purpose, "note": note, "cancel_requested": False, "rows": rows, "protocol": PROTOCOL,
                        "config": {"model": "ME-STGCN-TA", "window_seconds": window_seconds, "step_seconds": STEP_SECONDS,
                                   "confirm_windows": CONFIRM_WINDOWS, "threshold": float(self.engine().config["decision_threshold"]),
                                   "model_sha256": self.engine().signature}}
                    self.batches[identifier] = {"report": report, "paths": paths}
                total = 0
                for row, file in zip(rows, files):
                    path = self.runtime / f"batch-{identifier}-{row['index']}{Path(row['filename']).suffix.lower()}"
                    paths.append(path)
                    size = 0
                    with path.open("wb") as output:
                        while block := await file.read(1024 * 1024):
                            total += len(block)
                            size += len(block)
                            if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                                raise HTTPException(413, "Giới hạn 200 MB/video và 600 MB/lượt.")
                            output.write(block)
                    if size == 0:
                        raise HTTPException(400, "Có file rỗng.")
                with self.lock:
                    report["status"] = "running"
                self.persist(identifier)
                self.executor.submit(self.run, identifier)
                return {"id": identifier}
            except BaseException:
                if identifier:
                    with self.lock:
                        self.batches.pop(identifier, None)
                for path in paths:
                    self.clean_path(path)
                raise
            finally:
                for file in files:
                    await file.close()

        @router.get("/api/batches")
        def recent():
            paths = sorted(self.reports.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:10]
            results = []
            for path in paths:
                try:
                    report = self.snapshot(path.stem)
                    results.append({key: report[key] for key in ["id", "status", "created_at", "purpose", "note", "summary"]})
                except (ValueError, OSError, HTTPException):
                    logging.warning("Cannot read batch report: %s", path.name)
            return results

        @router.get("/api/batches/{identifier}")
        def get_report(identifier: str):
            return self.snapshot(identifier)

        @router.post("/api/batches/{identifier}/cancel")
        def cancel(identifier: str):
            with self.lock:
                if identifier not in self.batches or self.batches[identifier]["report"]["status"] != "running":
                    raise HTTPException(409, "Không có lượt đang chạy để dừng.")
                self.batches[identifier]["report"]["cancel_requested"] = True
            return {"message": "Sẽ dừng sau video đang xử lý; giữ kết quả đã có."}

        @router.get("/api/batches/{identifier}/download/{format}")
        def download(identifier: str, format: str):
            report = self.snapshot(identifier)
            if format == "json":
                content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                mime = "application/json"
            elif format == "csv":
                fields = ["index", "filename", "label", "onset_seconds", "status", "outcome", "reason", "eligible",
                    "duration", "has_alert", "alert_count", "first_alert_seconds", "matched_alert_seconds", "delay_seconds",
                    "early_alerts", "onset_during_warmup", "pose_rate", "window_count", "valid_windows", "whole_clip_label", "whole_clip_score"]
                stream = io.StringIO()
                writer = csv.writer(stream)
                writer.writerow(fields + ["purpose", "window_seconds", "step_seconds", "confirm_windows", "threshold", "model_sha256"])
                for row in report["rows"]:
                    values = [row.get(field) for field in fields] + [report["purpose"]] + [report["config"][field] for field in ["window_seconds", "step_seconds", "confirm_windows", "threshold", "model_sha256"]]
                    writer.writerow(["'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value for value in values])
                content, mime = "\ufeff" + stream.getvalue(), "text/csv"
            else:
                raise HTTPException(404, "Chỉ hỗ trợ JSON/CSV.")
            return Response(content, media_type=mime, headers={"Content-Disposition": f'attachment; filename="poselab-batch-{identifier[:8]}.{format}"'})

        return router
