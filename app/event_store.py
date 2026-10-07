from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sqlite3


REVIEW_STATES = {"pending", "acknowledged", "false_alarm"}


class LocalNotifications:
    name = "local_only"

    def prepare(self, event):
        return {"channel": self.name, "status": "disabled"}


class EventStore:
    def __init__(self, directory, notifications=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "history.sqlite3"
        self.notifications = notifications or LocalNotifications()
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL,
                    label TEXT NOT NULL, mode TEXT NOT NULL, score REAL, threshold REAL NOT NULL,
                    duration REAL NOT NULL, detection_rate REAL NOT NULL, elapsed_seconds REAL NOT NULL,
                    model_sha256 TEXT NOT NULL, report_path TEXT NOT NULL, video_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL, video_seconds REAL, ended_seconds REAL,
                    score REAL, detection_rate REAL, status TEXT NOT NULL DEFAULT 'pending',
                    reviewed_at TEXT, note TEXT NOT NULL DEFAULT '',
                    notification_channel TEXT NOT NULL, notification_status TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_session ON events(session_id);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, identifier, video, report, preview=None):
        with self.connect() as connection:
            if connection.execute("SELECT id FROM sessions WHERE id=?", (identifier,)).fetchone():
                return
        folder = self.directory / identifier
        if folder.resolve().parent != self.directory.resolve() or folder.name != identifier:
            raise ValueError("Đường dẫn phiên không hợp lệ.")
        folder.mkdir(exist_ok=True)
        video_path = folder / ("video" + Path(video).suffix.lower())
        report_path = folder / "report.json"
        try:
            shutil.copyfile(video, video_path)
            if preview is not None:
                shutil.copyfile(preview, folder / "preview.webm")
            report_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            events = []
            timeline = report.get("timeline") or {}
            for index, event in enumerate(timeline.get("events", [])):
                record = next((record for record in timeline.get("records", [])
                               if abs(record["end_seconds"] - event["alert_at"]) < 1e-6), {})
                events.append({"id": f"{identifier}-window-{index}", "kind": "timeline_alert",
                               "video_seconds": event["alert_at"], "ended_seconds": event["ended_at"],
                               "score": record.get("score"), "detection_rate": record.get("detection_rate")})
            if report["analysis_mode"] == "clip" and report["label"] == "fall":
                events.append({"id": f"{identifier}-clip", "kind": "clip_flag", "video_seconds": None,
                               "ended_seconds": None, "score": report["score"],
                               "detection_rate": report["detection_rate"]})
            with self.connect() as connection:
                connection.execute("""INSERT INTO sessions
                    (id,filename,created_at,label,mode,score,threshold,duration,detection_rate,
                     elapsed_seconds,model_sha256,report_path,video_path)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    identifier, report["filename"], datetime.now(timezone.utc).isoformat(), report["label"],
                    report["analysis_mode"], report["score"], report["threshold"], report["duration"],
                    report["detection_rate"], report["elapsed_seconds"], report["model_sha256"],
                    str(report_path), str(video_path)))
                for event in events:
                    notification = self.notifications.prepare(event)
                    connection.execute("""INSERT INTO events
                        (id,session_id,kind,video_seconds,ended_seconds,score,detection_rate,
                         notification_channel,notification_status) VALUES (?,?,?,?,?,?,?,?,?)""", (
                        event["id"], identifier, event["kind"], event["video_seconds"], event["ended_seconds"],
                        event["score"], event["detection_rate"], notification["channel"], notification["status"]))
        except BaseException:
            shutil.rmtree(folder)
            raise

    def overview(self):
        with self.connect() as connection:
            sessions = [dict(row) for row in connection.execute("""SELECT s.id,s.filename,s.created_at,
                s.label,s.mode,s.score,s.threshold,s.duration,s.detection_rate,s.elapsed_seconds,s.model_sha256,
                (SELECT COUNT(*) FROM events e WHERE e.session_id=s.id) AS event_count
                FROM sessions s ORDER BY s.created_at DESC LIMIT 200""")]
            events = [dict(row) for row in connection.execute("""SELECT e.*,s.filename,s.created_at,s.threshold,
                s.model_sha256 FROM events e JOIN sessions s ON s.id=e.session_id
                ORDER BY s.created_at DESC,e.video_seconds DESC LIMIT 500""")]
            counts = dict(connection.execute("""SELECT COUNT(*) AS sessions,
                COALESCE(SUM(label='insufficient'),0) AS insufficient FROM sessions""").fetchone())
            counts.update(dict(connection.execute("""SELECT COUNT(*) AS events,
                COALESCE(SUM(status='pending'),0) AS pending,
                COALESCE(SUM(status='acknowledged'),0) AS acknowledged,
                COALESCE(SUM(status='false_alarm'),0) AS false_alarm FROM events""").fetchone()))
        return {"sessions": sessions, "events": events, "counts": counts,
                "notifications": {"enabled": False, "channel": "local_only"},
                "limits": {"sessions": 200, "events": 500}}

    def session(self, identifier):
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return dict(row)

    def report(self, identifier):
        return json.loads(Path(self.session(identifier)["report_path"]).read_text(encoding="utf-8"))

    def review(self, identifier, status, note):
        if status not in REVIEW_STATES or not isinstance(note, str) or len(note) > 1000:
            raise ValueError("Trạng thái không hợp lệ hoặc ghi chú quá 1.000 ký tự.")
        with self.connect() as connection:
            updated = connection.execute("UPDATE events SET status=?,note=?,reviewed_at=? WHERE id=?", (
                status, note, datetime.now(timezone.utc).isoformat(), identifier))
            if updated.rowcount != 1:
                raise KeyError(identifier)

    def delete(self, identifier):
        session = self.session(identifier)
        folder = Path(session["report_path"]).parent.resolve()
        if folder.parent != self.directory.resolve() or folder.name != identifier:
            raise ValueError("Đường dẫn phiên không hợp lệ.")
        shutil.rmtree(folder)
        with self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id=?", (identifier,))
