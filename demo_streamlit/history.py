import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from demo_streamlit.reporting import report_data
from demo_streamlit.video import MAX_BYTES


DEFAULT_PATH = Path(__file__).resolve().parents[1] / ".runtime/fallscope/history.sqlite3"
MAX_ENTRIES = 20
MAX_STORAGE_BYTES = 250 * 1024 * 1024


def history_label(entry):
    local_time = datetime.fromisoformat(entry["created"]).astimezone(timezone(timedelta(hours=7)))
    return f'{local_time:%d/%m/%Y %H:%M:%S} (UTC+7) · {entry["model"]} · {entry["video_name"]}'


class HistoryStore:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute("""CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, created TEXT NOT NULL, model TEXT NOT NULL,
                video_name TEXT NOT NULL, result TEXT NOT NULL, preview BLOB,
                result_hash TEXT NOT NULL, preview_hash TEXT NOT NULL)""")
            with connection:
                yield connection
        finally:
            connection.close()

    def entries(self):
        if not self.path.exists():
            return []
        with self._connect() as connection:
            records = connection.execute(
                "SELECT id, created, model, video_name, preview IS NOT NULL FROM sessions ORDER BY created DESC").fetchall()
        return [dict(zip(("id", "created", "model", "video_name", "has_video"), record)) for record in records]

    def save(self, result, model_name, preview=None, include_video=False):
        if include_video and (not preview or len(preview) > MAX_BYTES):
            raise ValueError("Cần preview hợp lệ, tối đa 50 MB, để lưu kèm video.")
        payload = result if include_video else report_data(result)
        encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False)
        media = preview if include_video else None
        size = len(encoded.encode("utf-8")) + len(media or b"")
        if size > 64 * 1024 * 1024:
            raise ValueError("Phiên vượt giới hạn lưu trữ 64 MB.")
        identity = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count, used = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(result) + COALESCE(LENGTH(preview), 0)), 0) FROM sessions").fetchone()
            if count >= MAX_ENTRIES or used + size > MAX_STORAGE_BYTES:
                raise ValueError("Lịch sử đạt giới hạn 20 phiên / 250 MB. Hãy xóa phiên không cần trước khi lưu.")
            connection.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (identity, datetime.now(timezone.utc).isoformat(), model_name,
                                str(result.get("video_name", "Video tải lên"))[:200], encoded, media,
                                hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                                hashlib.sha256(media or b"").hexdigest()))
        return identity

    def load(self, identity):
        if not re.fullmatch(r"[a-f0-9]{32}", identity) or not self.path.exists():
            raise ValueError("Không tìm thấy phiên đã lưu.")
        with self._connect() as connection:
            record = connection.execute(
                "SELECT result, preview, model, result_hash, preview_hash FROM sessions WHERE id=?", (identity,)).fetchone()
        if record is None:
            raise ValueError("Phiên đã bị xóa hoặc không tồn tại.")
        encoded, preview, model, expected_result, expected_preview = record
        if (hashlib.sha256(encoded.encode("utf-8")).hexdigest() != expected_result
                or hashlib.sha256(preview or b"").hexdigest() != expected_preview):
            raise ValueError("Dữ liệu phiên đã thay đổi; không mở kết quả không toàn vẹn.")
        return preview, json.loads(encoded), model

    def delete(self, identity):
        self.load(identity)
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id=?", (identity,))
        with self._connect() as connection:
            connection.execute("VACUUM")
