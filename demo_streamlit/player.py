import base64
import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent


def player_height(result):
    if result is None:
        return 830
    return 900 if len(result["events"]) <= 1 else 1030


def player_html(preview=None, result=None, model_name="ME-STGCN-TA · A3"):
    payload = dict(result or {})
    payload["model_name"] = model_name
    payload["video"] = base64.b64encode(preview).decode("ascii") if preview else None
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    for character, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        encoded = encoded.replace(character, escaped)
    template = (DIRECTORY / "player.html").read_text(encoding="utf-8")
    return (template.replace("__STYLE__", (DIRECTORY / "player.css").read_text(encoding="utf-8"))
            .replace("__SCRIPT__", (DIRECTORY / "player.js").read_text(encoding="utf-8"))
            .replace("__PAYLOAD__", encoded))
