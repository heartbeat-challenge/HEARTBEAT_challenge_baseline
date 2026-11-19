import json
import os
import uuid
from pathlib import Path

ROOT = Path(os.environ.get("HEARTBEAT_EVAL_ROOT", "."))
FROZEN = os.environ.get("HEARTBEAT_DUPLEX_REPO")
TYPES = ["response_start", "pause", "cut", "backchannel", "background_speech"]
MODEL_NAME = os.environ.get("HEARTBEAT_MODEL", "venus")
MODELS = {"venus": os.environ.get("HEARTBEAT_BASE_MODEL")}
REF = os.environ.get("HEARTBEAT_REFERENCE_AUDIO")


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2))
    tmp.replace(p)


def rows(p):
    return [json.loads(s) for s in open(p) if s.strip()]
