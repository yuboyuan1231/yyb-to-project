
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Small JSON/JSONL helpers for CONQUER-RLEM artifacts."""

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


def open_text_auto(path: str, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def iter_jsonl(path: str, max_rows: Optional[int] = None) -> Iterator[Dict]:
    with open_text_auto(path, "rt") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: str, obj, pretty: bool = True) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2 if pretty else None, ensure_ascii=False)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
