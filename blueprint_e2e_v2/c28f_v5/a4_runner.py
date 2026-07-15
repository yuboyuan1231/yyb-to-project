"""Compatibility entrypoint for the compact C28F runtime."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

_RUNNER = Path(__file__).resolve(strict=True)
_REPO = _RUNNER.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from blueprint_e2e_v2.c28f_v5.runtime_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
