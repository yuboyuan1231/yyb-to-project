"""Repository-level entrypoint for the compact C28F F0/F1 runtime."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from blueprint_e2e_v2.c28f_v5.runtime_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
