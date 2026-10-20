"""Refresh the bundled snapshots in data/history (used when the live APIs are unreachable).

Usage:
    python scripts/build_history.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor_nepal.config import HISTORY_DIR  # noqa: E402
from monitor_nepal.history import sources  # noqa: E402


def main() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(f"wb_{k}", lambda k=k: sources.world_bank(k)) for k in sources.WB_INDICATORS]
    jobs += [(f"owid_{k}", lambda k=k: sources.owid(k)) for k in sources.OWID_SERIES]
    jobs += [(f"climate_{k}", lambda k=k: sources.climate(k)) for k in sources.CLIMATE]
    for name, fn in jobs:
        df = fn()
        if df.attrs.get("snapshot"):
            print(f"{name}: live fetch failed, kept existing snapshot")
            continue
        df.to_csv(HISTORY_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows ({int(df.year.min())}-{int(df.year.max())})")


if __name__ == "__main__":
    main()
