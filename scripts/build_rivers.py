"""Build the bundled river climatology (2004-2023) used to judge today's flows.

For each river: day-of-year percentiles of daily discharge (10th, 50th, 90th, pooled over a ±7-day window across
20 years), flood levels from annual maxima (2-year = median annual maximum, 10-year = 90th percentile), and the
annual peak/mean series.
Requests are spaced out to respect Open-Meteo's fair-use limits.

Usage:
    python scripts/build_rivers.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor_nepal.config import HISTORY_DIR  # noqa: E402
from monitor_nepal.live import rivers  # noqa: E402

START, END = "2004-01-01", "2023-12-31"
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "rivers"


def main() -> None:
    clim_rows, thr_rows, annual_rows = [], [], []
    for i, r in enumerate(rivers.RIVERS):
        if i and not (CACHE / f"{r.key}_{START}_{END}.csv").exists():
            time.sleep(90)
        raw = CACHE / f"{r.key}_{START}_{END}.csv"
        if raw.exists():
            df = pd.read_csv(raw, parse_dates=["date"])
        else:
            df = rivers.history(r, START, END)
            raw.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw, index=False)
        df = df.dropna()
        df["doy"] = df.date.dt.dayofyear.clip(upper=365)
        for doy in range(1, 366):
            window = ((df.doy - doy + 182) % 365 - 182).abs() <= 7
            q = df.discharge[window]
            clim_rows.append({"river": r.key, "doy": doy, "p10": q.quantile(0.10), "p50": q.quantile(0.50),
                              "p90": q.quantile(0.90)})
        annual_max = df.groupby(df.date.dt.year).discharge.max()
        yearly = df.groupby(df.date.dt.year).agg(peak=("discharge", "max"), mean=("discharge", "mean"))
        peak_day = df.loc[df.groupby(df.date.dt.year).discharge.idxmax(), "date"]
        for (year, row), day in zip(yearly.iterrows(), peak_day):
            annual_rows.append({"river": r.key, "year": int(year), "peak": round(float(row.peak), 1),
                                "mean": round(float(row["mean"]), 1), "peak_date": day.date().isoformat()})
        thr_rows.append({"river": r.key, "q2": float(np.median(annual_max)), "q10": float(annual_max.quantile(0.9)),
                         "mean": float(df.discharge.mean()), "years": int(annual_max.size)})
        print(f"{r.name}: mean {df.discharge.mean():,.0f} m3/s, 2-yr flood {thr_rows[-1]['q2']:,.0f}, "
              f"10-yr {thr_rows[-1]['q10']:,.0f} ({len(df)} days)", flush=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clim_rows).round(2).to_csv(HISTORY_DIR / "rivers_climatology.csv", index=False)
    pd.DataFrame(thr_rows).round(1).to_csv(HISTORY_DIR / "rivers_thresholds.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(HISTORY_DIR / "rivers_annual.csv", index=False)


if __name__ == "__main__":
    main()
