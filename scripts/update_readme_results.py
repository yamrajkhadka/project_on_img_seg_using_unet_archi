"""Write the evaluation numbers into README.md (between <!-- EVALUATION --> markers) from models/*.json.

Usage:
    python scripts/update_readme_results.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- EVALUATION -->", "<!-- /EVALUATION -->"


def table() -> str:
    ev = json.loads((ROOT / "models" / "evaluation.json").read_text())
    log_path = ROOT / "models" / "finetune_log.json"
    lines = [START, "", "**Held-out evaluation** — " + ev["description"], ""]
    lines += ["| Imagery | Reference | Original model | Nepal-adapted |", "|---|---|---|---|"]
    res = ev["results"]
    keys = sorted({(r["source"], r["reference"]) for r in res})
    for src, ref in keys:
        cells = []
        for model in ("paper", "nepal"):
            r = next((x for x in res if x["model_key"] == model and x["source"] == src and x["reference"] == ref), None)
            cells.append("—" if r is None else f"{r['overall'] * 100:.0f}% (κ {r['kappa']:.2f}, mIoU {r['miou']:.2f})")
        lines.append(f"| {src} | {ref} | {cells[0]} | {cells[1]} |")
    ir = ev.get("inter_reference")
    if ir:
        lines += ["", f"Pixel agreement (Cohen's κ, mean IoU over the six land classes). For scale, the two reference maps "
                      f"agree with each other on {ir['overall'] * 100:.0f}% of the same pixels (κ {ir['kappa']:.2f}) — the "
                      "practical ceiling for any model scored against them. Impact Observatory also supplied the "
                      "training labels, so ESA WorldCover is the fairer test."]
    if log_path.exists():
        log = json.loads(log_path.read_text())
        best = log["epochs"][log["best_epoch"] - 1]["val"]["all"]["miou"]
        base = log["baseline"]["all"]["miou"]
        done, planned = len(log["epochs"]), log.get("epochs_planned") or len(log["epochs"])
        status = (f" This is a **preliminary checkpoint** ({done} of {planned} planned epochs); finishing the run "
                  "(`scripts/finetune_nepal.py` resumes automatically, e.g. on a free Kaggle GPU) is expected to "
                  "improve it further." if done < planned else "")
        lines += ["", f"Validation mean IoU against the weak labels on the held-out districts: {base:.3f} for the original "
                      f"weights → {best:.3f} after fine-tuning (epoch {log['best_epoch']}).{status}"]
    lines += ["", END]
    return "\n".join(lines)


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = table()
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.S)
    elif START in text:
        text = text.replace(START, block)
    else:
        raise SystemExit("README has no <!-- EVALUATION --> marker")
    readme.write_text(text)
    print(block)


if __name__ == "__main__":
    main()
