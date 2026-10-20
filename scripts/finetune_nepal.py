"""Fine-tune the paper's U-Net on the weakly-labelled Nepal dataset.

* starts from the published weights (deepglobe_unet_jay.keras), same architecture;
* uses the paper's composite loss - 0.4 x weighted categorical cross-entropy + 0.6 x focal Tversky
  (alpha 0.7, beta 0.3, gamma 0.75) - with unlabeled pixels (255) ignored;
* class weights follow the paper's inverse-frequency recipe, softened with a square root because
  Nepal's rare classes (water, urban) are far rarer than DeepGlobe's;
* validates on districts never seen in training and keeps the epoch with the best mean IoU;
* exports ONNX (same NHWC interface as the original), a .keras file loadable with Keras, and a JSON log.

Usage:
    python scripts/finetune_nepal.py --data /data/nepal_lc --out models/ --epochs 14
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unet_torch import UNet, export_onnx, load_keras, save_keras  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NUM_CLASSES = 7
IGNORE = 255


def focal_tversky(probs, onehot, valid, alpha=0.7, beta=0.3, gamma=0.75):
    """Paper's focal Tversky loss (per image, per class, then averaged) with an ignore mask."""
    p = probs.clamp(1e-7, 1.0) * valid
    t = onehot * valid
    tp = (t * p).sum((2, 3))
    fp = ((1 - t) * p).sum((2, 3))
    fn = (t * (1 - p)).sum((2, 3))
    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return torch.pow(1 - tversky, gamma).mean()


def weighted_ce(logits, target, weights):
    return F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE)


def augment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometric (flips, 90-degree turns) + photometric jitter on a batch (N,3,H,W)/(N,H,W)."""
    if torch.rand(1) < 0.5:
        x, y = x.flip(3), y.flip(2)
    if torch.rand(1) < 0.5:
        x, y = x.flip(2), y.flip(1)
    k = int(torch.randint(0, 4, (1,)))
    if k:
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    n = x.shape[0]
    dev = x.device
    bright = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3
    mean = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * bright
    gray = x.mean(1, keepdim=True)
    sat = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3
    x = (x - gray) * sat + gray
    gamma = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4)
    x = x.clamp(1e-4, 1).pow(gamma)
    return x.clamp(0, 1), y


@torch.no_grad()
def evaluate(model, images, labels, sources, device, batch=32):
    model.eval()
    cms = {}
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255
        pred = model(x).argmax(1).cpu().numpy()
        lab = labels[i:i + batch]
        for j in range(len(pred)):
            ok = lab[j] < NUM_CLASSES - 1  # score the six land classes
            cm = cms.setdefault(sources[i + j], np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64))
            np.add.at(cm, (lab[j][ok], pred[j][ok]), 1)
    out = {}
    total = sum(cms.values())
    for name, cm in list(cms.items()) + [("all", total)]:
        inter = np.diag(cm)[:6]
        union = cm.sum(0)[:6] + cm.sum(1)[:6] - inter
        iou = inter / np.maximum(union, 1)
        present = cm.sum(1)[:6] > 0
        out[name] = {"oa": float(np.trace(cm) / max(1, cm.sum())), "miou": float(iou[present].mean()),
                     "iou": [round(float(v), 4) for v in iou]}
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--keras", default=str(ROOT / "deepglobe_unet_jay.keras"))
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--checkpoint", default=None, help="checkpoint file (default: <out>/finetune_checkpoint.pt); "
                                                       "training resumes from it automatically")
    ap.add_argument("--no-export", action="store_true",
                    help="only train and checkpoint (e.g. on Kaggle); export later with --export-only")
    ap.add_argument("--export-only", action="store_true",
                    help="write ONNX + .keras + log from the checkpoint's best epoch, without training")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out / "finetune_checkpoint.pt"
    if args.export_only:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        export(ck["best_state"], ck["log"], ck["best"], args.keras, out)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    data = Path(args.data)
    tr, va = np.load(data / "train.npz"), np.load(data / "val.npz")
    xtr, ytr = tr["images"], tr["labels"]
    xva, yva = va["images"], va["labels"]
    sva = [r["source"] for r in csv.DictReader(open(data / "val.csv"))]
    print(f"train {len(xtr)} / val {len(xva)} patches on {device}", flush=True)

    counts = np.bincount(ytr[ytr != IGNORE].ravel(), minlength=NUM_CLASSES).astype(np.float64)
    w = np.sqrt(counts.max() / (counts + 1e-6))
    w = np.minimum(w, 20.0)
    w = w / w.sum() * NUM_CLASSES
    weights = torch.tensor(w, dtype=torch.float32, device=device)
    print("class pixel share", np.round(counts / counts.sum() * 100, 2), "weights", np.round(w, 3), flush=True)

    # resuming needs only the architecture; a fresh run starts from the paper's Keras weights
    model = (UNet() if ckpt_path.exists() else load_keras(args.keras)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(xtr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    start_epoch, best, best_state = 1, -1.0, None
    if ckpt_path.exists():  # resume an interrupted run
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        planned = ck["log"].get("epochs_planned")
        if planned and planned != args.epochs:
            raise SystemExit(f"The checkpoint belongs to a {planned}-epoch run (its learning-rate schedule is fixed); "
                             f"resume with --epochs {planned}.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        start_epoch, best, best_state, log = ck["epoch"] + 1, ck["best"], ck["best_state"], ck["log"]
        print(f"resumed after epoch {ck['epoch']} (best val mIoU {best:.3f})", flush=True)
    else:
        base = evaluate(model, xva, yva, sva, device)
        print("epoch 0 (original weights):", json.dumps({k: round(v["miou"], 3) for k, v in base.items()}),
              flush=True)
        log = {"baseline": base, "epochs": [], "class_weights": w.tolist(), "train_patches": int(len(xtr)),
               "val_patches": int(len(xva)), "device": str(device), "epochs_planned": args.epochs}
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        t0, perm, running = time.time(), np.random.permutation(len(xtr)), 0.0
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * args.batch:(s + 1) * args.batch])
            x = torch.from_numpy(xtr[idx]).to(device).permute(0, 3, 1, 2).float() / 255
            y = torch.from_numpy(ytr[idx].astype(np.int64)).to(device)
            x, y = augment(x, y)
            logits = model(x)
            valid = (y != IGNORE).unsqueeze(1).float()
            onehot = F.one_hot(y.clamp(max=NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2).float()
            loss = 0.4 * weighted_ce(logits, y, weights) + 0.6 * focal_tversky(torch.softmax(logits, 1), onehot, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        metrics = evaluate(model, xva, yva, sva, device)
        entry = {"epoch": epoch, "loss": running / steps_per_epoch, "seconds": time.time() - t0, "val": metrics}
        log["epochs"].append(entry)
        print(f"epoch {epoch}: loss {entry['loss']:.4f} · val mIoU " +
              json.dumps({k: round(v['miou'], 3) for k, v in metrics.items()}) + f" · {entry['seconds']:.0f}s",
              flush=True)
        if metrics["all"]["miou"] > best:
            best = metrics["all"]["miou"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log["best_epoch"] = epoch
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"epoch": epoch, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "best": best,
                    "best_state": best_state, "log": log, "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, tmp)
        tmp.replace(ckpt_path)

    (out / "finetune_log.json").write_text(json.dumps(log, indent=1))
    if args.no_export:
        print(f"best epoch {log['best_epoch']} val mIoU {best:.3f}; checkpoint at {ckpt_path}", flush=True)
        return
    export(best_state, log, best, args.keras, out)


def export(best_state: dict, log: dict, best: float, keras_template: str, out: Path) -> None:
    """Best epoch -> ONNX (used by the app) and .keras (loads with keras.saving.load_model like the original)."""
    model = UNet()
    model.load_state_dict(best_state)
    desc = ("U-Net of Khadka & Pradhan (2026, doi:10.3126/joeis.v5i2.97825) fine-tuned for Nepal on weak labels "
            "(Impact Observatory 10 m LULC) over Esri, Sentinel-2 and Landsat imagery at 5.46 m. "
            "Input: NHWC float32 RGB in [0,1], H,W multiples of 16; output: softmax over 7 DeepGlobe classes.")
    export_onnx(model, out / "unet_nepal.onnx", desc)
    save_keras(model, keras_template, out / "deepglobe_unet_nepal.keras")
    (out / "finetune_log.json").write_text(json.dumps(log, indent=1))
    print(f"best epoch {log.get('best_epoch')} val mIoU {best:.3f}; wrote {out / 'unet_nepal.onnx'}", flush=True)


if __name__ == "__main__":
    main()
