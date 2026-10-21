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


def focal_tversky(probs, onehot, valid, alpha=0.7, beta=0.3, gamma=0.75, batch_level=False):
    """Paper's focal Tversky loss with an ignore mask.

    Per image (the first port): every class absent from an image adds a constant of about 1 with almost no
    gradient, so rare classes such as water barely train it. Batch level: true/false positives and false negatives
    are pooled over the batch and only the classes present in it are averaged.
    """
    p = probs.clamp(1e-7, 1.0) * valid
    t = onehot * valid
    dims = (0, 2, 3) if batch_level else (2, 3)
    tp = (t * p).sum(dims)
    fp = ((1 - t) * p).sum(dims)
    fn = (t * (1 - p)).sum(dims)
    loss = torch.pow(1 - (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6), gamma)
    if batch_level:
        present = t.sum((0, 2, 3)) > 0
        return loss[present].mean() if present.any() else probs.sum() * 0
    return loss.mean()


def zoom(x: torch.Tensor, y: torch.Tensor, max_zoom: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Random scale change of the whole batch (log-uniform in 1/max_zoom..max_zoom): objects look larger or smaller.

    Zooming in crops and enlarges; zooming out shrinks the patch and pads it with mirrored image and ignored labels.
    """
    s = math.exp(float(torch.empty(1).uniform_(-math.log(max_zoom), math.log(max_zoom))))
    h, w = x.shape[2:]
    if s > 1:
        ch, cw = int(h / s), int(w / s)
        i, j = int(torch.randint(0, h - ch + 1, (1,))), int(torch.randint(0, w - cw + 1, (1,)))
        x = F.interpolate(x[:, :, i:i + ch, j:j + cw], size=(h, w), mode="bilinear", align_corners=False)
        y = F.interpolate(y[:, None, i:i + ch, j:j + cw].float(), size=(h, w), mode="nearest")[:, 0].long()
    elif s < 1:
        nh, nw = int(h * s), int(w * s)
        top, left = int(torch.randint(0, h - nh + 1, (1,))), int(torch.randint(0, w - nw + 1, (1,)))
        pad = (left, w - nw - left, top, h - nh - top)
        x = F.pad(F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False), pad, mode="reflect")
        y = F.pad(F.interpolate(y[:, None].float(), size=(nh, nw), mode="nearest"), pad, value=IGNORE)[:, 0].long()
    return x, y


def lovasz_softmax(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovasz-softmax (Berman et al., CVPR 2018): a differentiable surrogate of the mean IoU itself."""
    c = probs.shape[1]
    p = probs.permute(0, 2, 3, 1).reshape(-1, c)
    y = labels.reshape(-1)
    keep = y != IGNORE
    p, y = p[keep], y[keep]
    losses = []
    for k in range(c):
        fg = (y == k).float()
        if not fg.any():
            continue
        err, order = torch.sort((fg - p[:, k]).abs(), descending=True)
        gt = fg[order]
        jaccard = 1 - (gt.sum() - gt.cumsum(0)) / (gt.sum() + (1 - gt).cumsum(0))
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        losses.append(torch.dot(err, jaccard))
    return torch.stack(losses).mean() if losses else probs.sum() * 0


def ignore_boundaries(y: torch.Tensor, px: int) -> torch.Tensor:
    """The labels are a 10 m map resampled to 5.46 m, so class edges are uncertain by about a pixel: ignore them."""
    yf = y.unsqueeze(1).float()
    hi = F.max_pool2d(yf, 2 * px + 1, 1, px)
    lo = -F.max_pool2d(-yf, 2 * px + 1, 1, px)
    y = y.clone()
    y[(hi != lo).squeeze(1)] = IGNORE
    return y


def weighted_ce(logits, target, weights, smoothing=0.0):
    return F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE, label_smoothing=smoothing)


def augment(x: torch.Tensor, y: torch.Tensor, strength: float = 1.0, per_sample: bool = False,
            blur: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometric (flips, 90-degree turns) + photometric jitter on a batch (N,3,H,W)/(N,H,W).

    ``per_sample`` draws the flips and turns for every image (not one for the whole batch) and adds a small
    per-channel colour shift; ``blur`` is the share of images degraded to 2-5 times coarser resolution, so the network also
    learns the softer look of Sentinel-2 and Landsat after resampling to 5.46 m.
    """
    n = x.shape[0]
    dev = x.device
    if per_sample:
        x, y = x.clone(), y.clone()
        m = torch.rand(n, device=dev) < 0.5
        x[m], y[m] = x[m].flip(3), y[m].flip(2)
        m = torch.rand(n, device=dev) < 0.5
        x[m], y[m] = x[m].flip(2), y[m].flip(1)
        k = torch.randint(0, 4, (n,), device=dev)
        for turn in (1, 2, 3):
            m = k == turn
            if m.any():
                x[m], y[m] = torch.rot90(x[m], turn, (2, 3)), torch.rot90(y[m], turn, (1, 2))
    else:
        if torch.rand(1) < 0.5:
            x, y = x.flip(3), y.flip(2)
        if torch.rand(1) < 0.5:
            x, y = x.flip(2), y.flip(1)
        k = int(torch.randint(0, 4, (1,)))
        if k:
            x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if blur:
        m = torch.rand(n, device=dev) < blur
        if m.any():
            h, w = x.shape[2:]
            f = int(torch.randint(2, 6, (1,)))  # 5.46 m -> about 11-27 m, i.e. Sentinel-2 to Landsat
            small = F.avg_pool2d(x[m], f, f, ceil_mode=True)
            x = x.clone()
            x[m] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    if per_sample:
        x = x * (1 + (torch.rand(n, 3, 1, 1, device=dev) - 0.5) * 0.1 * strength)
    bright = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3 * strength
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3 * strength
    mean = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * bright
    gray = x.mean(1, keepdim=True)
    sat = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3 * strength
    x = (x - gray) * sat + gray
    gamma = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4 * strength)
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
    ap.add_argument("--ce-weight", type=float, default=0.4, help="loss = w x CE + (1 - w) x focal Tversky (paper: 0.4)")
    ap.add_argument("--weight-power", type=float, default=0.5,
                    help="class weights = (max count / count) ** power (0.5 = square root)")
    ap.add_argument("--aug", type=float, default=1.0, help="photometric augmentation strength")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="evaluate an exponential moving average of the weights")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many epochs without improvement, never in the first half of the schedule "
                         "(a fresh learning rate first lowers the score before it climbs)")
    ap.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    ap.add_argument("--tversky", choices=("image", "batch"), default="image",
                    help="focal Tversky per image (first port) or pooled over the batch (rare classes train better)")
    ap.add_argument("--oversample", type=float, default=0.0,
                    help="0-1: share of patches drawn in proportion to their rare-class content instead of uniformly")
    ap.add_argument("--ignore-boundary", type=int, default=0, help="ignore this many pixels along label edges")
    ap.add_argument("--per-sample-aug", action="store_true", help="per-image flips/turns and per-channel colour")
    ap.add_argument("--blur", type=float, default=0.0, help="share of patches degraded to coarser resolution")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0 = Adam, as before)")
    ap.add_argument("--lovasz", type=float, default=0.0, help="add this weight x Lovasz-softmax (optimises IoU)")
    ap.add_argument("--tversky-alpha", type=float, default=0.7, help="false-positive weight (paper: 0.7)")
    ap.add_argument("--tversky-beta", type=float, default=0.3, help="false-negative weight (paper: 0.3)")
    ap.add_argument("--focal-gamma", type=float, default=0.75, help="focal exponent (paper: 0.75)")
    ap.add_argument("--zoom", type=float, default=1.0, help="random scale change up to this factor (1 = off)")
    ap.add_argument("--init", default=None, help="start a new run from these weights (a state_dict .pt, or a "
                    "checkpoint whose best epoch is used) instead of the paper's Keras weights")
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
    w = (counts.max() / (counts + 1e-6)) ** args.weight_power
    w = np.minimum(w, 20.0)
    w = w / w.sum() * NUM_CLASSES
    weights = torch.tensor(w, dtype=torch.float32, device=device)
    print("class pixel share", np.round(counts / counts.sum() * 100, 2), "weights", np.round(w, 3), flush=True)

    # resuming needs only the architecture; a fresh run starts from the paper's Keras weights
    if ckpt_path.exists():
        model = UNet()
    elif args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        model = UNet()
        model.load_state_dict(init["best_state"] if "best_state" in init else init)
    else:
        model = load_keras(args.keras)
    model = model.to(device)
    ema = None
    if args.ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if args.weight_decay:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    draw = None
    if args.oversample:  # rarer classes -> more often drawn: weight each patch by its class mix x class weights
        mix = np.zeros((len(ytr), NUM_CLASSES))
        for i in range(0, len(ytr), 256):
            part = ytr[i:i + 256]
            mix[i:i + 256] = np.stack([(part == c).mean((1, 2)) for c in range(NUM_CLASSES)], 1)
        score = mix @ w
        draw = (1 - args.oversample) / len(ytr) + args.oversample * score / score.sum()
        draw /= draw.sum()
        print(f"oversampling: most-drawn patch {draw.max() * len(ytr):.1f}x, least {draw.min() * len(ytr):.2f}x",
              flush=True)
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
        log = {"config": vars(args), "baseline": base, "epochs": [], "class_weights": w.tolist(), "train_patches": int(len(xtr)),
               "val_patches": int(len(xva)), "device": str(device), "epochs_planned": args.epochs}
        if args.init:  # a continuation never ends below where it started
            best, best_state, log["best_epoch"] = base["all"]["miou"], {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
    model.train()
    stale = 0
    for epoch in range(start_epoch, args.epochs + 1):
        perm = np.random.choice(len(xtr), len(xtr), p=draw) if draw is not None else np.random.permutation(len(xtr))
        t0, running = time.time(), 0.0
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * args.batch:(s + 1) * args.batch])
            x = torch.from_numpy(xtr[idx]).to(device).permute(0, 3, 1, 2).float() / 255
            y = torch.from_numpy(ytr[idx].astype(np.int64)).to(device)
            if args.zoom > 1 and torch.rand(1) < 0.5:
                x, y = zoom(x, y, args.zoom)
            x, y = augment(x, y, args.aug, args.per_sample_aug, args.blur)
            if args.ignore_boundary:
                y = ignore_boundaries(y, args.ignore_boundary)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            logits = logits.float()
            valid = (y != IGNORE).unsqueeze(1).float()
            if not valid.any():
                continue
            onehot = F.one_hot(y.clamp(max=NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2).float()
            probs = torch.softmax(logits, 1)
            loss = (args.ce_weight * weighted_ce(logits, y, weights, args.label_smoothing)
                    + (1 - args.ce_weight) * focal_tversky(probs, onehot, valid, args.tversky_alpha, args.tversky_beta,
                                                           args.focal_gamma, batch_level=args.tversky == "batch"))
            if args.lovasz:
                loss = loss + args.lovasz * lovasz_softmax(probs, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            running += loss.item()
        scored = ema.module if ema is not None else model
        metrics = evaluate(scored, xva, yva, sva, device)
        entry = {"epoch": epoch, "loss": running / steps_per_epoch, "seconds": time.time() - t0, "val": metrics}
        log["epochs"].append(entry)
        print(f"epoch {epoch}: loss {entry['loss']:.4f} · val mIoU " +
              json.dumps({k: round(v['miou'], 3) for k, v in metrics.items()}) + f" · {entry['seconds']:.0f}s",
              flush=True)
        if metrics["all"]["miou"] > best:
            best, stale = metrics["all"]["miou"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in scored.state_dict().items()}
            log["best_epoch"] = epoch
        else:
            stale += 1
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"epoch": epoch, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "best": best,
                    "best_state": best_state, "log": log, "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, tmp)
        tmp.replace(ckpt_path)
        if args.patience and stale >= args.patience and epoch >= args.epochs // 2:
            print(f"no improvement for {stale} epochs; stopping", flush=True)
            break

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
