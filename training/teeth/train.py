"""Train a clean-vs-food classifier (MobileNetV3-Small) on dataset/.

Reads dataset/labels.json written by synth.py, fine-tunes an ImageNet
MobileNetV3-Small with one output logit (sigmoid > 0.5 means food), keeps the
checkpoint with the lowest val loss, stops early when val loss stops
improving, then writes:

    runs/<name>/best.pt                 state_dict + config
    runs/<name>/metrics.json            val accuracy, food precision/recall,
                                        confusion matrix, recall per food type
    runs/<name>/teeth_mobilenetv3.onnx  input "image" (N,3,S,S) float32,
                                        output "logits" (N,1)
    runs/<name>/teeth_mobilenetv3.json  input size, mean/std, threshold

Runs on CUDA (e.g. Jetson), Apple MPS or CPU, picked automatically.
The first run downloads the ImageNet weights (~10 MB) from download.pytorch.org;
pass --no-pretrained to train from scratch offline, or --weights FILE.

Usage:
    python train.py [--size 224] [--epochs 15] [--batch 64] [--name mnv3s_224]
    python train.py --data dataset_v2 --size 320 --name v2_320
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "dataset"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class TeethSet(Dataset):
    def __init__(self, records, tf, data_dir):
        self.records, self.tf, self.data_dir = records, tf, data_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(self.data_dir / r["file"]).convert("RGB")
        return self.tf(img), torch.tensor([float(r["label"])]), i


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(pretrained, weights_file=None):
    weights = None
    if pretrained and not weights_file:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    if weights_file:
        state = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model


def transforms_for(size):
    # The images already carry brightness/colour/blur/JPEG augmentation from
    # synth.py; keep these light. Scale >= 0.85 so edge food bits stay in view.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def run_epoch(model, loader, device, loss_fn, opt=None):
    model.train(opt is not None)
    total, n, probs, idx = 0.0, 0, [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y, i in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(x)
            n += len(x)
            probs.append(torch.sigmoid(logits).detach().float().cpu().numpy()[:, 0])
            idx.append(i.numpy())
    return total / n, np.concatenate(probs), np.concatenate(idx)


def metrics(records, probs, idx, thr):
    y = np.array([records[i]["label"] for i in idx])
    p = (probs >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    per_type, per_count = {}, {}
    for k, i in enumerate(idx):
        r = records[i]
        if r["label"] != 1:
            continue
        hit = int(p[k] == 1)
        for t in {b["type"] for b in r["bits"]}:
            h, c = per_type.get(t, (0, 0))
            per_type[t] = (h + hit, c + 1)
        h, c = per_count.get(len(r["bits"]), (0, 0))
        per_count[len(r["bits"])] = (h + hit, c + 1)
    return {
        "threshold": thr,
        "n": int(len(y)),
        "accuracy": round((tp + tn) / len(y), 4),
        "food_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "food_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "confusion_matrix": {"rows=true, cols=pred, order": ["clean", "food"],
                             "matrix": [[tn, fp], [fn, tp]]},
        # An image counts for every food type it contains.
        "food_recall_by_type": {t: {"recall": round(h / c, 4), "n": c}
                                for t, (h, c) in sorted(per_type.items())},
        "food_recall_by_bit_count": {str(k): {"recall": round(h / c, 4), "n": c}
                                     for k, (h, c) in sorted(per_count.items())},
    }


def export_onnx(model, size, path):
    """Single-file ONNX, opset 17, dynamic batch.

    Uses the TorchScript exporter (dynamo=False): the newer dynamo exporter
    builds opset 18 and fails to convert MobileNetV3 down to 17.
    """
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, 3, size, size)
    for stale in (path, Path(str(path) + ".data")):
        stale.unlink(missing_ok=True)
    torch.onnx.export(model, dummy, str(path), opset_version=17, dynamo=False,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    # Check the exported graph matches PyTorch.
    import onnxruntime as ort
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        ["logits"], {"image": x.numpy()})[0]
    print(f"onnx export ok, max |diff| vs torch {np.abs(out - ref).max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224, help="input size, e.g. 224 or 320")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5, help="early stop on val loss")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--weights", help="local MobileNetV3-Small weights instead of downloading")
    ap.add_argument("--init", help="start from a finished run's best.pt (e.g. runs/v2_320/best.pt)")
    ap.add_argument("--name", default=None, help="run folder name under runs/")
    ap.add_argument("--data", type=Path, default=DATA_DIR,
                    help="dataset folder from synth.py (dataset = v1, dataset_v2 = v2)")
    ap.add_argument("--export-only", action="store_true",
                    help="skip training; export ONNX from the run's best.pt")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; re-run val metrics + ONNX export from best.pt")
    args = ap.parse_args()

    if args.export_only:
        run_dir = OUT_DIR / "runs" / (args.name or f"mnv3s_{args.size}")
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
        model = build_model(False)
        model.load_state_dict(ckpt["model"])
        export_onnx(model, ckpt["config"]["size"], run_dir / "teeth_mobilenetv3.onnx")
        (run_dir / "teeth_mobilenetv3.json").write_text(json.dumps(ckpt["config"], indent=2))
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    run_dir = OUT_DIR / "runs" / (args.name or f"mnv3s_{args.size}")
    run_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data if args.data.is_absolute() else OUT_DIR / args.data
    records = json.loads((data_dir / "labels.json").read_text())
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    train_tf, val_tf = transforms_for(args.size)
    pin = device.type == "cuda"
    train_dl = DataLoader(TeethSet(train_recs, train_tf, data_dir), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, pin_memory=pin,
                          drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(TeethSet(val_recs, val_tf, data_dir), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=pin,
                        persistent_workers=args.workers > 0)

    model = build_model(not args.no_pretrained and not args.init, args.weights)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"device {device}, train {len(train_recs)}, val {len(val_recs)}, size {args.size}")

    config = {"arch": "mobilenet_v3_small", "size": args.size, "mean": MEAN, "std": STD,
              "threshold": 0.5, "output": "logits[:,0]; sigmoid >= threshold means food"}
    meta_file = data_dir / "meta.json"  # crop rule the inputs must follow
    config["crop"] = (json.loads(meta_file.read_text()) if meta_file.exists()
                      else {"crop_rule": "outer_lip_bbox", "crop_margin": 0.25})
    best, bad, history = float("inf"), 0, []
    for epoch in range(1, 0 if args.eval_only else args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_dl, device, loss_fn, opt)
        va_loss, probs, idx = run_epoch(model, val_dl, device, loss_fn)
        sched.step()
        acc = metrics(val_recs, probs, idx, 0.5)["accuracy"]
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "val_loss": round(va_loss, 4), "val_acc": acc})
        print(f"epoch {epoch:2d} train {tr_loss:.4f} val {va_loss:.4f} acc {acc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if va_loss < best - 1e-4:
            best, bad = va_loss, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                       run_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: val loss did not improve for {args.patience} epochs")
                break

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    va_loss, probs, idx = run_epoch(model, val_dl, device, loss_fn)
    config = ckpt["config"]  # same as above when training; the saved one for --eval-only
    result = {"best_epoch": ckpt["epoch"], "val_loss": round(va_loss, 4),
              **metrics(val_recs, probs, idx, 0.5), "history": history or None,
              "device": str(device), "args": {k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()}}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    export_onnx(model, config["size"], run_dir / "teeth_mobilenetv3.onnx")
    (run_dir / "teeth_mobilenetv3.json").write_text(json.dumps(config, indent=2))
    print(f"best epoch {ckpt['epoch']}: acc {result['accuracy']} "
          f"food P {result['food_precision']} R {result['food_recall']}")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
