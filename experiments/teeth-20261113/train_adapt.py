"""Continue training the teeth model (v4) two ways, on the same data:
  --mode full : every weight trainable (classic fine-tune)
  --mode lora : base frozen; low-rank adapters (rank r) on every 1x1 conv and the
                hidden Linear, plus the 1-unit head trainable. Adapters are merged
                into the weights at the end, so the ONNX has the same size/speed.
BatchNorm stays in eval mode in both (the v4 weights come from a BN-fused ONNX).
Extra data can be mixed in with --extra (folder of crops + inner-lip json, label from subfolder)."""
import argparse, json, time, sys, random, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
sys.path.insert(0, "/home/user/work/")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class LoRAConv(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base; self.scale = alpha / r
        self.A = nn.Conv2d(base.in_channels, r, 1, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5)); nn.init.zeros_(self.B.weight)
    def forward(self, x): return self.base(x) + self.B(self.A(x)) * self.scale
    def merged(self):
        w = (self.B.weight[:, :, 0, 0] @ self.A.weight[:, :, 0, 0]) * self.scale
        self.base.weight.data += w[:, :, None, None]; return self.base

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base; self.scale = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False); self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5)); nn.init.zeros_(self.B.weight)
    def forward(self, x): return self.base(x) + self.B(self.A(x)) * self.scale
    def merged(self):
        self.base.weight.data += (self.B.weight @ self.A.weight) * self.scale; return self.base

def add_lora(model, r, alpha):
    for p in model.parameters(): p.requires_grad = False
    def walk(mod):
        for name, ch in mod.named_children():
            if isinstance(ch, nn.Conv2d) and ch.groups == 1 and ch.kernel_size == (1, 1):
                setattr(mod, name, LoRAConv(ch, r, alpha))
            elif isinstance(ch, nn.Linear) and ch.out_features > 1:
                setattr(mod, name, LoRALinear(ch, r, alpha))
            else: walk(ch)
    walk(model)
    for p in model.classifier[3].parameters(): p.requires_grad = True

def merge_lora(model):
    def walk(mod):
        for name, ch in mod.named_children():
            if isinstance(ch, (LoRAConv, LoRALinear)): setattr(mod, name, ch.merged())
            else: walk(ch)
    walk(model)

class Synth(Dataset):
    def __init__(self, d, split, tf):
        self.d = Path(d); self.tf = tf
        self.recs = [r for r in json.load(open(self.d / "labels.json")) if r["split"] == split]
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return self.tf(Image.open(self.d / r["file"]).convert("RGB")), torch.tensor([float(r["label"])])

class Real(Dataset):
    """Real photos: <dir>/{food,clean}/*.png|jpg + .json with inner_lip; random crop margin 1.2-1.55."""
    def __init__(self, items, tf, repeat=1):
        import cv2, crop_rule
        self.cv2, self.cr, self.tf = cv2, crop_rule, tf
        self.items = items * repeat
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, lab = self.items[i]
        img = self.cv2.imread(str(p)); inner = json.load(open(Path(p).with_suffix(".json")))["inner_lip"]
        c, _ = self.cr.crop(img, inner, margin=(random.uniform(1.2, 1.55) - 1) / 2)
        return self.tf(Image.fromarray(self.cv2.cvtColor(c, self.cv2.COLOR_BGR2RGB))), torch.tensor([float(lab)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--init", default="/home/user/work/runs/v4_384/best.pt")
    ap.add_argument("--data", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="real photo file:label pairs list json")
    ap.add_argument("--extra-repeat", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0); torch.set_num_threads(4)
    size = 384
    tr_tf = transforms.Compose([transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    sets = []
    if a.data: sets.append(Synth(a.data, "train", tr_tf))
    if a.extra:
        items = []
        for f in a.extra: items += [tuple(x) for x in json.load(open(f))]
        sets.append(Real(items, tr_tf, a.extra_repeat))
    dl = DataLoader(ConcatDataset(sets), batch_size=a.batch, shuffle=True, num_workers=3, drop_last=len(sets[0]) > a.batch)
    model = models.mobilenet_v3_small(); model.classifier[3] = nn.Linear(1024, 1)
    ck = torch.load(a.init, map_location="cpu"); model.load_state_dict(ck["model"])
    if a.mode == "lora": add_lora(model, a.rank, a.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params); n_all = sum(p.numel() for p in model.parameters())
    print(f"{a.mode}: trainable {n_tr:,} / {n_all:,} ({100*n_tr/n_all:.2f}%), train images {sum(len(s) for s in sets)}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4 if a.mode == "full" else 0.0)
    steps = a.epochs * len(dl) if not a.max_steps else a.max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss(); t0 = time.time(); step = 0; hist = []
    def bn_eval(m):
        for x in m.modules():
            if isinstance(x, nn.BatchNorm2d): x.eval()
    for ep in range(1, a.epochs + 1):
        model.train(); bn_eval(model); tot = n = 0
        for x, y in dl:
            loss = loss_fn(model(x), y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(x); n += len(x); step += 1
            if step % 50 == 0: print(f"  step {step}/{steps} loss {tot/n:.4f} {time.time()-t0:.0f}s", flush=True)
            if step >= steps: break
        hist.append({"epoch": ep, "train_loss": round(tot / n, 4), "sec": round(time.time() - t0)})
        print(hist[-1], flush=True)
        if step >= steps: break
    if a.mode == "lora": merge_lora(model)
    out = Path("/home/user/work/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    cfg = ck["config"]
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": len(hist)}, out / "best.pt")
    json.dump({"mode": a.mode, "trainable": n_tr, "total": n_all, "train_sec": round(time.time() - t0),
               "history": hist, "args": vars(a)}, open(out / "train.json", "w"), indent=1)
    model.eval(); dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(model, dummy, str(out / "teeth_mobilenetv3.onnx"), opset_version=17, dynamo=False,
                      input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    json.dump(cfg, open(out / "teeth_mobilenetv3.json", "w"), indent=2)
    print("done", out, f"{time.time()-t0:.0f}s")

if __name__ == "__main__": main()
