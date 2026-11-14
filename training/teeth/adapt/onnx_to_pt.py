"""Rebuild a torchvision MobileNetV3-Small state_dict from the exported (BN-fused) teeth ONNX."""
import sys, re, numpy as np, onnx, torch, torch.nn as nn
from onnx import numpy_helper
from torchvision import models
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
model = models.mobilenet_v3_small(); model.classifier[3] = nn.Linear(1024, 1)
sd = model.state_dict(); new = {}
for n in m.graph.node:
    if n.op_type not in ("Conv", "Gemm"): continue
    if n.output[0] == "logits":
        new["classifier.3.weight"] = torch.tensor(init[n.input[1]]); new["classifier.3.bias"] = torch.tensor(init[n.input[2]]); continue
    segs = [p for p in n.output[0].split("/") if p and not p.endswith("_output_0")]
    full, prev = segs[0], segs[0]
    for t in segs[1:]:
        full += "." + (t[len(prev) + 1:] if t.startswith(prev + ".") else t); prev = t
    W = init[n.input[1]]; B = init[n.input[2]] if len(n.input) > 2 else None
    if n.op_type == "Gemm":
        attrs = {a.name: a for a in n.attribute}
        if "transB" in attrs and attrs["transB"].i == 0: W = W.T
    key = full
    if key + ".bias" in sd:
        new[key + ".weight"] = torch.tensor(W); new[key + ".bias"] = torch.tensor(B)
    else:  # conv followed by BN at sibling index +1
        parts = key.split("."); bn = ".".join(parts[:-1] + [str(int(parts[-1]) + 1)])
        new[key + ".weight"] = torch.tensor(W)
        c = W.shape[0]; eps = model.get_submodule(bn).eps
        new[bn + ".weight"] = torch.ones(c); new[bn + ".bias"] = torch.tensor(B)
        new[bn + ".running_mean"] = torch.zeros(c); new[bn + ".running_var"] = torch.full((c,), 1 - eps)
missing = [k for k in sd if k not in new and not k.endswith("num_batches_tracked")]
print("missing", missing[:5], len(missing))
for k in new: assert new[k].shape == sd[k].shape, (k, new[k].shape, sd[k].shape)
model.load_state_dict(new, strict=False); model.eval()
import onnxruntime as ort
x = torch.randn(2, 3, 384, 384)
with torch.no_grad(): ref = model(x).numpy()
out = ort.InferenceSession(src).run(None, {"image": x.numpy()})[0]
print("max diff", np.abs(out - ref).max(), ref.ravel(), out.ravel())
import json
torch.save({"model": model.state_dict(), "config": json.load(open(src.replace(".onnx", ".json"))), "epoch": 0}, dst)
