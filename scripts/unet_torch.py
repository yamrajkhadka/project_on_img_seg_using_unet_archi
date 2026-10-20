"""PyTorch twin of the paper's Keras U-Net (same layers, same weights, same outputs).

Used for fast fine-tuning on a GPU / Apple-silicon MPS and for exporting ONNX.  ``load_keras`` copies
the weights out of ``deepglobe_unet_jay.keras`` and ``save_keras`` writes fine-tuned weights back into a
``.keras`` file that loads with ``keras.saving.load_model`` exactly like the original.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

BN_EPS = 1e-3  # Keras default


class Block(nn.Module):
    """Conv(3x3)+ReLU -> BN -> Conv(3x3)+ReLU -> BN   (the paper's conv_block: BN *after* ReLU)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.b1 = nn.BatchNorm2d(cout, eps=BN_EPS, momentum=0.01)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.b2 = nn.BatchNorm2d(cout, eps=BN_EPS, momentum=0.01)

    def forward(self, x):
        x = self.b1(torch.relu(self.c1(x)))
        return self.b2(torch.relu(self.c2(x)))


class UNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.e1, self.e2, self.e3, self.e4 = Block(3, f[0]), Block(f[0], f[1]), Block(f[1], f[2]), Block(f[2], f[3])
        self.b = Block(f[3], f[4])
        self.drop = nn.Dropout(0.5)
        self.u1, self.d1 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2), Block(f[4], f[3])
        self.u2, self.d2 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2), Block(f[3], f[2])
        self.u3, self.d3 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2), Block(f[2], f[1])
        self.u4, self.d4 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2), Block(f[1], f[0])
        self.head = nn.Conv2d(f[0], num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):  # NCHW in [0,1] -> logits NCHW
        c1 = self.e1(x)
        c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2))
        c4 = self.e4(self.pool(c3))
        b = self.drop(self.b(self.pool(c4)))
        c5 = self.d1(torch.cat([self.u1(b), c4], 1))
        c6 = self.d2(torch.cat([self.u2(c5), c3], 1))
        c7 = self.d3(torch.cat([self.u3(c6), c2], 1))
        c8 = self.d4(torch.cat([self.u4(c7), c1], 1))
        return self.head(c8)


def _modules_in_keras_order(m: UNet):
    blocks = [m.e1, m.e2, m.e3, m.e4, m.b, m.d1, m.d2, m.d3, m.d4]
    convs = [c for blk in blocks for c in (blk.c1, blk.c2)] + [m.head]
    bns = [b for blk in blocks for b in (blk.b1, blk.b2)]
    ups = [m.u1, m.u2, m.u3, m.u4]
    return convs, bns, ups


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def load_keras(path: str | Path, model: UNet | None = None) -> UNet:
    """Copy weights from a Keras 3 ``.keras`` archive of the paper's architecture."""
    model = model or UNet()
    with zipfile.ZipFile(path) as z:
        h5 = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")
    convs, bns, ups = _modules_in_keras_order(model)
    with torch.no_grad():
        for i, conv in enumerate(convs):
            g = h5[f"layers/conv2d{_suffix(i)}/vars"]
            conv.weight.copy_(torch.from_numpy(np.array(g["0"]).transpose(3, 2, 0, 1)))
            conv.bias.copy_(torch.from_numpy(np.array(g["1"])))
        for i, bn in enumerate(bns):
            g = h5[f"layers/batch_normalization{_suffix(i)}/vars"]
            bn.weight.copy_(torch.from_numpy(np.array(g["0"])))
            bn.bias.copy_(torch.from_numpy(np.array(g["1"])))
            bn.running_mean.copy_(torch.from_numpy(np.array(g["2"])))
            bn.running_var.copy_(torch.from_numpy(np.array(g["3"])))
        for i, up in enumerate(ups):
            g = h5[f"layers/conv2d_transpose{_suffix(i)}/vars"]
            up.weight.copy_(torch.from_numpy(np.array(g["0"]).transpose(3, 2, 0, 1)))
            up.bias.copy_(torch.from_numpy(np.array(g["1"])))
    return model


def save_keras(model: UNet, template: str | Path, out: str | Path) -> None:
    """Write the model's weights into a copy of the original ``.keras`` archive (optimizer state dropped)."""
    convs, bns, ups = _modules_in_keras_order(model)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as h5:
        def put(name, arrays):
            for j, a in enumerate(arrays):
                h5.create_dataset(f"layers/{name}/vars/{j}", data=a)
        for i, c in enumerate(convs):
            put(f"conv2d{_suffix(i)}", [c.weight.detach().cpu().numpy().transpose(2, 3, 1, 0),
                                       c.bias.detach().cpu().numpy()])
        for i, b in enumerate(bns):
            put(f"batch_normalization{_suffix(i)}", [b.weight.detach().cpu().numpy(), b.bias.detach().cpu().numpy(),
                                                    b.running_mean.cpu().numpy(), b.running_var.cpu().numpy()])
        for i, u in enumerate(ups):
            put(f"conv2d_transpose{_suffix(i)}", [u.weight.detach().cpu().numpy().transpose(2, 3, 1, 0),
                                                 u.bias.detach().cpu().numpy()])
        h5.create_group("layers/dropout/vars")
    with zipfile.ZipFile(template) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("metadata.json", zin.read("metadata.json"))
        zout.writestr("config.json", zin.read("config.json"))
        zout.writestr("model.weights.h5", buf.getvalue())


class NHWCSoftmax(nn.Module):
    """ONNX interface identical to the Keras export: NHWC float32 RGB in [0,1] -> NHWC softmax."""

    def __init__(self, net: UNet):
        super().__init__()
        self.net = net

    def forward(self, image):
        return torch.softmax(self.net(image.permute(0, 3, 1, 2)), dim=1).permute(0, 2, 3, 1)


def export_onnx(model: UNet, out: str | Path, description: str) -> None:
    import onnx

    model = model.eval().cpu()
    wrapper = NHWCSoftmax(model)
    dummy = torch.rand(1, 224, 224, 3)
    torch.onnx.export(wrapper, (dummy,), str(out), input_names=["image"], output_names=["probs"], opset_version=17,
                      dynamic_axes={"image": {0: "batch", 1: "height", 2: "width"},
                                    "probs": {0: "batch", 1: "height", 2: "width"}},
                      dynamo=False)
    proto = onnx.load(str(out))
    proto.producer_name = "monitor-nepal"
    meta = proto.metadata_props.add()
    meta.key, meta.value = "description", description
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))


def keras_class_weights_json(path: str | Path) -> dict:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("metadata.json"))
