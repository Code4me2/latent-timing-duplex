#!/usr/bin/env python3
"""C0 surprise export: JEPA 4096->512->512->256 + frozen P; MSE(pred, y@P)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ART = Path("/home/velvet/cs199-phase1-work/eval/phase2_c0_2026-09-25")
HIDDEN = Path("/home/velvet/cs199-phase1-work/ablations/precompute/hidden")
TARGET = Path("/home/velvet/cs199-phase1-work/ablations/precompute/target_h12")
CKPT = ART / "checkpoint.pt"
OUT = ART / "mid180_surprise_h12_lam0.01_c0.jsonl"
T_VALID = 2226
H_FRAMES = 12
LAMBDA = 0.01

class JEPAHead(nn.Module):
    def __init__(self, d_in=4096, width=512, embed=256, n_layers=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, embed),
        )
    def forward(self, h):
        return self.net(h)

def load_mat(path, key):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    t = obj[key] if isinstance(obj, dict) else obj
    return t.float() if hasattr(t, "float") else torch.as_tensor(t, dtype=torch.float32)

def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = JEPAHead()
    model.load_state_dict(ck["mlp_state_dict"])
    model.eval()
    P = ck["P"].float()
    assert P.shape == (4096, 256)
    files = sorted(HIDDEN.glob("*.pt"))
    n = 0
    with OUT.open("w") as fh:
        for hp in files:
            stem = hp.stem
            if stem.startswith("candor_"):
                sid = stem[len("candor_"):]
            elif stem.startswith("dc_"):
                sid = stem[len("dc_"):]
            else:
                sid = stem
            yp = TARGET / f"{stem}.pt"
            if not yp.is_file():
                print(f"SKIP missing target {stem}", flush=True)
                continue
            h = load_mat(hp, "hidden")[:T_VALID]
            y = load_mat(yp, "target")
            assert y.shape[0] == T_VALID and h.shape[0] == T_VALID
            with torch.no_grad():
                pred = model(h)  # [T,256]
                y_emb = y @ P    # [T,256] stopgrad already
                sur = ((pred - y_emb) ** 2).mean(dim=-1).cpu().numpy().astype(np.float64)
            rec = {
                "session_id": sid,
                "surprise": [float(x) for x in sur],
                "duration_s": 180.0,
                "window": "mid180",
                "horizon_frames": H_FRAMES,
                "lambda_reg": LAMBDA,
                "T": int(sur.shape[0]),
                "checkpoint": str(CKPT),
                "phase": "phase2_c0",
                "embed_dim": 256,
                "residual_inject": False,
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
            print(f"OK {sid} T={sur.shape[0]} mean={float(sur.mean()):.6f}", flush=True)
    print(f"DONE n={n} out={OUT}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
