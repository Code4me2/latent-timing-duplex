#!/usr/bin/env python3
"""Phase2 C1 residual injector g(s). residual-add only. No Moshi/FiLM/LoRA."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ART = Path("/home/velvet/cs199-phase1-work/eval/phase2_c1_2026-09-25")
C0 = Path("/home/velvet/cs199-phase1-work/eval/phase2_c0_2026-09-25")
VAD = Path("/home/velvet/cs199-phase1-work/eval/overnight_2026-09-12/mid180_channel_energy_vad_candor_plus_dc41.jsonl")
T_VALID, H, SEED = 2226, 12, 20260925
DT = 1.0 / 12.5
LOG, METRICS_JSONL, METRICS_JSON = ART/"train_c1.log", ART/"metrics.jsonl", ART/"metrics.json"
CKPT, PROGRESS = ART/"checkpoint.pt", ART/"PROGRESS.json"

def log(msg):
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line); sys.stdout.flush()
    with open(LOG, "a") as f: f.write(line)

class JEPAHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4096,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512,256))
    def forward(self, h): return self.net(h)

class Injector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256,512), nn.ReLU(), nn.Linear(512,4096))
    def forward(self, s): return self.net(s)

class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4096, 1)
    def forward(self, h): return self.lin(h).squeeze(-1)

def intervals_to_mask(ivs, T, dt=DT):
    m = np.zeros(T, dtype=np.float32)
    for a,b in ivs or []:
        i0 = max(0, int(math.floor(float(a)/dt))); i1 = min(T, int(math.ceil(float(b)/dt)))
        m[i0:i1] = 1.0
    return m

def load_vad():
    out = {}
    for ln in VAD.open():
        r = json.loads(ln); out[r["session_id"]] = r
    return out

def main():
    ART.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("")
    gate = json.loads((C0/"GATE.json").read_text())
    assert gate.get("decision") == "PASS", gate
    locked_c0 = json.loads((C0/"TRAIN_LOCKED.json").read_text())
    train_uuids = locked_c0["split"]["train_uuids"]; val_uuids = locked_c0["split"]["val_uuids"]
    inv = {f["uuid"]: f for f in json.loads((C0/"INVENTORY.json").read_text())["files"]}
    vad = load_vad()
    device = torch.device("cuda:0"); assert torch.cuda.is_available()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    _ = torch.zeros(1, device=device); torch.cuda.reset_peak_memory_stats(0)

    c0ck = torch.load(C0/"checkpoint.pt", map_location="cpu", weights_only=False)
    jepa = JEPAHead().to(device); jepa.load_state_dict(c0ck["mlp_state_dict"]); jepa.eval()
    for p in jepa.parameters(): p.requires_grad_(False)

    g = Injector().to(device); readout = Readout().to(device)
    n_inj = sum(p.numel() for p in g.parameters())
    log(f"C1 start injector_params={n_inj} train={len(train_uuids)} val={len(val_uuids)}")

    def build_split(uuids):
        hs, ss, ys = [], [], []
        for u in uuids:
            f = inv[u]
            h_obj = torch.load(f["hidden_path"], map_location="cpu", weights_only=False)
            h = (h_obj["hidden"] if isinstance(h_obj, dict) else h_obj).float()[:T_VALID]
            if u not in vad: continue
            user = intervals_to_mask(vad[u]["user_vad"], T_VALID)
            y = torch.from_numpy(user[H:T_VALID]).float()
            h_ctx = h[:T_VALID-H]
            with torch.no_grad():
                s = jepa(h_ctx.to(device)).cpu()
            hs.append(h_ctx); ss.append(s); ys.append(y)
        return torch.cat(hs), torch.cat(ss), torch.cat(ys)

    log("Loading train..."); Htr, Str, Ytr = build_split(train_uuids)
    log("Loading val..."); Hva, Sva, Yva = build_split(val_uuids)
    log(f"train N={Htr.shape[0]} val N={Hva.shape[0]}")

    train_locked = {
        "phase": "phase2_c1_train", "locked_before_any_optimizer_step": True, "seed": SEED,
        "H_frames": H, "architecture_injector": "Linear(256,512)+ReLU+Linear(512,4096)",
        "residual_add_only": True, "moshi_loaded": False, "film_lora": False,
        "expected_injector_params": n_inj,
        "split": {"train_uuids": train_uuids, "val_uuids": val_uuids},
        "loss": "BCE(readout(h+g(s)), y_user[t+H]) + 1e-4*||g(s)||^2",
        "optim": {"lr": 1e-3, "batch_size": 32, "max_steps": 8000, "val_every": 500, "patience": 6},
    }
    (ART/"TRAIN_LOCKED.json").write_text(json.dumps(train_locked, indent=2)+"\n")
    log("wrote TRAIN_LOCKED.json")

    opt = torch.optim.AdamW(list(g.parameters())+list(readout.parameters()), lr=1e-3, weight_decay=0.01)
    bs, max_steps, val_every, log_every = 32, 8000, 500, 50
    patience, min_delta = 6, 1e-4
    best, best_step, patience_left = float("inf"), 0, patience
    step = epoch = 0; t0 = time.time(); early = False; stop_reason = "max_steps"
    metrics = []; N = Htr.shape[0]
    g_epoch = torch.Generator(); g_epoch.manual_seed(SEED)
    open(METRICS_JSONL, "w").write("")

    def evaluate(Hd, Sd, Yd):
        g.eval(); readout.eval()
        tot_bce = tot_l2 = tot = 0.0; nb = 0
        with torch.no_grad():
            for i in range(0, Hd.shape[0], 512):
                hb = Hd[i:i+512].to(device); sb = Sd[i:i+512].to(device); yb = Yd[i:i+512].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    r = g(sb); logits = readout(hb + r)
                bce = F.binary_cross_entropy_with_logits(logits.float(), yb)
                l2 = r.float().pow(2).mean(); loss = bce + 1e-4 * l2
                tot_bce += float(bce); tot_l2 += float(l2); tot += float(loss); nb += 1
        return {"L": tot/nb, "L_bce": tot_bce/nb, "L_l2": tot_l2/nb}

    def condition_metrics():
        g.eval(); readout.eval(); out = {}
        with torch.no_grad():
            idx = torch.arange(0, Hva.shape[0], max(1, Hva.shape[0]//4000))
            hb = Hva[idx].to(device); sb0 = Sva[idx].to(device); yb = Yva[idx].to(device)
            for name, use_g, shuffle in [("inject", True, False), ("shuffle", True, True), ("none", False, False)]:
                sb = sb0
                if shuffle:
                    sb = sb[torch.randperm(sb.shape[0], generator=torch.Generator().manual_seed(SEED+1))]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = readout(hb + g(sb)) if use_g else readout(hb)
                p = torch.sigmoid(logits.float())
                out[name] = {
                    "speak_mass": float(p.mean()),
                    "overlap_mass": float((p*yb).mean()),
                    "gap_mass": float((p*(1-yb)).mean()),
                    "bce": float(F.binary_cross_entropy_with_logits(logits.float(), yb)),
                }
        return out

    while step < max_steps:
        epoch += 1
        perm = torch.randperm(N, generator=g_epoch)
        g.train(); readout.train()
        for start in range(0, N, bs):
            if step >= max_steps: break
            idx = perm[start:start+bs]
            if idx.numel() < 2: continue
            hb = Htr[idx].to(device); sb = Str[idx].to(device); yb = Ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                r = g(sb); logits = readout(hb + r)
            bce = F.binary_cross_entropy_with_logits(logits.float(), yb)
            l2 = r.float().pow(2).mean(); loss = bce + 1e-4 * l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(g.parameters())+list(readout.parameters()), 1.0)
            opt.step(); step += 1
            if step % log_every == 0 or step == 1:
                row = {"step": step, "epoch": epoch, "split": "train", "L": float(loss), "L_bce": float(bce), "L_l2": float(l2)}
                metrics.append(row)
                with open(METRICS_JSONL, "a") as jf: jf.write(json.dumps(row)+"\n")
                log(f"step={step} train L={row['L']:.6f} bce={row['L_bce']:.6f} l2={row['L_l2']:.6f}")
            if step % val_every == 0:
                v = evaluate(Hva, Sva, Yva); cond = condition_metrics()
                improved = v["L_bce"] < (best - min_delta)
                if v["L_bce"] < best:
                    best = v["L_bce"]; best_step = step
                    torch.save({"injector_state_dict": g.state_dict(), "readout_state_dict": readout.state_dict(),
                                "step": step, "best_val_bce": best, "residual_add_only": True, "moshi_loaded": False,
                                "hyperparams": train_locked, "condition_metrics": cond}, CKPT)
                patience_left = patience if improved else patience_left - 1
                row = {"step": step, "epoch": epoch, "split": "val", **v, "best_val_bce": best, "patience_left": patience_left, "conditions": cond}
                metrics.append(row)
                with open(METRICS_JSONL, "a") as jf: jf.write(json.dumps(row)+"\n")
                log(f"step={step} VAL bce={v['L_bce']:.6f} best={best:.6f} patience={patience_left} cond={cond}")
                PROGRESS.write_text(json.dumps({"step": step, "best_val_bce": best, "patience_left": patience_left, "conditions": cond}, indent=2)+"\n")
                if patience_left <= 0:
                    early = True; stop_reason = f"plateau best_bce={best:.6f}@step={best_step}"; log(stop_reason); break
        if early: break

    elapsed = time.time() - t0; cond = condition_metrics()
    summary = {"phase": "phase2_c1", "steps": step, "epochs": epoch, "stop_reason": stop_reason, "early_stopped": early,
               "best_val_bce": best, "best_step": best_step, "elapsed_sec": elapsed, "injector_params": n_inj,
               "condition_metrics_final": cond, "residual_add_only": True, "moshi_loaded": False,
               "gpu_peak_alloc_GiB": torch.cuda.max_memory_allocated(0)/(1024**3)}
    METRICS_JSON.write_text(json.dumps(summary, indent=2)+"\n")
    PROGRESS.write_text(json.dumps({"done": True, **summary}, indent=2)+"\n")
    log(f"DONE steps={step} best_bce={best:.6f} cond={cond} elapsed={elapsed:.1f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
