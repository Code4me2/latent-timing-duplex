#!/usr/bin/env python3
"""Phase2 C0: JEPA head only. 4096->512->512->256. No Moshi. No residual inject."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ART = Path("/home/velvet/cs199-phase1-work/eval/phase2_c0_2026-09-25")
LOCKED_PATH = ART / "TRAIN_LOCKED.json"
LOG_PATH = ART / "train_c0.log"
METRICS_JSONL = ART / "metrics.jsonl"
METRICS_JSON = ART / "metrics.json"
CKPT_PATH = ART / "checkpoint.pt"
PROGRESS_PATH = ART / "PROGRESS.json"

def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line); sys.stdout.flush()
    with open(LOG_PATH, "a") as f:
        f.write(line)

class JEPAHead(nn.Module):
    def __init__(self, d_in=4096, width=512, embed=256, n_layers=2):
        super().__init__()
        assert n_layers == 2
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, embed),
        )
    def forward(self, h):
        return self.net(h)

def build_P(d_in, embed, seed, device):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    A = torch.randn(d_in, embed, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    P = Q.contiguous().to(device=device, dtype=torch.float32)
    P.requires_grad_(False)
    return P

def iso_reg(pred):
    B, E = pred.shape
    mean = pred.mean(dim=0)
    r_mean = (mean * mean).sum()
    r_energy = 0.5 * pred.pow(2).sum(dim=-1).mean()
    if B < 2:
        z = pred.new_zeros(())
        return r_mean, z, z, r_mean, r_energy
    centered = pred - mean
    var = (centered * centered).mean(dim=0)
    r_var = ((var - 1.0) ** 2).mean()
    cov = (centered.T @ centered) / B
    off = cov - torch.diag(torch.diag(cov))
    r_cov = (off * off).sum() / E
    r_iso = r_mean + r_var + r_cov
    return r_mean, r_var, r_cov, r_iso, r_energy

def losses(pred, y_emb, lambda_reg):
    L_mse = F.mse_loss(pred, y_emb)
    r_mean, r_var, r_cov, r_iso, r_energy = iso_reg(pred)
    L = L_mse + float(lambda_reg) * r_iso
    return L, L_mse, r_mean, r_var, r_cov, r_iso, r_energy

def load_split_frames(files_by_uuid, uuids, T_valid):
    hs, ys = [], []
    for u in uuids:
        f = files_by_uuid[u]
        h_obj = torch.load(f["hidden_path"], map_location="cpu", weights_only=False)
        y_obj = torch.load(f["target_h12_path"], map_location="cpu", weights_only=False)
        h = h_obj["hidden"] if isinstance(h_obj, dict) else h_obj
        y = y_obj["target"] if isinstance(y_obj, dict) else y_obj
        h = h[:T_valid]
        assert h.shape[0] == y.shape[0] == T_valid, (u, tuple(h.shape), tuple(y.shape))
        assert h.shape[1] == y.shape[1] == 4096
        hs.append(h.to(torch.float32)); ys.append(y.to(torch.float32))
    return torch.cat(hs, 0), torch.cat(ys, 0)

@torch.no_grad()
def evaluate(model, H, Y, P, lambda_reg, device, batch=512):
    model.eval()
    n = H.shape[0]
    sums = {k: 0.0 for k in ["L","L_mse","r_mean","r_var","r_cov","r_iso","r_energy"]}
    nb = 0
    for i in range(0, n, batch):
        hb = H[i:i+batch].to(device=device, dtype=torch.bfloat16)
        yb = Y[i:i+batch].to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(hb)
        y_emb = (yb @ P).detach()
        L, L_mse, r_mean, r_var, r_cov, r_iso, r_energy = losses(pred.float(), y_emb, lambda_reg)
        for k,v in [("L",L),("L_mse",L_mse),("r_mean",r_mean),("r_var",r_var),("r_cov",r_cov),("r_iso",r_iso),("r_energy",r_energy)]:
            sums[k] += float(v.item())
        nb += 1
    return {k: sums[k]/max(nb,1) for k in sums}

def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    open(LOG_PATH, "w").write("")
    locked = json.loads(LOCKED_PATH.read_text())
    assert locked["locked_before_any_optimizer_step"] is True
    assert locked["residual_inject"] is False
    assert locked["moshi_loaded"] is False

    seed = int(locked["seed"])
    H_frames = int(locked["H_frames"])
    lambda_reg = float(locked["lambda_reg"])
    T_valid = int(locked["T_valid"])
    d_in = int(locked["D_in"])
    width = int(locked["width"])
    embed = int(locked["embed_dim"])
    n_layers = int(locked["n_layers"])
    oc = locked["optim"]
    lr = float(oc["lr"]); wd = float(oc["weight_decay"]); bs = int(oc["batch_size"])
    max_steps = int(oc["max_steps"]); val_every = int(oc["val_every"]); log_every = int(oc["log_every"])
    grad_clip = float(oc["grad_clip"])
    patience = int(oc["plateau"]["patience_vals"]); min_delta = float(oc["plateau"]["min_delta"])
    train_uuids = locked["split"]["train_uuids"]; val_uuids = locked["split"]["val_uuids"]
    inv = json.loads(Path(locked["data"]["inventory"]).read_text())
    files_list = inv.get("files") or inv.get("sessions") or []
    files_by_uuid = {}
    for f in files_list:
        u = f.get("uuid")
        hp = f.get("hidden_path") or f.get("hidden")
        tp = f.get("target_h12_path") or f.get("target_path") or f.get("target_h12")
        files_by_uuid[u] = {"uuid": u, "hidden_path": hp, "target_h12_path": tp, **f}

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    _ = torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats(0)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    log(f"C0 start seed={seed} H={H_frames} lambda={lambda_reg} arch={d_in}->{width}x{n_layers}->{embed} bs={bs} lr={lr}")
    log(f"train_files={len(train_uuids)} val_files={len(val_uuids)}")
    log("Loading train frames (CPU)...")
    H_train, Y_train = load_split_frames(files_by_uuid, train_uuids, T_valid)
    log("Loading val frames (CPU)...")
    H_val, Y_val = load_split_frames(files_by_uuid, val_uuids, T_valid)
    log(f"train_frames={H_train.shape[0]} val_frames={H_val.shape[0]}")

    model = JEPAHead(d_in=d_in, width=width, embed=embed, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    exp = int(locked["expected_param_count"])
    log(f"param_count={n_params} expected={exp}")
    assert n_params == exp and n_params < 10_000_000

    P = build_P(d_in, embed, seed, device)
    log(f"P frozen QR shape={tuple(P.shape)} seed={seed}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    metrics_rows = []
    best_val_mse = float("inf"); best_val = None; last_val = None
    n_val_points = 0; patience_left = patience
    early_stopped = False; stop_reason = "max_steps"
    N = H_train.shape[0]; step = 0; epoch = 0; t0 = time.time()

    def save_ckpt(step_i, epoch_i):
        torch.save({
            "mlp_state_dict": model.state_dict(),
            "P": P.detach().cpu(),
            "hyperparams": {
                "lr": lr, "weight_decay": wd, "batch_size": bs, "lambda_reg": lambda_reg,
                "grad_clip": grad_clip, "amp": "bf16", "seed": seed,
                "architecture": f"Linear({d_in},{width})+ReLU+Linear({width},{width})+ReLU+Linear({width},{embed})",
                "d_in": d_in, "width": width, "embed": embed, "n_layers": n_layers,
                "H_frames": H_frames, "T_valid": T_valid,
            },
            "split": {"train_uuids": train_uuids, "val_uuids": val_uuids, "seed": seed},
            "step": step_i, "epoch": epoch_i, "param_count": n_params,
            "best_val_L_mse": best_val_mse, "residual_inject": False, "moshi_loaded": False,
        }, CKPT_PATH)

    model.eval()
    with torch.no_grad():
        idx0 = torch.arange(min(bs, N))
        hb = H_train[idx0].to(device=device, dtype=torch.bfloat16)
        yb = Y_train[idx0].to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(hb)
        y_emb = (yb @ P).detach()
        L, L_mse, r_mean, r_var, r_cov, r_iso, r_energy = losses(pred.float(), y_emb, lambda_reg)
    row0 = {"step":0,"epoch":0,"split":"train","L":float(L.item()),"L_mse":float(L_mse.item()),
            "r_mean":float(r_mean.item()),"r_var":float(r_var.item()),"r_cov":float(r_cov.item()),
            "r_iso":float(r_iso.item()),"r_energy":float(r_energy.item())}
    metrics_rows.append(row0)
    with open(METRICS_JSONL,"w") as jf: jf.write(json.dumps(row0)+"\n")
    log(f"step=0 train L={row0['L']:.6f} L_mse={row0['L_mse']:.6f} r_iso={row0['r_iso']:.6f}")

    v0 = evaluate(model, H_val, Y_val, P, lambda_reg, device)
    n_val_points += 1
    last_val = {"step":0, **v0}; best_val_mse = v0["L_mse"]; best_val = dict(last_val)
    vrow = {"step":0,"epoch":0,"split":"val", **v0}
    metrics_rows.append(vrow)
    with open(METRICS_JSONL,"a") as jf: jf.write(json.dumps(vrow)+"\n")
    log(f"step=0 VAL L_mse={v0['L_mse']:.6f} L={v0['L']:.6f} r_iso={v0['r_iso']:.6f}")
    save_ckpt(0, 0)

    g_epoch = torch.Generator(); g_epoch.manual_seed(seed)
    while step < max_steps:
        epoch += 1
        perm = torch.randperm(N, generator=g_epoch)
        model.train()
        for start in range(0, N, bs):
            if step >= max_steps: break
            idx = perm[start:start+bs]
            if idx.numel() < 2: continue
            hb = H_train[idx].to(device=device, dtype=torch.bfloat16)
            yb = Y_train[idx].to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(hb)
            y_emb = (yb @ P).detach()
            L, L_mse, r_mean, r_var, r_cov, r_iso, r_energy = losses(pred.float(), y_emb, lambda_reg)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            step += 1

            if step % log_every == 0 or step == 1:
                row = {"step":step,"epoch":epoch,"split":"train","L":float(L.item()),"L_mse":float(L_mse.item()),
                       "r_mean":float(r_mean.item()),"r_var":float(r_var.item()),"r_cov":float(r_cov.item()),
                       "r_iso":float(r_iso.item()),"r_energy":float(r_energy.item())}
                metrics_rows.append(row)
                with open(METRICS_JSONL,"a") as jf: jf.write(json.dumps(row)+"\n")
                log(f"step={step} epoch={epoch} train L={row['L']:.6f} L_mse={row['L_mse']:.6f} r_iso={row['r_iso']:.6f}")

            if step % val_every == 0:
                v = evaluate(model, H_val, Y_val, P, lambda_reg, device)
                n_val_points += 1
                last_val = {"step": step, **v}
                improved = v["L_mse"] < (best_val_mse - min_delta)
                if v["L_mse"] < best_val_mse:
                    best_val_mse = v["L_mse"]; best_val = dict(last_val); save_ckpt(step, epoch)
                patience_left = patience if improved else patience_left - 1
                vrow = {"step":step,"epoch":epoch,"split":"val", **v, "best_val_L_mse":best_val_mse, "patience_left":patience_left}
                metrics_rows.append(vrow)
                with open(METRICS_JSONL,"a") as jf: jf.write(json.dumps(vrow)+"\n")
                log(f"step={step} VAL L_mse={v['L_mse']:.6f} best={best_val_mse:.6f} patience_left={patience_left}")
                PROGRESS_PATH.write_text(json.dumps({
                    "step":step,"epoch":epoch,"best_val_L_mse":best_val_mse,
                    "last_val_L_mse":v["L_mse"],"patience_left":patience_left,
                    "elapsed_sec":time.time()-t0}, indent=2)+"\n")
                model.train()
                if patience_left <= 0:
                    early_stopped = True
                    stop_reason = (
                        f"plateau: no val L_mse improve by {min_delta} for {patience} vals; "
                        f"best={best_val_mse:.6f} @ step={best_val['step']}"
                    )
                    log(stop_reason); break
        if early_stopped: break
        if step >= max_steps:
            stop_reason = f"hit_max_steps={max_steps}"; break

    elapsed = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    train_rows = [r for r in metrics_rows if r.get("split")=="train"]
    first_train, last_train = train_rows[0], train_rows[-1]
    summary = {
        "phase":"phase2_c0","host":"spark-61dd","seed":seed,"H_frames":H_frames,"lambda_reg":lambda_reg,
        "architecture": f"Linear({d_in},{width})+ReLU+Linear({width},{width})+ReLU+Linear({width},{embed})",
        "param_count": n_params, "param_count_M": round(n_params/1e6,6),
        "steps": step, "epochs": epoch, "stop_reason": stop_reason, "early_stopped": early_stopped,
        "elapsed_sec": elapsed,
        "gpu_peak_alloc_GiB": peak_alloc/(1024**3), "gpu_peak_reserved_GiB": peak_reserved/(1024**3),
        "train_first": {k: first_train[k] for k in ("step","L","L_mse","r_iso","r_energy")},
        "train_last": {k: last_train[k] for k in ("step","L","L_mse","r_iso","r_energy")},
        "train_L_mse_decreased": last_train["L_mse"] < first_train["L_mse"],
        "best_val": best_val, "last_val": last_val, "n_val_points": n_val_points,
        "n_train_frames": int(H_train.shape[0]), "n_val_frames": int(H_val.shape[0]),
        "moshi_loaded": False, "residual_inject": False,
        "paths": {"TRAIN_LOCKED": str(LOCKED_PATH), "checkpoint": str(CKPT_PATH),
                  "metrics_jsonl": str(METRICS_JSONL), "metrics_json": str(METRICS_JSON), "log": str(LOG_PATH)},
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2)+"\n")
    PROGRESS_PATH.write_text(json.dumps({"done":True,"step":step,"best_val_L_mse":best_val_mse,
        "stop_reason":stop_reason,"elapsed_sec":elapsed}, indent=2)+"\n")
    log(f"DONE steps={step} epochs={epoch} best_val_L_mse={best_val_mse:.6f} elapsed_sec={elapsed:.1f} reason={stop_reason}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
