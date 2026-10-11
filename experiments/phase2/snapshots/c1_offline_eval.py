#!/usr/bin/env python3
"""C1 offline teacher-forced: inject vs shuffle vs none on speak-mass/overlap/gap."""
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

class JEPAHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4096, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256),
        )
    def forward(self, h):
        return self.net(h)

class Injector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 4096))
    def forward(self, s):
        return self.net(s)

class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4096, 1)
    def forward(self, h):
        return self.lin(h).squeeze(-1)

def intervals_to_mask(ivs, T, dt=DT):
    m = np.zeros(T, dtype=np.float32)
    for a, b in ivs or []:
        i0 = max(0, int(math.floor(float(a) / dt)))
        i1 = min(T, int(math.ceil(float(b) / dt)))
        m[i0:i1] = 1.0
    return m

def load_vad():
    out = {}
    for ln in VAD.open():
        r = json.loads(ln)
        out[r["session_id"]] = r
    return out

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    locked = json.loads((ART / "TRAIN_LOCKED.json").read_text())
    val_uuids = locked["split"]["val_uuids"]
    inv = {f["uuid"]: f for f in json.loads((C0 / "INVENTORY.json").read_text())["files"]}
    vad = load_vad()

    c0ck = torch.load(C0 / "checkpoint.pt", map_location="cpu", weights_only=False)
    jepa = JEPAHead().to(device)
    jepa.load_state_dict(c0ck["mlp_state_dict"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    ck = torch.load(ART / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck.get("residual_add_only") is True
    assert ck.get("moshi_loaded") is False
    g = Injector().to(device)
    readout = Readout().to(device)
    g.load_state_dict(ck["injector_state_dict"])
    readout.load_state_dict(ck["readout_state_dict"])
    g.eval(); readout.eval()

    hs, ss, ys, sids = [], [], [], []
    for u in val_uuids:
        f = inv[u]
        h_obj = torch.load(f["hidden_path"], map_location="cpu", weights_only=False)
        h = (h_obj["hidden"] if isinstance(h_obj, dict) else h_obj).float()[:T_VALID]
        if u not in vad:
            continue
        user = intervals_to_mask(vad[u]["user_vad"], T_VALID)
        y = torch.from_numpy(user[H:T_VALID]).float()
        h_ctx = h[: T_VALID - H]
        with torch.no_grad():
            s = jepa(h_ctx.to(device)).cpu()
        hs.append(h_ctx); ss.append(s); ys.append(y); sids.append(u)
    H_all = torch.cat(hs); S_all = torch.cat(ss); Y_all = torch.cat(ys)
    print(f"val_frames={H_all.shape[0]} n_sessions={len(sids)} ckpt_step={ck['step']} best_bce={ck['best_val_bce']:.6f}")

    def run_condition(name, use_g, shuffle, seed=SEED):
        g.eval(); readout.eval()
        tot_bce = 0.0; n_bat = 0
        speak_sum = over_sum = gap_sum = 0.0
        n = 0
        # session-level aggregates
        per_sess = []
        # build session frame ranges
        offsets = []
        cur = 0
        for t in [x.shape[0] for x in hs]:
            offsets.append((cur, cur + t)); cur += t
        with torch.no_grad():
            # global tensors optionally shuffled once
            Sb = S_all
            if shuffle:
                gen = torch.Generator().manual_seed(seed + 7)
                Sb = S_all[torch.randperm(S_all.shape[0], generator=gen)]
            for i0 in range(0, H_all.shape[0], 512):
                i1 = min(H_all.shape[0], i0 + 512)
                hb = H_all[i0:i1].to(device)
                sb = Sb[i0:i1].to(device)
                yb = Y_all[i0:i1].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
                    if use_g:
                        logits = readout(hb + g(sb))
                    else:
                        logits = readout(hb)
                p = torch.sigmoid(logits.float())
                bce = F.binary_cross_entropy_with_logits(logits.float(), yb, reduction="sum")
                tot_bce += float(bce)
                speak_sum += float(p.sum())
                over_sum += float((p * yb).sum())
                gap_sum += float((p * (1.0 - yb)).sum())
                n += (i1 - i0)
                n_bat += 1
            # per-session
            for sid, (a, b) in zip(sids, offsets):
                hb = H_all[a:b].to(device)
                sb = Sb[a:b].to(device)
                yb = Y_all[a:b].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
                    logits = readout(hb + g(sb)) if use_g else readout(hb)
                p = torch.sigmoid(logits.float())
                per_sess.append({
                    "session_id": sid,
                    "speak_mass": float(p.mean()),
                    "overlap_mass": float((p * yb).mean()),
                    "gap_mass": float((p * (1 - yb)).mean()),
                    "bce": float(F.binary_cross_entropy_with_logits(logits.float(), yb)),
                    "user_frame_frac": float(yb.mean()),
                })
        return {
            "condition": name,
            "speak_mass": speak_sum / n,
            "overlap_mass": over_sum / n,
            "gap_mass": gap_sum / n,
            "bce": tot_bce / n,  # mean per-frame BCE (sum/n)
            "n_frames": n,
            "n_sessions": len(sids),
            "per_session_mean_bce": float(np.mean([x["bce"] for x in per_sess])),
            "per_session": per_sess,
        }

    t0 = time.time()
    results = {}
    for name, use_g, shuffle in [
        ("inject", True, False),
        ("shuffle", True, True),
        ("none", False, False),
    ]:
        results[name] = run_condition(name, use_g, shuffle)
        r = results[name]
        print(f"{name}: speak={r['speak_mass']:.4f} overlap={r['overlap_mass']:.4f} gap={r['gap_mass']:.4f} bce={r['bce']:.4f}")

    # deltas relative to none and shuffle
    inj, shuf, none = results["inject"], results["shuffle"], results["none"]
    deltas = {
        "inject_minus_none": {
            "speak_mass": inj["speak_mass"] - none["speak_mass"],
            "overlap_mass": inj["overlap_mass"] - none["overlap_mass"],
            "gap_mass": inj["gap_mass"] - none["gap_mass"],
            "bce": inj["bce"] - none["bce"],
        },
        "inject_minus_shuffle": {
            "speak_mass": inj["speak_mass"] - shuf["speak_mass"],
            "overlap_mass": inj["overlap_mass"] - shuf["overlap_mass"],
            "gap_mass": inj["gap_mass"] - shuf["gap_mass"],
            "bce": inj["bce"] - shuf["bce"],
        },
        "shuffle_minus_none": {
            "speak_mass": shuf["speak_mass"] - none["speak_mass"],
            "overlap_mass": shuf["overlap_mass"] - none["overlap_mass"],
            "gap_mass": shuf["gap_mass"] - none["gap_mass"],
            "bce": shuf["bce"] - none["bce"],
        },
    }
    # interpretation helpers (desired: lower BCE, higher overlap, lower gap vs shuffle)
    verdict = {
        "inject_beats_shuffle_on_bce": inj["bce"] < shuf["bce"],
        "inject_beats_none_on_bce": inj["bce"] < none["bce"],
        "inject_beats_shuffle_on_overlap": inj["overlap_mass"] > shuf["overlap_mass"],
        "inject_lower_gap_than_none": inj["gap_mass"] < none["gap_mass"],
        "note": "Lower BCE / higher overlap_mass / lower gap_mass are preferred. Shuffle control should degrade vs inject if g(s) uses s meaningfully.",
    }

    out = {
        "phase": "phase2_c1_offline_eval",
        "host": "spark-61dd",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": str(ART / "checkpoint.pt"),
        "ckpt_step": ck["step"],
        "best_val_bce_train": ck["best_val_bce"],
        "residual_add_only": True,
        "moshi_loaded": False,
        "film_lora_peft": False,
        "live_rl": False,
        "protocol": {
            "description": "Offline teacher-forced residual-add: readout(h+g(s)) predicts user VAD at t+H=12; conditions inject / shuffle(s) / none",
            "speak_mass": "mean sigmoid(logit) over frames",
            "overlap_mass": "mean sigmoid(logit)*y_user over frames",
            "gap_mass": "mean sigmoid(logit)*(1-y_user) over frames",
            "bce": "mean per-frame BCEWithLogits",
            "vad_proxy": "DuplexChat/CANDOR channel-energy VAD — not gold turns",
            "val_uuids_n": len(val_uuids),
            "seed_shuffle": SEED + 7,
        },
        "conditions": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_session"}
            for k, v in results.items()
        },
        "per_session_by_condition": {k: v["per_session"] for k, v in results.items()},
        "deltas": deltas,
        "verdict": verdict,
        "elapsed_sec": time.time() - t0,
        "gpu_peak_alloc_GiB": (torch.cuda.max_memory_allocated(0) / (1024**3)) if device.type == "cuda" else 0.0,
    }
    (ART / "OFFLINE_EVAL.json").write_text(json.dumps(out, indent=2) + "\n")
    # slim metrics without per-session for quick read
    slim = {k: v for k, v in out.items() if k != "per_session_by_condition"}
    (ART / "metrics_offline.json").write_text(json.dumps(slim, indent=2) + "\n")
    print("WROTE", ART / "OFFLINE_EVAL.json")
    print("verdict", verdict)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
