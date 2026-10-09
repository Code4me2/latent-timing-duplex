#!/usr/bin/env python3
"""Phase2 C2: one-block LoRA on late Moshi layer + C1 residual-add.
TRAIN_LOCKED is written before any optimizer step. Live decode: inject/shuffle/none/lora_off.
"""
from __future__ import annotations
import json, math, os, sys, time, traceback, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

os.environ.setdefault("HF_HOME", "/home/velvet/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/home/velvet/.cache/huggingface/hub")
os.environ["NO_CUDA_GRAPH"] = "1"
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ART = Path("/home/velvet/cs199-phase1-work/eval/phase2_c2_2026-09-25")
C0 = Path("/home/velvet/cs199-phase1-work/eval/phase2_c0_2026-09-25")
C1 = Path("/home/velvet/cs199-phase1-work/eval/phase2_c1_2026-09-25")
L1 = Path("/home/velvet/cs199-phase1-work/eval/phase2_c1_live_2026-09-25")
HF_REPO = "kyutai/moshiko-pytorch-bf16"
SEED = 20260925
PT = timezone(timedelta(hours=-7))
LORA_RANK = int(os.environ.get("C2_LORA_RANK", "8"))
LORA_SCALING = float(os.environ.get("C2_LORA_SCALING", "2.0"))
MAX_STEPS = int(os.environ.get("C2_MAX_STEPS", "200"))
T_CHUNK = int(os.environ.get("C2_T_CHUNK", "48"))
TRAIN_N = int(os.environ.get("C2_TRAIN_N", "8"))
MAX_SEC = float(os.environ.get("C2_MAX_SEC", "180"))
CORE_N = int(os.environ.get("C2_CORE_N", "12"))
SKIP_TRAIN = os.environ.get("C2_SKIP_TRAIN", "0") == "1"
SKIP_LIVE = os.environ.get("C2_SKIP_LIVE", "0") == "1"

WAV_ROOTS = [
    Path("/home/velvet/cs199-phase1-work/ablations/windows/candor"),
    Path("/home/velvet/cs199-candor-work/window180_mid/candor"),
    Path("/home/velvet/cs199-candor-work/window180_mid/dc"),
    Path("/home/velvet/cs199-phase1-work/eval/overnight_2026-09-04/windows/dc"),
    Path("/home/velvet/cs199-phase1-work/eval/overnight_2026-09-06/windows/dc"),
    Path("/home/velvet/cs199-phase1-work/eval/overnight_2026-09-07/windows/dc"),
    Path("/home/velvet/cs199-phase1-work/eval/overnight_2026-09-12/windows/dc"),
    Path("/home/velvet/cs199-phase1-work/ablations/windows/dc"),
]

def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line); sys.stdout.flush()
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / "c2.log", "a") as f:
        f.write(line); f.flush()

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

def find_wav(uuid: str):
    for root in WAV_ROOTS:
        if not root.exists():
            continue
        for name in (f"{uuid}_mid180_24k.wav", f"{uuid}_mid180.wav"):
            p = root / name
            if p.is_file():
                return p
        hits = sorted(root.glob(f"*{uuid}*mid180*.wav")) or sorted(root.glob(f"*{uuid}*.wav"))
        if hits:
            return hits[0]
    return None

def load_stereo(path: Path, sr_target: int):
    import sphn
    wav, sr = sphn.read(str(path), sample_rate=sr_target)
    wav = np.asarray(wav)
    if wav.ndim == 1:
        return wav.astype(np.float32), np.zeros_like(wav, dtype=np.float32)
    if wav.shape[0] == 2:
        return wav[0].astype(np.float32), wav[1].astype(np.float32)
    if wav.shape[-1] == 2:
        return wav[:, 0].astype(np.float32), wav[:, 1].astype(np.float32)
    u = wav[0] if wav.shape[0] < wav.shape[-1] else wav[:, 0]
    return u.astype(np.float32), np.zeros_like(u, dtype=np.float32)

def frame_energy_vad(x, sr, frame_ms=25.0, hop_ms=10.0, k_mad=3.0):
    frame = int(round(frame_ms * 1e-3 * sr))
    hop = int(round(hop_ms * 1e-3 * sr))
    if len(x) < frame:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    n = 1 + (len(x) - frame) // hop
    rms = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = x[i * hop:i * hop + frame]
        rms[i] = math.sqrt(float(np.mean(seg * seg)) + 1e-12)
    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med)))
    if mad < 1e-12:
        T = float(np.percentile(rms, 80))
        if T <= med:
            T = med + 1e-12
    else:
        T = med + k_mad * mad
    return rms > T, rms

def align_mask(mask, n):
    if mask.size == 0:
        return np.zeros(n, dtype=bool)
    idx = np.linspace(0, len(mask) - 1, n).astype(np.int64)
    return mask[idx]

def metrics_from_masks(user_m, asst_m):
    n = len(user_m)
    if n == 0:
        return {"speak_mass": 0.0, "overlap_mass": 0.0, "gap_mass": 0.0,
                "mutual_silence_frac": 0.0, "n_frames": 0}
    return {
        "speak_mass": float(asst_m.mean()),
        "overlap_mass": float((asst_m & user_m).mean()),
        "gap_mass": float((asst_m & ~user_m).mean()),
        "mutual_silence_frac": float((~asst_m & ~user_m).mean()),
        "n_frames": int(n),
    }

def silence_collapse(asst_m, asst_rms_f, frame_hz=12.5):
    DT = 1.0 / frame_hz
    sil = ~asst_m
    collapse = False
    n_long = 0
    long_vars = []
    i = 0
    T = len(sil)
    while i < T:
        if not sil[i]:
            i += 1
            continue
        j = i + 1
        while j < T and sil[j]:
            j += 1
        if (j - i) * DT >= 2.0:
            n_long += 1
            chunk = asst_rms_f[i:j]
            v = float(np.var(chunk)) if chunk.size else 0.0
            long_vars.append(v)
            if v < 1e-8 or (float(np.mean(chunk)) < 1e-4 and float(np.std(chunk)) < 1e-6):
                collapse = True
        i = j
    return {
        "collapse_flag": bool(collapse),
        "n_long_silence_runs": n_long,
        "long_silence_var_mean": float(np.mean(long_vars)) if long_vars else None,
    }

def bootstrap_ci(values, seed=SEED, B=2000):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    means = np.sort(np.array([float(v[rng.integers(0, len(v), size=len(v))].mean()) for _ in range(B)]))
    return {"mean": float(v.mean()), "ci95": [float(means[int(0.025 * B)]), float(means[int(0.975 * B)])], "n": int(v.size)}

def apply_lora_one_block(lm, block_index, rank, scaling):
    from moshi.modules.lora import replace_all_linear_with_lora
    layers = lm.transformer.layers
    replace_all_linear_with_lora(layers[block_index], rank, scaling)
    return block_index

def freeze_non_lora(lm):
    from moshi.modules.lora import LoRALinear
    for p in lm.parameters():
        p.requires_grad_(False)
    for m in lm.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_A.parameters():
                p.requires_grad_(True)
            for p in m.lora_B.parameters():
                p.requires_grad_(True)

def lora_param_list(lm):
    from moshi.modules.lora import LoRALinear
    params = []
    for m in lm.modules():
        if isinstance(m, LoRALinear):
            params += list(m.lora_A.parameters()) + list(m.lora_B.parameters())
    return params

def set_lora_scaling(lm, scaling: float):
    from moshi.modules.lora import LoRALinear
    for m in lm.modules():
        if isinstance(m, LoRALinear):
            m.scaling = scaling

def pack_codes(lm, user_codes, moshi_codes):
    B, Ku, T = user_codes.shape
    Tuse = min(T, moshi_codes.shape[-1])
    user_codes = user_codes[..., :Tuse]
    moshi_codes = moshi_codes[..., :Tuse]
    K = lm.num_codebooks
    zero_id = getattr(lm, "zero_token_id", getattr(lm, "ungenerated_token_id", -1))
    text_pad = getattr(lm, "text_padding_token_id", getattr(lm, "existing_text_padding_id", 3))
    codes = torch.full((B, K, Tuse), zero_id, device=user_codes.device, dtype=torch.long)
    codes[:, 0, :] = text_pad
    dep_q = lm.dep_q
    codes[:, 1:1 + dep_q, :] = moshi_codes[:, :dep_q, :]
    user_slots = lm.n_q - dep_q
    codes[:, 1 + dep_q:1 + dep_q + user_slots, :] = user_codes[:, :user_slots, :]
    return codes

def encode_wav(mimi, wav_b1n, frame_size):
    n = wav_b1n.shape[-1]
    nn = (n // frame_size) * frame_size
    if nn < frame_size:
        raise RuntimeError("wav too short")
    with torch.no_grad():
        with mimi.streaming(1):
            return mimi.encode(wav_b1n[..., :nn])

def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    open(ART / "c2.log", "w").write("")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    assert torch.cuda.is_available()

    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    apps = [ln.strip() for ln in smi.stdout.splitlines() if ln.strip()]
    log(f"C2 start gpu_apps={apps!r} MAX_STEPS={MAX_STEPS} T_CHUNK={T_CHUNK}")

    gate = json.loads((L1 / "L1_GATE.json").read_text())
    assert gate.get("decision") == "PASS" and gate.get("c2_allowed", True), gate

    locked_c1 = json.loads((C1 / "TRAIN_LOCKED.json").read_text())
    train_uuids = locked_c1["split"]["train_uuids"]
    val_uuids = locked_c1["split"]["val_uuids"]

    train_sess, val_sess, missing = [], [], []
    for u in train_uuids:
        w = find_wav(u)
        if w is not None:
            train_sess.append({"uuid": u, "wav": str(w)})
        if len(train_sess) >= TRAIN_N:
            break
    for u in val_uuids:
        w = find_wav(u)
        if w is None:
            missing.append(u)
        else:
            val_sess.append({"uuid": u, "wav": str(w)})
    if len(val_sess) > CORE_N:
        val_sess = val_sess[:CORE_N]
    assert train_sess, "no train wavs resolved"

    # ---- TRAIN_LOCKED BEFORE optimizer ----
    train_locked = {
        "phase": "phase2_c2_train",
        "locked_before_any_optimizer_step": True,
        "locked_at_local": datetime.now(PT).isoformat(),
        "seed": SEED,
        "hf_repo": HF_REPO,
        "architecture": {
            "lora": "one_block LoRA on late Moshi StreamingTransformer layer",
            "lora_block_policy": "last transformer layer (n_layers-1)",
            "lora_rank": LORA_RANK,
            "lora_scaling": LORA_SCALING,
            "residual_add": "h' = h + g(stopgrad(JEPA(h))); JEPA=C0 frozen; g=C1 Injector frozen",
            "readout": "Linear(4096,1) trained with LoRA",
            "film": False,
            "monologue_untouched": True,
        },
        "optim": {
            "lr": 1e-4,
            "max_steps": MAX_STEPS,
            "t_chunk": T_CHUNK,
            "train_n_sessions": len(train_sess),
            "loss": "BCE(readout(h+residual), y_user_vad_proxy) + 1e-4*||residual||^2",
            "label_caveat": "Channel-energy VAD proxies only — never gold turn labels.",
        },
        "c0_ckpt": str(C0 / "checkpoint.pt"),
        "c1_ckpt": str(C1 / "checkpoint.pt"),
        "l1_gate": str(L1 / "L1_GATE.json"),
        "train_sessions": train_sess,
        "val_sessions_for_live": val_sess,
        "missing_wav_val": missing,
    }
    (ART / "TRAIN_LOCKED.json").write_text(json.dumps(train_locked, indent=2) + "\n")
    log(f"TRAIN_LOCKED written n_train={len(train_sess)} n_val_live={len(val_sess)}")

    c0 = torch.load(C0 / "checkpoint.pt", map_location="cpu", weights_only=False)
    c1 = torch.load(C1 / "checkpoint.pt", map_location="cpu", weights_only=False)
    jepa = JEPAHead().to(device).eval()
    jepa.load_state_dict(c0["mlp_state_dict"])
    for p in jepa.parameters():
        p.requires_grad_(False)
    inj = Injector().to(device).eval()
    inj.load_state_dict(c1["injector_state_dict"])
    for p in inj.parameters():
        p.requires_grad_(False)
    readout = Readout().to(device)
    if "readout_state_dict" in c1:
        readout.load_state_dict(c1["readout_state_dict"])
    log("loaded C0/C1 heads")

    from moshi.models import loaders, LMGen
    from moshi.models.lm import _delay_sequence
    from moshi.modules.lora import LoRALinear

    t0 = time.time()
    log(f"loading Moshi {HF_REPO}")
    info = loaders.CheckpointInfo.from_hf_repo(HF_REPO)
    mimi = info.get_mimi(device=device)
    mimi.eval()
    lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    n_layers = len(lm.transformer.layers)
    block_i = n_layers - 1
    apply_lora_one_block(lm, block_i, LORA_RANK, LORA_SCALING)
    freeze_non_lora(lm)
    lora_params = lora_param_list(lm)
    n_lora = sum(p.numel() for p in lora_params)
    log(f"Moshi+LoRA ready in {time.time()-t0:.1f}s layers={n_layers} block={block_i} "
        f"lora_params={n_lora} mem={torch.cuda.memory_allocated()/1024**3:.2f}GiB")

    train_locked["architecture"]["lora_block_index"] = block_i
    train_locked["architecture"]["n_layers"] = n_layers
    train_locked["architecture"]["lora_param_count"] = n_lora
    (ART / "TRAIN_LOCKED.json").write_text(json.dumps(train_locked, indent=2) + "\n")

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    sr = int(mimi.sample_rate)
    frame_hz = float(mimi.frame_rate)
    ckpt_path = ART / "checkpoint.pt"

    if not SKIP_TRAIN:
        opt = torch.optim.AdamW(list(lora_params) + list(readout.parameters()), lr=1e-4, weight_decay=0.01)
        log(f"optimizer created AFTER TRAIN_LOCKED; steps={MAX_STEPS}")
        sil = torch.zeros(1, 1, T_CHUNK * frame_size, device=device)
        sil_codes = encode_wav(mimi, sil, frame_size)

        cached = []
        for sess in train_sess:
            user, _ = load_stereo(Path(sess["wav"]), sr)
            user = user[: int(30 * sr)]
            wav_t = torch.from_numpy(user).float().to(device).view(1, 1, -1)
            try:
                uc = encode_wav(mimi, wav_t, frame_size)
            except Exception as e:
                log(f"encode fail {sess['uuid']}: {e}")
                continue
            user_m, _ = frame_energy_vad(user, sr)
            y = align_mask(user_m, uc.shape[-1]).astype(np.float32)
            cached.append({"uuid": sess["uuid"], "user_codes": uc.cpu(), "y": torch.from_numpy(y)})
            log(f"cached train {sess['uuid']} T={uc.shape[-1]}")
        assert cached, "no cached train codes"

        best_loss = float("inf")
        metrics_rows = []
        t_train = time.time()
        rng = np.random.default_rng(SEED)
        for step in range(1, MAX_STEPS + 1):
            item = cached[int(rng.integers(0, len(cached)))]
            uc = item["user_codes"].to(device)
            y = item["y"].to(device)
            T = uc.shape[-1]
            t0c = 0 if T <= T_CHUNK + 1 else int(rng.integers(0, T - T_CHUNK))
            uc_c = uc[:, :, t0c:t0c + T_CHUNK]
            y_c = y[t0c:t0c + T_CHUNK]
            sc = sil_codes[:, :, : uc_c.shape[-1]]
            codes = pack_codes(lm, uc_c, sc)
            B, K, Tc = codes.shape
            initial = lm._get_initial_token().expand(B, -1, -1)
            delayed = _delay_sequence(lm.delays, codes, initial)
            delayed = torch.cat([initial, delayed], dim=2)

            for m in lm.modules():
                if isinstance(m, LoRALinear):
                    m.train()
                else:
                    m.eval()
            tout, _ = lm.forward_text(delayed[:, :, :-1], None, None)
            h = tout.float()
            T_h = min(h.shape[1], y_c.shape[0])
            h = h[:, :T_h]
            y_c = y_c[:T_h]
            hf = h[0]
            with torch.no_grad():
                s = jepa(hf.detach())
                r = inj(s)
            # residual detached from JEPA/injector but added into h for readout+LoRA path:
            # LoRA grads flow through hf; residual is constant w.r.t. LoRA
            h_p = hf + r.detach()
            logits = readout(h_p)
            bce = F.binary_cross_entropy_with_logits(logits, y_c)
            l2 = (r.float() ** 2).mean()
            loss = bce + 1e-4 * l2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            row = {"step": step, "loss": float(loss.item()), "bce": float(bce.item()),
                   "l2": float(l2.item()), "uuid": item["uuid"]}
            metrics_rows.append(row)
            if step % 10 == 0 or step == 1:
                log(f"step {step}/{MAX_STEPS} loss={loss.item():.4f} bce={bce.item():.4f}")
                (ART / "PROGRESS.json").write_text(json.dumps(
                    {"phase": "train", "step": step, "loss": row["loss"]}, indent=2) + "\n")
            if loss.item() < best_loss:
                best_loss = loss.item()
                lora_sd = {k: v.detach().cpu() for k, v in lm.state_dict().items() if "lora_" in k}
                torch.save({
                    "lora_state_dict": lora_sd,
                    "readout_state_dict": readout.state_dict(),
                    "block_index": block_i,
                    "lora_rank": LORA_RANK,
                    "lora_scaling": LORA_SCALING,
                    "step": step,
                    "best_loss": best_loss,
                    "residual_add_only": True,
                    "monologue_untouched": True,
                }, ckpt_path)
            del tout, h, hf, h_p, logits, loss, bce, l2, codes, delayed
            if step % 20 == 0:
                torch.cuda.empty_cache()

        (ART / "train_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in metrics_rows) + "\n")
        (ART / "train_metrics.json").write_text(json.dumps({
            "n_steps": len(metrics_rows), "best_loss": best_loss,
            "elapsed_sec": time.time() - t_train, "lora_params": n_lora, "block_index": block_i,
        }, indent=2) + "\n")
        log(f"train done best_loss={best_loss:.4f}")
    else:
        log("SKIP_TRAIN")

    if SKIP_LIVE:
        log("SKIP_LIVE")
        return 0

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        miss, unexp = lm.load_state_dict(ck["lora_state_dict"], strict=False)
        log(f"loaded lora ckpt step={ck.get('step')} miss={len(miss)} unexp={len(unexp)}")
        readout.load_state_dict(ck["readout_state_dict"])
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.eval()
    set_lora_scaling(lm, LORA_SCALING)

    design = {
        "phase": "phase2_c2_live",
        "locked_at_local": datetime.now(PT).isoformat(),
        "seed": SEED,
        "hf_repo": HF_REPO,
        "lora_block_index": block_i,
        "lora_rank": LORA_RANK,
        "residual": "h' = h + g(stopgrad(JEPA(h))) with C1; LoRA on late Moshi block",
        "conditions": ["inject", "shuffle", "none", "lora_off"],
        "monologue_untouched": True,
        "film": False,
        "sessions": val_sess,
        "missing_wav": missing,
        "max_sec": MAX_SEC,
        "duplexchat_label_caveat": "Channel-energy VAD proxies only — never gold turn labels.",
    }
    (ART / "DESIGN_LOCKED.json").write_text(json.dumps(design, indent=2) + "\n")

    def run_condition(mode: str, user_wav: np.ndarray) -> dict:
        n_align = (len(user_wav) // frame_size) * frame_size
        n_align = min(n_align, int(MAX_SEC * sr) // frame_size * frame_size)
        user_wav = user_wav[:n_align]
        n_frames = n_align // frame_size
        s_buf = []
        rng = torch.Generator(device="cpu")
        rng.manual_seed(SEED + (0 if mode != "shuffle" else 99))
        asst_codes_frames = []
        transformer_norms, residual_norms = [], []

        lora_on = mode != "lora_off"
        set_lora_scaling(lm, LORA_SCALING if lora_on else 0.0)
        inject_mode = "inject" if mode == "lora_off" else mode

        lm_gen = LMGen(lm, use_sampling=False, temp=0.0, temp_text=0.0)
        with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
            state = lm_gen._streaming_state

            def main_with_inject(input_tokens, condition_sum, condition_cross):
                tout, text_logits = lm.forward_text(input_tokens, condition_sum, condition_cross)
                h = tout
                hf = h[:, 0, :].float()
                s = jepa(hf)
                r = torch.zeros_like(hf)
                if inject_mode == "inject":
                    r = inj(s)
                    h = h + r.to(h.dtype).unsqueeze(1)
                elif inject_mode == "shuffle":
                    s_buf.append(s.detach())
                    if len(s_buf) > 64:
                        s_buf.pop(0)
                    if len(s_buf) >= 2:
                        j = int(torch.randint(0, len(s_buf) - 1, (1,), generator=rng).item())
                        s_sh = s_buf[j]
                        if torch.allclose(s_sh, s):
                            s_sh = s_buf[(j + 1) % len(s_buf)]
                        r = inj(s_sh)
                    else:
                        r = inj(s)
                    h = h + r.to(h.dtype).unsqueeze(1)
                transformer_norms.append(float(hf.norm(dim=-1).mean().item()))
                residual_norms.append(float(r.norm(dim=-1).mean().item()) if inject_mode != "none" else 0.0)
                return h, text_logits

            state.graphed_main = main_with_inject
            for fi in range(n_frames):
                chunk = torch.from_numpy(user_wav[fi * frame_size:(fi + 1) * frame_size]).float().to(device)
                chunk = chunk.view(1, 1, -1)
                codes = mimi.encode(chunk)
                out = lm_gen.step(codes)
                if out is None:
                    continue
                audio = out[:, 1:1 + lm.dep_q, 0]
                asst_codes_frames.append(audio[0].detach().to("cpu"))

        if not asst_codes_frames:
            return {"error": "no_audio_tokens", "n_frames_in": n_frames}
        codes_TxQ = torch.stack(asst_codes_frames, dim=-1)
        codes_b = codes_TxQ.unsqueeze(0).to(device)
        with torch.no_grad():
            Ku = mimi.num_codebooks
            if codes_b.shape[1] < Ku:
                pad = torch.zeros(1, Ku - codes_b.shape[1], codes_b.shape[2], device=device, dtype=codes_b.dtype)
                codes_b = torch.cat([codes_b, pad], dim=1)
            elif codes_b.shape[1] > Ku:
                codes_b = codes_b[:, :Ku]
            wav_out = mimi.decode(codes_b)
        asst_np = wav_out[0, 0].detach().float().cpu().numpy()
        user_m, _ = frame_energy_vad(user_wav, sr)
        asst_m, asst_rms = frame_energy_vad(asst_np, sr)
        user_mf = align_mask(user_m, len(asst_codes_frames))
        asst_mf = align_mask(asst_m, len(asst_codes_frames))
        asst_rms_f = np.interp(
            np.linspace(0, max(len(asst_rms) - 1, 0), len(asst_codes_frames)),
            np.arange(len(asst_rms)) if len(asst_rms) else [0],
            asst_rms if len(asst_rms) else [0.0],
        )
        met = metrics_from_masks(user_mf, asst_mf)
        met.update(silence_collapse(asst_mf, asst_rms_f, frame_hz=frame_hz))
        met["n_decode_frames"] = len(asst_codes_frames)
        met["n_input_frames"] = n_frames
        met["mean_transformer_norm"] = float(np.mean(transformer_norms)) if transformer_norms else None
        met["mean_residual_norm"] = float(np.mean(residual_norms)) if residual_norms else None
        met["mode"] = mode
        met["lora_enabled"] = lora_on
        met["decoded_sec"] = float(len(asst_np) / sr)
        return met

    modes = ["inject", "shuffle", "none", "lora_off"]
    results = {m: [] for m in modes}
    open(ART / "per_session.jsonl", "w").write("")
    t_run = time.time()
    for si, sess in enumerate(val_sess):
        uuid = sess["uuid"]
        log(f"==== session {si+1}/{len(val_sess)} {uuid}")
        try:
            user, _ = load_stereo(Path(sess["wav"]), sr)
            row = {"uuid": uuid, "wav": sess["wav"], "conditions": {}}
            for mode in modes:
                log(f"  condition={mode}")
                t1 = time.time()
                met = run_condition(mode, user)
                met["elapsed_sec"] = time.time() - t1
                row["conditions"][mode] = met
                if "error" not in met:
                    results[mode].append(met)
                log(f"  {mode}: speak={met.get('speak_mass')} overlap={met.get('overlap_mass')} "
                    f"gap={met.get('gap_mass')} collapse={met.get('collapse_flag')} sec={met['elapsed_sec']:.1f}")
                (ART / "PROGRESS.json").write_text(json.dumps({
                    "phase": "live", "session_i": si, "uuid": uuid, "mode": mode,
                    "done_sessions": si, "n_total": len(val_sess), "elapsed_sec": time.time() - t_run,
                }, indent=2) + "\n")
            with open(ART / "per_session.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            log(f"ERROR session {uuid}: {e}\n{traceback.format_exc()}")

    def agg(mode):
        rows = results[mode]
        if not rows:
            return None
        out = {}
        for k in ("speak_mass", "overlap_mass", "gap_mass", "mutual_silence_frac", "mean_residual_norm"):
            vals = [r[k] for r in rows if r.get(k) is not None]
            out[k] = bootstrap_ci(vals) if vals else None
        out["n_sessions"] = len(rows)
        out["n_collapse"] = sum(1 for r in rows if r.get("collapse_flag"))
        out["collapse_rate"] = out["n_collapse"] / max(len(rows), 1)
        return out

    summary_cond = {m: agg(m) for m in modes}

    def mean_of(mode, k):
        a = summary_cond[mode]
        return None if a is None or a[k] is None else a[k]["mean"]

    deltas = {}
    for base in ("none", "shuffle", "lora_off"):
        deltas[f"inject_minus_{base}"] = {
            k: (None if mean_of("inject", k) is None or mean_of(base, k) is None
                else mean_of("inject", k) - mean_of(base, k))
            for k in ("speak_mass", "overlap_mass", "gap_mass")
        }

    finished = all(summary_cond[m] and summary_cond[m]["n_sessions"] == len(val_sess) for m in modes)
    gate_out = {
        "phase": "C2_GATE",
        "decision": "PASS" if finished else "FAIL",
        "finished_all_conditions": finished,
        "n_sessions": len(val_sess),
        "missing_wav": missing,
        "reason": ("all conditions finished live Moshi LoRA+residual decode; silence logged"
                   if finished else "incomplete"),
        "lora_block_index": block_i,
    }
    (ART / "C2_GATE.json").write_text(json.dumps(gate_out, indent=2) + "\n")
    summary = {
        "phase": "phase2_c2_live",
        "host": "spark-61dd",
        "seed": SEED,
        "hf_repo": HF_REPO,
        "lora_block_index": block_i,
        "lora_rank": LORA_RANK,
        "n_sessions": len(val_sess),
        "missing_wav": missing,
        "max_sec": MAX_SEC,
        "conditions": summary_cond,
        "deltas": deltas,
        "gate": gate_out,
        "elapsed_sec": time.time() - t_run,
        "gpu_peak_alloc_GiB": torch.cuda.max_memory_allocated(0) / 1024 ** 3,
        "duplexchat_caveat": design["duplexchat_label_caveat"],
        "paths": {
            "artifact_root": str(ART),
            "TRAIN_LOCKED": str(ART / "TRAIN_LOCKED.json"),
            "DESIGN_LOCKED": str(ART / "DESIGN_LOCKED.json"),
            "C2_GATE": str(ART / "C2_GATE.json"),
            "REPORT": str(ART / "REPORT.md"),
            "SUMMARY": str(ART / "SUMMARY.json"),
            "checkpoint": str(ckpt_path),
        },
    }
    (ART / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ART / "metrics.json").write_text(json.dumps(
        {"conditions": summary_cond, "deltas": deltas, "gate": gate_out}, indent=2) + "\n")

    def fmt(ci):
        if not ci:
            return "n/a"
        return f"{ci['mean']:.4f} [{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}] (n={ci['n']})"

    md = [
        "# Phase2 C2 — one-block LoRA + residual-add live decode", "",
        f"Host spark-61dd. Seed {SEED}. LoRA block={block_i} rank={LORA_RANK}.",
        "Residual-add from frozen C1; monologue untouched; no FiLM.", "",
        "## Metrics (channel-energy VAD proxies — NOT gold turns)", "",
        "| condition | speak_mass | overlap_mass | gap_mass | n_collapse |",
        "|-----------|----------:|-------------:|---------:|-----------:|",
    ]
    for m in modes:
        a = summary_cond[m]
        if not a:
            md.append(f"| {m} | n/a | n/a | n/a | n/a |")
        else:
            md.append(f"| **{m}** | {fmt(a['speak_mass'])} | {fmt(a['overlap_mass'])} | "
                      f"{fmt(a['gap_mass'])} | {a['n_collapse']}/{a['n_sessions']} |")
    md += ["", "### Deltas", "```json", json.dumps(deltas, indent=2), "```", "",
           f"## C2_GATE: **{gate_out['decision']}**", f"Reason: {gate_out['reason']}", "",
           "## Caveat", design["duplexchat_label_caveat"],
           "Do not claim timing control beyond these proxy metrics.", ""]
    (ART / "REPORT.md").write_text("\n".join(md) + "\n")
    log(f"DONE gate={gate_out['decision']} elapsed={summary['elapsed_sec']:.1f}")
    (ART / "PROGRESS.json").write_text(json.dumps({"done": True, "gate": gate_out["decision"]}, indent=2) + "\n")
    return 0 if gate_out["decision"] == "PASS" else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        ART.mkdir(parents=True, exist_ok=True)
        with open(ART / "c2.log", "a") as f:
            f.write(f"FATAL {e}\n{traceback.format_exc()}\n")
        (ART / "C2_GATE.json").write_text(json.dumps(
            {"decision": "FAIL", "reason": f"fatal: {e}"}, indent=2) + "\n")
        raise
