#!/usr/bin/env python3
"""Phase2 L1 LIVE Moshi residual-add decode (NOT offline Linear(4096,1) readout)."""
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

ART = Path("/home/velvet/cs199-phase1-work/eval/phase2_c1_live_2026-09-25")
RUN_TAG = os.environ.get("L1_RUN_TAG", "").strip()  # "", "smoke", or "full"
C0 = Path("/home/velvet/cs199-phase1-work/eval/phase2_c0_2026-09-25")
C1 = Path("/home/velvet/cs199-phase1-work/eval/phase2_c1_2026-09-25")
HF_REPO = "kyutai/moshiko-pytorch-bf16"
SEED = 20260925
PT = timezone(timedelta(hours=-7))
MAX_SEC = float(os.environ.get("L1_MAX_SEC", "90"))
CORE_N = int(os.environ.get("L1_CORE_N", "6"))

def _tag(name: str) -> Path:
    """SUMMARY.json vs SUMMARY_smoke.json when RUN_TAG set."""
    if not RUN_TAG:
        return ART / name
    if name.endswith(".json"):
        stem = name[:-5]
        return ART / f"{stem}_{RUN_TAG}.json"
    if name.endswith(".md"):
        stem = name[:-3]
        return ART / f"{stem}_{RUN_TAG}.md"
    if name.endswith(".jsonl"):
        stem = name[:-6]
        return ART / f"{stem}_{RUN_TAG}.jsonl"
    if name.endswith(".log"):
        stem = name[:-4]
        return ART / f"{stem}_{RUN_TAG}.log"
    return ART / f"{name}_{RUN_TAG}"

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
    with open(_tag("l1_live.log"), "a") as f:
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
        return {"speak_mass": 0.0, "overlap_mass": 0.0, "gap_mass": 0.0, "mutual_silence_frac": 0.0, "n_frames": 0}
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

def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    open(_tag("l1_live.log"), "w").write("")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True)
    apps = [ln.strip() for ln in smi.stdout.splitlines() if ln.strip()]
    log(f"L1 start gpu_apps={apps!r} MAX_SEC={MAX_SEC} CORE_N={CORE_N}")

    locked_c1 = json.loads((C1 / "TRAIN_LOCKED.json").read_text())
    val_uuids = locked_c1["split"]["val_uuids"]
    sessions, missing = [], []
    for u in val_uuids:
        w = find_wav(u)
        if w is None:
            missing.append(u)
        else:
            sessions.append({"uuid": u, "wav": str(w)})
    if len(sessions) > CORE_N:
        sessions = sessions[:CORE_N]
        log(f"CORE subset -> {len(sessions)}")

    design = {
        "phase": "phase2_c1_live_L1",
        "locked_at_local": datetime.now(PT).isoformat(),
        "seed": SEED,
        "hf_repo": HF_REPO,
        "injection_site": "LMGen streaming: transformer_out AFTER forward_text / BEFORE depformer (pre-depformer)",
        "residual": "h' = h + g(stopgrad(JEPA(h))); JEPA=C0; g=C1 Injector",
        "conditions": ["inject", "shuffle", "none"],
        "monologue_untouched": True,
        "film_lora": False,
        "live_rl": False,
        "val_uuids_locked": val_uuids,
        "sessions_with_wav": sessions,
        "sessions_missing_wav": missing,
        "core_n": CORE_N,
        "L1_MAX_SEC": MAX_SEC,
        "subset_policy": f"Predeclared CORE of up to {CORE_N} locked-val sessions with resolvable mid180 wav; audio clipped to {MAX_SEC}s. LIVE Moshi decode — not offline readout.",
        "duplexchat_label_caveat": "Channel-energy VAD proxies only — never gold turn labels.",
        "c0_ckpt": str(C0 / "checkpoint.pt"),
        "c1_ckpt": str(C1 / "checkpoint.pt"),
    }
    (_tag("DESIGN_LOCKED.json")).write_text(json.dumps(design, indent=2) + "\n")
    log(f"DESIGN_LOCKED n_sessions={len(sessions)} missing={len(missing)}")
    if not sessions:
        (_tag("L1_GATE.json")).write_text(json.dumps({"decision": "FAIL", "reason": "no wav"}, indent=2) + "\n")
        return 2

    device = torch.device("cuda:0")
    assert torch.cuda.is_available()

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
    log(f"loaded heads c1_step={c1.get('step')} residual_add_only={c1.get('residual_add_only')}")

    from moshi.models import loaders, LMGen
    t0 = time.time()
    log(f"loading Moshi {HF_REPO}")
    info = loaders.CheckpointInfo.from_hf_repo(HF_REPO)
    mimi = info.get_mimi(device=device); mimi.eval()
    lm = info.get_moshi(device=device, dtype=torch.bfloat16); lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    log(f"Moshi loaded in {time.time()-t0:.1f}s dep_q={getattr(lm,'dep_q',None)} "
        f"num_cb={lm.num_codebooks} mem={torch.cuda.memory_allocated()/1024**3:.2f}GiB")

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    sr = int(mimi.sample_rate)
    frame_hz = float(mimi.frame_rate)

    def run_condition(mode: str, user_wav: np.ndarray) -> dict:
        n_align = (len(user_wav) // frame_size) * frame_size
        n_align = min(n_align, int(MAX_SEC * sr) // frame_size * frame_size)
        user_wav = user_wav[:n_align]
        n_frames = n_align // frame_size
        s_buf = []
        # torch.randint + Generator must be CPU-device in this torch build
        rng = torch.Generator(device="cpu")
        rng.manual_seed(SEED + (0 if mode != "shuffle" else 99))
        asst_codes_frames = []
        transformer_norms, residual_norms = [], []

        lm_gen = LMGen(lm, use_sampling=False, temp=0.0, temp_text=0.0)
        with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
            state = lm_gen._streaming_state

            def main_with_inject(input_tokens, condition_sum, condition_cross):
                tout, text_logits = lm.forward_text(input_tokens, condition_sum, condition_cross)
                h = tout
                hf = h[:, 0, :].float()
                s = jepa(hf)
                r = torch.zeros_like(hf)
                if mode == "inject":
                    r = inj(s)
                    h = h + r.to(h.dtype).unsqueeze(1)
                elif mode == "shuffle":
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
                residual_norms.append(float(r.norm(dim=-1).mean().item()) if mode != "none" else 0.0)
                return h, text_logits

            state.graphed_main = main_with_inject

            for fi in range(n_frames):
                chunk = torch.from_numpy(user_wav[fi * frame_size:(fi + 1) * frame_size]).float().to(device)
                chunk = chunk.view(1, 1, -1)
                codes = mimi.encode(chunk)
                out = lm_gen.step(codes)
                if out is None:
                    continue
                dep_q = lm.dep_q
                audio = out[:, 1:1 + dep_q, 0]
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
        met["decoded_sec"] = float(len(asst_np) / sr)
        return met

    results = {m: [] for m in ("inject", "shuffle", "none")}
    per_session = []
    t_run = time.time()
    open(_tag("per_session.jsonl"), "w").write("")
    for si, sess in enumerate(sessions):
        uuid = sess["uuid"]
        wav_path = Path(sess["wav"])
        log(f"==== session {si+1}/{len(sessions)} {uuid}")
        try:
            user, _ = load_stereo(wav_path, sr)
            row = {"uuid": uuid, "wav": str(wav_path), "conditions": {}}
            for mode in ("inject", "shuffle", "none"):
                log(f"  condition={mode}")
                t1 = time.time()
                met = run_condition(mode, user)
                met["elapsed_sec"] = time.time() - t1
                row["conditions"][mode] = met
                if "error" not in met:
                    results[mode].append(met)
                log(f"  {mode}: speak={met.get('speak_mass')} overlap={met.get('overlap_mass')} "
                    f"gap={met.get('gap_mass')} collapse={met.get('collapse_flag')} sec={met.get('elapsed_sec'):.1f}")
                (_tag("PROGRESS.json")).write_text(json.dumps({
                    "session_i": si, "uuid": uuid, "mode": mode,
                    "done_sessions": si, "n_total": len(sessions),
                    "elapsed_sec": time.time() - t_run,
                }, indent=2) + "\n")
            per_session.append(row)
            with open(_tag("per_session.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            log(f"ERROR session {uuid}: {e}\n{traceback.format_exc()}")
            per_session.append({"uuid": uuid, "error": str(e)})

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

    summary_cond = {m: agg(m) for m in ("inject", "shuffle", "none")}

    def mean_of(mode, k):
        a = summary_cond[mode]
        return None if a is None or a[k] is None else a[k]["mean"]

    deltas = {}
    for base in ("none", "shuffle"):
        deltas[f"inject_minus_{base}"] = {
            k: (None if mean_of("inject", k) is None or mean_of(base, k) is None
                else mean_of("inject", k) - mean_of(base, k))
            for k in ("speak_mass", "overlap_mass", "gap_mass")
        }

    finished_all = all(summary_cond[m] and summary_cond[m]["n_sessions"] == len(sessions)
                       for m in ("inject", "shuffle", "none"))
    n_err = sum(1 for r in per_session if "error" in r and "conditions" not in r)
    gate_pass = bool(finished_all and n_err == 0 and len(sessions) >= 1)
    nullish = False
    if gate_pass and deltas["inject_minus_shuffle"]["speak_mass"] is not None:
        d = deltas["inject_minus_shuffle"]
        nullish = all(abs(d[k]) < 1e-4 for k in d)

    gate = {
        "phase": "L1_GATE",
        "decision": "PASS" if (gate_pass and not nullish) else "FAIL",
        "finished_all_3_conditions": finished_all,
        "n_sessions_planned": len(sessions),
        "n_sessions_missing_wav": len(missing),
        "n_session_hard_errors": n_err,
        "nullish_deltas": nullish,
        "reason": (
            "all 3 conditions finished on CORE live Moshi residual-add decode; silence logged"
            if gate_pass and not nullish else
            ("inject≈shuffle null effect" if nullish else
             f"incomplete/crashed finished_all={finished_all} n_err={n_err}")
        ),
        "note_missing_wav": missing,
        "c2_allowed": bool(gate_pass and not nullish),
    }
    (_tag("L1_GATE.json")).write_text(json.dumps(gate, indent=2) + "\n")

    summary = {
        "phase": "phase2_c1_live_L1",
        "host": "spark-61dd",
        "seed": SEED,
        "hf_repo": HF_REPO,
        "injection_site": design["injection_site"],
        "n_sessions": len(sessions),
        "missing_wav": missing,
        "max_sec": MAX_SEC,
        "conditions": summary_cond,
        "deltas": deltas,
        "gate": gate,
        "elapsed_sec": time.time() - t_run,
        "gpu_peak_alloc_GiB": torch.cuda.max_memory_allocated(0) / 1024**3,
        "duplexchat_caveat": design["duplexchat_label_caveat"],
        "paths": {
            "artifact_root": str(ART),
            "DESIGN_LOCKED": str(_tag("DESIGN_LOCKED.json")),
            "L1_GATE": str(_tag("L1_GATE.json")),
            "REPORT": str(_tag("REPORT.md")),
            "SUMMARY": str(_tag("SUMMARY.json")),
            "START": str(ART / "START.json"),
            "log": str(_tag("l1_live.log")),
        },
    }
    (_tag("SUMMARY.json")).write_text(json.dumps(summary, indent=2) + "\n")
    (_tag("metrics.json")).write_text(json.dumps({"conditions": summary_cond, "deltas": deltas, "gate": gate}, indent=2) + "\n")

    def fmt(ci):
        if not ci:
            return "n/a"
        return f"{ci['mean']:.4f} [{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}] (n={ci['n']})"

    md = []
    md.append("# Phase2 L1 — LIVE Moshi residual-add decode")
    md.append("")
    md.append(f"Host spark-61dd. Seed {SEED}. Repo `{HF_REPO}`.")
    md.append(f"Injection: **{design['injection_site']}**.")
    md.append("**LIVE Moshi+Mimi decode** — not the C1 offline Linear(4096,1) readout.")
    md.append("")
    md.append("## Subset")
    md.append(f"- Locked val uuids: {len(val_uuids)}")
    md.append(f"- CORE with wav (run): **{len(sessions)}** (L1_CORE_N={CORE_N}, L1_MAX_SEC={MAX_SEC})")
    md.append(f"- Missing wav: {missing}")
    md.append("")
    md.append("## Metrics (channel-energy VAD proxies — NOT gold turns)")
    md.append("")
    md.append("| condition | speak_mass | overlap_mass | gap_mass | n_collapse |")
    md.append("|-----------|----------:|-------------:|---------:|-----------:|")
    for m in ("inject", "shuffle", "none"):
        a = summary_cond[m]
        if not a:
            md.append(f"| {m} | n/a | n/a | n/a | n/a |")
            continue
        md.append(f"| **{m}** | {fmt(a['speak_mass'])} | {fmt(a['overlap_mass'])} | {fmt(a['gap_mass'])} | {a['n_collapse']}/{a['n_sessions']} |")
    md.append("")
    md.append("### Deltas inject − baseline")
    md.append("```json")
    md.append(json.dumps(deltas, indent=2))
    md.append("```")
    md.append("")
    md.append(f"## L1_GATE: **{gate['decision']}**")
    md.append(f"Reason: {gate['reason']}")
    md.append(f"C2 allowed: {gate['c2_allowed']}")
    md.append("")
    md.append("## Caveat")
    md.append(design["duplexchat_label_caveat"])
    md.append("Do not claim timing control beyond these proxy metrics.")
    md.append("")
    md.append("## Paths")
    for k, pth in summary["paths"].items():
        md.append(f"- {k}: `{pth}`")
    (_tag("REPORT.md")).write_text("\n".join(md) + "\n")
    log(f"DONE gate={gate['decision']} elapsed={summary['elapsed_sec']:.1f}")
    (_tag("PROGRESS.json")).write_text(json.dumps({"done": True, "gate": gate["decision"]}, indent=2) + "\n")
    return 0 if gate["decision"] == "PASS" else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        ART.mkdir(parents=True, exist_ok=True)
        with open(_tag("l1_live.log"), "a") as f:
            f.write(f"FATAL {e}\n{traceback.format_exc()}\n")
        (_tag("L1_GATE.json")).write_text(json.dumps({"decision": "FAIL", "reason": f"fatal: {e}"}, indent=2) + "\n")
        raise
