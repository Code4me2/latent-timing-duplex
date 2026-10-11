#!/usr/bin/env python3
"""Build a causal silence-checkpoint JSONL manifest from two-channel VAD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from latent_timing_duplex.silence import extract_silence_checkpoints, sample_episodes


def _records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            return [value]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _intervals(obj: dict[str, Any], keys: tuple[str, ...]) -> list[tuple[float, float]]:
    for key in keys:
        if key not in obj:
            continue
        rows: list[tuple[float, float]] = []
        for item in obj[key]:
            if isinstance(item, dict):
                start = item.get("start_s", item.get("start"))
                end = item.get("end_s", item.get("end"))
            else:
                start, end = item[:2]
            rows.append((float(start), float(end)))
        return rows
    return []


def _split(lineage_id: str, seed: int) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{lineage_id}".encode()).digest()[:8], "big"
    ) / 2**64
    return "train" if value < 0.70 else "development" if value < 0.85 else "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=float,
        nargs="+",
        default=[0.24, 0.48, 0.96, 2.0, 3.0, 5.0, 10.0],
    )
    parser.add_argument("--max-per-session", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()

    checkpoints = []
    for obj in _records(args.vad):
        session_id = str(
            obj.get("session_id") or obj.get("conversation_id") or obj.get("uuid") or ""
        )
        if not session_id:
            raise ValueError("every VAD record requires a session_id")
        lineage_id = str(obj.get("lineage_id") or obj.get("podcast_id") or session_id)
        user = _intervals(obj, ("user_vad", "user", "left", "LEFT"))
        assistant = _intervals(
            obj, ("assistant_vad", "assistant", "agent_vad", "right", "RIGHT")
        )
        observed_end = max((end for _, end in user + assistant), default=0.0)
        duration_s = float(obj.get("duration_s", observed_end))
        checkpoints.extend(
            extract_silence_checkpoints(
                session_id,
                duration_s,
                user,
                assistant,
                checkpoints_s=args.checkpoints,
                source=str(obj.get("source", "unknown")),
                lineage_id=lineage_id,
            )
        )
    checkpoints = sample_episodes(
        checkpoints, max_per_session=args.max_per_session, seed=args.seed
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in checkpoints:
        lines.append(
            json.dumps(
                {
                    **row.__dict__,
                    "context": row.context.value,
                    "quality": row.quality.value,
                    "split": _split(row.lineage_id or row.session_id, args.seed),
                },
                sort_keys=True,
            )
        )
    payload = "\n".join(lines) + ("\n" if lines else "")
    args.output.write_text(payload, encoding="utf-8")
    lock = {
        "schema": "semantic-silence-manifest/v1",
        "input_sha256": hashlib.sha256(args.vad.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "checkpoints_s": args.checkpoints,
        "max_per_session": args.max_per_session,
        "split_seed": args.seed,
        "n_checkpoints": len(checkpoints),
        "n_episodes": len({row.episode_id for row in checkpoints}),
    }
    args.output.with_suffix(args.output.suffix + ".lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
