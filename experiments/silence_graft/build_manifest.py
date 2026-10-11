#!/usr/bin/env python3
"""Build locked full and pilot silence-checkpoint manifests from JSONL VAD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from latent_timing_duplex.config import repo_root
from latent_timing_duplex.silence import (
    SilenceCheckpoint,
    SilenceContext,
    extract_silence_checkpoints,
    sample_episodes,
)

try:
    from experiments.silence_graft.protocol_v1 import load_protocol_v1
    from experiments.silence_graft.receipts import read_hmac_key, receipt_bytes, sign_receipt
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from protocol_v1 import load_protocol_v1
    from receipts import read_hmac_key, receipt_bytes, sign_receipt

_PROTOCOL = "DATA_LABELING_PROTOCOL.md"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _records(payload: bytes) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("VAD input must be UTF-8") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_strict_object, parse_constant=_nonfinite)
        except (json.JSONDecodeError, ValueError) as error:
            message = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
            raise ValueError(f"malformed JSON on line {line_number}: {message}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSON row {line_number} must be an object")
        records.append(row)
    return records


def _required_string(obj: Mapping[str, Any], key: str, line_number: int) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"JSON row {line_number} requires an explicit non-empty {key}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _intervals(
    obj: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    duration_s: float,
    channel: str,
) -> list[tuple[float, float]]:
    aliases = [key for key in keys if key in obj]
    if len(aliases) != 1:
        raise ValueError(f"{channel} VAD requires exactly one explicit alias")
    value = obj[aliases[0]]
    if not isinstance(value, list):
        raise ValueError(f"{channel} VAD must be a list")
    intervals: list[tuple[float, float]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            if "start_s" in item or "end_s" in item:
                start, end = item.get("start_s"), item.get("end_s")
            else:
                start, end = item.get("start"), item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError(f"{channel} VAD interval {index} must be a pair or object")
        intervals.append(
            (
                _number(start, f"{channel} VAD interval {index} start"),
                _number(end, f"{channel} VAD interval {index} end"),
            )
        )
    intervals.sort()
    previous_end = -math.inf
    for start, end in intervals:
        if start < 0 or end > duration_s:
            raise ValueError(f"{channel} VAD timestamps must lie within duration_s")
        if start >= end:
            raise ValueError(f"{channel} VAD interval start must precede end")
        if start < previous_end:
            raise ValueError(f"within-channel {channel} VAD intervals must not overlap")
        previous_end = end
    return intervals


def _sessions(payload: bytes) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, obj in enumerate(_records(payload), 1):
        session_id = _required_string(obj, "session_id", line_number)
        if session_id in seen:
            raise ValueError(f"duplicate session_id {session_id!r}")
        seen.add(session_id)
        lineage_id = _required_string(obj, "lineage_id", line_number)
        source = _required_string(obj, "source", line_number)
        if "duration_s" not in obj:
            raise ValueError(f"JSON row {line_number} requires an explicit duration_s")
        duration_s = _number(obj["duration_s"], "duration_s")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        challenge = obj.get("challenge", False)
        if not isinstance(challenge, bool):
            raise ValueError(f"JSON row {line_number} challenge must be a boolean")
        sessions.append(
            {
                "session_id": session_id,
                "lineage_id": lineage_id,
                "source": source,
                "duration_s": duration_s,
                "challenge": challenge,
                "user": _intervals(
                    obj,
                    ("user_vad", "user", "left", "LEFT"),
                    duration_s=duration_s,
                    channel="user",
                ),
                "assistant": _intervals(
                    obj,
                    ("assistant_vad", "assistant", "agent_vad", "right", "RIGHT"),
                    duration_s=duration_s,
                    channel="assistant",
                ),
            }
        )
    return sessions


def assign_lineage_splits(
    lineage_ids: Iterable[str], seed: int, challenge_lineages: Iterable[str] = ()
) -> dict[str, str]:
    """Assign each lineage to challenge or one deterministic 70/15/15 split."""
    challenges = set(challenge_lineages)
    splits: dict[str, str] = {}
    for lineage_id in sorted(set(lineage_ids)):
        if lineage_id in challenges:
            splits[lineage_id] = "challenge"
            continue
        value = int.from_bytes(
            hashlib.sha256(f"{seed}:{lineage_id}".encode()).digest()[:8], "big"
        ) / 2**64
        splits[lineage_id] = (
            "train" if value < 0.70 else "development" if value < 0.85 else "test"
        )
    return splits


def elapsed_stratum(elapsed_s: float, boundaries_s: Sequence[float], labels: Sequence[str]) -> str:
    for boundary, label in zip(boundaries_s, labels):
        if elapsed_s < boundary:
            return label
    return labels[-1]


def _locked_pilot_splits(
    allowed_splits: Sequence[str], excluded_splits: Sequence[str]
) -> tuple[set[str], set[str]]:
    if isinstance(allowed_splits, (str, bytes)) or isinstance(
        excluded_splits, (str, bytes)
    ):
        raise ValueError("pilot split settings must be lists")
    allowed, excluded = list(allowed_splits), list(excluded_splits)
    if len(allowed) != 2 or set(allowed) != {"train", "development"}:
        raise ValueError("pilot_splits must be exactly train and development")
    if not {"test", "challenge"} <= set(excluded):
        raise ValueError("pilot_excluded_sets must include test and challenge")
    if set(allowed) & set(excluded):
        raise ValueError("pilot allowed and excluded splits must be disjoint")
    return set(allowed), set(excluded)


def select_pilot_items(
    rows: Sequence[SilenceCheckpoint],
    lineage_splits: Mapping[str, str],
    *,
    target_episodes: int,
    min_lineages: int,
    allowed_splits: Sequence[str],
    excluded_splits: Sequence[str],
    boundaries_s: Sequence[float],
    stratum_labels: Sequence[str],
    seed: int,
) -> list[SilenceCheckpoint]:
    """Select one JOINT checkpoint per episode, balancing that row's source/stratum."""
    if target_episodes < 1 or min_lineages < 1 or min_lineages > target_episodes:
        raise ValueError("pilot target and minimum lineages are inconsistent")
    if len(stratum_labels) != len(boundaries_s) + 1:
        raise ValueError("elapsed-stratum labels must have one more item than boundaries")
    allowed, excluded = _locked_pilot_splits(allowed_splits, excluded_splits)

    episode_metadata: dict[str, tuple[str, str, str]] = {}
    candidates: list[SilenceCheckpoint] = []
    for row in rows:
        lineage = row.lineage_id or row.session_id
        metadata = (row.session_id, lineage, row.source)
        previous = episode_metadata.setdefault(row.episode_id, metadata)
        if previous != metadata:
            raise ValueError(f"episode {row.episode_id!r} has inconsistent session, lineage, or source")
        split = lineage_splits.get(lineage)
        if row.context is SilenceContext.JOINT and split in allowed and split not in excluded:
            candidates.append(row)

    candidate_episodes = {row.episode_id for row in candidates}
    if len(candidate_episodes) < target_episodes:
        raise ValueError(
            f"pilot requires {target_episodes} eligible episodes; found {len(candidate_episodes)}"
        )
    candidate_lineages = {row.lineage_id or row.session_id for row in candidates}
    if len(candidate_lineages) < min_lineages:
        raise ValueError(
            f"pilot requires {min_lineages} eligible lineages; found {len(candidate_lineages)}"
        )

    selected: list[SilenceCheckpoint] = []
    selected_episodes: set[str] = set()
    selected_lineages: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str]] = Counter()
    while len(selected) < target_episodes:
        pool = [row for row in candidates if row.episode_id not in selected_episodes]
        if len(selected_lineages) < min_lineages:
            unseen = [
                row
                for row in pool
                if (row.lineage_id or row.session_id) not in selected_lineages
            ]
            if unseen:
                pool = unseen

        def rank(row: SilenceCheckpoint) -> tuple[int, int, int, int, bytes, str, float]:
            lineage = row.lineage_id or row.session_id
            stratum = elapsed_stratum(row.elapsed_s, boundaries_s, stratum_labels)
            return (
                cell_counts[row.source, stratum],
                source_counts[row.source],
                stratum_counts[stratum],
                selected_lineages[lineage],
                hashlib.sha256(
                    f"{seed}:pilot:{row.episode_id}:{row.elapsed_s!r}".encode()
                ).digest(),
                row.episode_id,
                row.elapsed_s,
            )

        row = min(pool, key=rank)
        selected.append(row)
        selected_episodes.add(row.episode_id)
        lineage = row.lineage_id or row.session_id
        stratum = elapsed_stratum(row.elapsed_s, boundaries_s, stratum_labels)
        selected_lineages[lineage] += 1
        source_counts[row.source] += 1
        stratum_counts[stratum] += 1
        cell_counts[row.source, stratum] += 1

    if len(selected_lineages) < min_lineages:
        raise ValueError("pilot lineage minimum was not met")
    if any(
        lineage_splits[row.lineage_id or row.session_id] not in allowed
        or lineage_splits[row.lineage_id or row.session_id] in excluded
        for row in selected
    ):
        raise ValueError("a protected split entered the pilot")
    return selected


def _jsonl(rows: Sequence[SilenceCheckpoint], splits: Mapping[str, str]) -> bytes:
    lines = []
    for row in rows:
        lineage_id = row.lineage_id or row.session_id
        lines.append(
            json.dumps(
                {
                    **row.__dict__,
                    "context": row.context.value,
                    "quality": row.quality.value,
                    "split": splits[lineage_id],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return ("\n".join(lines) + ("\n" if lines else "")).encode()


def companion_paths(output: Path) -> tuple[Path, Path, Path]:
    pilot = output.with_name(f"{output.stem}.pilot{output.suffix}")
    splits = output.with_name(f"{output.stem}.splits.json")
    lock = output.with_suffix(output.suffix + ".lock.json")
    return pilot, splits, lock


def _publish_artifacts(artifacts: Sequence[tuple[Path, bytes]]) -> None:
    directory = artifacts[0][0].parent
    if any(path.parent != directory for path, _ in artifacts):
        raise ValueError("all manifest artifacts must share one directory")
    directory.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(directory_fd)
    temporary: list[tuple[str, str]] = []
    published: list[str] = []
    try:
        for path, payload in artifacts:
            name = f".{path.name}.{secrets.token_hex(8)}.tmp"
            descriptor = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory_fd,
            )
            temporary.append((name, path.name))
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary_name, target_name in temporary:
            os.link(
                temporary_name, target_name,
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            published.append(target_name)
            os.unlink(temporary_name, dir_fd=directory_fd)
        current = directory.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError("manifest output directory was replaced")
        os.fsync(directory_fd)
    except Exception as error:
        for name in published:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        for name, _ in temporary:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.fsync(directory_fd)
        if isinstance(error, FileExistsError):
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {error.filename}"
            ) from error
        raise
    finally:
        os.close(directory_fd)


def build_manifests(
    vad: Path,
    output: Path,
    config: str | Path = "silence.yaml",
    *,
    hmac_key: Path,
) -> tuple[Path, Path, Path, Path]:
    """Build and authenticate all locked artifacts without replacing existing ones."""
    vad, output = Path(vad), Path(output)
    pilot_path, splits_path, lock_path = companion_paths(output)
    paths = (output, pilot_path, splits_path, lock_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifact(s): {', '.join(existing)}")

    config_path, config_payload, cfg = load_protocol_v1(config)
    protocol_path = repo_root() / "docs" / _PROTOCOL
    vad_payload = vad.read_bytes()
    protocol_payload = protocol_path.read_bytes()
    key = read_hmac_key(hmac_key)
    silence, annotation = cfg["silence"], cfg["annotation"]
    checkpoints = silence["checkpoints_s"]
    boundaries = silence["duration_strata_s"]
    labels = silence["duration_strata_labels"]
    max_per_lineage = silence["max_episodes_per_lineage"]
    split_seed = silence["split_seed"]
    target = annotation["pilot_target_episodes"]
    minimum = annotation["pilot_min_lineages"]

    sessions = _sessions(vad_payload)
    lineage_bindings: dict[str, tuple[str, str]] = {}
    for session in sessions:
        binding = (session["source"], "challenge" if session["challenge"] else "ordinary")
        previous = lineage_bindings.setdefault(session["lineage_id"], binding)
        if previous != binding:
            raise ValueError("a lineage cannot span sources or challenge status")
    splits = assign_lineage_splits(
        (session["lineage_id"] for session in sessions),
        split_seed,
        (
            session["lineage_id"]
            for session in sessions
            if session["challenge"]
        ),
    )
    rows: list[SilenceCheckpoint] = []
    for session in sessions:
        rows.extend(
            extract_silence_checkpoints(
                session["session_id"],
                session["duration_s"],
                session["user"],
                session["assistant"],
                checkpoints_s=checkpoints,
                source=session["source"],
                lineage_id=session["lineage_id"],
            )
        )
    full = sample_episodes(rows, max_per_lineage=max_per_lineage, seed=split_seed)
    pilot = select_pilot_items(
        full,
        splits,
        target_episodes=target,
        min_lineages=minimum,
        allowed_splits=annotation["pilot_splits"],
        excluded_splits=annotation["pilot_excluded_sets"],
        boundaries_s=boundaries,
        stratum_labels=labels,
        seed=split_seed,
    )

    full_payload = _jsonl(full, splits)
    pilot_payload = _jsonl(pilot, splits)
    splits_payload = (
        json.dumps(
            {
                "schema": cfg["schema"],
                "protocol_version": cfg["protocol_version"],
                "split_seed": split_seed,
                "lineages": splits,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    selected_ids = {row.episode_id for row in pilot}
    receipt = {
        "schema": cfg["schema"],
        "protocol_version": cfg["protocol_version"],
        "label_schema": cfg["label_schema"],
        "sha256": {
            "input_vad": _sha256(vad_payload),
            "config": _sha256(config_payload),
            "protocol_doc": _sha256(protocol_payload),
            "full_manifest": _sha256(full_payload),
            "pilot_manifest": _sha256(pilot_payload),
            "splits": _sha256(splits_payload),
        },
        "counts": {
            "input_sessions": len(sessions),
            "input_lineages": len(splits),
            "split_lineages": dict(sorted(Counter(splits.values()).items())),
            "full_episodes": len({row.episode_id for row in full}),
            "full_checkpoints": len(full),
            "pilot_episodes": len(selected_ids),
            "pilot_checkpoints": len(pilot),
            "pilot_lineages": len(
                {row.lineage_id or row.session_id for row in pilot}
            ),
            "pilot_by_source": dict(
                sorted(Counter(row.source for row in pilot).items())
            ),
            "pilot_by_elapsed_stratum": dict(
                sorted(
                    Counter(
                        elapsed_stratum(row.elapsed_s, boundaries, labels)
                        for row in pilot
                    ).items()
                )
            ),
        },
    }
    lock_payload = receipt_bytes(sign_receipt(receipt, key))

    _publish_artifacts(
        (
            (output, full_payload),
            (pilot_path, pilot_payload),
            (splits_path, splits_payload),
            (lock_path, lock_payload),
        )
    )
    return output, pilot_path, splits_path, lock_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="silence.yaml", help="config path or name")
    parser.add_argument("--hmac-key", type=Path, required=True)
    args = parser.parse_args()
    build_manifests(args.vad, args.output, args.config, hmac_key=args.hmac_key)


if __name__ == "__main__":
    main()
