#!/usr/bin/env python3
"""Render causal pilot prefixes separately from coordinator-only reveals."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import shutil
import stat
import tempfile
import wave
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from latent_timing_duplex.config import repo_root

try:
    from experiments.silence_graft.receipts import (
        read_hmac_key,
        receipt_bytes,
        require_digest,
        sha256_bytes,
        sha256_file,
        sign_receipt,
        verify_receipt,
    )
    from experiments.silence_graft.protocol_v1 import config_path, parse_protocol_v1
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from receipts import read_hmac_key, receipt_bytes, require_digest, sha256_bytes, sha256_file, sign_receipt, verify_receipt
    from protocol_v1 import config_path, parse_protocol_v1

_PROTOCOL = "DATA_LABELING_PROTOCOL.md"
_ALLOWED_SPLITS = {"train", "development"}
_READY = "READY"


def _sha256(payload: bytes) -> str:
    return sha256_bytes(payload)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _jsonl(payload: bytes, name: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_object)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid {name} line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{name} line {line_number} must be a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    return rows


def _string(row: Mapping[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} requires non-empty string {key}")
    return value


def _finite(row: Mapping[str, Any], key: str, where: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} requires numeric {key}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} {key} must be finite")
    return result


def _config_path(value: str | Path) -> Path:
    return config_path(value)


def _validate_config(payload: bytes, path: Path) -> dict[str, Any]:
    return parse_protocol_v1(payload, path)


def _validate_lock(
    lock: Mapping[str, Any],
    *,
    manifest_payload: bytes,
    manifest_rows: list[dict[str, Any]],
    config_payload: bytes,
    protocol_payload: bytes,
    config: Mapping[str, Any],
    key: bytes,
) -> None:
    verify_receipt(lock, key, "manifest lock receipt")
    for field in ("schema", "protocol_version", "label_schema"):
        if field not in lock or lock[field] != config[field]:
            raise ValueError(f"lock receipt {field} does not exactly match config")
    hashes = lock.get("sha256")
    required_hashes = {"input_vad", "config", "protocol_doc", "full_manifest", "pilot_manifest", "splits"}
    if not isinstance(hashes, dict) or not required_hashes <= hashes.keys():
        raise ValueError("lock receipt requires all manifest SHA-256 commitments")
    for field in required_hashes:
        require_digest(hashes.get(field), f"lock receipt sha256.{field}")
    expected = {
        "pilot_manifest": _sha256(manifest_payload),
        "config": _sha256(config_payload),
        "protocol_doc": _sha256(protocol_payload),
    }
    for field, digest in expected.items():
        if not hmac.compare_digest(hashes[field], digest):
            raise ValueError(f"{field} SHA-256 does not match lock receipt")
    counts = lock.get("counts")
    required_counts = {
        "input_sessions", "input_lineages", "split_lineages", "full_episodes",
        "full_checkpoints", "pilot_episodes", "pilot_checkpoints", "pilot_lineages",
        "pilot_by_source", "pilot_by_elapsed_stratum",
    }
    if not isinstance(counts, dict) or not required_counts <= counts.keys():
        raise ValueError("lock receipt requires all manifest count commitments")
    scalar_counts = required_counts - {"split_lineages", "pilot_by_source", "pilot_by_elapsed_stratum"}
    if any(isinstance(counts[name], bool) or not isinstance(counts[name], int) or counts[name] < 0 for name in scalar_counts):
        raise ValueError("lock receipt count commitments must be non-negative integers")
    for name in ("split_lineages", "pilot_by_source", "pilot_by_elapsed_stratum"):
        if not isinstance(counts[name], dict) or any(
            not isinstance(label, str) or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for label, value in counts[name].items()
        ):
            raise ValueError(f"lock receipt counts.{name} is invalid")
    if (
        sum(counts["split_lineages"].values()) != counts["input_lineages"]
        or counts["full_episodes"] > counts["full_checkpoints"]
        or counts["pilot_episodes"] > counts["pilot_checkpoints"]
        or sum(counts["pilot_by_source"].values()) != counts["pilot_checkpoints"]
        or sum(counts["pilot_by_elapsed_stratum"].values()) != counts["pilot_checkpoints"]
    ):
        raise ValueError("lock receipt count commitments are inconsistent")
    boundaries = config["silence"]["duration_strata_s"]
    labels = config["silence"]["duration_strata_labels"]
    recomputed = {
        "pilot_episodes": len({row["episode_id"] for row in manifest_rows}),
        "pilot_checkpoints": len(manifest_rows),
        "pilot_lineages": len({row["lineage_id"] for row in manifest_rows}),
        "pilot_by_source": dict(sorted(Counter(row["source"] for row in manifest_rows).items())),
        "pilot_by_elapsed_stratum": dict(sorted(Counter(
            next((label for boundary, label in zip(boundaries, labels) if row["elapsed_s"] < boundary), labels[-1])
            for row in manifest_rows
        ).items())),
    }
    if any(counts[name] != value for name, value in recomputed.items()):
        raise ValueError("pilot manifest counts do not match lock receipt")


def _validate_manifest(
    rows: list[dict[str, Any]],
    *,
    checkpoints_s: list[float],
    prefix_s: float,
    item_range: list[int],
    min_lineages: int,
    max_per_lineage: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    episodes: set[str] = set()
    sessions: set[str] = set()
    session_bindings: dict[str, tuple[str, str, str]] = {}
    lineage_bindings: dict[str, tuple[str, str]] = {}
    lineage_counts: Counter[str] = Counter()
    for number, row in enumerate(rows, 1):
        where = f"pilot manifest row {number}"
        session = _string(row, "session_id", where)
        episode = _string(row, "episode_id", where)
        source = _string(row, "source", where)
        lineage = _string(row, "lineage_id", where)
        if episode in episodes:
            raise ValueError(f"duplicate pilot episode_id {episode!r}")
        episodes.add(episode)
        sessions.add(session)
        if row.get("context") != "joint_silence":
            raise ValueError(f"{where} is not joint_silence")
        split = row.get("split")
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"{where} is not train/development")
        session_binding = (source, lineage, split)
        if session_bindings.setdefault(session, session_binding) != session_binding:
            raise ValueError(f"session {session!r} has inconsistent source, lineage, or split")
        lineage_binding = (source, split)
        if lineage_bindings.setdefault(lineage, lineage_binding) != lineage_binding:
            raise ValueError(f"lineage {lineage!r} has inconsistent source or split")
        lineage_counts[lineage] += 1
        onset = _finite(row, "onset_s", where)
        end = _finite(row, "silence_end_s", where)
        cutoff = _finite(row, "model_cutoff_s", where)
        elapsed = _finite(row, "elapsed_s", where)
        if elapsed > prefix_s + 1e-9:
            raise ValueError(f"{where} elapsed silence exceeds the prefix duration")
        if not any(abs(elapsed - checkpoint) <= 1e-9 for checkpoint in checkpoints_s):
            raise ValueError(f"{where} elapsed_s is not a configured checkpoint")
        if onset < 0 or not onset <= cutoff <= end or end <= onset:
            raise ValueError(f"{where} has invalid silence boundaries")
        if abs((cutoff - onset) - elapsed) > 1e-6:
            raise ValueError(f"{where} has a timing mismatch")
        if "right_censored" in row and not isinstance(row["right_censored"], bool):
            raise ValueError(f"{where} right_censored must be boolean")
    if not item_range[0] <= len(rows) <= item_range[1]:
        raise ValueError(
            f"production pilot requires {item_range[0]}–{item_range[1]} items; found {len(rows)}"
        )
    if len(lineage_counts) < min_lineages:
        raise ValueError(
            f"production pilot requires at least {min_lineages} lineages; found {len(lineage_counts)}"
        )
    if any(count > max_per_lineage for count in lineage_counts.values()):
        raise ValueError(f"production pilot exceeds {max_per_lineage} items per lineage")
    return rows, sessions


def _media_rows(payload: bytes, pilot_sessions: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(_jsonl(payload, "media index"), 1):
        where = f"media index row {number}"
        session = _string(row, "session_id", where)
        if session in result:
            raise ValueError(f"duplicate media index session_id {session!r}")
        wav_path = Path(_string(row, "wav_path", where))
        if not wav_path.is_absolute():
            raise ValueError(f"{where} wav_path must be an absolute local path")
        source_digest = require_digest(row.get("wav_sha256"), f"{where} wav_sha256")
        channels: list[int] = []
        for key in ("user_channel", "assistant_channel"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{where} {key} must be an integer")
            channels.append(value)
        if channels[0] == channels[1]:
            raise ValueError(f"{where} user and assistant channels must be distinct")
        result[session] = {**row, "wav_path": wav_path, "wav_sha256": source_digest}
    missing, extra = pilot_sessions - result.keys(), result.keys() - pilot_sessions
    if missing:
        raise ValueError(f"media index is missing pilot session(s): {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"media index has non-pilot session(s): {', '.join(sorted(extra))}")
    return result


def _nearest_frame(seconds: float, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate + 0.5))


def _item_id(key: bytes, episode_id: str, cutoff_s: float) -> str:
    identity = f"{episode_id}\0{cutoff_s.hex()}".encode("utf-8")
    return hmac.new(key, identity, hashlib.sha256).hexdigest()


def _wav_info(path: Path) -> tuple[int, int, int, int]:
    try:
        with wave.open(str(path), "rb") as source:
            channels, width = source.getnchannels(), source.getsampwidth()
            rate, frames, compression = source.getframerate(), source.getnframes(), source.getcomptype()
            if compression != "NONE" or channels < 2 or width not in (1, 2, 3, 4) or rate <= 0:
                raise ValueError(f"unsupported PCM WAV {path}")
            remaining = frames
            while remaining:
                count = min(remaining, 65_536)
                payload = source.readframes(count)
                if len(payload) != count * channels * width:
                    raise ValueError(f"truncated PCM WAV {path}")
                remaining -= count
    except (FileNotFoundError, OSError, EOFError, wave.Error) as error:
        raise ValueError(f"unsupported or unreadable PCM WAV {path}: {error}") from error
    return channels, width, rate, frames


def _read_channels(path: Path, *, source_channels: int, width: int, start: int, end: int, selected: tuple[int, ...]) -> bytes:
    with wave.open(str(path), "rb") as source:
        source.setpos(start)
        payload = source.readframes(end - start)
    frame_width = source_channels * width
    if len(payload) != (end - start) * frame_width:
        raise ValueError(f"truncated PCM WAV {path}")
    return b"".join(
        payload[offset + channel * width : offset + (channel + 1) * width]
        for frame in range(end - start)
        for offset in (frame * frame_width,)
        for channel in selected
    )


def _write_wav(path: Path, channels: int, width: int, rate: int, payload: bytes) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(payload)
    path.chmod(0o600)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _lines(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n").encode()


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def _reserve(path: Path) -> tuple[Path, int, tuple[int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}") from error
    descriptor = -1
    identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        identity = (info.st_dev, info.st_ino)
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
        return staging, descriptor, identity
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if identity is not None:
            _remove_owned_directory(path, identity)
        raise


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    return info.st_dev, info.st_ino


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        if _directory_identity(path) == identity:
            path.rmdir()  # Never recursively remove a target an attacker may have replaced.
    except (FileNotFoundError, OSError):
        pass


def _publish_staging(
    staging: Path, target: Path, target_fd: int, identity: tuple[int, int]
) -> None:
    staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for child in staging.iterdir():
            os.rename(child.name, child.name, src_dir_fd=staging_fd, dst_dir_fd=target_fd)
        marker = os.open(
            _READY, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=target_fd,
        )
        try:
            os.write(marker, b"ready\n")
            os.fsync(marker)
        finally:
            os.close(marker)
        os.fsync(target_fd)
        if _directory_identity(target) != identity:
            raise RuntimeError("reserved publication target was replaced")
    finally:
        os.close(staging_fd)
    staging.rmdir()


def _snapshot_wavs(media: Mapping[str, Mapping[str, Any]], root: Path) -> dict[Path, Path]:
    snapshots: dict[Path, Path] = {}
    digests: dict[Path, str] = {}
    identities: dict[tuple[int, int], Path] = {}
    for row in media.values():
        source = row["wav_path"]
        if source not in snapshots:
            try:
                source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError as error:
                raise ValueError("source WAV must be a readable regular non-symlink file") from error
            target_fd = -1
            try:
                info = os.fstat(source_fd)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("source WAV must be a regular file")
                identity = (info.st_dev, info.st_ino)
                if identity in identities:
                    original = identities[identity]
                    snapshots[source], digests[source] = snapshots[original], digests[original]
                else:
                    target_fd, target_name = tempfile.mkstemp(prefix="source-", suffix=".wav", dir=root)
                    os.fchmod(target_fd, 0o600)
                    digest = hashlib.sha256()
                    while chunk := os.read(source_fd, 1024 * 1024):
                        digest.update(chunk)
                        view = memoryview(chunk)
                        while view:
                            view = view[os.write(target_fd, view):]
                    os.fsync(target_fd)
                    snapshots[source], digests[source] = Path(target_name), digest.hexdigest()
                    identities[identity] = source
            finally:
                os.close(source_fd)
                if target_fd >= 0:
                    os.close(target_fd)
        if not hmac.compare_digest(digests[source], row["wav_sha256"]):
            raise ValueError("source WAV SHA-256 mismatch")
    return snapshots


def _remove_ready(target_fd: int) -> None:
    try:
        os.unlink(_READY, dir_fd=target_fd)
        os.fsync(target_fd)
    except FileNotFoundError:
        pass


def _require_ready(root: Path, where: str) -> None:
    marker = root / _READY
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"{where} is not ready")


def render_clips(
    pilot_manifest: Path,
    lock_receipt: Path,
    media_index: Path,
    prefix_output_dir: Path,
    coordinator_output_dir: Path,
    hmac_key: Path,
    config: str | Path = "silence.yaml",
) -> tuple[Path, Path]:
    """Render into separately reserved prefix and coordinator packages."""
    pilot_manifest, lock_receipt, media_index = map(Path, (pilot_manifest, lock_receipt, media_index))
    prefix_output_dir, coordinator_output_dir = map(Path, (prefix_output_dir, coordinator_output_dir))
    prefix_resolved, coordinator_resolved = prefix_output_dir.resolve(), coordinator_output_dir.resolve()
    if (
        prefix_resolved == coordinator_resolved
        or prefix_resolved in coordinator_resolved.parents
        or coordinator_resolved in prefix_resolved.parents
    ):
        raise ValueError("prefix and coordinator output directories must be separate")

    config_path = _config_path(config)
    protocol_path = repo_root() / "docs" / _PROTOCOL
    manifest_payload, lock_payload, media_payload = (
        pilot_manifest.read_bytes(), lock_receipt.read_bytes(), media_index.read_bytes()
    )
    config_payload, protocol_payload = config_path.read_bytes(), protocol_path.read_bytes()
    key = read_hmac_key(hmac_key)
    cfg = _validate_config(config_payload, config_path)
    manifest, sessions = _validate_manifest(
        _jsonl(manifest_payload, "pilot manifest"),
        checkpoints_s=[float(value) for value in cfg["silence"]["checkpoints_s"]],
        prefix_s=float(cfg["pilot_media"]["prefix_s"]),
        item_range=cfg["annotation"]["pilot_review_range_episodes"],
        min_lineages=cfg["annotation"]["pilot_min_lineages"],
        max_per_lineage=cfg["silence"]["max_episodes_per_lineage"],
    )
    _validate_lock(
        _json(lock_payload, "lock receipt"),
        manifest_payload=manifest_payload,
        manifest_rows=manifest,
        config_payload=config_payload,
        protocol_payload=protocol_payload,
        config=cfg,
        key=key,
    )
    media = _media_rows(media_payload, sessions)

    prefix_staging: Path | None = None
    coordinator_staging: Path | None = None
    prefix_fd = coordinator_fd = -1
    prefix_identity: tuple[int, int] | None = None
    coordinator_identity: tuple[int, int] | None = None
    snapshot_directory = Path(tempfile.mkdtemp(prefix="silence-sources-"))
    snapshot_directory.chmod(0o700)
    try:
        snapshots = _snapshot_wavs(media, snapshot_directory)
        prefix_staging, prefix_fd, prefix_identity = _reserve(prefix_output_dir)
        coordinator_staging, coordinator_fd, coordinator_identity = _reserve(coordinator_output_dir)
        (prefix_staging / "prefix").mkdir()
        (coordinator_staging / "reveal_user").mkdir()
        wavs: dict[Path, tuple[int, int, int, int]] = {}
        index_rows: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []
        render_rows: list[dict[str, Any]] = []
        seen_items: set[str] = set()
        media_cfg = cfg["pilot_media"]

        for row in manifest:
            session, source = row["session_id"], media[row["session_id"]]
            source_path = source["wav_path"]
            wav_path = snapshots[source_path]
            if wav_path not in wavs:
                wavs[wav_path] = _wav_info(wav_path)
            channels, width, rate, frame_count = wavs[wav_path]
            user, assistant = source["user_channel"], source["assistant_channel"]
            if user < 0 or assistant < 0 or user >= channels or assistant >= channels:
                raise ValueError(f"invalid channel mapping for session {session!r}")

            cutoff_s = float(row["model_cutoff_s"])
            duration_s = frame_count / rate
            if cutoff_s > duration_s:
                raise ValueError(f"model cutoff exceeds WAV duration for session {session!r}")
            if float(row["silence_end_s"]) > duration_s + 0.5 / rate:
                raise ValueError(f"manifest/WAV timing mismatch for session {session!r}")
            if row.get("right_censored") and abs(float(row["silence_end_s"]) - duration_s) > 0.5 / rate:
                raise ValueError(f"right-censored boundary does not match WAV for session {session!r}")

            cutoff = _nearest_frame(cutoff_s, rate)
            prefix_frames = _nearest_frame(float(media_cfg["prefix_s"]), rate)
            reveal_frames = _nearest_frame(float(media_cfg["reveal_s"]), rate)
            prefix_start, reveal_end = max(0, cutoff - prefix_frames), min(frame_count, cutoff + reveal_frames)
            if not 0 <= prefix_start <= cutoff <= reveal_end <= frame_count:
                raise ValueError(f"invalid rounded clip boundaries for session {session!r}")

            item = _item_id(key, row["episode_id"], cutoff_s)
            if item in seen_items:
                raise ValueError("opaque item ID collision")
            seen_items.add(item)
            prefix_rel, reveal_rel = f"prefix/{item}.wav", f"reveal_user/{item}.wav"
            _write_wav(
                prefix_staging / prefix_rel, 2, width, rate,
                _read_channels(wav_path, source_channels=channels, width=width, start=prefix_start, end=cutoff, selected=(user, assistant)),
            )
            _write_wav(
                coordinator_staging / reveal_rel, 1, width, rate,
                _read_channels(wav_path, source_channels=channels, width=width, start=cutoff, end=reveal_end, selected=(user,)),
            )
            prefix_hash, reveal_hash = sha256_file(prefix_staging / prefix_rel), sha256_file(coordinator_staging / reveal_rel)
            index_rows.append({
                "item_id": item,
                "prefix_clip": prefix_rel,
                "prefix_sha256": prefix_hash,
                "protocol_version": cfg["protocol_version"],
                "reveal_clip": reveal_rel,
                "reveal_sha256": reveal_hash,
            })
            mapping_rows.append({
                "assistant_channel": assistant,
                "episode_id": row["episode_id"],
                "gain_db": 0.0,
                "item_id": item,
                "lineage_id": row["lineage_id"],
                "onset_s": float(row["onset_s"]),
                "elapsed_s": float(row["elapsed_s"]),
                "checkpoint_elapsed_s": float(row["elapsed_s"]),
                "model_cutoff_s": cutoff_s,
                "session_id": session,
                "source": row["source"],
                "split": row["split"],
                "user_channel": user,
                "wav_sha256": source["wav_sha256"],
            })
            for package, kind, relative, digest, start, end, labels in (
                ("prefix", "prefix", prefix_rel, prefix_hash, prefix_start, cutoff, ["user", "assistant"]),
                ("coordinator", "reveal_user", reveal_rel, reveal_hash, cutoff, reveal_end, ["user"]),
            ):
                render_rows.append({
                    "channels": labels, "end_frame": end, "gain_db": 0.0, "item_id": item,
                    "kind": kind, "package": package, "path": relative, "sample_rate": rate,
                    "sample_width_bytes": width, "sha256": digest, "start_frame": start,
                })

        metadata = {
            "coordinator_media_index.jsonl": _lines(index_rows),
            "private_source_mapping.jsonl": _lines(mapping_rows),
            "render_records.jsonl": _lines(render_rows),
        }
        for name, payload in metadata.items():
            _write(coordinator_staging / name, payload)
        prefix_outputs = {
            str(path.relative_to(prefix_staging)): sha256_file(path)
            for path in sorted(prefix_staging.rglob("*")) if path.is_file()
        }
        coordinator_outputs = {
            str(path.relative_to(coordinator_staging)): sha256_file(path)
            for path in sorted(coordinator_staging.rglob("*")) if path.is_file()
        }
        receipt = sign_receipt({
            "schema": "semantic-silence-renderer-receipt/v1",
            "protocol_version": cfg["protocol_version"],
            "sha256": {
                "config": _sha256(config_payload), "lock_receipt": _sha256(lock_payload),
                "media_index": _sha256(media_payload), "pilot_manifest": _sha256(manifest_payload),
                "protocol_doc": _sha256(protocol_payload),
            },
            "counts": {"clips": len(render_rows), "items": len(index_rows), "sessions": len(sessions)},
            "prefix_outputs": prefix_outputs,
            "outputs": coordinator_outputs,
            "sources": [
                {
                    "assistant_channel": media[session]["assistant_channel"],
                    "session_id": session,
                    "user_channel": media[session]["user_channel"],
                    "wav_sha256": media[session]["wav_sha256"],
                }
                for session in sorted(media)
            ],
        }, key)
        _write(coordinator_staging / "renderer_receipt.json", receipt_bytes(receipt))
        _publish_staging(prefix_staging, prefix_output_dir, prefix_fd, prefix_identity)
        prefix_staging = None
        _publish_staging(coordinator_staging, coordinator_output_dir, coordinator_fd, coordinator_identity)
        coordinator_staging = None
        os.close(prefix_fd)
        prefix_fd = -1
        os.close(coordinator_fd)
        coordinator_fd = -1
    except Exception:
        if prefix_fd >= 0:
            _remove_ready(prefix_fd)
        if coordinator_fd >= 0:
            _remove_ready(coordinator_fd)
        if prefix_staging is not None:
            shutil.rmtree(prefix_staging, ignore_errors=True)
        if coordinator_staging is not None:
            shutil.rmtree(coordinator_staging, ignore_errors=True)
        if prefix_identity is not None:
            _remove_owned_directory(prefix_output_dir, prefix_identity)
        if coordinator_identity is not None:
            _remove_owned_directory(coordinator_output_dir, coordinator_identity)
        raise
    finally:
        if prefix_fd >= 0:
            os.close(prefix_fd)
        if coordinator_fd >= 0:
            os.close(coordinator_fd)
        shutil.rmtree(snapshot_directory, ignore_errors=True)
    return prefix_output_dir, coordinator_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--lock-receipt", type=Path, required=True)
    parser.add_argument("--media-index", type=Path, required=True)
    parser.add_argument("--prefix-output-dir", type=Path, required=True)
    parser.add_argument("--coordinator-output-dir", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--config", default="silence.yaml", help="config path or name")
    args = parser.parse_args()
    render_clips(
        args.pilot_manifest, args.lock_receipt, args.media_index,
        args.prefix_output_dir, args.coordinator_output_dir, args.hmac_key, args.config,
    )


if __name__ == "__main__":
    main()
