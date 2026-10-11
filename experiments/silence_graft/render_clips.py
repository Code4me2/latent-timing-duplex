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
import tempfile
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from latent_timing_duplex.config import find_config, repo_root

try:
    from experiments.silence_graft.receipts import (
        read_hmac_key,
        receipt_bytes,
        require_digest,
        sha256_bytes,
        sha256_file,
        sign_receipt,
    )
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from receipts import read_hmac_key, receipt_bytes, require_digest, sha256_bytes, sha256_file, sign_receipt

_PROTOCOL = "DATA_LABELING_PROTOCOL.md"
_ALLOWED_SPLITS = {"train", "development"}
_LOCK_IDENTIFIERS = {
    "schema": "semantic-silence-config/v1",
    "protocol_version": "semantic-silence-pilot/v1",
    "label_schema": "semantic-silence-labels/v1",
}
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
    path = Path(value)
    return path.resolve() if path.is_file() else find_config(str(value)).resolve()


def _validate_config(payload: bytes, path: Path) -> dict[str, Any]:
    config = yaml.safe_load(payload)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    for key, value in _LOCK_IDENTIFIERS.items():
        if config.get(key) != value:
            raise ValueError(f"config {key} must be exactly {value!r}")
    if "pilot_media" not in config or not isinstance(config.get("silence"), dict):
        raise ValueError("config requires pilot_media and silence mappings")
    expected = {
        "audio_only": True,
        "prefix_s": 20.0,
        "reveal_s": 8.0,
        "reveal_channel": "user",
        "reveal_clip_at_session_end": True,
        "boundary_rounding": "nearest_sample",
        "gain_scope": "conversation_or_session",
        "gain_db": 0.0,
        "transcript": False,
    }
    media = config["pilot_media"]
    if not isinstance(media, dict) or any(media.get(key) != value for key, value in expected.items()):
        raise ValueError("config pilot_media does not match the locked pilot protocol")
    checkpoints = config["silence"].get("checkpoints_s")
    if not isinstance(checkpoints, list) or not checkpoints or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        for value in checkpoints
    ):
        raise ValueError("config requires finite positive silence checkpoints")
    return config


def _validate_lock(
    lock: Mapping[str, Any],
    *,
    manifest_payload: bytes,
    config_payload: bytes,
    protocol_payload: bytes,
    config: Mapping[str, Any],
) -> None:
    for field in ("schema", "protocol_version", "label_schema"):
        if field not in lock or lock[field] != config[field]:
            raise ValueError(f"lock receipt {field} does not exactly match config")
    hashes = lock.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("lock receipt requires sha256")
    expected = {
        "pilot_manifest": _sha256(manifest_payload),
        "config": _sha256(config_payload),
        "protocol_doc": _sha256(protocol_payload),
    }
    for field, digest in expected.items():
        require_digest(hashes.get(field), f"lock receipt sha256.{field}")
        if not hmac.compare_digest(hashes[field], digest):
            raise ValueError(f"{field} SHA-256 does not match lock receipt")


def _validate_manifest(
    rows: list[dict[str, Any]], *, checkpoints_s: list[float], prefix_s: float
) -> tuple[list[dict[str, Any]], set[str]]:
    episodes: set[str] = set()
    sessions: set[str] = set()
    for number, row in enumerate(rows, 1):
        where = f"pilot manifest row {number}"
        session = _string(row, "session_id", where)
        episode = _string(row, "episode_id", where)
        _string(row, "source", where)
        _string(row, "lineage_id", where)
        if episode in episodes:
            raise ValueError(f"duplicate pilot episode_id {episode!r}")
        episodes.add(episode)
        sessions.add(session)
        if row.get("context") != "joint_silence":
            raise ValueError(f"{where} is not joint_silence")
        if row.get("split") not in _ALLOWED_SPLITS:
            raise ValueError(f"{where} is not train/development")
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


def _reserve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}") from error
    try:
        return Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    except Exception:
        path.rmdir()
        raise


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    return info.st_dev, info.st_ino


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        if _directory_identity(path) == identity:
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _publish_staging(staging: Path, target: Path) -> None:
    for child in staging.iterdir():
        os.replace(child, target / child.name)
    _write(target / _READY, b"ready\n")
    staging.rmdir()
    descriptor = os.open(target, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    _validate_lock(
        _json(lock_payload, "lock receipt"),
        manifest_payload=manifest_payload,
        config_payload=config_payload,
        protocol_payload=protocol_payload,
        config=cfg,
    )
    manifest, sessions = _validate_manifest(
        _jsonl(manifest_payload, "pilot manifest"),
        checkpoints_s=[float(value) for value in cfg["silence"]["checkpoints_s"]],
        prefix_s=float(cfg["pilot_media"]["prefix_s"]),
    )
    media = _media_rows(media_payload, sessions)

    initial_hashes: dict[Path, str] = {}
    for source in media.values():
        path, expected = source["wav_path"], source["wav_sha256"]
        actual = initial_hashes.setdefault(path, sha256_file(path))
        if not hmac.compare_digest(actual, expected):
            raise ValueError(f"source WAV SHA-256 mismatch: {path}")

    prefix_staging: Path | None = None
    coordinator_staging: Path | None = None
    prefix_identity: tuple[int, int] | None = None
    coordinator_identity: tuple[int, int] | None = None
    try:
        prefix_staging = _reserve(prefix_output_dir)
        prefix_identity = _directory_identity(prefix_output_dir)
        coordinator_staging = _reserve(coordinator_output_dir)
        coordinator_identity = _directory_identity(coordinator_output_dir)
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
            wav_path = source["wav_path"]
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
                "model_cutoff_s": cutoff_s,
                "session_id": session,
                "source": row["source"],
                "split": row["split"],
                "user_channel": user,
                "wav_path": str(wav_path),
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

        for path, before in initial_hashes.items():
            if not hmac.compare_digest(sha256_file(path), before):
                raise ValueError(f"source WAV changed during rendering: {path}")

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
        _publish_staging(prefix_staging, prefix_output_dir)
        prefix_staging = None
        _publish_staging(coordinator_staging, coordinator_output_dir)
        coordinator_staging = None
    except Exception:
        if prefix_staging is not None:
            shutil.rmtree(prefix_staging, ignore_errors=True)
        if coordinator_staging is not None:
            shutil.rmtree(coordinator_staging, ignore_errors=True)
        if prefix_identity is not None:
            _remove_owned_directory(prefix_output_dir, prefix_identity)
        if coordinator_identity is not None:
            _remove_owned_directory(coordinator_output_dir, coordinator_identity)
        raise
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
