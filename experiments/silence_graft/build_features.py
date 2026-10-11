#!/usr/bin/env python3
"""Build exact-cutoff probe features from authenticated prefix PCM and frozen states."""

from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from latent_timing_duplex.silence import CausalFeatureRecord

try:
    from experiments.silence_graft.build_assignments import _publish_packages, _sha256, _verify_output_map
    from experiments.silence_graft.compile_annotations import _exact_assignment_packages, _load_assignment_map
    from experiments.silence_graft.compile_r2_dataset import _exact_authenticated_package
    from experiments.silence_graft.protocol_v1 import load_protocol_v1
    from experiments.silence_graft.receipts import (
        read_hmac_key,
        receipt_bytes,
        require_digest,
        require_disjoint_output,
        sha256_file,
        sign_receipt,
        verify_receipt,
    )
    from experiments.silence_graft.render_clips import _json, _jsonl, _nearest_frame, _require_ready, _wav_info
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from build_assignments import _publish_packages, _sha256, _verify_output_map
    from compile_annotations import _exact_assignment_packages, _load_assignment_map
    from compile_r2_dataset import _exact_authenticated_package
    from protocol_v1 import load_protocol_v1
    from receipts import read_hmac_key, receipt_bytes, require_digest, require_disjoint_output, sha256_file, sign_receipt, verify_receipt
    from render_clips import _json, _jsonl, _nearest_frame, _require_ready, _wav_info

_WINDOWS_MS = (80, 240, 960)
_VAD_THRESHOLD = 1e-4
_LOG_FLOOR = 1e-12
_EXTERNAL_SCHEMA = "semantic-silence-frozen-feature-receipt/v1"
_OUTPUT_SCHEMA = "semantic-silence-r2-feature-receipt/v1"
_DATASET_SCHEMA = "semantic-silence-r2-dataset-receipt/v2"
_ADJUDICATION_SCHEMA = "semantic-silence-adjudication-receipt/v1"
_RENDERER_SCHEMA = "semantic-silence-renderer-receipt/v1"
_CAUSAL_SCHEMA = "semantic-silence-r2-causal-label/v1"
_OPTIONAL_EXTERNAL = (
    "hidden_previous",
    "predicted_latent",
    "untrained_latent",
    "shuffled_target_latent",
)


def _strings(value: np.ndarray, name: str, n: int) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype.kind not in "US" or any(not item for item in value.astype(str)):
        raise ValueError(f"external archive {name} must be a non-empty string row vector")
    return value.astype(str)


def _vector(value: np.ndarray, name: str, n: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f"external archive {name} must be a finite row vector")
    return value


def _matrix(value: np.ndarray, name: str, n: int) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in "iuf" or value.ndim != 2 or value.shape[0] != n or value.shape[1] == 0:
        raise ValueError(f"external archive {name} must be a non-ragged numeric matrix")
    result = value.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"external archive {name} must be finite")
    return result


def _load_external(payload: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            required = {"item_id", "prefix_sha256", "available_at_s", "model_cutoff_s", "hidden"}
            unknown = set(archive.files) - required - set(_OPTIONAL_EXTERNAL)
            missing = required - set(archive.files)
            if missing or unknown:
                raise ValueError(
                    f"external archive has missing {sorted(missing)} or unknown {sorted(unknown)} arrays"
                )
            raw = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("external archive"):
            raise
        raise ValueError(f"invalid external frozen-feature archive: {error}") from error
    item_raw = np.asarray(raw["item_id"])
    if item_raw.ndim != 1:
        raise ValueError("external archive item_id must be a string row vector")
    n = len(item_raw)
    items = _strings(item_raw, "item_id", n)
    if len(set(items.tolist())) != n:
        raise ValueError("external archive has duplicate item_id rows")
    result = {
        "item_id": items,
        "prefix_sha256": _strings(raw["prefix_sha256"], "prefix_sha256", n),
        "available_at_s": _vector(raw["available_at_s"], "available_at_s", n),
        "model_cutoff_s": _vector(raw["model_cutoff_s"], "model_cutoff_s", n),
        "hidden": _matrix(raw["hidden"], "hidden", n),
    }
    for name in _OPTIONAL_EXTERNAL:
        if name in raw:
            result[name] = _matrix(raw[name], name, n)
    for value in result["prefix_sha256"]:
        require_digest(value, "external archive prefix_sha256")
    if np.any(result["available_at_s"] < 0) or np.any(result["model_cutoff_s"] < 0):
        raise ValueError("external archive timing must be non-negative")
    if "hidden_previous" in result and result["hidden_previous"].shape != result["hidden"].shape:
        raise ValueError("external hidden_previous must have the same shape as hidden")
    return result


def _pcm(path: Path) -> tuple[np.ndarray, int, int]:
    channels, width, rate, frames = _wav_info(path)
    if channels != 2:
        raise ValueError("authenticated prefix WAV must be stereo")
    with wave.open(str(path), "rb") as source:
        payload = source.readframes(frames)
    if width == 1:
        values = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        octets = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        signed = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        values = signed.astype(np.float64) / 8388608.0
    else:
        values = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    return values.reshape(frames, channels), rate, width


def _window_stats(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[float] = []
    deltas: list[float] = []
    truncated: list[float] = []
    n = samples.shape[0]
    active = np.abs(samples) >= _VAD_THRESHOLD
    for window_ms in _WINDOWS_MS:
        width = max(1, _nearest_frame(window_ms / 1000.0, rate))
        bounds = ((max(0, n - width), n), (max(0, n - 2 * width), max(0, n - width)))
        logs: list[np.ndarray] = []
        onset_values: list[np.ndarray] = []
        for start, end in bounds:
            segment = samples[start:end]
            rms = np.sqrt(np.mean(segment * segment, axis=0)) if len(segment) else np.zeros(2)
            logs.append(np.log(np.maximum(rms, _LOG_FLOOR)))
            occupancy = active[start:end].mean(axis=0) if end > start else np.zeros(2)
            previous = active[start - 1] if start else np.zeros(2, dtype=bool)
            transitions = np.concatenate([previous[None, :], active[start:end]], axis=0)
            onsets = np.sum((~transitions[:-1]) & transitions[1:], axis=0)
            onset_values.append(onsets.astype(np.float64))
            features.extend(logs[-1].tolist())
            features.extend(occupancy.tolist())
            features.extend(onsets.astype(np.float64).tolist())
            truncated.append(float(end - start < width))
        delta = logs[0] - logs[1]
        features.extend(delta.tolist())
        deltas.extend(delta.tolist())
        deltas.extend(onset_values[0].tolist())
        deltas.extend(onset_values[1].tolist())
    return np.asarray(features), np.asarray(deltas), np.asarray(truncated)


def _renderer(
    prefix_root: Path,
    renderer_root: Path,
    key: bytes,
    protocol: str,
    config_hash: str,
    assignment_receipt: Mapping[str, Any],
) -> tuple[bytes, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require_ready(prefix_root, "prefix package")
    _require_ready(renderer_root, "renderer coordinator package")
    receipt_path = renderer_root / "renderer_receipt.json"
    payload = receipt_path.read_bytes()
    receipt = _json(payload, "renderer receipt")
    verify_receipt(receipt, key, "renderer receipt")
    if receipt.get("schema") != _RENDERER_SCHEMA or receipt.get("protocol_version") != protocol:
        raise ValueError("renderer receipt schema or protocol mismatch")
    hashes = receipt.get("sha256")
    if not isinstance(hashes, dict) or hashes.get("config") != config_hash:
        raise ValueError("renderer receipt does not bind the unchanged config")
    assignment_hashes = assignment_receipt.get("sha256")
    if not isinstance(assignment_hashes, dict) or assignment_hashes.get("renderer_receipt") != _sha256(payload):
        raise ValueError("assignment receipt is not chained to this renderer receipt")
    prefix_outputs = _verify_output_map(prefix_root, receipt.get("prefix_outputs"), "renderer prefix package")
    outputs = _verify_output_map(renderer_root, receipt.get("outputs"), "renderer coordinator package")
    for root, locked, extras, where in (
        (prefix_root, prefix_outputs, {"READY"}, "prefix package"),
        (renderer_root, outputs, {"renderer_receipt.json", "READY"}, "renderer coordinator package"),
    ):
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError(f"{where} must not contain symlinks")
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        if actual != set(locked) | extras:
            raise ValueError(f"{where} contains unrecorded or missing files")
    records_payload = (renderer_root / "render_records.jsonl").read_bytes()
    index_payload = (renderer_root / "coordinator_media_index.jsonl").read_bytes()
    if outputs.get("render_records.jsonl") != _sha256(records_payload) or outputs.get(
        "coordinator_media_index.jsonl"
    ) != _sha256(index_payload):
        raise ValueError("renderer metadata is not receipt-locked")
    records: dict[str, dict[str, Any]] = {}
    for row in _jsonl(records_payload, "render records"):
        if row.get("package") == "prefix" and row.get("kind") == "prefix":
            item = row.get("item_id")
            if not isinstance(item, str) or item in records:
                raise ValueError("render records have invalid or duplicate prefix item_id")
            records[item] = row
    index: dict[str, dict[str, Any]] = {}
    for row in _jsonl(index_payload, "coordinator media index"):
        item = row.get("item_id")
        if not isinstance(item, str) or item in index:
            raise ValueError("coordinator media index has invalid or duplicate item_id")
        index[item] = row
    if set(records) != set(index):
        raise ValueError("renderer prefix records and media index differ")
    return payload, records, index


def build_features(
    dataset_root: Path,
    adjudication_root: Path,
    prefix_root: Path,
    renderer_root: Path,
    annotator_root: Path,
    assignment_root: Path,
    frozen_features: Path,
    frozen_receipt: Path,
    hmac_key: Path,
    output_dir: Path,
    config: str | Path = "silence.yaml",
) -> Path:
    """Publish one deterministic, authenticated train/development probe archive."""
    require_disjoint_output(
        output_dir,
        [dataset_root, adjudication_root, prefix_root, renderer_root, annotator_root, assignment_root],
    )
    key = read_hmac_key(hmac_key)
    frozen_features, frozen_receipt = Path(frozen_features), Path(frozen_receipt)
    if any(path.is_symlink() or not path.is_file() for path in (frozen_features, frozen_receipt)):
        raise ValueError("external frozen features and receipt must be regular non-symlink files")
    _, config_payload, cfg = load_protocol_v1(config)
    protocol, config_hash = cfg["protocol_version"], _sha256(config_payload)

    assignment_payload, _, assignment_outputs = _exact_assignment_packages(
        Path(annotator_root), Path(assignment_root), key
    )
    assignment_receipt = _json(assignment_payload, "assignment receipt")
    if assignment_receipt.get("protocol_version") != protocol:
        raise ValueError("assignment protocol does not match config")
    assignment_hashes = assignment_receipt.get("sha256")
    if not isinstance(assignment_hashes, dict) or assignment_hashes.get("config") != config_hash:
        raise ValueError("assignment receipt does not bind the unchanged config")
    assignment_rows, reveals, private_payload, _ = _load_assignment_map(
        Path(assignment_root), assignment_outputs, cfg
    )

    adjudication_payload, adjudication, adjudication_outputs = _exact_authenticated_package(
        Path(adjudication_root), receipt_name="adjudication_receipt.json",
        receipt_schema=_ADJUDICATION_SCHEMA, key=key, where="adjudication package",
    )
    dataset_payload, dataset_receipt, dataset_outputs = _exact_authenticated_package(
        Path(dataset_root), receipt_name="dataset_receipt.json", receipt_schema=_DATASET_SCHEMA,
        key=key, where="R2 dataset package",
    )
    if adjudication.get("protocol_version") != protocol or dataset_receipt.get("protocol_version") != protocol:
        raise ValueError("adjudication or dataset protocol mismatch")
    dataset_hashes = dataset_receipt.get("sha256")
    if not isinstance(dataset_hashes, dict) or dataset_hashes.get("adjudication_receipt") != _sha256(adjudication_payload):
        raise ValueError("dataset is not chained to this adjudication receipt")
    if dataset_hashes.get("assignment_receipt") != _sha256(assignment_payload) or dataset_hashes.get(
        "private_assignment_source_map"
    ) != _sha256(private_payload):
        raise ValueError("dataset is not chained to these private assignments")
    if dataset_hashes.get("adjudicated") != adjudication_outputs.get("adjudicated.jsonl"):
        raise ValueError("dataset and adjudication output hashes differ")

    renderer_payload, render_records, media_index = _renderer(
        Path(prefix_root), Path(renderer_root), key, protocol, config_hash, assignment_receipt
    )

    rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "development"):
        relative = f"causal_labels/{split}.jsonl"
        payload = (Path(dataset_root) / relative).read_bytes()
        if dataset_outputs.get(relative) != _sha256(payload):
            raise ValueError("causal labels are not locked by dataset receipt")
        if not payload.strip():
            continue
        for row in _jsonl(payload, f"{split} causal labels"):
            item = row.get("item_id")
            if row.get("schema") != _CAUSAL_SCHEMA or row.get("protocol_version") != protocol:
                raise ValueError("invalid causal-label schema or protocol")
            if not isinstance(item, str) or item in rows or row.get("split") != split:
                raise ValueError("causal labels have duplicate items or inconsistent splits")
            rows[item] = row
    if not rows:
        raise ValueError("dataset contains no train/development causal labels")

    assignments: dict[str, dict[str, Any]] = {}
    prefix_hashes: dict[str, str] = {}
    for row in assignment_rows:
        if row["repeat"]:
            continue
        item = row["item_id"]
        identity = {field: row[field] for field in (
            "item_id", "session_id", "source", "lineage_id", "split", "elapsed_s", "model_cutoff_s"
        )}
        if assignments.setdefault(item, identity) != identity:
            raise ValueError("private assignment map has ambiguous item identity")
        digest = reveals[row["assignment_id"]]["prefix_sha256"]
        if prefix_hashes.setdefault(item, digest) != digest:
            raise ValueError("private assignment map has mixed prefix hashes")
    if not set(rows) <= set(assignments):
        raise ValueError("dataset contains items absent from private assignments")

    frozen_payload = frozen_receipt.read_bytes()
    external_payload = frozen_features.read_bytes()
    frozen_lock = _json(frozen_payload, "frozen-feature receipt")
    verify_receipt(frozen_lock, key, "frozen-feature receipt")
    required_provenance = {
        "schema", "protocol_version", "frozen", "causal_input", "model_id", "model_revision",
        "layer", "checkpoint_sha256", "extractor_code_sha256", "feature_npz_sha256",
        "renderer_receipt_sha256", "config_sha256", "context_start", "state_reset",
        "history_truncation", "key_fingerprint", "hmac_sha256",
    }
    if set(frozen_lock) != required_provenance:
        raise ValueError("frozen-feature receipt has missing or unknown provenance fields")
    if (
        frozen_lock["schema"] != _EXTERNAL_SCHEMA
        or frozen_lock["protocol_version"] != protocol
        or frozen_lock["frozen"] is not True
        or frozen_lock["causal_input"] != "authenticated_prefix_clip_only"
        or frozen_lock["context_start"] != "authenticated_render_record_start_frame"
        or frozen_lock["state_reset"] != "per_prefix_clip"
        or frozen_lock["history_truncation"] != "renderer_prefix_or_session_start"
    ):
        raise ValueError("frozen-feature receipt violates the causal extractor contract")
    for field in ("model_id", "model_revision"):
        if not isinstance(frozen_lock[field], str) or not frozen_lock[field].strip():
            raise ValueError(f"frozen-feature receipt requires {field}")
    layer = frozen_lock["layer"]
    if not ((isinstance(layer, str) and layer.strip()) or (isinstance(layer, int) and not isinstance(layer, bool) and layer >= 0)):
        raise ValueError("frozen-feature receipt requires a layer name or index")
    for field in (
        "checkpoint_sha256", "extractor_code_sha256", "feature_npz_sha256",
        "renderer_receipt_sha256", "config_sha256",
    ):
        require_digest(frozen_lock[field], f"frozen-feature receipt {field}")
    if frozen_lock["feature_npz_sha256"] != _sha256(external_payload):
        raise ValueError("external frozen-feature NPZ hash mismatch")
    if frozen_lock["renderer_receipt_sha256"] != _sha256(renderer_payload):
        raise ValueError("external frozen features bind a different renderer receipt")
    if frozen_lock["config_sha256"] != config_hash:
        raise ValueError("external frozen features bind a different config")
    external = _load_external(external_payload)
    if set(external["item_id"].tolist()) != set(rows):
        raise ValueError("external frozen features must exactly cover dataset items")
    external_index = {item: i for i, item in enumerate(external["item_id"].tolist())}

    columns: dict[str, list[Any]] = {
        name: [] for name in (
            "item_id", "session_id", "source", "lineage_id", "split", "action", "context",
            "elapsed_s", "model_cutoff_s", "available_at_s", "prefix_sha256", "context_start_s",
            "render_boundary_s",
            "sampling_weight_status", "acoustic", "energy_change", "history_truncated", "hidden",
        )
    }
    optional = [name for name in _OPTIONAL_EXTERNAL if name in external]
    for name in optional:
        columns[name] = []
    if "hidden_previous" in external:
        columns["hidden_velocity"] = []

    for item in sorted(rows):
        label, identity = rows[item], assignments[item]
        for field in ("source", "lineage_id", "split", "elapsed_s"):
            if label.get(field) != identity[field]:
                raise ValueError(f"item {item!r} has inconsistent {field} across dataset and assignments")
        if float(label["model_cutoff_s"]) != float(identity["model_cutoff_s"]):
            raise ValueError(f"item {item!r} has inconsistent configured cutoff")
        record, media = render_records.get(item), media_index.get(item)
        if record is None or media is None:
            raise ValueError(f"item {item!r} is missing authenticated renderer metadata")
        if record.get("path") != media.get("prefix_clip") or record.get("sha256") != media.get("prefix_sha256"):
            raise ValueError("renderer record and media index prefix binding mismatch")
        if record.get("channels") != ["user", "assistant"]:
            raise ValueError("prefix channel order is not authenticated user/assistant stereo")
        for field in ("start_frame", "end_frame", "sample_rate", "sample_width_bytes"):
            if isinstance(record.get(field), bool) or not isinstance(record.get(field), int):
                raise ValueError(f"render record requires integer {field}")
        start, end, rate = record["start_frame"], record["end_frame"], record["sample_rate"]
        configured_cutoff = float(identity["model_cutoff_s"])
        if rate <= 0 or not 0 <= start <= end or _nearest_frame(configured_cutoff, rate) != end:
            raise ValueError("render record does not match the configured exact cutoff")
        render_boundary = end / rate
        acoustic_available = (end - 1) / rate if end > start else start / rate
        if acoustic_available > configured_cutoff + 1e-9:
            raise ValueError("rendered prefix contains a sample after the configured cutoff")
        path = Path(prefix_root) / record["path"]
        if sha256_file(path) != record["sha256"] or prefix_hashes[item] != record["sha256"]:
            raise ValueError("prefix bytes do not match renderer and assignment receipts")
        samples, wav_rate, wav_width = _pcm(path)
        if wav_rate != rate or wav_width != record["sample_width_bytes"] or len(samples) != end - start:
            raise ValueError("prefix PCM dimensions do not match signed render boundaries")
        acoustic, energy_change, truncated = _window_stats(samples, rate)
        ext = external_index[item]
        if external["prefix_sha256"][ext] != record["sha256"]:
            raise ValueError("external frozen feature prefix hash mismatch")
        if abs(external["model_cutoff_s"][ext] - configured_cutoff) > 1e-9:
            raise ValueError("external frozen feature cutoff does not match the configured checkpoint")
        hidden_available = float(external["available_at_s"][ext])
        if hidden_available > configured_cutoff + 1e-9:
            raise ValueError("external frozen feature observes beyond the configured cutoff")
        available = max(acoustic_available, hidden_available)
        CausalFeatureRecord(item, configured_cutoff, acoustic_available, tuple(acoustic), "acoustic")
        CausalFeatureRecord(item, configured_cutoff, hidden_available, tuple(external["hidden"][ext]), "hidden")
        values = {
            "item_id": item,
            "session_id": identity["session_id"],
            "source": identity["source"],
            "lineage_id": identity["lineage_id"],
            "split": identity["split"],
            "action": label["action"],
            "context": "joint_silence",
            "elapsed_s": float(identity["elapsed_s"]),
            "model_cutoff_s": configured_cutoff,
            "available_at_s": available,
            "prefix_sha256": record["sha256"],
            "context_start_s": start / rate,
            "render_boundary_s": render_boundary,
            "sampling_weight_status": label.get("sampling_weight_status"),
            "acoustic": acoustic,
            "energy_change": energy_change,
            "history_truncated": truncated,
            "hidden": external["hidden"][ext],
        }
        if values["sampling_weight_status"] != "not_identifiable":
            raise ValueError("pilot sampling weights must remain not_identifiable")
        for name in optional:
            values[name] = external[name][ext]
        if "hidden_previous" in external:
            values["hidden_velocity"] = external["hidden"][ext] - external["hidden_previous"][ext]
        for name, value in values.items():
            columns[name].append(value)

    arrays = {name: np.asarray(value) for name, value in columns.items()}
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    feature_payload = stream.getvalue()
    feature_hash = _sha256(feature_payload)
    receipt = sign_receipt({
        "schema": _OUTPUT_SCHEMA,
        "protocol_version": protocol,
        "counts": {"items": len(rows), "train": int(np.sum(arrays["split"] == "train")), "development": int(np.sum(arrays["split"] == "development"))},
        "acoustic_recipe": {
            "input": "authenticated_prefix_clip_only",
            "channel_order": ["user", "assistant"],
            "trailing_windows_ms": list(_WINDOWS_MS),
            "window_rounding": "nearest_sample_minimum_one",
            "vad_absolute_pcm_threshold": _VAD_THRESHOLD,
            "vad_context_boundary_state": "inactive",
            "log_rms_floor": _LOG_FLOOR,
            "includes_previous_window": True,
            "includes_log_rms_delta": True,
            "includes_vad_occupancy_and_onset_count": True,
            "includes_history_truncation_indicators": True,
            "energy_change_order_per_window": [
                "log_rms_delta_user", "log_rms_delta_assistant",
                "current_vad_onsets_user", "current_vad_onsets_assistant",
                "previous_vad_onsets_user", "previous_vad_onsets_assistant",
            ],
            "column_order_per_window": [
                "current_log_rms_user", "current_log_rms_assistant",
                "current_vad_occupancy_user", "current_vad_occupancy_assistant",
                "current_vad_onsets_user", "current_vad_onsets_assistant",
                "previous_log_rms_user", "previous_log_rms_assistant",
                "previous_vad_occupancy_user", "previous_vad_occupancy_assistant",
                "previous_vad_onsets_user", "previous_vad_onsets_assistant",
                "log_rms_delta_user", "log_rms_delta_assistant",
            ],
            "history_truncation_order_per_window": ["current", "previous"],
        },
        "frozen_feature_provenance": {
            field: frozen_lock[field] for field in (
                "model_id", "model_revision", "layer", "checkpoint_sha256", "extractor_code_sha256",
                "context_start", "state_reset", "history_truncation",
            )
        },
        "sha256": {
            "features.npz": feature_hash,
            "dataset_receipt": _sha256(dataset_payload),
            "adjudication_receipt": _sha256(adjudication_payload),
            "assignment_receipt": _sha256(assignment_payload),
            "private_assignment_source_map": _sha256(private_payload),
            "renderer_receipt": _sha256(renderer_payload),
            "external_feature_npz": frozen_lock["feature_npz_sha256"],
            "external_feature_receipt": _sha256(frozen_payload),
            "config": config_hash,
        },
        "outputs": {"features.npz": feature_hash},
    }, key)
    return _publish_packages({Path(output_dir): {
        "features.npz": feature_payload,
        "feature_receipt.json": receipt_bytes(receipt),
    }})[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--adjudication-root", type=Path, required=True)
    parser.add_argument("--prefix-root", type=Path, required=True)
    parser.add_argument("--renderer-root", type=Path, required=True)
    parser.add_argument("--annotator-root", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--frozen-features", type=Path, required=True)
    parser.add_argument("--frozen-receipt", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="silence.yaml")
    args = parser.parse_args()
    build_features(
        args.dataset_root, args.adjudication_root, args.prefix_root, args.renderer_root,
        args.annotator_root, args.assignment_root, args.frozen_features, args.frozen_receipt,
        args.hmac_key, args.output_dir, args.config,
    )


if __name__ == "__main__":
    main()
