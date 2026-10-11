#!/usr/bin/env python3
"""Build blinded prefix assignments and release presentation-specific reveals."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import random
import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

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
    from experiments.silence_graft.render_clips import (
        _config_path,
        _item_id,
        _json,
        _jsonl,
        _lines,
        _directory_identity,
        _publish_staging,
        _remove_owned_directory,
        _require_ready,
        _reserve,
        _write,
    )
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from receipts import read_hmac_key, receipt_bytes, require_digest, sha256_bytes, sha256_file, sign_receipt, verify_receipt
    from render_clips import (
        _config_path, _directory_identity, _item_id, _json, _jsonl, _lines,
        _publish_staging, _remove_owned_directory, _require_ready, _reserve, _write,
    )

_QUEUE_SCHEMA = "semantic-silence-prefix-assignment/v1"
_RECEIPT_SCHEMA = "semantic-silence-assignment-receipt/v1"
_RELEASE_SCHEMA = "semantic-silence-reveal-assignment/v1"
_RELEASE_RECEIPT_SCHEMA = "semantic-silence-release-receipt/v1"
_AUDITION_SCHEMA = "semantic-silence-channel-audition/v1"
_COMPLETION_SCHEMA = "semantic-silence-prefix-completion-receipt/v1"
_RENDERER_SCHEMA = "semantic-silence-renderer-receipt/v1"
_PUBLIC_PREFIX_KEYS = {"assignment_id", "prefix_clip", "presentation_id", "protocol_version", "schema"}
_PUBLIC_REVEAL_KEYS = {"assignment_id", "presentation_id", "protocol_version", "reveal_clip", "schema"}


def _sha256(payload: bytes) -> str:
    return sha256_bytes(payload)


def _safe_path(root: Path, relative: Any, where: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{where} must be a relative path")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{where} escapes its package") from error
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe {where}: {relative!r}")
    return path


def _check_hash(path: Path, expected: Any, where: str) -> None:
    digest = require_digest(expected, where)
    if not hmac.compare_digest(sha256_file(path), digest):
        raise ValueError(f"{where} SHA-256 mismatch")


def _checked_bytes(path: Path, expected: Any, where: str) -> bytes:
    digest = require_digest(expected, where)
    payload = path.read_bytes()
    if not hmac.compare_digest(_sha256(payload), digest):
        raise ValueError(f"{where} SHA-256 mismatch")
    return payload


def _verify_output_map(root: Path, outputs: Any, where: str) -> dict[str, str]:
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f"{where} requires output hashes")
    checked: dict[str, str] = {}
    for relative, digest in outputs.items():
        if not isinstance(relative, str):
            raise ValueError(f"{where} output path must be a string")
        checked[relative] = require_digest(digest, f"{where} output {relative!r}")
        _check_hash(_safe_path(root, relative, f"{where} output"), digest, f"{where} output {relative!r}")
    return checked


def _load_annotators(payload: bytes) -> list[str]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid annotator ID JSON: {error}") from error
    if isinstance(value, dict):
        value = value.get("annotators")
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("annotator ID JSON must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError("annotator IDs must be distinct")
    if len(value) < 3:
        raise ValueError("at least 3 annotators are required")
    return sorted(value)


def _opaque(key: bytes, domain: str, *values: str) -> str:
    payload = "\0".join((domain, *values)).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _rng(seed: int, domain: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{domain}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _roots_overlap(*roots: Path) -> bool:
    resolved = [root.resolve() for root in roots]
    return any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(resolved) for right in resolved[index + 1:]
    )


def _publish_packages(packages: Mapping[Path, Mapping[str, bytes]]) -> tuple[Path, ...]:
    targets = list(packages)
    staging: dict[Path, Path] = {}
    reserved: dict[Path, tuple[int, int]] = {}
    try:
        for target, files in packages.items():
            staging[target] = _reserve(target)
            reserved[target] = _directory_identity(target)
            for relative, payload in files.items():
                path = staging[target] / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                _write(path, payload)
        for target in targets:
            _publish_staging(staging.pop(target), target)
    except Exception:
        for path in staging.values():
            shutil.rmtree(path, ignore_errors=True)
        for path, identity in reserved.items():
            _remove_owned_directory(path, identity)
        raise
    return tuple(targets)


def _scheduled(rows: list[dict[str, Any]], episodes: Mapping[str, str], rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(episodes[row["item_id"]], []).append(row)
    for values in groups.values():
        rng.shuffle(values)
    if groups and max(map(len, groups.values())) > (len(rows) + 1) // 2:
        raise ValueError("cannot avoid adjacent assignments from the same episode")
    rank = list(groups)
    rng.shuffle(rank)
    priority = {episode: index for index, episode in enumerate(rank)}
    result: list[dict[str, Any]] = []
    previous: str | None = None
    while groups:
        choices = [episode for episode in groups if episode != previous]
        episode = min(choices, key=lambda value: (-len(groups[value]), priority[value]))
        result.append(groups[episode].pop())
        if not groups[episode]:
            del groups[episode]
        previous = episode
    return result


def _sentinels(path: Path | None, config_value: Any, audition: Mapping[str, Any]) -> tuple[set[str], bytes | None]:
    if config_value is None:
        if path is not None:
            raise ValueError("sentinel list is forbidden when config value is null")
        return set(), None
    configured_digest = require_digest(config_value, "config coordinator_locked_sentinel_list")
    if path is None:
        raise ValueError("configured sentinel list is required")
    payload = path.read_bytes()
    digest = _sha256(payload)
    if not hmac.compare_digest(configured_digest, digest):
        raise ValueError("sentinel list SHA-256 does not match config")
    hashes = audition.get("sha256")
    if not isinstance(hashes, dict) or hashes.get("sentinel_list") != digest:
        raise ValueError("sentinel list is not locked by the authenticated audition receipt")
    value = json.loads(payload.decode("utf-8"))
    if isinstance(value, dict):
        value = value.get("item_ids")
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("sentinel list must contain item IDs")
    if len(value) != len(set(value)):
        raise ValueError("sentinel list contains duplicate item IDs")
    return set(value), payload


def _validate_audited_at(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("channel-audition receipt requires audited_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("channel-audition audited_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("channel-audition audited_at must include a timezone")


def build_assignments(
    prefix_root: Path,
    coordinator_root: Path,
    annotator_ids: Path,
    hmac_key: Path,
    audition_receipt: Path,
    annotator_output_dir: Path,
    coordinator_output_dir: Path,
    config: str | Path = "silence.yaml",
    sentinel_list: Path | None = None,
) -> tuple[Path, Path]:
    """Build deterministic public and coordinator assignment packages."""
    prefix_root, coordinator_root, annotator_ids, audition_receipt = map(
        Path, (prefix_root, coordinator_root, annotator_ids, audition_receipt)
    )
    annotator_output_dir, coordinator_output_dir = map(Path, (annotator_output_dir, coordinator_output_dir))
    if _roots_overlap(annotator_output_dir, coordinator_output_dir):
        raise ValueError("annotator and coordinator assignment output directories must be separate")
    if any(
        _roots_overlap(target, source)
        for target in (annotator_output_dir, coordinator_output_dir)
        for source in (prefix_root, coordinator_root)
    ):
        raise ValueError("assignment outputs must be separate from renderer packages")
    _require_ready(prefix_root, "prefix package")
    _require_ready(coordinator_root, "coordinator package")
    key = read_hmac_key(hmac_key)
    config_path = _config_path(config)
    config_payload = config_path.read_bytes()
    protocol_payload = (repo_root() / "docs" / "DATA_LABELING_PROTOCOL.md").read_bytes()
    annotator_payload = annotator_ids.read_bytes()
    renderer_path = coordinator_root / "renderer_receipt.json"
    renderer_payload = renderer_path.read_bytes()
    renderer = _json(renderer_payload, "renderer receipt")
    verify_receipt(renderer, key, "renderer receipt")
    if renderer.get("schema") != _RENDERER_SCHEMA:
        raise ValueError("invalid renderer receipt schema")

    cfg = yaml.safe_load(config_payload)
    annotation = cfg.get("annotation") if isinstance(cfg, dict) else None
    silence = cfg.get("silence") if isinstance(cfg, dict) else None
    if not isinstance(annotation, dict) or not isinstance(silence, dict):
        raise ValueError("config requires annotation and silence mappings")
    if annotation.get("annotators_per_item") != 3:
        raise ValueError("config must require exactly 3 annotators per item")
    if annotation.get("repeat_fraction") != 0.05 or isinstance(annotation.get("repeat_fraction"), bool):
        raise ValueError("protocol v1 requires repeat_fraction exactly 0.05")
    if annotation.get("automatic_sentinels") is not False:
        raise ValueError("automatic sentinels must be disabled")
    sentinel_config = annotation.get("coordinator_locked_sentinel_list")
    if sentinel_config is not None:
        require_digest(sentinel_config, "config coordinator_locked_sentinel_list")
    seed = silence.get("split_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("config silence.split_seed must be an integer")
    protocol = cfg.get("protocol_version")
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("config requires protocol_version")
    if renderer.get("protocol_version") != protocol:
        raise ValueError("renderer protocol version does not match config")
    hashes = renderer.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("renderer receipt requires input hashes")
    for name, digest in hashes.items():
        require_digest(digest, f"renderer receipt sha256.{name}")
    if hashes.get("config") != _sha256(config_payload) or hashes.get("protocol_doc") != _sha256(protocol_payload):
        raise ValueError("config or protocol document SHA-256 does not match renderer receipt")

    prefix_outputs = _verify_output_map(prefix_root, renderer.get("prefix_outputs"), "renderer prefix package")
    coordinator_outputs = _verify_output_map(coordinator_root, renderer.get("outputs"), "renderer coordinator package")
    if any("reveal" in relative.lower() for relative in prefix_outputs):
        raise ValueError("reveal output must never exist under the prefix root")
    if any(path.is_symlink() for path in prefix_root.rglob("*")):
        raise ValueError("prefix package must not contain symlinks")
    actual_prefix = {str(path.relative_to(prefix_root)) for path in prefix_root.rglob("*") if path.is_file()}
    if actual_prefix != set(prefix_outputs) | {"READY"}:
        raise ValueError("prefix package contains unrecorded or missing files")
    if any(path.is_symlink() for path in coordinator_root.rglob("*")):
        raise ValueError("coordinator package must not contain symlinks")
    actual_coordinator = {
        str(path.relative_to(coordinator_root))
        for path in coordinator_root.rglob("*")
        if path.is_file()
    }
    if actual_coordinator != set(coordinator_outputs) | {"renderer_receipt.json", "READY"}:
        raise ValueError("coordinator package contains unrecorded or missing files")

    index_path = coordinator_root / "coordinator_media_index.jsonl"
    source_path = coordinator_root / "private_source_mapping.jsonl"
    index_payload, source_payload = index_path.read_bytes(), source_path.read_bytes()
    if coordinator_outputs.get("coordinator_media_index.jsonl") != _sha256(index_payload):
        raise ValueError("coordinator media index is not locked by renderer receipt")
    if coordinator_outputs.get("private_source_mapping.jsonl") != _sha256(source_payload):
        raise ValueError("private source mapping is not locked by renderer receipt")

    index_rows, source_rows = _jsonl(index_payload, "coordinator media index"), _jsonl(source_payload, "private source mapping")
    by_item: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    episodes: dict[str, str] = {}
    for number, row in enumerate(index_rows, 1):
        item = row.get("item_id")
        if not isinstance(item, str) or item in by_item:
            raise ValueError(f"invalid or duplicate item_id in coordinator media index row {number}")
        prefix_path = _safe_path(prefix_root, row.get("prefix_clip"), "prefix clip")
        reveal_path = _safe_path(coordinator_root, row.get("reveal_clip"), "reveal clip")
        _check_hash(prefix_path, row.get("prefix_sha256"), "prefix clip")
        _check_hash(reveal_path, row.get("reveal_sha256"), "reveal clip")
        if prefix_outputs.get(row["prefix_clip"]) != row["prefix_sha256"]:
            raise ValueError("prefix clip is not locked by renderer receipt")
        if coordinator_outputs.get(row["reveal_clip"]) != row["reveal_sha256"]:
            raise ValueError("reveal clip is not locked by renderer receipt")
        if row.get("protocol_version") != protocol:
            raise ValueError("media index protocol version does not match config")
        by_item[item] = {**row, "prefix_path": prefix_path, "reveal_path": reveal_path}

    session_bindings: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(source_rows, 1):
        item = row.get("item_id")
        required_strings = ("episode_id", "source", "lineage_id", "session_id")
        if item not in by_item or item in sources:
            raise ValueError(f"private source mapping row {number} has unknown or duplicate item_id")
        if any(not isinstance(row.get(field), str) or not row[field] for field in required_strings):
            raise ValueError(f"private source mapping row {number} requires source and lineage identity")
        require_digest(row.get("wav_sha256"), f"private source mapping row {number} wav_sha256")
        cutoff = row.get("model_cutoff_s")
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
            raise ValueError(f"private source mapping row {number} requires model_cutoff_s")
        for field in ("user_channel", "assistant_channel"):
            if isinstance(row.get(field), bool) or not isinstance(row.get(field), int):
                raise ValueError(f"private source mapping row {number} requires integer {field}")
        if not hmac.compare_digest(item, _item_id(key, row["episode_id"], float(cutoff))):
            raise ValueError("item ID does not match private HMAC key and source mapping")
        binding = {
            "assistant_channel": row["assistant_channel"], "session_id": row["session_id"],
            "user_channel": row["user_channel"], "wav_sha256": row["wav_sha256"],
        }
        previous = session_bindings.setdefault(row["session_id"], binding)
        if previous != binding:
            raise ValueError("private source mapping has inconsistent session bindings")
        sources[item], episodes[item] = row, row["episode_id"]
    if set(sources) != set(by_item):
        raise ValueError("coordinator media index and private source mapping items differ")
    receipt_sources = renderer.get("sources")
    if (
        not isinstance(receipt_sources, list)
        or any(not isinstance(row, dict) for row in receipt_sources)
        or sorted(receipt_sources, key=lambda row: row.get("session_id", ""))
        != [session_bindings[key] for key in sorted(session_bindings)]
    ):
        raise ValueError("renderer receipt source bindings do not match private mapping")

    audition_payload = audition_receipt.read_bytes()
    audition = _json(audition_payload, "channel-audition receipt")
    verify_receipt(audition, key, "channel-audition receipt")
    if audition.get("schema") != _AUDITION_SCHEMA or audition.get("protocol_version") != protocol:
        raise ValueError("invalid channel-audition receipt schema or protocol")
    if not isinstance(audition.get("auditor_identity"), str) or not audition["auditor_identity"].strip():
        raise ValueError("channel-audition receipt requires auditor_identity")
    _validate_audited_at(audition.get("audited_at"))
    audition_hashes = audition.get("sha256")
    expected_hashes = {
        "config": _sha256(config_payload), "coordinator_media_index": _sha256(index_payload),
        "private_source_mapping": _sha256(source_payload), "renderer_receipt": _sha256(renderer_payload),
    }
    if not isinstance(audition_hashes, dict) or any(audition_hashes.get(name) != digest for name, digest in expected_hashes.items()):
        raise ValueError("channel-audition receipt hashes do not match inputs")
    for name, digest in audition_hashes.items():
        require_digest(digest, f"channel-audition receipt sha256.{name}")
    coverage = audition.get("sessions")
    if not isinstance(coverage, list):
        raise ValueError("channel-audition receipt requires session coverage")
    approved: dict[str, dict[str, Any]] = {}
    for row in coverage:
        if not isinstance(row, dict) or not isinstance(row.get("session_id"), str):
            raise ValueError("invalid channel-audition session coverage")
        if row.get("user_channel_auditioned") is not True or row.get("assistant_channel_auditioned") is not True:
            raise ValueError("both channels must be explicitly auditioned for every session")
        session = row["session_id"]
        if session in approved:
            raise ValueError("duplicate channel-audition session coverage")
        approved[session] = row
    if set(approved) != set(session_bindings):
        raise ValueError("channel audition must cover exactly every pilot session")
    for session, binding in session_bindings.items():
        for field in ("session_id", "wav_sha256", "user_channel", "assistant_channel"):
            if approved[session].get(field) != binding[field]:
                raise ValueError(f"channel-audition mapping mismatch for session {session!r}")

    sentinel_ids, sentinel_payload = _sentinels(
        Path(sentinel_list) if sentinel_list is not None else None, sentinel_config, audition
    )
    if sentinel_ids - by_item.keys():
        raise ValueError("sentinel list contains unknown item IDs")

    annotators = _load_annotators(annotator_payload)
    shuffled_items, shuffled_annotators = sorted(by_item), annotators.copy()
    _rng(seed, "item-balance").shuffle(shuffled_items)
    _rng(seed, "annotator-balance").shuffle(shuffled_annotators)
    queues: dict[str, list[dict[str, Any]]] = {annotator: [] for annotator in annotators}
    for index, item in enumerate(shuffled_items):
        for slot in range(3):
            annotator = shuffled_annotators[(index * 3 + slot) % len(annotators)]
            queues[annotator].append({
                "annotator_id": annotator,
                "assignment_id": _opaque(key, "assignment", annotator, item, "original"),
                "item_id": item,
                "presentation_id": _opaque(key, "presentation", annotator, item, "original"),
                "repeat": False,
                "sentinel": item in sentinel_ids,
            })

    private_rows: list[dict[str, Any]] = []
    reveal_rows: list[dict[str, Any]] = []
    annotator_files: dict[str, bytes] = {}
    queue_counts: dict[str, int] = {}
    for annotator in annotators:
        originals = queues[annotator]
        repeat_count = min(len(originals), max(1, int(len(originals) * 0.05 + 0.5))) if originals else 0
        choices = originals.copy()
        _rng(seed, f"repeat\0{annotator}").shuffle(choices)
        for original in choices[:repeat_count]:
            item = original["item_id"]
            originals.append({
                **original,
                "assignment_id": _opaque(key, "assignment", annotator, item, "repeat"),
                "presentation_id": _opaque(key, "presentation", annotator, item, "repeat"),
                "repeat": True,
            })
        ordered = _scheduled(originals, episodes, _rng(seed, f"queue\0{annotator}"))
        opaque_annotator = _opaque(key, "annotator", annotator)
        package_relative = f"annotators/{opaque_annotator}"
        queue_relative = f"{package_relative}/queue.jsonl"
        public_rows: list[dict[str, Any]] = []
        for position, row in enumerate(ordered):
            item, presentation = row["item_id"], row["presentation_id"]
            prefix_alias = f"media/prefix/{presentation}.wav"
            annotator_files[f"{package_relative}/{prefix_alias}"] = _checked_bytes(
                by_item[item]["prefix_path"], by_item[item]["prefix_sha256"], "prefix clip"
            )
            public = {
                "assignment_id": row["assignment_id"], "prefix_clip": prefix_alias,
                "presentation_id": presentation, "protocol_version": protocol, "schema": _QUEUE_SCHEMA,
            }
            if set(public) != _PUBLIC_PREFIX_KEYS:
                raise AssertionError("prefix queue contains a private field")
            public_rows.append(public)
            private_rows.append({
                **sources[item], **row, "annotator_opaque_id": opaque_annotator,
                "position": position, "prefix_alias": f"{package_relative}/{prefix_alias}", "queue": queue_relative,
            })
            reveal_rows.append({
                **row, "annotator_opaque_id": opaque_annotator,
                "prefix_sha256": by_item[item]["prefix_sha256"], "protocol_version": protocol,
                "reveal_sha256": by_item[item]["reveal_sha256"],
                "reveal_source_path": str(by_item[item]["reveal_path"].resolve()),
            })
        annotator_files[queue_relative] = _lines(public_rows)
        queue_counts[opaque_annotator] = len(public_rows)

    coordinator_files = {
        "private_assignment_source_map.jsonl": _lines(private_rows),
        "coordinator_reveal_release_map.jsonl": _lines(reveal_rows),
    }
    annotator_output_hashes = {name: _sha256(payload) for name, payload in sorted(annotator_files.items())}
    coordinator_output_hashes = {name: _sha256(payload) for name, payload in sorted(coordinator_files.items())}
    receipt = sign_receipt({
        "counts": {
            "annotators": len(annotators), "items": len(by_item),
            "prefix_assignments": len(private_rows), "queue_presentations": queue_counts,
        },
        "annotator_outputs": annotator_output_hashes,
        "outputs": coordinator_output_hashes,
        "protocol_version": protocol,
        "schema": _RECEIPT_SCHEMA,
        "sha256": {
            "annotator_ids": _sha256(annotator_payload), "audition_receipt": _sha256(audition_payload),
            "config": _sha256(config_payload), "coordinator_media_index": _sha256(index_payload),
            "protocol_doc": _sha256(protocol_payload), "renderer_receipt": _sha256(renderer_payload),
            **({"sentinel_list": _sha256(sentinel_payload)} if sentinel_payload is not None else {}),
        },
    }, key)
    coordinator_files["assignment_receipt.json"] = receipt_bytes(receipt)
    return _publish_packages({annotator_output_dir: annotator_files, coordinator_output_dir: coordinator_files})


def release(
    annotator_root: Path,
    coordinator_root: Path,
    completions: Path,
    completion_receipt: Path,
    hmac_key: Path,
    output_dir: Path,
) -> Path:
    """Release aliased reveals only after every signed completion is valid."""
    annotator_root, coordinator_root, completions, completion_receipt, output_dir = map(
        Path, (annotator_root, coordinator_root, completions, completion_receipt, output_dir)
    )
    if _roots_overlap(annotator_root, coordinator_root):
        raise ValueError("annotator and coordinator assignment roots must be separate")
    if any(_roots_overlap(output_dir, root) for root in (annotator_root, coordinator_root)):
        raise ValueError("release output must be separate from the assignment packages")
    _require_ready(annotator_root, "annotator assignment package")
    _require_ready(coordinator_root, "coordinator assignment package")
    key = read_hmac_key(hmac_key)
    assignment_receipt_path = coordinator_root / "assignment_receipt.json"
    assignment_payload = assignment_receipt_path.read_bytes()
    assignment_data = _json(assignment_payload, "assignment receipt")
    verify_receipt(assignment_data, key, "assignment receipt")
    if assignment_data.get("schema") != _RECEIPT_SCHEMA:
        raise ValueError("invalid assignment receipt schema")
    annotator_outputs = _verify_output_map(
        annotator_root, assignment_data.get("annotator_outputs"), "assignment receipt annotator package"
    )
    assignment_outputs = _verify_output_map(
        coordinator_root, assignment_data.get("outputs"), "assignment receipt coordinator package"
    )
    for root, outputs, extra, where in (
        (annotator_root, annotator_outputs, {"READY"}, "annotator assignment package"),
        (coordinator_root, assignment_outputs, {"assignment_receipt.json", "READY"}, "coordinator assignment package"),
    ):
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError(f"{where} must not contain symlinks")
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        if actual != set(outputs) | extra:
            raise ValueError(f"{where} contains unrecorded or missing files")
    reveal_relative = "coordinator_reveal_release_map.jsonl"
    reveal_path = coordinator_root / reveal_relative
    reveal_payload = reveal_path.read_bytes()
    if assignment_outputs.get(reveal_relative) != _sha256(reveal_payload):
        raise ValueError("coordinator reveal map is not locked by assignment receipt")
    protocol = assignment_data.get("protocol_version")

    completion_payload, lock_payload = completions.read_bytes(), completion_receipt.read_bytes()
    lock = _json(lock_payload, "completion receipt")
    verify_receipt(lock, key, "completion receipt")
    if lock.get("schema") != _COMPLETION_SCHEMA or lock.get("protocol_version") != protocol:
        raise ValueError("invalid completion receipt schema or protocol")
    hashes = lock.get("sha256")
    if not isinstance(hashes, dict) or hashes.get("completions") != _sha256(completion_payload):
        raise ValueError("completion records SHA-256 does not match receipt")
    if hashes.get("assignment_receipt") != _sha256(assignment_payload):
        raise ValueError("assignment receipt SHA-256 does not match completion receipt")
    for name, digest in hashes.items():
        require_digest(digest, f"completion receipt sha256.{name}")

    expected_rows = _jsonl(reveal_payload, "coordinator reveal-release map")
    expected: dict[str, dict[str, Any]] = {}
    for row in expected_rows:
        assignment = row.get("assignment_id")
        if not isinstance(assignment, str) or assignment in expected:
            raise ValueError("duplicate or invalid assignment in reveal-release map")
        expected[assignment] = row
    seen: set[str] = set()
    for number, row in enumerate(_jsonl(completion_payload, "completion records"), 1):
        assignment = row.get("assignment_id")
        if not isinstance(assignment, str) or assignment not in expected:
            raise ValueError(f"unknown completion assignment at row {number}")
        if assignment in seen:
            raise ValueError(f"duplicate completion assignment at row {number}")
        wanted = expected[assignment]
        for field in ("assignment_id", "presentation_id", "annotator_id", "item_id", "prefix_sha256"):
            if row.get(field) != wanted[field]:
                raise ValueError(f"tampered completion {field} at row {number}")
        require_digest(row.get("annotation_sha256"), f"completion row {number} annotation_sha256")
        seen.add(assignment)
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"missing {len(missing)} expected prefix completion(s); no reveals released")

    files: dict[str, bytes] = {}
    public_by_annotator: dict[str, list[dict[str, Any]]] = {}
    for row in expected_rows:
        source = Path(row["reveal_source_path"])
        if not source.is_absolute() or not source.is_file() or source.is_symlink():
            raise ValueError("missing or unsafe coordinator reveal source")
        payload = _checked_bytes(source, row["reveal_sha256"], "coordinator reveal source")
        alias = f"media/reveal/{row['presentation_id']}.wav"
        files[f"annotators/{row['annotator_opaque_id']}/{alias}"] = payload
        public = {
            "assignment_id": row["assignment_id"], "presentation_id": row["presentation_id"],
            "protocol_version": protocol, "reveal_clip": alias, "schema": _RELEASE_SCHEMA,
        }
        if set(public) != _PUBLIC_REVEAL_KEYS:
            raise AssertionError("reveal queue contains a private field")
        public_by_annotator.setdefault(row["annotator_opaque_id"], []).append(public)
    for annotator, rows in sorted(public_by_annotator.items()):
        files[f"annotators/{annotator}/queue.jsonl"] = _lines(rows)

    outputs = {name: _sha256(payload) for name, payload in sorted(files.items())}
    receipt = sign_receipt({
        "counts": {"items": len({row["item_id"] for row in expected_rows}), "reveal_presentations": len(expected_rows)},
        "outputs": outputs,
        "protocol_version": protocol,
        "schema": _RELEASE_RECEIPT_SCHEMA,
        "sha256": {
            "assignment_receipt": _sha256(assignment_payload), "completion_receipt": _sha256(lock_payload),
            "completions": _sha256(completion_payload), "reveal_release_map": _sha256(reveal_payload),
        },
    }, key)
    files["release_receipt.json"] = receipt_bytes(receipt)
    return _publish_packages({output_dir: files})[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="operation", required=True)
    build = commands.add_parser("build")
    build.add_argument("--prefix-root", type=Path, required=True)
    build.add_argument("--coordinator-root", type=Path, required=True)
    build.add_argument("--annotator-ids", type=Path, required=True)
    build.add_argument("--hmac-key", type=Path, required=True)
    build.add_argument("--audition-receipt", type=Path, required=True)
    build.add_argument("--annotator-output-dir", type=Path, required=True)
    build.add_argument("--coordinator-output-dir", type=Path, required=True)
    build.add_argument("--config", default="silence.yaml")
    build.add_argument("--sentinel-list", type=Path)
    unlock = commands.add_parser("release")
    unlock.add_argument("--annotator-root", type=Path, required=True)
    unlock.add_argument("--coordinator-root", type=Path, required=True)
    unlock.add_argument("--completions", type=Path, required=True)
    unlock.add_argument("--completion-receipt", type=Path, required=True)
    unlock.add_argument("--hmac-key", type=Path, required=True)
    unlock.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "build":
        build_assignments(
            args.prefix_root, args.coordinator_root, args.annotator_ids, args.hmac_key,
            args.audition_receipt, args.annotator_output_dir, args.coordinator_output_dir,
            args.config, args.sentinel_list,
        )
    else:
        release(
            args.annotator_root, args.coordinator_root, args.completions,
            args.completion_receipt, args.hmac_key, args.output_dir,
        )


if __name__ == "__main__":
    main()
