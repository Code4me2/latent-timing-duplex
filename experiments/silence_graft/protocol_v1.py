"""Strict shared loader for the locked semantic-silence protocol v1 config."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from latent_timing_duplex.config import find_config

try:
    from experiments.silence_graft.receipts import require_digest
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from receipts import require_digest


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _mapping(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def config_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_file() else find_config(str(value)).resolve()


def _reject_nonfinite(value: Any, where: str = "config") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} contains a non-finite YAML value")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{where}[{index}]")


def _same(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(value) == len(expected) and all(_same(left, right) for left, right in zip(value, expected))
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(_same(value[key], child) for key, child in expected.items())
    return value == expected


def _exact(
    mapping: Any,
    expected: dict[str, Any],
    where: str,
    *,
    nested: tuple[str, ...] = (),
) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"config {where} must be a mapping")
    if set(mapping) != set(expected) | set(nested):
        raise ValueError(f"config {where} has missing or unknown keys")
    for key, value in expected.items():
        if not _same(mapping.get(key), value):
            raise ValueError(f"config {where}.{key} must be exactly {value!r}")


def parse_protocol_v1(payload: bytes, path: str | Path = "config") -> dict[str, Any]:
    try:
        config = yaml.load(payload, Loader=_UniqueSafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise ValueError(f"invalid protocol config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    _reject_nonfinite(config)
    _exact(config, {
        "schema": "semantic-silence-config/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "label_schema": "semantic-silence-labels/v1",
        "prefix_response_schema": "semantic-silence-prefix-response/v1",
    }, "root", nested=("silence", "annotation", "pilot_media", "acceptable_onset", "gates"))
    _exact(config.get("silence"), {
        "checkpoints_s": [0.24, 0.48, 0.96, 2.0, 3.0, 5.0, 10.0],
        "duration_strata_field": "checkpoint_elapsed_s",
        "duration_strata_s": [1.0, 3.0],
        "duration_strata_labels": ["<1s", "1–<3s", ">=3s"],
        "use_realized_duration_for_strata": False,
        "max_episodes_per_lineage": 5,
        "split_seed": 20261009,
    }, "silence", nested=("gate",))
    _exact(config["silence"].get("gate"), {
        "threshold": 0.6,
        "consecutive_frames": 2,
    }, "silence.gate")
    _exact(config.get("annotation"), {
        "primary_actions": ["wait", "ack", "respond", "repair", "unratable"],
        "annotators_per_item": 3,
        "pilot_target_episodes": 75,
        "pilot_review_range_episodes": [60, 90],
        "pilot_min_lineages": 30,
        "pilot_min_pairable_fraction": 0.80,
        "pilot_min_pairable_lineages": 30,
        "pilot_splits": ["train", "development"],
        "pilot_excluded_sets": ["test", "challenge"],
        "agreement_gate": 0.80,
        "alpha_metric": "krippendorff_nominal",
        "unratable_for_action_alpha": "missing",
        "collapse_map": {"ack": "respond", "wait": "wait", "respond": "respond", "repair": "repair"},
        "collapsed_reliability_requires_new_pilot": True,
        "repeat_fraction": 0.05,
        "automatic_sentinels": False,
        "agreement_reports": ["per_class_positive_agreement", "onset_endpoint_disagreement", "repeat_consistency"],
    }, "annotation", nested=("coordinator_locked_sentinel_list", "bootstrap"))
    sentinel = config["annotation"].get("coordinator_locked_sentinel_list")
    if sentinel is not None:
        require_digest(sentinel, "config coordinator_locked_sentinel_list")
    _exact(config["annotation"].get("bootstrap"), {
        "unit": "lineage",
        "seed": 20261009,
        "replicates": 10000,
        "min_valid_fraction": 0.90,
    }, "annotation.bootstrap")
    _exact(config.get("pilot_media"), {
        "audio_only": True,
        "prefix_s": 20.0,
        "reveal_s": 8.0,
        "reveal_channel": "user",
        "reveal_clip_at_session_end": True,
        "boundary_rounding": "nearest_sample",
        "gain_scope": "conversation_or_session",
        "gain_db": 0.0,
        "transcript": False,
    }, "pilot_media")
    _exact(config.get("acceptable_onset"), {
        "supervision": "normative_interval_censored_assistant_entry",
        "recorded_partner_onset": "outcome_only",
        "user_resumption_tie": "cancel_assistant_entry",
    }, "acceptable_onset")
    _exact(config.get("gates"), {
        "r1_requires_pilot_agreement_receipt": True,
        "test_evaluation_requires_analysis_lock_receipt": True,
    }, "gates")
    return config


def load_protocol_v1(value: str | Path = "silence.yaml") -> tuple[Path, bytes, dict[str, Any]]:
    path = config_path(value)
    payload = path.read_bytes()
    return path, payload, parse_protocol_v1(payload, path)
