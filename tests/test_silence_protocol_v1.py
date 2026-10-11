"""Locked protocol-v1 config parsing regressions."""

from pathlib import Path

import pytest

from experiments.silence_graft.protocol_v1 import parse_protocol_v1

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_config_rejects_duplicate_yaml_keys() -> None:
    payload = (ROOT / "configs" / "silence.yaml").read_bytes() + b"schema: semantic-silence-config/v1\n"
    with pytest.raises(ValueError, match="duplicate YAML key"):
        parse_protocol_v1(payload)


def test_protocol_config_rejects_unknown_keys() -> None:
    payload = (ROOT / "configs" / "silence.yaml").read_bytes() + b"unknown: true\n"
    with pytest.raises(ValueError, match="missing or unknown keys"):
        parse_protocol_v1(payload)


def test_protocol_config_rejects_nonfinite_yaml_values() -> None:
    payload = (ROOT / "configs" / "silence.yaml").read_text().replace(
        "gain_db: 0.0", "gain_db: .nan"
    ).encode()
    with pytest.raises(ValueError, match="non-finite YAML value"):
        parse_protocol_v1(payload)
