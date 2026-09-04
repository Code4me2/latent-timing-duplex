"""Spark filename aliases: candor_ / dc_ and .pt stems."""

from __future__ import annotations

from latent_timing_duplex.phase1.artifacts import (
    list_session_ids_in_dir,
    resolve_named_file,
    session_id_aliases,
    session_id_from_filename,
    strip_session_prefix,
)


def test_strip_and_aliases() -> None:
    assert strip_session_prefix("candor_abc-uuid") == "abc-uuid"
    assert strip_session_prefix("dc_ep1") == "ep1"
    aliases = session_id_aliases("abc-uuid")
    assert "candor_abc-uuid" in aliases
    assert "dc_abc-uuid" in aliases
    assert "abc-uuid" in aliases


def test_resolve_candor_pt_by_bare_uuid(tmp_path) -> None:
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    target = hidden / "candor_deadbeef.pt"
    target.write_bytes(b"not-a-real-tensor")
    found = resolve_named_file(hidden, "deadbeef")
    assert found == target
    found2 = resolve_named_file(hidden, "candor_deadbeef")
    assert found2 == target
    assert list_session_ids_in_dir(hidden) == ["deadbeef"]


def test_resolve_dc_pt(tmp_path) -> None:
    folder = tmp_path / "tgt"
    folder.mkdir()
    (folder / "dc_episode99.pt").write_bytes(b"x")
    assert resolve_named_file(folder, "episode99") is not None
    assert session_id_from_filename(folder / "dc_episode99.pt") == "episode99"
