"""Strict assigned-response and pilot-agreement checks."""

from __future__ import annotations

import hashlib
import json

import pytest

from latent_timing_duplex.silence import (
    AcceptableOnset,
    assigned_agreement_report,
    elapsed_stratum_agreement_reports,
    lineage_bootstrap_ci,
    load_assigned_annotations,
    load_assigned_annotations_bytes,
    nominal_krippendorff_alpha,
    onset_endpoint_disagreement,
    per_class_positive_agreement,
)


def _raw(assignment: str = "a1", **updates):
    row = {
        "schema": "semantic-silence-prefix-response/v1",
        "protocol_version": "semantic-silence-pilot/v1",
        "assignment_id": assignment,
        "presentation_id": f"p-{assignment}",
        "annotator_id": "annotator-1",
        "action": "wait",
        "confidence": 4,
        "also_acceptable": [],
        "tags": ["floor_holding"],
        "acceptable_onset": {
            "earliest_elapsed_s": None,
            "latest_elapsed_s": None,
            "wait_for_user_resume": True,
        },
        "quality": "clean",
        "ratable": True,
    }
    row.update(updates)
    return row


def _load(*rows):
    payload = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    return load_assigned_annotations_bytes(payload)


def test_assigned_parser_is_strict_and_preserves_canonical_digest() -> None:
    raw = _raw()
    annotation = _load(raw)[0]
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    assert annotation.annotation_sha256 == hashlib.sha256(canonical).hexdigest()

    for field, value, match in (
        ("confidence", True, "integer"),
        ("ratable", "true", "boolean"),
        ("schema", "semantic-silence-labels/v1", "schema"),
        ("protocol_version", "other", "protocol_version"),
    ):
        with pytest.raises(ValueError, match=match):
            _load(_raw(**{field: value}))
    with pytest.raises(ValueError, match="unknown assigned response fields"):
        _load(_raw(retrospective_tags=[]))
    missing_confidence = _raw()
    del missing_confidence["confidence"]
    with pytest.raises(ValueError, match="missing fields"):
        _load(missing_confidence)
    with pytest.raises(ValueError, match="zero or one"):
        _load(_raw(also_acceptable=["ack", "respond"]))
    for protected in ("listen", "assistant_speaking"):
        with pytest.raises(ValueError, match="invalid assigned response enum"):
            _load(_raw(action=protected))
    with pytest.raises(ValueError, match="non-finite"):
        load_assigned_annotations_bytes(
            json.dumps(_raw()).replace('"confidence": 4', '"confidence": NaN').encode()
        )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_assigned_annotations_bytes(
            b'{"schema":"semantic-silence-prefix-response/v1","schema":"x"}\n'
        )
    with pytest.raises(ValueError, match="duplicate assignment_id"):
        _load(raw, raw)
    with pytest.raises(ValueError, match="finite"):
        _load(_raw(acceptable_onset={
            "earliest_elapsed_s": float("inf"),
            "latest_elapsed_s": float("inf"),
            "wait_for_user_resume": False,
        }))


def test_assigned_path_loader_reads_exact_bytes(tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_bytes((json.dumps(_raw()) + "\n").encode())
    assert load_assigned_annotations(path)[0].row_sha256


def test_assigned_parser_rejects_onset_and_unratable_contradictions() -> None:
    with pytest.raises(ValueError, match="cannot include an onset window"):
        _load(
            _raw(
                acceptable_onset={
                    "earliest_elapsed_s": 1.0,
                    "latest_elapsed_s": 2.0,
                    "wait_for_user_resume": True,
                }
            )
        )
    with pytest.raises(ValueError, match="ratable rows require onset"):
        _load(_raw(acceptable_onset=None))
    with pytest.raises(ValueError, match="unratable rows"):
        _load(_raw(action="unratable", ratable=True, acceptable_onset=None))
    with pytest.raises(ValueError, match="alternate"):
        _load(
            _raw(
                action="unratable",
                ratable=False,
                acceptable_onset=None,
                also_acceptable=["wait"],
            )
        )


def test_nominal_alpha_perfect_disagreement_and_missing() -> None:
    assert nominal_krippendorff_alpha(
        {"one": ["wait", "wait"], "two": ["respond", "respond"]}
    ) == pytest.approx(1.0)
    assert nominal_krippendorff_alpha(
        [["wait", "respond"], ["wait", "respond"]]
    ) == pytest.approx(-0.5)
    assert nominal_krippendorff_alpha([["wait", "wait"], ["wait", "wait"]]) is None
    assert nominal_krippendorff_alpha(
        [["wait", "wait", "unratable"], ["respond", "respond", "unratable"]]
    ) == pytest.approx(1.0)
    assert nominal_krippendorff_alpha([["wait", "unratable"]]) is None


def test_positive_agreement_and_onset_mismatch() -> None:
    positive = per_class_positive_agreement(
        [["wait", "wait"], ["wait", "respond"]]
    )
    assert positive["wait"] == pytest.approx(2 / 3)
    assert positive["respond"] == 0.0
    assert positive["ack"] is None

    disagreement = onset_endpoint_disagreement(
        [[
            AcceptableOnset(1.0, 2.0),
            AcceptableOnset(1.5, 3.0),
            AcceptableOnset(None, None, True),
        ]]
    )
    assert disagreement["earliest_mean_absolute_disagreement_s"] == 0.5
    assert disagreement["latest_mean_absolute_disagreement_s"] == 1.0
    assert disagreement["wait_window_mismatches"] == 2
    assert disagreement["wait_window_mismatch_rate"] == pytest.approx(2 / 3)


def test_report_excludes_repeats_from_inter_rater_and_reports_strata() -> None:
    rows = _load(
        _raw("a1", annotator_id="one"),
        _raw("a2", annotator_id="two"),
        _raw("b1", annotator_id="one", action="respond", acceptable_onset={
            "earliest_elapsed_s": 3.0, "latest_elapsed_s": 4.0,
            "wait_for_user_resume": False,
        }),
        _raw("b2", annotator_id="two", action="respond", acceptable_onset={
            "earliest_elapsed_s": 3.0, "latest_elapsed_s": 4.0,
            "wait_for_user_resume": False,
        }),
        _raw("repeat", annotator_id="one"),
    )
    metadata = [
        {
            "assignment_id": row.assignment_id,
            "presentation_id": row.presentation_id,
            "annotator_id": row.annotator_id,
            "item_id": "short" if row.assignment_id in {"a1", "a2", "repeat"} else "long",
            "lineage_id": "lineage-1" if row.assignment_id in {"a1", "a2", "repeat"} else "lineage-2",
            "source": "corpus-a" if row.assignment_id in {"a1", "a2", "repeat"} else "corpus-b",
            "checkpoint_elapsed_s": 0.48 if row.assignment_id in {"a1", "a2", "repeat"} else 3.0,
            "repeat": row.assignment_id == "repeat",
        }
        for row in rows
    ]
    report = assigned_agreement_report(rows, metadata, n_boot=20, seed=7)
    assert report["overall"]["annotations"] == 4
    assert report["overall"]["items"] == 2
    assert report["repeat_consistency"]["action_consistency"] == 1.0
    assert set(report["by_corpus"]) == {"corpus-a", "corpus-b"}
    assert report["by_elapsed_stratum"]["<1s"]["items"] == 1
    assert report["by_elapsed_stratum"][">=3s"]["items"] == 1
    assert report["overall"]["lineage_bootstrap_95_ci"][
        "nominal_krippendorff_alpha"
    ]["n_boot"] == 20
    assert elapsed_stratum_agreement_reports(rows, metadata)["1–<3s"]["items"] == 0


def test_grouped_lineage_bootstrap_is_seeded_and_parameterized() -> None:
    groups = {"l1": [1.0, 3.0], "l2": [5.0], "l3": [7.0]}
    statistic = lambda values: sum(values) / len(values)
    first = lineage_bootstrap_ci(groups, statistic, n_boot=25, seed=11)
    second = lineage_bootstrap_ci(groups, statistic, n_boot=25, seed=11)
    assert first == second
    assert first["estimate"] == 4.0
    assert first["n_boot"] == 25
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]
