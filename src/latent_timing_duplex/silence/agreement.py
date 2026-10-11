"""Dependency-free agreement metrics for assigned silence responses."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any, TypeVar

from .types import AcceptableOnset, AssignedSilenceAnnotation, SilenceAction

RATABLE_ACTIONS = (
    SilenceAction.WAIT,
    SilenceAction.ACK,
    SilenceAction.RESPOND,
    SilenceAction.REPAIR,
)
_T = TypeVar("_T")


def nominal_krippendorff_alpha(
    item_ratings: Mapping[Hashable, Sequence[Any]] | Iterable[Sequence[Any]],
) -> float | None:
    """Nominal Krippendorff alpha; ``UNRATABLE`` and ``None`` are missing."""
    coincidences: dict[SilenceAction, float] = {}
    observed_disagreement = 0.0
    total = 0
    for group in _rating_groups(item_ratings):
        ratings = [action for value in group if (action := _action(value)) is not None]
        size = len(ratings)
        if size < 2:
            continue
        total += size
        counts = {action: ratings.count(action) for action in set(ratings)}
        observed_disagreement += (
            size * size - sum(count * count for count in counts.values())
        ) / (size - 1)
        for action, count in counts.items():
            coincidences[action] = coincidences.get(action, 0.0) + count
    if total < 2:
        return None
    observed_disagreement /= total
    expected_disagreement = (
        total * total - sum(count * count for count in coincidences.values())
    ) / (total * (total - 1))
    if expected_disagreement == 0.0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


# Conventional spelling retained as an equally stable public alias.
krippendorff_alpha_nominal = nominal_krippendorff_alpha


def pairwise_confusion(
    item_ratings: Mapping[Hashable, Sequence[Any]] | Iterable[Sequence[Any]],
    *,
    classes: Sequence[SilenceAction | str] = RATABLE_ACTIONS,
) -> dict[str, dict[str, int]]:
    """Symmetric ordered-pair confusion counts, excluding missing ratings."""
    actions = tuple(_required_action(value) for value in classes)
    result = {left.value: {right.value: 0 for right in actions} for left in actions}
    allowed = set(actions)
    for group in _rating_groups(item_ratings):
        ratings = [action for value in group if (action := _action(value)) in allowed]
        for left_index, left in enumerate(ratings):
            for right_index, right in enumerate(ratings):
                if left_index != right_index:
                    result[left.value][right.value] += 1
    return result


def per_class_positive_agreement(
    item_ratings: Mapping[Hashable, Sequence[Any]] | Iterable[Sequence[Any]],
    *,
    classes: Sequence[SilenceAction | str] = RATABLE_ACTIONS,
) -> dict[str, float | None]:
    """Pairwise positive agreement for each ratable action."""
    matrix = pairwise_confusion(item_ratings, classes=classes)
    result: dict[str, float | None] = {}
    for label, row in matrix.items():
        diagonal = row[label]
        disagreements = sum(value for other, value in row.items() if other != label)
        disagreements += sum(
            other_row[label]
            for other, other_row in matrix.items()
            if other != label
        )
        denominator = 2 * diagonal + disagreements
        result[label] = None if denominator == 0 else 2 * diagonal / denominator
    return result


def onset_endpoint_disagreement(
    item_annotations: Mapping[Hashable, Sequence[Any]] | Iterable[Sequence[Any]],
) -> dict[str, float | int | None]:
    """Mean finite endpoint disagreement and explicit wait/window mismatch."""
    earliest: list[float] = []
    latest: list[float] = []
    wait_window_mismatches = 0
    compared_pairs = 0
    finite_pairs = 0
    for group in _rating_groups(item_annotations):
        onsets = [onset for value in group if (onset := _onset(value)) is not None]
        for left_index, left in enumerate(onsets):
            for right in onsets[left_index + 1 :]:
                compared_pairs += 1
                if left.wait_for_user_resume != right.wait_for_user_resume:
                    wait_window_mismatches += 1
                elif not left.wait_for_user_resume:
                    finite_pairs += 1
                    earliest.append(abs(left.earliest_elapsed_s - right.earliest_elapsed_s))  # type: ignore[operator]
                    latest.append(abs(left.latest_elapsed_s - right.latest_elapsed_s))  # type: ignore[operator]
    return {
        "compared_pairs": compared_pairs,
        "finite_pairs": finite_pairs,
        "earliest_mean_absolute_disagreement_s": _mean(earliest),
        "latest_mean_absolute_disagreement_s": _mean(latest),
        "wait_window_mismatches": wait_window_mismatches,
        "wait_window_mismatch_rate": (
            None if compared_pairs == 0 else wait_window_mismatches / compared_pairs
        ),
    }


def unratable_rate(annotations: Iterable[Any]) -> float | None:
    actions = [_raw_action(value) for value in annotations]
    return None if not actions else sum(action is SilenceAction.UNRATABLE for action in actions) / len(actions)


def repeat_consistency(
    annotations: Iterable[AssignedSilenceAnnotation],
    assignment_map: Iterable[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    """Action consistency for repeat/original pairs from the same annotator."""
    joined = _join(annotations, assignment_map)
    originals = {
        (row.annotation.annotator_id, row.item_id): row.annotation
        for row in joined
        if not row.repeat
    }
    pairs = [
        (originals[(row.annotation.annotator_id, row.item_id)], row.annotation)
        for row in joined
        if row.repeat and (row.annotation.annotator_id, row.item_id) in originals
    ]
    agreements = sum(left.action is right.action for left, right in pairs)
    repeat_count = sum(row.repeat for row in joined)
    return {
        "repeat_assignments": repeat_count,
        "paired_repeats": len(pairs),
        "action_agreements": agreements,
        "action_consistency": None if not pairs else agreements / len(pairs),
        "unpaired_repeats": repeat_count - len(pairs),
    }


def lineage_bootstrap_ci(
    groups_by_lineage: Mapping[Hashable, Sequence[_T]],
    statistic: Callable[[list[_T]], float | None],
    *,
    n_boot: int = 10_000,
    seed: int = 20_261_009,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Percentile CI from resampled lineage groups, never individual rows."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    lineages = sorted(groups_by_lineage, key=repr)
    if len(lineages) < 2:
        raise ValueError("lineage bootstrap needs at least two lineages")
    observed = statistic(list(chain.from_iterable(groups_by_lineage[key] for key in lineages)))
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sampled = [rng.choice(lineages) for _ in lineages]
        value = statistic(list(chain.from_iterable(groups_by_lineage[key] for key in sampled)))
        if value is not None and math.isfinite(value):
            values.append(float(value))
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": observed,
        "ci_low": _quantile(values, tail),
        "ci_high": _quantile(values, 1.0 - tail),
        "n_boot": n_boot,
        "valid_replicates": len(values),
        "seed": seed,
    }


def corpus_agreement_reports(
    annotations: Iterable[AssignedSilenceAnnotation],
    assignment_map: Iterable[Mapping[str, Any]],
    *,
    n_boot: int = 10_000,
    seed: int = 20_261_009,
) -> dict[str, dict[str, Any]]:
    rows = [row for row in _join(annotations, assignment_map) if not row.repeat]
    return {
        corpus: _report_with_bootstrap(
            [row for row in rows if row.corpus == corpus], n_boot=n_boot, seed=seed
        )
        for corpus in sorted({row.corpus for row in rows})
    }


def elapsed_stratum_agreement_reports(
    annotations: Iterable[AssignedSilenceAnnotation],
    assignment_map: Iterable[Mapping[str, Any]],
    *,
    boundaries_s: Sequence[float] = (1.0, 3.0),
    labels: Sequence[str] = ("<1s", "1–<3s", ">=3s"),
    n_boot: int = 10_000,
    seed: int = 20_261_009,
) -> dict[str, dict[str, Any]]:
    rows = [row for row in _join(annotations, assignment_map) if not row.repeat]
    _validate_strata(boundaries_s, labels)
    return {
        label: _report_with_bootstrap(
            [row for row in rows if _elapsed_stratum(row.elapsed_s, boundaries_s, labels) == label],
            n_boot=n_boot,
            seed=seed,
        )
        for label in labels
    }


def assigned_agreement_report(
    annotations: Iterable[AssignedSilenceAnnotation],
    assignment_map: Iterable[Mapping[str, Any]],
    *,
    boundaries_s: Sequence[float] = (1.0, 3.0),
    stratum_labels: Sequence[str] = ("<1s", "1–<3s", ">=3s"),
    n_boot: int = 10_000,
    seed: int = 20_261_009,
) -> dict[str, Any]:
    """Locked pilot report: originals inter-rater, repeats intra-rater."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    annotations = list(annotations)
    assignment_map = list(assignment_map)
    joined = _join(annotations, assignment_map)
    originals = [row for row in joined if not row.repeat]
    _validate_strata(boundaries_s, stratum_labels)
    overall = _inter_report(originals)
    overall["lineage_bootstrap_95_ci"] = _agreement_bootstrap(
        originals, n_boot=n_boot, seed=seed
    )
    by_corpus = {
        corpus: _report_with_bootstrap(
            [row for row in originals if row.corpus == corpus], n_boot=n_boot, seed=seed
        )
        for corpus in sorted({row.corpus for row in originals})
    }
    by_stratum = {
        label: _report_with_bootstrap(
            [
                row for row in originals
                if _elapsed_stratum(row.elapsed_s, boundaries_s, stratum_labels) == label
            ],
            n_boot=n_boot,
            seed=seed,
        )
        for label in stratum_labels
    }
    return {
        "overall": overall,
        "repeat_consistency": repeat_consistency(annotations, assignment_map),
        "by_corpus": by_corpus,
        "by_elapsed_stratum": by_stratum,
        "bootstrap": {
            "unit": "lineage", "replicates": n_boot,
            "requested_replicates": n_boot, "seed": seed,
        },
    }


pilot_agreement_report = assigned_agreement_report


@dataclass(frozen=True)
class _Joined:
    annotation: AssignedSilenceAnnotation
    item_id: str
    lineage_id: str
    corpus: str
    elapsed_s: float
    repeat: bool


def _join(
    annotations: Iterable[AssignedSilenceAnnotation],
    assignment_map: Iterable[Mapping[str, Any]],
) -> list[_Joined]:
    metadata: dict[str, Mapping[str, Any]] = {}
    for number, row in enumerate(assignment_map, 1):
        assignment = row.get("assignment_id")
        if not isinstance(assignment, str) or not assignment or assignment in metadata:
            raise ValueError(f"assignment map row {number} has invalid or duplicate assignment_id")
        metadata[assignment] = row
    result: list[_Joined] = []
    seen: set[str] = set()
    for annotation in annotations:
        if annotation.assignment_id in seen:
            raise ValueError(f"duplicate annotation assignment {annotation.assignment_id!r}")
        seen.add(annotation.assignment_id)
        try:
            row = metadata[annotation.assignment_id]
        except KeyError as exc:
            raise ValueError(f"unknown annotation assignment {annotation.assignment_id!r}") from exc
        for field, expected in (
            ("presentation_id", annotation.presentation_id),
            ("annotator_id", annotation.annotator_id),
        ):
            if row.get(field) != expected:
                raise ValueError(f"assignment {annotation.assignment_id!r} has mismatched {field}")
        item, lineage = row.get("item_id"), row.get("lineage_id")
        corpus = row.get("source", row.get("corpus"))
        repeat = row.get("repeat")
        elapsed = row.get("checkpoint_elapsed_s", row.get("elapsed_s"))
        if any(not isinstance(value, str) or not value for value in (item, lineage, corpus)):
            raise ValueError("assignment map requires item_id, lineage_id, and source")
        if not isinstance(repeat, bool):
            raise ValueError("assignment map repeat must be a boolean")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("assignment map requires finite non-negative checkpoint_elapsed_s")
        onset = annotation.acceptable_onset
        if (
            onset is not None
            and onset.earliest_elapsed_s is not None
            and onset.earliest_elapsed_s + 1e-9 < elapsed
        ):
            raise ValueError("acceptable onset cannot precede the assignment checkpoint")
        result.append(_Joined(annotation, item, lineage, corpus, float(elapsed), repeat))
    return result


def _inter_report(rows: Sequence[_Joined]) -> dict[str, Any]:
    grouped = _group_items(rows)
    ratings = [tuple(row.annotation for row in group) for group in grouped]
    return {
        "items": len(grouped),
        "annotations": len(rows),
        "nominal_krippendorff_alpha": nominal_krippendorff_alpha(ratings),
        "pairwise_confusion": pairwise_confusion(ratings),
        "per_class_positive_agreement": per_class_positive_agreement(ratings),
        "onset_endpoint_disagreement": onset_endpoint_disagreement(ratings),
        "unratable_rate": unratable_rate(row.annotation for row in rows),
    }


def _report_with_bootstrap(
    rows: Sequence[_Joined], *, n_boot: int, seed: int
) -> dict[str, Any]:
    report = _inter_report(rows)
    if len({row.lineage_id for row in rows}) >= 2:
        report["lineage_bootstrap_95_ci"] = _agreement_bootstrap(
            rows, n_boot=n_boot, seed=seed
        )
    return report


def _agreement_bootstrap(
    rows: Sequence[_Joined], *, n_boot: int, seed: int
) -> dict[str, dict[str, float | int | None]]:
    item_groups = _group_items(rows)
    by_lineage: dict[str, list[tuple[AssignedSilenceAnnotation, ...]]] = {}
    for group in item_groups:
        by_lineage.setdefault(group[0].lineage_id, []).append(
            tuple(row.annotation for row in group)
        )
    if len(by_lineage) < 2:
        return {}
    statistics: dict[str, Callable[[list[tuple[AssignedSilenceAnnotation, ...]]], float | None]] = {
        "nominal_krippendorff_alpha": nominal_krippendorff_alpha,
        "unratable_rate": lambda groups: unratable_rate(chain.from_iterable(groups)),
        "onset_earliest_mean_absolute_disagreement_s": lambda groups: onset_endpoint_disagreement(groups)["earliest_mean_absolute_disagreement_s"],  # type: ignore[return-value]
        "onset_latest_mean_absolute_disagreement_s": lambda groups: onset_endpoint_disagreement(groups)["latest_mean_absolute_disagreement_s"],  # type: ignore[return-value]
        "onset_wait_window_mismatch_rate": lambda groups: onset_endpoint_disagreement(groups)["wait_window_mismatch_rate"],  # type: ignore[return-value]
    }
    for action in RATABLE_ACTIONS:
        statistics[f"positive_agreement_{action.value}"] = (
            lambda groups, action=action: per_class_positive_agreement(groups)[action.value]
        )
    return {
        name: lineage_bootstrap_ci(
            by_lineage, statistic, n_boot=n_boot, seed=seed, confidence=0.95
        )
        for name, statistic in statistics.items()
    }


def _group_items(rows: Sequence[_Joined]) -> list[list[_Joined]]:
    grouped: dict[str, list[_Joined]] = {}
    for row in rows:
        group = grouped.setdefault(row.item_id, [])
        if group and any(
            (existing.lineage_id, existing.corpus, existing.elapsed_s)
            != (row.lineage_id, row.corpus, row.elapsed_s)
            for existing in group
        ):
            raise ValueError(f"item {row.item_id!r} has inconsistent assignment metadata")
        if any(existing.annotation.annotator_id == row.annotation.annotator_id for existing in group):
            raise ValueError(f"item {row.item_id!r} has duplicate original annotator ratings")
        group.append(row)
    return [grouped[item] for item in sorted(grouped)]


def _rating_groups(
    values: Mapping[Hashable, Sequence[Any]] | Iterable[Sequence[Any]],
) -> Iterable[Sequence[Any]]:
    return values.values() if isinstance(values, Mapping) else values


def _raw_action(value: Any) -> SilenceAction:
    raw = value.action if hasattr(value, "action") else value
    try:
        return raw if isinstance(raw, SilenceAction) else SilenceAction(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid silence action {raw!r}") from exc


def _action(value: Any) -> SilenceAction | None:
    if value is None:
        return None
    action = _raw_action(value)
    return None if action is SilenceAction.UNRATABLE else action


def _required_action(value: SilenceAction | str) -> SilenceAction:
    action = _raw_action(value)
    if action is SilenceAction.UNRATABLE:
        raise ValueError("UNRATABLE is missing, not an agreement class")
    return action


def _onset(value: Any) -> AcceptableOnset | None:
    onset = value.acceptable_onset if hasattr(value, "acceptable_onset") else value
    if onset is None:
        return None
    if not isinstance(onset, AcceptableOnset):
        raise ValueError("onset agreement requires AcceptableOnset values or annotations")
    return onset


def _validate_strata(boundaries: Sequence[float], labels: Sequence[str]) -> None:
    if len(labels) != len(boundaries) + 1:
        raise ValueError("stratum labels must have one more value than boundaries")
    if any(isinstance(value, bool) or not math.isfinite(value) for value in boundaries) or any(
        right <= left for left, right in zip(boundaries, boundaries[1:])
    ):
        raise ValueError("stratum boundaries must be finite and increasing")


def _elapsed_stratum(elapsed: float, boundaries: Sequence[float], labels: Sequence[str]) -> str:
    for boundary, label in zip(boundaries, labels):
        if elapsed < boundary:
            return label
    return labels[-1]


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    if lower == len(ordered) - 1:
        return ordered[lower]
    return ordered[lower] * (1.0 - fraction) + ordered[lower + 1] * fraction
