"""Offline replay utility: do personal research references help, before enrollment?

Scope
-----
This module is an OFFLINE evaluation aid only. It never fits, calibrates,
saves, enrolls or promotes anything, performs no file/network IO, and applies
no thresholds. It NEVER certifies sign quality: candidate captures are
user-requested positives of unverified quality, not certified signs and not
enrollment-eligible.

The caller freezes its existing DTW / ML / hybrid pair models, supplies the
already-computed baseline outcome plus the pair scores of the extra research
references, and this module reproduces the classifier's reference-bank
aggregation (min finite DTW distance, max finite ML/hybrid probability).
"""

from copy import deepcopy
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

METHODS = ("dtw_only", "ml_only", "hybrid")

_HEXDIGITS = set("0123456789abcdefABCDEF")
_REQUIRED_PARTITION = "development"
_REQUIRED_QUALITY_BASIS = "user_requested_positive_only"
_EXCLUSION_REASON = "QUERY_ID_OR_SOURCE_HASH"
_STATUS_RANKED = "RANKED"
_STATUS_TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


# --------------------------------------------------------------------------- #
# shared validation helpers
# --------------------------------------------------------------------------- #
def _check_vocabulary(vocabulary: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(vocabulary, (str, bytes)) or not isinstance(vocabulary, Sequence):
        raise ValueError("vocabulary must be a sequence of label strings")
    words = tuple(vocabulary)
    if not words:
        raise ValueError("vocabulary must be non-empty")
    for word in words:
        if not isinstance(word, str) or not word:
            raise ValueError("vocabulary labels must be non-empty strings")
    if len(set(words)) != len(words):
        raise ValueError("vocabulary labels must be unique")
    return words


def _check_sha256(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEXDIGITS for c in value):
        raise ValueError("%s must be exactly 64 hex characters" % what)
    return value.lower()


def _check_score(value: Any, method: str, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a real number" % what)
    score = float(value)
    if not isfinite(score):
        raise ValueError("%s must be finite" % what)
    if method == "dtw_only":
        if score < 0.0:
            raise ValueError("%s (DTW distance) must be >= 0" % what)
    elif not 0.0 <= score <= 1.0:
        raise ValueError("%s (probability) must lie in [0, 1]" % what)
    return score


def _check_failed_pairs(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("failed_pairs must be a non-negative int (bool is not accepted)")
    return value


# --------------------------------------------------------------------------- #
# reference selection
# --------------------------------------------------------------------------- #
def _check_candidate(candidate: Any, words: Tuple[str, ...], index: int) -> Tuple[str, str]:
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate %d must be a Mapping" % index)
    sample_id = candidate.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("candidate %d: sample_id must be a non-empty string" % index)
    source_sha = _check_sha256(candidate.get("source_sha256"), "candidate %d: source_sha256" % index)
    if candidate.get("label") not in words:
        raise ValueError("candidate %s: label is outside the vocabulary" % sample_id)
    if candidate.get("partition") != _REQUIRED_PARTITION:
        raise ValueError("candidate %s: partition must be 'development' (final data is rejected)" % sample_id)
    if candidate.get("positive") is not True:
        raise ValueError("candidate %s: positive must be exactly True" % sample_id)
    if candidate.get("quality_basis") != _REQUIRED_QUALITY_BASIS:
        raise ValueError("candidate %s: quality_basis must be 'user_requested_positive_only'" % sample_id)
    if "seq" not in candidate:
        raise ValueError("candidate %s: opaque seq is missing" % sample_id)
    return sample_id, source_sha


def select_research_references(
    candidates: Iterable[Mapping[str, Any]],
    vocabulary: Sequence[str],
    query_sample_id: str,
    query_source_sha256: str,
) -> Tuple[Tuple[Mapping[str, Any], ...], Dict[str, Any]]:
    """Pick leave-one-out personal research references for one query capture.

    Selection is metadata-only: no target label, no score, no threshold and no
    guess about capture quality is involved. The whole candidate bank is
    validated *before* any filtering, so a malformed or duplicated entry is an
    error even when it would have been excluded anyway.

    Returns ``(selected, exclusion)`` where ``selected`` holds the original
    candidate mappings (not copied, never mutated) and ``exclusion`` reports
    the excluded sample IDs plus counts.
    """
    words = _check_vocabulary(vocabulary)
    if not isinstance(query_sample_id, str) or not query_sample_id:
        raise ValueError("query_sample_id must be a non-empty string")
    query_sha = _check_sha256(query_source_sha256, "query_source_sha256")

    bank: List[Tuple[Mapping[str, Any], str, str]] = []
    seen_ids: set = set()
    seen_shas: set = set()
    for index, candidate in enumerate(candidates):
        sample_id, source_sha = _check_candidate(candidate, words, index)
        if sample_id in seen_ids:
            raise ValueError("duplicate sample_id in candidate bank: %s" % sample_id)
        if source_sha in seen_shas:
            raise ValueError("duplicate source_sha256 in candidate bank (alias of %s)" % sample_id)
        seen_ids.add(sample_id)
        seen_shas.add(source_sha)
        bank.append((candidate, sample_id, source_sha))

    selected: List[Mapping[str, Any]] = []
    excluded: List[str] = []
    for candidate, sample_id, source_sha in bank:
        if sample_id == query_sample_id or source_sha == query_sha:
            excluded.append(sample_id)
        else:
            selected.append(candidate)

    exclusion = {
        "excluded_sample_ids": excluded,
        "reason": _EXCLUSION_REASON,
        "candidates_seen": len(bank),
        "selected_count": len(selected),
        "excluded_count": len(excluded),
    }
    return tuple(selected), exclusion


# --------------------------------------------------------------------------- #
# outcome augmentation
# --------------------------------------------------------------------------- #
def augment_outcome(
    baseline_outcome: Mapping[str, Any],
    pair_results: Iterable[Mapping[str, Any]],
    method: str,
    vocabulary: Sequence[str],
) -> Dict[str, Any]:
    """Fold extra reference pair scores into a frozen baseline outcome.

    Aggregation mirrors the existing classifier's reference bank: a class score
    is the minimum finite DTW distance or the maximum finite ML/hybrid
    probability over that class's pairs. Nothing is calibrated, no threshold is
    applied and the query's declared target is never consulted.

    Limitation: a ``TECHNICAL_FAILURE`` baseline is returned as a deep copy,
    unchanged (reason preserved). Personalization is not used to rescue a
    technical failure, because the failure means the query itself produced no
    trustworthy scores.
    """
    words = _check_vocabulary(vocabulary)
    if method not in METHODS:
        raise ValueError("method must be one of %r" % (METHODS,))
    if not isinstance(baseline_outcome, Mapping):
        raise ValueError("baseline_outcome must be a Mapping")

    status = baseline_outcome.get("status")
    if status == _STATUS_TECHNICAL_FAILURE:
        return deepcopy(dict(baseline_outcome))
    if status != _STATUS_RANKED:
        raise ValueError("baseline status must be 'RANKED' or 'TECHNICAL_FAILURE'")

    raw = baseline_outcome.get("raw_class_scores")
    if not isinstance(raw, Mapping) or set(raw) != set(words):
        raise ValueError("baseline raw_class_scores keys must match the vocabulary exactly")
    scores = {w: _check_score(raw[w], method, "baseline score for %r" % w) for w in words}
    failed_pairs = _check_failed_pairs(baseline_outcome.get("failed_pairs"))

    better = min if method == "dtw_only" else max
    for index, pair in enumerate(pair_results):
        if not isinstance(pair, Mapping):
            raise ValueError("pair_result %d must be a Mapping" % index)
        label = pair.get("label")
        if label not in words:
            raise ValueError("pair_result %d: label is outside the vocabulary" % index)
        ok = pair.get("ok")
        if not isinstance(ok, bool):
            raise ValueError("pair_result %d: ok must be a bool" % index)
        if not ok:
            failed_pairs += 1
            continue
        value = _check_score(pair.get("value"), method, "pair_result %d value" % index)
        scores[label] = better(scores[label], value)

    ranking = sorted(words, key=lambda w: (scores[w] if method == "dtw_only" else -scores[w], w))
    top1 = ranking[0]
    top1_score = scores[top1]
    margin = None if len(ranking) == 1 else abs(top1_score - scores[ranking[1]])

    return {
        "status": _STATUS_RANKED,
        "top1": top1,
        "score": top1_score,
        "q": -top1_score if method == "dtw_only" else top1_score,
        "margin": margin,
        "raw_class_scores": dict(scores),
        "failed_pairs": failed_pairs,
    }
