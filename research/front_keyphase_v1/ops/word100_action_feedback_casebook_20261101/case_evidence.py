"""Read-only summaries of already saved independent hand evidence.

This module neither extracts observations nor computes differences or alignment.
Saved measurements, reference descriptions, correspondence context and scoped RGB
reviews remain separate. An attached word, position result or posterior cannot
alter the hand evidence. Source pointers are relative to the input hand record;
the caller supplies that record's external path/hash.
"""
from collections import Counter
from copy import deepcopy
import math


SCHEMA = "wis_action_case_evidence_v1"
GROUP_DIMENSIONS = {hand + suffix: dimensions for hand in ("L", "R")
                    for suffix, dimensions in (("_finger_angles14", 14),
                                               ("_hand_axis2d", 1),
                                               ("_hand_forearm_relative2d", 1))}
STATUSES = {"COMPARED", "PARTIAL_SHARED_SUPPORT", "NOT_COMPARABLE"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pointer_token(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def _provenance(value, label):
    _require(isinstance(value, dict), label + " source must be a mapping")
    _require(isinstance(value.get("path"), str) and bool(value["path"]), label + " source path missing")
    digest = value.get("sha256")
    _require(isinstance(digest, str) and len(digest) == 64 and
             all(c in "0123456789abcdefABCDEF" for c in digest), label + " source SHA256 missing")
    _require(isinstance(value.get("json_pointer"), str) and
             (value["json_pointer"] == "" or value["json_pointer"].startswith("/")),
             label + " source JSON pointer missing")


def _matrix(a, name, n, d):
    values = a[name]
    _require(isinstance(values, list) and len(values) == n and
             all(isinstance(row, list) and len(row) == d for row in values),
             name + " shape differs from saved feature/time opportunities")
    return values


def _axis(a, n, d, pointer):
    _require(a["possible_shape"] == [n, d], "Saved axis shape differs")
    labels = a["dimension_labels"]
    _require(len(labels) == d, "Dimension identity missing")
    for key in ("frame_id", "timestamp_us"):
        values = a[key]
        _require(len(values) == n and all(type(v) is int for v in values), "Original " + key + " missing")
        _require(all(x < y for x, y in zip(values, values[1:])), "Original " + key + " order differs")
    own = _matrix(a, "own_observed", n, d)
    shared = _matrix(a, "matched_support", n, d)
    mixed = _matrix(a, "mixed_counterpart_missing", n, d)
    status = _matrix(a, "cell_status", n, d)
    reason = _matrix(a, "cell_reason", n, d)
    own_reason = _matrix(a, "own_reason", n, d)
    means = _matrix(a, "mean_abs_deg", n, d)
    maxima = _matrix(a, "max_abs_deg", n, d)
    totals = dict(possible=n*d, own_observed=0, own_unobserved=0, matched_support=0,
                  direct_complete=0, partial_shared=0, not_comparable=0,
                  observed_without_shared_counterpart=0)
    dimensions = []
    for j in range(d):
        indices = dict(direct_complete=[], partial_shared=[], not_comparable=[], own_unobserved=[])
        for i in range(n):
            o, s, m = own[i][j], shared[i][j], mixed[i][j]
            _require(all(type(v) is bool for v in (o, s, m)), "Evidence masks must be Boolean")
            _require((not s or o) and (not m or s), "Shared/partial support contradicts own observation")
            expected = "PARTIAL_SHARED_SUPPORT" if m else ("COMPARED" if s else "NOT_COMPARABLE")
            _require(status[i][j] in STATUSES and status[i][j] == expected, "Saved cell status disagrees with support")
            if s:
                lo, hi = means[i][j], maxima[i][j]
                _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                             math.isfinite(v) for v in (lo, hi)) and 0 <= lo and 0 <= hi
                         and (lo <= hi or math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12)),
                         "Shared cell must retain finite nonnegative mean/max")
            else:
                _require(means[i][j] is None and maxima[i][j] is None, "Unsupported cell cannot have numeric zero")
            key = "partial_shared" if m else ("direct_complete" if s else "not_comparable")
            indices[key].append(i)
            if not o:
                indices["own_unobserved"].append(i)
            totals["own_observed"] += int(o)
            totals["own_unobserved"] += int(not o)
            totals["matched_support"] += int(s)
            totals[key] += 1
            totals["observed_without_shared_counterpart"] += int(o and not s)
        dimensions.append(dict(dimension=j, label=labels[j], possible=n,
                               counts={key: len(v) for key, v in indices.items()},
                               frame_indices_by_support=indices,
                               raw_cell_pointer_template=pointer + "/{field}/{frame_index}/" + str(j)))
    _require(a["counts"] == totals, "Saved axis denominator/count mismatch")
    _require(a["own_reason_counts"] == dict(Counter(v for row in own_reason for v in row)),
             "Saved observation reasons disagree")
    # Numeric arrays stay in the frozen record. The complete index partitions
    # retain every feature/time opportunity, including all unsupported cells.
    fields = ("mean_abs_deg", "max_abs_deg", "cell_status", "cell_reason", "own_reason",
              "own_observed", "matched_support", "mixed_counterpart_missing", "values_degrees",
              "aligned_link_count", "matched_link_count", "missing_counterpart_link_count",
              "counterpart_observed_link_count", "original_source_mask", "diagnostics")
    return dict(source_pointer=pointer, possible_shape=[n, d], dimension_labels=deepcopy(labels),
                counts=deepcopy(totals), own_reason_counts=deepcopy(a["own_reason_counts"]),
                cell_reason_counts=dict(Counter(v for row in reason for v in row)),
                unaligned_frame_indices=deepcopy(a["unaligned_frame_indices"]),
                frame_id_pointer=pointer + "/frame_id", timestamp_us_pointer=pointer + "/timestamp_us",
                raw_fields={key: pointer + "/" + key for key in fields}, dimensions=dimensions,
                all_opportunities_retained=True, numeric_values_recomputed=False)


def _location(evidence, axes, index):
    qi, ri = evidence["path"][index]
    return dict(link_index=index,
                query=dict(frame_index=qi, frame_id=axes["query"]["frame_id"][qi],
                           timestamp_us=axes["query"]["timestamp_us"][qi]),
                reference=dict(frame_index=ri, frame_id=axes["reference"]["frame_id"][ri],
                               timestamp_us=axes["reference"]["timestamp_us"][ri]))


def summarize(hand_record, description=None, correspondence=None, visual_review=None):
    """Join independently scoped saved evidence without a correctness decision.

    Optional wrappers carry ``source={path,sha256,json_pointer}``. Description
    cards and old group decisions are copied, never applied to new degree units.
    All peak indices resolve through ``source_path`` and ``source_axes``; the
    first-peak example is not a replacement for other tied peaks.
    """
    evidence = hand_record["evidence"]
    _require(evidence["schema"] == "wis_independent_hand_evidence_v1", "Unexpected hand evidence schema")
    _require(set(evidence["groups"]) == set(GROUP_DIMENSIONS), "All six independent hand groups are required")
    _require(hand_record.get("kind") in ("natural", "technical"), "Record kind must explicitly identify synthetic evidence")
    synthetic = hand_record["kind"] == "technical" or hand_record.get("technical_edit") is not None
    first_group = next(iter(evidence["groups"].values()))
    axes = {axis: {key: deepcopy(first_group[axis][key]) for key in ("frame_id", "timestamp_us")}
            for axis in ("query", "reference")}
    n, m = len(axes["query"]["frame_id"]), len(axes["reference"]["frame_id"])
    path = evidence["path"]
    _require(isinstance(path, list), "Saved path missing")
    for pair in path:
        _require(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair)
                 and 0 <= pair[0] < n and 0 <= pair[1] < m, "Saved path index outside original source axes")
    groups, observation_groups, difference_groups = {}, {}, {}
    for name, d in GROUP_DIMENSIONS.items():
        group = evidence["groups"][name]
        pointer = "/evidence/groups/" + _pointer_token(name)
        _require(group["units"] == "degrees", "Do not mix saved units")
        summaries = group["summary_by_dimension"]
        _require(len(summaries) == d, "Every original dimension must remain")
        out = {key: deepcopy(group[key]) for key in ("definition", "source_columns", "source_feature_names", "units", "circular")}
        out["source_pointer"] = pointer
        for axis, length in (("query", n), ("reference", m)):
            a = group[axis]
            _require(all(a[key] == axes[axis][key] for key in ("frame_id", "timestamp_us")), "Group source axes differ")
            out[axis] = _axis(a, length, d, pointer + "/" + axis)
        shared = _matrix(group["links"], "shared_support", len(path), d)
        absolute = _matrix(group["links"], "absolute_difference_deg", len(path), d)
        out["summary_by_dimension"] = []
        for j, saved in enumerate(summaries):
            _require(saved["dimension"] == j and saved["label"] == group["query"]["dimension_labels"][j]
                     == group["reference"]["dimension_labels"][j], "Saved dimension identity differs")
            count = sum(row[j] for row in shared)
            _require(saved["possible_links"] == len(path) and saved["shared_links"] == count, "Saved link denominator differs")
            lo, hi = saved["mean_abs_deg"], saved["max_abs_deg"]
            peaks = saved["peak_link_indices"]
            if count:
                _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                             for v in (lo, hi)) and 0 <= lo and 0 <= hi
                         and (lo <= hi or math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12)),
                         "Saved link mean/max invalid")
                _require(peaks == [k for k in range(len(path)) if shared[k][j] and absolute[k][j] == hi],
                         "Every tied peak must be retained")
                _require(bool(peaks), "Shared evidence has no peak")
            else:
                _require(lo is None and hi is None and peaks == [], "Unknown links cannot become numeric zero")
            summary = deepcopy(saved)
            summary.update(source_pointer=pointer + "/summary_by_dimension/" + str(j),
                           measurement_status="SHARED_MEASUREMENTS" if count else "NO_SHARED_OBSERVATION",
                           first_peak=_location(evidence, axes, peaks[0]) if peaks else None,
                           peak_source_mapping=dict(all_peak_indices_field="peak_link_indices",
                               summary_path_pointer="/source_path", summary_axes_pointer="/source_axes",
                               original_path_pointer="/evidence/path", dimension=j,
                               original_absolute_difference_pointer=pointer + "/links/absolute_difference_deg"),
                           all_tied_peaks_retained=True, numerical_difference_is_educational_error=False)
            out["summary_by_dimension"].append(summary)
        out["signed_and_absolute_link_values_pointer"] = pointer + "/links"
        out["scalar_agreement"] = None
        groups[name] = out
        observation_groups[name] = {axis: deepcopy(out[axis]["counts"]) for axis in ("query", "reference")}
        observation_groups[name]["summary_pointer"] = "/groups/" + _pointer_token(name)
        difference_groups[name] = dict(units=group["units"], circular=group["circular"],
            summary_pointer="/groups/" + _pointer_token(name) + "/summary_by_dimension",
            original_source_pointer=pointer, no_shared_dimensions=[s["dimension"] for s in summaries if not s["shared_links"]])
    if description is not None:
        _provenance(description.get("source"), "Description")
        _require(isinstance(description.get("cards"), list), "Original source cards must be a list")
    if correspondence is not None:
        _provenance(correspondence.get("source"), "Correspondence")
    if visual_review is not None:
        _provenance(visual_review.get("source"), "Visual review")
        _require(visual_review.get("status") == "REVIEWED" and bool(visual_review.get("windows"))
                 and bool(visual_review.get("observations")), "REVIEWED requires original scoped windows and observations")
        _require(not synthetic, "Synthetic feature edits do not have a corresponding performed RGB video")
    cards = description["cards"] if description is not None else []
    return dict(schema=SCHEMA, record_id=hand_record.get("id"), source_record_pointer="",
        identity={key: deepcopy(hand_record.get(key)) for key in ("kind", "query", "reference", "parent_id", "query_range", "reference_range")},
        synthetic=synthetic, evidence_role="SYNTHETIC_FEATURE_EDIT" if synthetic else "PUBLIC_NORMAL_DEVELOPMENT_OBSERVATION",
        technical_edit=deepcopy(hand_record.get("technical_edit")),
        source_axes=axes, source_path=deepcopy(path), groups=groups,
        observation_support=dict(status="PER_FEATURE_CELL", groups=observation_groups,
            partial_and_unknown_preserved=True, source_of_support="Saved runtime input masks and derived validity; not word or guide labels"),
        correspondence_support=dict(status="SAVED_PATH_NOT_SEMANTICALLY_VERIFIED",
            source_path_pointer="/evidence/path", path_full_scope=evidence["path_full_scope"],
            original_path_retained=True, alignment_verified=False, observed_action_order_verified=False,
            provided=deepcopy(correspondence), uncertainty_does_not_erase_measurements=True),
        measurement_difference=dict(status="SAVED_NUMERIC_DIFFERENCES_ONLY", groups=difference_groups,
            aggregation="Existing per-dimension link means and maxima; no new pooled score",
            scalar_difference=None, educational_error=None, zero_is_correct_performance=False,
            position_or_posterior_used_for_gating=False, threshold_applied=False),
        visible_difference_review=dict(status="REVIEWED_SCOPED_OBSERVATIONS" if visual_review is not None else "NOT_REVIEWED",
            provided=deepcopy(visual_review), inferred_from_measurements=False,
            review_applies_to_whole_parent=False, review_is_training_truth=False),
        description_support=dict(status="REFERENCE_SOURCE_CONTEXT" if cards else ("PHYSICAL_CONTEXT_ONLY" if description else "NOT_PROVIDED"),
            source_role="Reference expectations and historical diagnostics, not query observations",
            cards_count=len(cards), source_status_counts=dict(Counter(card.get("source_status", "NOT_SPECIFIED") for card in cards)),
            provided=deepcopy(description), full_span_semantic_verified=False,
            legacy_decisions_applied_to_new_measurements=False, source_links_and_qualifiers_preserved=True),
        linguistic_importance=dict(status="UNESTABLISHED", important_difference=None,
            allowable_difference=None, required_feature=None, educational_error=None),
        source_role=dict(measurements="SAVED_OBSERVATIONS_OR_SYNTHETIC_EDITS",
            descriptions="REFERENCE_EXPECTATION_CONTEXT", visual_review="SCOPED_REVIEW_ONLY", word="REFERENCE_ROUTING_ONLY"),
        semantic_verdict=None, whole_sign_verdict=None, correction_advice=None,
        missing_values_imputed=False, new_comparison=False, new_alignment=False, new_inference=False,
        input_records_unchanged=True, all_groups_and_dimensions_retained=True,
        limits=deepcopy(evidence.get("limits", [])) + [
            "Measured differences and source descriptions do not establish educational importance or correctness.",
            "Source cards and RGB reviews apply only to their stated anchors/windows, not an entire parent.",
            "The original path is a correspondence hypothesis; monotonicity is not proof of performed action order."])
