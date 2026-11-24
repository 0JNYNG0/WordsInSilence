"""Evaluation-only physical-draft joins on fixed reference frames.

Public API: evaluate(query_alias, reference_alias, saved_or_new_path) -> JSON.
aggregate(results) requires exactly the fixed12 pairs of one condition.

Do not import this module into a matcher or pass its outputs to path search.
It reads saved evaluation drafts, never signals, model outputs or RGB, and
does not rerun or evaluate any archived path by itself.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from pathlib import Path

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
SCOPE_FILE = OPS / "word100_action_feedback_casebook_20261101/INPUTS.json"
PHYSICAL_FILE = OPS / "word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json"
PHASE_FILE = OPS / "word100_core_stage_evidence_20261031/PHASE_DATA.json"
EXPECTED_PAIRS = 12
EXPECTED_PARENTS = 39
EXPECTED_REFERENCE_FRAMES = 440
EXPECTED_PHYSICAL_CERTAIN = 360
EXPECTED_REFERENCE_BOUNDARY_OR_UNKNOWN = 80


def _read(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), {
        "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()
    }


@lru_cache(maxsize=1)
def _sources():
    case, scope_pin = _read(SCOPE_FILE)
    physical, physical_pin = _read(PHYSICAL_FILE)
    phase, phase_pin = _read(PHASE_FILE)
    parents = case["natural"]
    pairs = {(r["query"], r["reference"]) for r in parents}
    if len(parents) != EXPECTED_PARENTS or len(pairs) != EXPECTED_PAIRS:
        raise ValueError("Saved fixed39/12 evaluation scope changed")
    pc = {c["alias"]: c for c in physical["clips"]}
    bc = {c["alias"]: c for c in phase["clips"]}
    frame_maps = {}
    for alias, clip in pc.items():
        by_frame = {}
        for region in clip["regions"]:
            for frame in range(region["outer"][0], region["outer"][1]+1):
                if frame in by_frame:
                    raise ValueError("Overlapping physical draft regions")
                by_frame[frame] = region
        frame_maps[alias] = by_frame
    total = certain = 0
    for p in parents:
        for ri in range(p["reference_range"][0], p["reference_range"][1]+1):
            region = frame_maps[p["reference"]].get(ri)
            total += 1
            certain += int(bool(region and region.get("kind")))
    if (total, certain, total-certain) != (
        EXPECTED_REFERENCE_FRAMES, EXPECTED_PHYSICAL_CERTAIN,
        EXPECTED_REFERENCE_BOUNDARY_OR_UNKNOWN
    ):
        raise ValueError("Frozen reference440/certain360/unknown80 denominator changed")
    return {
        "parents": parents, "pairs": pairs, "physical": pc, "phase": bc,
        "frame_maps": frame_maps, "kinds": physical["kinds"],
        "pins": [scope_pin, physical_pin, phase_pin],
    }


def _physical(alias, frame, sources):
    r = sources["frame_maps"][alias].get(frame)
    if r is None:
        return {
            "region_id": None, "region_type": "outside_saved_reviewed_region",
            "kind": None, "physical_family": None, "occurrence": None,
            "motion_subphase": None,
        }
    kind = r.get("kind")
    family = sources["kinds"].get(kind, {}).get("physical_kind", kind) if kind else None
    return {
        "region_id": r["id"], "region_type": r["region_type"], "kind": kind,
        "physical_family": family, "occurrence": r.get("occurrence"),
        "motion_subphase": r.get("motion_subphase"),
    }


def _broad_phase(alias, frame, sources):
    matches = [
        {"segment_id": s["segment_id"], "role": s["role"], "core": s["core"]}
        for s in sources["phase"][alias]["segments"]
        if s["core"][0] <= frame <= s["core"][1]
    ]
    return {
        "memberships": matches,
        "unknown_unannotated_gap": not bool(matches),
        "linguistic_role_confirmed": False,
    }


def _path(path, n, m):
    output = []
    for item in path:
        if len(item) != 2 or any(isinstance(x, bool) or not isinstance(x, Integral) for x in item):
            raise ValueError("Path requires integer [query_index, reference_index] rows")
        qi, ri = int(item[0]), int(item[1])
        if not (0 <= qi < n and 0 <= ri < m):
            raise ValueError("Path index outside full saved native axes")
        output.append([qi, ri])
    if not output or output[0] != [0, 0] or output[-1] != [n-1, m-1]:
        raise ValueError("Full native endpoints must be retained")
    for before, after in zip(output, output[1:]):
        step = after[0]-before[0], after[1]-before[1]
        if step not in {(1,0), (0,1), (1,1)}:
            raise ValueError("Full path must use the three monotone unit moves")
    if {x[0] for x in output} != set(range(n)) or {x[1] for x in output} != set(range(m)):
        raise ValueError("Every original query/reference index must occur")
    return output


def _frame_record(parent, ri, qindices, query, reference, sources):
    r = _physical(reference, ri, sources)
    qs = [_physical(query, qi, sources) for qi in qindices]
    rknown = r["kind"] is not None
    qknown = [q for q in qs if q["kind"] is not None]
    any_qunknown = any(q["kind"] is None for q in qs)
    qfamilies = sorted({q["physical_family"] for q in qknown})
    qkinds = sorted({q["kind"] for q in qknown})
    same_family = rknown and r["physical_family"] in qfamilies
    different_family = rknown and any(f != r["physical_family"] for f in qfamilies)
    same_exact = rknown and r["kind"] in qkinds
    different_exact = rknown and any(k != r["kind"] for k in qkinds)
    occurrence_comparable = [
        q for q in qknown if rknown and q["occurrence"] is not None and r["occurrence"] is not None
    ]
    # These are overlapping flags. For example one reference can be linked to
    # same-kind, different-kind and unknown query frames at the same time.
    flags = {
        "any_same_physical_family": bool(same_family),
        "any_different_physical_family": bool(different_family),
        "any_unknown_comparison": (not rknown) or any_qunknown,
        "any_query_unknown": any_qunknown,
        "reference_kind_unknown": not rknown,
        "any_same_exact_kind": bool(same_exact),
        "any_different_exact_kind": bool(different_exact),
        "any_same_motion_subphase": bool(rknown and r["motion_subphase"] is not None and any(q["motion_subphase"] == r["motion_subphase"] for q in qknown)),
        "any_different_motion_subphase": bool(rknown and r["motion_subphase"] is not None and any(q["motion_subphase"] is not None and q["motion_subphase"] != r["motion_subphase"] for q in qknown)),
        "any_same_recorded_occurrence": any(q["occurrence"] == r["occurrence"] for q in occurrence_comparable),
        "any_different_recorded_occurrence": any(q["occurrence"] != r["occurrence"] for q in occurrence_comparable),
        "any_occurrence_unknown": r["occurrence"] is None or any(q["occurrence"] is None for q in qs),
    }
    return {
        "reference_index": ri, "reference_physical_draft": r,
        "reference_broad_phase_draft": _broad_phase(reference, ri, sources),
        "matched_query_indices": qindices,
        "matched_query_kind_set": qkinds,
        "matched_query_physical_family_set": qfamilies,
        "matched_query_motion_subphase_set": sorted({q["motion_subphase"] for q in qknown if q["motion_subphase"] is not None}),
        "matched_query_regions": [
            {"query_index": qi, "physical_draft": q,
             "broad_phase_draft": _broad_phase(query, qi, sources)}
            for qi, q in zip(qindices, qs)
        ],
        "flags": flags,
        "family_conflict_flag": bool(different_family),
        "unknown_is_retained_even_when_same_family_exists": True,
        "unique_correct_query_frame": None,
        "educational_or_learner_error_verdict": None,
    }


def _counts(frames):
    certain = [x for x in frames if not x["flags"]["reference_kind_unknown"]]
    unknown = [x for x in frames if x["flags"]["reference_kind_unknown"]]
    flag_counts = dict(Counter(k for x in frames for k,v in x["flags"].items() if v))
    # Include zero flags explicitly for stable comparisons.
    for x in frames:
        for k in x["flags"]:
            flag_counts.setdefault(k, 0)
    patterns = Counter(
        "same="+str(int(x["flags"]["any_same_physical_family"]))+
        ",different="+str(int(x["flags"]["any_different_physical_family"]))+
        ",unknown="+str(int(x["flags"]["any_unknown_comparison"]))
        for x in frames
    )
    return {
        "reference_frames": len(frames),
        "reference_physical_certain": len(certain),
        "reference_boundary_or_unknown": len(unknown),
        "flags_nonexclusive": flag_counts,
        "family_flag_patterns_partition": dict(patterns),
        "reference_certain_any_different_family": sum(x["flags"]["any_different_physical_family"] for x in certain),
        "reference_certain_any_query_unknown": sum(x["flags"]["any_query_unknown"] for x in certain),
        "reference_certain_same_and_different_family": sum(x["flags"]["any_same_physical_family"] and x["flags"]["any_different_physical_family"] for x in certain),
        "reference_certain_same_and_unknown": sum(x["flags"]["any_same_physical_family"] and x["flags"]["any_unknown_comparison"] for x in certain),
        "path_link_count": sum(len(x["matched_query_indices"]) for x in frames),
        "reference_axis_weight": "one per original reference frame; linked query multiplicity does not change denominator",
    }


def evaluate(query_alias, reference_alias, path):
    """Evaluate a caller-supplied path. Never construct, search or modify it."""
    sources = _sources()
    if (query_alias, reference_alias) not in sources["pairs"]:
        raise ValueError("Only the fixed12 current development/reference pairs are allowed")
    n = sources["physical"][query_alias]["native_frame_count"]
    m = sources["physical"][reference_alias]["native_frame_count"]
    p = _path(path, n, m)
    by_reference = {}
    for qi, ri in p:
        by_reference.setdefault(ri, []).append(qi)
    parent_records = []
    for parent in sources["parents"]:
        if (parent["query"], parent["reference"]) != (query_alias, reference_alias):
            continue
        lo, hi = parent["reference_range"]
        frames = [
            _frame_record(parent, ri, by_reference[ri], query_alias, reference_alias, sources)
            for ri in range(lo, hi+1)
        ]
        parent_records.append({
            "id": parent["id"], "parent_id": parent["parent_id"],
            "reference_scope_inclusive": [lo, hi],
            "fixed_parent_physical_description": parent["parent_physical_description"],
            "frames": frames, "counts": _counts(frames),
            "same_kind_is_not_a_uniquely_correct_time": True,
        })
    all_frames = [x for p in parent_records for x in p["frames"]]
    return {
        "schema": "main_phase_natural_evaluation.v1",
        "query": query_alias, "reference": reference_alias,
        "path_sha256": hashlib.sha256(json.dumps(p, separators=(",",":")).encode("utf-8")).hexdigest(),
        "full_path_query_frames": n, "full_path_reference_frames": m,
        "full_path_links": len(p), "all_native_axes_retained": True,
        "parents": parent_records, "counts": _counts(all_frames),
        "source_pins": sources["pins"],
        "evaluation_drafts_not_matcher_or_training_inputs": True,
        "physical_family_definition": "Existing kinds[kind].physical_kind, falling back to exact kind; no newly invented semantic merging",
        "conflict_definition": "At least one known matched query family differs from this known reference family; mixed same/different/unknown links remain separately counted",
        "limitations": [
            "Existing nonblind physical observation drafts are not independent linguistic or learner-correctness truth.",
            "Different motion/hold families may reflect optional pauses or normal speed variation; conflict does not itself prove erroneous signing or alignment.",
            "Unknown/boundary frames remain in the fixed reference denominator, including where a same-family query also exists.",
            "Subphase and occurrence fields are recorded separately and never establish required repetition.",
        ],
        "educational_accuracy": None, "unique_true_time_accuracy": None,
        "new_alignment_or_inference": 0,
    }


def aggregate(results):
    """Aggregate one condition of exactly12 unique pairs, never mix panels."""
    results = list(results)
    sources = _sources()
    keys = [(x["query"], x["reference"]) for x in results]
    if len(keys) != EXPECTED_PAIRS or len(set(keys)) != EXPECTED_PAIRS or set(keys) != sources["pairs"]:
        raise ValueError("Aggregate requires every fixed pair exactly once in one condition")
    expected = {p["id"]: p for p in sources["parents"]}
    for r in results:
        allowed = {p["id"] for p in sources["parents"] if (p["query"],p["reference"]) == (r["query"],r["reference"])}
        if {p["id"] for p in r["parents"]} != allowed:
            raise ValueError("Parent attached to a different query/reference pair")
        if r["counts"] != _counts([f for p in r["parents"] for f in p["frames"]]):
            raise ValueError("Pair summary inconsistent with saved frame flags")
        if r["source_pins"] != sources["pins"]:
            raise ValueError("Cannot aggregate results from different evaluation drafts")
    parents = [p for r in results for p in r["parents"]]
    if len(parents) != EXPECTED_PARENTS or {p["id"] for p in parents} != set(expected):
        raise ValueError("Aggregate must preserve all39 fixed parent scopes")
    frames = []
    for p in parents:
        source = expected[p["id"]]
        lo, hi = source["reference_range"]
        if p["reference_scope_inclusive"] != [lo, hi] or [x["reference_index"] for x in p["frames"]] != list(range(lo,hi+1)):
            raise ValueError("Reference denominator or original frame order changed")
        if p["counts"] != _counts(p["frames"]):
            raise ValueError("Parent summary inconsistent with saved frame flags")
        frames.extend(p["frames"])
    counts = _counts(frames)
    if (counts["reference_frames"],counts["reference_physical_certain"],counts["reference_boundary_or_unknown"]) != (440,360,80):
        raise ValueError("Expected fixed440/certain360/unknown80")
    return {
        "schema": "main_phase_natural_aggregate.v1",
        "unique_pairs":12, "parent_instances":39, "counts":counts,
        "pair_counts":[{"query":r["query"],"reference":r["reference"],"counts":r["counts"]} for r in results],
        "conflict_reference_keys":[{"query":r["query"],"reference":r["reference"],"parent_id":p["parent_id"],"reference_index":f["reference_index"]}
                                   for r in results for p in r["parents"] for f in p["frames"] if f["family_conflict_flag"]],
        "source_pins":sources["pins"],
        "no_drop_of_unknown":True, "no_path_cell_weighting":True,
        "educational_accuracy":None, "unique_true_time_accuracy":None,
        "counts_are_evaluation_draft_diagnostics_only":True,
    }
