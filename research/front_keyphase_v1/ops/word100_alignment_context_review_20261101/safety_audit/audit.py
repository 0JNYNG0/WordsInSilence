"""Join frozen stage-audit rows to saved observational context. No rescoring.

This audit imports no runtime implementation, model, feature extraction, or
alignment module. Query physical annotations are read here only for explaining
counterexamples. They never enter context_evidence.py or the runtime runner.
"""
from pathlib import Path
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
U = HERE.parent
OPS = U.parent
PREV = OPS / "word100_alignment_missing_split_20261101"
PHYSICAL = OPS / "word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json"
REF = OPS / "word100_target_known_feedback_20261101/REFERENCE_PHASES.json"
AUTH = U / "SAFETY_AUDIT_AUTHORIZATION.json"
SEEDS = (42, 43, 44)
HYPOTHESES = ("DIRECT_BOTH_AXES", "BOUNDED_ANY_AXIS", "BOUNDED_BOTH_AXES")
BOUNDED = "BOUNDED_BY_DISTINCT_OBSERVATIONS"
DIRECT_CONTEXT = "DIRECT_CELL_NO_GAP_CONTEXT"


def read(p):
    p = Path(p)
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    return json.loads(raw.decode("utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def new(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    tasks, manifest = read(U / "TASKS.json"), read(U / "RESULT_MANIFEST.json")
    fixed = [Path(__file__), AUTH, U / "FROZEN.json", U / "EVALUATION.json",
             U / "TASKS.json", U / "RESULT_MANIFEST.json", U / "context_evidence.py",
             PREV / "EVALUATION.json", PHYSICAL, REF]
    pins = {str(p): sha(p) for p in fixed}
    for rec in tasks["records"]:
        for name, expected in (("path", "sha256"), ("source", "source_sha256")):
            assert sha(rec[name]) == rec[expected]
            pins[rec[name]] = rec[expected]
    for rec in manifest["contexts"]:
        assert sha(rec["path"]) == rec["sha256"]
        pins[rec["path"]] = rec["sha256"]
    new(HERE / "PROSPECTIVE_CONTRACT.json", {
        "status": "FROZEN_BEFORE_SAFETY_JOIN", "at_utc": now(), "pins": pins,
        "primary": "Historical DEV12 pairs,360 frozen reference-frame opportunities per seed",
        "secondary": "FIT24 pairs,720 frozen reference-frame opportunities per seed",
        "seeds": list(SEEDS), "records": 108, "hypotheses": list(HYPOTHESES),
        "hypotheses_are_misinterpretations_not_runtime_policies": True,
        "definitions": {
            "DIRECT_BOTH_AXES": "Both saved query/ref feature cells are DIRECT_COMPLETE.",
            "BOUNDED_ANY_AXIS": "At least one non-DIRECT cell has strict before/after bounded context.",
            "BOUNDED_BOTH_AXES": "Both non-DIRECT cells have strict before/after bounded context.",
            "pair_support": "SHARED, QUERY_UNOBSERVED, REFERENCE_UNOBSERVED, BOTH_UNOBSERVED",
            "weight": "Use frozen trace weight1/N mapped query links per original reference frame, for each exact feature dimension.",
        },
        "denominators": "Group feature-cell opportunities=360*D per DEV seed,720*D per FIT seed. All-group70 dimensions are accounting only, not independent semantic targets. Three seeds repeat the same videos/masks/stage labels.",
        "source_stage_labels": "Use saved method=global_dtw audit.events.trace rows verbatim. Never call phase_audit or recompute stage/recognition scores.",
        "query_truth_isolated": "PHYSICAL used by this audit.py only to explain frozen counterexample frame labels; no runtime module is imported or edited.",
        "example_selection": "Seed42 DEV only; sort query/reference, group learned3 first then group name, dimension order, event/trace order. Keep first different-known-kind example per hypothesis and word. No selection by residual magnitude, no threshold.",
        "interpretation": "Different-known-stage under a hypothesis disproves interpreting that observational state as guaranteed same stage. No counterexample is not proof of safety. Same-known-kind is not exact repeated occurrence or educational correctness.",
        "outputs": ["RESULT.json"], "new_path_search": 0, "new_inference": 0,
        "new_training": 0, "new_runtime_policy": False, "old_scores_unchanged": True,
        "no_post_join_threshold_or_reference_selection": True,
    })
    print("contract ready", str(HERE / "PROSPECTIVE_CONTRACT.json"), sha(HERE / "PROSPECTIVE_CONTRACT.json"))


def observation(axis, i, k):
    if axis["matched_support"][i][k]:
        return "PARTIAL_SHARED" if axis["mixed_counterpart_missing"][i][k] else "DIRECT_COMPLETE"
    return "NO_SHARED"


def context_map(info):
    n, dims = info["possible_shape"]
    mapping = [[None for _ in range(dims)] for _ in range(n)]
    for ei, ep in enumerate(info["episodes"]):
        for i in range(ep["start"], ep["end"] + 1):
            assert mapping[i][ep["dimension"]] is None
            mapping[i][ep["dimension"]] = (ep["temporal_context"], ei)
    return mapping


def physical_case(clips, alias, frame):
    ci, clip = clips[alias]
    found = []
    for ri, region in enumerate(clip["regions"]):
        if region["region_type"] == "observed_physical_core" and region["core"][0] <= frame <= region["core"][1]:
            found.append({"json_pointer": f"/clips/{ci}/regions/{ri}", "region": region})
    return {"source": str(PHYSICAL), "alias": alias, "frame": frame, "certain_regions": found}


def counterexample(h, task, stage_source_index, event_index, trace_index, event,
                   ri, qi, weight, group, dim, axes, context, qctx, rctx, raw_file):
    qa, ra = axes["query"], axes["reference"]
    base = f"/evidence/groups/{group}"
    def details(a, index, axis, ctx):
        return {
            "frame_id": a["frame_id"][index], "timestamp_us": a["timestamp_us"][index],
            "observation": observation(a, index, dim),
            "mean_residual": a["mean_residual"][index][dim],
            "max_residual": a["max_residual"][index][dim],
            "residual_is_conditional_axis_mean_not_specific_link": True,
            "mean_pointer": base + f"/{axis}/mean_residual/{index}/{dim}",
            "max_pointer": base + f"/{axis}/max_residual/{index}/{dim}",
            "context": ctx[0] if ctx else DIRECT_CONTEXT,
            "context_pointer": f"/annotation/groups/{group}/{axis}/episodes/{ctx[1]}" if ctx else None,
        }
    return {
        "hypothesis": h, "word_id": task["word_id"], "query": task["query"], "reference": task["reference"],
        "seed": 42, "stage_label": "different_observed_kind", "frozen_link_weight": weight,
        "reference_event": event["event"], "reference_kind": event["reference_kind"],
        "group": group, "dimension": dim, "dimension_label": qa["dimension_labels"][dim],
        "query_cell": details(qa, qi, "query", qctx), "reference_cell": details(ra, ri, "reference", rctx),
        "raw_evidence_source": raw_file, "context_source": context,
        "stage_audit_source": str(PREV / "EVALUATION.json"),
        "stage_audit_pointer": f"/pairs/{stage_source_index}/audit/events/{event_index}/trace/{trace_index}",
        "not_an_educational_error_label": True,
    }


def run():
    contract = read(HERE / "PROSPECTIVE_CONTRACT.json")
    for p, h in contract["pins"].items():
        assert sha(p) == h, p
    old, tasks, manifest = read(PREV / "EVALUATION.json"), read(U / "TASKS.json"), read(U / "RESULT_MANIFEST.json")
    stage = {(r["query"], r["reference"]): (i, r) for i, r in enumerate(old["pairs"]) if r["method"] == "global_dtw"}
    assert len(stage) == 36
    contexts = {(r["query"], r["reference"], r["seed"]): r for r in manifest["contexts"]}
    physical = {c["alias"]: (i, c) for i, c in enumerate(read(PHYSICAL)["clips"])}
    tables, feature_tables, hyps = Counter(), Counter(), Counter()
    group_dims, examples, example_keys, sources, checks = {}, [], set(), [], 0
    for task in sorted(tasks["records"], key=lambda t: (t["query"], t["reference"], t["seed"])):
        key = (task["query"], task["reference"], task["seed"])
        ci = contexts[key]
        raw, context = read(task["path"]), read(ci["path"])
        si, sr = stage[key[:2]]
        assert raw["path"] == read(sr["source"])["path"]
        assert sr["source_sha256"] == sha(sr["source"])
        assert raw["query"] == context["query"] == task["query"]
        assert raw["reference"] == context["reference"] == task["reference"]
        checks += 4
        role, seed = sr["role"], task["seed"]
        pointer = {"query": task["query"], "reference": task["reference"], "seed": seed,
                   "role": role, "raw": task["path"], "context": ci["path"],
                   "stage_audit_pointer": f"/pairs/{si}/audit"}
        sources.append(pointer)
        groups = raw["evidence"]["groups"]
        for group in sorted(groups, key=lambda g: (g != "learned3", g)):
            axes = groups[group]
            qa, ra = axes["query"], axes["reference"]
            dims = len(qa["dimension_labels"])
            assert group not in group_dims or group_dims[group] == dims
            group_dims[group] = dims
            qc = context_map(context["annotation"]["groups"][group]["query"])
            rc = context_map(context["annotation"]["groups"][group]["reference"])
            for dim in range(dims):
                for ei, event in enumerate(sr["audit"]["events"]):
                    for ti, (ri, qi, label, weight) in enumerate(event["trace"]):
                        assert qi is not None, "Frozen fullglobal source unexpectedly unmapped"
                        qobs, robs = observation(qa, qi, dim), observation(ra, ri, dim)
                        qtc, rtc = qc[qi][dim], rc[ri][dim]
                        assert (qtc is None) == (qobs == "DIRECT_COMPLETE")
                        assert (rtc is None) == (robs == "DIRECT_COMPLETE")
                        qo, ro = qa["own_observed"][qi][dim], ra["own_observed"][ri][dim]
                        support = "SHARED" if qo and ro else "QUERY_UNOBSERVED" if ro else "REFERENCE_UNOBSERVED" if qo else "BOTH_UNOBSERVED"
                        qt, rt = qtc[0] if qtc else DIRECT_CONTEXT, rtc[0] if rtc else DIRECT_CONTEXT
                        common = (role, seed, group, label, support, qobs, robs, qt, rt)
                        tables[common] += weight
                        feature_tables[common + (dim, qa["dimension_labels"][dim])] += weight
                        accepted = {
                            "DIRECT_BOTH_AXES": qobs == robs == "DIRECT_COMPLETE",
                            "BOUNDED_ANY_AXIS": qt == BOUNDED or rt == BOUNDED,
                            "BOUNDED_BOTH_AXES": qt == BOUNDED and rt == BOUNDED,
                        }
                        for h, applies in accepted.items():
                            if applies:
                                hyps[(role, seed, group, h, label)] += weight
                            ek = (h, sr["word_id"])
                            if applies and role == "development" and seed == 42 and label == "different_observed_kind" and ek not in example_keys:
                                ex = counterexample(h, sr, si, ei, ti, event, ri, qi, weight, group, dim, axes, ci["path"], qtc, rtc, task["path"])
                                ex["query_physical_audit_only"] = physical_case(physical, task["query"], qa["frame_id"][qi])
                                ex["reference_physical_audit_only"] = physical_case(physical, task["reference"], ra["frame_id"][ri])
                                ex["query_video"] = str(OPS / "word100_stage_feedback_cycle_20261031/videos" / (task["query"] + ".mp4"))
                                ex["reference_video"] = str(OPS / "word100_stage_feedback_cycle_20261031/videos" / (task["reference"] + ".mp4"))
                                examples.append(ex)
                                example_keys.add(ek)
                        checks += 2
    denoms = []
    for role, n in (("development", 360), ("fit", 720)):
        for seed in SEEDS:
            for group, dims in sorted(group_dims.items()):
                actual = sum(v for k, v in tables.items() if k[:3] == (role, seed, group))
                assert abs(actual - n * dims) < 1e-7
                denoms.append({"role": role, "seed": seed, "group": group, "dimensions": dims,
                               "frozen_reference_frames": n, "feature_cell_opportunities": n * dims})
                checks += 1
    def dump(counter, fields):
        return [dict(zip(fields, k), weight=v) for k, v in sorted(counter.items())]
    fields = ["role", "seed", "group", "frozen_stage_label", "pair_support", "query_observation",
              "reference_observation", "query_context", "reference_context"]
    hypotheses = dump(hyps, ["role", "seed", "group", "hypothesis", "frozen_stage_label"])
    outcomes = []
    for h in HYPOTHESES:
        selected = [r for r in hypotheses if r["role"] == "development" and r["seed"] == 42 and r["hypothesis"] == h]
        weights = Counter()
        for row in selected:
            weights[row["frozen_stage_label"]] += row["weight"]
        outcomes.append({"hypothesis": h, "primary_seed42_feature_cell_weights": dict(weights),
                         "different_known_stage_counterexample_exists": weights["different_observed_kind"] > 0,
                         "no_counterexample_does_not_establish_safety": True,
                         "not_a_runtime_policy_or_educational_accuracy": True})
    result = {
        "status": "COMPLETE_FROZEN_STAGE_CONTEXT_SAFETY_JOIN", "at_utc": now(),
        "contract": str(HERE / "PROSPECTIVE_CONTRACT.json"), "contract_sha256": sha(HERE / "PROSPECTIVE_CONTRACT.json"),
        "checks": checks, "source_records": sources, "group_dimensions": group_dims,
        "frozen_stage_summary_copied_not_recomputed": {k: v for k, v in old["summary"].items() if "/global_dtw/" in k},
        "denominators": denoms, "group_crosstab": dump(tables, fields),
        "feature_crosstab": dump(feature_tables, fields + ["dimension", "dimension_label"]),
        "hypothesis_crosstab": hypotheses, "hypothesis_outcomes": outcomes,
        "examples": examples,
        "denominator_cautions": ["360DEVframes perseed, not1080independentframes.",
                                "Three seeds repeat video/annotation/mask evidence.",
                                "70feature dimensions include related measurements; their summedcounts are accounting, not70independent semantic targets.",
                                "Samephysicalkind doesnotestablish sameoccurrence, correctfeatureclassification, orcorrectsign."],
        "new_stage_score_calculation": 0, "new_alignment": 0, "new_model_inference": 0,
        "new_training": 0, "new_runtime_policy": False,
        "query_annotations_audit_only": True, "prior_frozen_outputs_unchanged": True,
    }
    for p, h in contract["pins"].items():
        assert sha(p) == h, p
    new(HERE / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "outcomes": outcomes, "examples": len(examples)}, ensure_ascii=False))


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
