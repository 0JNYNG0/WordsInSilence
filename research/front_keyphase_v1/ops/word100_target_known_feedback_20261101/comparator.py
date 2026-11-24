"""Known-practice-word, reference-local saved-output comparison.

No model inference or path search. Query labels never enter RuntimeStore or
compare_practice. All references, methods and seeds remain individual. Local
observations are useful despite unrelated missing frames; they are never a
whole-sign or mandatory linguistic correctness verdict.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

U = Path(__file__).resolve().parent
OPS = U.parent
OLD = OPS / "word100_stage_feedback_cycle_20261031"
REL = OPS / "word100_relational_sequence_compare_20261031"
CORE = OPS / "word100_core_context_split_20261101"
PHYSICAL = OPS / "word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json"
PHASE = OPS / "word100_core_stage_evidence_20261031/PHASE_DATA.json"
TARGETS = ("R_index_only", "R_open_four", "L_open_four")
RELATIONS = ("hands_wrist_delta_x", "hands_wrist_delta_y", "hands_wrist_distance")
METHODS = ("global_dtw", "core_first")
SEEDS = (42, 43, 44)
CONFIDENCE = .8
DIFFERENCE = .6
SCHEMA = "wis_target_known_local_comparison_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def source_support(array, names):
    """Original estimate_support rule: detector evidence, not visual truth."""
    if "mask" not in array:
        raise ValueError("Explicit input observation mask required")
    mask = np.asarray(array["mask"])
    x = np.asarray(array["x"])
    if mask.dtype != np.bool_ or mask.shape != x.shape or mask.shape[1] != len(names):
        raise ValueError("Input observation schema/mask mismatch")
    out = np.zeros((len(mask), len(TARGETS)), dtype=bool)
    for j, feature in enumerate(TARGETS):
        side = feature[0]
        wrist = [names.index(side + "_hand_wrist_" + a) for a in ("x", "y")]
        fingers = [i for i, name in enumerate(names)
                   if name.startswith(side + "_") and "_angle" in name and "elbow" not in name]
        out[:, j] = mask[:, wrist].all(1) & (mask[:, fingers].sum(1) >= 10)
    return out


def prepare():
    """Create FIT-only runtime expectations and explicit historical test tasks."""
    acceptance = read(U / "COMPARATOR_FREEZE_ACCEPTANCE.json")
    if acceptance["proposal_sha256"] != sha(U / "COMPARATOR_FROZEN.json"):
        raise ValueError("Proposal changed after root approval")
    bank = read(U / "REFERENCE_BANK.json")
    physical = read(PHYSICAL)
    phase = {c["alias"]: c for c in read(PHASE)["clips"]}
    old_rows = {r["alias"]: r for r in read(OLD / "LEARNING_INPUTS.json")["rows"]}
    allowed_refs = {r["legacy_alias"] for w in bank["words"] for r in w["references"]
                    if r["phase_reference_available"]}
    ref_clips = {}
    for clip in physical["clips"]:
        if clip["alias"] not in allowed_refs:
            continue
        if clip["role"] != "fit":
            raise ValueError("Non-FIT reference")
        parents = [p for p in phase[clip["alias"]]["segments"] if p["role"] == "core_candidate"]
        parent_ids = {p["segment_id"] for p in parents}
        parent_by = {p["segment_id"]:p for p in parents}
        for region in clip["regions"]:
            if region["region_type"] == "observed_physical_core" and region["source_segment_id"] not in parent_ids:
                raise ValueError("Child has no saved parent phase")
            if region["region_type"] == "observed_physical_core":
                a,b=region["core"];pa,pb=parent_by[region["source_segment_id"]]["core"]
                if not pa<=a<=b<=pb:
                    raise ValueError("Child interval escapes original parent phase")
        ref_clips[clip["alias"]] = {"alias": clip["alias"], "role": "fit",
            "native_frame_count": clip["native_frame_count"], "regions": clip["regions"],
            "parents": parents, "word_id": old_rows[clip["alias"]]["word_id"]}
    phase_bank = {"status": "FIT_REFERENCE_ONLY_NO_QUERY_ANNOTATIONS", "clips": ref_clips,
                  "source_pins": {str(PHYSICAL): sha(PHYSICAL), str(PHASE): sha(PHASE)}}
    save_new(U / "REFERENCE_PHASES.json", phase_bank)
    # Query word metadata is used only here to construct explicit test requests.
    public_clips = [c for c in physical["clips"] if c["alias"] in old_rows]
    tasks = []
    for clip in public_clips:
        row = old_rows[clip["alias"]]
        tasks.append({"practice_word_id": row["word_id"], "query_alias": clip["alias"],
                      "role": clip["role"], "scope": "historical_public_descriptive"})
    paths = {}
    expected_parents = expected_children = pair_count = 0
    for task in tasks:
        for ref in ref_clips.values():
            if ref["word_id"] != task["practice_word_id"] or ref["alias"] == task["query_alias"]:
                continue
            name = "B_same_path_" + task["query_alias"] + "_" + ref["alias"]
            for path in (REL / "pairs" / (name + ".json"), CORE / "results" / (name + ".json")):
                paths[str(path)] = sha(path)
            pair_count += 1
            expected_parents += len(ref["parents"])
            expected_children += sum(r["region_type"] == "observed_physical_core" for r in ref["regions"])
    if pair_count != 36 or expected_parents != 117:
        raise ValueError("Original pair/parent denominator changed")
    save_new(U / "COMPARATOR_TASKS.json", {"tasks": tasks, "pairs": pair_count,
        "original_parent_instances": expected_parents, "refined_child_instances": expected_children,
        "saved_path_pins": paths, "no_fresh_evaluation": True})


class RuntimeStore:
    """Saved observations/model outputs/automatic paths; no query truth loader."""
    def __init__(self, bank):
        info = read(OLD / "LEARNING_INPUTS.json")
        self.rows = {r["alias"]: r for r in info["rows"]}
        self.index = {r["alias"]: i for i, r in enumerate(info["rows"])}
        self.target_indices = [info["targets"].index(f) for f in TARGETS]
        self.phase_refs = read(U / "REFERENCE_PHASES.json")["clips"]
        self.probabilities = {seed: np.load(OLD / "runs" / f"pose_plus_signrep_{seed}" / "probabilities.npy",
                                               mmap_mode="r", allow_pickle=False) for seed in SEEDS}
        self.cache = {}
        self.bank = bank

    def clip(self, alias):
        if alias not in self.rows:
            raise ValueError("No stored native output for query alias")
        if alias not in self.cache:
            row = self.rows[alias]
            if sha(row["input"]) != row["input_sha256"]:
                raise ValueError("Saved native input hash changed")
            with np.load(row["input"], allow_pickle=False) as data:
                a = {k: data[k] for k in ("x", "mask", "frame_id", "timestamp_us")}
            n = len(a["frame_id"])
            if not np.array_equal(a["frame_id"], np.arange(n)):
                raise ValueError("Saved path native ordinal/frame ID equivalence needs explicit adapter")
            if not np.all(np.diff(a["timestamp_us"]) > 0):
                raise ValueError("Native timestamps are not ordered")
            a["support"] = source_support(a, row["feature_names"])
            a["names"] = row["feature_names"]
            a["probs"] = {s: np.asarray(p[self.index[alias], :n])[:, self.target_indices]
                          for s, p in self.probabilities.items()}
            self.cache[alias] = a
        return self.cache[alias]

    def paths(self, query, reference):
        name = "B_same_path_" + query + "_" + reference
        global_file = REL / "pairs" / (name + ".json")
        local_file = CORE / "results" / (name + ".json")
        if not global_file.exists() or not local_file.exists():
            return None
        global_path, local = read(global_file), read(local_file)
        if any(x["query"] != query or x["reference"] != reference for x in (global_path, local)):
            raise ValueError("Saved path identity mismatch")
        return {"global_dtw": global_path["path"],
            "core_first": {a["segment_id"]: {"path": a["selected"]["core_path"] if a.get("selected") else [],
                             "saved_state": a["state"], "reference_core": a["reference_core"],
                             "disjoint_core_tie":a.get("disjoint_core_tie"),
                             "context_tie_resolved":a.get("context_tie_resolved"),
                             "selected_tie_index":a.get("selected_tie_index"),
                             "candidate_query_cores":[t.get("query_core") for t in a.get("core_ties",[])],
                             "within_window_path_ties_enumerated":a.get("within_window_path_ties_enumerated"),
                             "diagnostic_hold_reasons":a.get("diagnostic_hold_reasons",[])}
                           for a in local["anchors"]},
            "sources": {str(global_file): sha(global_file), str(local_file): sha(local_file)}}


def check_path(path, nq, nr):
    if len({tuple(p) for p in path}) != len(path):
        raise ValueError("Duplicate saved path cells")
    if any(len(p) != 2 or any(type(v) is not int for v in p) or not (0 <= p[0] < nq and 0 <= p[1] < nr)
           for p in path):
        raise ValueError("Out-of-range or invalid native path cell")
    if any(c < a or d < b for (a, b), (c, d) in zip(path, path[1:])):
        raise ValueError("Within-phase path must be monotone")


def probability_status(qp, rp, qs, rs):
    if not rs:
        return "REFERENCE_UNSUPPORTED"
    if not qs:
        return "QUERY_UNSUPPORTED"
    if not np.isfinite(qp) or not np.isfinite(rp):
        return "OUTPUT_UNAVAILABLE"
    if max(qp, 1-qp) < CONFIDENCE or max(rp, 1-rp) < CONFIDENCE:
        return "LOW_CONFIDENCE"
    if abs(qp-rp) >= DIFFERENCE:
        return "ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE"
    if (qp >= .5) == (rp >= .5):
        return "ESTIMATED_FEATURE_AGREEMENT"
    return "UNCERTAIN_FEATURE_DIFFERENCE"


def local_compare(query, reference, path, core, seed):
    """Each original reference frame retains total weight one per feature."""
    check_path(path, len(query["frame_id"]), len(reference["frame_id"]))
    start, end = core
    if not 0 <= start <= end < len(reference["frame_id"]):
        raise ValueError("Reference child core outside original frame axis")
    by_reference = defaultdict(list)
    for qi, ri in path:
        if start <= ri <= end:
            by_reference[ri].append(qi)
    counts = {f: Counter() for f in TARGETS}
    records = []
    for ri in range(start, end + 1):
        query_indices = by_reference[ri]
        links = []
        for qi in query_indices:
            states = [probability_status(float(query["probs"][seed][qi,j]),
                        float(reference["probs"][seed][ri,j]), bool(query["support"][qi,j]),
                        bool(reference["support"][ri,j])) for j in range(len(TARGETS))]
            for feature, state in zip(TARGETS, states):
                counts[feature][state] += 1 / len(query_indices)
            links.append({"query_frame": int(query["frame_id"][qi]),
                "query_pts_us": int(query["timestamp_us"][qi]),
                "query_probabilities": [float(v) if np.isfinite(v) else None for v in query["probs"][seed][qi]],
                "query_input_support": query["support"][qi].tolist(),
                "states": states, "weight_per_feature": 1 / len(query_indices)})
        if not links:
            for j, feature in enumerate(TARGETS):
                counts[feature]["NO_SAVED_CORRESPONDENCE" if reference["support"][ri,j]
                                else "REFERENCE_UNSUPPORTED"] += 1
        records.append({"reference_frame": int(reference["frame_id"][ri]),
            "reference_pts_us": int(reference["timestamp_us"][ri]),
            "reference_probabilities": [float(v) if np.isfinite(v) else None for v in reference["probs"][seed][ri]],
            "reference_input_support": reference["support"][ri].tolist(), "links": links})
    expected = end-start+1
    summaries = {}
    for j, feature in enumerate(TARGETS):
        count = counts[feature]
        if not np.isclose(sum(count.values()), expected):
            raise ValueError("Reference denominator was lost")
        usable = sum(count[s] for s in ("ESTIMATED_FEATURE_AGREEMENT", "ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE"))
        common = sum(count[s] for s in ("ESTIMATED_FEATURE_AGREEMENT", "ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE",
                                      "LOW_CONFIDENCE", "UNCERTAIN_FEATURE_DIFFERENCE"))
        # Deliberately explicit counterfactual; this is not production's existing decision.
        global_gate = bool(query["support"][:,j].all() and reference["support"][:,j].all()
                           and np.isfinite(query["probs"][seed][:,j]).all()
                           and np.isfinite(reference["probs"][seed][:,j]).all())
        summaries[feature] = {"expected_reference_frames": expected, "weighted_states": dict(count),
            "input_supported_and_finite_weight": common, "confidence_qualified_weight": usable,
            "input_coverage": common/expected, "confidence_coverage": usable/expected,
            "global_complete_input_gate_counterfactual": global_gate,
            "local_qualified_weight_retained_despite_global_gate": usable if not global_gate else 0,
            "contains_difference_candidate": count["ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE"] > 0,
            "educational_warning": None}
    return {"seed": seed, "target_order": list(TARGETS), "reference_core": core,
            "frames": records, "features": summaries, "whole_sign_verdict": None,
            "eligibility_from_query_annotations": False}


def measured_relations(query, reference, path, core):
    by = defaultdict(list)
    for qi, ri in path:
        if core[0] <= ri <= core[1]:
            by[ri].append(qi)
    out = []
    for feature in RELATIONS:
        qj, rj = query["names"].index(feature), reference["names"].index(feature)
        rows, known_weight = [], 0.0
        for ri in range(core[0], core[1]+1):
            rv = float(reference["x"][ri,rj])
            rok = bool(reference["mask"][ri,rj]) and np.isfinite(rv)
            links = []
            for qi in by[ri]:
                qv = float(query["x"][qi,qj]); qok = bool(query["mask"][qi,qj]) and np.isfinite(qv)
                weight = 1/len(by[ri])
                if rok and qok:
                    known_weight += weight
                links.append({"query_frame": int(query["frame_id"][qi]), "query_value": qv if qok else None,
                    "raw_difference": qv-rv if rok and qok else None, "weight": weight})
            rows.append({"reference_frame": int(reference["frame_id"][ri]),
                         "reference_value": rv if rok else None, "links": links})
        out.append({"feature": feature, "expected_reference_frames": core[1]-core[0]+1,
            "jointly_observed_weight": known_weight, "frames": rows,
            "linguistic_tolerance": None, "difference_warning": None,
            "kind": "saved_normalized_coordinate_measurement_not_contact_or_semantic_truth"})
    return out


def projected_order(stages, method):
    events = []
    for stage in stages:
        qi = sorted({p[0] for p in stage["saved_path"]})
        events.append({"reference_event_id": stage["reference_event"]["id"],
            "reference_core": stage["reference_event"]["core"],
            "projected_query_span": [min(qi), max(qi)] if qi else None,
            "query_original_frames": qi,
            "saved_candidate_ambiguity":stage.get("saved_candidate_ambiguity",False),
            "saved_candidate_metadata":stage.get("saved_candidate_metadata"),
            "unique_physical_correspondence_claim":False})
    relations = []
    for first, second in zip(events, events[1:]):
        a, b = first["projected_query_span"], second["projected_query_span"]
        status = "UNMAPPED_EVENT" if a is None or b is None else (
            "FORWARD_DISJOINT" if a[1] < b[0] else "REVERSE_DISJOINT" if b[1] < a[0] else "OVERLAPPING_PROJECTIONS")
        ambiguous=first["saved_candidate_ambiguity"] or second["saved_candidate_ambiguity"]
        relations.append({"first": first["reference_event_id"], "second": second["reference_event_id"],
            "status":"AMBIGUOUS_CANDIDATE_ORDER" if ambiguous else status,
            "selected_projection_relation":status,"source_candidate_ambiguity":ambiguous})
    return {"events": events, "relations": relations, "all_event_instances_retained": True,
        "method_forces_monotonic_order": method == "global_dtw",
        "independent_correct_order_claim": False, "required_repetition_count": None}


def compare_practice(practice_word_id, query_alias, bank, runtime_store):
    """Explicit target argument selects refs. No query word/kind/role/label input."""
    word = next((w for w in bank["words"] if w["word_id"] == practice_word_id), None)
    if word is None:
        return {"schema": SCHEMA, "practice_word_id": practice_word_id, "query": query_alias,
                "status": "UNSUPPORTED_PRACTICE_WORD", "references": [], "whole_sign_verdict": None}
    output = {"schema": SCHEMA, "practice_word_id": practice_word_id, "query": query_alias,
        "status": "TARGET_ROUTED", "references": [],
        "guide_candidates": word["guide_candidates"], "guide_can_override_observation": False,
        "mandatory_linguistic_rules_enabled": False, "whole_sign_verdict": None}
    for ref in word["references"]:
        ra = ref["legacy_alias"]
        if ra is not None and ra == query_alias:
            # A FIT clip used as a historical query must not reference itself.
            output["references"].append({"reference_alias": ra, "status": "SELF_REFERENCE_EXCLUDED",
                                        "whole_sign_verdict": None})
            continue
        record = {"reference_alias": ra, "reference_clip_id": ref["clip_id"],
                  "reference_source": ref["source"], "methods": {}, "whole_sign_verdict": None}
        if not ref["phase_reference_available"] or ra not in runtime_store.phase_refs:
            record["status"] = "PHASE_COMPARISON_UNSUPPORTED"
            output["references"].append(record); continue
        paths = runtime_store.paths(query_alias, ra)
        if paths is None:
            record["status"] = "SAVED_AUTOMATIC_PATH_UNAVAILABLE"
            output["references"].append(record); continue
        query, reference = runtime_store.clip(query_alias), runtime_store.clip(ra)
        rph = runtime_store.phase_refs[ra]
        parents = {p["segment_id"]: p for p in rph["parents"]}
        if set(paths["core_first"]) != set(parents):
            raise ValueError("Original parent lineage mismatch")
        record.update(status="LOCAL_ESTIMATED_EVIDENCE", saved_path_sources=paths["sources"],
            original_parent_ids=list(parents), reference_original_regions=rph["regions"],
            reference_native_frames=len(reference["frame_id"]), query_native_frames=len(query["frame_id"]))
        for method in METHODS:
            stages = []
            for event in rph["regions"]:
                if event["region_type"] != "observed_physical_core":
                    continue
                parent = event["source_segment_id"]
                ca,cb=event["core"];pa,pb=parents[parent]["core"]
                if not pa<=ca<=cb<=pb:
                    raise ValueError("Child interval escapes original parent phase")
                if paths["core_first"][parent]["reference_core"] != parents[parent]["core"]:
                    raise ValueError("Saved path parent bounds do not match source")
                source_path = paths[method] if method == "global_dtw" else paths[method][parent]["path"]
                check_path(source_path, len(query["frame_id"]), len(reference["frame_id"]))
                subpath = [p for p in source_path if event["core"][0] <= p[1] <= event["core"][1]]
                candidate=None if method=="global_dtw" else {k:v for k,v in paths[method][parent].items() if k!="path"}
                ambiguity=bool(candidate and (candidate.get("disjoint_core_tie")
                    or "ambig" in str(candidate.get("saved_state","")).lower()
                    or len(candidate.get("candidate_query_cores",[]))>1))
                stages.append({"reference_event": event, "parent_segment_id": parent,
                    "parent_reference_core": parents[parent]["core"], "saved_path": subpath,
                    "saved_path_state": None if method == "global_dtw" else paths[method][parent]["saved_state"],
                    "saved_candidate_metadata":candidate,"saved_candidate_ambiguity":ambiguity,
                    "local_evidence_conditional_on_saved_projection":True,
                    "unique_physical_correspondence_claim":False,
                    "seeds": [local_compare(query, reference, subpath, event["core"], seed) for seed in SEEDS],
                    "measured_relations": measured_relations(query, reference, subpath, event["core"]),
                    "mandatory_stage": False, "query_physical_annotation_used": False})
            record["methods"][method] = {"stages": stages, "projected_order": projected_order(stages, method)}
        output["references"].append(record)
    return output


def selftest():
    checks = []

    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})
        if not condition:
            raise AssertionError(name)

    names = read(OLD / "LEARNING_INPUTS.json")["rows"][0]["feature_names"]
    n = 8
    a = {"x": np.zeros((n,len(names)),dtype=np.float32),
         "mask": np.ones((n,len(names)),dtype=bool), "frame_id": np.arange(n),
         "timestamp_us": np.arange(n)*33333, "names": names,
         "probs": {s: np.tile([.95,.05,.95],(n,1)) for s in SEEDS}}
    a["support"] = source_support(a,names)
    path = [[i,i] for i in range(n)]
    base = local_compare(a,a,path,[2,5],42)
    check("all_local_frames_retained", len(base["frames"]) == 4)
    check("identity_qualified", all(f["confidence_qualified_weight"] == 4 for f in base["features"].values()))
    check("no_identity_difference", not any(f["contains_difference_candidate"] for f in base["features"].values()))
    b = copy.deepcopy(a); b["support"][0,:] = False
    partial = local_compare(b,a,path,[2,5],42)
    check("unrelated_unknown_not_global_block", all(f["confidence_qualified_weight"] == 4
          and f["local_qualified_weight_retained_despite_global_gate"] == 4 for f in partial["features"].values()))
    b["support"][3,0] = False
    missing = local_compare(b,a,path,[2,5],42)
    check("missing_feature_local_only", missing["features"]["R_index_only"]["confidence_qualified_weight"] == 3
          and missing["features"]["R_open_four"]["confidence_qualified_weight"] == 4)
    b = copy.deepcopy(a); b["probs"][42][3,0] = .01
    changed = local_compare(b,a,path,[2,5],42)
    check("one_frame_change_not_smoothed", changed["features"]["R_index_only"]["contains_difference_candidate"])
    b["probs"][42][3,0] = .5
    uncertain = local_compare(b,a,path,[2,5],42)
    check("low_confidence_not_agreement", uncertain["features"]["R_index_only"]["weighted_states"]["LOW_CONFIDENCE"] == 1)
    b["probs"][42][3,0] = np.nan
    unknown = local_compare(b,a,path,[2,5],42)
    check("nonfinite_not_negative", unknown["features"]["R_index_only"]["weighted_states"]["OUTPUT_UNAVAILABLE"] == 1)
    no_map = local_compare(a,a,[p for p in path if p[1] != 3],[2,5],42)
    check("unmapped_keeps_denominator", no_map["features"]["R_index_only"]["expected_reference_frames"] == 4
          and no_map["features"]["R_index_only"]["weighted_states"]["NO_SAVED_CORRESPONDENCE"] == 1)
    dual = [[i,j] for j in range(2,6) for i in (j,j+1)]
    stretched = local_compare(a,a,dual,[2,5],42)
    check("warped_weight_not_inflated", stretched["features"]["R_index_only"]["confidence_qualified_weight"] == 4)
    stages = [{"reference_event":{"id":str(i),"core":[i,i]},"saved_path":[[q,i]]}
              for i,q in enumerate([6,3,1])]
    order = projected_order(stages,"core_first")
    check("reverse_order_preserved", all(r["status"] == "REVERSE_DISJOINT" for r in order["relations"]))
    stages[1]["saved_path"] = []
    check("missing_stage_not_discarded", len(projected_order(stages,"core_first")["events"]) == 3)
    check("global_not_order_proof", projected_order(stages,"global_dtw")["method_forces_monotonic_order"])
    stages[0]["saved_candidate_ambiguity"]=True
    check("ambiguous_projection_not_unique_order",projected_order(stages,"core_first")["relations"][0]["status"]=="AMBIGUOUS_CANDIDATE_ORDER")
    for name, call in [("missing_input_mask", lambda: source_support({"x":a["x"]},names)),
                       ("duplicate_path", lambda: check_path([[1,1],[1,1]],n,n)),
                       ("out_of_bounds", lambda: check_path([[n,1]],n,n)),
                       ("reverse_within_stage", lambda: check_path([[2,1],[1,2]],n,n))]:
        try:
            call()
        except ValueError:
            check(name,True)
        else:
            check(name,False)
    support_test = copy.deepcopy(a)
    fingers = [i for i,name in enumerate(names) if name.startswith("R_") and "_angle" in name and "elbow" not in name]
    support_test["mask"][0,fingers] = False
    support_test["mask"][0,fingers[:10]] = True
    check("original_10_angle_rule", bool(source_support(support_test,names)[0,0]))
    support_test["mask"][0,fingers[9]] = False
    check("nine_angles_unsupported", not source_support(support_test,names)[0,0])
    # No lookup by hidden query word; explicit target word is the only selector.
    stub = type("Stub",(),{})(); stub.phase_refs = {}
    bank = {"words":[{"word_id":"chosen","guide_candidates":[],"references":[
        {"legacy_alias":"r1","clip_id":"clip1","source":"source1","phase_reference_available":False}]}]}
    stub.query_truth = {"word_id":"other","kind":"wrongshape","target_mask":[False]}
    first = compare_practice("chosen","q",bank,stub)
    stub.query_truth = {"word_id":"chosen","kind":"expectedshape","target_mask":[True]}
    check("query_truth_cannot_override_routing", first == compare_practice("chosen","q",bank,stub))
    check("wrong_explicit_word_not_replaced", compare_practice("absent","q",bank,stub)["status"] == "UNSUPPORTED_PRACTICE_WORD")
    bank["words"][0]["references"].append({"legacy_alias":"q","clip_id":"same","source":"source","phase_reference_available":False})
    routed = compare_practice("chosen","q",bank,stub)
    check("self_reference_exclusion", routed["references"][-1]["status"] == "SELF_REFERENCE_EXCLUDED")
    check("unsupported_ref_not_match", routed["references"][0]["status"] == "PHASE_COMPARISON_UNSUPPORTED")
    check("word_does_not_create_observation", all("methods" not in r or not r["methods"] for r in routed["references"]))
    class CompleteStub:
        def __init__(self):
            event={"id":"r1_phase", "region_type":"observed_physical_core", "core":[2,5],
                   "source_segment_id":"parent", "kind":"observed_kind"}
            self.phase_refs={"r1":{"parents":[{"segment_id":"parent","core":[2,5]}],"regions":[event]}}
            self.query_truth={"word_id":"wrong","kind":"wrongshape","target_mask":[False]}
        def clip(self,alias):return a
        def paths(self,query,reference):
            return {"global_dtw":path,"core_first":{"parent":{"path":path,"reference_core":[2,5],"saved_state":"saved"}},"sources":{}}
    full_stub=CompleteStub()
    bank["words"][0]["references"][0]["phase_reference_available"]=True
    first=compare_practice("chosen","q",bank,full_stub)
    full_stub.query_truth={"word_id":"chosen","kind":"expected","target_mask":[True]}
    second=compare_practice("chosen","q",bank,full_stub)
    check("query_annotation_invariance_with_usable_evidence",first==second
          and first["references"][0]["methods"]["global_dtw"]["stages"][0]["seeds"][0]["features"]["R_index_only"]["confidence_qualified_weight"]==4)
    translated=copy.deepcopy(a)
    for side in ("L","R"):
        translated["x"][:,names.index(side+"_hand_wrist_x")]+=.002
    rel=measured_relations(translated,a,path,[2,5])
    check("common_body_translation_is_architectural_only",all(link["raw_difference"]==0
        for feature in rel for row in feature["frames"] for link in row["links"]))
    none_bank={"words":[{"word_id":"new","guide_candidates":[],"references":[
        {"legacy_alias":None,"clip_id":"newclip","source":"source","phase_reference_available":False}]}]}
    check("none_aliases_are_not_self_reference",compare_practice("new",None,none_bank,stub)["references"][0]["status"]
          =="PHASE_COMPARISON_UNSUPPORTED")
    return {"status":"PASS","checks":checks,"count":len(checks),"code_sha256":sha(__file__),
            "educational_accuracy":None,"new_inference":0}


def freeze_execution():
    tasks = read(U / "COMPARATOR_TASKS.json")
    pins = dict(tasks["saved_path_pins"])
    pins.update(read(U / "COMPARATOR_FROZEN.json")["pins"])
    for p in [Path(__file__), U/"COMPARATOR_FROZEN.json",U/"COMPARATOR_FREEZE_ACCEPTANCE.json",
              U/"REFERENCE_PHASES.json",U/"COMPARATOR_TASKS.json",U/"COMPARATOR_SELFTEST_FROZEN.json",
              CORE/"RESULT_MANIFEST.json",REL/"PAIR_PROVENANCE.json"]:
        pins[str(p)] = sha(p)
    old = {r["alias"]:r for r in read(OLD/"LEARNING_INPUTS.json")["rows"]}
    for task in tasks["tasks"]:
        row = old[task["query_alias"]]
        if sha(row["input"]) != row["input_sha256"]:
            raise ValueError("Input changed")
        pins[row["input"]] = row["input_sha256"]
    save_new(U / "COMPARATOR_EXECUTION.json", {"status":"FROZEN_FINAL_CODE_AND_EXACT_SAVED_INPUTS_BEFORE_NEW_REPLAY",
        "pins":pins,"pairs":36,"original_parent_instances":117,"refined_child_instances":tasks["refined_child_instances"],
        "reference_child_frames_per_method_seed":1080,"feature_cells_per_method_seed":3240,
        "methods":list(METHODS),"seeds":list(SEEDS),"query_annotations_runtime":False})


def verify_execution():
    frozen = read(U / "COMPARATOR_EXECUTION.json")
    for path,digest in frozen["pins"].items():
        if sha(path) != digest:
            raise ValueError("Frozen execution input changed: "+path)
    return frozen


def controlled_saved_tests(store):
    """Transform saved arrays only; mapping is supplied, no aligner accuracy claim."""
    results = []
    clips = read(PHYSICAL)["clips"]  # technical-control case construction only
    for clip in clips:
        alias = clip["alias"]; original = store.clip(alias); n=len(original["frame_id"])
        cores = [r["core"] for r in clip["regions"] if r["region_type"] == "observed_physical_core"]
        indices = sorted({i for a,b in cores for i in range(a,b+1)})
        # Whole recorded axis is retained, not a chosen high-score slice.
        core=[0,n-1]; identity=[[i,i] for i in range(n)]
        source_index=np.repeat(np.arange(n),1+np.arange(n)%3)
        warped={k:copy.deepcopy(v) for k,v in original.items()}
        for k in ("x","mask","support"):
            warped[k]=original[k][source_index].copy()
        warped["probs"]={s:original["probs"][s][source_index].copy() for s in SEEDS}
        warped["frame_id"]=np.arange(len(source_index));warped["timestamp_us"]=np.arange(len(source_index))*33333
        known_path=[[qi,int(ri)] for qi,ri in enumerate(source_index)]
        for seed in SEEDS:
            a=local_compare(original,original,identity,core,seed)
            b=local_compare(warped,original,known_path,core,seed)
            invariant=all(np.isclose(a["features"][f]["confidence_qualified_weight"],b["features"][f]["confidence_qualified_weight"])
                and not b["features"][f]["contains_difference_candidate"] for f in TARGETS)
            results.append({"case":f"{alias}/{seed}/identity_timewarp","pass":bool(invariant),
                "kind":"saved_probability_timewarp_technical","expected_frames":n,
                "usable_weights":{f:a["features"][f]["confidence_qualified_weight"] for f in TARGETS}})
            for j,feature in enumerate(TARGETS):
                eligible=[i for i in indices if original["support"][i,j]
                    and np.isfinite(original["probs"][seed][i,j])
                    and max(original["probs"][seed][i,j],1-original["probs"][seed][i,j])>=CONFIDENCE]
                if not eligible:
                    results.append({"case":f"{alias}/{seed}/{feature}","kind":"shape_missing_control",
                        "status":"NO_ELIGIBLE_FIXED_SOURCE_FRAME","pass":None});continue
                index=eligible[0]; altered=copy.deepcopy(original)
                altered["probs"][seed][index,j]=.01 if original["probs"][seed][index,j]>=.5 else .99
                changed=local_compare(altered,original,identity,core,seed)
                difference=changed["features"][feature]["weighted_states"].get("ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE",0)
                altered["support"][index,j]=False
                missing=local_compare(altered,original,identity,core,seed)
                masked=missing["features"][feature]["weighted_states"].get("QUERY_UNSUPPORTED",0)
                original_missing=a["features"][feature]["weighted_states"].get("QUERY_UNSUPPORTED",0)
                dropped=local_compare(original,original,[p for p in identity if p[1]!=index],core,seed)
                no_path=dropped["features"][feature]["weighted_states"].get("NO_SAVED_CORRESPONDENCE",0)
                pass_test=bool(np.isclose(difference,1) and np.isclose(masked-original_missing,1)
                    and np.isclose(no_path,1) and missing["features"][feature]["expected_reference_frames"]==n)
                results.append({"case":f"{alias}/{seed}/{feature}","kind":"shape_missing_control","source_frame":index,
                    "source_pts_us":int(original["timestamp_us"][index]),"difference_weight":difference,
                    "masked_weight_increment":masked-original_missing,"removed_path_weight":no_path,"pass":pass_test})
    # Order bookkeeping can detect reversed independent projections; global DTW remains constrained.
    for name,spans,expected in [("forward",[1,4,7],"FORWARD_DISJOINT"),
                                ("reversal",[7,4,1],"REVERSE_DISJOINT"),
                                ("overlap",[4,4,4],"OVERLAPPING_PROJECTIONS")]:
        stages=[{"reference_event":{"id":str(i),"core":[i,i]},"saved_path":[[q,i]]} for i,q in enumerate(spans)]
        order=projected_order(stages,"core_first")
        results.append({"case":"projection_"+name,"kind":"supplied_projection_technical",
            "pass":all(r["status"]==expected for r in order["relations"]),"result":order})
    return {"status":"PASS_TECHNICAL_ONLY" if all(r["pass"] is not False for r in results) else "FAIL",
        "cases":results,"executed":sum(r["pass"] is not None for r in results),
        "unsupported":sum(r["pass"] is None for r in results),"actual_learner_errors":0,
        "new_inference":0,"automatic_alignment_on_transformed_input_tested":False}


def summarize(results):
    groups={}
    for result in results:
        role=result["test_role"]
        for ref in result["references"]:
            if ref.get("status") != "LOCAL_ESTIMATED_EVIDENCE":continue
            for method,data in ref["methods"].items():
                for stage in data["stages"]:
                    for seed in stage["seeds"]:
                        key=f"{role}/{method}/{seed['seed']}"
                        row=groups.setdefault(key,{"parent_ids":set(),"pair_ids":set(),"child_events":0,
                            "expected_feature_frame_cells":0,"confidence_qualified_weight":0.,
                            "input_supported_weight":0.,"agreement_weight":0.,"difference_candidate_weight":0.,
                            "retained_despite_global_gate_weight":0.,"candidate_child_events":0})
                        row["pair_ids"].add((result["query"],ref["reference_alias"]))
                        row["parent_ids"].add((result["query"],ref["reference_alias"],stage["parent_segment_id"]))
                        row["child_events"]+=1
                        any_difference=False
                        for feature in seed["features"].values():
                            states=feature["weighted_states"]
                            row["expected_feature_frame_cells"]+=feature["expected_reference_frames"]
                            row["confidence_qualified_weight"]+=feature["confidence_qualified_weight"]
                            row["input_supported_weight"]+=feature["input_supported_and_finite_weight"]
                            row["agreement_weight"]+=states.get("ESTIMATED_FEATURE_AGREEMENT",0)
                            row["difference_candidate_weight"]+=states.get("ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE",0)
                            row["retained_despite_global_gate_weight"]+=feature["local_qualified_weight_retained_despite_global_gate"]
                            any_difference|=feature["contains_difference_candidate"]
                        row["candidate_child_events"]+=any_difference
    for row in groups.values():
        row["pairs"]=len(row.pop("pair_ids"));row["original_parent_instances"]=len(row.pop("parent_ids"))
        row["confidence_coverage"]=row["confidence_qualified_weight"]/row["expected_feature_frame_cells"]
        row["input_coverage"]=row["input_supported_weight"]/row["expected_feature_frame_cells"]
    return groups


def audit_projection(results):
    """Query physical labels enter only after frozen runtime outputs exist."""
    physical={c["alias"]:c for c in read(PHYSICAL)["clips"]}
    output=[]
    for result in results:
        query_truth=physical[result["query"]]
        for ref in result["references"]:
            if ref.get("status")!="LOCAL_ESTIMATED_EVIDENCE":continue
            for method,data in ref["methods"].items():
                for stage in data["stages"]:
                    core=stage["reference_event"]["core"];kind=stage["reference_event"]["kind"]
                    by=defaultdict(list)
                    for qi,ri in stage["saved_path"]:by[ri].append(qi)
                    counts=Counter();trace=[]
                    for ri in range(core[0],core[1]+1):
                        if not by[ri]:counts["unmapped"]+=1;trace.append({"reference_frame":ri,"query_frame":None,"status":"unmapped","weight":1});continue
                        for qi in by[ri]:
                            hits=[r for r in query_truth["regions"] if r["outer"][0]<=qi<=r["outer"][1]]
                            state="unreviewed_or_uncertain"
                            if len(hits)==1:
                                h=hits[0]
                                certain=(h.get("region_type")=="observed_physical_core" and h.get("core")
                                         and h["core"][0]<=qi<=h["core"][1] and h.get("kind") is not None)
                                state=("same_observed_kind" if h["kind"]==kind else "different_observed_kind") \
                                    if certain else "boundary_or_uncertain"
                            weight=1/len(by[ri]);counts[state]+=weight
                            trace.append({"reference_frame":ri,"query_frame":qi,"status":state,"weight":weight})
                    output.append({"query":result["query"],"reference":ref["reference_alias"],"role":result["test_role"],
                        "method":method,"reference_event":stage["reference_event"]["id"],"expected_reference_frames":core[1]-core[0]+1,
                        "weighted_states":dict(counts),"trace":trace,"query_annotations_used_for_runtime":False,
                        "same_kind_does_not_validate_shape_or_semantics":True})
    return {"status":"SAVED_AUTOMATIC_PROJECTION_DESCRIPTIVE_AUDIT","events":output,
            "query_truth_source":str(PHYSICAL),"query_truth_sha256":sha(PHYSICAL),"educational_accuracy":None}


def replay():
    frozen=verify_execution();bank=read(U/"REFERENCE_BANK.json");store=RuntimeStore(bank)
    results=[];routed_cache={};routing_results=[]
    for request in bank["development_queries"]:
        query=request["query"];alias=query.get("legacy_alias") or query["alias"]
        r=compare_practice(request["practice_word_id"],alias,bank,store)
        routed_cache[(request["practice_word_id"],alias)]=r
        routing_results.append({"practice_word_id":request["practice_word_id"],"query":alias,
            "reference_statuses":[{"reference_alias":ref["reference_alias"],"status":ref["status"]}
                                  for ref in r["references"]],
            "has_phase_local_evidence":any(ref["status"]=="LOCAL_ESTIMATED_EVIDENCE" for ref in r["references"]),
            "whole_sign_verdict":None})
    for task in read(U/"COMPARATOR_TASKS.json")["tasks"]:
        r=routed_cache.get((task["practice_word_id"],task["query_alias"]))
        if r is None:r=compare_practice(task["practice_word_id"],task["query_alias"],bank,store)
        r["test_role"]=task["role"];results.append(r)
    summary=summarize(results)
    for method in METHODS:
        for seed in SEEDS:
            cells=sum(r["expected_feature_frame_cells"] for k,r in summary.items() if k.endswith(f"/{method}/{seed}"))
            if cells!=3240:raise ValueError("Frozen local cell denominator changed")
    output={"status":"COMPLETED_SAVED_AUTOMATIC_LOCAL_COMPARISON_NOT_EDUCATIONAL_VALIDATION",
        "schema":SCHEMA,"execution_sha256":sha(U/"COMPARATOR_EXECUTION.json"),"code_sha256":sha(__file__),
        "bank_query_denominator":len(routing_results),
        "phase_supported_development_queries":sum(r["has_phase_local_evidence"] for r in routing_results),
        "phase_unsupported_development_queries":sum(not r["has_phase_local_evidence"] for r in routing_results),
        "routing_results":routing_results,"scope":"historical FIT/descriptive DEV, not fresh or learner errors",
        "summary":summary,"results":results,"new_inference":0,"new_training":0,
        "guide_override":False,"whole_sign_correctness":None,"educational_accuracy":None}
    save_new(U/"COMPARISON_RESULTS.json",output)
    # Evaluation only occurs after runtime output has been persisted.
    save_new(U/"PROJECTION_AUDIT.json",audit_projection(results))
    save_new(U/"COMPARATOR_CONTROLS.json",controlled_saved_tests(store))
    verify_execution()
    print(json.dumps({"status":output["status"],"summary":summary}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","selftest","freeze","replay"))
    args=parser.parse_args()
    if args.mode=="prepare":prepare()
    elif args.mode=="selftest":
        result=selftest();save_new(U/"COMPARATOR_SELFTEST.json",result);print(json.dumps({"status":result["status"],"checks":result["count"]}))
    elif args.mode=="freeze":freeze_execution()
    else:replay()


if __name__=="__main__":main()
