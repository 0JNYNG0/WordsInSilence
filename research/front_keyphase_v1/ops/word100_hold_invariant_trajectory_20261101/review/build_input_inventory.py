"""Read-only saved-input/interval inventory. No extraction/model/path search."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
import numpy as np

HERE = Path(__file__).resolve().parent
U = HERE.parent
OPS = U.parent
JOINT = OPS / "word100_joint_stage_alignment_20261101"
STAGE = OPS / "word100_stage_feedback_cycle_20261031"
TARGET = OPS / "word100_target_known_feedback_20261101"
CONTEXT = OPS / "word100_alignment_context_review_20261101"
MISSING = OPS / "word100_alignment_missing_split_20261101"
pins = {}


def read(p):
    p = Path(p)
    raw = p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix == ".gz" else raw.decode("utf-8-sig"))


def pin(p, expected=None):
    p = str(Path(p).resolve())
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert expected is None or h == expected, p
    pins[p] = h
    return h


def new(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def geometry(a, names):
    x = np.asarray(a["x"], np.float64)
    known = np.asarray(a["mask"], bool) & np.isfinite(x)
    def columns(keys):
        idx = [names.index(k) for k in keys]
        return x[:, idx], known[:, idx]
    out = {}
    for side, opposite in (("L", "R"), ("R", "L")):
        wrist, wm = columns([side + "_hand_wrist_" + k for k in ("x", "y")])
        out[side + "_wrist_xy"] = (wrist, wm)
        opw, pm = columns([opposite + "_pose_wrist_" + k for k in ("x", "y")])
        elbow, em = columns([opposite + "_elbow_" + k for k in ("x", "y")])
        direction = opw - elbow
        displacement = wrist - elbow
        norm2 = (direction ** 2).sum(axis=1)
        ok = wm.all(1) & pm.all(1) & em.all(1) & (norm2 > 1e-8)
        values = np.zeros((len(x), 2))
        values[ok, 0] = (displacement[ok] * direction[ok]).sum(1) / norm2[ok]
        values[ok, 1] = (direction[ok, 0] * displacement[ok, 1] - direction[ok, 1] * displacement[ok, 0]) / norm2[ok]
        out[side + "_to_" + opposite + "_forearm"] = (values, np.repeat(ok[:, None], 2, axis=1))
    out["hands_delta_xy"] = columns(["hands_wrist_delta_x", "hands_wrist_delta_y"])
    return out


def support(mask):
    point = mask.all(axis=1)
    spans = []
    for want in (True, False):
        changes = np.diff(np.r_[0, (point == want).astype(int), 0])
        spans.append([[int(a), int(b) - 1] for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))])
    return dict(frames=len(point), observed_2d_points=int(point.sum()),
                observed_dimension_cells=int(mask.sum()), possible_dimension_cells=int(mask.size),
                fully_observed=bool(point.all()), observed_runs=spans[0], unknown_runs=spans[1])


tasks = read(JOINT / "TASKS.json")
inputs = read(STAGE / "LEARNING_INPUTS.json")
rows = {r["alias"]: r for r in inputs["rows"]}
references = read(TARGET / "REFERENCE_PHASES.json")
refs = references["clips"]
for p in (JOINT / "TASKS.json", JOINT / "FROZEN.json", JOINT / "run.py", JOINT / "joint_alignment.py",
          STAGE / "LEARNING_INPUTS.json", TARGET / "REFERENCE_PHASES.json", TARGET / "REFERENCE_BANK.json",
          TARGET / "comparator.py", CONTEXT / "NEXT_PLAN_KO.txt", CONTEXT / "FROZEN.json",
          CONTEXT / "EVALUATION.json", CONTEXT / "safety_audit/RESULT.json",
          CONTEXT / "review/RESULT_VERIFICATION.json", CONTEXT / "review/SAFETY_RESULT_VERIFICATION.json",
          MISSING / "READOUT_ONLY_RESULT.json", MISSING / "READOUT_ONLY_CONTRACT.json"):
    pin(p)
for p, h in references["source_pins"].items():
    pin(p, h)
arrays, geoms, clip_inventory, parents = {}, {}, [], []
roles = {t["query_alias"]: t["role"] for t in tasks["clips"]}
assert len(roles) == 16 and len(refs) == 12
for alias in sorted(roles):
    row = rows[alias]
    pin(row["input"], row["input_sha256"])
    with np.load(row["input"], allow_pickle=False) as z:
        keys = z.files
        a = {k: z[k].copy() for k in ("x", "mask", "frame_id", "timestamp_us")}
        schema = {k: dict(shape=list(z[k].shape), dtype=str(z[k].dtype)) for k in z.files}
    n = len(a["x"])
    assert a["x"].shape == a["mask"].shape == (n, 69) and a["mask"].dtype == np.bool_
    assert np.array_equal(a["frame_id"], np.arange(n)) and np.all(np.diff(a["timestamp_us"]) > 0)
    assert len(row["feature_names"]) == 69 and len(set(row["feature_names"])) == 69
    arrays[alias], geoms[alias] = a, geometry(a, row["feature_names"])
    clip_inventory.append(dict(alias=alias, word_id=row["word_id"], role=roles[alias], person=row["person"],
        clip_id=row["clip_id"], input=row["input"], input_sha256=row["input_sha256"], frames=n,
        original_frame_range=[int(a["frame_id"][0]), int(a["frame_id"][-1])],
        timestamp_range_us=[int(a["timestamp_us"][0]), int(a["timestamp_us"][-1])],
        stored_schema=schema, runtime_geometry_fields=["x", "mask", "frame_id", "timestamp_us"],
        rgb_or_targets_used=False, component_support={k: support(v[1]) for k, v in geoms[alias].items()}))
    if alias in refs:
        assert roles[alias] == refs[alias]["role"] == "fit"
        for pi, parent in enumerate(refs[alias]["parents"]):
            lo, hi = parent["core"]
            assert parent["role"] == "core_candidate" and 0 <= lo <= hi < n
            assert parent["linguistic_core"] is None and parent["obligatory_linguistic_order"] is None
            parents.append(dict(reference=alias, word_id=row["word_id"], segment_id=parent["segment_id"],
                reference_core=[lo, hi], frames=hi - lo + 1, source=str(TARGET / "REFERENCE_PHASES.json"),
                pointer=f"/clips/{alias}/parents/{pi}", core_pts_us=parent["core_pts_us"],
                support=parent["support"], core_interpretation=parent["core_interpretation"],
                linguistic_core=None, obligatory_linguistic_order=None,
                child_ids=[r["id"] for r in refs[alias]["regions"] if r.get("source_segment_id") == parent["segment_id"]],
                component_support={k: support(v[1][lo:hi + 1]) for k, v in geoms[alias].items()}))

pair_inventory, parent_instances = [], []
for task in tasks["pairs"]:
    path_source = task["baselines"]["global_dtw"]
    pin(path_source, tasks["pins"][path_source])
    old = read(path_source)
    path = np.asarray(old["path"], dtype=int)
    qn, rn = len(arrays[task["query"]]["x"]), len(arrays[task["reference"]]["x"])
    assert old["query"] == task["query"] and old["reference"] == task["reference"]
    assert path[0].tolist() == [0, 0] and path[-1].tolist() == [qn - 1, rn - 1]
    assert np.all(np.diff(path, axis=0) >= 0) and len(set(map(tuple, path))) == len(path)
    assert set(path[:, 0]) == set(range(qn)) and set(path[:, 1]) == set(range(rn))
    pair_inventory.append({**{k: task[k] for k in ("query", "reference", "role", "word_id")},
                           "source": path_source, "path_cells": len(path), "query_frames": qn, "reference_frames": rn})
    for parent in refs[task["reference"]]["parents"]:
        rlo, rhi = parent["core"]
        keep = np.flatnonzero((path[:, 1] >= rlo) & (path[:, 1] <= rhi))
        assert len(keep) and np.array_equal(keep, np.arange(keep[0], keep[-1] + 1))
        segment = path[keep[0]:keep[-1] + 1]
        qlo, qhi = int(segment[:, 0].min()), int(segment[:, 0].max())
        assert segment[0].tolist() == [qlo, rlo] and segment[-1].tolist() == [qhi, rhi]
        assert set(segment[:, 0]) == set(range(qlo, qhi + 1)) and set(segment[:, 1]) == set(range(rlo, rhi + 1))
        components = {}
        for name in geoms[task["query"]]:
            qm = geoms[task["query"]][name][1][qlo:qhi + 1]
            rm = geoms[task["reference"]][name][1][rlo:rhi + 1]
            components[name] = dict(query=support(qm), reference=support(rm),
                                    whole_parent_comparable=bool(qm.all() and rm.all()))
        parent_instances.append(dict(query=task["query"], reference=task["reference"], role=task["role"], word_id=task["word_id"],
            parent_id=parent["segment_id"], reference_core=[rlo, rhi], query_range_from_saved_path=[qlo, qhi],
            source_path=path_source, path_slice_inclusive=[int(keep[0]), int(keep[-1])], path_cells=len(segment),
            query_frames=qhi - qlo + 1, reference_frames=rhi - rlo + 1,
            boundary_query_also_linked_outside_ref_core=bool(np.any((path[:, 0] == qlo) & (path[:, 1] < rlo)) or
                                                           np.any((path[:, 0] == qhi) & (path[:, 1] > rhi))),
            whole_axes_covered=True, query_truth_used=False, components=components))
assert len(pair_inventory) == 36 and len(parent_instances) == 117

speed_inventory = []
for c in tasks["controls"]:
    if c["control"] not in ("speed_075", "speed_125"):
        continue
    pin(c["baseline"], tasks["pins"][c["baseline"]])
    pin(c["array"], tasks["pins"][c["array"]])
    metadata = read(c["baseline"])
    ids = np.asarray(c["metadata"]["original_indices"], dtype=int)
    with np.load(c["array"], allow_pickle=False) as z:
        q = {k: z[k].copy() for k in ("x", "mask", "frame_id", "timestamp_us")}
    base = arrays[c["alias"]]
    assert np.array_equal(q["x"], base["x"][ids], equal_nan=True) and np.array_equal(q["mask"], base["mask"][ids])
    assert metadata["array_sha256"] == pins[str(Path(c["array"]).resolve())]
    assert np.all(np.diff(ids) >= 0) and ids[0] == 0 and ids[-1] == len(base["x"]) - 1
    speed_inventory.append(dict(name=c["name"], alias=c["alias"], role=c["role"], control=c["control"],
        array=c["array"], metadata_source=c["baseline"], source_frames=len(base["x"]), resampled_frames=len(ids),
        unique_original_frames=len(set(ids.tolist())), repeated_rows=len(ids) - len(set(ids.tolist())),
        omitted_original_frames=len(base["x"]) - len(set(ids.tolist())),
        exact_coordinate_and_mask_source_index_copy=True, original_indices=ids.tolist(),
        resampled_frame_id_is_original_source_identity=bool(np.array_equal(q["frame_id"], base["frame_id"][ids])),
        stored_array_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in q.items()}))
assert len(speed_inventory) == 32

readouts = read(MISSING / "READOUT_ONLY_RESULT.json")["records"]
by_pair = defaultdict(list)
for row in readouts:
    pin(row["path"], row["sha256"])
    by_pair[row["query"], row["reference"]].append(row)
seed_aliases = []
for pair, source_rows in sorted(by_pair.items()):
    assert sorted(r["seed"] for r in source_rows) == [42, 43, 44]
    first = read(source_rows[0]["path"])
    geometry_evidence = {k: v for k, v in first["evidence"]["groups"].items() if k != "learned3"}
    for row in source_rows[1:]:
        current = read(row["path"])
        assert current["path"] == first["path"]
        assert {k: v for k, v in current["evidence"]["groups"].items() if k != "learned3"} == geometry_evidence
    seed_aliases.append(dict(query=pair[0], reference=pair[1], geometry_record_count=1,
                             original_seed_records=[{k: r[k] for k in ("seed", "path", "sha256")} for r in source_rows],
                             path_and_all15_nonlearned_groups_exactly_equal=True))

role_counts = Counter(r["role"] for r in parent_instances)
support_counts = {k: dict(Counter("fully_observed" if r["components"][k]["whole_parent_comparable"] else "has_unknown"
                                for r in parent_instances)) for k in geoms[next(iter(geoms))]}
report = dict(schema="wis_hold_invariant_saved_input_inventory_v1", status="PASS_READ_ONLY_INPUT_AND_SCOPE_AUDIT",
    at_utc=datetime.now(timezone.utc).isoformat(), reviewer="Codex semantic_data_audit",
    totals=dict(public_words=4, public_clips=16, native_frames=sum(r["frames"] for r in clip_inventory),
                fit_reference_clips=12, unique_fit_parents=len(parents), fixed_global_pairs=36,
                parent_instances=117, parent_instances_by_role=dict(role_counts),
                component_instances=117 * 5, geometry_seeds_needed=1, preserved_seed_alias_records=108,
                pair_parent_reference_frame_opportunities=sum(r["reference_frames"] for r in parent_instances),
                pair_parent_query_frame_opportunities=sum(r["query_frames"] for r in parent_instances),
                saved_speed_controls=32),
    component_names=list(geoms[next(iter(geoms))]), feature_names_69=rows["P033"]["feature_names"],
    component_support_inventory=support_counts, clips=clip_inventory, fit_parents=parents,
    pairs=pair_inventory, parent_instances=parent_instances, speed_controls=speed_inventory,
    geometry_seed_aliases=seed_aliases,
    baseline_scope_rule="For each saved FIT parent core [rlo,rhi], take every original global path row with rlo<=r<=rhi. Monotonicity makes this an existing contiguous slice. Its min/max query indices define the query interval; preserve every original point on both closed intervals. No query GT or new path search.",
    baseline_scope_validation="All117 parent instances have a contiguous original-path slice with endpoints [qlo,rlo] and[qhi,rhi] and complete integer query/ref axis coverage. Query boundary points can also link outside the chosen reference core; this ambiguity is retained.",
    feature_formula="Reuse saved L/R_hand_wrist_xy and hands_wrist_delta_xy. Hand-to-opposite-forearm: d=opposite_pose_wrist-opposite_elbow; v=hand_wrist-opposite_elbow; along=dot(v,d)/dot(d,d); cross=(d.x*v.y-d.y*v.x)/dot(d,d). Both output coordinates known only when all six source coordinates are known/finite and dot(d,d)>1e-8. This is existing joint_alignment.features geometry, not a new direction normalization.",
    geometry_reuse="All15 nonlearned saved evidence groups and every original global path are exactly equal across42/43/44 for each36pair. One geometry comparison may be indexed by all3 original seed records. Keep learned3 and old shape/readout evidence distinct and unchanged.",
    limits=["Public reused development data, not fresh independent evaluation.",
            "FIT parent core spans are sampled RGB physical drafts, not dense truth or mandatory linguistic stages.",
            "Direct geometric observations do not identify educational correctness.",
            "speed075 preserves all original sample points by repetition; speed125 removes samples and can change discrete-curve distances at turns. Do not require universal exact0 for subsampling.",
            "Do not bridge missing runs or turn the maximum over remaining observed links into a complete-trajectory agreement claim."],
    pins=pins, protected=tasks["protected"],
    actions=dict(raw_extraction=0, model_inference=0, alignment_or_path_search=0, scoring=0, training=0,
                 raw_video_access=0, private_or_mentor_access=0, old_files_modified=0),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
destination = HERE / "SOURCE_INVENTORY.json"
new(destination, report)
print(json.dumps(dict(status=report["status"], totals=report["totals"], support=support_counts,
                      output=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), pins=len(pins))))
