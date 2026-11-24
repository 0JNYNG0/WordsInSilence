"""Independent vectorized join of saved trace labels and context cells.

No stage label/score is generated. No runtime, model, DP or producer import.
Trace weights/labels are copied; numpy unique/bincount groups trace-major cells,
independent of the producer's dimension-major event loops.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import ast
import gzip
import hashlib
import json
import math
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
U = HERE.parent
OPS = U.parent
AUDIT = U / "safety_audit"
OLD = OPS / "word100_alignment_missing_split_20261101/EVALUATION.json"
PHYSICAL = OPS / "word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json"
OUTPUT = HERE / "SAFETY_RESULT_VERIFICATION.json"
STATE = ["DIRECT_COMPLETE", "PARTIAL_SHARED", "NO_SHARED"]
TEMPORAL = ["DIRECT_CELL_NO_GAP_CONTEXT", "BOUNDED_BY_DISTINCT_OBSERVATIONS",
            "OPEN_BOUNDARY", "NO_ANCHORS", "MAPPING_COMPRESSED_OR_OVERLAPPING"]
SUPPORT = ["SHARED", "QUERY_UNOBSERVED", "REFERENCE_UNOBSERVED", "BOTH_UNOBSERVED"]
HYPOTHESES = ["DIRECT_BOTH_AXES", "BOUNDED_ANY_AXIS", "BOUNDED_BOTH_AXES"]
FIELDS = ["role", "seed", "group", "frozen_stage_label", "pair_support", "query_observation",
          "reference_observation", "query_context", "reference_context"]
checks, failures, pins = 0, [], {}
group_table, feature_table, hypothesis_table = Counter(), Counter(), Counter()
source_weights = Counter()
expected_examples, representatives, group_dims = {}, {}, {}


def read(p):
    p = Path(p)
    raw = p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix == ".gz" else raw.decode("utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(p, expected=None):
    p = str(Path(p).resolve())
    actual = sha(p)
    check(expected is None or expected == actual, "SHA " + p)
    pins[p] = actual


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        failures.append(label)


def require(condition, label):
    check(condition, label)
    if not condition:
        raise AssertionError(label)


def resolve(obj, pointer):
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def states(axis):
    shared = np.asarray(axis["matched_support"], dtype=bool)
    partial = np.asarray(axis["mixed_counterpart_missing"], dtype=bool)
    return np.where(shared, np.where(partial, 1, 0), 2)


def contexts(info, state):
    result = np.zeros(state.shape, dtype=np.int64)
    episode_index = np.full(state.shape, -1, dtype=np.int64)
    for i, e in enumerate(info["episodes"]):
        view = (slice(e["start"], e["end"] + 1), e["dimension"])
        require(np.all(episode_index[view] == -1), "Saved context intervals do not overlap")
        result[view] = TEMPORAL.index(e["temporal_context"])
        episode_index[view] = i
    check(np.array_equal(episode_index == -1, state == 0), "DIRECT cells precisely lack gap episodes")
    return result, episode_index


def vector_add(counter, prefix, codes, weights, labels, dimension_labels=None):
    unique, inverse = np.unique(codes, axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=weights)
    for key, amount in zip(unique, sums):
        label, support, qo, ro, qt, rt = (int(x) for x in key[:6])
        decoded = prefix + (labels[label], SUPPORT[support], STATE[qo], STATE[ro], TEMPORAL[qt], TEMPORAL[rt])
        if dimension_labels is not None:
            dim = int(key[6])
            decoded += (dim, dimension_labels[dim])
        counter[decoded] += float(amount)


def compare_table(saved, expected, fields, name):
    actual = {tuple(r[k] for k in fields): r["weight"] for r in saved}
    require(len(actual) == len(saved), name + " unique table keys")
    check(set(actual) == set(expected), name + " exact category inventory")
    errors = [abs(actual[k] - expected[k]) for k in set(actual) & set(expected)]
    check(all(math.isclose(actual[k], expected[k], abs_tol=1e-8, rel_tol=1e-12)
              for k in set(actual) & set(expected)), name + " weighted sums")
    return dict(rows=len(actual), max_absolute_difference=max(errors, default=0.0))


def physical_description(physical, alias, frame):
    index, clip = next((i, c) for i, c in enumerate(physical["clips"]) if c["alias"] == alias)
    found = [dict(json_pointer=f"/clips/{index}/regions/{j}", region=r)
             for j, r in enumerate(clip["regions"])
             if r["region_type"] == "observed_physical_core" and r["core"][0] <= frame <= r["core"][1]]
    return dict(source=str(PHYSICAL), alias=alias, frame=frame, certain_regions=found)


def inspect_example(ex, old, physical):
    pin(ex["raw_evidence_source"])
    pin(ex["context_source"])
    raw, context = read(ex["raw_evidence_source"]), read(ex["context_source"])
    trace = resolve(old, ex["stage_audit_pointer"])
    ri, qi, label, weight = trace
    check(label == ex["stage_label"] == "different_observed_kind" and weight == ex["frozen_link_weight"],
          "Example frozen trace label/weight exact")
    group, dim = ex["group"], ex["dimension"]
    check(ex["stage_audit_source"] == str(OLD) and ex["seed"] == 42 and ex["not_an_educational_error_label"] is True,
          "Example audit-only provenance")
    for axis, i in (("query", qi), ("reference", ri)):
        cell = ex[axis + "_cell"]
        a = raw["evidence"]["groups"][group][axis]
        check(cell["frame_id"] == a["frame_id"][i] and cell["timestamp_us"] == a["timestamp_us"][i] and
              cell["observation"] == STATE[int(states(a)[i, dim])], "Example native cell identity/observation")
        for field in ("mean", "max"):
            check(resolve(raw, cell[field + "_pointer"]) == cell[field + "_residual"] == a[field + "_residual"][i][dim],
                  "Example exact original residual pointer")
        check(cell["residual_is_conditional_axis_mean_not_specific_link"] is True,
              "Example axis aggregate not mislabeled as per-link residual")
        if cell["context_pointer"] is None:
            check(cell["observation"] == "DIRECT_COMPLETE" and cell["context"] == TEMPORAL[0],
                  "Example direct cell context")
        else:
            e = resolve(context, cell["context_pointer"])
            check(e["dimension"] == dim and e["start"] <= i <= e["end"] and
                  e["temporal_context"] == cell["context"], "Example exact saved episode pointer")
        physical_saved = ex[axis + "_physical_audit_only"]
        check(physical_saved == physical_description(physical, ex[axis], cell["frame_id"]),
              "Example public physical-region pointers exact; audit-only")
    return True


def static_review():
    p = AUDIT / "audit.py"
    source = p.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    imports = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            imports.add(n.module)
    check(imports <= {"pathlib", "collections", "datetime", "gzip", "hashlib", "json", "sys"},
          "Audit imports no runtime/model/feature/alignment implementation")
    function_calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check(not function_calls.intersection({"phase_audit", "align", "solve", "train", "predict", "infer", "eval", "exec"}),
          "No stage-score/model/path-search call")
    runtime = (U / "run.py").read_text(encoding="utf-8-sig")
    check("reference=copy.deepcopy(refs[rec['reference']])" in runtime and
          "context.annotate(source['evidence'],source['path'],reference)" in runtime and
          "PHYSICAL_CORRESPONDENCE" not in runtime and "safety_audit" not in runtime,
          "Runtime remains isolated from query physical audit")


def main():
    contract = read(AUDIT / "PROSPECTIVE_CONTRACT.json")
    result = read(AUDIT / "RESULT.json")
    pin(AUDIT / "PROSPECTIVE_CONTRACT.json", result["contract_sha256"])
    pin(AUDIT / "RESULT.json")
    for p, h in contract["pins"].items():
        pin(p, h)
    pin(HERE / "RESULT_VERIFICATION.json")
    require(read(HERE / "RESULT_VERIFICATION.json")["status"] == "PASS", "Underlying full context verification passed")
    static_review()
    old, physical = read(OLD), read(PHYSICAL)
    saved_stage = {(r["query"], r["reference"]): (i, r) for i, r in enumerate(old["pairs"]) if r["method"] == "global_dtw"}
    require(len(saved_stage) == 36, "Exact36 original global trace records")
    tasks = read(U / "TASKS.json")["records"]
    manifest = read(U / "RESULT_MANIFEST.json")
    cmap = {(r["query"], r["reference"], r["seed"]): r for r in manifest["contexts"]}
    labels = sorted({trace[2] for _, r in saved_stage.values() for e in r["audit"]["events"] for trace in e["trace"]})
    expected_sources = []
    for task in sorted(tasks, key=lambda r: (r["query"], r["reference"], r["seed"])):
        query, reference, seed = task["query"], task["reference"], task["seed"]
        ci = cmap[query, reference, seed]
        raw, context = read(task["path"]), read(ci["path"])
        si, sr = saved_stage[query, reference]
        role, word = sr["role"], sr["word_id"]
        expected_sources.append(dict(query=query, reference=reference, seed=seed, role=role,
                                     raw=task["path"], context=ci["path"], stage_audit_pointer=f"/pairs/{si}/audit"))
        flat = [(ei, ti, trace) for ei, event in enumerate(sr["audit"]["events"]) for ti, trace in enumerate(event["trace"])]
        ri = np.array([t[0] for _, _, t in flat], dtype=int)
        qi = np.array([t[1] for _, _, t in flat], dtype=int)
        label_codes = np.array([labels.index(t[2]) for _, _, t in flat], dtype=int)
        weights = np.array([t[3] for _, _, t in flat], dtype=float)
        fixed_path = {tuple(p) for p in raw["path"]}
        require(all((int(q), int(r)) in fixed_path for q, r in zip(qi, ri)), "Frozen traces all address the unchanged path")
        by_ref = Counter(p[1] for p in raw["path"])
        check(np.allclose(weights, [1.0 / by_ref[int(r)] for r in ri], atol=1e-15, rtol=0),
              "Frozen trace weights1/N mapped query links per original reference frame")
        for ei, event in enumerate(sr["audit"]["events"]):
            event_trace = [t for e, _, t in flat if e == ei]
            reference_frames = range(event["reference_core"][0], event["reference_core"][1] + 1)
            counter = Counter()
            for r, _, _, weight in event_trace:
                counter[r] += weight
            check(set(counter) == set(reference_frames) and all(abs(v - 1) < 1e-12 for v in counter.values()),
                  "Every frozen event reference frame has weight1")
        source_weights[role, seed] += float(weights.sum())
        for group in sorted(raw["evidence"]["groups"], key=lambda g: (g != "learned3", g)):
            axes = raw["evidence"]["groups"][group]
            qa, ra = axes["query"], axes["reference"]
            dims = len(qa["dimension_labels"])
            require(group not in group_dims or group_dims[group] == dims, "Stable group dimension inventory")
            group_dims[group] = dims
            qs, rs = states(qa), states(ra)
            qt, qe = contexts(context["annotation"]["groups"][group]["query"], qs)
            rt, re = contexts(context["annotation"]["groups"][group]["reference"], rs)
            qstate, rstate, qcontext, rcontext = qs[qi], rs[ri], qt[qi], rt[ri]
            qo = np.asarray(qa["own_observed"], bool)[qi]
            ro = np.asarray(ra["own_observed"], bool)[ri]
            support = np.where(qo, np.where(ro, 0, 2), np.where(ro, 1, 3))
            codes = np.column_stack((np.repeat(label_codes, dims), support.ravel(), qstate.ravel(), rstate.ravel(),
                                     qcontext.ravel(), rcontext.ravel()))
            cell_weights = np.repeat(weights, dims)
            prefix = (role, seed, group)
            vector_add(group_table, prefix, codes, cell_weights, labels)
            vector_add(feature_table, prefix, np.column_stack((codes, np.tile(np.arange(dims), len(flat)))),
                       cell_weights, labels, qa["dimension_labels"])
            applies = [(qstate == 0) & (rstate == 0),
                       (qcontext == 1) | (rcontext == 1), (qcontext == 1) & (rcontext == 1)]
            for h, mask in zip(HYPOTHESES, applies):
                for li, label in enumerate(labels):
                    amount = float(np.sum(np.broadcast_to(weights[:, None], mask.shape)[mask & (label_codes[:, None] == li)]))
                    if amount:
                        hypothesis_table[prefix + (h, label)] += amount
                exkey = (h, word)
                if role == "development" and seed == 42 and exkey not in expected_examples:
                    indices = np.argwhere((mask & (label_codes[:, None] == labels.index("different_observed_kind"))).T)
                    if len(indices):
                        dim, ti = (int(x) for x in indices[0])
                        ei, trace_index, _ = flat[ti]
                        expected_examples[exkey] = (query, reference, group, dim, f"/pairs/{si}/audit/events/{ei}/trace/{trace_index}")
            if role == "development" and seed == 42:
                for ti, (ei, trace_index, trace) in enumerate(flat):
                    category = "same" if trace[2] == "same_observed_kind" else "different" if trace[2] == "different_observed_kind" else "unknown"
                    if category not in representatives:
                        r, q, label, weight = trace
                        representatives[category] = dict(
                            selection="First sorted DEV seed42 task/group learned3 first/dimension0/trace row for this label category; no residual selection",
                            query=query, reference=reference, seed=seed, group=group, dimension=0,
                            label_copied=label, trace=trace, source=str(OLD),
                            trace_pointer=f"/pairs/{si}/audit/events/{ei}/trace/{trace_index}",
                            raw_source=task["path"], context_source=ci["path"],
                            query_mean_pointer=f"/evidence/groups/{group}/query/mean_residual/{q}/0",
                            reference_mean_pointer=f"/evidence/groups/{group}/reference/mean_residual/{r}/0",
                            query_mean=qa["mean_residual"][q][0], reference_mean=ra["mean_residual"][r][0],
                            query_context=TEMPORAL[int(qt[q, 0])], reference_context=TEMPORAL[int(rt[r, 0])],
                            query_physical_audit_only=physical_description(physical, query, qa["frame_id"][q]),
                            reference_physical_audit_only=physical_description(physical, reference, ra["frame_id"][r]))
    check(result["source_records"] == expected_sources and len(expected_sources) == 108, "All108 source/trace pointers exact")
    check(result["group_dimensions"] == group_dims and sum(group_dims.values()) == 70 and len(group_dims) == 16,
          "Exact16 groups/70 correlated feature dimensions")
    expected_denoms = []
    for role, frames in (("development", 360), ("fit", 720)):
        for seed in (42, 43, 44):
            check(abs(source_weights[role, seed] - frames) < 1e-9, f"{role}/{seed} unique-reference trace-weight denominator")
            for group, dims in sorted(group_dims.items()):
                expected_denoms.append(dict(role=role, seed=seed, group=group, dimensions=dims,
                                           frozen_reference_frames=frames, feature_cell_opportunities=frames * dims))
    check(result["denominators"] == expected_denoms, "Every360/720 times dimension denominator, seeds separate")
    table_differences = dict(
        group=compare_table(result["group_crosstab"], group_table, FIELDS, "Group cross-tab"),
        feature=compare_table(result["feature_crosstab"], feature_table, FIELDS + ["dimension", "dimension_label"], "Feature cross-tab"),
        hypothesis=compare_table(result["hypothesis_crosstab"], hypothesis_table,
                                 ["role", "seed", "group", "hypothesis", "frozen_stage_label"], "Hypothesis cross-tab"))
    for outcome in result["hypothesis_outcomes"]:
        h = outcome["hypothesis"]
        aggregate = Counter()
        for (role, seed, group, hypothesis, label), amount in hypothesis_table.items():
            if (role, seed, hypothesis) == ("development", 42, h):
                aggregate[label] += amount
        actual = outcome["primary_seed42_feature_cell_weights"]
        check(set(actual) == set(aggregate) and all(math.isclose(actual[k], aggregate[k], abs_tol=1e-8) for k in aggregate),
              "Outcome copied from independent feature-cell join " + h)
        check(outcome["different_known_stage_counterexample_exists"] == (aggregate["different_observed_kind"] > 0) and
              outcome["no_counterexample_does_not_establish_safety"] is True and
              outcome["not_a_runtime_policy_or_educational_accuracy"] is True, "Outcome narrow interpretation " + h)
    produced_examples = {(e["hypothesis"], e["word_id"]): (e["query"], e["reference"], e["group"], e["dimension"], e["stage_audit_pointer"])
                         for e in result["examples"]}
    check(len(produced_examples) == len(result["examples"]) and produced_examples == expected_examples,
          "All and only preregistered first counterexamples, no residual-dependent selection")
    for example in result["examples"]:
        inspect_example(example, old, physical)
    require(set(representatives) == {"same", "different", "unknown"}, "Same/different/unknown representative source rows")
    for sample in representatives.values():
        check(resolve(old, sample["trace_pointer"]) == sample["trace"], "Representative exact frozen trace pointer")
        raw = read(sample["raw_source"])
        check(resolve(raw, sample["query_mean_pointer"]) == sample["query_mean"] and
              resolve(raw, sample["reference_mean_pointer"]) == sample["reference_mean"], "Representative source-value pointers")
    check(result["frozen_stage_summary_copied_not_recomputed"] == {k: v for k, v in old["summary"].items() if "/global_dtw/" in k},
          "Old stage summary copied exactly; no score recalculation")
    check(all(result[k] == 0 for k in ("new_stage_score_calculation", "new_alignment", "new_model_inference", "new_training")) and
          result["new_runtime_policy"] is False and result["query_annotations_audit_only"] is True and
          result["prior_frozen_outputs_unchanged"] is True, "Audit-only no execution/policy claim")
    for p, h in list(pins.items()):
        check(sha(p) == h, "Source unchanged throughout audit " + p)
    return table_differences, result["hypothesis_outcomes"]


if __name__ == "__main__":
    if OUTPUT.exists():
        raise SystemExit("First result must be preserved; no overwrite.")
    error, comparisons, outcomes = None, None, None
    try:
        comparisons, outcomes = main()
    except Exception:
        error = traceback.format_exc()
        failures.append("Verification exception")
    report = dict(schema="wis_independent_safety_join_verification_v1", status="PASS" if not failures else "FAIL",
                  reviewer="Codex semantic_data_audit", at_utc=datetime.now(timezone.utc).isoformat(),
                  checks=checks, failure_count=len(failures), failures=failures[:80], exception=error,
                  method="Trace-major numpy unique/bincount weighted join, compared against producer dimension-major tables. Saved stage labels/weights are never reclassified.",
                  denominators=dict(development_reference_frames_per_seed=360, fit_reference_frames_per_seed=720,
                                    correlated_feature_dimensions=70, seeds=[42, 43, 44],
                                    development_feature_cells_per_seed=25200, development_all_seed_feature_cells=75600,
                                    fit_feature_cells_per_seed=50400, all_roles_and_seeds_feature_cells=226800),
                  source_trace_weight_by_role_seed={str(k): v for k, v in source_weights.items()},
                  table_comparisons=comparisons, hypothesis_outcomes=outcomes,
                  representative_source_pointers=representatives,
                  interpretation=["DIRECT establishes observation availability only, not same physical subphase correspondence.",
                                  "Different-known-kind here is a physical movement/hold subphase contrast between public normal performances; it is not a learner error, invalid sign, or rejectable variation label.",
                                  "Same physical kind is not proof of same occurrence or educational correctness.",
                                  "No BOUNDED counterexample in this frozen sample does not establish safety or accuracy.",
                                  "Three seeds repeat source videos/masks/labels;70 related dimensions are accounting, not independent semantic targets."],
                  actions=dict(new_alignment=0, inference=0, training=0, stage_score_recalculation=0,
                               new_labels=0, new_runtime_policy=False, raw_video_access=0, old_files_modified=0),
                  pins=pins, script_sha256=sha(__file__))
    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(dict(status=report["status"], checks=checks, failures=len(failures), output=str(OUTPUT), sha256=sha(OUTPUT))))
    if failures:
        raise SystemExit(1)
