"""Independent saved-input verification. Never imports producer or matcher."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

U = Path(__file__).resolve().parents[1]
OPS = U.parent
OUT = U / "review/PREPARATION_DATA_REVIEW.json"
assert not OUT.exists(), "Preserve first independent result."
def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}
checks = []
def check(v, name):
    checks.append({"name": name, "pass": bool(v)})
def eq(a, b):
    return np.array_equal(a, b, equal_nan=True)

d = read(U / "INPUTS.json")
ledger = read(U / "review/EVALUATION_LEDGER.json")
old = read(OPS / "word100_action_feedback_casebook_20261101/INPUTS.json")
rec = read(OPS / "word100_observation_lineage_repair_20261101/RECOVERY_RESULT_MANIFEST.json")
overlays = {r["alias"]: r for r in rec["records"]}
check(set(d["inputs"]) == set(old["rows"]), "exact16_aliases")
check(d["parents"] == old["natural"], "all39_parent_metadata_unchanged")
check(d["scope"] == {"clips":16,"natural_pairs":12,"parent_intervals":39,"saved_controls":126,"new_controls":48}, "scope_exact")
for p,h in d["pins"].items():
    check(sha(p) == h, "input_pin:" + p)
keys = {"x","mask","frame_id","timestamp_us"}
totals = collections.Counter()
bases = {}
for alias, r in d["inputs"].items():
    base = arrays(r["original"])
    base = {k: base[k] for k in keys}
    bases[alias] = base
    merged = arrays(r["merged"])
    check(set(merged) == keys, alias + ":runtime_keys_only")
    check(r["original"] == old["rows"][alias]["input"], alias + ":original_exact_source")
    check(sha(r["original"]) == old["rows"][alias]["input_sha256"], alias + ":original_exact_sha")
    expected = {k:v.copy() for k,v in base.items()}
    addition = 0
    if alias in overlays:
        z = arrays(overlays[alias]["output"])
        check(sha(overlays[alias]["output"]) == overlays[alias]["output_sha256"], alias+":overlay_source_sha")
        ids = z["source_index"]
        check(len(set(ids.tolist())) == len(ids), alias+":unique_overlay_rows")
        for k in ["frame_id","timestamp_us"]:
            check(eq(base[k][ids], z[k]), alias+":join_"+k)
        for k in ["x","mask"]:
            expected[k][ids] = z[k]
        addition = int((z["mask"] & ~z["baseline_mask"]).sum())
        totals["overlay_rows"] += len(ids)
        totals["overlay_clips"] += 1
    for k in keys:
        check(eq(expected[k], merged[k]), alias+":full_merged_exact_"+k)
    check(np.all(merged["mask"][base["mask"]]), alias+":all_old_support_preserved")
    check(eq(merged["x"][base["mask"]],base["x"][base["mask"]]), alias+":all_old_valid_values_preserved")
    check(d["overlays"][alias]["new_supported_cells"] == addition, alias+":new_support_count")
    check(int((merged["mask"] & ~base["mask"]).sum()) == addition, alias+":no_extra_support_outside_overlay")
    totals["new_cells"] += addition
check(dict(totals) == {"overlay_rows":303,"overlay_clips":8,"new_cells":604}, "fixed_overlay_totals")

expected_pairs = {(r["query"],r["reference"]) for r in old["natural"]}
check({(r["query"],r["reference"]) for r in d["natural"]} == expected_pairs, "exact12_pairs")
for r in d["natural"]:
    for key,prefix in [("global_path","B_same_path"),("same_cost_path","C_relational_path")]:
        expected = OPS/"word100_relational_sequence_compare_20261031/pairs"/f'{prefix}_{r["query"]}_{r["reference"]}.json'
        check(Path(r[key]).resolve() == expected.resolve(), f'{r["query"]}_{r["reference"]}:baseline_{key}')

expected_controls = {(c["alias"],c["control"]):c for c in ledger["saved_current_controls"]}
seen_old, seen_new = set(), set()
for c in d["controls"]:
    a = c["alias"]
    query = arrays(c["input"])
    reference = arrays(c["reference"])
    for label,z in [("q",query),("r",reference)]:
        check(keys <= set(z), c["id"]+":"+label+"_required_keys")
        if c["new"]:
            check(set(z) == keys, c["id"]+":"+label+"_no_truth_arrays")
    if not c["new"]:
        key = (a,c["kind"])
        source = expected_controls[key]
        seen_old.add(key)
        check(c["input"] == source["input_npz"]["path"], c["id"]+":old_control_input_exact")
        check(c["baseline"] == source["json"]["path"], c["id"]+":old_C_path_exact")
        check(c["reference"] == d["inputs"][a]["original"], c["id"]+":old_ref_original")
        check(c["metadata"] == source["metadata"], c["id"]+":metadata_unchanged")
        continue
    seen_new.add((a,c["kind"]))
    base = bases[a]
    n = len(base["x"])
    count = {"hold15":1, "repeat2_same":2, "repeat3_same":3}[c["kind"]]
    if count == 1:
        qi = list(range(n//2+1)) + [n//2]*15 + list(range(n//2+1,n))
        qo = [0]*len(qi)
    else:
        # Independently enumerate the specified linspace and ties-to-even round rule.
        percopy = round(n/.75)
        local = [int(round(i*(n-1)/(percopy-1))) for i in range(percopy)]
        qi = local*count
        qo = [cycle for cycle in range(count) for _ in local]
    ri = list(range(n))*count
    ro = [cycle for cycle in range(count) for _ in range(n)]
    m = c["metadata"]
    for field,value in [("query_source_frame",qi),("query_occurrence",qo),("reference_source_frame",ri),("reference_occurrence",ro),
                        ("query_unrolled_index",[f+o*n for f,o in zip(qi,qo)]),("reference_unrolled_index",list(range(n*count)))]:
        check(m[field] == value, c["id"]+":truth_"+field)
    check(m["equal_repeat_count"] == count, c["id"]+":repeatcount")
    check(c["baseline"] is None, c["id"]+":no_fabricated_saved_baseline")
    for label,z,ids in [("q",query,qi),("r",reference,ri)]:
        for k in ["x","mask"]:
            check(eq(z[k],base[k][ids]), c["id"]+":"+label+"_copied_"+k)
        check(eq(z["frame_id"],np.arange(len(ids))), c["id"]+":"+label+"_unrolled_ids")
        check(eq(z["timestamp_us"],np.arange(len(ids))*33333), c["id"]+":"+label+"_unrolled_time")
check(seen_old == set(expected_controls), "all126_saved_controls_once")
check(seen_new == {(a,k) for a in bases for k in ["hold15","repeat2_same","repeat3_same"]}, "all48_new_controls_once")
check(len(d["controls"]) == len(seen_old)+len(seen_new) == 174, "174_unique_controls")
result = {
    "schema":"phase_preparation_data_review.v1",
    "status":"PASS" if all(c["pass"] for c in checks) else "FAIL",
    "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checks":len(checks), "failures":[c for c in checks if not c["pass"]],
    "totals":dict(totals), "counts":{"inputs":16,"natural_pairs":12,"parents":39,"saved_controls":126,"new_controls":48,"all_controls":174},
    "source_pins":[{"path":str(p.resolve()),"sha256":sha(p)} for p in [U/"INPUTS.json",U/"prepare.py",U/"review/EVALUATION_LEDGER.json",Path(__file__)]],
    "code_review":{
        "prepare_runtime_npz_keys":["x","mask","frame_id","timestamp_us"],
        "truth_only_in_metadata":True,
        "frame_id_timestamp_are_renumbered_in_new_controls":True,
        "source_index_is_used_for_overlay_join_only":True,
        "new_condition_selection":"fixed all16, word-independent hold midpoint, same repeat counts and speed075",
        "natural_parent_descriptions_present_in_manifest":"Audit/reference context only. Runner must not pass query physical descriptions, parent query ranges, W06/W07 labels, or control metadata into matcher.",
        "runtime_execution_not_reviewed_yet":True
    },
    "limits":["Passing input preparation does not validate matching accuracy or educational correctness.",
              "Repeated seams are synthetic; report interior/seam and occurrence-aware source-time errors separately.",
              "Primary original69/C and recovery merged69 E0/E1 panels must remain distinct; no panel selection after scores."],
    "actions":{"new_alignment":0,"new_inference":0,"new_feature_extraction":0,"new_image_review":0,"producer_modifications":0},
    "all_checks":checks
}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:result[k] for k in ["status","checks","failures","totals","counts"]},ensure_ascii=False))
print(sha(OUT))
