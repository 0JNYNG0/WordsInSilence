"""Independently inspect all saved paths, axes, truth errors and summary.

No producer import, model, feature extraction, dynamic program or path search.
Run only after the root reports RESULT_MANIFEST.json and SUMMARY.json complete.
The first verification record is exclusively created and never overwritten.
"""
from pathlib import Path
from collections import Counter
import ast
import datetime
import hashlib
import json
import math
import statistics
import numpy as np

U = Path(__file__).resolve().parents[1]
OUT = U / "review/RESULT_VERIFICATION.json"
NORMAL = {"identity", "speed_075", "speed_125", "hold15", "repeat2_same", "repeat3_same"}
KEYS = ("x", "mask", "frame_id", "timestamp_us")
checks = 0
failures = []
pins_checked = {}
cache = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def check(condition, label):
    global checks
    checks += 1
    if not bool(condition):
        failures.append({"check": label})

def same(actual, expected, label):
    if isinstance(expected, dict):
        check(isinstance(actual, dict), label + ":dict")
        if not isinstance(actual, dict):
            return
        check(set(actual) == set(expected), label + ":keys")
        for k, v in expected.items():
            if k in actual:
                same(actual[k], v, label + "/" + str(k))
    elif isinstance(expected, (list, tuple)):
        check(isinstance(actual, (list, tuple)), label + ":list")
        if not isinstance(actual, (list, tuple)):
            return
        check(len(actual) == len(expected), label + ":length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, label + "/" + str(i))
    elif isinstance(expected, float):
        check(isinstance(actual, (int, float)) and math.isclose(float(actual), expected, abs_tol=1e-10, rel_tol=1e-10), label)
    else:
        check(actual == expected, label)

def pincheck(path, expected):
    path = str(Path(path).resolve())
    if path not in pins_checked:
        pins_checked[path] = sha(path)
    check(pins_checked[path] == expected, "pin:" + path)

def load(path):
    path = str(path)
    if path not in cache:
        with np.load(path, allow_pickle=False) as z:
            cache[path] = {k: z[k].copy() for k in KEYS}
    return cache[path]

def digest(q, r):
    return {
        side: {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
               for k, v in a.items()}
        for side, a in (("query", q), ("reference", r))
    }

def path_audit(path, n, m, label):
    p = np.asarray(path)
    check(p.ndim == 2 and p.shape[1:] == (2,) and len(p) > 0, label + ":path_shape")
    check(np.issubdtype(p.dtype, np.integer), label + ":path_integer")
    p = p.astype(np.int64)
    check(p[0].tolist() == [0, 0] and p[-1].tolist() == [n-1, m-1], label + ":full_endpoints")
    check(all(0 <= i < n and 0 <= j < m for i, j in p), label + ":index_bounds")
    check(set(map(tuple, np.diff(p, axis=0))).issubset({(1,0),(0,1),(1,1)}), label + ":strict_monotone_moves")
    check(np.array_equal(np.unique(p[:,0]), np.arange(n)), label + ":every_query_frame")
    check(np.array_equal(np.unique(p[:,1]), np.arange(m)), label + ":every_reference_frame")
    return p

def independent_evidence(q, r, p):
    """Accumulate all link evidence using indexed reductions, not producer loops."""
    qi, ri = p[:,0], p[:,1]
    qx, rx = q["x"], r["x"]
    qm, rm = q["mask"], r["mask"]
    edge_known = qm[qi] & rm[ri] & np.isfinite(qx[qi]) & np.isfinite(rx[ri])
    edge_difference = np.where(edge_known, np.minimum(abs(qx[qi]-rx[ri]),1),0)
    supported, maxima = [], []
    for ids, source in ((qi,qx),(ri,rx)):
        support = np.zeros(source.shape, dtype=bool)
        peak = np.zeros_like(source)
        np.logical_or.at(support, ids, edge_known)
        np.maximum.at(peak, ids, edge_difference)
        supported.append(support)
        maxima.append(peak)
    axes = {}
    for k, (name, mask) in enumerate((("query",qm),("reference",rm))):
        support, peak = supported[k], maxima[k]
        axes[name] = {
            "total_slots": int(mask.size), "observed_slots": int(np.count_nonzero(mask)),
            "unobserved_slots": int(mask.size-np.count_nonzero(mask)),
            "compared_slots": int(np.count_nonzero(support)),
            "observed_without_comparison": int(np.count_nonzero(mask & ~support)),
            "not_compared_slots": int(support.size-np.count_nonzero(support)),
            "shape": {
                h: {"observed":int(np.count_nonzero(mask[:,cols])),
                    "compared":int(np.count_nonzero(support[:,cols])),
                    "peak_difference":float(np.max(peak[:,cols],initial=0)) if support[:,cols].any() else None}
                for h,cols in (("L",range(11,25)),("R",range(39,53)))
            }
        }
    relation_known = qm[qi][:,[1,29]].all(1) & rm[ri][:,[1,29]].all(1)
    opposite = ((qx[qi,1]-qx[qi,29])*(rx[ri,1]-rx[ri,29]) < 0) & relation_known
    axes.update(height_conflicts=p[opposite].tolist(),
                height_supported_path_cells=int(np.count_nonzero(relation_known)),
                height_relation_is_not_linguistic_truth=True, no_correctness_verdict=True)
    return axes, supported, maxima

def independent_truth(p, meta, kind, n, m):
    if "query_unrolled_index" in meta:
        qloc, rloc = meta["query_unrolled_index"], meta["reference_unrolled_index"]
        qocc, rocc = meta["query_occurrence"], meta["reference_occurrence"]
    elif "original_indices" in meta:
        qloc, rloc = meta["original_indices"], range(m)
        qocc, rocc = [0]*n, [0]*m
    elif n == m:
        qloc, rloc, qocc, rocc = range(n), range(m), [0]*n, [0]*m
    else:
        return None
    qerrors = [[] for _ in range(n)]
    rerrors = [[] for _ in range(m)]
    qwrong, rwrong, nwrong = set(), set(), 0
    qseam = sorted({j for i in range(1,n) if qocc[i]!=qocc[i-1] for j in (i-1,i)})
    rseam = sorted({j for i in range(1,m) if rocc[i]!=rocc[i-1] for j in (i-1,i)})
    seam_mismatch = 0
    for qi, ri in p:
        e = abs(qloc[qi]-rloc[ri])
        qerrors[qi].append(e)
        rerrors[ri].append(e)
        if qocc[qi] != rocc[ri]:
            qwrong.add(int(qi)); rwrong.add(int(ri)); nwrong += 1
            seam_mismatch += int(qi in qseam or ri in rseam)
    # Every original axis index contributes once after worst-link aggregation.
    qmax = [max(x) for x in qerrors]
    rmax = [max(x) for x in rerrors]
    qm, rm = sum(qmax)/n, sum(rmax)/m
    qinterior = [v for i,v in enumerate(qmax) if i not in qseam]
    rinterior = [v for i,v in enumerate(rmax) if i not in rseam]
    return {"query_mean_max_error_frames":float(qm),
            "reference_mean_max_error_frames":float(rm),
            "mean_two_axes":float((qm+rm)/2),
            "peak_error_frames":float(max(max(qmax),max(rmax))),
            "occurrence_mismatch_cells":nwrong,
            "query_wrong_occurrence_frames":len(qwrong),
            "reference_wrong_occurrence_frames":len(rwrong),
            "interior_mean_two_axes":float((sum(qinterior)/len(qinterior)+sum(rinterior)/len(rinterior))/2),
            "synthetic_seam":{
                "query_frames":qseam,"reference_frames":rseam,
                "query_mean_max_error":float(sum(qmax[i] for i in qseam)/len(qseam)) if qseam else None,
                "reference_mean_max_error":float(sum(rmax[i] for i in rseam)/len(rseam)) if rseam else None,
                "mismatched_path_cells_touching_seam":seam_mismatch
            },
            "source_truth_not_passed_to_candidate":True,
            "counterfactual_feature_edit":kind not in NORMAL}

def independent_shape(c, q, bsupport, csupport, bpeak, cpeak):
    changed = c["metadata"].get("changed_frames", [])
    if not changed:
        return None
    cols = list(range(11,25) if c["kind"]=="left_short_shape_change_5f" else range(39,53))
    selector = np.ix_(changed,cols)
    known = q["mask"][selector]
    before = np.max(bpeak[selector],axis=1)
    after = np.max(cpeak[selector],axis=1)
    return {"frames":changed,"before_max":before.tolist(),"after_max":after.tolist(),
            "nonzero_before":int(sum(v>1e-6 for v in before)),
            "nonzero_after":int(sum(v>1e-6 for v in after)),
            "actual_observed_changed_cells":c["metadata"].get("actual_observed_changed_cells",0),
            "known_edited_cells":int(known.sum()),
            "baseline_compared_edited_cells":int((known&bsupport[selector]).sum()),
            "candidate_compared_edited_cells":int((known&csupport[selector]).sum()),
            "lost_compared_edited_cells":int((known&bsupport[selector]&~csupport[selector]).sum())}

def candidate_scope(d, q, r, bp, cp, label):
    c = d["candidate"]
    check(c["alignment_changed"] == (cp.tolist()!=bp.tolist()), label+":alignment_changed_truth")
    if c["fallback"] or c["status"] == "BASELINE_ALREADY_PASSES_OBSERVED_ANCHORS":
        check(np.array_equal(cp,bp), label+":fallback_or_already_passes_exact")
    check(c["baseline_exact_when_fallback"] is True, label+":fallback_claim")
    for axis, a in (("query",q),("reference",r)):
        same(c[axis+"_frame_ids"],a["frame_id"].tolist(),label+":"+axis+"_native_ids")
        same(c[axis+"_timestamp_us"],a["timestamp_us"].tolist(),label+":"+axis+"_native_pts")
    for field in ("physical_occurrence_verified","alignment_verified","semantic_order_verified","source_values_changed"):
        check(c[field] is False,label+":"+field)
    for field in ("whole_sign_verdict","correction_advice"):
        check(c[field] is None,label+":"+field)
    check(c["new_inference"]==c["new_training"]==0,label+":no_new_model")
    check(c["source_scope_preserved"] and c["all_feature_masks_preserved"],label+":preservation_claim")
    same(d["input_digests_checked_before_and_after"],digest(q,r),label+":saved_runtime_digest_vs_source")

def main():
    assert not OUT.exists(), "First verification output is immutable; use separately named supplemental verifier if needed."
    assert (U/"RESULT_MANIFEST.json").exists() and (U/"SUMMARY.json").exists(), "Wait for root execution-complete authorization."
    inp = read(U/"INPUTS.json")
    frozen = read(U/"FROZEN.json")
    manifest = read(U/"RESULT_MANIFEST.json")
    summary = read(U/"SUMMARY.json")
    for path, h in {**frozen["pins"],**inp["pins"]}.items():
        pincheck(path,h)
    check(manifest["frozen_sha256"]==sha(U/"FROZEN.json"),"manifest_frozen_pin")
    controls_by_id = {c["id"]:c for c in inp["controls"]}
    natural_by_id = {f'{panel}_{r["query"]}_{r["reference"]}':(r,panel)
                     for r in inp["natural"] for panel in ("original","merged")}
    check(len(manifest["controls"])==174,"174_manifest_control_records")
    check(len(manifest["natural"])==24,"24_manifest_natural_records")
    check({r["id"] for r in manifest["controls"]}==set(controls_by_id),"exact_control_result_inventory")
    check({r["id"] for r in manifest["natural"]}==set(natural_by_id),"exact_natural_result_inventory")
    check(len({r["file"] for r in manifest["controls"]+manifest["natural"]})==198,"unique198_result_files")
    recomputed_controls, recomputed_natural = [],[]
    scope_denominators = Counter()
    path_counts = Counter()
    shape_support = []
    provenance = []
    for group in ("controls","natural"):
        for entry in manifest[group]:
            label = entry["id"]
            pincheck(entry["file"],entry["sha256"])
            d = read(entry["file"])
            if group=="controls":
                spec = controls_by_id[label]
                qfile,rfile = spec["input"],spec["reference"]
                same(d["id"],label,label+":id")
                same(d["kind"],spec["kind"],label+":kind")
                same(d["input"],qfile,label+":input_path")
                same(d["reference"],rfile,label+":reference_path")
                same(d["baseline_source"],spec["baseline"],label+":baseline_source")
                baseline_source = spec["baseline"]
                baseline_kind = "archived_C" if baseline_source else "new_same_input_unconstrained"
            else:
                spec,panel = natural_by_id[label]
                qfile = inp["inputs"][spec["query"]][panel]
                rfile = inp["inputs"][spec["reference"]][panel]
                same(d["input"],qfile,label+":input_path")
                same(d["reference_input"],rfile,label+":reference_path")
                same(d["query"],spec["query"],label+":query_alias")
                same(d["reference"],spec["reference"],label+":reference_alias")
                same(d["panel"],panel,label+":panel")
                same(d["global_saved_source"],spec["global_path"],label+":global_context_source")
                same(d["parent_ids"],[r["id"] for r in inp["parents"] if r["query"]==spec["query"] and r["reference"]==spec["reference"]],label+":all_parent_ids")
                same(d["true_semantic_time_correspondence"],None,label+":no_unique_time_GT")
                baseline_source = spec["same_cost_path"] if panel=="original" else None
                same(d["baseline_source"],baseline_source or "new matched-input unconstrained control",label+":baseline_source")
                baseline_kind = "archived_C" if baseline_source else "new_same_input_unconstrained"
            q,r = load(qfile),load(rfile)
            n,m = len(q["x"]),len(r["x"])
            bp = path_audit(d["baseline_path"],n,m,label+":baseline")
            cp = path_audit(d["candidate"]["path"],n,m,label+":candidate")
            if baseline_source:
                same(d["baseline_path"],read(baseline_source)["path"],label+":archived_baseline_exact")
            candidate_scope(d,q,r,bp,cp,label)
            be,bs,bmax = independent_evidence(q,r,bp)
            ce,cs,cmax = independent_evidence(q,r,cp)
            same(d["baseline_evidence"],be,label+":baseline_evidence")
            same(d["candidate_evidence"],ce,label+":candidate_evidence")
            for ax in ("query","reference"):
                for key in ("total_slots","observed_slots","unobserved_slots"):
                    check(be[ax][key]==ce[ax][key],label+":"+ax+":unchanged_"+key)
                scope_denominators[group+"_"+ax+"_slots"] += be[ax]["total_slots"]
                scope_denominators[group+"_"+ax+"_observed"] += be[ax]["observed_slots"]
                check(be[ax]["observed_slots"]+be[ax]["unobserved_slots"]==be[ax]["total_slots"],label+":"+ax+":full_partition")
                check(ce[ax]["compared_slots"]+ce[ax]["not_compared_slots"]==ce[ax]["total_slots"],label+":"+ax+":comparison_partition")
            path_counts[group+"_"+baseline_kind] += 1
            path_counts["all_paths_checked"] += 2
            path_counts["fallback_exact"] += int(d["candidate"]["fallback"])
            provenance.append({"id":label,"baseline_kind":baseline_kind,"baseline_source":baseline_source,
                               "query_frames":n,"reference_frames":m,"baseline_links":len(bp),"candidate_links":len(cp)})
            if group=="controls":
                bt = independent_truth(bp,spec["metadata"],spec["kind"],n,m)
                ct = independent_truth(cp,spec["metadata"],spec["kind"],n,m)
                sv = independent_shape(spec,q,bs[0],cs[0],bmax[0],cmax[0])
                same(d["baseline_truth"],bt,label+":baseline_truth")
                same(d["candidate_truth"],ct,label+":candidate_truth")
                same(d["changed_shape"],sv,label+":changed_shape")
                if sv:
                    check(sv["actual_observed_changed_cells"]==sv["known_edited_cells"],label+":actual_known_edited_denominator")
                    source = load(inp["inputs"][spec["alias"]]["original"])
                    columns = list(range(11,25) if spec["kind"]=="left_short_shape_change_5f" else range(39,53))
                    idx = np.ix_(sv["frames"],columns)
                    actual = q["mask"][idx] & (q["x"][idx]!=source["x"][idx])
                    check(int(actual.sum())==sv["known_edited_cells"],label+":actual_nonzero_known_edits")
                    shape_support.append({"id":label,**sv})
                recomputed_controls.append({**d,"baseline_evidence":be,"candidate_evidence":ce,
                                            "baseline_truth":bt,"candidate_truth":ct,"changed_shape":sv})
            else:
                recomputed_natural.append({**d,"baseline_evidence":be,"candidate_evidence":ce})

    regress,improve,shape_regress,scope_fail = [],[],[],[]
    for c in recomputed_controls:
        b,a = c["baseline_truth"],c["candidate_truth"]
        if c["kind"] in NORMAL:
            bad = any(a[k]>b[k]+1e-9 for k in ("query_mean_max_error_frames","reference_mean_max_error_frames","peak_error_frames","occurrence_mismatch_cells","query_wrong_occurrence_frames","reference_wrong_occurrence_frames"))
            if bad: regress.append(c["id"])
            if not bad and a["mean_two_axes"]<b["mean_two_axes"]-1e-9 and a["interior_mean_two_axes"]<b["interior_mean_two_axes"]-1e-9:
                improve.append(c["id"])
        s = c["changed_shape"]
        if s and s["actual_observed_changed_cells"] and (s["lost_compared_edited_cells"]>0 or s["nonzero_after"]<s["nonzero_before"] or any(a+1e-6<b for a,b in zip(s["after_max"],s["before_max"]))):
            shape_regress.append(c["id"])
        for axis in ("query","reference"):
            if any(c["baseline_evidence"][axis][k]!=c["candidate_evidence"][axis][k] for k in ("total_slots","observed_slots","unobserved_slots")):
                scope_fail.append(c["id"])
    groups = []
    for kind in sorted({c["kind"] for c in recomputed_controls}):
        cs = [c for c in recomputed_controls if c["kind"]==kind]
        groups.append({"kind":kind,"cases":len(cs),
                       "changed_paths":sum(c["candidate"]["path"]!=c["baseline_path"] for c in cs),
                       "candidate_states":dict(Counter(c["candidate"].get("status","unspecified") for c in cs)),
                       "baseline_mean_two_axis_error":statistics.mean(c["baseline_truth"]["mean_two_axes"] for c in cs if c["baseline_truth"]),
                       "candidate_mean_two_axis_error":statistics.mean(c["candidate_truth"]["mean_two_axes"] for c in cs if c["candidate_truth"]),
                       "worse_normal_ids":[c["id"] for c in cs if c["id"] in regress],
                       "better_normal_ids":[c["id"] for c in cs if c["id"] in improve]})
    nsum = []
    residual = []
    for d in recomputed_natural:
        b,a = d["baseline_evidence"],d["candidate_evidence"]
        for axis in ("query","reference"):
            if any(b[axis][k]!=a[axis][k] for k in ("total_slots","observed_slots","unobserved_slots")):
                scope_fail.append(d["panel"]+"_"+d["query"]+"_"+d["reference"])
        nsum.append({"query":d["query"],"reference":d["reference"],"panel":d["panel"],
                     "changed_path":d["candidate"]["path"]!=d["baseline_path"],"status":d["candidate"].get("status"),
                     "height_conflicts_before":len(b["height_conflicts"]),"height_conflicts_after":len(a["height_conflicts"]),
                     "height_observed_path_cells_before":b["height_supported_path_cells"],
                     "height_observed_path_cells_after":a["height_supported_path_cells"],"no_semantic_accuracy_claim":True})
        if (d["query"],d["reference"])!=("P066","P063"): continue
        q,r=load(d["input"]),load(d["reference_input"])
        for condition,path in (("global_saved",read(d["global_saved_source"])["path"]),("same_input_unconstrained",d["baseline_path"]),("candidate",d["candidate"]["path"])):
            for frame in [68,71,72,73,74,75,76,77,78,79]:
                matches=[]
                for i,j in path:
                    if i!=frame: continue
                    valid = bool(q["mask"][i,[1,29]].all() and r["mask"][j,[1,29]].all())
                    sign = bool((q["x"][i,1]-q["x"][i,29])*(r["x"][j,1]-r["x"][j,29])<0) if valid else None
                    matches.append({"reference_frame":j,"both_wrist_heights_observed":valid,"opposite_height_order":sign})
                residual.append({"panel":d["panel"],"condition":condition,"query_frame":frame,"correspondences":matches,"unique_true_reference_frame":None})
    gates = {"whole_input_axes_preserved":not scope_fail,"known_normal_time_no_regression":not regress,
             "short_shape_peaks_preserved":not shape_regress,"known_time_useful_improvement":bool(improve),
             "requires_independent_natural_semantic_validation":True}
    useful = all(gates[k] for k in ("whole_input_axes_preserved","known_normal_time_no_regression","short_shape_peaks_preserved","known_time_useful_improvement"))
    expected = {
        "counts":{"controls":174,"normal_time_controls":sum(c["kind"] in NORMAL for c in recomputed_controls),
                  "natural_panels":24,"unique_natural_pairs":12,"parent_intervals_kept":39},
        "control_summary":groups,"gates":gates,"normal_regressions":regress,
        "known_time_improvements":improve,"short_shape_regressions":shape_regress,
        "scope_failures":scope_fail,"natural_summary":nsum,"source_review_followup":residual,
        "decision":"RESEARCH_PATH_CANDIDATE_ONLY_PENDING_NATURAL_TRUTH" if useful else "NOT_ADOPTED"
    }
    for k,v in expected.items(): same(summary[k],v,"summary/"+k)
    timing=[c["seconds"] for c in recomputed_controls+recomputed_natural]
    same(summary["timing"]["candidate_seconds_median"],statistics.median(timing),"summary/timing_median")
    same(summary["timing"]["candidate_seconds_max"],max(timing),"summary/timing_max")
    check(summary["new_training"]==summary["new_detector_inference"]==0,"summary_no_new_models")
    check(summary["C_T3F_operational_DTW_unchanged"] is True,"summary_protected_operational_claim")
    # Static runtime data route: align receives q/r measured arrays and baseline path,
    # not parents, control metadata, frame truth or stored W06/W07 review.
    runtree = ast.parse((U/"run.py").read_text(encoding="utf-8"))
    aligncalls = [n for n in ast.walk(runtree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="align"]
    check(len(aligncalls)==1 and [ast.unparse(a) for a in aligncalls[0].args]==["q","r","bp"] and not aligncalls[0].keywords,"static_align_only_measured_arrays_and_baseline")
    for p,h in {**frozen["pins"],**inp["pins"]}.items():
        check(sha(p)==h,"post_verification_unchanged:"+p)
    result={
        "schema":"phase_correspondence_independent_result_verification.v1",
        "status":"PASS" if not failures else "FAIL",
        "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checks":checks,"failures":failures,
        "result_count":198,"all_path_count":path_counts["all_paths_checked"],
        "path_counts":dict(path_counts),"full_axis_denominators":dict(scope_denominators),
        "recomputed_decision":expected["decision"],"recomputed_gates":gates,
        "normal_regressions":regress,"known_time_improvements":improve,"short_shape_regressions":shape_regress,
        "short_shape_support":shape_support,"path_provenance":provenance,
        "source_pins":[{"path":str(p.resolve()),"sha256":sha(p)} for p in (U/"INPUTS.json",U/"FROZEN.json",U/"RESULT_MANIFEST.json",U/"SUMMARY.json",U/"run.py",U/"summarize.py",Path(__file__))],
        "limitations":[
            "Runtime before/after mutation guard is producer-code evidence; independent saved digest verification confirms equality to pinned source arrays.",
            "All198 baseline/candidate saved paths were checked; no path or baseline DP was rerun and numeric optimality is outside this source audit.",
            "Known source-time and occurrence labels are construction truth only, never educational correctness.",
            "All198 full axes contribute to scope checks, including natural24; no denominator dropped.",
            "W06/W07 followup is saved correspondence arithmetic with exact-time truth left null; no new RGB interpretation.",
            "Original and merged natural panels are reported separately; no result-driven panel choice."
        ],
        "actions":{"producer_imports":0,"model_inference":0,"new_alignment_or_DP":0,"new_feature_extraction":0,"new_RGB_review":0,"original_files_modified":0}
    }
    with OUT.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","checks","failures","result_count","recomputed_decision","recomputed_gates")},ensure_ascii=False))
    print(sha(OUT))

if __name__ == "__main__":
    main()
