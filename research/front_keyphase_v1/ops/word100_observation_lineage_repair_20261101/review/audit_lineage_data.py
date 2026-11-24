"""Read-only mask lineage over exactly the prior fixed 303 public frames.

No producer imports, new detection, feature-value extraction, RGB decoding,
alignment, score, source mutation, or label modification.
"""
from pathlib import Path
import json, hashlib, datetime as dt
from collections import Counter, defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
OPS=HERE.parent.parent
CASE=OPS/"word100_action_feedback_casebook_20261101"
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
pins={};checks=Counter();failures=[]
def pin(p,expected=None):
    p=str(Path(p).resolve());actual=sha(p)
    if expected is not None: check(actual==expected,"source_hash",p)
    pins[p]=actual
    return p
def check(ok,name,detail=None):
    checks[name]+=1
    if not bool(ok): failures.append({"check":name,"detail":detail})

for name in ["INPUTS.json","review/WINDOW_SELECTION.json","review/WINDOW_ACK.json","review/RGB_MANIFEST.json","review/VISUAL_REVIEW.json"]:
    pin(CASE/name)
inputs=read(CASE/"INPUTS.json")
manifest=read(CASE/"review/RGB_MANIFEST.json")
selection=read(CASE/"review/WINDOW_SELECTION.json")
scope=defaultdict(set)
for f in manifest["full_frames"]: scope[f["alias"]].add(f["frame_id"])
check(sum(map(len,scope.values()))==303,"fixed_unique_frames")
check(len(scope)==8,"fixed_public_clips")
check(sha(CASE/"review/WINDOW_SELECTION.json")=="e4a406079828ba530808ae19b00634e2b5d495eb1c6cf5596a47c4e8754e7d67","original_selection")
for p in [OPS/"word100_stage_feedback_cycle_20261031/prepare_bundle.py",
          OPS/"word100_signrep_temporal_head_20261029/prepare_inputs.py",
          OPS/"word100_guide_video_feedback_20261028/motion_observer.py"]:
    pin(p)

triplets=[(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),
          (0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]
counts=Counter(); by_alias={};frame_rows=[];clip_sources=[]
for alias,frames in sorted(scope.items()):
    row=inputs["rows"][alias]
    rec=read(pin(row["record"],row["record_sha256"]))
    cap=rec["native_capture"]
    capdir=Path(cap["candidate"]).parent
    rawpath=pin(cap["raw"],cap["pins"][cap["raw"]])
    candidate=pin(cap["candidate"],cap["pins"][cap["candidate"]])
    obs_path=pin(row["observation"],rec["outputs"][row["observation"]])
    input_path=pin(row["input"],row["input_sha256"])
    assign_path=pin(capdir/"assignment.npz",cap["pins"].get(str(capdir/"assignment.npz")))
    trace_path=pin(capdir/"trace.json",cap["pins"].get(str(capdir/"trace.json")))
    sidecar_path=pin(Path(candidate).with_suffix(".sidecar.json"),cap["pins"].get(str(Path(candidate).with_suffix(".sidecar.json"))))
    raw=load(rawpath);obs=load(obs_path);cand=load(candidate);final=load(input_path);ass=load(assign_path);trace=read(trace_path)
    check(rec["rest_wrist_filter"]=="OFF","rest_filter_off",alias)
    check(row["box"]==rec["calibration"]["box_source_pixels"],"source_roi_equal",alias)
    check(cap.get("box") is None,"full_native_not_sensor_capture",alias)
    x0,y0,x1,y1=row["box"]
    local=Counter()
    for fid in sorted(frames):
        indices={}
        for name,data in [("raw",raw),("native",obs),("candidate",cand),("final",final)]:
            ids=np.flatnonzero(data["frame_id"]==fid)
            check(len(ids)==1,"unique_native_id",[alias,fid,name])
            indices[name]=int(ids[0])
        i=indices["raw"];j=indices["native"];k=indices["final"];c=indices["candidate"]
        pts=int(raw["timestamp_us"][i])
        for name,data in [("native",obs),("candidate",cand),("final",final)]:
            check(int(data["timestamp_us"][indices[name]])==pts,"same_native_pts",[alias,fid,name])
        for key in ["xy_px","observed","usable","conf","hand_score"]:
            check(np.array_equal(obs[key][j],cand[key][c],equal_nan=True),"candidate_native_exact",[alias,fid,key])
        t=next(t for t in trace if t["frame"]==fid)
        nraw=int(raw["raw_count"][i]); sides=ass["sides"][i]
        check(t["assigned_sides"]==sides[:nraw].tolist(),"trace_assignment_equal",[alias,fid])
        check(int(t["timestamp_us"])==pts,"trace_pts",[alias,fid])
        check(not t.get("coordinates_generated",False),"no_generated_coordinates",[alias,fid])
        fr={"alias":alias,"frame_id":fid,"timestamp_us":pts,"raw_count":nraw,
            "raw_candidates":[{"index":u,"raw_label_name":str(raw["raw_label_name"][i,u]),"raw_label":int(raw["raw_label"][i,u]),
                               "raw_handedness_score":float(raw["raw_score"][i,u]),"assigned_side":int(sides[u]),
                               "trace_reason":t["reasons"][u],"finite_points":int(raw["hand_finite"][i,u].sum()),
                               "raw_high_score":bool(raw["raw_high_score"][i,u])} for u in range(nraw)],
            "roi_box":row["box"],"hands":{}}
        for si,(side,off,coloff) in enumerate([("L",33,0),("R",54,28)]):
            matched=np.flatnonzero(sides[:nraw]==si)
            check(len(matched)<=1,"one_assignment_per_side",[alias,fid,side])
            xy=obs["xy_px"][j,off:off+21].astype(float)
            observed=obs["observed"][j,off:off+21]
            usable=obs["usable"][j,off:off+21]
            finite=np.isfinite(xy).all(1)
            inside=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
            native_valid=observed&usable&finite
            roi_valid=native_valid&inside
            if len(matched):
                u=int(matched[0]);rxy=raw["hand_xy_px"][i,u].astype(np.float32)
                check(np.array_equal(xy.astype(np.float32),rxy,equal_nan=True),"assigned_raw_xy_float32_exact",[alias,fid,side])
                check(observed.all(),"assigned_observed21",[alias,fid,side])
                width=float(raw["width"][i]);height=float(raw["height"][i])
                quality=np.isfinite(rxy).all(1)&(rxy[:,0]>=-.05*width)&(rxy[:,0]<=1.05*width)&(rxy[:,1]>=-.05*height)&(rxy[:,1]<=1.05*height)&(raw["raw_score"][i,u]>=.5)
                check(np.array_equal(usable,quality),"native_quality_formula",[alias,fid,side])
            else:
                check(not observed.any() and not usable.any(),"unassigned_no_observed_points",[alias,fid,side])
            reasons=[];expected=[]
            for ids in triplets+[(0,9),(0,9)]:
                ids=list(ids)
                if len(ids)==3:
                    a,b,d=xy[ids]
                    product=float(np.linalg.norm(a-b)*np.linalg.norm(d-b))
                    geometry=np.isfinite(product) and product>1e-12
                else:
                    length=float(np.linalg.norm(xy[9]-xy[0]))
                    geometry=np.isfinite(length) and length>1e-8
                before=bool(native_valid[ids].all() and geometry)
                after=bool(roi_valid[ids].all() and geometry)
                expected.append(after)
                if after: reason="SUPPORTED"
                elif not observed[ids].all():
                    reason="NO_ASSIGNED_NATIVE_OBSERVATION"
                elif not (usable[ids]&finite[ids]).all(): reason="NATIVE_QUALITY_OR_FINITE_EXCLUSION"
                elif not geometry: reason="DEGENERATE_GEOMETRY"
                elif before and not inside[ids].all(): reason="EXPLICIT_FIXED_ROI_POINT_EXCLUSION"
                else: reason="UNRESOLVED"
                reasons.append(reason);local[reason]+=1;counts[reason]+=1
            cols=list(range(coloff+11,coloff+25))+[coloff+25,coloff+26]
            saved=final["mask"][k,cols]
            check(np.array_equal(saved,expected),"independent_finger_axis_mask_equal",[alias,fid,side])
            check((final["x"][k,cols][~saved]==0).all(),"unknown_zero_mask_retained",[alias,fid,side])
            mode="full" if saved.all() else "none" if not saved.any() else "partial"
            local["handframes_"+mode]+=1;counts["handframes_"+mode]+=1
            if len(matched)==0:
                absent="raw_zero" if nraw==0 else "raw_other_side_only" if all(int(s)>=0 and int(s)!=si for s in sides[:nraw]) else "assignment_withheld_candidate_present"
                local[absent]+=1;counts[absent]+=1
            else: absent=None
            fr["hands"][side]={"assigned_candidate_index":int(matched[0]) if len(matched) else None,
                "absence_lineage":absent,"native_observed_points":int(observed.sum()),"native_usable_points":int(usable.sum()),
                "native_valid_points":int(native_valid.sum()),"inside_roi_valid_points":int(roi_valid.sum()),
                "roi_removed_native_point_ids":np.flatnonzero(native_valid&~inside).tolist(),
                "native_y_range": [float(xy[finite,1].min()),float(xy[finite,1].max())] if finite.any() else None,
                "feature_columns":cols,"saved_mask":saved.astype(int).tolist(),"mask_reason_by_column":reasons,
                "quality_not_anatomical_truth":True}
        frame_rows.append(fr)
    by_alias[alias]=dict(local)
    clip_sources.append({"alias":alias,"frame_ids":sorted(frames),"row_source":str(CASE/"INPUTS.json"),
       "record":row["record"],"raw":rawpath,"assignment":assign_path,"trace":trace_path,"candidate":candidate,
       "native_observation":obs_path,"final_69_input":input_path,"sidecar":sidecar_path,"native_capture_reused":cap["reused"],
       "box":row["box"],"sensor_capture_not_used":True,"rest_wrist_filter":"OFF"})

check(len(frame_rows)==303,"all_fixed_frames")
check(sum(counts[k] for k in ["handframes_full","handframes_none","handframes_partial"])==606,"all_handframes")
check(sum(counts[k] for k in ["SUPPORTED","NO_ASSIGNED_NATIVE_OBSERVATION","NATIVE_QUALITY_OR_FINITE_EXCLUSION","DEGENERATE_GEOMETRY","EXPLICIT_FIXED_ROI_POINT_EXCLUSION","UNRESOLVED"])==9696,"all_32_feature_cells")
result={"schema":"read_only_fixed303_observation_lineage_v1","status":"PASS" if not failures else "FAIL",
 "at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"author":"Codex semantic_data_audit",
 "scope":{"public_clips":8,"unique_fixed_frames":303,"hand_frame_opportunities":606,"finger_axis_cells":9696,
          "cell_definition":"14 projected internal finger angles + 2 projected palm-axis coordinates per hand; no shoulder normalizer dependency.",
          "not_recomputed":"No 69 feature values, learned posterior, previous comparison, model, alignment, RGB, or score was computed."},
 "counts":dict(counts),"by_alias":by_alias,"clip_sources":clip_sources,"frames":frame_rows,
 "checks":dict(checks),"check_total":sum(checks.values()),"failures":failures,"source_pins":pins,"script_sha256":sha(__file__),
 "findings":[
  {"case":"P048 f72-74","finding":"Stored full-native raw capture contains one candidate, assigned Right. No Left observation exists before ROI or feature-mask processing.",
   "scope":"No claim about internal detector failure mechanism or which unreturned physical hand candidate the model rejected; no rerun was made.",
   "repair_eligibility":"No lost Left coordinates available in this actual native lineage to restore by downstream mask repair."},
  {"case":"P036 fixed f64-132 Left","finding":"Raw assigned Left observations exist, but exact fixed bottom884 ROI excludes dependent hand points; final mask follows the explicit pointwise ROI rule.",
   "scope":"This is scope exclusion, not evidence that source RGB hand is invisible. Linguistic importance is not inferred from the crop rule.",
   "repair_eligibility":"Do not automatically undo an authorized ROI or call it a bug; preserve separate outside-ROI observation lineage."}
 ],
 "interpretation":["Raw handedness category score is not an anatomical/landmark-quality truth score.",
  "Observed/usable and ROI inclusion are different stages. rest_wrist_filter OFF does not mean ROI OFF.",
  "A returned/assigned candidate is not human-verified anatomical identity.",
  "No evidence here authorizes filling an absent hand or selecting a different saved detector run by favorable outcome.",
  "P066 saved temporal correspondence mismatch remains a separate problem, not a handshape classifier error label."],
 "actions":{"new_detection":0,"inference":0,"rgb_decode_or_open":0,"new_alignment":0,"old_score_recalculation":0,"source_mutation":0,"label_changes":0}}
out=HERE/"LINEAGE_DATA_AUDIT.json"
assert not out.exists(),"Preserve first audit result"
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":result["status"],"checks":result["check_total"],"failures":len(failures),"counts":dict(counts),"sha256":sha(out)},ensure_ascii=False))
if failures: raise SystemExit(1)
