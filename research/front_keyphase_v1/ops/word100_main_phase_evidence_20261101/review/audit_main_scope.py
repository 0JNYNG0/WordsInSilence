"""Join frozen39 parent scopes, saved paths and masks. No matching or scoring."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import numpy as np

U=Path(__file__).resolve().parents[1]
OPS=U.parent
PHASE=OPS/"word100_phase_correspondence_20261101"
CASE=OPS/"word100_action_feedback_casebook_20261101"
OUT=U/"review/MAIN_SCOPE_LEDGER.json"
assert not OUT.exists(),"Preserve the first saved audit."
pins={}
checks=0
failures=[]
cache={}
def pin(p):
    p=Path(p).resolve()
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    pins[str(p)]=h
    return {"path":str(p),"sha256":h}
def read(p):
    pin(p)
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def check(ok,label):
    global checks
    checks+=1
    if not bool(ok):failures.append(label)
def load(p):
    p=str(p)
    if p not in cache:
        pin(p)
        with np.load(p,allow_pickle=False) as z:cache[p]={k:z[k].copy() for k in ("x","mask","frame_id","timestamp_us")}
    return cache[p]
def runs(ids):
    ids=sorted(set(map(int,ids)))
    out=[]
    for x in ids:
        if out and x==out[-1][1]+1:out[-1][1]=x
        else:out.append([x,x])
    return out
def span_info(a,ids):
    ids=sorted(set(map(int,ids)))
    return {"sequence_indices":ids,"index_runs_inclusive":runs(ids),
            "frame_ids":a["frame_id"][ids].tolist(),"timestamp_us":a["timestamp_us"][ids].tolist(),
            "count":len(ids),"index_range_inclusive":[ids[0],ids[-1]] if ids else None}

case=read(CASE/"INPUTS.json")
inp=read(PHASE/"INPUTS.json")
manifest=read(PHASE/"RESULT_MANIFEST.json")
broad_path=OPS/"word100_core_stage_evidence_20261031/PHASE_DATA.json"
physical_path=OPS/"word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json"
broad=read(broad_path)
physical=read(physical_path)
feature_path=OPS/"word100_independent_hand_evidence_20261101/review/INPUT_AUDIT.json"
feature=read(feature_path)
visual_path=CASE/"review/VISUAL_REVIEW.json"
visual=read(visual_path)
read(PHASE/"review/POSTEXECUTION_REVIEW.json")
pin(PHASE/"NEXT_PLAN_KO.txt")
check(case["natural"]==inp["parents"],"fixed39_parent_rows_unchanged")
check(len(case["natural"])==39 and len(inp["natural"])==12 and len(inp["inputs"])==16,"fixed39_12_16")
phase_clips={c["alias"]:c for c in broad["clips"]}
physical_clips={c["alias"]:c for c in physical["clips"]}
kind_defs=physical["kinds"]
natural_result={}
for rec in manifest["natural"]:
    d=read(rec["file"])
    check(pins[str(Path(rec["file"]).resolve())]==rec["sha256"],"result_pin:"+rec["id"])
    natural_result[(d["query"],d["reference"],d["panel"])]=(d,pin(rec["file"]))
pair_sources={(p["query"],p["reference"]):p for p in inp["natural"]}
column_names={c["index"]:c["name"] for c in feature["feature_columns"]}
groups={}
for side,off in (("L",0),("R",28)):
    for name,cols in [
        ("hand_wrist_body_xy",[0,1]),("pose_forearm_body_xy",[2,3,4,5]),
        ("index_tip_body_xy",[6,7]),("palm_center_body_xy",[8,9]),
        ("elbow_angle",[10]),("finger_angles14",list(range(11,25))),
        ("hand_axis2d",[25,26]),("palm_width",[27])]:
        groups[side+"_"+name]={"columns":[off+i for i in cols],"kind":"stored69_columns"}
    groups[side+"_hand_forearm_relative_dependencies"]={"columns":[off+i for i in [2,3,4,5,25,26]],"kind":"dependency_support_only_no_new_angle","nondegenerate_forearm":True}
    other=28-off
    groups[side+"_opposite_forearm_relation_dependencies"]={"columns":[off,off+1,other+2,other+3,other+4,other+5],"kind":"dependency_support_only_no_new_relation","nondegenerate_opposite_forearm":True}
groups.update({
    "interhand_wrist_relation":{"columns":[56,57,58],"kind":"stored69_columns"},
    "L_hand_pose_estimator_distance":{"columns":[59],"kind":"stored69_columns"},
    "R_hand_pose_estimator_distance":{"columns":[60],"kind":"stored69_columns"},
    "L_velocity_xy":{"columns":[61,62],"kind":"stored69_columns"},
    "L_motion_direction_xy":{"columns":[63,64],"kind":"stored69_columns"},
    "R_velocity_xy":{"columns":[65,66],"kind":"stored69_columns"},
    "R_motion_direction_xy":{"columns":[67,68],"kind":"stored69_columns"}})
for g in groups.values():g["column_names"]=[column_names[i] for i in g["columns"]]

def valid(a):
    return a["mask"] & np.isfinite(a["x"])
def family_mask(a,name):
    g=groups[name]
    m=valid(a)[:,g["columns"]].copy()
    if g.get("nondegenerate_forearm") or g.get("nondegenerate_opposite_forearm"):
        side=0 if name.startswith("L") else 28
        if g.get("nondegenerate_opposite_forearm"):side=28-side
        delta=a["x"][:,side+2:side+4]-a["x"][:,side+4:side+6]
        good=np.isfinite(delta).all(1)&(np.linalg.norm(delta,axis=1)>1e-8)
        m &= good[:,None]
    return m
def support(a,ids):
    ids=np.asarray(sorted(set(map(int,ids))),dtype=int)
    base=valid(a)[ids]
    out={"frames":len(ids),"all69":{"possible_cells":int(base.size),"supported_cells":int(base.sum()),
                                  "unknown_cells":int(base.size-base.sum()),"support_fraction":float(base.mean()) if base.size else None},"families":{}}
    for name in groups:
        m=family_mask(a,name)[ids]
        full=m.all(1);any_known=m.any(1)
        out["families"][name]={"possible_cells":int(m.size),"supported_cells":int(m.sum()),"unknown_cells":int(m.size-m.sum()),
            "support_fraction":float(m.mean()) if m.size else None,"fully_supported_frames":int(full.sum()),
            "partially_supported_frames":int((any_known&~full).sum()),"no_support_frames":int((~any_known).sum()),
            "fully_supported_index_runs":runs(ids[full])}
    return out

def broad_members(alias,frame):
    out=[]
    for s in phase_clips[alias]["segments"]:
        if s["core"][0]<=frame<=s["core"][1]:
            out.append({"segment_id":s["segment_id"],"role":s["role"]})
    return out
def physical_member(alias,frame):
    found=[]
    for s in physical_clips[alias]["regions"]:
        lo,hi=s["outer"]
        if lo<=frame<=hi:found.append(s)
    check(len(found)<=1,f"{alias}:{frame}:physical_region_nonoverlap")
    return found[0] if found else None
def core_kind(s):
    if not s or not s.get("kind"):return None
    return kind_defs.get(s["kind"],{}).get("physical_kind",s["kind"])
def phase_coverage(alias,ids):
    ids=sorted(set(map(int,ids)))
    memberships={}
    for i in ids:
        matches=broad_members(alias,i)
        key="|".join(x["segment_id"] for x in matches) if matches else "UNKNOWN_UNANNOTATED_GAP"
        memberships.setdefault(key,{"roles":[x["role"] for x in matches],"indices":[]})["indices"].append(i)
    return [{"phase_ids":key,"roles":v["roles"],"frame_count":len(v["indices"]),"index_runs":runs(v["indices"])} for key,v in memberships.items()]
def physical_coverage(alias,ids):
    out={}
    for i in sorted(set(map(int,ids))):
        s=physical_member(alias,i)
        key=s["id"] if s else "OUTSIDE_SAVED_REVIEWED_REGION"
        item=out.setdefault(key,{"region_id":key,"region_type":s["region_type"] if s else "unreviewed",
                "kind":s.get("kind") if s else None,"physical_kind":core_kind(s),
                "occurrence":s.get("occurrence") if s else None,"motion_subphase":s.get("motion_subphase") if s else None,
                "description_ko":s["description_ko"] if s else None,"indices":[]})
        item["indices"].append(i)
    for x in out.values():
        x["frame_count"]=len(x["indices"]);x["index_runs"]=runs(x.pop("indices"))
    return list(out.values())

def paired_support(q,r,p):
    qids=np.unique(p[:,0]);rids=np.unique(p[:,1])
    out={}
    for name in groups:
        qm,rm=family_mask(q,name),family_mask(r,name)
        links=qm[p[:,0]] & rm[p[:,1]]
        qu=np.zeros_like(qm);ru=np.zeros_like(rm)
        np.logical_or.at(qu,p[:,0],links);np.logical_or.at(ru,p[:,1],links)
        # Query ranges can change with paths: record counts and do not compare
        # conditional fractions as an accuracy metric.
        out[name]={"path_link_possible_cells":int(links.size),"path_link_shared_cells":int(links.sum()),
                   "query_unique_possible_cells":int(qu[qids].size),"query_unique_compared_cells":int(qu[qids].sum()),
                   "reference_fixed_possible_cells":int(ru[rids].size),"reference_fixed_compared_cells":int(ru[rids].sum()),
                   "reference_fixed_uncompared_cells":int(ru[rids].size-ru[rids].sum())}
    return out

def physical_link_table(qalias,ralias,p):
    c=Counter();examples={}
    for qi,ri in p:
        q=physical_member(qalias,int(qi));r=physical_member(ralias,int(ri))
        if not q or not r or not q.get("kind") or not r.get("kind"):
            state="UNKNOWN_OR_BOUNDARY"
        elif q["kind"]==r["kind"]:
            state="SAME_EXACT_DRAFT_KIND"
        elif core_kind(q)==core_kind(r):
            state="SAME_PHYSICAL_FAMILY_DIFFERENT_SUBPHASE"
        else:state="DIFFERENT_PHYSICAL_DRAFT_KIND_NOT_EDUCATIONAL_ERROR"
        occ="unknown" if not q or not r or q.get("occurrence") is None or r.get("occurrence") is None else "same" if q["occurrence"]==r["occurrence"] else "different"
        key=(state,occ,q["id"] if q else None,r["id"] if r else None)
        c[key]+=1
        examples.setdefault(key,[]).append([int(qi),int(ri)])
    return [{"state":k[0],"occurrence_relation":k[1],"query_region":k[2],"reference_region":k[3],
             "path_cells":n,"all_source_pairs":examples[k]} for k,n in c.items()]

def candidates(parent,qalias,ralias,qoriginal,qmerged):
    result=[]
    rr=parent["reference_range"]
    refparts=[s for s in physical_clips[ralias]["regions"] if s.get("kind") and s["outer"][0]<=rr[1] and s["outer"][1]>=rr[0]]
    for s in refparts:
        exact=[];family=[]
        for q in physical_clips[qalias]["regions"]:
            if not q.get("kind"):continue
            if core_kind(q)!=core_kind(s):continue
            item={"region_id":q["id"],"core":q["core"],"kind":q["kind"],"physical_kind":core_kind(q),
                  "occurrence":q.get("occurrence"),"same_recorded_occurrence":None if q.get("occurrence") is None or s.get("occurrence") is None else q["occurrence"]==s["occurrence"],
                  "description_ko":q["description_ko"],"motion_subphase":q.get("motion_subphase"),
                  "original_support":support(qoriginal,range(q["core"][0],q["core"][1]+1)),
                  "merged_support":support(qmerged,range(q["core"][0],q["core"][1]+1))}
            (exact if q["kind"]==s["kind"] else family).append(item)
        result.append({"reference_region_id":s["id"],"reference_core":s["core"],"parent_intersection":[max(rr[0],s["core"][0]),min(rr[1],s["core"][1])],
                       "reference_kind":s["kind"],"reference_occurrence":s.get("occurrence"),"exact_kind_query_candidates":exact,
                       "same_physical_family_other_subphase_candidates":family,"unique_correct_time":None,
                       "acceptable_correspondence_ground_truth":None,"status":"EVALUATOR_DRAFT_CANDIDATES_NOT_SELECTED_BY_DISTANCE"})
    return result

records=[];method_aggregate={};path_records={}
for query,reference in sorted(pair_sources):
    spec=pair_sources[(query,reference)]
    original,op=natural_result[(query,reference,"original")]
    merged,mp=natural_result[(query,reference,"merged")]
    bg=read(spec["global_path"]);c=read(spec["same_cost_path"])
    methods=[
        ("B_global_original","original",bg["path"],pin(spec["global_path"]),"/path"),
        ("C_relational_original","original",c["path"],pin(spec["same_cost_path"]),"/path"),
        ("phase_candidate_original","original",original["candidate"]["path"],op,"/candidate/path"),
        ("recovery_E0","merged",merged["baseline_path"],mp,"/baseline_path"),
        ("phase_candidate_recovery_E1","merged",merged["candidate"]["path"],mp,"/candidate/path")]
    for name,panel,path,source,pointer in methods:
        path_records[(query,reference,name)]=(panel,np.asarray(path,dtype=int),source,pointer)
    check(original["baseline_path"]==c["path"],f"{query}_{reference}:C_baseline_exact")

for ni,parent in enumerate(case["natural"]):
    qa,ra=parent["query"],parent["reference"]
    qold=load(inp["inputs"][qa]["original"]);qnew=load(inp["inputs"][qa]["merged"])
    rold=load(inp["inputs"][ra]["original"]);rnew=load(inp["inputs"][ra]["merged"])
    lo,hi=parent["reference_range"];refids=list(range(lo,hi+1))
    check(parent["id"] not in {r["id"] for r in records},"unique_parent:"+parent["id"])
    r={
        "id":parent["id"],"query":qa,"reference":ra,"parent_id":parent["parent_id"],"label":parent["label"],
        "source_pointer":f"/natural/{ni}","reference_core_range_inclusive":parent["reference_range"],
        "legacy_B_query_range_inclusive":parent["query_range"],"reference_description_ko":parent["parent_physical_description"],
        "reference_scope":span_info(rold,refids),"reference_broad_phase_coverage":phase_coverage(ra,refids),
        "reference_physical_regions":physical_coverage(ra,refids),
        "reference_support_original":support(rold,refids),"reference_support_merged":support(rnew,refids),
        "prior_hand_evidence":parent["hand_record"],"prior_source_description_link":parent["legacy"],
        "source_card_count":parent["source_card_count"],
        "query_physical_candidate_intervals":candidates(parent,qa,ra,qold,qnew),
        "methods":[],"method_differences":[],"unique_correct_query_interval":None,
        "linguistic_required_action_or_learner_error":None}
    subset_lookup={}; allq=set()
    for name in ["B_global_original","C_relational_original","phase_candidate_original","recovery_E0","phase_candidate_recovery_E1"]:
        panel,full,source,pointer=path_records[(qa,ra,name)]
        p=full[(full[:,1]>=lo)&(full[:,1]<=hi)]
        check(len(p)>0,f'{parent["id"]}:{name}:nonempty')
        check(np.array_equal(np.unique(p[:,1]),np.arange(lo,hi+1)),f'{parent["id"]}:{name}:all_fixed_reference_frames')
        qids=np.unique(p[:,0]);rids=np.unique(p[:,1]);allq.update(map(int,qids))
        q,rarr=(qold,rold) if panel=="original" else (qnew,rnew)
        qscope=span_info(q,qids)
        if name=="B_global_original":check(qscope["index_range_inclusive"]==parent["query_range"],f'{parent["id"]}:legacy_query_range')
        mapping=[]
        for ri in refids:
            qs=p[p[:,1]==ri,0]
            mapping.append({"reference_index":ri,"reference_frame_id":int(rarr["frame_id"][ri]),
                            "query_indices":qs.tolist(),"query_index_runs":runs(qs),
                            "query_physical_regions":physical_coverage(qa,qs)})
        table=physical_link_table(qa,ra,p)
        mi={"method":name,"panel":panel,"path_source":source,"path_json_pointer":pointer,
            "reference_core_is_fixed":True,"full_path_links":len(full),"parent_path_links":len(p),
            "query_scope":qscope,"query_broad_phase_coverage":phase_coverage(qa,qids),
            "query_physical_region_coverage":physical_coverage(qa,qids),
            "query_support":support(q,qids),"shared_support_by_family":paired_support(q,rarr,p),
            "physical_draft_link_cross_tab":table,"all_reference_frame_correspondences":mapping}
        r["methods"].append(mi);subset_lookup[name]=p
        ag=method_aggregate.setdefault(name,Counter())
        ag["parents"]+=1;ag["fixed_reference_frames"]+=len(refids);ag["linked_unique_query_frames"]+=len(qids);ag["path_cells"]+=len(p)
        for t in table:ag[t["state"]]+=t["path_cells"]
    r["common_query_envelope_across_saved_methods"]=span_info(qold,allq)
    r["common_query_envelope_support_original"]=support(qold,allq)
    r["common_query_envelope_support_merged"]=support(qnew,allq)
    for left,right in [("B_global_original","C_relational_original"),("C_relational_original","phase_candidate_original"),
                       ("recovery_E0","phase_candidate_recovery_E1"),("C_relational_original","recovery_E0")]:
        lp,rp=subset_lookup[left],subset_lookup[right]
        ls,rs=set(map(tuple,lp)),set(map(tuple,rp))
        r["method_differences"].append({"before":left,"after":right,"same_parent_correspondences":np.array_equal(lp,rp),
                                       "removed_source_pairs":[list(x) for x in sorted(ls-rs)],
                                       "added_source_pairs":[list(x) for x in sorted(rs-ls)],
                                       "query_frames_removed":sorted(set(lp[:,0].tolist())-set(rp[:,0].tolist())),
                                       "query_frames_added":sorted(set(rp[:,0].tolist())-set(lp[:,0].tolist())),
                                       "accuracy_improvement":None})
    r["previous_visual_review"]=[{"window_id":w["window_id"],"source":str(visual_path),"pointer":f"/windows/{wi}",
                                  "query_reviewed_frames":w["query_axis"]["reviewed_frame_ids"],
                                  "reference_reviewed_frames":w["reference_axis"]["reviewed_frame_ids"]}
                                 for wi,w in enumerate(visual["windows"])
                                 if (w["query"],w["reference"],w["parent_id"])==(qa,ra,parent["parent_id"])]
    records.append(r)

check(len(records)==39,"all39_records")
check(len({(r["query"],r["reference"]) for r in records})==12,"all12_pairs")
for name,ag in method_aggregate.items():check(ag["parents"]==39,"all39_method:"+name)
check(len({ag["fixed_reference_frames"] for ag in method_aggregate.values()})==1,"same_reference_axis_denominator_all_methods")
beforepins=dict(pins)
for p,h in beforepins.items():check(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,"source_unchanged:"+p)
pin(Path(__file__))
result={
    "schema":"main_phase_evidence_scope_ledger.v1","status":"PASS_SCOPE_JOIN_NOT_ACCURACY" if not failures else "FAIL",
    "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"check_count":checks,"failures":failures,
    "scope":{"parents":39,"pairs":12,"clips":16,"words":4,"methods_per_parent":5,"new_observation":0,"new_path":0},
    "feature_contract":{"columns":feature["feature_columns"],"groups":groups,
        "availability_only":"Support counts quantify stored finite known observations, not discriminative validity, accuracy or indispensability.",
        "overlapping_dependencies_not_additive":"Dependency families reuse raw columns and must not be summed as independent evidence dimensions.",
        "finger_angles":"14 projected internal angles per hand, non-circular and normalized by pi.",
        "rotation_limits":"Saved wrist-to-MCP 2D direction is not palm-normal/pronation, true3D rotation or anatomical wrist-flexion truth.",
        "dependency_support":"Hand/forearm derived-value dependencies require all listed masks and finite nonzero forearm; no new derived angle or residual is computed."},
    "evaluation_contract":{"all_labels_evaluator_only":True,
        "broad_roles":"PHASE_DATA sampled draft preparation/core_candidate/return scopes remain separate from finer physical kinds.",
        "unknown":"Outside a saved broad core is UNKNOWN_UNANNOTATED_GAP; physical boundary/unreviewed remains unknown. Neither is padded by adjacent kinds.",
        "same_kind_not_truth":"Same draft physical kind/recorded occurrence is a candidate range, not unique true frame or linguistic acceptability.",
        "different_kind_not_error":"Different physical subphase can be optional hold/normal timing variation; do not score as learner error or automatically reject a normal reference.",
        "fixed_denominator":"Every fixed reference-parent frame retained per method. Linked query extent can differ; common union envelope reported separately. More path cells are not more original observations.",
        "candidate_selection":"All same-kind and same-family draft candidates are retained; no residual search, best-reference selection or new thresholds.",
        "old_outputs":"Old handshape classifier and independent hand evidence remain baseline provenance only; not rerun or claimed updated after recovery."},
    "method_aggregates":{k:dict(v) for k,v in method_aggregate.items()},
    "changed_parent_counts":{f"{a}__{b}":sum(not next(x for x in r["method_differences"] if x["before"]==a and x["after"]==b)["same_parent_correspondences"] for r in records)
                            for a,b in [("B_global_original","C_relational_original"),("C_relational_original","phase_candidate_original"),("recovery_E0","phase_candidate_recovery_E1"),("C_relational_original","recovery_E0")]},
    "records":records,"source_pins":[{"path":p,"sha256":h} for p,h in pins.items()],
    "actions":{"new_inference":0,"new_alignment":0,"new_feature_extraction":0,"new_residual_scoring":0,"new_RGB_review":0,"original_or_prior_result_modifications":0},
    "limits":["Same Codex reused prior nonblind physical review, not independent human KSL ground truth.",
              "Source-supported candidate ranges and observation availability alone do not establish a repairable alignment error.",
              "Normal hand lowering, boundary transition and unseen information are separate states; missing is not meaninglessness.",
              "No role/physical label or acceptable-time candidate is licensed as runtime alignment input or a supervised training target."]
}
with OUT.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
print(json.dumps({k:result[k] for k in ["status","check_count","failures","scope","method_aggregates","changed_parent_counts"]},ensure_ascii=False))
print(hashlib.sha256(OUT.read_bytes()).hexdigest())
