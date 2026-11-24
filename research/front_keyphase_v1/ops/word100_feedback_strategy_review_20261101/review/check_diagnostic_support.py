"""Read-only label/scope join, no predictions, paths or scores recomputed."""
from pathlib import Path
import collections,hashlib,json,datetime
import numpy as np
U=Path(__file__).resolve().parents[1];OPS=U.parent
pins={}
def read(p):
    p=Path(p);pins[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding="utf-8"))
def pin(p):
    p=Path(p);pins[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
case=read(OPS/"word100_action_feedback_casebook_20261101/INPUTS.json")
data=read(OPS/"word100_semantic_feature_cycle_20261101/DATASET.json")
contract=read(OPS/"word100_semantic_feature_cycle_20261101/FEATURE_CONTRACT.json")
learn=read(OPS/"word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json")
phys=read(OPS/"word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json")
guide=read(OPS/"word100_action_feedback_plan_20261101/review/DATA_PLAN_REVIEW.json")
prior=read(OPS/"word100_semantic_feature_cycle_20261101/review/DATA_AUDIT.json")
shape=read(OPS/"word100_independent_hand_evidence_20261101/review/SHAPE_GUIDE_LINKS.json")
pin(U/"PLAN_REVIEW_KO.txt")
alias_by_clip={r["clip_id"]:a for a,r in case["rows"].items()}
targets=data["targets"];labels={};conflicts=[]
for a,row in case["rows"].items():
    pin(row["targets_file"])
    with np.load(row["targets_file"],allow_pickle=False) as z:
        for t in targets[:3]:
            col=learn["all_targets"].index(t)
            for f in np.flatnonzero(z["target_mask"][:,col]):
                labels[(a,int(f),t)]=int(z["y"][f,col])
centers=[r for r in data["rows"] if r["clip_id"] in alias_by_clip]
for r in centers:
    a=alias_by_clip[r["clip_id"]]
    for i,t in enumerate(targets):
        if not r["target_mask"][i]:continue
        key=(a,r["center_frame"],t);value=int(r["y"][i])
        if key in labels and labels[key]!=value:conflicts.append(key)
        labels[key]=value
for key in conflicts:labels.pop(key,None)
pc={r["alias"]:r for r in phys["clips"]}
def region(a,f):
    return next((r for r in pc[a]["regions"] if r.get("kind") and r["core"][0]<=f<=r["core"][1]),None)
def family(r):
    return phys["kinds"].get(r["kind"],{}).get("physical_kind",r["kind"])
paths={}
for p in case["natural"]:
    key=p["query"],p["reference"]
    if key not in paths:
        paths[key]=read(OPS/"word100_relational_sequence_compare_20261031/pairs"/f"C_relational_path_{key[0]}_{key[1]}.json")["path"]
counts={};anchor_opportunities=[]
for t in targets:
    cnt=collections.Counter()
    for p in case["natural"]:
        qa,ra=p["query"],p["reference"];path=paths[(qa,ra)]
        for rf in range(p["reference_range"][0],p["reference_range"][1]+1):
            cnt["fixed_reference_opportunities"]+=1
            if (ra,rf,t) not in labels:continue
            cnt["reference_direct_feature_known"]+=1
            qs=[q for q,r in path if r==rf]
            current_known=all((qa,q,t) in labels for q in qs)
            if current_known:cnt["reference_and_all_saved_Cmatched_query_features_known"]+=1
            rr=region(ra,rf)
            if not rr:continue
            cnt["reference_direct_feature_and_physical_core_known"]+=1
            candidates=[r for r in pc[qa]["regions"] if r.get("kind") and family(r)==family(rr)]
            full=[r for r in candidates if all((qa,f,t) in labels for f in range(r["core"][0],r["core"][1]+1))]
            known_anchors=[{"query_frame":f,"region_id":r["id"],"region_core":r["core"],"kind":r["kind"],
                           "occurrence":r.get("occurrence"),"label":labels[(qa,f,t)]}
                          for r in candidates for f in range(r["core"][0],r["core"][1]+1) if (qa,f,t) in labels]
            if full:cnt["full_samefamily_query_core_with_all_feature_labels"]+=1
            if current_known and full:cnt["full_interval_common_2x2_source_opportunities"]+=1
            if current_known and known_anchors:
                cnt["sparse_anchor_common_source_opportunities"]+=1
                anchor_opportunities.append({"target":t,"parent_id":p["id"],"query":qa,"reference":ra,
                    "reference_frame":rf,"reference_label":labels[(ra,rf,t)],"reference_kind":rr["kind"],
                    "reference_occurrence":rr.get("occurrence"),"current_query_indices":qs,
                    "current_query_labels":[labels[(qa,q,t)] for q in qs],
                    "same_family_known_query_anchors":known_anchors,"unique_correct_query_time":None,
                    "eligible_as_confirmed_stage_oracle":False})
    counts[t]=dict(cnt)
support={t:{role:{"cells":sum(1 for (a,f,tt) in labels if tt==t and case["rows"][a]["role"]==role),
                 "positive":sum(v==1 for (a,f,tt),v in labels.items() if tt==t and case["rows"][a]["role"]==role),
                 "negative":sum(v==0 for (a,f,tt),v in labels.items() if tt==t and case["rows"][a]["role"]==role)}
            for role in ("fit","development")} for t in targets}
out={
 "schema":"feedback_strategy_diagnostic_feasibility_review.v1",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "status":"REVISE_BEFORE_EXECUTION_FULL_2X2_NOT_READY",
 "reviewer":"Codex data reviewer; metadata/label join only, not independent human KSL adjudication",
 "verdict_ko":[
   "현재 자료로 39구간 전체의 확인특징×확인단계 2×2를 그대로 실행하는 계획은 통과시키기 어렵다. 직접 확인된 특징은 대부분 중심 프레임이고 단계는 물리 초안이므로, 구간 정답으로 확장하면 근거가 바뀐다.",
   "동일 출처의 정확한 프레임·항목에서 제한적인 보유근거 교집합은 조사할 수 있지만 이를 독립적인 단계 정답이나 교육 정확도로 보고해서는 안 된다.",
   "전체 단어 주석이나 추가 모델 탐색을 필수로 만들지 말고, 아래 지원 gate를 통과하는 한 항목의 작은 공통 분모가 실제 있는지 먼저 고정해야 한다."
 ],
 "scope":{"current_clips":16,"current_words":4,"development_queries":4,"fit_references":12,"natural_pairs":12,
          "parent_scopes":39,"fixed_reference_frames":440,"physical_certain_reference_frames":360,"physical_boundary_reference_frames":80,
          "seven_target_center_rows_in_current_clips":len(centers),"label_merge_conflicts":conflicts},
 "direct_label_support_after_exact_dedup":support,
 "common_denominator_audit":{"current_connection_inspected":"saved C_relational only for label availability, not re-scored or chosen by outcome",
      "per_target":counts,"sparse_anchor_opportunities":anchor_opportunities,
      "full_interval_common_count":sum(c.get("full_interval_common_2x2_source_opportunities",0) for c in counts.values()),
      "sparse_anchor_common_count":len(anchor_opportunities),
      "strict_full_interval_count_is_not_a_requirement_to_label_every_frame":"Zero full intervals rejects only full-interval replacement. A separately declared sparse-anchor diagnostic can use exact known cells without claiming interval truth.",
      "current_path_not_yet_fixed_in_plan":"C/U/W/learned3 routes differ. A diagnostic must name one frozen current baseline; availability on C cannot be silently generalized to other paths."},
 "overlap_and_absence":{
   "learned3":shape["targets"],"learned7":targets,
   "common3_are_independent_binary_attributes":"Both-negative agreement does not identify an equal handshape. Open_four0 is not a fist; index_only0 is not open_four.",
   "seven_are_center_targets":"13-native-frame context input, direct center-only output/label; not13 labeled frame targets or mandatory action phases.",
   "dense_prior_shape":"Existing36 windows/468frames from old shape bundle are currently FIT only. Current four DEV clips do not acquire dense truth from these files. Dedup overlaps with stage targets_v2.",
   "physical":"16 existing clips; physical family/subphase/occurrence drafts. Broad PHASE and physical certain/boundary masks are different layers; no unique correct cross-person time.",
   "guide":"All4 full guide/form links remain provisional.12 of39 parents have30 existing clause links concentrated in recorded person01:24 partial,3 proximity,3 unverified;27 parents physical-only.",
   "absent":["independent human language correctness/tolerance","actual learner-error labels","exact cross-person acceptable correspondence set","dense DEV7-target truth","3D palm rotation/contact/wrist bend truth","mandatory repetition/active-vs-support-hand training targets"],
   "input_schema_blocker":"7 binary features cannot replace raw69 geometry, hand trajectories or C13 costs as if they were corrected coordinates. Freeze an attribute-level comparator if using those labels; leave other channels untouched."
 },
 "circularity_and_leakage":{
   "same_source_observations":"Feature labels, PHYSICAL kinds and source-linked descriptions largely come from the same public RGB and same Codex review, sometimes incorporating already exposed examples. They are not independent corroboration.",
   "label_identity_circularity":"Kinds such as open hand or thumb-up/higher contain the same property tested as a feature. Selecting same-kind ranges can make feature agreement follow from selection; omit that feature from the independent stage evidence or report non-identifiability.",
   "oracle_metric_circularity":"Replacing a prediction with its label then scoring against that same label trivially improves that attribute; it does not identify an end-to-end causal bottleneck or establish that learning is needed.",
   "training_leakage":"Keep manual intervention values, stage intervals and branch outputs out of model training, hyperparameter/candidate selection and operational automatic-performance totals. A new implementation after diagnosis needs separate unused evaluation.",
   "exposure":"Current4 expressions/person04 DEV are repeatedly exposed; historical held names changed roles. No currently unused same-support evaluation cohort was established by this bounded review.",
   "seed_inflation":"3 classifier seeds are repeated predictions, not3 independent people or additional RGB labels."
 },
 "minimum_support_gates":[
   {"id":"G1_UNIT","rule":"Before outputs, freeze clip/native frame/side/attribute identity, one current predictor/path and one attribute-level comparison rule. Current and confirmed feature values must share schema."},
   {"id":"G2_IDENTICAL_DENOMINATOR","rule":"Every selected opportunity remains in all4 branches with the same reference/query universe. Known subset may be reported but unknown/transition/non-identifiable remainder and excluded reasons remain on the full ledger."},
   {"id":"G3_DIRECT_FEATURE","rule":"Use only exact direct RGB-confirmed feature cells with original target_mask and compatible definition; never propagate center labels through13-frame windows or PHYSICAL ranges, infer0 fromunknown, or synthesize corrected69 coordinates."},
   {"id":"G4_STAGE_SUPPORT","rule":"A confirmed-stage intervention needs a genuinely supported allowed range/occurrence set for that feature opportunity, defined without reading the candidate score. Same-kind label alone is a diagnostic candidate, not proof. Preserve multiple ranges/unknown boundaries."},
   {"id":"G5_NONCIRCULAR_ENDPOINT","rule":"Use an independent downstream observable claim not mechanically identical to intervention labels. If feature and stage confirmation share defining evidence, mark attribution non-identifiable rather than deciding feature-vs-alignment bottleneck."},
   {"id":"G6_USEFUL_CONTRAST","rule":"For any claimed ability to find differences, require observed agreement and disagreement examples for that exact attribute; all-negative/all-normal common cells permit preservation only. No minimum global word count or whole-corpus relabeling."},
   {"id":"G7_DIAGNOSTIC_NOT_ADOPTION","rule":"A passing bounded diagnostic licenses a causal hypothesis only, not model training by default. First check code, masks, scope and comparison logic; only independent training support plus held evaluation justifies supervised work."}
 ],
 "bounded_next_action":{
   "first":"Freeze the existing-label support inventory and one narrow observable attribute. Confirm whether sparse common opportunities actually have independent stage evidence and a noncircular endpoint.",
   "if_sparse_common_exists":"Use all eligible existing anchor opportunities in a declared pilot, retain all alternative same-family anchors as uncertainty, and call it sensitivity/consistency diagnosis. Do not call either intervention an oracle upper bound.",
   "if_stage_or_feature_confirmation_missing":"Do not build a fabricated4-cell table. Report the identifiable1-factor comparison or two linked audits; conditional independence and interaction remain unknown.",
   "optional_minimum_extra_review":"Only if a concrete unresolved opportunity blocks an otherwise valid common unit, review that exact existing public query/reference event window once, with a cap of one event window per supported expression (maximum4, no new words/persons). Annotate only the one feature and allowed interval/unknown, preserving same-Codex provenance. This is a proposed bounded action, not executed or mandatory.",
   "stop":"If no noncircular endpoint/common support appears within that cap, close the2×2 branch as infeasible on present data and state what truth is missing. Do not require full16/39/100-word reannotation.",
   "alternative_now":"Saved physical-label availability and exact categorical comparison logic can be audited without new paths/models. A technical synthetic known-source test can test mechanics, never learner errors."
 },
 "open_unknowns":["which frozen current feature model/output route the2x2 uses","whether any allowed-stage range is independently justified beyond same-kind descriptions","whether a noncircular downstream feedback endpoint exists","educational salience/tolerance of each observable difference","unused public support-matched evaluation availability","end-to-end latency of an implemented new flow"],
 "actions":{"new_experiment":0,"new_training":0,"new_inference":0,"new_path_or_rescoring":0,"new_RGB_review":0,"private_data_access":0},
 "source_pins":[{"path":p,"sha256":h} for p,h in pins.items()]
}
dest=U/"review/DIAGNOSTIC_FEASIBILITY_REVIEW.json"
assert not dest.exists(),"Preserve initial review."
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":out["status"],"center_rows":len(centers),"full_interval_common":out["common_denominator_audit"]["full_interval_common_count"],"sparse_anchor_common":len(anchor_opportunities),"anchor_words":sorted({case["rows"][x["query"]]["word_id"] for x in anchor_opportunities}),"per_target":counts},ensure_ascii=False))
print(hashlib.sha256(dest.read_bytes()).hexdigest())
