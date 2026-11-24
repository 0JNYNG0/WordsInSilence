"""Metadata-only frozen RGB-window specification. Never opens/decodes images or video."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np

R=Path(__file__).resolve().parent;U=R.parent;OPS=U.parent
pins={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);b=p.read_bytes();pins[str(p)]=hashlib.sha256(b).hexdigest()
 return json.loads(b.decode("utf-8-sig"))
def pin(p,expected=None):
 p=Path(p);h=sha(p)
 assert expected is None or h==expected,str(p)
 pins[str(p)]=h;return h
plan=read(OPS/"word100_action_feedback_plan_20261101/PLAN.json")
data_review=read(OPS/"word100_action_feedback_plan_20261101/review/DATA_PLAN_REVIEW.json")
links=read(OPS/"word100_independent_hand_evidence_20261101/review/SHAPE_GUIDE_LINKS.json")
tasks=read(OPS/"word100_hold_invariant_trajectory_20261101/TASKS.json")
visual=read(OPS/"word100_stage_feedback_cycle_20261031/VISUAL_INDEX.json")
physical=read(OPS/"word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json")
old_review=read(OPS/"word100_independent_hand_evidence_20261101/review/RGB_VISIBILITY_REVIEW.json")
vis={r["alias"]:{x["frame_id"]:x for x in r["frames"]} for r in visual["rows"]}
by={(r["query"],r["reference"],r["parent_id"]):r for r in links["records"]}
choices=[
 ("P036","P033","P033_s02",["shape"],"Saved physical description explicitly identifies right open-finger outline near lower face. Check shape/support only, not a compulsory linguistic stage."),
 ("P036","P033","P033_s03",["movement"],"Saved physical description identifies hand closing while descending toward lateral chest. Examine full existing event and boundary context."),
 ("P036","P033","P033_s04",["relation"],"Saved physical description identifies closed right hand/forearm reorientation to shoulder-side relation; do not interpret screen orientation as3D wrist flexion."),
 ("P048","P045","P045_s02",["shape","relation","movement"],"The same saved physical event explicitly contains open right/closed left shapes, right hand versus opposite forearm and movement along that arm. Share one window across all3 categories instead of duplicating review."),
 ("P066","P063","P063_s02",["shape"],"Saved physical description identifies thumb-raised closed outline and the first hand-height arrangement. Existing native3 posterior does not classify thumbs or complete fists."),
 ("P066","P063","P063_s03",["relation"],"Saved physical description identifies the first exchange of the two hands' vertical arrangement. Check simultaneous hand/arm visibility and projected relation."),
 ("P066","P063","P063_s04",["movement"],"Saved physical description identifies the next opposite height exchange. Preserve order and motion/hold context without making a repeat count mandatory."),
 ("P072","P069","P069_s02",["relation"],"Saved physical description identifies two hands approaching a central triangular gap. Existing source link proves proximity only, not fingertip contact."),
 ("P072","P069","P069_s03",["movement"],"Saved physical description identifies separation toward the shoulders with lowering. Existing known preparation/core boundary mismatch remains a possibility, not an error verdict."),
 ("P072","P069","P069_s04",["shape"],"Saved physical description identifies flat, grouped-finger outlines at the final separated posture. Existing source suffix gathering is edge-on/unverified and must not become a label.")
]
cache={}
def axis(alias,core,context):
 row=tasks["rows"][alias]
 pin(row["input"],row["input_sha256"]);pin(row["source"],row["source_sha256"])
 if alias not in cache:
  with np.load(row["input"],allow_pickle=False) as z:
   cache[alias]=(z["frame_id"].copy(),z["timestamp_us"].copy())
 frame,pts=cache[alias]
 assert np.array_equal(frame,np.arange(len(frame)))
 assert 0<=context[0]<=core[0]<=core[1]<=context[1]<len(frame)
 sampled=[dict(vis[alias][i],sha256=pin(vis[alias][i]["file"])) for i in range(context[0],context[1]+1) if i in vis[alias]]
 return dict(alias=alias,clip_id=row["clip_id"],role=row["role"],person=row["person"],
  source=row["source"],source_sha256=row["source_sha256"],input=row["input"],input_sha256=row["input_sha256"],
  core_frame_range_inclusive=core,context_frame_range_inclusive=context,
  core_timestamp_us_inclusive=[int(pts[i]) for i in core],context_timestamp_us_inclusive=[int(pts[i]) for i in context],
  context_frame_ids=frame[context[0]:context[1]+1].tolist(),context_timestamp_us=pts[context[0]:context[1]+1].tolist(),
  existing_full_rgb_samples=sampled,
  previously_reviewed_physical_context=True,
  planned_new_activity="Paired cause/visibility review of this fixed window; saved full-frame JPEGs reused when exact native frame exists, otherwise local public source RGB decoding only.",
  new_extraction_or_image_opening_done=False)
windows=[]
for n,(q,r,parent,categories,reason) in enumerate(choices,1):
 link=by[q,r,parent];p=link["reference_parent"]
 ref_core=link["reference_range"];query_core=link["query_range"]
 outer=[p["start_bracket"][0],p["end_bracket"][1]]
 baseline=Path(link["baseline"]);d=read(baseline);path=d["path"]
 selected=[pair for pair in path if outer[0]<=pair[1]<=outer[1]]
 query_outer=[min(pair[0] for pair in selected),max(pair[0] for pair in selected)]
 corepath=[pair for pair in path if ref_core[0]<=pair[1]<=ref_core[1]]
 assert query_core==[min(pair[0] for pair in corepath),max(pair[0] for pair in corepath)]
 windows.append(dict(window_id=f"W{n:02}",query=q,reference=r,parent_id=parent,categories=categories,
  word_id=link["word_id"],source_description=p["description_ko"],selection_reason=reason,
  baseline_path_source=str(baseline),baseline_sha256=pins[str(baseline)],
  query_scope_rule="All query native indices touched by unchanged saved global path where reference lies in the fixed reference core/context. No query GT or new alignment.",
  reference_context_rule="Existing parent start_bracket lower endpoint through end_bracket upper endpoint, inclusive. No extra context selected after seeing RGB.",
  query_axis=axis(q,query_core,query_outer),reference_axis=axis(r,ref_core,outer),
  saved_core_correspondence=corepath,saved_context_correspondence=selected,
  previous_reference_rgb_evidence_frames=p.get("evidence_frames",[]),
  previous_source_links=p.get("source_links",[]),
  previous_physical_annotations_source=str(OPS/"word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json"),
  new_pair_review_status="NOT_STARTED",prior_evidence_reuse="Existing physical descriptions/phase/brackets/source anchors are reused. Their historical RGB review is not counted as new or independent truth.",
  prohibited_promotions=["visible proximity to actual contact","palm-axis2D to3D fist rotation or anatomical wrist flexion","normal-reference difference to learner error","unobserved or unreviewed to agreement"]))
assert len(windows)==10
category=[]
for q in plan["scope"]["query_aliases"]:
 for c in ("shape","relation","movement"):
  ids=[w["window_id"] for w in windows if w["query"]==q and c in w["categories"]]
  assert len(ids)<=1
  category.append(dict(query=q,category=c,window_ids=ids,status="EXISTING_PHYSICAL_DESCRIPTION_SUPPORTS_REVIEW" if ids else "NO_DOCUMENTED_EVENT_FOR_CATEGORY_NO_FORCED_FILL"))
assert sum(len(r["window_ids"]) for r in category)==12
for path,h in pins.items():assert sha(path)==h,path
result=dict(schema="wis_action_casebook_fixed_rgb_window_selection_v1",
 status="FIXED_SELECTION_AWAITING_ROOT_ACK_BEFORE_RGB_OPEN_OR_EXTRACTION",
 at_utc=datetime.now(timezone.utc).isoformat(),author="Codex semantic_data_audit",
 user_authorization="Root delegated execution of the user-approved reviewed action-feedback plan. This step only fixes RGB review windows.",
 selection_basis=["Existing saved physical parent descriptions, original parent boundaries and existing source anchors.",
  "Reference person01 chosen uniformly because the earlier source-clause anchors reside there; no residual or model-score ranking.",
  "One shape/relation/movement window at most per expression; multiple categories sharing the identical parent are deduplicated.",
  "Old results and annotations are historically used data, not a blinded or fresh cohort."],
 residual_or_probability_summary_consulted_for_selection=False,
 scope=dict(max_unique_windows=12,selected_unique_windows=10,category_slots=12,unique_videos=8,
  query_cases=4,newly_reviewed_pairs=4,full_casebook_pairs_preserved=12,full_parent_opportunities_preserved=39),
 all_three_references_preserved=True,other_references="P034/P035,P046/P047,P064/P065,P070/P071 retain existing physical/source/readout evidence; no new RGB review claims for them.",
 categories=category,windows=windows,
 review_protocol=dict(sequence="Root acknowledges this exact SHA before any new image opening/decoding. Then inspect both original RGB windows, not just isolated peak frames. Preserve all predetermined ranges and uncertain results.",
  record_fields=["window_id","reviewed native frameIDs/PTS and frame source hashes","query/reference visible hand/arm evidence","visible_difference_review with explicit evidence","observation_missing versus saved detector support","correspondence_contribution possible/uncertain","multiple coexisting explanations","unknown3D/contact/linguistic importance"],
  observations_not_labels=True,new_query_truth_not_used_at_runtime=True,
  stop="No replacement window or beyond-context search after review. Unresolved cause stays unresolved and the bounded unit can close.",
  result_path=str(R/"RGB_WINDOW_REVIEW.json")),
 source_pins=pins,script_sha256=sha(__file__),
 actions=dict(new_image_views=0,video_decoding=0,feature_extraction=0,model_inference=0,training=0,new_alignment=0,
  source_video_writes=0,old_artifact_writes=0,private_or_mentor_access=0))
out=R/"WINDOW_SELECTION.json"
with out.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=result["status"],windows=len(windows),category_slots=12,pins=len(pins),sha256=sha(out),
 frames_in_windows=sum(len(w[a]["context_frame_ids"]) for w in windows for a in ("query_axis","reference_axis")))))
