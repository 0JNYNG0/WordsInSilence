"""Planning-only audit of saved JSON/text evidence; no model/arrays/video execution."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json

R=Path(__file__).resolve().parent
U=R.parent;OPS=U.parent
PINS={}
def read(rel):
 p=OPS/rel;b=p.read_bytes();PINS[str(p)]=hashlib.sha256(b).hexdigest()
 return json.loads(b.decode("utf-8-sig"))
def pin(rel):
 p=OPS/rel;PINS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

shape=read("word100_independent_hand_evidence_20261101/review/SHAPE_GUIDE_LINKS.json")
input_audit=read("word100_independent_hand_evidence_20261101/review/INPUT_AUDIT.json")
rgb=read("word100_independent_hand_evidence_20261101/review/RGB_VISIBILITY_REVIEW.json")
phase=read("word100_core_stage_evidence_20261031/PHASE_DATA.json")
physical=read("word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json")
refs=read("word100_target_known_feedback_20261101/REFERENCE_PHASES.json")
source=read("word100_source_token_alignment_20261027/SOURCE_FORM_LINKS.json")
source_ready=read("word100_source_token_alignment_20261027/ANNOTATION_READINESS.json")
source_learn=read("word100_source_token_alignment_20261027/LEARNING_CONTRACT.json")
shared=read("word100_shared_feature_bundle_20261027/RESULT.json")
shared_manifest=read("word100_shared_feature_bundle_20261027/LEARNING_MANIFEST.json")
guide_map=read("word100_stage_feedback_cycle_20261031/review/guide_stage_map.json")
learn_audit=read("word100_semantic_feature_cycle_20261101/review/DATA_AUDIT.json")
learn_contract=read("word100_semantic_feature_cycle_20261101/FEATURE_CONTRACT.json")
old_description_summary=read("word100_description_evidence_feedback_20261101/SUMMARY.json")
old_description_manifest=read("word100_description_evidence_feedback_20261101/RESULT_MANIFEST.json")
old_description_report=read("word100_description_evidence_feedback_20261101/REPORT_DATA.json")
for rel in ("word100_source_token_alignment_20261027/RESULT_KO.txt",
 "word100_shared_feature_bundle_20261027/RESULT_KO.txt",
 "word100_semantic_feature_cycle_20261101/RESULT_KO.txt",
 "word100_independent_hand_evidence_20261101/RESULT_KO.txt",
 "word100_independent_hand_evidence_20261101/NEXT_PLAN_KO.txt",
 "word100_independent_hand_evidence_20261101/FINAL_VERIFICATION.json",
 "word100_independent_hand_evidence_20261101/review/RESULT_VERIFICATION.json",
 "word100_independent_hand_evidence_20261101/review/RESULT_SUMMARY_VERIFICATION.json",
 "word100_description_evidence_feedback_20261101/RESULT_KO.txt",
 "word100_description_evidence_feedback_20261101/review/SAVED_VERIFICATION.json"):
 pin(rel)
draft=read("word100_action_feedback_plan_20261101/PLAN_DRAFT.json")
final_plan=read("word100_action_feedback_plan_20261101/PLAN.json")
assert final_plan["execution_status"]=="NOT_STARTED_PLAN_ONLY"
assert final_plan["scope"]["query_aliases"]==["P036","P048","P066","P072"]
assert final_plan["scope"]["pairs"]==12 and final_plan["scope"]["parent_opportunities"]==39
assert final_plan["scope"]["linked_parent_opportunities"]==12
assert final_plan["scope"]["unlinked_parent_opportunities_retained"]==27
assert final_plan["scope"]["source_clause_links"]==30
unique={r["parent_id"]:r for r in shape["records"]}
supported={k:r for k,r in unique.items() if r["reference_parent"].get("source_links")}
other={k:r for k,r in unique.items() if not r["reference_parent"].get("source_links")}
assert len(unique)==39 and len(supported)==12 and len(other)==27
first_instance_count=sum(r["parent_id"] in supported for r in shape["records"])
assert first_instance_count==36
dev=[r for r in shape["records"] if r["role"]=="development"]
assert len(dev)==39 and len({(r["query"],r["reference"]) for r in dev})==12
assert sum(r["parent_id"] in supported for r in dev)==12
dev_cases=[]
for q in ("P036","P048","P066","P072"):
 rs=[r for r in dev if r["query"]==q]
 ls=[r for r in rs if r["parent_id"] in supported]
 dev_cases.append(dict(query=q,references=sorted({r["reference"] for r in rs}),parent_instances=len(rs),
  source_linked_parent_instances=len(ls),physical_only_parent_instances=len(rs)-len(ls),
  clause_link_rows=sum(len(r["reference_parent"]["source_links"]) for r in ls)))
assert old_description_summary["development"]["pairs"]==12
assert old_description_summary["development"]["stages"]==39
assert old_description_summary["development"]["source_context_stages"]==12
assert old_description_summary["development"]["source_cards"]==30
words={r["word_id"] for r in unique.values()}
cards=shape["guide_cards"]
assert len(cards)==4 and all(c["link"]["state"]=="SOURCE_CANDIDATE_NOT_FORM_VERIFIED" for c in cards)
candidates=[c for card in cards for c in card["candidates"]]
assert len(candidates)==9 and all(c["status"]=="PROVISIONAL_EXPECTATION_ONLY" for c in candidates)
byword=[]
features={
 "WORD0739":["closed right-hand outline at listed samples","right forearm lateral/chest/shoulder relation","observed lowering/reorientation; requiredness unresolved"],
 "WORD1133":["open right-hand outline and closed left hand","right hand relative to left forearm","movement toward distal forearm; physical contact/rubbing not certified"],
 "WORD1567":["thumb-raised closed outline at listed samples","alternating hand-height arrangement","observed alternation/order, not mandatory repeat count"],
 "WORD2523":["fingertip proximity only","oblique symmetric projected arrangement","separation/lowering; final finger gathering remains edge-on/unverified"]
}
status=Counter();anchors=set();clause_rows=[]
for word in sorted(words):
 wr=[r for r in unique.values() if r["word_id"]==word]
 linked=[r for r in wr if r["parent_id"] in supported]
 ll=[(r,s) for r in linked for s in r["reference_parent"]["source_links"]]
 counts=Counter(s["status"] for r,s in ll);status.update(counts)
 af={(r["reference"],f) for r,s in ll for f in s.get("anchor_frames",[])};anchors|=af
 for r,s in ll:
  assert s["training_target"] is False
  assert all(r["reference_range"][0]<=f<=r["reference_range"][1] for f in s["anchor_frames"])
  clause_rows.append(dict(reference=r["reference"],parent_id=r["parent_id"],word_id=word,
    parent_range=r["reference_range"],original_clause=s,
    interpretation="Existing saved source-link status, not a new label or full-span truth."))
 byword.append(dict(word_id=word,label=next(c["label"] for c in cards if c["word_id"]==word),
  total_unique_fit_parents=len(wr),linked_unique_parents=len(linked),
  remaining_physical_only=len(wr)-len(linked),linked_reference_aliases=sorted({r["reference"] for r in linked}),
  linked_parent_ids=[r["parent_id"] for r in linked],clause_link_rows=len(ll),
  source_status_counts=dict(counts),unique_rgb_anchor_frames=len(af),feature_scope=features[word]))
assert len(clause_rows)==30 and len(anchors)==30
shared_current=[r for r in shared_manifest["sequence"] if r["word_id"] in words]
assert len(shared_current)==12
source_overlap=[r["word_id"] for r in source["links"] if r["word_id"] in words]
assert source_overlap==[] and source["full_variant_confirmed_count"]==0
learning={}
for t in learn_audit["core_supported_static_fit_targets"]:
 learning[t]=learn_audit["core_static_support_recount"]["fit"][t]

result=dict(
 schema="wis_action_feedback_plan_data_review_v1",
 status="PASS_WITH_EXPLICIT_SCOPE_LIMITS_PLAN_ONLY_NO_EXPERIMENT_AUTHORIZATION",
 at_utc=datetime.now(timezone.utc).isoformat(),
 reviewer="Codex semantic_data_audit; saved-artifact planning review, not new expert language annotation",
 answer_ko=[
  "네 표현의 사전 전체 표현형은 모두 잠정 연결이다. 그러나 기존 RGB 물리 동작 설명과 일부 지정 시점 연결까지 미검토인 것은 아니다.",
  "첫 설명 연결은 기존 12고유부모·30문구연결을 그대로 재사용할 수 있다. 그중24는 부분일치,3은 근접만 확인,3은 여전히미확인이며 전부 기록ID01의4참고영상에 몰려 있다.",
  "첫 산출물은 기존 기능을 재개발하지 않고4DEV×3FIT의12쌍39부모를4사례 묶음으로 검토한다. 그중12부모만 사전문구 연결이 있고 나머지27은 물리설명/관측근거만 유지한다. 원117전체 결과도 보존하며 다시 계산·수동전수검토하지 않는다.",
  "주요 동작의 관측 차이를 설명하는 작은 흐름은 가능하지만, 무엇을 고쳐야 하는지 또는 허용 변이인지를 확정할 언어·교육 정답은 없다."
 ],
 current_scope=dict(words=4,clips=16,recorded_people=[1,2,3,4],fit_clips=12,development_clips=4,
  pairs=36,parent_instances=117,unique_fit_parents=39,source_linked_unique_parents=12,
  source_linked_parent_instances=36,remaining_unique_physical_only_parents=27,
  remaining_parent_instances=81,prior_position_records=585,prior_shape_seed_links=351,
  source_linked_reference_clips=4,source_linked_recorded_people=[1],
  distinct_source_anchor_frames=30,clause_link_rows=30,clause_status_counts=dict(status)),
 first_deliverable=dict(kind="Reuse existing description_evidence_feedback cards/trace; bounded review, no new comparison engine.",
  development_queries=dev_cases,pairs=12,parent_instances=39,source_linked_parent_instances=12,
  physical_only_parent_instances=27,clause_link_rows=30,
  original_full_scope="117 full instances remain preserved. Original36 linked/81physical-only differs from selected DEV12 linked/27physical-only.",
  existing_unit=str(OPS/"word100_description_evidence_feedback_20261101"),
  existing_development_summary=old_description_summary["development"],
  no_duplicate_engine=True,
  denominator_warning="Previous84 linked measurement opportunities and507 all group opportunities are not accuracy, videos or new32-dimensional feature cells. Within-small-reference-sample is not normal/correct truth."),
 guide_status=dict(all_four_cards_provisional=True,all_nine_candidates_provisional=True,
  fully_verified_language_variants=0,mandatory_semantic_stage_labels=0,
  distinction="Full dictionary sense/form equivalence is unconfirmed; the frozen physical RGB descriptions and limited source-anchor observations are separately reusable.",
  latest_reconciliation_summary=guide_map["summary"],cards=[dict(word_id=c["word_id"],label=c["label"],
    link=c["link"],candidates=[dict(entry_id=x["entry_id"],status=x["status"],description=x.get("description")) for x in c["candidates"]]) for c in cards]),
 first_connection_by_word=byword,existing_clause_rows=clause_rows,
 temporal_evidence_layers=[
  dict(source=str(OPS/"word100_core_stage_evidence_20261031/PHASE_DATA.json"),
   scope=phase["summary"],reuse="Original parent descriptions, evidence frames/PTS, core spans and boundary brackets;39 FIT parent definitions reused as metadata and physical context.",
   not_proven="Dense semantic truth, mandatory start/end/hold stages, tolerance and pedagogical correctness."),
  dict(source=str(OPS/"word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json"),
   status=physical["status"],scope=physical["scope"],neural_training_target=physical["neural_training_target"],linguistic_truth=physical["linguistic_truth"],
   reuse="Known physical kind/occurrence and uncertain boundary provenance for after-output audit. FIT projection is in REFERENCE_PHASES; DEV query annotations remain audit-only.",
   not_proven="Physical movement-vs-hold mismatch is not automatically a wrong sign. Do not relabel these audit annotations as a new ready training target."),
  dict(source=str(OPS/"word100_stage_feedback_cycle_20261031/review/guide_stage_map.json"),
   reuse="Latest saved reconciliation of old source text, sparse visual samples and dense short-window observations; retains prior corrections and dedup lineage.",
   not_proven="No fully verified guide variants or mandatory semantic stage labels were created.")
 ],
 older_sources=dict(
  source_token=dict(source=str(OPS/"word100_source_token_alignment_20261027/SOURCE_FORM_LINKS.json"),
   current_four_word_overlap=source_overlap,full_variant_confirmed_count=source["full_variant_confirmed_count"],
   reusable=[dict(word_id=r["word_id"],label=r["label"],state=r["state"],source_ids=r["source_ids"],rejected_ids=r["rejected_ids"],use=r["use"],unknown=r["unknown"]) for r in source["links"]],
   limits="Different four expressions. Useful precedent for homonym exclusion, coarse shape/relation and not forcing mismatched guides; cannot certify current four variants. Source spans are sampled observations, not dense frame labels."),
  shared_feature=dict(source=str(OPS/"word100_shared_feature_bundle_20261027/LEARNING_MANIFEST.json"),
   counts=shared["counts"],roles=shared["roles"],
   current_four_legacy_clips=[{k:r[k] for k in ("alias","clip_id","word_id","recorded_person_id","role")} for r in shared_current],
   reusable="Exact clip/frame/PTS/anatomical-side definitions;7 sparse static attribute drafts and sampled shape-change presence, including current12 FIT clips.",
   limits="Two Codex observations agreeing is not independent human truth. Sequence57/72 describes sampled change presence, not exact repeats or full temporal boundaries. Historical preparation-not-ready is not a claim that later approved seven-target training never occurred.",
   later_role_change="Historical held-expression Egypt and person3 are current FIT references; preserve role history. Do not present these clips as fresh held-out data now.")),
 learning_support=dict(
  current_plan_training=False,
  already_available_static_seven_fit_support=learning,
  source=str(OPS/"word100_semantic_feature_cycle_20261101/review/DATA_AUDIT.json"),
  later_training_already_completed="Existing96 clips/288 center frames and7 static outputs were already used in the frozen semantic cycle; preserve its results. Current action plan does not authorize a retraining or new selection sweep.",
  label_granularity="Central-frame static target only within13-frame context. Unknown cells excluded per target, not whole dataset; cannot spread center labels to all13frames.",
  shape_target_limit="Closed-four excludes thumb, so it is not a complete fist/precise handshape label. Native protected3 has no closed, thumb, orientation or wrist-bend output.",
  unsupported_new_targets=["3D fist axial rotation","anatomical wrist flexion","exact contact","linguistic mandatory repetition/order","acceptable deviations and true learner errors","normal crossing/hand-lowering educational acceptability"],
  training_gate_proposal="No learning in this plan. If a later separately authorized output has an identifiable observer error, use only directly scoped positive/negative labels, keep clip/person/expression role and dedup lineage, and predefine that output's evidence threshold. Existing other supported outputs need not wait for ambiguous ones.",
  no_new_global_gate="Do not require micro-down/hold or all guide variants to be resolved before reusing supported static/physical evidence."),
 minimal_plan=dict(
  first_pass="Reuse existing saved cards/trace for4 DEV queries x3 FIT references =12pairs/39parent instances. Review one metadata pass over their39 unique FIT parent definitions. Preserve12 linked/30 original clauses and27 physical-only instances. Original117 results remain immutable; no new all117 card pipeline.",
  required_record_fields=["practice word supplied by caller","reference clip and recorded role","parent and occurrence IDs","physical description with provenance","dictionary candidate and source-link status separately","original core/brackets/frameID/PTS","per-reference shape/relation/movement/order evidence pointers","own/counterpart missing masks","correspondence ambiguity","no correctness/correction claim"],
  key_reuse="Do not recompute927 readouts,585 positions,108 paths or models. Existing5 RGB frames may illustrate limitations; do not relabel from them.",
  no_global_variant_gate=True,manual117_review_required=False,
  optional_later_visual_review_cap=dict(current_plan_new_image_views=0,
   if_later_authorized="Maximum3 predeclared event windows per DEV query,12 windows total, one pass over already saved native parent/query range and its original RGB context. Fix exact reference/query frame IDs/PTS before that visual review; no full117 manual review, new data collection or post-review substitution. Keep entire needed motion context; isolated stills cannot establish movement or required repeats.",
   exposure="Selection is fixed before the additional visual review, but prior results have already been examined. This is not untouched or blinded evaluation.",
   authorization="Proposed future ceiling only; no visual review or experiment authorized in this plan-only turn."),
  success="Four case bundles reuse the existing event cards and exact trace, explain supported physical observations/differences, and separate observation missing, saved-path ambiguity and unsupported language claims. Do not redevelop the existing engine. No-evidence and unused parents remain counted.",
  terminate_when="After the one bounded pass, unresolved form/3D/tolerance claims stay unsupported. Deliver the limited observation-based result plus exact missing-data list; stop this branch rather than endlessly collecting labels or opening new model/distance loops.",
  if_no_semantic_truth="Explicitly end at observed-difference explanation. Offer no automatic correction, educational accuracy or requiredness judgment; future expert/independent data request is a separate decision.")
 ,
 evaluation_partition=dict(
  historical_not_fresh=True,
  evidence="The16 public clips, person4 DEV and4 expressions have already informed multiple analysis cycles. All older bundles retain priorTCN TRAIN lineage. Shared-feature historical split and current FIT role differ.",
  primary_scope="Known practice word with individual FIT references. Unseen-word transfer is a separate diagnostic, not a universal gate for this limited source-supported product.",
  separation=["Runtime observer consumes saved numeric evidence, not dictionary expected answers or query stage truth.",
   "FIT descriptions can name a reference event; DEV physical annotations are audit-only and cannot supply query ranges/correspondence.",
   "Preserve original values, unknowns, all117 opportunities and individual references; never choose the best reference after outcome.",
   "Synthetic edits test preservation and provenance only, not actual learner-error sensitivity.",
   "Report support/abstention and visible-difference coverage together. All-abstain is not evidence of success.",
   "A later new generalization claim requires a genuinely unused independently adjudicated cohort; current repeated DEV cannot supply it."]),
 draft_review=dict(reviewed_status=draft["status"],acceptable="Steps1-3 can proceed as bounded source-supported explanation after separate execution approval.",
  required_clarifications=["Use12/30 initial links with24partial+3proximity+3unverified; no all30 confirmed claim.",
  "Retain DEV27 physical-only instances in first-delivery denominator; original117 full-scope27unique/81instance physical-only also remains preserved.",
   "State initial source clauses cover4 ID01 references; do not transfer language requiredness to other people.",
   "Keep step4 a decision record with no training by default; current learning is neither required nor authorized.",
   "No new raw/video review in this planning turn; optional future ceiling and termination branch explicit."]),
 final_plan_review=dict(status="PASS_NO_IMPORTANT_REMAINING_DATA_SCOPE_MISMATCH",
  reviewed_plan_status=final_plan["status"],
  source=str(U/"PLAN.json"),
  confirms=["Existing description connection is already complete; the increment is cause-explanation in four DEV case bundles.",
   "Four queries x3 fixed FIT references =12pairs/39parents,12 source-linked/27physical-only,30 status-preserving clause links.",
   "No automatic reconstruction of language truth; ID01 concentration,24partial/3proximity/3unverified and old DEV exposure are explicit.",
   "Maximum12 event windows, explicit unresolved exit branch, no repeated117/manual or927/model/path execution.",
   "Learning and production adoption are not authorized by this planning review."]),
 actions=dict(new_experiment=0,training=0,inference=0,path_or_score_recalculation=0,raw_array_loading=0,
  new_visual_review=0,new_labels=0,private_or_mentor_access=0,root_document_changes=0),
 source_pins=PINS,script_sha256=sha(__file__))
for p,h in PINS.items():
 if Path(p).name not in ("PLAN_DRAFT.json","PLAN.json"):assert sha(p)==h,p
out=R/"DATA_PLAN_REVIEW.json"
with out.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=result["status"],source_pins=len(PINS),linked_parents=12,clause_rows=30,
 status_counts=dict(status),anchor_frames=len(anchors),sha256=sha(out))))
