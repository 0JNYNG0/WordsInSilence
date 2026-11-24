"""Add denominator/contrast detail from the already saved availability audit."""
from pathlib import Path
import json,hashlib,collections
U=Path(__file__).resolve().parent
p=U/"DIAGNOSTIC_FEASIBILITY_REVIEW.json"
initial=U/"DIAGNOSTIC_FEASIBILITY_REVIEW_INITIAL.json"
assert not initial.exists()
raw=p.read_bytes();initial.write_bytes(raw)
d=json.loads(raw.decode("utf-8"));a=d["common_denominator_audit"]["sparse_anchor_opportunities"]
detail={
 "target_opportunities":len(a),
 "unique_pair_reference_frames":len({(x["query"],x["reference"],x["reference_frame"]) for x in a}),
 "unique_parents":len({x["parent_id"] for x in a}),
 "unique_pairs":len({(x["query"],x["reference"]) for x in a}),
 "by_query_target_opportunities":dict(collections.Counter(x["query"] for x in a)),
 "current_connection_direct_feature_disagreement_opportunities":sum(any(v!=x["reference_label"] for v in x["current_query_labels"]) for x in a),
 "any_same_family_anchor_direct_feature_disagreement_opportunities":sum(any(v["label"]!=x["reference_label"] for v in x["same_family_known_query_anchors"]) for x in a),
 "all30_have_current_and_candidate_confirmed_attribute_agreement":all(all(v==x["reference_label"] for v in x["current_query_labels"]) and all(v["label"]==x["reference_label"] for v in x["same_family_known_query_anchors"]) for x in a),
 "what_can_be_diagnosed":"Agreement-preservation or physical-attribute false-discrepancy sensitivity on these exact cells may be examined. Predictions could still be wrong. No direct true-feature disagreement in this common subset supports difference-detection effectiveness or proves a stage bottleneck.",
 "what_cannot_be_diagnosed":"The30 target opportunities are9 shared reference-frame instances, not30 independent clips. Confirmed same-kind ranges remain candidate hypotheses; unchanged confirmed attributes do not provide a naturally occurring wrong-stage contrast."
}
d["common_denominator_audit"]["sparse_refinement"]=detail
d["bounded_next_action"]["zero_full_intervals_does_not_mean_all_data_unusable"]=True
d["bounded_next_action"]["minimum_existing_only_example"]={
 "query":"P036","reference":"P034",
 "observed_anchor_pairs":[
   {"query_frame":84,"reference_frame":57,"R_index_only":[0,0],"R_open_four":[1,1],"R_closed_four":[0,0],"draft_kind":"right_open_hand_near_lower_face"},
   {"query_frame":122,"reference_frame":91,"R_index_only":[0,0],"R_open_four":[0,0],"R_closed_four":[1,1],"draft_kind":"closed_hand_reorientation_hold"}
 ],
 "reuse":"All values already appear in the stored support ledger. No new RGB review is necessary to test that two R_index_only negatives do not prove the same complete handshape.",
 "optional_mechanics_control":"A deliberately cross-paired84↔91 or122↔57 test would provide a known attribute contrast from existing anchors, but is a constructed wrong-connection control, not an existing automatic alignment error or learner error. This is proposed only; not run.",
 "source":"sparse_anchor_opportunities, target entries R_index_only/R_open_four/R_closed_four for P036/P034"
}
d["bounded_next_action"]["minimum_extra_review_scope"]={
 "needed_now_for_existing_logic_example":0,
 "only_if_identifiable_pilot_otherwise_blocked":"Select a predeclared existing saved-path endpoint and a directly confirmed center from the same proposed opportunity; review just the one missing side/attribute and its short existing context. Do not extend a label to a whole parent.",
 "upper_bound":"At most one paired event window per currently supported expression, maximum4 paired windows total; stop at that bound. This is a suggested optional cap, not a requirement to annotate all4 expressions.",
 "no_outcome_selection":"Choose by missing support and existing annotation boundaries before model outputs, not by highest errors or desired effect. If the attribute remains invisible or allowed-stage alternatives remain multiple, keep unknown and close that item.",
 "not_executed":True
}
d["verdict_ko"].append("구간 전체 교집합0은 보유자료 전체가 무용하다는 뜻이 아니다. sparse30항목기회는9고유참고프레임/8쌍에 모여 있고 모두 확인특징 일치라, 현재는 물리특징 일치 보존·불필요한 차이신호 진단에 제한된다.")
d["source_pins"].append({"path":str(Path(__file__).resolve()),"sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
d["initial_review_preserved"]={"path":str(initial.resolve()),"sha256":hashlib.sha256(raw).hexdigest(),"reason":"Final support refinement; no initial finding retracted or experiment rerun."}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(detail,ensure_ascii=False))
print(hashlib.sha256(p.read_bytes()).hexdigest())
