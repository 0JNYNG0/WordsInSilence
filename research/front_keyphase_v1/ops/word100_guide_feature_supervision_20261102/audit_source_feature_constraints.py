"""Source/logic audit only; reuse immutable FIT evidence, no inference or labels."""
import ast
import collections
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

AE = Path(__file__).resolve().parent
O = AE.parent
P = O / 'word100_learned_guide_feedback_20261102'
Q = O / 'word100_predicate_supervision_20261102'
S = O / 'word100_source_component_connector_20261102'
X = O / 'word100_learned_component_extent_20261102'
CARD = O / 'word100_guide_video_feedback_20261028/GUIDE_CARDS.json'
REPORT = O / 'word100_description_evidence_feedback_20261101/REPORT_DATA.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pointer(data, path):
    for s in path.strip('/').split('/'):
        s = s.replace('~1', '/').replace('~0', '~')
        data = data[int(s)] if isinstance(data, list) else data[s]
    return data


def spec(features, typed, relation, temporal, unsupported, observation_note):
    return dict(supported_component_if_any=features, existing_typed_component_ids=typed,
                logical_implication=relation, temporal_binding_requirements=temporal,
                additional_unsupported=unsupported, actual_observation_scope_note=observation_note)


SPECS = {
 'obedience_closed_hand': spec(
    [{'feature':'non_thumb_fingers_closed','hand':'right','relation':'necessary_partial_condition_of_explicit_fist_description'}],
    ['right_non_thumb_closed'],
    'Under the explicit fist reading, flexed/closed non-thumb fingers are a partial shape condition. The reverse is invalid: closed_four does not establish thumb placement, palm occlusion details, a complete named fist, or correctness.',
    ['Bind only a directly observed closed-finger point to the relevant described posture. Initial open-hand face approach and transitional images are not automatically fist targets.', 'Center-frame truth remains a point; do not fill every frame between canonical anchors.'],
    ['All thumb configurations', 'A whole-stage fist label from one center image', 'Fist inferred from open_four=0 or index_only=0'],
    'S already contains direct FIT right_non_thumb_closed targets. Reuse those limited physical labels; do not promote the earlier P context card to full phrase truth.'),
 'obedience_lateral_arm_then_lower': spec(
    [{'feature':'elbow_lateral_to_torso','hand':'right','relation':'partial_body_relative_configuration'}, {'feature':'elbow_lowers_relative_torso','hand':'right','relation':'partial_later_movement'}],
    ['right_elbow_lowers_hand_remains_raised'],
    'The text describes a lateral arm posture/action followed by lowering. It does not identify the initial wrist rise or face-to-torso wrist descent as that action. The stored elbow-lowering/hand-raised predicate is a directly observed partial relation, not a literal full paraphrase.',
    ['Track the same anatomical right arm, its lateral elbow relation, hand configuration and later elbow change.', 'Distinguish source-related arm action from initial approach and final rest; overlap alone cannot bind them.', 'The existing hand-remains-raised observation is not an additional universal requirement inferred from this sentence.'],
    ['A particular elbow angle or shoulder height', 'right_hand_descends_face_to_torso equals this clause', 'Initial R_up equals lateral arm raising', 'Whole lateral-then-lower action from one point'],
    'Existing S sequence evidence is reusable within its explicit frames. It does not resolve all borders or the whole dictionary variant.'),
 'well_palm_left_arm': spec(
    [{'feature':'hand_near_opposite_arm_projection','hand':'right','other_hand':'left','relation':'visible_spatial_component_of_named_body_relation'}],
    ['right_hand_at_left_arm_projection'],
    'The clause identifies the right palm and left arm surface. Visible right-hand proximity to the left arm is a partial observable relation. The wording does NOT logically entail R_open_four=1: using the palm does not specify every long finger is fully extended.',
    ['Require contemporaneous right-hand and left-arm observations with stable side identity.', 'Attach any separately observed open fingers only at their reviewed times, not as a mandatory condition supplied by the word palm.'],
    ['R_open_four is mandatory from this text alone', 'Palm normal/facing direction from proximity', 'Physical contact, pressure, or exact back-of-arm surface from 2D overlap', 'Substituting simple screen-down motion for the relation'],
    'S provides direct projection-relation targets and X already exposes this component. Separately recorded R_open_four remains visual evidence, not a source-derived requirement.'),
 'well_brush': spec(
    [{'feature':'hand_traverses_opposite_arm_toward_wrist','hand':'right','other_hand':'left','relation':'reference_bound_partial_path_component'}],
    ['right_hand_traverses_left_arm_toward_wrist'],
    'Read with the previous palm/left-arm phrase, the source describes brushing/downward travel along that arm. The direct FIT reference supports a proximal-to-wrist projected path; that path is not proof of brushing contact or an exact required endpoint.',
    ['Use at least an ordered path/interval on the same left arm, not an isolated endpoint.', 'Keep the right-hand-to-arm relation and arm-relative direction together; screen y alone is insufficient.', 'Separate the demonstrated travel from later hand lowering to rest.'],
    ['Exact touching surface or pressure', 'A mandatory stop or exact endpoint frame', 'Universal screen-down direction under changing body pose', 'Full stroke completion from any directed-motion tail'],
    'S interval/sequence targets and X motion bridge already exist. Their partial physical meaning must be retained when used beside the source text.'),
 'well_alternative_thumb': spec(
    [{'feature':'non_thumb_fingers_closed','hand':'right','relation':'conditional_on_separately_verified_entry_41'}, {'feature':'thumb_extended_upright_projected','hand':'right','relation':'conditional_on_separately_verified_entry_41'}],
    [],
    'The separate entry41 literally describes the right thumb extended from a closed hand, a fore/aft motion and two movements. Its lexical count can be stored as text metadata, but it is not a count requirement for the arm-brushing entry9543 or a labelled count for these recordings.',
    ['Verify that the recording depicts entry41 before using any of its components.', 'A same-gloss label or word ID cannot merge entry41 with entry9543.', 'A depth-sensitive fore/aft event and its count would require separate direct temporal evidence; none is created here.'],
    ['Applying this variant to the arm-brushing recordings', 'Required two repetitions for entry9543', 'Fore/aft depth from hand size alone', 'Present recordings satisfy this alternative'],
    'No matching current S typed target or P canonical source-card binding exists for this alternative; keep it as a separate unbound source variant.'),
 'ancestor_thumbs': spec(
    [{'feature':'non_thumb_fingers_closed','hand':'both','relation':'necessary_partial_shape_condition'}, {'feature':'thumb_extended_upright_projected','hand':'both','relation':'separate_thumb_evidence_required'}],
    ['both_visible_thumbs_up_with_closed_finger_masses'],
    'In the stored source terminology 5지 denotes the thumb, not five open fingers. The described compound requires evidence about both closed non-thumb finger masses and both extended/upright thumbs. Closed_four alone cannot verify the thumb component.',
    ['Require both hand components at the same directly inspected point or explicitly reviewed linked posture.', 'Do not borrow one thumb from another time or fill an occluded thumb from the target word.', 'A positive compound point is not a whole-stage shape label.'],
    ['5지 means five open fingers', 'Both thumbs from R/L_closed_four alone', 'Exact joint angles or hidden finger positions', 'Whole action correct from a compound point'],
    'Later S contains direct compound FIT thumb observations, including contrast points. These are genuine progress beyond the older P context-only card, but remain scoped physical observations.'),
 'ancestor_alternation': spec(
    [{'feature':'bilateral_alternating_relative_height','hand':'both','relation':'ordered_two_hand_sequence_component'}],
    ['bilateral_height_alternation'],
    'Alternation is an ordered two-hand event, not one R_up or a single R_higher/L_higher point. Relative-height exchange is a visible component; full upward action additionally needs the relevant hand trajectories, since one hand lowering can also change their relative order.',
    ['Track both anatomical hand identities through more than one relative ordering state.', 'Use explicit sequence witnesses; a point overlap or DTW correspondence is not sequence truth.', 'Preserve different normal speeds and repeated occurrences without inventing a mandatory numeric count.'],
    ['A required cycle count from 번갈아', 'R_higher or one interval proves bilateral alternation', 'Identity certainty during unresolved crossing', 'Generic reversed video is automatically semantic error'],
    'S has explicit FIT bilateral sequence witnesses. The existing P two height columns alone do not supply their temporal meaning.'),
 'egypt_tip_meeting': spec(
    [{'feature':'opposed_hand_upper_ends_near_projected','hand':'both','relation':'visible_projection_component_not_contact'}],
    ['bilateral_upper_ends_converged'],
    'The source mentions opposing fingertips. Directly visible convergence of the upper ends is a useful partial component; it does not establish which fingertips truly touch or their depth.',
    ['Require both hand upper ends in the same observed frame and the relevant raised configuration.', 'Bind only the inspected near posture, not every frame before/after it or a wrist-distance surrogate.'],
    ['Actual fingertip contact', 'All fingertip identities', 'Wrist proximity equals tip proximity', 'Near hands in any part of the clip satisfy the whole source action'],
    'S direct near-end labels and X runtime projection component already exist; preserve contact as unverified.'),
 'egypt_oblique_pose': spec(
    [{'feature':'paired_long_finger_axes_oblique_projected','hand':'both','relation':'partial_visible_orientation_configuration'}],
    ['bilateral_oblique_long_finger_axes'],
    '비스듬히 세웠다가 describes a slanted configuration followed by another action. It does not mean an upward wrist velocity. Paired visible long-finger axes can support the projected configuration without determining palm normals or exact 3D slant.',
    ['Use contemporaneous visible axes on both hands in the relevant posture.', 'If asserting a before/after relation, bind an actual subsequent action separately; a point pose cannot supply the whole sequence.'],
    ['Exact angle or 3D palm orientation', '세웠다가 equals R_up', 'A fully horizontal numerical angle as the only valid opposite', 'Every side-on hand is unobservable or negative'],
    'S already adds directly reviewed oblique FIT positives and counterpoints/unknowns beyond older Q draft links. Those scoped targets can be reused, but X does not have this component among its four validated bridge targets.'),
 'egypt_lower_shared': spec(
    [{'feature':'both_hands_lower_relative_body','hand':'both','relation':'source_conditional_sequence_not_currently_bound'}],
    ['source_variant_specific_lowering'],
    'Entry5990 says lower after the oblique pose; entry23905 also uses 내려 before a gathering suffix. The common lowering wording does not merge their suffixes. Hand separation is a different measurement from lowering.',
    ['Verify the matching source variant and inspect both hands over the intended lowering action.', 'Separate any earlier body-relative lowering from final return to rest.', 'Do not equate the existing near-to-apart process with downward motion or count both dictionary variants twice.'],
    ['hands_move_apart proves lowering', 'One R_down proves both hands lower', 'Final rest proves the described lowering', 'Cross-entry suffix requirements'],
    'S currently preserves nine FIT source-variant lowering observations as unknown. This audit keeps them unknown; a text implication cannot fill that evidence gap.'),
 'egypt_gather_suffix': spec(
    [{'feature':'within_each_hand_visible_fingertips_converge','hand':'both','relation':'variant_specific_component_needs_direct_evidence'}],
    ['each_hand_all_fingertips_gathered'],
    '각각 means fingertips converge within each hand, not that the two hands approach one another. This suffix belongs to entry23905 and is not present in entry5990. Non-extension of four fingers or a closed finger mass does not prove all fingertips are gathered.',
    ['Confirm entry23905 correspondence before treating the suffix as applicable.', 'Inspect the individual fingertips on each hand at the relevant later action; side-on ambiguity remains unknown.', 'Do not transfer a gathering label to an entire stage from an unrelated earlier shape.'],
    ['open_four=0 means fingertips gathered', 'closed_four means all tips gathered', 'Bilateral hand proximity means within-hand gathering', 'Missing suffix in entry5990 is a learner error'],
    'S preserves all nine FIT suffix observations as unknown. The earlier report unverified_edge_on status is retained; no actual gathering truth is created here.'),
}


def main():
    output = AE / 'SOURCE_FEATURE_CONSTRAINT_AUDIT.json'
    assert not output.exists(), 'Do not overwrite a completed audit.'
    paths = [AE/'SCOPE.json', P/'GUIDE_SUPPORT.json', P/'INPUT_MANIFEST.json', CARD, REPORT,
             Q/'CLAUSE_COMPONENT_LINKS.json', S/'DIRECT_TARGETS.json', X/'source_bridge.py',
             O/'word100_interval_support_contract_20261102/GUIDE_SCOPE_ADDENDUM.json']
    pins = [{'path':str(p), 'sha256':sha(p)} for p in paths]
    guide, cards, report = read(paths[1]), read(CARD), read(REPORT)
    fit = [(i,r) for i,r in enumerate(read(S/'DIRECT_TARGETS.json')['rows']) if r['role']=='fit']
    assert all(1 <= r['person'] <= 9 for _,r in fit)
    runtime = set()
    for node in ast.parse((X/'source_bridge.py').read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['POINTS','MOTIONS'] for t in node.targets):
            runtime.update(ast.literal_eval(node.value))
    assert len(runtime)==4
    assert len(guide['source_clauses'])==len(SPECS)==11
    rows=[]; checks=[]

    def check(name, ok):
        checks.append({'name':name,'pass':bool(ok)})
        assert ok,name

    for i,c in enumerate(guide['source_clauses']):
        cid=c['id']; src=c['source']; card=pointer(cards,src['json_pointer'])
        check(cid+': source file pin',sha(src['path'])==src['sha256'])
        check(cid+': exact entry/url/text',card['entry_id']==c['source_entry_id'] and card['url']==c['source_url'] and c['source_span'] in card['description'])
        bindings=[]
        for gi,g in enumerate(guide['guides']):
            for ci,b in enumerate(g['source_cards']):
                if b['clause_id']!=cid:continue
                old=pointer(report,b['saved_card_pointer']); case=report['cases'][int(b['saved_card_pointer'].split('/')[2])]
                check(cid+': original report clause',old['clause_id']==cid and old['text']==c['source_span'])
                check(cid+': FIT report only',case['role']=='fit' and case['reference_person']==1 and 1<=case['query_person']<=9)
                check(cid+': unchanged uncertainty',b['context_only'] is True and old['training_target'] is False and all(old[k] is False and b[k] is False for k in ['named_shape_verified','source_action_verified','full_span_semantic_verified']))
                anchors=b['canonical_reference_anchor_frames']
                check(cid+': anchor preservation',anchors==old['reference_anchor_frames'])
                bindings.append({
                    'guide_pointer':f'/guides/{gi}/source_cards/{ci}',
                    'description_report_pointer':b['saved_card_pointer'],
                    'canonical_stage_id':g['canonical_stage_id'],
                    'canonical_reference':'registered ID01, saved reference frame axis',
                    'context_bounds':g['frames'], 'context_description':g['description'],
                    'source_status':old['source_status'], 'anchor_frames':anchors,
                    'anchor_pts_us':[round(case['reference_pts'][f]*1e6) for f in anchors],
                    'query_anchor_links_are_DTW_context_not_truth':True,
                    'context_only':True,'named_shape_verified':False,'source_action_verified':False,
                    'full_span_semantic_verified':False,'training_target':False,
                })
        typed=[]
        for target in SPECS[cid]['existing_typed_component_ids']:
            evidence=[(j,r) for j,r in fit if r['target']==target]
            linked=[(j,r) for j,r in evidence if cid in r['source_clause_ids']]
            count=collections.Counter('unknown' if not r['mask'] else str(r['y']) for _,r in evidence)
            check(cid+': existing typed evidence present',bool(evidence))
            check(cid+': old physical not whole phrase',all(r['whole_clause_truth'] is None and r['learner_error'] is None for _,r in evidence))
            typed.append({
                'target':target,'artifact':{'path':str(S/'DIRECT_TARGETS.json'),'sha256':sha(S/'DIRECT_TARGETS.json')},
                'fit_all_same_definition_counts':dict(count),'fit_registered_ids':sorted({r['person'] for _,r in evidence}),
                'source_linked_row_pointers':[f'/rows/{j}' for j,_ in linked],
                'source_linked_records':len(linked),
                'all_same_definition_record_count_is_not_independent_sample_count':True,
                'existing_runtime_bridge_support':target in runtime,
                'new_image_review_in_this_audit':False,
                'use_policy':'Reuse immutable direct physical provenance, native point/interval/sequence scope and uncertainty. A linked row does not verify the entire phrase.',
            })
        specrow=dict(SPECS[cid]);extra=specrow.pop('additional_unsupported')
        rows.append({
            'clause_id':cid,'word_id_for_source_lookup_only':c['word_id'],
            'source_clause_pointer':f'/source_clauses/{i}','source':src,
            'url':c['source_url'],'entry_id':c['source_entry_id'],'text':c['source_span'],
            'exact_entry_description':card['description'],
            'saved_capture_content_sha256':sorted({x['source']['content_sha256'] for x in card['claims']}),
            'capture_hash_scope':'Recorded original capture identity from immutable source card; this audit verifies the local card hash and exact text/pointer, and does not download or re-fetch the site.',
            'source_card_status':card['status'],
            **specrow,
            'unsupported':list(dict.fromkeys(c['unsupported_components']+extra)),
            'earlier_context_bindings_preserved':bindings,
            'later_direct_physical_evidence_reused':typed,
            'source_logical_condition_is_video_observation':False,
            'source_text_supplies_framewise_truth':False,
            'current_training_label_from_this_audit':None,
            'expected_feature_value_from_old_catalog_preserved':c['expected_feature_value'],
            'requiredness':c['requiredness'],'tolerance':c['tolerance'],
            'whole_clause_verified':False,'whole_sign_correctness':None,
        })
    vocab={}
    for r in rows:
        for f in r['supported_component_if_any']:
            vocab.setdefault(f['feature'],{'parameterization':'hand identity and other-hand/body part are parameters, not a word-specific runtime exception','mentioned_by_source_clauses':[]})['mentioned_by_source_clauses'].append(r['clause_id'])
    for p in pins:check('unchanged input '+p['path'],sha(p['path'])==p['sha256'])
    out={
        'status':'SOURCE_COMPONENT_CONSTRAINT_AUDIT_COMPLETE_NO_NEW_SUPERVISION',
        'at':datetime.now(timezone.utc).isoformat(),'rows':rows,
        'common_feature_vocabulary':vocab,
        'already_implemented_runtime_components':sorted(runtime),
        'existing_progress_acknowledged': 'The older P context cards are not the entire current system. Later Q/S direct physical bindings and X source_bridge already support four runtime components. This audit neither removes those scoped capabilities nor upgrades them to full source-clause correctness.',
        'source_to_observation_contract':[
            'Text can name an expected partial property conditional on the verified expression variant; it does not prove that a frame exhibits it.',
            'Use an independent existing direct physical point/interval/sequence label at its native source time. A center-frame label cannot be propagated over a stage.',
            'DTW and canonical anchor extents propose correspondence. They are neither fresh image observations nor exact action boundaries.',
            'Necessary partial component support is not sufficient for the full phrase. A missing or hidden component is not a contrary observation.',
            'An actually visible contrary component can support a component difference only when source variant and same-stage timing are supported. It is not automatically a whole-sign educational error.',
            'Do not merge same-gloss dictionary variants; preserve literal counts as variant text metadata without creating mandatory repetition labels.',
            'Use common handshape/relation/movement predicates with hand/body parameters. Word IDs select source descriptions; they never supply observed truth.',
        ],
        'summary_ko':'11개 출처 문구와 원문·시간 연결을 감사했다. 긴 네 손가락 닫힘, 반대팔 부근 관계, 팔을 따른 이동, 양손 끝의 근접 등 기존 직접 관측 연결을 재사용할 수 있다. 다만 엄지·접촉·3D 방향·필수 반복 수·표현 변이까지 확인한 것은 아니다. 손바닥이라는 문장만으로 네 손가락 완전 펼침을 강제하지 않는다. 기존 context_only 기록과 이후 직접 물리 정답의 진전을 함께 보존했다.',
        'pins':pins,'checks':checks,'check_count':len(checks),
        'source_clauses':11,'new_physical_labels':0,'new_images_reviewed':0,
        'FIT_evidence_only':True,'DEV_images_or_model_results_accessed':False,
        'new_inference':0,'new_training':0,'downloads':0,
        'previous_artifacts_modified':0,'independent_KSL_expert_validation':False,
        'whole_goal_complete':False,'performance_gain_established':False,
    }
    output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':out['status'],'clauses':len(rows),'features':len(vocab),'checks':len(checks),'sha256':sha(output),'path':str(output)},ensure_ascii=True))


if __name__=='__main__':main()
