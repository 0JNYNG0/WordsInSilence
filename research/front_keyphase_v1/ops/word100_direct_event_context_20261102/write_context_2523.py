"""Write one-pass, direct event-context review; no prior feature labels read."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json

AF=Path(__file__).resolve().parent
AE=AF.parent/'word100_guide_feature_supervision_20261102'
QUEUE=AE/'CONTEXT_REVIEW_QUEUE.json'
PLAN=AE/'NEXT_TIME_BINDING_REVIEW.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


DISPLAYED={
1:[0,25,27,29,31,33,34,35,36,38,40,42,44,46,48,49,50,52,54,56,58,59,61,63,65,67,69,84],
4:[0,68,70,73,75,78,80,82,83,85,87,90,92,94,95,97,99,101,102,104,106,107,108,109,111,113,116,118,121,123,143],
7:[0,30,32,34,36,37,39,41,42,43,44,45,47,49,51,52,53,54,55,56,58,60,61,62,64,66,67,69,71,73,88],
}
BODY={
1:{'approach':[25,27,29,31],'present':[33,34,35,36,38],'change':[40,42],'later':[44,46,48,49,50,52,54,56],'return':[58,59,61,63,65,67,69], 'uncertain_bracket':[38,44]},
4:{'approach':[68,70,73,75],'present':[78,80,82,83,85,87],'change':[90,92,94],'later':[95,97,99,101,102,104,106,107,108,109],'return':[111,113,116,118,121,123], 'uncertain_bracket':[87,95]},
7:{'approach':[30,32,34,36,37],'present':[39,41,42,43,44,45],'change':[47,49,51],'later':[52,53,54,55,56,58,60,61,62],'return':[64,66,67,69,71,73], 'uncertain_bracket':[45,52]},
}


def make_reason(cid, value, b):
    if value==1:
        return (
          '앞서 양팔이 몸 앞쪽으로 함께 올라온 뒤, 두 팔꿈치를 몸통 양옆에 둔 채 전완을 얼굴 아래 앞쪽으로 향하게 한 제시 맥락이다. 뒤의 전완 방향 전환 및 어깨 앞 후속 자세와 구별된다. 이 판정은 손끝이 실제 맞닿았는지, 손가락이 맞는 모양인지의 판정이 아니다.',
          {'preceding_bilateral_arm_approach':b['approach'], 'current_body_context':'both elbows beside torso, forearms presented in front of lower face/upper torso', 'following_forearm_reorientation':b['change'],'later_posture':b['later'], 'independence_limit':'Arm trajectories are additional visible evidence, but these samples do not supply a wrong-tip-shape example within the same initial event.'})
    if value==0:
        return (
          '초기 양팔 제시 뒤에 전완 방향이 바뀌고 각 손목이 어깨 앞쪽에 놓이는 후속 자세이다. 앞의 전환과 뒤의 유지·팔 내림 흐름으로 다른 맥락임을 구분했다. 손끝 사이 거리 자체를 적용 여부의 단독 기준으로 삼지 않았고, 정상적인 다음 동작을 수어 오류로 부르지 않는다.',
          {'preceding_forearm_reorientation':b['change'], 'current_body_context':'later forearm configuration with wrists in front of the corresponding shoulder areas', 'following_later_posture':b['later'],'following_arm_return':b['return'], 'independence_limit':'This separates the visible episodes; it does not establish a sharp boundary or guarantee a future context model will avoid proximity shortcuts.'})
    if cid=='egypt_lower_shared':
        return (
          '양팔 제시에서 후속 자세로 바뀌는 흐름은 보이지만, 이 변화의 어느 부분이 사전의 내린다에 해당하는지 손·팔의 하강값 자체와 독립적으로 확정할 수 없다. 양손이 벌어진다는 사실을 내림의 증거로 사용하지 않으며, 마지막 팔 내림과도 구분해야 하므로 적용 맥락은 미확인이다.',
          {'observed_broad_episode':'bilateral presentation followed by forearm reorientation and a later posture', 'preceding_presentation':b['present'],'surrounding_transition':b['change'],'later_posture':b['later'], 'unresolved':'Whether this exact sampled point belongs to the source-specific lowering event rather than its antecedent or another normal change.'})
    return (
      '후속 어깨 앞 자세라는 점은 팔의 배치와 앞뒤 흐름으로 알 수 있다. 하지만 이 순간이 23905의 각 손끝 모음에 해당하는지는 손 모양 그 자체 또는 사전 변이 선택에 의존한다. 5990에는 같은 모음 문구가 없으므로 시간상 뒤에 있다는 이유만으로 적용 맥락을 확정하지 않았다.',
      {'observed_broad_episode':'later shoulder-front posture after bilateral forearm reorientation', 'preceding_forearm_reorientation':b['change'],'later_posture':b['later'],'following_arm_return':b['return'], 'unresolved':'Entry23905-specific gathering event versus another normal later posture; the source variant remains unresolved.'})


def main():
    output=AF/'CONTEXT_2523.json'
    assert not output.exists(), 'Do not overwrite completed one-pass annotations.'
    q=json.loads(QUEUE.read_text(encoding='utf-8'))
    source_meta={r['clip_id']:r['source_sha256'] for r in json.loads(PLAN.read_text(encoding='utf-8'))['pilot_packets'] if r['word_id']=='WORD2523'}
    selected=[r for r in q['rows'] if r['word_id']=='WORD2523' and r['registered_id'] in DISPLAYED]
    assert len(selected)==3
    results=[];checks=[]
    for packet in selected:
        person=packet['registered_id'];frames=DISPLAYED[person];b=BODY[person]
        assert sha(packet['sheet']['path'])==packet['sheet']['sha256']
        claims=[]
        for c in packet['claims']:
            cid=c['clause_id'];points=[]
            for f,ts in zip(c['witness_frames'],c['witness_timestamps_us']):
                assert f in frames
                # These values are this actual image review, not prior support/y.
                if cid in ['egypt_tip_meeting','egypt_oblique_pose'] and f in b['present']:
                    context=1
                elif cid=='egypt_tip_meeting' and f in b['later']:
                    context=0
                else:
                    context=None
                reason,evidence=make_reason(cid,context,b)
                points.append({
                    'frame':f,'timestamp_us':ts,'context':context,
                    'reason_ko':reason,'independent_body_or_trajectory_evidence':evidence,
                    'preceding_frames':[x for x in frames if x<f][-2:],
                    'following_frames':[x for x in frames if x>f][:2],
                    'tested_feature_used_to_assign_context':False,
                    'if_tested_feature_is_required_for_resolution':'leave context null, rather than read a shape/proximity value as applicability',
                    'sampled_context_only':True,
                })
            entry='NIKL_legacy:5990' if cid=='egypt_lower_shared' else 'NIKL_legacy:23905'
            definition=(
                'The initial paired raised-arm presentation after the bilateral approach and before the later forearm reorientation; local shared-prefix event context only. This does not require correct fingertip proximity or oblique finger shape.'
                if cid in ['egypt_tip_meeting','egypt_oblique_pose'] else
                'The source-specific lowering event, separate from later rest return; current broad visible transition does not independently resolve its applicability.'
                if cid=='egypt_lower_shared' else
                'The entry23905-specific within-each-hand fingertip-gathering event; a later position in the action alone does not establish applicability.')
            claims.append({
                'claim_id':c['claim_id'],'clause_id':cid,'source_text':c['source_text'],
                'source':c['source'],'source_entry_id':entry,'query_kind':c['query_kind'],
                'original_witness_frames':c['witness_frames'],'original_witness_timestamps_us':c['witness_timestamps_us'],
                'local_context_definition':definition,
                'source_variant_status':{
                    'whole_recording_variant':'UNRESOLVED_23905_VS_5990',
                    'shared_prefix_context':'locally compatible paired raised-arm event; no full source-form confirmation',
                    'suffix_policy':'23905 gathering suffix is not imposed on5990; hand separation is not lowering',
                },
                'context_points':points,
                'preceding_context_witnesses':b['approach'] if cid in ['egypt_tip_meeting','egypt_oblique_pose'] else b['present'],
                'following_context_witnesses':b['change']+b['later']+b['return'],
                'uncertain_boundary_frames_or_brackets':[b['uncertain_bracket']],
                'tested_feature_used_to_assign_context':False,
                'unresolved_dimensions':['whole source variant','exact event boundary','unsampled intervening frames','actual contact/3D','required repetitions'],
                'whole_sign_correctness':None,'educational_error':None,'required_repetition_count':None,'exact_event_boundary_truth':None,
                'dense_label_propagation':False,'copied_physical_label':False,
            })
        results.append({
            'clip_id':packet['clip_id'],'word_id':packet['word_id'],'person':person,'role':'fit',
            'reviewer':'Codex egypt_source_binding_review; same-team direct image review, not independent human/KSL expert',
            'reviewed_sheet':packet['sheet'],'actual_displayed_frames':frames,
            'sheet_open_count_this_pass':1,'zoom_views':[],
            'source_sha256':source_meta[packet['clip_id']],
            'original_video_hash_status':'source identity copied from authorized NEXT_TIME_BINDING_REVIEW pilot_packets metadata; old-support metadata and original video were not reopened for this assignment',
            'claims':claims,
            'shortcut_challenge':{
                'question':'Could the same local event still be recognized if fingertip proximity or handshape differed?',
                'independent_cues_observed':'Bilateral arm approach, elbow relation to torso, forearm presentation and subsequent reorientation provide evidence beyond finger configuration or tip distance.',
                'actual_target_contrary_example_within_same_context_observed':False,
                'same_tested_value_across_contexts_established':False,
                'hidden_target_feature_challenge_executed':False,
                'conclusion':'Local event annotations have separate body/trajectory reasons, but these sheets alone do not establish empirical shortcut robustness. Do not call a future learned context observer independently validated.',
            },
            'whole_sign_correctness':None,'educational_error':None,'required_repetition_count':None,'exact_event_boundary_truth':None,
        })
    counts=collections.Counter('unknown' if p['context'] is None else str(p['context']) for r in results for c in r['claims'] for p in c['context_points'])
    assert counts=={'1':34,'0':9,'unknown':17}
    result={
        'status':'ONE_PASS_DIRECT_CONTEXT_2523_COMPLETE_NOT_LEARNING_RELEASE',
        'at':datetime.now(timezone.utc).isoformat(),'rows':results,
        'queue':{'path':str(QUEUE),'sha256':sha(QUEUE)},
        'source_identity_metadata':{'path':str(PLAN),'sha256':sha(PLAN),'pointer':'/pilot_packets'},
        'scope':{'path':str(AF/'SCOPE.json'),'sha256':sha(AF/'SCOPE.json')},
        'summary':{'clips':3,'claims':15,'context_queries':60,'unique_clip_frame_queries':34,'actual_sheet_opens':3,'actual_displayed_image_slots':90,'extra_zooms':0,'context_counts':dict(counts)},
        'target_readiness':{
            'egypt_tip_meeting':'Each of three records has local and later-context witnesses with arm/trajectory reasons; empirical shortcut challenge is still unproven, so this is not an automatic training pass.',
            'egypt_oblique_pose':'Only local-context witness times were supplied for this clause; no new off-context times invented.',
            'egypt_lower_shared':'Independent source-specific lowering applicability unresolved; preserve null.',
            'egypt_gather_suffix':'Source variant and suffix applicability unresolved independently of the tested shape; preserve null.',
        },
        'historical_project_knowledge_present':True,'independent_blinded_human_review':False,
        'prior_support_y_mask_predictions_or_DEV_read_for_this_assignment':False,
        'old_labels_modified':0,'new_extraction':0,'new_model_inference':0,'new_training':0,
        'whole_goal_complete':False,
        'limitations':['Chronological sampled sheets do not establish all intermediate frames or exact boundaries.', 'Direct context1 does not prove compared physical feature1.', 'Context0 is another normal action context, not wrong signing.', 'The explicit shortcut limitation must not be hidden by reporting only the1/0 counts.'],
    }
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(output),'sha256':sha(output),'summary':result['summary']},ensure_ascii=True))


if __name__=='__main__':main()
