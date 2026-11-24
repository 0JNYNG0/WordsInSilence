"""Record one completed visual pass; no old label/prediction or model reads."""
from pathlib import Path
import json,hashlib
from collections import Counter

U=Path(__file__).resolve().parent
Q=U.parent/'word100_guide_feature_supervision_20261102/CONTEXT_REVIEW_QUEUE.json'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Read from the three displayed sheets, not imported from prior annotations.
DISPLAYED={
1:[0,40,42,45,47,49,51,54,56,58,60,63,64,65,67,68,69,72,73,74,76,77,78,81,83,85,87,90,92,128],
4:[0,65,67,69,71,74,76,78,80,81,82,84,86,88,91,93,95,97,98,99,101,102,103,105,108,110,112,114,143],
7:[0,35,37,39,41,44,46,47,48,49,50,52,54,56,58,60,61,62,63,65,67,69,71,72,73,75,78,80,82,84,116],
}
BODY={
1:'f40에는 팔이 아래에 있고 f42~49에서 왼팔을 몸 앞에 올려 제시한다. f51~81에서 왼팔의 받침 자세와 몸통 방향이 대체로 유지된다. f83~90에는 팔을 내리고 f92에는 내려간 자세이다.',
4:'f65~71에서 왼팔이 몸 앞쪽으로 올라오고, f74~103에서 제시한 왼팔과 몸통의 관계가 대체로 유지된다. f105~114에는 팔들이 내려오고 f143에는 아래의 휴식 자세이다.',
7:'f35~44에서 왼팔을 앞으로 들어 제시한다. f46~73에서 왼팔을 들어 둔 자세와 몸통 방향이 대체로 유지된다. f75~84에서는 왼팔과 팔꿈치가 아래로 내려가며 f116에는 휴식 자세이다.',
}
RELATION_REASON={
1:{54:'왼팔을 이미 몸 앞에 들어 제시한 상태이며 전후 f49·51과 f56·58에서도 그 받침 자세가 이어진다.',
   64:'앞서 제시한 왼팔이 몸 앞에 유지되고 이후 f65·67에서도 같은 받침 자세가 계속된다.',
   73:'왼팔을 제시한 자세가 f69·72와 f74·76 사이에도 이어진다. 아직 f83~90의 팔 내림 맥락으로 바뀌었다고 볼 근거는 없다.'},
4:{80:'f74·76·78에 이어 왼팔을 앞쪽에 제시한 몸-팔 자세가 유지된다. 뒤 f81·82에서도 이 받침 자세가 이어진다.',
   93:'f86·88·91부터 들어 둔 왼팔의 자세가 유지되며 f95·97에서도 같은 몸통-왼팔 맥락이다.',
   102:'f98·99·101과 f103에서 왼팔의 제시 자세가 유지된다. f108 이후의 팔 내림과 구분되는 넓은 제시 맥락이다.'},
7:{49:'f44 이후 제시한 왼팔과 몸통의 관계가 f46·47·48 및 f50·52에서 이어진다.',
   60:'왼팔을 들어 둔 자세가 f54·56·58 및 f61·62에 계속 보인다.',
   69:'f63·65·67에 이어 왼팔의 받침 자세가 f71·72·73까지 유지된다. 이후 f75~84의 팔 내림과는 구분된다.'},
}

def main():
    out=U/'CONTEXT_1133.json';assert not out.exists()
    queue=read(Q);rr=[r for r in queue['rows'] if r['word_id']=='WORD1133']
    assert {r['registered_id'] for r in rr}=={1,4,7}
    rows=[];labels=Counter();queries=0
    for r in rr:
        pid=r['registered_id'];frames=DISPLAYED[pid]
        assert sha(r['sheet']['path'])==r['sheet']['sha256']
        claims=[]
        for claim in r['claims']:
            points=[]
            for f,t in zip(claim['witness_frames'],claim['witness_timestamps_us']):
                assert f in frames, 'All requested witnesses were actually displayed'
                before=[i for i in frames if i<f][-3:];after=[i for i in frames if i>f][:3]
                if claim['clause_id']=='well_palm_left_arm':
                    value=1
                    reason=RELATION_REASON[pid][f]+' 이 판단은 왼팔의 제시와 전후 팔 올림·내림에 근거한다. 오른손이 왼팔에 얼마나 가까운지, 손가락 모양이 맞는지, 실제로 접촉하는지는 맥락 라벨의 조건으로 쓰지 않았다.'
                    definition='왼팔을 몸 앞에 제시해 상대 손과의 상호작용을 검토할 수 있는 넓은 받침-팔 제시 맥락. 실제 오른손-왼팔 근접·손바닥면·접촉·동작 정답을 포함하지 않음.'
                    independent={'available':True,'evidence_ko':BODY[pid],
                         'cue':'receiving_left_arm_presentation_relative_torso_and_its_raise_lower_context',
                         'counterfactual':'오른손을 왼팔에서 떨어뜨리거나 손모양을 바꾸어도, 같은 왼팔 제시 자세라면 이 넓은 context1은 유지되어야 한다.'}
                else:
                    value=None
                    reason='이 시점 주변에 왼팔을 제시한 넓은 맥락은 보인다. 그러나 그것만으로 출처의 실제 쓸기 진행 시점인지 이후 자세를 유지하는 시점인지 구분할 수 없다. 오른손의 팔을 따른 이동 또는 이동 없음으로 해당 구간을 선택하면 바로 검사할 움직임으로 맥락을 정하는 순환이 된다. 따라서 특정 쓸기 구간 적용성은 null로 둔다.'
                    definition='출처의 쓸기 동작을 비교할 국소 시점/과정 맥락. 단순히 왼팔을 제시한 넓은 구간이나 오른손 이동 자체와 동일시하지 않음.'
                    independent={'available':False,'available_broad_context_ko':BODY[pid],
                         'missing':'왼팔 제시가 유지되는 동안 쓸기 진행과 이후 유지 구간을 구분할 독립된 몸/다른 팔의 사건 신호가 없다.',
                         'shortcut_challenge':'같은 왼팔 제시 자세에서 오른손 이동을 생략하거나 반대로 움직인 경우에도 맥락을 배제하면 안 된다. 이동 유무를 보고 context1/0을 정하지 않았다.'}
                points.append(dict(frame=f,timestamp_us=t,event_context=value,context=value,
                    reason_ko=reason,independent_body_or_trajectory_evidence=independent,
                    preceding_frames=before,following_frames=after,tested_feature_used_to_assign_context=False,
                    source_action_match=None,whole_sign_correctness=None,educational_error=None,
                    exact_event_boundary_truth=None))
                labels['unknown' if value is None else str(value)]+=1;queries+=1
            claims.append(dict(claim_id=claim['claim_id'],clause_id=claim['clause_id'],source_text=claim['source_text'],
                  source=claim['source'],query_kind=claim['query_kind'],context_points=points,
                  local_context_definition=definition,source_variant_status='Source card candidate0 arm-related description is the comparison hypothesis; dictionary-form identity and full semantic agreement are not newly labelled.',
                  uncertain_boundaries=['준비에서 제시로 넘어가는 정확한 첫 프레임','쓸기 종료와 이후 유지 사이의 정확한 경계','제시 자세에서 마무리 팔 내림으로 넘어가는 정확한 첫 프레임'],
                  unresolved_dimensions=['손바닥의 정확한 면/방향','실제 접촉 여부','쓸기 구간 전체 수행','필수 거리·속도·횟수','교육적 정오'],
                  whole_sign_correctness=None,educational_error=None,exact_event_boundary_truth=None))
        rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=pid,role='fit',
             reviewer='Codex predicate_contract_review; prior project context known, not blinded expert',
             sheet=r['sheet'],sheet_display_count=1,actual_displayed_frames=frames,zoom_views=[],
             claims=claims,local_context_definition='넓은 왼팔 제시 맥락은 확인; 그 내부의 출처 쓸기 진행 구간 적용성은 독립 근거 부족으로 보류.',
             source_variant_status='candidate0 only; no transfer to an alternative same-name expression',
             uncertain_boundaries='Sparse sheet; no exact dense event boundaries inferred.',
             unresolved_dimensions=['Broad receiving-arm context is not a labelled stroke interval.','No queued other-context0 examples; visible preparation/rest cells were not added as new queries.'],
             timestamp_note=('REAL04 sheet and zoom seconds are display estimates different from queued native microseconds. Labels bind displayed frame IDs and preserve witness_timestamps_us exactly; no timestamp correction.' if pid==4 else 'Queued native timestamps preserved; only listed displayed frame IDs reviewed.'),
             whole_sign_correctness=None,educational_error=None,exact_event_boundary_truth=None))
    report=dict(status='DIRECT_VISUAL_PASS_COMPLETE_LIMITED_CONTEXT_SUPPORT',rows=rows,
        scope_sha256=sha(U/'SCOPE.json'),queue_sha256=sha(Q),review_code_sha256=sha(Path(__file__)),
        actually_opened_sheets=3,actually_opened_zooms=0,displayed_frame_cells=sum(map(len,DISPLAYED.values())),
        claims=9,context_queries=queries,context_counts=dict(labels),
        direct_unique_query_frames=sum(len(set(f for c in r['claims'] for f in c['witness_frames'])) for r in rr),
        readiness='NOT_READY_FOR_CONTEXT_DISCRIMINATION_FROM_THIS_QUEUE: relation context has9 applicable witnesses but no queued other-context0; motion phase has22 unresolved queries. Do not turn post-action holds into negatives using absence of the tested motion.',
        same_codex_not_independent_human_expert=True,old_support_or_predictions_read_for_assignment=False,
        dense_label_propagation=False,new_frame_extraction=0,new_model_inference=0,new_training=0,
        whole_goal_complete=False)
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ['status','actually_opened_sheets','displayed_frame_cells','context_queries','context_counts','direct_unique_query_frames']},ensure_ascii=False))

if __name__=='__main__':main()
