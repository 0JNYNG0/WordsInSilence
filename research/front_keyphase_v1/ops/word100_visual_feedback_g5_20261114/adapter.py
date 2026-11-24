"""Display-only projection of qualified G4 evidence. No model/threshold changes."""
from copy import deepcopy

AGREE = {'measured_features_agree', 'measured_features_and_goal_agree'}
SHAPE_HAND = {0:'R',1:'R',2:'L',3:'R',4:'L'}

def spans_in(value):
    if isinstance(value, dict):
        return spans_in(value['span']) if 'span' in value else []
    if isinstance(value, list):
        if len(value)==2 and all(isinstance(v,int) for v in value): return [value]
        return [s for item in value for s in spans_in(item)]
    return []

def project(row, guides, frames):
    """Keep exact provenance. Unknown cards never inherit raw orange evidence."""
    out=[]
    candidates={c['card']:c for c in row['original_candidates']}
    for source in row['joint_cards']:
        spec=next(g for g in guides if g['id']==source['card'])
        card=source['card']; state=source['state']; candidate=candidates.get(card,{})
        role=candidate.get('role_confirmation',{}).get('matching',[])
        base=spans_in(candidate.get('chosen') or {}) or spans_in(role)
        observations=[]
        qualified=state=='attributable_feature_difference'
        # Each visual cue has a local evidence path and only its actual span/hands.
        if qualified:
            for i,e in enumerate(source['differences']):
                if e.get('task')=='direction' and e.get('state')=='opposite_direction':
                    observations.append(dict(kind='direction',message='움직인 방향이 참고와 반대로 관측됐어요.',action=spec['instruction'],spans=spans_in(e.get('span')),hands=spec['hands'],evidence=f'differences/{i}'))
                elif e.get('task')=='shape' and e.get('state')=='difference' and e.get('feature') in SHAPE_HAND:
                    hand=SHAPE_HAND[e['feature']]; name='오른손' if hand=='R' else '왼손'
                    expected=e.get('expected',1); feature=e['feature']
                    action=(f'{name}의 네 손가락을 펴 보세요.' if (feature in (1,2) and expected==1) or (feature in (3,4) and expected==0) else f'{name}의 네 손가락을 접어 보세요.') if feature else f'{name}의 검지 모양을 참고와 비교해 보세요.'
                    observations.append(dict(kind='shape',message=f'{name}의 손모양에 차이가 관측됐어요.',action=action,spans=spans_in(e.get('span')),hands=[hand],evidence=f'differences/{i}'))
            for i,e in enumerate(source['order_evidence']):
                if e['state']=='reverse':
                    pair=[next(g for g in guides if g['id']==cid) for cid in e['cards']]
                    observations.append(dict(kind='order',message='표시한 두 행동의 순서가 반대로 관측됐어요.',action='먼저 '+pair[0]['instruction']+' 그다음 '+pair[1]['instruction'],spans=spans_in(e['witness_spans']),card_ids=e['cards'],hands=sorted(set(h for g in pair for h in g['hands'])),evidence=f'order_evidence/{i}'))
        status='difference' if observations else 'similar' if state in AGREE else 'uncertain'
        reason={
            'evidence_conflict':'같은 구간의 판단이 엇갈렸어요. 맞거나 틀렸다고 확정하지 않을게요.',
            'unimplemented_sequence_requirement':'손의 움직임은 보이지만, 반복 전체가 맞는지는 아직 판단하지 못해요.',
            'component_link_uncertain':'보이는 특징을 이 설명의 행동으로 확실하게 연결하지 못했어요.',
            'action_link_uncertain':'이 설명에 해당하는 행동 구간을 확실하게 찾지 못했어요.',
            'observed_features_only':'일부 움직임은 보이지만, 필요한 특징을 모두 확인하지 못했어요.',
            'observation_insufficient':'손·팔 관측이 부족해요. 오답으로 판단하지 않을게요.',
            'goal_pose_without_observed_start':'끝 자세는 보이지만, 그곳까지 움직인 과정은 확인하지 못했어요.'
        }.get(state,'이 부분은 근거가 부족해 교정 여부를 보류했어요.')
        if status=='similar':
            message=next((d['text'] for d in source.get('drafts',[]) if d['kind']=='observed_agreement'),'확인 가능한 특징은 설명과 비슷해요.')
        else: message=observations[0]['message'] if observations else reason
        spans=[]
        for s in base:
            if len(s)!=2 or not (0<=s[0]<=s[1]<frames): raise ValueError((card,'bad base span',s,frames))
            if s not in spans:spans.append(s)
        for cue in observations:
            cue['spans']=sorted(cue['spans'])
            if not cue['spans']:raise ValueError((card,'difference without time'))
            for a,b in cue['spans']:
                if not 0<=a<=b<frames:raise ValueError((card,'cue span outside recording'))
        out.append(dict(id=card,status=status,raw_state=state,instruction=spec['instruction'],hands=spec['hands'],spans=spans,cues=observations,message=message,whole_sign_correct=None,completion=None,source_card=deepcopy(source)))
    return out
