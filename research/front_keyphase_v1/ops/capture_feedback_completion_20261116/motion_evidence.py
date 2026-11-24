"""Preserve observed progress separately from confirmed endpoint evidence.

This is a common relation-motion component, not a KSL word completion judge.
It deliberately retains the frozen arrival detector; it does not turn a partial
movement into an arrival or infer error from uncertainty. No word/person input.
"""
import numpy as np

def movement_evidence(static, motion, arrivals, available, radius=3):
    static=np.asarray(static);motion=np.asarray(motion);available=np.asarray(available,dtype=bool)
    n=len(static)
    if motion.shape!=(n,) or available.shape!=(n,):
        raise ValueError('State/motion/observation lengths must agree')
    confirmed=[]
    for event in arrivals:
        a,b=int(event['start']),int(event['end'])
        if not 0<=a<=b<n:
            raise ValueError('Arrival outside the actual recording')
        if static[b]!=2 or not available[a:b+1].all():
            continue
        confirmed.append(dict(event,endpoint_status='observed_near',physical_contact_verified=False))
    candidates=[];i=0
    while i<n:
        # The learned label belongs to the entire real radius-3 window.
        a,b=i-radius,i+radius
        valid=a>=0 and b<n and available[a:b+1].all()
        if motion[i]!=1 or not valid:
            i+=1;continue
        first=i;i+=1
        while i<n and motion[i]==1 and i+radius<n and available[i-radius:i+radius+1].all():
            i+=1
        if i-first<2:
            continue
        a,b=first-radius,i-1+radius
        # Require some observed in-action relation, not a background-only run.
        if not np.isin(static[a:b+1],[1,2]).any():
            continue
        linked=[j for j,e in enumerate(confirmed) if max(a,e['start'])<=min(b,e['end'])]
        candidates.append(dict(source_span=[int(a),int(b)],prediction_centres=[int(first),int(i-1)],
            movement='hands_approaching',status='progress_with_arrival' if linked else 'progress_endpoint_unconfirmed',
            arrival_indices=linked,description_bound=False))
    return dict(confirmed_arrivals=confirmed,movement_candidates=candidates,
        confirmed_arrival_count=len(confirmed),
        unconfirmed_progress_count=sum(not c['arrival_indices'] for c in candidates),
        whole_word_correct=None,lesson_complete=False,
        scope='Projected relative hand approach only; shape, side, orientation, description binding and order need separate evidence.')

def repetition_feedback(evidence,required_count):
    """An observation summary; even sufficient approaches do not pass a lesson."""
    if type(required_count) is not int or required_count<1:
        raise ValueError('A reviewed positive repetition requirement is required')
    count=evidence['confirmed_arrival_count'];partial=evidence['unconfirmed_progress_count']
    if count>=required_count:
        return dict(status='observed_component',text=f'손이 가까워지는 동작 {count}회를 확인했어요. 손모양과 방향도 확인해야 해요.',
                    lesson_complete=False)
    if partial:
        return dict(status='uncertain_endpoint',text='손이 가까워지는 움직임은 보였어요. 끝까지 닿는 부분은 확인이 어려워요.',
                    lesson_complete=False)
    if count:
        return dict(status='partial_observation',text=f'가까워지는 동작은 {count}회 확인했어요. 나머지 동작은 확인이 필요해요.',
                    lesson_complete=False)
    return dict(status='insufficient_observation',text='이 동작을 확인할 근거가 부족해요.',lesson_complete=False)
