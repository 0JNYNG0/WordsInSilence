"""Pure additive readout of already saved local-event probabilities.

No files, models, DTW, word-specific thresholds or score-selected scopes here.
Source descriptors and proposals are frozen independently by replay.py.
"""
import copy
import hashlib
import json
import re
import numpy as np

VERSION = 'local-event-guide-additive-20261103-v1'
HIGH = .8
LOW = .2
JOINT_SUPPORT = .5
NAMES = [
    '오른손이 왼팔을 따라 왼손목 쪽으로 이동하는 흐름',
    '두 손이 서로 벌어지는 움직임',
    '오른손이 왼손에 비해 올라가는 움직임',
    '오른손이 얼굴 부근에서 몸통 쪽으로 내려오는 움직임',
    '오른손이 왼손에 비해 내려가는 움직임',
]

def value_sha(value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def integer_axis(values,name):
    a=np.asarray(values)
    if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iu' or (a<0).any() or (np.diff(a.astype('int64'))<=0).any():
        raise ValueError('Invalid strictly increasing native axis: '+name)
    return a

def stage_guard(stage):
    names=['neural_stage_agreement','image_time_coverage','unique_query_frames']
    values={k:stage.get(k) for k in names}
    known=all(isinstance(v,(float,int)) and not isinstance(v,bool) and np.isfinite(v) for v in values.values())
    supported=bool(known and values[names[0]]>=.5 and values[names[1]]>=.8 and values[names[2]]>=2)
    return dict(guide_id=stage['guide_id'],guard_scope_native_frames=copy.deepcopy(stage.get('frame_ids',[])),
                measurements=values,thresholds=dict(stage_agreement=.5,image_time_coverage=.8,unique_query_frames=2),
                status='SUPPORTED' if supported else 'UNCONFIRMED',
                scope='Existing guide-stage consistency only; not direct verification of all source-proposal times.',
                source_applicability_verified=None)

def validate_identity(base,envelope,prepared,predictions,proposals):
    if 'local_event_feedback' in base:
        raise ValueError('Refuse duplicate or replacement attachment')
    for key in ['source_sha256','input_sha256','AH_prediction_sha256']:
        if not re.fullmatch('[0-9a-f]{64}',envelope.get(key,'')):
            raise ValueError('Missing verified identity: '+key)
    input_identity=prepared.get('input_identity',{})
    for key in ['clip_id','word_id','source_sha256','input_sha256']:
        if input_identity.get(key)!=envelope.get(key):
            raise ValueError('Prepared cache identity differs: '+key)
    if base.get('word_id')!=envelope['word_id'] or value_sha(base)!=envelope['base_value_sha256']:
        raise ValueError('Existing analysis identity differs')
    for key in ['source_sha256','input_sha256','clip_id']:
        if key in base and base[key]!=envelope[key]:
            raise ValueError('Analysis envelope conflicts with existing '+key)
    ids=integer_axis(prepared['ids'],'prepared ids');ts=integer_axis(prepared['ts'],'prepared time')
    pi=integer_axis(predictions['frame_id'],'prediction ids');pt=integer_axis(predictions['timestamp_us'],'prediction time')
    if not (len(ids)==len(ts) and np.array_equal(ids,pi) and np.array_equal(ts,pt)):
        raise ValueError('Saved prediction/input native axis differs')
    if ids.tolist()!=envelope['native_frame_ids'] or ts.tolist()!=envelope['timestamp_us']:
        raise ValueError('Native timeline is not the frozen source')
    n=len(ids);p=np.asarray(predictions['probabilities'])
    if p.shape!=(n,5) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Saved probabilities must be finite native x5')
    for key,prepared_key in [('motion_anchor','motion_anchor'),('evidence_available','evidence_available'),('rgb_available','available')]:
        a=np.asarray(predictions[key]);b=np.asarray(prepared[prepared_key])
        if a.shape!=(n,) or a.dtype.kind!='b' or not np.array_equal(a,b):
            raise ValueError('Saved/actual availability differs: '+key)
    if prepared['mask'].shape!=(n,73) or prepared['mask'].dtype.kind!='b':
        raise ValueError('Prepared geometry mask differs')
    path=np.asarray(base['path'])
    if path.ndim!=2 or path.shape[1]!=2 or path.dtype.kind not in 'iu' or (path<0).any() or path[:,0].max()>=n or (np.diff(path,axis=0)<0).any():
        raise ValueError('Invalid already-computed path')
    if base.get('path_axes')!=['query_native_index','learned_canonical_index']:
        raise ValueError('Unexpected path coordinate semantics')
    if value_sha(proposals)!=envelope['proposals_value_sha256']:
        raise ValueError('Fixed proposal plan differs')
    seen=set()
    for proposal in proposals:
        j=proposal['target_index'];ii=proposal['indices']
        if type(j) is not int or j not in range(5) or j in seen:
            raise ValueError('Ambiguous or unsupported physical target')
        seen.add(j)
        if len(ii)!=2 or any(type(v) is not int for v in ii) or not 0<=ii[0]<=ii[1]<n:
            raise ValueError('Proposal outside native axis')
        if ids[ii[0]:ii[1]+1].tolist()!=proposal['native_frame_ids'] or ts[ii[0]:ii[1]+1].tolist()!=proposal['timestamp_us']:
            raise ValueError('Fixed proposal frame/time mapping differs')
        source=proposal.get('source_constraint')
        if source is not None and not (j==0 and source.get('word_id')==envelope['word_id'] and source['clause_id']=='well_brush' and source['entry_id']=='NIKL_legacy:9543' and source['component_id']=='right_hand_traverses_left_arm_toward_wrist' and source['condition_relation']=='RELATED_ONLY'):
            raise ValueError('Unsupported source-clause substitution')
    return ids,ts,p

def attach(base,envelope,prepared,predictions,proposals,structural_support):
    ids,ts,p=validate_identity(base,envelope,prepared,predictions,proposals)
    stages={s['guide_id']:s for s in base['stages']}
    if len(stages)!=len(base['stages']):
        raise ValueError('Duplicate stage identity')
    observations=[]
    for proposal in proposals:
        j=proposal['target_index'];lo,hi=proposal['indices'];ii=np.arange(lo,hi+1)
        active=np.asarray(predictions['motion_anchor'])[ii]
        observed=np.asarray(predictions['evidence_available'])[ii]
        complete=bool(active.all() and observed.all() and len(ii)>=2)
        eligible=ii[active]
        maximum=float(p[eligible,j].max()) if len(eligible) else None
        best=int(eligible[np.argmax(p[eligible,j])]) if len(eligible) else None
        high=eligible[p[eligible,j]>=HIGH]
        if len(high):
            status='LOCAL_MOVEMENT_EVIDENCE_OBSERVED'
        elif not complete:
            status='INSUFFICIENT_OBSERVATION'
        elif maximum is not None and maximum<=LOW:
            status='LOCAL_MOVEMENT_NOT_CONFIRMED'
        else:
            status='LOCAL_MOVEMENT_UNCERTAIN'
        if proposal['guide_id'] not in stages:
            raise ValueError('Proposed guide missing from old result')
        guard=stage_guard(stages[proposal['guide_id']])
        source=copy.deepcopy(proposal.get('source_constraint'))
        binding=('RELATED_OBSERVATION_WITH_STAGE_CONSISTENCY' if guard['status']=='SUPPORTED'
                 else 'RELATED_OBSERVATION_STAGE_UNCONFIRMED') if source else 'REFERENCE_OBSERVATION_ONLY'
        if len(high):
            msg=NAMES[j]+'의 근거가 이 구간에서 추정됩니다.'
            if not complete:msg+=' 관측이 빠진 다른 순간까지 확인한 것은 아닙니다.'
            if guard['status']!='SUPPORTED':msg+=' 설명의 이 단계에 해당하는지는 확인이 더 필요합니다.'
        elif status=='INSUFFICIENT_OBSERVATION':
            msg='이 구간은 관측이 부족해 '+NAMES[j]+'을 충분히 확인하지 못했습니다.'
        else:
            msg='이 구간에서 '+NAMES[j]+'을 확실히 확인하지 못했습니다. 동작 오류로 판정하지 않습니다.'
        selected=np.asarray(prepared['geometry_valid'])[:73]>.5
        joint=float(prepared['mask'][best,selected].mean()) if best is not None and selected.any() else None
        mode=('RGB_AND_JOINTS' if joint is not None and joint>=JOINT_SUPPORT else 'IMAGE_TIME_ESTIMATE') if best is not None and prepared['available'][best] else ('JOINT_TIME_ESTIMATE' if best is not None else 'NO_AVAILABLE_ANCHOR')
        observations.append(dict(target_index=j,physical_name=NAMES[j],proposal=copy.deepcopy(proposal),
            guide_description=stages[proposal['guide_id']].get('description'),
            probability=maximum,status=status,high_anchor_native_frames=ids[high].tolist(),
            high_anchor_timestamp_us=ts[high].tolist(),
            maximum_anchor_native_frame=int(ids[best]) if best is not None else None,
            maximum_anchor_timestamp_us=int(ts[best]) if best is not None else None,
            maximum_anchor_evidence_mode=mode,maximum_anchor_geometry_coverage=joint,
            native_point_estimates=[dict(frame_id=int(ids[i]),timestamp_us=int(ts[i]),probability=float(p[i,j]),
                motion_anchor=bool(predictions['motion_anchor'][i]),evidence_available=bool(predictions['evidence_available'][i]),
                rgb_available=bool(predictions['rgb_available'][i])) for i in ii],
            scope_evidence_complete=complete,
            unavailable_native_frames=ids[ii[~observed]].tolist(),
            nonanchor_native_frames=ids[ii[~active]].tolist(),
            native_scope_denominator=len(ii),anchor_count=int(active.sum()),
            guide_time_binding=guard,source_binding=binding,source_constraint=source,
            source_variant_applicability_verified=None,source_local_applicability_verified=None,
            source_time_scope='PROPOSED_DTW_NOT_DIRECT_EVENT_BOUNDARY',
            structural_context_support=copy.deepcopy(structural_support[j]),
            estimate_scope='Weak-trained contextual occurrence anchors, not dense action truth or exact onset.',
            probability_is_calibrated_linguistic_truth=False,
            message_ko=msg,whole_clause_truth=None,whole_sign_correctness=None,
            complete_action=None,educational_error=None,required_repetitions=None,correction_enabled=False))
    out=copy.deepcopy(base)
    out['local_event_feedback']=dict(version=VERSION,identity=copy.deepcopy(envelope),observations=observations,
        thresholds=dict(high=HIGH,low=LOW,joint_support=JOINT_SUPPORT),
        primary_scopes_fixed_before_readout=True,scope_search=False,full_clip_argmax_used=False,
        new_model_loads=0,new_model_forwards=0,new_DTW=0,new_training=0,
        old_outputs_preserved=True,whole_clause_truth=None,whole_sign_correctness=None,
        complete_action=None,educational_error=None,correction_enabled=False)
    if value_sha({k:v for k,v in out.items() if k!='local_event_feedback'})!=value_sha(base):
        raise AssertionError('Non-additive change to existing output')
    return out
