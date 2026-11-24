"""Shared physical observations -> sourced phrase and automatic native time.

The neural observers remain frozen. Target identity selects the expected
components; it never supplies missing image observations or a correct answer.
"""
from pathlib import Path
import json,copy,time
import numpy as np
X=Path(__file__).resolve().parent;O=X.parent
S=O/'word100_source_component_connector_20261102'
Q=O/'word100_predicate_supervision_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

POINTS={
 'right_non_thumb_closed':('retained_probability',3,'오른손의 긴 네 손가락이 접힌 모양',[39,40,41,42,43,44,45,46,47,48,49,50,51,52]),
 'right_hand_at_left_arm_projection':('relation_probability',2,'오른손이 왼팔 부근에 놓인 관계',[2,3,4,5,36,37]),
 'bilateral_upper_ends_converged':('relation_probability',1,'양손의 위쪽 끝이 가까워진 배치',[6,7,34,35])
}
MOTIONS={'right_hand_traverses_left_arm_toward_wrist':(0,'오른손이 왼팔을 따라 왼손목 쪽으로 이동하는 흐름')}

class SourceBridge:
 def __init__(self):
  self.catalog={r['clause_id']:r for r in read(Q/'CLAUSE_COMPONENT_LINKS.json')['rows']}
  self.nominal={}
  for r in read(S/'DIRECT_TARGETS.json')['rows']:
   if r['role']!='fit' or r['person']!=1 or not r['mask'] or r['y']!=1 or not r['source_clause_ids']:continue
   k=(r['word_id'],r['target']);v=self.nominal.setdefault(k,dict(frames=[],clauses=set()))
   v['frames']+=r['native_frames'];v['clauses'].update(r['source_clause_ids'])

 def attach(self,result,a):
  before=copy.deepcopy(result);out=copy.deepcopy(result);path=np.array(result['path'],int)
  ids=np.asarray(a['frame_id']);ts=np.asarray(a['timestamp_us']);n=len(ids)
  assert path.ndim==2 and path.shape[1]==2 and path[:,0].min()>=0 and path[:,0].max()<n
  assert np.all(np.diff(ids)>0) and np.all(np.diff(ts)>0)
  available=np.asarray(a.get('image_available',np.ones(n,bool)),bool)
  components=[];t0=time.perf_counter()
  for (word,target),nominal in self.nominal.items():
   if word!=result['word_id']:continue
   lo,hi=min(nominal['frames']),max(nominal['frames'])
   indices=np.unique(path[(path[:,1]>=lo)&(path[:,1]<=hi),0]);assert len(indices)
   full=np.arange(indices.min(),indices.max()+1);frames=ids[full]
   clauses=[]
   for cid in sorted(nominal['clauses']):
    c=self.catalog[cid];clauses.append(dict(id=cid,text=c['text'],url=c['url'],source=c['source'],unchecked=c['unsupported_components']))
   item=dict(component_id=target,source_clauses=clauses,canonical_anchor_extent=[lo,hi],
     frame_ids=frames.tolist(),start_seconds=float(ts[full[0]]/1e6),end_seconds=float(ts[full[-1]]/1e6),
     scope='learned_prototype_DTW_proposal_not_exact_event_boundary',
     complete_action_verified=False,whole_clause_correctness=None,whole_sign_correctness=None,
     source_description_is_observation=False,correction_enabled=False)
   if target in POINTS:
    field,j,name,relevant=POINTS[target];p=np.asarray(a[field])[full,j]
    geom=np.asarray(a['mask'],bool)[full][:,relevant].mean(1)
    usable=available[full]&np.isfinite(p)
    state=np.where(~usable,'UNOBSERVED',np.where(p>=.8,'FEATURE_OBSERVED',np.where(p<=.2,'FEATURE_NOT_CONFIRMED','UNCERTAIN')))
    model_state=state.copy()
    state[(state=='FEATURE_OBSERVED')&(geom<.5)]='IMAGE_TIME_ESTIMATE'
    item.update(name=name,evidence_mode='frozen_neural_point_observer',
      confirmed_frames=frames[state=='FEATURE_OBSERVED'].tolist(),
      unconfirmed_frames=frames[state=='FEATURE_NOT_CONFIRMED'].tolist(),
      image_estimated_frames=frames[state=='IMAGE_TIME_ESTIMATE'].tolist(),
      uncertain_frames=frames[np.isin(state,['UNOBSERVED','UNCERTAIN','IMAGE_TIME_ESTIMATE'])].tolist(),
      point_evidence=[dict(frame=int(f),seconds=float(tt/1e6),probability=float(v) if np.isfinite(v) else None,
        status=str(st),model_status=str(ms),joint_support=float(g),image_available=bool(av),
        evidence='RGB_AND_JOINTS' if g>=.5 else 'RGB_TIME_ESTIMATE')
        for f,tt,v,st,ms,g,av in zip(frames,ts[full],p,state,model_state,geom,available[full])],
      status='POINT_EVIDENCE_AVAILABLE' if usable.any() else 'INSUFFICIENT_OBSERVATION',
      message='이 설명과 관련된 특징을 시간별로 표시했습니다. 중간의 다른 모양이나 낮은 확신을 곧바로 잘못된 동작으로 판정하지 않습니다.')
   elif target in MOTIONS:
    j,name=MOTIONS[target]
    same=[m for m in result.get('motion_observations',[]) if m['target']==j and m['frame_ids']==frames.tolist()]
    if len(same)==1:
     m=same[0];observed=m['status']=='DIRECTED_MOTION_OBSERVED' and available[full].mean()>=.8
     binding=m.get('guide_time_binding','UNCONFIRMED')
     item.update(name=name,evidence_mode='retained_RGB_joint_temporal_observer',
       probability=m['probability'],continuity_probability=m['continuity_probability'],
       image_coverage=float(available[full].mean()),old_motion_status=m['status'],guide_time_binding=binding,
       status='RELATED_MOVEMENT_OBSERVED' if observed else 'RELATED_MOVEMENT_UNCONFIRMED',
       message=(name+'을 확인했습니다. 이 관측만으로 전체 순서나 마무리를 확인한 것은 아닙니다.') if observed else
        '설명과 관련된 이동을 충분히 확인하지 못했습니다. 관측 부족 또는 시간 연결의 문제도 있어 동작 오류로 확정하지 않습니다.')
     if available[full].mean()<.8:item['status']='INSUFFICIENT_OBSERVATION'
     elif observed and binding!='SUPPORTED':
      item['status']='MOVEMENT_SEEN_SOURCE_UNCONFIRMED'
      item['message']='관련된 이동은 보이지만, 사전 설명의 이 부분과 연결되는 시간인지는 충분히 확인하지 못했습니다.'
    else:item.update(name=name,status='EXACT_SCOPE_OBSERVER_UNAVAILABLE',message='이 설명 구간과 정확히 같은 범위의 이동 관측이 없어 확인을 보류합니다.')
   else:
    item.update(name=target,status='COMPONENT_NOT_RUNTIME_VALIDATED',
       message='이 설명의 해당 세부 특징은 현재 데모에서 검증된 관측기가 없습니다. 확인한 것으로 표시하지 않습니다.')
   components.append(item)
  out['source_components']=components
  out['source_component_contract']=dict(version='source-bridge-20261102-v1',
   old_model_outputs_preserved=True,new_inference=0,new_training=0,
   per_reference_video_comparisons=0,learned_prototype_comparisons=1,
   automatic_scope_is_not_completion=True,completion_truth_available=False,
   unsupported_source_requirements_never_confirmed=True,bridge_seconds=time.perf_counter()-t0)
  # Preserve every old field: additive presentation must not silently change
  # neural probabilities, sensor gaps, feedback scope, or the learned path.
  for k,v in before.items():assert out[k]==v,k
  return out
