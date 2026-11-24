from pathlib import Path
import time,hashlib,json,copy
from collections import defaultdict,Counter
import torch,numpy as np
from relation_model import V,read,RelationNet,dense
from train_relations import save
P=V.parent/'word100_learned_guide_feedback_20261102';U=V.parent/'word100_learned_feature_retention_20261102'
TARGET_KO=['오른손이 얼굴 앞에 있음','양손의 긴 손가락 끝이 한곳으로 모임','오른손이 왼팔 부근에 있음']
def support():
 labels=read(V/'RELATION_TARGETS.json')['rows'];manifest=read(P/'INPUT_MANIFEST.json')['rows'];guides=read(P/'GUIDE_SUPPORT.json')['guides'];result={}
 for g in guides:
  pools=[[] for _ in range(3)]
  for r in manifest:
   if r['role']!='fit':continue
   span=next((s for s in r['stages'] if s['canonical_stage_id']==g['canonical_stage_id']),None)
   if span is None:continue
   for l in labels:
    if l['clip_id']!=r['clip_id'] or not span['frames'][0]<=l['center_frame']<=span['frames'][1]:continue
    for j in range(3):
     if l['mask'][j]:pools[j].append(dict(person=r['person'],frame=l['center_frame'],value=l['y'][j],canonical=r['person']==1))
  # Canonical positive + multi-person positive evidence; a word name alone never sets an expected relation.
  active=[]
  for j,q in enumerate(pools):
   people={a['person'] for a in q if a['value']};rate=np.mean([a['value'] for a in q]) if q else 0
   if len(people)>=3 and rate>=.8 and any(a['canonical'] and a['value'] for a in q):active.append(j)
  result[g['canonical_stage_id']]=dict(positive_targets=active,reviewed_centers=pools,expected_scope='physically observed FIT reference relation; linguistic requiredness not proved')
 return result
def attach(base,a,prob,supports):
 out=copy.deepcopy(base);ids=np.asarray(a['frame_id']);available=np.asarray(a.get('image_available',np.ones(len(ids),bool)))
 for s in out['stages']:
  ix=np.flatnonzero(np.isin(ids,s['frame_ids']));items=[]
  for j in supports[s['guide_id']]['positive_targets']:
   p=prob[ix,j];valid=np.isfinite(p)&available[ix];coverage=float(valid.mean());found=valid&(p>=.8)
   consecutive=any(found[:-1]&found[1:]) if len(found)>1 else False
   low=bool(valid.any() and np.max(p[valid])<=.2)
   time_ok=s['neural_stage_agreement']>=.5 and len(ix)>=2
   if not time_ok or coverage<.8:state='RELATION_UNCERTAIN'
   elif consecutive:state='POSITIVE_RELATION_OBSERVED'
   elif low:state='EXPECTED_RELATION_NOT_OBSERVED'
   else:state='RELATION_UNCERTAIN'
   items.append(dict(target_index=j,name=TARGET_KO[j],status=state,coverage=coverage,
     probability_max=float(np.max(p[valid])) if valid.any() else None,
     observed_frame_ids=ids[ix[found]].tolist(),whole_sign_error=False,
     message=('이 구간에서 참고에 보였던 관계를 확인하지 못했습니다. 촬영 근거나 실제 동작 차이를 확인해야 합니다.' if state=='EXPECTED_RELATION_NOT_OBSERVED' else
              '영상·관절에서 이 관계를 추정했습니다. 접촉이나 동작 전체의 정답을 뜻하지 않습니다.' if state=='POSITIVE_RELATION_OBSERVED' else '이 관계를 확실하게 판단하기 어려워 보류합니다.')))
  s['relations']=items
  if any(t['status']!='POSITIVE_RELATION_OBSERVED' for t in items) and s['status']=='SUPPORTED_FEATURES_SIMILAR':
   s['status']='IMPORTANT_RELATION_UNCONFIRMED';s['message']='손모양 일부는 비슷하지만 설명의 신체 관계를 확인하지 못해 이 구간은 확인이 필요합니다.'
 out['relation_head']='public_supervised_goal_blind_3_relations';out['whole_sign_correctness']=None
 return out
def main():
 torch.set_num_threads(4);assert not (V/'RELATION_COMPARISON.json').exists()
 supports=support();save(V/'GUIDE_RELATION_SUPPORT.json',supports)
 save(V/'FEEDBACK_CONTRACT.json',dict(positive_threshold=.8,negative_threshold=.2,minimum_consecutive_frames=2,minimum_coverage=.8,
  min_fit_people=3,min_positive_rate=.8,canonical_positive_required=True,source='before actual-control relation predictions',
  absence_is_not_linguistic_error=True,unknown_not_normal=True,no_development_calibration=True))
 stats=read(V/'NORMALIZATION.json');cases=read(U/'VIDEO_CONTROL_RESULTS.json')['rows'];reports=[]
 for cond in ['geometry_only','rgb_geometry']:
  for seed in [42,43,44]:
   model=RelationNet(cond=='rgb_geometry');model.load_state_dict(torch.load(V/'runs'/f'{cond}_{seed}'/'model.pt',weights_only=True));model.eval()
   for row in cases:
    folder=Path(row['result_path']).parent;a=dict(np.load(folder/'analysis.npz'));base=read(folder/'RESULT.json')
    t=time.perf_counter();prob=dense(model,a,stats);seconds=time.perf_counter()-t
    out=attach(base,a,prob,supports);name=f"{cond}_{seed}_{row['word_id']}_{row['kind']}";d=V/'control_predictions';d.mkdir(exist_ok=True)
    np.savez_compressed(d/f'{name}.npz',probability=prob,frame_id=a['frame_id'],timestamp_us=a['timestamp_us'])
    save(V/'control_feedback'/f'{name}.json',out)
    reports.append(dict(condition=cond,seed=seed,word_id=row['word_id'],kind=row['kind'],head_seconds=seconds,
     statuses=dict(Counter(s['status'] for s in out['stages'])),relations=[dict(guide=s['guide_id'],**t) for s in out['stages'] for t in s['relations']],
     evidence_gaps=len(out['evidence_gaps']),learned_dtw_preserved=out['path']==base['path'],
     shape_features_preserved=all(s['features']==old['features'] for s,old in zip(out['stages'],base['stages']))))
 save(V/'RELATION_COMPARISON.json',dict(rows=reports,source_videos_reextracted=False,new_relation_head_inference=True,
  old_models_unchanged=True,full_KSL_correctness=False,controls_are_physical_interventions_not_learner_errors=True))
 for c in ['geometry_only','rgb_geometry']:
  for kind in ['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest','action_cut','main_blackout']:
   rr=[r for r in reports if r['condition']==c and r['kind']==kind];counts=Counter(t['status'] for r in rr for t in r['relations'])
   print(c,kind,dict(counts),flush=True)
 print('SUPPORT', {g:v['positive_targets'] for g,v in supports.items()},flush=True)
if __name__=='__main__':main()
