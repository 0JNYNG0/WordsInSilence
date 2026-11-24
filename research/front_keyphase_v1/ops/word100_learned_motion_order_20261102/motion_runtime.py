"""Goal-blind learned motion observations, queried on the learned DTW flow."""
import copy,time
import numpy as np,torch
from motion_model import M,P,read,features,normalize,predict,MotionNet
NAMES=['오른손이 왼팔을 따라 왼손목 쪽으로 이동','두 손이 서로 벌어짐','오른손이 왼손에 비해 올라감','얼굴 앞 오른손이 몸통 쪽으로 내려감','오른손이 왼손에 비해 내려감']
def support():
 labels=read(M/'MOTION_TARGETS.json')['rows'];guides=read(P/'GUIDE_SUPPORT.json')['guides'];out=[]
 for r in labels:
  if r['person']!=1 or r['kind']!='directed':continue
  j=next(j for j in range(5) if r['mask'][j] and r['y'][j]);people={q['person'] for q in labels if q['role']=='fit' and q['word_id']==r['word_id'] and q['mask'][j] and q['y'][j]}
  assert len(people)>=3
  candidates=[g for g in guides if g['word_id']==r['word_id']]
  g=max(candidates,key=lambda g:max(0,min(g['frames'][1],r['frames'][1])-max(g['frames'][0],r['frames'][0])+1))
  out.append(dict(word_id=r['word_id'],target=j,name=NAMES[j],canonical_frames=r['frames'],guide_id=g['canonical_stage_id'],description=g['description'],
   source_phrases=[v['text'] for v in g['source_cards']],fit_positive_people=sorted(people),
   expected_scope='Observed FIT reference motion; context association, not all text requirements or certified repeat count'))
 return out
class MotionObserver:
 def __init__(self,condition='ordered_gru',seed=42):
  self.model=MotionNet(condition);self.model.load_state_dict(torch.load(M/'runs'/f'{condition}_{seed}'/'model.pt',map_location='cpu',weights_only=True));self.model.eval();self.model.requires_grad_(False)
  self.stats=read(M/'NORMALIZATION.json');self.support=read(M/'MOTION_GUIDE_SUPPORT.json');self.condition=condition;self.seed=seed
 def attach(self,base,a,shape,relations):
  out=copy.deepcopy(base);z,mask=features(a,shape,relations);x=normalize(z,mask,self.stats);path=np.asarray(base['path']);ids=np.asarray(a['frame_id']);ts=np.asarray(a['timestamp_us'])/1e6
  available=np.asarray(a.get('image_available',np.ones(len(ids),bool)),bool);descriptors=[];seq=[]
  for s in self.support:
   if s['word_id']!=base['word_id']:continue
   lo,hi=s['canonical_frames'];q=np.unique(path[(path[:,1]>=lo)&(path[:,1]<=hi),0]);assert len(q)
   ix=np.arange(q.min(),q.max()+1);seq.append(x[ix]);descriptors.append((s,ix))
  t=time.perf_counter();pp=predict(self.model,seq);seconds=time.perf_counter()-t;items=[]
  for (s,ix),p in zip(descriptors,pp):
   j=s['target'];im=float(available[ix].mean());coverage=float(mask[ix].mean());continuity=float(p[5]);value=float(p[j]);opposite={2:4,4:2}.get(j)
   # Missing pixels never become a positive result due to stale pose or context.
   if im<.8 or len(ix)<3 or coverage<.5:status='MOTION_OBSERVATION_INSUFFICIENT'
   elif continuity<.8:status='MOTION_CONTINUITY_UNCONFIRMED'
   elif value>=.8:status='DIRECTED_MOTION_OBSERVED'
   elif value<=.2 and opposite is not None and p[opposite]>=.8:status='OPPOSITE_DIRECTION_CANDIDATE'
   else:status='EXPECTED_MOTION_UNCONFIRMED'
   items.append(dict(**s,status=status,start_seconds=float(ts[ix[0]]),end_seconds=float(ts[ix[-1]]),frame_ids=ids[ix].tolist(),
    probability=value,continuity_probability=continuity,feature_coverage=coverage,image_coverage=im,all_motion_probabilities=p[:5].tolist(),
    whole_sign_error=False,latent_time_alignment_is_not_motion_proof=True))
  out['motion_observations']=items;out['motion_head_seconds']=seconds;out['motion_condition']=self.condition;out['motion_seed']=self.seed
  for stage in out['stages']:
   stage['motions']=[i for i in items if i['guide_id']==stage['guide_id']]
   if any(i['status']!='DIRECTED_MOTION_OBSERVED' for i in stage['motions']) and stage['status']=='SUPPORTED_FEATURES_SIMILAR':
    stage['status']='IMPORTANT_MOTION_UNCONFIRMED';stage['message']='손모양 일부는 비슷하지만 설명에 연결된 이동 과정은 확인이 더 필요합니다.'
  out['motion_scope']='Five directed physical motions and a learned continuity estimate; no guaranteed semantic correctness, wrist rotation, contact or required repetition count'
  return out
