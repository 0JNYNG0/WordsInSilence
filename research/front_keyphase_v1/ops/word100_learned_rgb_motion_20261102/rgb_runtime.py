import copy,time,numpy as np,torch
from rgb_model import R,M,RGBMotionNet,rgb_input,read,predict
from motion_model import features,normalize
from motion_runtime import MotionObserver

class RGBMotionObserver(MotionObserver):
 def __init__(self,condition,seed):
  super().__init__('ordered_gru',seed);self.model=RGBMotionNet(condition);self.model.load_state_dict(torch.load(R/'runs'/f'{condition}_{seed}/model.pt',map_location='cpu',weights_only=True))
  self.model.eval();self.model.requires_grad_(False);self.condition=condition;self.rgb_stats=read(R/'RGB_NORMALIZATION.json')
 def attach(self,base,a,shape,relations):
  out=copy.deepcopy(base);z,mask=features(a,shape,relations)
  video=rgb_input(a,self.rgb_stats);x=np.concatenate([normalize(z,mask,self.stats),video],axis=1)
  path=np.asarray(base['path']);ids=np.asarray(a['frame_id']);ts=np.asarray(a['timestamp_us'])/1e6
  available=video[:,-1]>.5;descriptors=[];seq=[]
  for s in self.support:
   if s['word_id']!=base['word_id']:continue
   lo,hi=s['canonical_frames'];q=np.unique(path[(path[:,1]>=lo)&(path[:,1]<=hi),0]);assert len(q)
   ix=np.arange(q.min(),q.max()+1);seq.append(x[ix]);descriptors.append((s,ix))
  t=time.perf_counter();pp=predict(self.model,seq);seconds=time.perf_counter()-t;items=[]
  for (s,ix),p in zip(descriptors,pp):
   j=s['target'];im=float(available[ix].mean());coverage=float(mask[ix].mean());continuity=float(p[5]);value=float(p[j]);opposite={2:4,4:2}.get(j)
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
  out['motion_scope']='Five physical directions, direct pretrained video evidence and learned continuity; not complete semantic correctness, wrist rotation, contact or required repeat count'
  return out

