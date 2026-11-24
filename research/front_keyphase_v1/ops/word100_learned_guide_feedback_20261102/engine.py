"""Learned-guide runtime: one video representation + target, no reference bank."""
from pathlib import Path
from collections import defaultdict
import json,time
import hashlib
import numpy as np
import torch
from model import GuideFeedbackNet,collate
from alignment import dtw
U=Path(__file__).resolve().parent
WORDS=('WORD0739','WORD1133','WORD1567','WORD2523')
FEATURE_KO=['오른손 검지만 펴짐','오른손 긴 네 손가락 펼침','왼손 긴 네 손가락 펼침','오른손 긴 네 손가락 접힘','왼손 긴 네 손가락 접힘','오른손이 더 높은 배치','왼손이 더 높은 배치']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def build_item(arrays,stats,word):
 n=len(arrays['frame_id']);x=np.asarray(arrays['x']);mask=np.asarray(arrays['mask'],bool)
 assert x.shape==mask.shape==(n,69) and n>=16
 assert np.all(np.diff(arrays['frame_id'])>0) and np.all(np.diff(arrays['timestamp_us'])>0)
 pm=np.asarray(stats['pose_mean']);ps=np.asarray(stats['pose_std']);pose=np.where(mask,(x-pm)/ps,0).astype(np.float32)
 return dict(pose=np.concatenate([pose,mask],1).astype(np.float32),pose_y=pose,pose_m=mask.astype(np.float32),
   rgb=np.asarray(arrays['rgb'],np.float32),prelinear=np.asarray(arrays['prelinear'],np.float32),word=WORDS.index(word),
   y=np.zeros((n,7),np.float32),ym=np.zeros((n,7),np.float32),stage_y=np.full(n,-1,np.int64),stage_w=np.zeros(n,np.float32),canonical_index=np.full(n,-1,np.int64))
class LearnedGuideEngine:
 def __init__(self,condition='projection_finetune',seed=42,calibration=True):
  torch.set_num_threads(4);self.condition=condition;self.seed=seed;self.stats=read(U/'NORMALIZATION.json')
  self.guides=read(U/'GUIDE_SUPPORT.json')['guides'];self.ids=[g['canonical_stage_id'] for g in self.guides]
  self.model=GuideFeedbackNet(np.load(U/'PRETRAINED_PROJECTION.npy'),np.asarray(self.stats['guide_vectors']),np.asarray(self.stats['guide_words']),self.stats['prototype_lengths'],condition=='projection_finetune')
  checkpoint=U/'runs'/f'{condition}_{seed}'/'model.pt'
  assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==read(checkpoint.parent/'COMPLETE.json')['checkpoint_sha256']
  self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);self.model.eval()
  self.normal=read(U/'calibration'/f'{condition}_{seed}.json') if calibration else None
 @torch.inference_mode()
 def infer(self,arrays,word):
  item=build_item(arrays,self.stats,word);b=collate([item]);start=time.perf_counter()
  out=self.model(b['pose'],b['rgb'],b['prelinear'],b['lengths'],b['word'])
  return dict(latent=out['latent'][0].numpy(),probability=out['attributes'][0].sigmoid().numpy(),stage_probability=out['stage'][0].softmax(-1).numpy(),head_seconds=time.perf_counter()-start)
 def compare(self,prediction,arrays,word,calibration=False):
  wi=WORDS.index(word);n=int(self.model.prototype_lengths[wi]);prototype=self.model.prototype[wi,:n].detach().numpy()
  start=time.perf_counter();al=dtw(prediction['latent'],prototype);path=al['path'];p=prediction['probability'];sp=prediction['stage_probability'];ts=np.asarray(arrays['timestamp_us'])/1e6
  image_available=np.asarray(arrays.get('image_available',np.ones(len(ts),bool)),bool)
  stages=[]
  missing=np.flatnonzero(~image_available & ~np.asarray(arrays['mask']).any(1));gaps=[]
  if len(missing):
   for part in np.split(missing,np.flatnonzero(np.diff(missing)>1)+1):
    gaps.append(dict(start_seconds=float(ts[part[0]]),end_seconds=float(ts[part[-1]]),frames=part.tolist(),
      status='NO_OBSERVATION_NOT_A_CORRECT_ACTION',reason='RGB and joint inputs unavailable; retained independently of DTW path'))
  for g in self.guides:
   if g['word_id']!=word:continue
   lo,hi=g['frames'];ix=np.unique(path[(path[:,1]>=lo)&(path[:,1]<=hi),0]);assert len(ix)>0
   full=np.arange(ix.min(),ix.max()+1);sindex=self.ids.index(g['canonical_stage_id'])
   agreement=float((sp[full].argmax(1)==sindex).mean());observed=float(image_available[full].mean())
   cost=float(al['cost'][path[(path[:,1]>=lo)&(path[:,1]<=hi),0],path[(path[:,1]>=lo)&(path[:,1]<=hi),1]].mean())
   span=dict(guide_id=g['canonical_stage_id'],description=g['description'],source_phrases=[c['text'] for c in g.get('source_cards') or []],
    source_phrase_scope='saved partial association; not every phrase applies throughout this stage',
    start_seconds=float(ts[full[0]]),end_seconds=float(ts[full[-1]]),frame_ids=np.asarray(arrays['frame_id'])[full].tolist(),
    neural_stage_agreement=agreement,image_time_coverage=observed,unique_query_frames=len(ix),reference_stage_frames=hi-lo+1,
    flow_cost=cost,attribute_mean=p[full].mean(0).tolist(),features=[],status='UNSCORED',whole_sign_correctness=None)
   if not calibration:
    norm=self.normal['stages'][g['canonical_stage_id']];differences=[];unknown=[]
    # Agreement is a consistency guard, not independent evidence or a KSL rule.
    time_supported=agreement>=.5 and observed>=.8 and len(ix)>=2
    for f in norm['supported_attributes']:
     j=f['target_index'];mean=float(p[full,j].mean());expected=f['expected'];dist=abs(mean-expected)
     # A posterior is confidence in a binary property, not the physical amount
     # of finger flexion. Lower confidence in the SAME class is not a mistake.
     predicted=int(mean>=.5);opposite=predicted!=int(expected)
     value=('DIFFERENCE_CANDIDATE' if opposite else 'CLASSIFICATION_CONFIDENCE_LOW') if dist>f['distance_limit'] else 'WITHIN_PUBLIC_TRAINING_RANGE'
     if not time_supported:value='INSUFFICIENT_TIME_EVIDENCE'
     # For fingers, distinguish image/time estimates without direct joint support.
     geom=np.asarray(arrays['mask'])[full];side=(1 if j in [0,1,3,5] else 0)
     relevant=geom[:,(39 if side else 11):(53 if side else 25)] if j<5 else geom[:,[8,9,36,37]]
     geometry=float(relevant.mean())
     evidence='RGB_AND_JOINTS' if geometry>=.5 else 'RGB_TIME_ESTIMATE'
     if evidence=='RGB_TIME_ESTIMATE' and value=='DIFFERENCE_CANDIDATE':value='ESTIMATED_DIFFERENCE_NEEDS_CONFIRMATION'
     span['features'].append(dict(name=FEATURE_KO[j],expected=expected,predicted=predicted,observed_mean=mean,difference=dist,limit=f['distance_limit'],status=value,evidence=evidence,probability_is_physical_difference=False))
     if value=='DIFFERENCE_CANDIDATE':differences.append(FEATURE_KO[j])
     if 'INSUFFICIENT' in value or 'CONFIRMATION' in value or value=='CLASSIFICATION_CONFIDENCE_LOW':unknown.append(FEATURE_KO[j])
    if not time_supported:
     span['status']='INSUFFICIENT_TIME_EVIDENCE';span['message']='설명에 해당하는 시점을 충분히 연결하지 못했습니다. 동작이 틀렸다는 뜻은 아닙니다.'
    elif differences:
     span['status']='FEATURE_DIFFERENCE_CANDIDATE';span['message']='이 구간에서 공개 자료로 배운 특징과 차이가 보입니다: '+', '.join(differences)+'.'
    elif unknown:
     span['status']='IMAGE_TIME_ESTIMATE_ONLY';span['message']='확신이 낮거나 영상과 앞뒤 흐름으로 추정한 항목이 있습니다. 동작 오류로 확정하지 않습니다.'
    elif cost>norm['flow_limit']:
     span['status']='FLOW_DIFFERENCE_NEEDS_REVIEW';span['message']='이 구간의 흐름이 학습 범위와 다릅니다. 정상 변이인지 중요한 차이인지 추가 확인이 필요합니다.'
    elif not span['features']:
     span['status']='STAGE_LOCATED_FEATURE_SUPPORT_LIMITED';span['message']='대응 구간은 찾았지만 이 설명의 세부 특징을 판정할 학습 근거가 부족합니다.'
    else:
     span['status']='SUPPORTED_FEATURES_SIMILAR';span['message']='현재 확인할 수 있는 특징은 학습한 공개 범위와 비슷합니다. 단어 전체의 정답 판정은 아닙니다.'
   stages.append(span)
  return dict(word_id=word,stages=stages,path=path.tolist(),path_axes=['query_native_index','learned_canonical_index'],
   reference_video_reads=0,individual_reference_comparisons=0,learned_prototype_comparisons=1,
   head_seconds=prediction.get('head_seconds'),alignment_seconds=time.perf_counter()-start,
   evidence_gaps=gaps,whole_sign_correctness=None,policy_version='v2_posterior_is_not_motion_difference',
   unchecked=['엄지의 정확한 모양·손바닥 방향·접촉 여부','짧은 주요 동작의 누락과 정확한 반복 횟수','단어 전체의 교육적 정오'],
   feedback_scope='learned physical features and guide-linked flow; not certified KSL correctness')
