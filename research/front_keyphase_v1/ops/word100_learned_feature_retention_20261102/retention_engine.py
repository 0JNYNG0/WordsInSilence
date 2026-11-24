"""Protected local neural observations + learned whole-video flow.

No target word enters the physical attribute branch. No extra RGB backbone.
Model posterior means physical classification confidence, not angle error.
"""
from pathlib import Path
import sys,time
import numpy as np
U=Path(__file__).resolve().parent;P=U.parent/'word100_learned_guide_feedback_20261102'
sys.path.insert(0,str(P))
from engine import LearnedGuideEngine,FEATURE_KO,read
from prepare_projection import load_module
class RetentionEngine(LearnedGuideEngine):
 def __init__(self,seed=42,calibration=True):
  super().__init__('projection_finetune',seed,False)
  self.retained_norm=read(U/'calibration'/f'{seed}.json') if calibration else None
 def infer(self,arrays,word):
  p=super().infer(arrays,word)
  # Frozen local heads consume only observations, never word/guide/prototype.
  adapter=load_module('_retained_public_observer',U.parent/'word100_multipeople_live_feedback_20261101/adapter.py') if '_retained_public_observer' not in sys.modules else sys.modules['_retained_public_observer']
  t=time.perf_counter();retained,centers=adapter._heads(arrays)
  p['retained_probability']=retained;p['retained_head_seconds']=time.perf_counter()-t
  return p
 def compare(self,prediction,arrays,word,calibration=False):
  result=super().compare(prediction,arrays,word,True)
  unavailable=np.flatnonzero(~np.asarray(arrays.get('image_available',np.ones(len(arrays['frame_id']),bool)),bool))
  if len(unavailable):
   result['evidence_gaps']=[]
   for part in np.split(unavailable,np.flatnonzero(np.diff(unavailable)>1)+1):
    result['evidence_gaps'].append(dict(start_seconds=float(arrays['timestamp_us'][part[0]]/1e6),end_seconds=float(arrays['timestamp_us'][part[-1]]/1e6),
     frames=part.tolist(),status='CURRENT_IMAGE_UNAVAILABLE',reason='Blank/unavailable current pixels cannot be certified by temporal model or stale coordinates'))
  p=np.asarray(prediction['retained_probability']);frame_id=np.asarray(arrays['frame_id']);mask=np.asarray(arrays['mask'])
  for s in result['stages']:
   ix=np.flatnonzero(np.isin(frame_id,s['frame_ids']));finite=np.isfinite(p[ix]);means=np.array([p[ix,j][finite[:,j]].mean() if finite[:,j].any() else np.nan for j in range(7)])
   s['attribute_mean']=[float(x) if np.isfinite(x) else None for x in means];s['posterior_coverage']=finite.mean(0).tolist()
   if calibration:continue
   norm=self.retained_norm['stages'][s['guide_id']];features=[];differences=[];unknown=[]
   time_ok=s['neural_stage_agreement']>=.5 and s['image_time_coverage']>=.8 and s['unique_query_frames']>=2
   for f in norm['supported_attributes']:
    j=f['target_index'];mean=means[j];expected=f['expected'];coverage=s['posterior_coverage'][j]
    geom=mask[ix];side=1 if j in [0,1,3,5] else 0
    relevant=geom[:,(39 if side else 11):(53 if side else 25)] if j<5 else geom[:,[8,9,36,37]]
    geometry=float(relevant.mean());evidence='RGB_AND_JOINTS' if geometry>=.5 else 'RGB_TIME_ESTIMATE'
    if not time_ok or coverage<.8 or not np.isfinite(mean):value='INSUFFICIENT_TIME_EVIDENCE'
    else:
     distance=abs(float(mean)-expected);opposite=int(mean>=.5)!=int(expected)
     value=('DIFFERENCE_CANDIDATE' if opposite else 'CLASSIFICATION_CONFIDENCE_LOW') if distance>f['distance_limit'] else 'WITHIN_PUBLIC_TRAINING_RANGE'
     # Lack of joints does not erase an RGB estimate. It also cannot certify
     # visible finger support. Do not turn a guessed expected positive into pass.
     if evidence=='RGB_TIME_ESTIMATE' and (expected>.5 or opposite):
      value='ESTIMATED_DIFFERENCE_NEEDS_CONFIRMATION' if opposite else 'IMAGE_ESTIMATE_NEEDS_CONFIRMATION'
    features.append(dict(name=FEATURE_KO[j],expected=expected,observed_mean=float(mean) if np.isfinite(mean) else None,
      status=value,evidence=evidence,posterior_coverage=coverage,geometry_coverage=geometry,probability_is_physical_difference=False))
    if value=='DIFFERENCE_CANDIDATE':differences.append(FEATURE_KO[j])
    if value not in ['DIFFERENCE_CANDIDATE','WITHIN_PUBLIC_TRAINING_RANGE']:unknown.append(FEATURE_KO[j])
   s['features']=features;s['whole_sign_correctness']=None
   if not time_ok:s['status']='INSUFFICIENT_TIME_EVIDENCE';s['message']='설명과 대응하는 시점의 근거가 부족합니다. 동작 오류로 확정하지 않습니다.'
   elif differences:s['status']='FEATURE_DIFFERENCE_CANDIDATE';s['message']='배운 특징과 다르게 보이는 항목: '+', '.join(differences)+'. 이 구간을 확인해 주세요.'
   elif unknown:s['status']='IMAGE_TIME_ESTIMATE_ONLY';s['message']='영상과 흐름에서 추정한 결과가 있지만 확인 근거가 부족한 항목은 보류합니다.'
   elif s['flow_cost']>norm['flow_limit']:s['status']='FLOW_DIFFERENCE_NEEDS_REVIEW';s['message']='흐름이 학습 범위와 다릅니다. 허용 가능한 변이인지 추가 확인이 필요합니다.'
   elif not features:s['status']='STAGE_LOCATED_FEATURE_SUPPORT_LIMITED';s['message']='구간을 찾았지만 설명의 세부 특징을 평가할 학습 근거가 부족합니다.'
   else:s['status']='SUPPORTED_FEATURES_SIMILAR';s['message']='확인한 특징은 배운 범위와 비슷합니다. 단어 전체의 정답 판정은 아닙니다.'
  result.update(observer='retained_protected_neural_branch',phase='learned_partial_finetune_temporal_branch',policy_version='retained_branch_v1',
   new_training=0,retained_head_seconds=prediction.get('retained_head_seconds'),all_semantic_requirements_checked=False)
  return result
