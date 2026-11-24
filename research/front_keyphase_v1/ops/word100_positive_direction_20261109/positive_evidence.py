"""Post-model evidence contract. No word-ID branches, coordinate repair or training.

An attribute probability is model evidence, not an educational correctness score.
Signed projection is a 2D movement observation, not proof of physical contact.
"""
from pathlib import Path
import json,copy
import numpy as np
U=Path(__file__).resolve().parent
CONFIG=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))
CONTRACTS=json.loads((U/'GUIDE_CONTRACTS.json').read_text(encoding='utf-8'))['contracts']
OPPOSITE={1:3,3:1,2:4,4:2,5:6,6:5}
NAMES={0:'오른손 검지',1:'오른손 펴기',2:'왼손 펴기',3:'오른손 닫기',4:'왼손 닫기',5:'오른손의 상대 높이',6:'왼손의 상대 높이'}

def runs(mask,ts,min_ms=60):
 out=[];start=None;dt=float(np.median(np.diff(ts))) if len(ts)>1 else 0
 for i,on in enumerate(np.r_[mask,False]):
  if on and start is None:start=i
  elif not on and start is not None:
   if ts[i-1]-ts[start]+dt>=min_ms*1000:out.append([int(start),int(i-1)])
   start=None
 return out

def feature_available(x,j):
 # Model indices: R index, R open, L open, R closed, L closed, R high, L high.
 valid=x[:,186:237].astype(bool)
 if j in (5,6):return valid[:,[9,30]].all(1)
 base=9 if j in (2,4) else 30
 # Four-finger claims need those four chains, not five arbitrary detected points.
 pts=[base]+[base+k for k in range(5,21)]
 return valid[:,pts].all(1)

def projected_shape_support(x,j):
 """Necessary 2D consistency for a correction cue, NOT a second independent AI.

 A projection cannot prove a hidden finger is open or closed. Clear disagreement
 with projected chains downgrades a model contradiction instead of inventing a fix.
 """
 if j not in (1,2,3,4):return np.ones(len(x),bool)
 base=9 if j in (2,4) else 30;p=x[:,:102].reshape(-1,51,2)
 ratios=[];ok=np.ones(len(x),bool)
 for k in [5,9,13,17]:
  q=p[:,base+k:base+k+4];length=np.linalg.norm(np.diff(q,axis=1),axis=2).sum(1)
  ok&=length>1e-4
  ratios.append(np.linalg.norm(q[:,-1]-q[:,0],axis=1)/np.maximum(length,1e-6))
 r=np.stack(ratios,1)
 # j describes the EXPECTED shape. Test whether the opposite model prediction
 # also has the necessary visible projected structure to justify a review cue.
 return ok&((r<.85).sum(1)>=3) if j in (1,2) else ok&(r>.90).all(1)

def shape_evidence(attrs,x,ts,span,features,late=False):
 out=[]
 if span is None:return out
 a,b=span
 if late:a=a+max(0,(b-a)*2//3)
 for key,expected in features.items():
  j=int(key);available=feature_available(x,j);scope=np.zeros(len(x),bool);scope[a:b+1]=True
  good=available&scope;ratio=float(available[a:b+1].mean());p=attrs[:,j] if expected else 1-attrs[:,j]
  cfg=CONFIG['positive'];th=cfg['threshold'];opp=OPPOSITE.get(j)
  contradict=(p<=1-th)
  if expected and opp is not None:contradict&=attrs[:,opp]>=th
  elif expected:contradict&=False # Low index-only alone does not identify another handshape.
  neural_br=runs(contradict&good,ts,cfg['min_evidence_ms'])
  corroborated=contradict&good&projected_shape_support(x,j)
  br=runs(corroborated,ts,cfg['min_evidence_ms']);sr=runs((p>=th)&good,ts,cfg['min_evidence_ms'])
  mean=float(p[good].mean()) if good.any() else None
  state='unknown'
  conflict=bool(expected and opp is not None and runs((p>=th)&(attrs[:,opp]>=th)&good,ts,cfg['min_evidence_ms']))
  if ratio>=cfg['min_available_fraction']:
   if br:state='contradicted'
   elif sr and mean>=cfg['mean_support_threshold'] and not conflict and not neural_br:state='supported'
  out.append(dict(feature=j,name=NAMES[j],expected=int(expected),state=state,available_fraction=ratio,mean_expected_probability=mean,
                  supported_spans=sr,contradicted_spans=br,neural_contradicted_spans=neural_br,projection_consistency_required=True,
                  opposite_feature=opp,conflicting_predictions=conflict,projection_is_not_independent_truth=True))
 return out

def along_forearm(x,ts,phase,contract):
 cfg=CONFIG['direction'];points=x[:,:102].reshape(-1,51,2);valid=x[:,186:237].astype(bool)
 e,w=(3,5) if contract['anchor_arm']=='L' else (4,6)
 h=9 if contract['moving_hand']=='L' else 30
 # Palm centre, not one fingertip: stable under an open/closed hand intervention.
 palm=[h,h+5,h+9,h+13,h+17];v=points[:,w]-points[:,e];vv=(v*v).sum(1)
 available=valid[:,[e,w]+palm].all(1)&(vv>1e-4)
 rel=points[:,palm].mean(1)-points[:,e]
 u=np.divide((rel*v).sum(1),vv,out=np.zeros(len(x)),where=vv>1e-4)
 groups=np.asarray(contract['phase_groups'])+1
 ids=np.flatnonzero(phase[:,groups].sum(1)>=cfg['phase_support'])
 out=dict(kind='along_forearm',state='unknown',moving_hand=contract['moving_hand'],anchor_arm=contract['anchor_arm'],expected_sign=contract['expected_sign'],
          span=None,contact_verified=False,exact_extent_required=False,trajectory=[],available_fraction=0.)
 if len(ids)<cfg['min_frames']:return out
 a,b=int(ids[0]),int(ids[-1]);scope=np.arange(a,b+1);good=scope[available[scope]];out.update(span=[a,b],available_fraction=float(available[scope].mean()))
 out['trajectory']=[dict(index=int(i),progress=float(u[i]) if available[i] else None) for i in scope]
 if len(good)<cfg['min_frames'] or out['available_fraction']<cfg['min_available_fraction']:return out
 if np.max(np.diff(ts[good]))>cfg['max_gap_ms']*1000:return out
 k=max(2,int(np.ceil(len(scope)*cfg['end_fraction'])))
 first=scope[:k];last=scope[-k:];first=first[available[first]];last=last[available[last]]
 if len(first)<2 or len(last)<2:return out
 start=float(np.median(u[first]));end=float(np.median(u[last]));delta=(end-start)*contract['expected_sign']
 # Remove the median step (actual motion) before estimating jitter. This avoids
 # treating fast normal motion as sensor noise; no smoothing modifies the input.
 steps=np.diff(u[good]);noise=float(np.median(np.abs(steps-np.median(steps))))
 floor=max(cfg['min_displacement_forearm_units'],cfg['noise_multiple']*noise)
 tau=(ts[good]-ts[good[0]])/max(1.,ts[good[-1]]-ts[good[0]])
 trend=float(np.polyfit(tau,u[good],1)[0])*contract['expected_sign']
 state='supported' if delta>floor and trend>0 else 'contradicted' if delta<-floor and trend<0 else 'unknown'
 out.update(state=state,start_progress=start,end_progress=end,signed_change=float(delta),trend=trend,noise=noise,deadband=floor)
 return out

def refine(base,attrs,x,ts,fid,phase):
 result=copy.deepcopy(base);direction_cache={}
 for card in result['cards']:
  previous=card['status'];span=card.get('query_indices');contract=CONTRACTS.get(card['id']);motion=None
  shapes=shape_evidence(attrs,x,ts,span,card['features'],card.get('shape_scope')=='late')
  if contract:
   key=json.dumps(contract,sort_keys=True)
   if key not in direction_cache:direction_cache[key]=along_forearm(x,ts,phase,contract)
   motion=copy.deepcopy(direction_cache[key])
  card.update(previous_status=previous,shape_evidence=shapes,motion_evidence=motion,positive_evidence_required=True)
  contradicted=[s for s in shapes if s['state']=='contradicted']
  shape_ok=bool(shapes) and all(s['state']=='supported' for s in shapes)
  motion_bad=motion is not None and motion['state']=='contradicted'
  reason=[]
  if contradicted:reason.append('handshape')
  if motion_bad:reason.append('direction')
  # Existing missing/ordering warnings are not promoted to semantic certainty.
  # An optional endpoint hold still carries a required stroke-direction contract.
  if reason:
   card['status']='difference_candidate'
   if span is None and motion_bad:span=motion['span']
  elif previous=='supported':
   if (card['features'] and not shape_ok) or (motion is not None and motion['state']!='supported') or (not shapes and motion is None):card['status']='unknown'
  if previous=='unknown' and base.get('long_missing_runs'):
   if span is None or any(not(b<span[0] or a>span[1]) for a,b in base['long_missing_runs']):card['status']='unknown';reason=[]
  card['difference_components']=reason
  card['difference_spans']=[s for q in contradicted for s in q['contradicted_spans']]+([motion['span']] if motion_bad and motion['span'] else [])
  card['checked_components']=[s['name'] for s in shapes if s['state']=='supported']+(['팔을 따른 이동 방향'] if motion and motion['state']=='supported' else [])
  if span is not None:
   card.update(query_indices=span,query_frames=[int(fid[k]) for k in span],query_seconds=[float(ts[k]/1e6) for k in span])
  status=card['status'];card['reference_only']=span is None
  card['display_status']='similar' if status=='supported' else 'difference' if reason and status=='difference_candidate' else 'optional' if status=='optional' else 'uncertain'
  if status=='supported':
   card['message']=' · '.join(card['checked_components'])+' 특징이 비슷하게 관측돼요.'
  elif card['display_status']=='difference':
   phrases=[]
   if contradicted:
    side='오른손' if contradicted[0]['feature'] in (0,1,3) else '왼손'
    expected_open=contradicted[0]['feature'] in (1,2)
    phrases.append(f'{side}이 '+('닫힌 모양으로 보여요. 펴는 부분을 확인해 보세요.' if expected_open else '펴진 모양으로 보여요. 닫는 부분을 확인해 보세요.'))
   if motion_bad:phrases.append('손목에서 팔꿈치 쪽으로 움직인 것으로 보여요. 참고처럼 팔꿈치 쪽에서 손목 쪽으로 쓸어 보세요.' if contract['expected_sign']==1 else '참고와 반대 방향으로 움직인 것으로 보여요. 참고의 화살표 방향을 확인해 보세요.')
   card['message']=' '.join(phrases)
  elif status=='optional':card['message']='마지막에 멈추는 길이는 같지 않아도 돼요. 이동 방향의 확인 결과는 앞 구간에서 확인하세요.'
  elif status=='missing_candidate':card['message']='이 동작을 이어서 했는지 참고와 확인해 주세요.'
  elif status=='order_candidate':card['message']='동작 순서를 참고와 확인해 주세요.'
  else:card['message']='손모양 또는 이동 방향의 근거가 부족해요. 틀렸다는 뜻은 아니에요.'
  card['candidate_message']=card['message'];card['educational_wrong']=False
 result.update(evidence_version='positive_shape_and_signed_direction_v1',educational_correctness_validated=False)
 return result
