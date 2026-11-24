"""Describe supported differences; missing events are not manufactured by DTW."""
from task import *
import numpy as np

def runs(mask,ts,min_ms=60):
 out=[];start=None;dt=float(np.median(np.diff(ts))) if len(ts)>1 else 33333
 for i,v in enumerate(np.r_[mask,False]):
  if v and start is None:start=i
  elif not v and start is not None:
   end=i-1
   if (ts[end]-ts[start]+dt)/1000>=min_ms:out.append((start,end))
   start=None
 return out

def assess(word,phase,d0,d1,raw_x,ts,fid):
 cfg=read(U/'FROZEN.json')['phase'];cards=read(U/'CARDS.json')['words'][word];threshold=cfg['threshold'];n=len(ts)
 hand=np.stack([raw_x[:,237:258].sum(1)>=5,raw_x[:,258:279].sum(1)>=5],1)
 active=phase[:,1:].max(1)>.4;activeids=np.flatnonzero(active)
 active_range=(int(activeids[0]),int(activeids[-1])) if len(activeids) else (0,n-1)
 tail=(ts>=ts[-1]-cfg['rest_tail_ms']*1000);returned=bool(tail.sum()>=3 and np.mean(phase[tail,0]>.65)>=.8)
 result=[];previous_end=-1;some_present=False
 for card in cards:
  sides=[0 if h=='L' else 1 for h in card['hands']];available=hand[:,sides].all(1);lo,hi=active_range;unknown=float((~available[lo:hi+1]).mean())
  evidence=[];unordered=False
  for group in card['groups']:
   prob=phase[:,np.asarray(group)+1].sum(1);rr=runs((prob>=threshold)&available,ts,cfg['min_evidence_ms'])
   choices=[r for r in rr if r[1]>=previous_end]
   if not choices and rr:unordered=True
   chosen=max(choices,key=lambda z:float(prob[z[0]:z[1]+1].max())) if choices else None
   evidence.append(dict(group=group,span=list(chosen) if chosen else None,peak=float(prob.max()),available_runs=[list(r) for r in rr]))
  supported=all(e['span'] for e in evidence)
  if supported and len(evidence)>1:
   centres=[sum(e['span'])/2 for e in evidence];supported=max(centres)-min(centres)>=1
  span=[min(e['span'][0] for e in evidence),max(e['span'][1] for e in evidence)] if supported else None
  status='supported' if supported else 'unknown';message='確認' # replaced below, never a language-correctness score
  missing=not supported and returned and some_present and unknown<=cfg['max_unknown_fraction'] and card.get('required',True)
  if missing:status='missing_candidate'
  elif unordered and unknown<=cfg['max_unknown_fraction']:status='order_candidate'
  if not card.get('required',True) and not supported:status='optional'
  differences=[]
  if supported:
   a,b=span;some_present=True;previous_end=max(previous_end,a)
   if card.get('shape_scope')=='late':a=a+max(0,(b-a)*2//3)
   for feat,expected in card['features'].items():
    j=int(feat);prob=(d1 if expected else d0)[:,j];bad=(prob>=threshold)&available
    bad[:a]=False;bad[b+1:]=False;rr=runs(bad,ts,cfg['min_evidence_ms'])
    if rr:differences.append(dict(feature=j,expected=expected,span=list(rr[0]),peak=float(prob[rr[0][0]:rr[0][1]+1].max())))
   if differences:status='difference_candidate'
  if status=='supported':message='확인한 움직임은 참고와 비슷해요.'
  elif status=='difference_candidate':message=card['instruction']
  elif status=='missing_candidate':message='이 동작까지 이어서 해보세요.'
  elif status=='order_candidate':message='참고의 흐름을 보고 순서대로 이어 보세요.'
  elif status=='optional':message='참고의 마지막 위치예요. 잠깐 멈추는 길이는 같지 않아도 돼요.'
  else:message='이 부분은 확인하기 어려워요. 손이 보이게 다시 해보세요.'
  qspan=([int(fid[span[0]]),int(fid[span[1]])] if span else None)
  result.append(dict(**card,status=status,message=message,query_indices=span,query_frames=qspan,query_seconds=[float(ts[k]/1e6) for k in span] if span else None,phase_evidence=evidence,unknown_fraction=unknown,differences=differences,
    reference_only=status in ['missing_candidate','unknown','order_candidate'],educational_wrong=False))
 # Long missing input must never be presented as proved success or an action omission.
 allhands=np.any(hand,1);missing_runs=runs(~allhands,ts,300)
 if missing_runs:
  for r in result:
   if r['status']!='optional' and (r['query_indices'] is None or any(not (b<r['query_indices'][0] or a>r['query_indices'][1]) for a,b in missing_runs)):
    r.update(status='unknown',message='손이 가려져 확인하기 어려워요. 손이 보이게 다시 해보세요.',reference_only=True)
 return dict(cards=result,return_to_rest_evidence=returned,active_indices=list(active_range),long_missing_runs=[list(x) for x in missing_runs],educational_correctness_validated=False,
  evidence_before_dtw=True,missing_stage_cannot_be_completed_by_alignment=True,observed_hand_mask=hand.tolist(),phase_probability=phase.tolist())
