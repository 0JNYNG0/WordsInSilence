"""Combine existing learned phase and actual body-role correspondence before ordering."""
from graph import *
from prepare_linker import load
from collections import Counter
import sys,numpy as np,torch
O=U.parent;G=O/'word100_guide_requirement_learning_20261111';R=O/'word100_candidate_role_learning_20261111';D=O/'word100_action_feedback_demo_20261108'
def observed_runs(mask,ts,min_ms):
 out=[];edges=np.diff(np.r_[False,mask,False].astype(int));dt=np.median(np.diff(ts)) if len(ts)>1 else 33333
 for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1):
  if (ts[b]-ts[a]+dt)/1000>=min_ms:out.append([int(a),int(b)])
 return out
def main():
 assert not (U/'JOINT_CORRESPONDENCE_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_joint_corr_req',G/'requirement_core.py');sys.modules['requirement_core']=rc;om=load('_joint_corr_order',G/'order_binding.py');cfg=read(D/'FROZEN.json')['phase'];src={(s['clip_id'],s['kind']):s for s in read(R/'EVALUATION_INPUTS.json')['rows']};saved=read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows'];cache={};results=[]
 write(U/'JOINT_CORRESPONDENCE_FROZEN.json',dict(reason='Phase-only correspondence and body-only replacement each fail; combine actual candidate evidence before selecting, not veto after choosing a wrong interval.',source_phase='unchanged runtime phase network',body_role='unchanged R seed-specific role model',score='sqrt(phase-group probability * actual body-role probability)',threshold=cfg['threshold'],minimum_ms=cfg['min_evidence_ms'],selector='original maximum peak run; expected chronological filter removed',scope='Conditional correspondence ensemble, not independent evidence or certified semantic correctness',all_occurrences_preserved=True,new_training=False,not_fresh_test=True))
 nets={}
 for seed in [42,43,44]:
  n=om.OrderedBinding();n.load_state_dict(torch.load(G/'order_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval();nets[seed]=n
 for s in saved:
  cid,kind=s['clip_id'],s['kind'];seed=s['seed'];meta=src[cid,kind]
  if (cid,kind) not in cache:cache[cid,kind]={k:v for k,v in np.load(meta['path']).items()}
  c=cache[cid,kind];b=read(meta['baseline']);r=read(s['result']);phase=np.asarray(b['phase_probability']);role=np.asarray(r['role_probabilities']);ts=c['ts'];hand=np.asarray(b['observed_hand_mask']);word=cid.split('_')[2];cards=[]
  for spec in rc.FAMILIES[word]:
   card=next(k for k in rc.CARDS[word] if k['id']==spec['id']);f=spec['families'][0];valid=hand[:,[0 if h=='L' else 1 for h in card['hands']]].all(1);groups=[]
   for group in card['groups']:
    score=np.sqrt(phase[:,np.array(group)+1].sum(1)*role[:,f]);runs=observed_runs((score>=cfg['threshold'])&valid,ts,cfg['min_evidence_ms']);span=max(runs,key=lambda e:float(score[e[0]:e[1]+1].max())) if runs else None;groups.append(dict(group=group,selected=span,runs=runs))
   span=[min(e['selected'][0] for e in groups),max(e['selected'][1] for e in groups)] if all(e['selected'] for e in groups) else None
   cards.append(dict(id=card['id'],family=f,span=span,groups=groups,required=spec['required'],parent=spec.get('parent')))
  pairs=[];required=[k for k in cards if k['required'] and not k['parent']]
  for a,bb in zip(required,required[1:]):
   if a['family']==bb['family']:continue
   aa,bs=a['span'],bb['span'];p=None;st='association_unknown'
   if aa is not None and bs is not None:
    lo,hi=sorted([int(round(sum(aa)/2)),int(round(sum(bs)/2))]);ga=next(k['groups'][0][0] for k in rc.CARDS[word] if k['id']==a['id']);gb=next(k['groups'][0][0] for k in rc.CARDS[word] if k['id']==bb['id'])
    with torch.inference_mode():p=float(nets[seed](torch.tensor(rc.observation(c['fh'],lo,hi)[None]),torch.tensor(rc.requirement('order',guides=[ga,gb])[None])).sigmoid()[0])
    st='supported' if aa[1]<bs[0] and p>=.65 else 'reversed' if bs[1]<aa[0] and p<=.35 else 'ambiguous'
   pairs.append(dict(cards=[a['id'],bb['id']],state=st,spans=[aa,bs],order_probability=p))
  results.append(dict(seed=seed,clip_id=cid,kind=kind,cards=cards,pairs=pairs))
 write(U/'JOINT_CORRESPONDENCE_RESULTS.json',results);summary={str(s):{k:dict(Counter(p['state'] for r in results if r['seed']==s and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in results})} for s in [42,43,44]};write(U/'JOINT_CORRESPONDENCE_COMPLETE.json',dict(summary=summary,web_changed=False));print(summary)
if __name__=='__main__':main()
