"""Arithmetic-only ablation: do not discard distinctions between near-one scores."""
from framing_boundary_core import *
import copy,itertools
V=R/'numeric_ranking';E=R/'evaluation'
def score(p):
 with np.errstate(divide='ignore'):s=np.log(p.astype(np.float64)).sum(1)
 return np.where(p[:,0]>=.5,s,-np.inf)
def main():
 assert not (V/'COMPARISON.json').exists();assert read(V/'FROZEN.json')['primary']=='crop_boundary';src=read(E/'COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in src['predictions']};out=[];preds=[];audit=[]
 # Mathematical unit case: the old clipping flattened these two distinct values.
 fixture=np.array([[1-5e-7,1,1],[1-1e-7,1,1]],np.float32);oldscore=np.log(np.clip(fixture,1e-6,1-1e-6)).sum(1);assert oldscore[0]==oldscore[1] and score(fixture)[1]>score(fixture)[0]
 for r in src['rows']:
  saved=pm[r['seed'],r['clip_id'],r['kind']];assert sha(saved['path'])==saved['sha256'];p=np.load(saved['path']);raw=np.load(saved['source']);q=raw['z'][:,292:392].reshape(-1,10,10);cards=[]
  for i,c in enumerate(r['conditions']['crop_boundary']['cards']):
   cc=copy.deepcopy(c);pp=p[f'crop_boundary_p_{i}'];sp=p[f'crop_boundary_spans_{i}'];s=score(pp);old_s=np.where(pp[:,0]>=.5,np.log(np.clip(pp,1e-6,1-1e-6)).sum(1),-np.inf);best=None
   if len(s) and np.isfinite(s.max()):
    k=int(np.argmax(s));a,b=map(int,sp[k]);axis=AX[c['family']]*2;obs=np.arange(a,b+1);obs=obs[q[obs,axis,8]>.5];best=dict(span=[a,b],source='saved_probability_float64_log',body_observed=obs.tolist(),hand_qualified=obs[q[obs,axis,9]>.5].tolist(),candidate_probability=pp[k].tolist(),score=float(s[k]),semantic_completion=None)
   cc['chosen']=best
   if 'truth'in cc:
    obs=set(best['body_observed']) if best else set();cc['truth']['all_witness']=set(cc['truth']['witness'])<=obs;cc['truth']['wrong_points']=len(set(cc['truth']['wrong_witness'])&obs)
   cards.append(cc);audit.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],old_top_ties=int((old_s==old_s.max()).sum()) if len(s) and np.isfinite(old_s.max()) else 0,new_top_ties=int((s==s.max()).sum()) if len(s) and np.isfinite(s.max()) else 0,changed=bool(c['chosen'] and best and c['chosen']['span']!=best['span']),new_unrankable=bool(c['chosen'] is not None and best is None)))
  orders=[]
  for i,j in itertools.combinations(range(len(cards)),2):
   if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
   a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse';orders.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
  out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=dict(original_clipped=r['conditions']['crop_boundary'],unclipped_probability=dict(cards=cards,order=orders))));preds.append(saved)
 summary=[]
 for seed,mode,kind in itertools.product([42,43,44],['original_clipped','unclipped_probability'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][mode]['cards']];tt=[c['truth'] for c in cc if 'truth'in c];oo=[o['state'] for r in rr for o in r['conditions'][mode]['order']];summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cc),missing=sum(c['chosen'] is None for c in cc),known_groups=len(tt),all_witness=sum(t['all_witness'] for t in tt),wrong_points=sum(t['wrong_points'] for t in tt),order={k:oo.count(k) for k in set(oo)}))
 write(V/'COMPARISON.json',dict(at=stamp(),rows=out,predictions=preds,summary=summary,tie_audit=audit,new_inference=False,new_training=False,web_changed=False));write(V/'SHADOW_CONTRACT.json',dict(selection=dict(conditions=['original_clipped','unclipped_probability'])))
 for s in summary:
  if s['condition']=='unclipped_probability' and s['kind']in['full_normal','dynamic_complete_blocks']:print(s)
if __name__=='__main__':main()
