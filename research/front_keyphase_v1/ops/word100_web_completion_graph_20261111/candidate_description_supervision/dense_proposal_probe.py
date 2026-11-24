"""Fixed diagnostic: proposal recall versus candidate ranking. No new training."""
from core import *
import itertools,time,traceback
D=W/'dense_proposal_probe';V=W/'direction_neutral_candidates'

def pooled(z,spans):
 v=z[:,228:456].astype(np.float64);s=np.vstack([np.zeros(228),v.cumsum(0)]);s2=np.vstack([np.zeros(228),(v*v).cumsum(0)])
 a=spans[:,0];b=spans[:,1]+1;n=b-a;q=n//3;r=n%3;b1=a+q+(r>0);b2=b1+q+(r>1)
 av=lambda x,y:(s[y]-s[x])/(y-x)[:,None]
 var=np.maximum((s2[b]-s2[a])/n[:,None]-av(a,b)**2,0)
 return np.concatenate([av(a,b1),av(b1,b2),av(b2,b),np.sqrt(var)],1).astype(np.float32)

def main():
 assert not (D/'COMPARISON.json').exists();D.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
 old=read(V/'COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in old['predictions']};models={}
 cfg=dict(at=stamp(),scope='Exhaustive visible spans for short public sequences; causal diagnostic, not web adoption',selection=dict(conditions=['baseline','previous_normal_body','dense_candidates']),minimum_samples=4,body_fraction=.8,presence_gate=.5,score='unchanged sum log presence/purity/coverage',seeds=[42,43,44],no_desired_direction_gate=True,no_edit_metadata=True,no_gold_in_selection=True,new_training=False,pins={str(p):sha(p) for p in [Path(__file__),V/'COMPARISON.json',W/'TRAINING_COMPLETE.json']});write(D/'FROZEN.json',cfg)
 for r in read(W/'TRAINING_COMPLETE.json')['rows']:
  assert sha(r['model'])==r['model_sha256'];st=torch.load(r['model'],weights_only=True,map_location='cpu');m=CandidateHead(st['mean'].numpy(),st['std'].numpy());m.load_state_dict(st);m.eval();models[r['seed']]=m
 out=[];preds=[];timing=[];verified_pool=False
 for rec in old['rows']:
  pr=pm[rec['seed'],rec['clip_id'],rec['kind']];assert sha(pr['source'])==pr['source_sha256'];a=np.load(pr['source']);z=a['z'];T=len(z);q=z[:,292:392].reshape(T,10,10);start=time.perf_counter();cards=[];saved=[]
  for c in rec['conditions']['baseline']['cards']:
   k=AX[c['family']]*2;j=F.index(c['family']);valid=q[:,k,8]>.5;cs=np.r_[0,valid.cumsum()];sp=np.array([(l,h) for l in range(T-3) for h in range(l+3,T) if valid[l] and valid[h] and cs[h+1]-cs[l]>=.8*(h-l+1)],dtype=np.int32).reshape(-1,2);scores=[];ps=[]
   for off in range(0,len(sp),512):
    ss=sp[off:off+512];xx=pooled(z,ss)
    if not verified_pool:
     for idx in [0,len(ss)//2,-1]:assert np.allclose(xx[idx],features(z,*ss[idx]),atol=3e-5,rtol=3e-5)
     verified_pool=True
    with torch.inference_mode():pp=models[rec['seed']](torch.tensor(xx)).sigmoid().numpy()[:,j]
    ps.extend(pp);scores.extend(np.where(pp[:,0]>=.5,np.log(np.clip(pp,1e-6,1-1e-6)).sum(1),-np.inf))
   best=None
   if scores and np.isfinite(max(scores)):
    i=int(np.argmax(scores));lo,hi=map(int,sp[i]);ix=np.arange(lo,hi+1);obs=ix[valid[ix]];best=dict(span=[lo,hi],source='dense_visible_span',body_observed=obs.tolist(),hand_qualified=obs[q[obs,k,9]>.5].tolist(),candidate_probability=np.array(ps[i]).tolist(),score=float(scores[i]),semantic_completion=None)
   cc=dict(card=c['card'],family=c['family'],chosen=best,semantic_completion=None)
   if 'truth'in c:
    obs=set(best['body_observed']) if best else set();cc['truth']=dict(c['truth'],all_witness=set(c['truth']['witness'])<=obs,wrong_points=len(set(c['truth']['wrong_witness'])&obs))
   cards.append(cc);saved.append(dict(card=c['card'],family=c['family'],candidate_count=len(sp)))
  orders=[]
  for i,j in itertools.combinations(range(len(cards)),2):
   if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
   a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse';orders.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
  out.append(dict(seed=rec['seed'],clip_id=rec['clip_id'],kind=rec['kind'],conditions=dict(baseline=rec['conditions']['baseline'],previous_normal_body=rec['conditions']['normal_body'],dense_candidates=dict(cards=cards,order=orders)),candidate_counts=saved));preds.append(pr);timing.append(dict(seed=rec['seed'],clip_id=rec['clip_id'],kind=rec['kind'],seconds=time.perf_counter()-start))
  if len(out)%30==0:print('dense',len(out),flush=True)
 summary=[]
 for seed,mode,kind in itertools.product([42,43,44],cfg['selection']['conditions'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][mode]['cards']];tt=[c['truth'] for c in cc if 'truth'in c];orders=[o['state'] for r in rr for o in r['conditions'][mode]['order']];summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cc),missing=sum(c['chosen'] is None for c in cc),known_groups=len(tt),all_witness=sum(t['all_witness'] for t in tt),wrong_points=sum(t['wrong_points'] for t in tt),order={k:orders.count(k) for k in set(orders)}))
 write(D/'COMPARISON.json',dict(at=stamp(),rows=out,summary=summary,predictions=preds,timing=timing,upstream_inference_calls=0,web_changed=False,pooled_verified_against_direct=True))
 for s in summary:
  if s['seed']==42 and s['condition']=='dense_candidates':print(s)
if __name__=='__main__':
 try:main()
 except BaseException:D.mkdir(exist_ok=True);(D/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
