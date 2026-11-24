from source_boundary_core import *
import itertools,time,traceback
E=B/'evaluation'

def main():
 assert not (E/'COMPARISON.json').exists();E.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1);old=read(R/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in old['predictions']};rm={(r['seed'],r['clip_id'],r['kind']):r for r in old['rows']};heads={}
 for run in read(B/'TRAINING_COMPLETE.json')['rows']:
  assert sha(run['model'])==run['model_sha256'];st=torch.load(run['model'],weights_only=True,map_location='cpu');n=(BodyWindow if run['condition']=='body_window' else BoundaryHead)(st['mean'].numpy(),st['std'].numpy());n.load_state_dict(st);n.eval();heads[run['condition'],run['seed']]=n
 cfg=dict(at=stamp(),selection=read(B/'FROZEN.json')['selection'],same82source_scenarios=True,seeds=[42,43,44],body_fraction=.8,min_frames=4,window_score='sum log-sigmoid logits in float64; presence logit>=0',boundary_score='start logit at a + end logit at b + mean log-sigmoid presence inside; mean presence>=.5',no_expected_order=True,all_visible_spans_retained=True,semantic_completion=None,pins={str(p):sha(p) for p in [Path(__file__),W/'source_boundary_core.py',B/'FROZEN.json',B/'TRAINING_COMPLETE.json',R/'evaluation/COMPARISON.json']});write(E/'FROZEN.json',cfg);out=[];preds=[];tim=[]
 for cid,kind in dict.fromkeys((r['clip_id'],r['kind']) for r in old['rows']):
  t=time.perf_counter();src=pm[42,cid,kind];row=rm[42,cid,kind];assert sha(src['source'])==src['source_sha256'];z=np.load(src['source']);qr=z['z'][:,292:392].reshape(-1,10,10);x=body_context(qr,z['ts']);ss=[];T=len(x)
  for c in row['conditions']['baseline']['cards']:
   v=qr[:,AX[c['family']]*2,8]>.5;cs=np.r_[0,v.cumsum()];ss.append(np.asarray([(a,b) for a in range(T-3) for b in range(a+3,T) if v[a] and v[b] and cs[b+1]-cs[a]>=.8*(b-a+1)],np.int32).reshape(-1,2))
  union=sorted(set(tuple(s) for sp in ss for s in sp));spans=np.asarray(union,np.int32).reshape(-1,2);f=pool_windows(x,spans);lookup={s:i for i,s in enumerate(union)};ii=[np.asarray([lookup[tuple(s)] for s in sp],int) for sp in ss]
  for seed in cfg['seeds']:
   original=rm[seed,cid,kind];conditions=dict(baseline=original['conditions']['baseline']);arrays={}
   with torch.inference_mode():wl=heads['body_window',seed](torch.tensor(f)).numpy();bl=heads['direct_boundary',seed](torch.tensor(x)).numpy()
   arrays['boundary_logits']=bl
   for mode in ['body_window','direct_boundary']:
    cards=[]
    for ci,c in enumerate(original['conditions']['baseline']['cards']):
     j=F.index(c['family']);k=AX[c['family']]*2;sp=ss[ci];scores=rank_window(wl[ii[ci],j]) if mode=='body_window' else rank_boundary(bl,sp,j);best=None;arrays[f'spans_{ci}']=sp;arrays[f'window_logits_{ci}']=wl[ii[ci],j]
     if len(scores) and np.isfinite(scores.max()):
      at=int(np.argmax(scores));a,b=map(int,sp[at]);obs=np.arange(a,b+1);obs=obs[qr[obs,k,8]>.5];best=dict(span=[a,b],source=mode,body_observed=obs.tolist(),hand_qualified=obs[qr[obs,k,9]>.5].tolist(),score=float(scores[at]),semantic_completion=None)
     card=dict(card=c['card'],family=c['family'],chosen=best,semantic_completion=None)
     if 'truth'in c:
      obs=set(best['body_observed']) if best else set();card['truth']=dict(c['truth'],all_witness=set(c['truth']['witness'])<=obs,wrong_points=len(set(c['truth']['wrong_witness'])&obs))
     cards.append(card)
    orders=[]
    for i,j in itertools.combinations(range(len(cards)),2):
     if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
     a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse';orders.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
    conditions[mode]=dict(cards=cards,order=orders)
   p=E/'predictions'/str(seed)/f'{cid}_{kind}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays);preds.append(dict(pm[seed,cid,kind],path=str(p),sha256=sha(p)));out.append(dict(seed=seed,clip_id=cid,kind=kind,conditions=conditions));tim.append(dict(seed=seed,clip_id=cid,kind=kind,seconds=time.perf_counter()-t,scope='cached relation features -> two research heads/exhaustive candidates, no RTMW/camera/UI'))
  print('source boundary compare',len(out),flush=True)
 summary=[]
 for seed,mode,kind in itertools.product([42,43,44],cfg['selection']['conditions'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][mode]['cards']];tt=[c['truth'] for c in cc if 'truth'in c];oo=[o['state'] for r in rr for o in r['conditions'][mode]['order']];summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cc),missing=sum(c['chosen'] is None for c in cc),known_groups=len(tt),all_witness=sum(t['all_witness'] for t in tt),wrong_points=sum(t['wrong_points'] for t in tt),order={k:oo.count(k) for k in set(oo)}))
 write(E/'COMPARISON.json',dict(at=stamp(),rows=out,predictions=preds,timing=tim,summary=summary,web_changed=False,source_cache_reused=True))
 for s in summary:
  if s['condition']=='direct_boundary' and s['kind']in['full_normal','dynamic_complete_blocks']:print(s)
if __name__=='__main__':
 try:main()
 except BaseException:E.mkdir(exist_ok=True);(E/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
