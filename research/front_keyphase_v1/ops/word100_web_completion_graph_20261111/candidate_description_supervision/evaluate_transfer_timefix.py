from transfer_local_timefix_core import *
import itertools,time,traceback
E=TQ/'evaluation'

def main():
 assert not (E/'COMPARISON.json').exists();E.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
 old=read(Q/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in old['predictions']};base=encoder();heads={}
 for run in read(TQ/'TRAINING_COMPLETE.json')['rows']:
  assert sha(run['model'])==run['model_sha256'];st=torch.load(run['model'],weights_only=True,map_location='cpu');n=TransferHead(st['mean'].numpy(),st['std'].numpy());n.load_state_dict(st);n.eval();heads[run['seed']]=n
 cfg=dict(at=stamp(),selection=read(TQ/'FROZEN.json')['selection'],scope='Same visible exhaustive spans, candidate-local common100 encoding; no global word context or expected direction gate',seeds=[42,43,44],body_fraction=.8,min_frames=4,presence_gate=.5,score='unchanged sum log three heads',gold_only_for_diagnostics=True,pins={str(p):sha(p) for p in [Path(__file__),TQ/'FROZEN.json',TQ/'TRAINING_COMPLETE.json',Q/'evaluation/COMPARISON.json']});write(E/'FROZEN.json',cfg)
 out=[];preds=[];tim=[];features={}
 for row in old['rows']:
  key=row['seed'],row['clip_id'],row['kind'];src=pm[key];case=key[1:];t=time.perf_counter()
  if case not in features:
   assert sha(src['source'])==src['source_sha256'];z=np.load(src['source']);pose=normalized_pose(z['x']);q=z['z'][:,292:392];qr=q.reshape(-1,10,10);T=len(q);ss=[]
   for c in row['conditions']['baseline']['cards']:
    v=qr[:,AX[c['family']]*2,8]>.5;cs=np.r_[0,v.cumsum()];ss.append(np.asarray([(a,b) for a in range(T-3) for b in range(a+3,T) if v[a] and v[b] and cs[b+1]-cs[a]>=.8*(b-a+1)],np.int32).reshape(-1,2))
   union=sorted(set(tuple(s) for sp in ss for s in sp));spans=np.asarray(union,np.int32).reshape(-1,2);xf=[]
   for off in range(0,len(spans),128):xf.extend(encode_windows(base,pose,q,spans[off:off+128]))
   f=np.asarray(xf,np.float32).reshape(-1,1160);lookup={s:i for i,s in enumerate(union)};ii=[np.asarray([lookup[tuple(s)] for s in sp],int) for sp in ss]
   features[case]=dict(f=f,ss=ss,ii=ii,q=qr,encode_seconds=time.perf_counter()-t)
  cache=features[case];cards=[];arrays={};counts=[]
  with torch.inference_mode():allp=heads[row['seed']](torch.tensor(cache['f'])).sigmoid().numpy()
  for ci,c in enumerate(row['conditions']['baseline']['cards']):
   j=F.index(c['family']);k=AX[c['family']]*2;sp=cache['ss'][ci];pp=allp[cache['ii'][ci],j];valid=cache['q'][:,k,8]>.5;scores=np.where(pp[:,0]>=.5,np.log(np.clip(pp,1e-6,1-1e-6)).sum(1),-np.inf);best=None
   if len(scores) and np.isfinite(scores.max()):
    at=int(np.argmax(scores));a,b=map(int,sp[at]);obs=np.arange(a,b+1);obs=obs[valid[obs]];best=dict(span=[a,b],source='dense_local_pretrained',body_observed=obs.tolist(),hand_qualified=obs[cache['q'][obs,k,9]>.5].tolist(),candidate_probability=pp[at].tolist(),score=float(scores[at]),semantic_completion=None)
   card=dict(card=c['card'],family=c['family'],chosen=best,semantic_completion=None)
   if 'truth'in c:
    obs=set(best['body_observed']) if best else set();card['truth']=dict(c['truth'],all_witness=set(c['truth']['witness'])<=obs,wrong_points=len(set(c['truth']['wrong_witness'])&obs))
   cards.append(card);arrays[f'spans_{ci}']=sp;arrays[f'p_{ci}']=pp;counts.append(dict(card=c['card'],candidate_count=len(sp)))
  orders=[]
  for i,j in itertools.combinations(range(len(cards)),2):
   if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
   a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse';orders.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
  p=E/'predictions'/str(row['seed'])/f"{row['clip_id']}_{row['kind']}.npz";p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays);preds.append(dict(src,path=str(p),sha256=sha(p)));out.append(dict(seed=row['seed'],clip_id=row['clip_id'],kind=row['kind'],conditions=dict(baseline=row['conditions']['baseline'],local_from_scratch=row['conditions']['local_sequence'],local_pretrained=dict(cards=cards,order=orders)),candidate_counts=counts));tim.append(dict(seed=row['seed'],clip_id=row['clip_id'],kind=row['kind'],wall_seconds=time.perf_counter()-t,source_encode_seconds=cache['encode_seconds']))
  if len(out)%12==0:print('transfer eval',len(out),flush=True)
 summary=[]
 for seed,mode,kind in itertools.product([42,43,44],cfg['selection']['conditions'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][mode]['cards']];tt=[c['truth'] for c in cc if 'truth'in c];oo=[o['state'] for r in rr for o in r['conditions'][mode]['order']];summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cc),missing=sum(c['chosen'] is None for c in cc),known_groups=len(tt),all_witness=sum(t['all_witness'] for t in tt),wrong_points=sum(t['wrong_points'] for t in tt),order={k:oo.count(k) for k in set(oo)}))
 write(E/'COMPARISON.json',dict(at=stamp(),rows=out,predictions=preds,timing=tim,summary=summary,source_encoder_calls=len(features),web_changed=False,existing_full_clip_results_reexecuted=False))
 for s in summary:
  if s['condition']=='local_pretrained' and s['kind']in['full_normal','dynamic_complete_blocks']:print(s)
if __name__=='__main__':
 try:main()
 except BaseException:E.mkdir(exist_ok=True);(E/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
