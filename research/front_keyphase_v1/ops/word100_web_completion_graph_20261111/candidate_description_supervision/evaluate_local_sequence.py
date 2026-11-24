from local_sequence_core import *
import itertools,time,traceback
E=Q/'evaluation'

def main():
 assert not (E/'COMPARISON.json').exists();E.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1);old=read(N/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in old['predictions']};norm=read(O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json');mean=np.array(norm['mean'])[102:186];std=np.array(norm['std'])[102:186];heads={}
 for run in read(Q/'TRAINING_COMPLETE.json')['rows']:
  assert sha(run['model'])==run['model_sha256'];net=LocalCandidate();net.load_state_dict(torch.load(run['model'],weights_only=True,map_location='cpu'));net.eval();heads[run['seed']]=net
 cfg=dict(at=stamp(),scope='Same exhaustive body-visible spans as pooled baseline, but local candidate sequence encoder. All actual window samples retained.',selection=dict(conditions=['baseline','previous_window_pool','local_sequence']),seeds=[42,43,44],body_fraction=.8,min_frames=4,presence_gate=.5,score='unchanged sum log three heads',no_source_gold_or_edit_metadata_in_selection=True,new_upstream_inference=False,pins={str(p):sha(p) for p in [Path(__file__),Q/'TRAINING_COMPLETE.json',Q/'FROZEN.json',N/'evaluation/COMPARISON.json']});write(E/'FROZEN.json',cfg)
 out=[];preds=[];timing=[]
 for row in old['rows']:
  key=row['seed'],row['clip_id'],row['kind'];src=pm[key];assert sha(src['source'])==src['source_sha256'];z=np.load(src['source']);x=z['x'].copy();mask=np.repeat(x[:,237:279]>.5,2,1);x[:,102:186]=np.where(mask,(x[:,102:186]-mean)/std,0);seq=local_input(z['z'],x);q=z['z'][:,292:392].reshape(-1,10,10);T=len(seq);t=time.perf_counter();cards=[];arrays={};counts=[]
  for ci,c in enumerate(row['conditions']['baseline']['cards']):
   k=AX[c['family']]*2;j=F.index(c['family']);valid=q[:,k,8]>.5;cs=np.r_[0,valid.cumsum()];sp=np.array([(a,b) for a in range(T-3) for b in range(a+3,T) if valid[a] and valid[b] and cs[b+1]-cs[a]>=.8*(b-a+1)],np.int32).reshape(-1,2);pp=[]
   for off in range(0,len(sp),128):
    xb,lens=batch([seq[a:b+1] for a,b in sp[off:off+128]])
    with torch.inference_mode():pp.extend(heads[row['seed']](xb,lens).sigmoid().numpy()[:,j])
   pp=np.array(pp).reshape(-1,3);scores=np.where(pp[:,0]>=.5,np.log(np.clip(pp,1e-6,1-1e-6)).sum(1),-np.inf) if len(pp) else [];best=None
   if len(scores) and np.isfinite(np.max(scores)):
    ind=int(np.argmax(scores));a,b=map(int,sp[ind]);ii=np.arange(a,b+1);obs=ii[valid[ii]];best=dict(span=[a,b],source='dense_local_sequence',body_observed=obs.tolist(),hand_qualified=obs[q[obs,k,9]>.5].tolist(),candidate_probability=pp[ind].tolist(),score=float(scores[ind]),semantic_completion=None)
   card=dict(card=c['card'],family=c['family'],chosen=best,semantic_completion=None)
   if 'truth'in c:
    obs=set(best['body_observed']) if best else set();card['truth']=dict(c['truth'],all_witness=set(c['truth']['witness'])<=obs,wrong_points=len(set(c['truth']['wrong_witness'])&obs))
   cards.append(card);arrays[f'spans_{ci}']=sp;arrays[f'p_{ci}']=pp;counts.append(dict(card=c['card'],candidate_count=len(sp)))
  orders=[]
  for i,j in itertools.combinations(range(len(cards)),2):
   if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
   a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse';orders.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
  p=E/'predictions'/str(row['seed'])/f"{row['clip_id']}_{row['kind']}.npz";p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays);preds.append(dict(src,path=str(p),sha256=sha(p)));out.append(dict(seed=row['seed'],clip_id=row['clip_id'],kind=row['kind'],conditions=dict(baseline=row['conditions']['baseline'],previous_window_pool=row['conditions']['dense_candidates'],local_sequence=dict(cards=cards,order=orders)),candidate_counts=counts));timing.append(dict(seed=row['seed'],clip_id=row['clip_id'],kind=row['kind'],seconds=time.perf_counter()-t))
  if len(out)%12==0:print('local eval',len(out),flush=True)
 summary=[]
 for seed,mode,kind in itertools.product([42,43,44],cfg['selection']['conditions'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][mode]['cards']];tt=[c['truth'] for c in cc if 'truth'in c];oo=[o['state'] for r in rr for o in r['conditions'][mode]['order']];summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cc),missing=sum(c['chosen'] is None for c in cc),known_groups=len(tt),all_witness=sum(t['all_witness'] for t in tt),wrong_points=sum(t['wrong_points'] for t in tt),order={k:oo.count(k) for k in set(oo)}))
 write(E/'COMPARISON.json',dict(at=stamp(),rows=out,predictions=preds,timing=timing,summary=summary,upstream_inference_calls=0,web_changed=False))
 for s in summary:
  if s['condition']=='local_sequence' and s['kind']in ['full_normal','dynamic_complete_blocks']:print(s)
if __name__=='__main__':
 try:main()
 except BaseException:E.mkdir(exist_ok=True);(E/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
