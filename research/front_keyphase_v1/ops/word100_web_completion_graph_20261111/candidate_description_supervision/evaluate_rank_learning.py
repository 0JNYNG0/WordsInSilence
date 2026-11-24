from ranking_core import *
import time,itertools,traceback
E=K/'evaluation'
def main():
 E.mkdir(exist_ok=True);assert not(E/'COMPARISON.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);prior=read(B/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in prior['predictions']};rm={(r['seed'],r['clip_id'],r['kind']):r for r in prior['rows']};models={}
 for r in read(K/'TRAINING_COMPLETE.json')['rows']:
  assert sha(r['model'])==r['model_sha256'];st=torch.load(r['model'],weights_only=True,map_location='cpu');m=(BoundaryHead if r['condition']=='boundary_extended'else RankHead)(st['mean'].numpy(),st['std'].numpy());m.load_state_dict(st);m.eval();models[r['condition'],r['seed']]=m
 cfg=dict(at=stamp(),selection=read(K/'FROZEN.json')['selection'],primary='rank_extended',same82source_scenarios=True,seeds=[42,43,44],source_boundary_v3_reused=True,body_window_presence_reused=True,rank_score='raw score; frozen body_window presence logit>=0',boundary_score='unchanged start+end+mean logsigmoid presence;mean presence>=.5',no_expected_order=True,semantic_completion=None,pins={str(p):sha(p)for p in[Path(__file__),W/'ranking_core.py',W/'source_boundary_core.py',K/'FROZEN.json',K/'TRAINING_COMPLETE.json',B/'evaluation/COMPARISON.json']});write(E/'FROZEN.json',cfg);rows=[];preds=[];tim=[]
 for cid,kind in dict.fromkeys((r['clip_id'],r['kind'])for r in prior['rows']):
  start=time.perf_counter();r=pm[42,cid,kind];assert sha(r['source'])==r['source_sha256'];z=np.load(r['source']);qr=z['z'][:,292:392].reshape(-1,10,10);x=body_context(qr,z['ts']);old=np.load(r['path']);cc=rm[42,cid,kind]['conditions']['baseline']['cards'];ss=[old[f'spans_{i}']for i in range(len(cc))];union=sorted(set(tuple(s)for sp in ss for s in sp));sp=np.asarray(union,np.int32).reshape(-1,2);features=pool_windows(x,sp);lu={s:i for i,s in enumerate(union)};ids=[np.array([lu[tuple(s)]for s in a],int)for a in ss]
  for seed in cfg['seeds']:
   rr=rm[seed,cid,kind];pred=pm[seed,cid,kind];old=np.load(pred['path']);conditions=dict(rr['conditions']);arrays={}
   with torch.inference_mode():out={m:models[m,seed](torch.tensor(x if m=='boundary_extended'else features)).numpy()for m in['rank_original','rank_extended','boundary_extended']}
   arrays['boundary_logits']=out['boundary_extended']
   for mode in out:
    cards=[]
    for i,c in enumerate(cc):
     j=F.index(c['family']);axis=AX[c['family']]*2;spans=ss[i];base=old[f'window_logits_{i}'];arrays[f'spans_{i}']=spans;arrays[f'presence_{i}']=base[:,0]
     if mode=='boundary_extended':score=rank_boundary(out[mode],spans,j)
     else:
      logits=out[mode][ids[i],j];arrays[f'{mode}_logits_{i}']=logits;score=np.where(base[:,0]>=0,logits,-np.inf)
     chosen=None
     if len(score)and np.isfinite(score.max()):
      at=int(score.argmax());a,b=map(int,spans[at]);obs=np.arange(a,b+1);obs=obs[qr[obs,axis,8]>.5];chosen=dict(span=[a,b],source=mode,body_observed=obs.tolist(),hand_qualified=obs[qr[obs,axis,9]>.5].tolist(),score=float(score[at]),semantic_completion=None)
     card=dict(card=c['card'],family=c['family'],chosen=chosen,semantic_completion=None)
     if 'truth'in c:
      obs=set(chosen['body_observed'])if chosen else set();card['truth']=dict(c['truth'],all_witness=set(c['truth']['witness'])<=obs,wrong_points=len(set(c['truth']['wrong_witness'])&obs))
     cards.append(card)
    order=[]
    for i,j in itertools.combinations(range(len(cards)),2):
     if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
     a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved'if a is None or b is None else'ambiguous_overlap'if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1])else'forward'if a['span'][1]<b['span'][0]else'reverse';order.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
    conditions[mode]=dict(cards=cards,order=order)
   p=E/'predictions'/str(seed)/f'{cid}_{kind}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays);preds.append(dict(pred,path=str(p),sha256=sha(p)));rows.append(dict(seed=seed,clip_id=cid,kind=kind,conditions=conditions));tim.append(dict(seed=seed,clip_id=cid,kind=kind,seconds=time.perf_counter()-start,scope='cached body relations, research exhaustive candidates; not capture/RTMW/UI'))
  print('rank compare',len(rows),flush=True)
 summary=[]
 for seed,mode,kind in itertools.product([42,43,44],cfg['selection']['conditions'],sorted({r['kind']for r in rows})):
  rr=[r for r in rows if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][mode]['cards']];tt=[c['truth']for c in cc if 'truth'in c];oo=[o['state']for r in rr for o in r['conditions'][mode]['order']];summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cc),missing=sum(c['chosen']is None for c in cc),known_groups=len(tt),all_witness=sum(t['all_witness']for t in tt),wrong_points=sum(t['wrong_points']for t in tt),order={k:oo.count(k)for k in set(oo)}))
 write(E/'COMPARISON.json',dict(at=stamp(),rows=rows,predictions=preds,timing=tim,summary=summary,web_changed=False))
 for s in summary:
  if s['condition']=='rank_extended'and s['kind']in['full_normal','dynamic_complete_blocks']:print(s)
if __name__=='__main__':
 try:main()
 except BaseException:E.mkdir(exist_ok=True);(E/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
