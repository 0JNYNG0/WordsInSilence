from core import *
import itertools,time,traceback
from collections import defaultdict
V=W/'direction_neutral_candidates';B=U/'episode_stage_contrast'
def main():
 assert not (V/'COMPARISON.json').exists();cfg=read(V/'FROZEN.json');assert all(sha(p)==h for p,h in cfg['pins'].items());torch.set_num_threads(4);torch.set_num_interop_threads(1);base=defaultdict(list)
 for r in read(B/'BINDING_V2_SCORES.json')['rows']:base[r['seed'],r['clip_id'],r['kind']].append(r)
 oldpred={(r['seed'],r['clip_id'],r['kind']):r for r in read(W/'COMPARISON.json')['predictions']};models={};out=[];predictions=[];timing=[];reused=0;new_normal=0
 for folder,name in [(W,'normal_body'),(V,'neutral_body')]:
  for r in read(folder/'TRAINING_COMPLETE.json')['rows']:
   assert sha(r['model'])==r['model_sha256'];st=torch.load(r['model'],weights_only=True,map_location='cpu');net=CandidateHead(st['mean'].numpy(),st['std'].numpy());net.load_state_dict(st);net.eval();models[r['seed'],name]=net
 for e in read(B/'EVALUATION_COMPLETE.json')['rows']:
  key=e['seed'],e['clip_id'],e['kind'];rs=base[key];assert sha(e['source'])==e['source_sha256'];z=np.load(e['source'])['z'];op=oldpred[key];assert sha(op['path'])==op['sha256'];old=np.load(op['path']);lookup={}
  for (ri,ci,j),p in zip(old['spec'],old['p']):
   c=[v for v in rs[ri]['candidates'] if v['has_motion']][ci];lookup[ri,tuple(c['span'])]=p[j]
  started=time.perf_counter();cs=[];xx=[];spec=[]
  for ri,r in enumerate(rs):
   temp=[]
   for c in r['candidates']:
    if len(c['body_observed'])<3 or len(c['body_observed'])/(c['span'][1]-c['span'][0]+1)<.8:continue
    temp.append(dict(c));xx.append(features(z,*c['span']));spec.append((ri,len(temp)-1,F.index(r['family'])))
   cs.append(temp)
  x=np.array(xx);normal=np.zeros((len(xx),3),np.float32);missing=[]
  for k,(ri,ci,j) in enumerate(spec):
   value=lookup.get((ri,tuple(cs[ri][ci]['span'])))
   if value is None:missing.append(k)
   else:normal[k]=value;reused+=1
  with torch.inference_mode():
   if missing:
    pp=models[e['seed'],'normal_body'](torch.tensor(x[missing])).sigmoid().numpy()
    for k,p in zip(missing,pp):normal[k]=p[spec[k][2]]
    new_normal+=len(missing)
   pp=models[e['seed'],'neutral_body'](torch.tensor(x)).sigmoid().numpy() if len(x) else np.empty((0,5,3))
  neutral=np.array([p[j] for p,(_,_,j) in zip(pp,spec)],np.float32).reshape(-1,3)
  for k,(ri,ci,j) in enumerate(spec):cs[ri][ci]['normal_body']=normal[k].tolist();cs[ri][ci]['neutral_body']=neutral[k].tolist()
  modes={}
  for mode in cfg['selection']['conditions']:
   cards=[]
   for r,possible in zip(rs,cs):
    if mode=='baseline':chosen=r['chosen']
    else:
     eligible=[c for c in possible if c[mode][0]>=.5];chosen=max(eligible,key=lambda c:np.log(np.clip(c[mode],1e-6,1-1e-6)).sum()) if eligible else None
    card=dict(card=r['card'],family=r['family'],chosen=chosen,semantic_completion=None)
    if 'truth' in r:
     t=r['truth'];obs=set(chosen['body_observed']) if chosen else set();card['truth']=dict(witness=t['witness'],wrong_witness=t['wrong_witness'],all_witness=set(t['witness'])<=obs,wrong_points=len(set(t['wrong_witness'])&obs))
    cards.append(card)
   order=[]
   for i,j in itertools.combinations(range(len(cards)),2):
    if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
    a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse';order.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
   modes[mode]=dict(cards=cards,order=order)
  dst=V/'predictions'/str(e['seed'])/f"{e['clip_id']}_{e['kind']}.npz";dst.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst,normal=normal,neutral=neutral,spec=np.array(spec,dtype=np.int32).reshape(-1,3));predictions.append(dict(e,path=str(dst),sha256=sha(dst)));out.append(dict(seed=e['seed'],clip_id=e['clip_id'],kind=e['kind'],conditions=modes,candidates=cs));timing.append(dict(seed=e['seed'],clip_id=e['clip_id'],kind=e['kind'],seconds=time.perf_counter()-started))
 summary=[]
 for seed,mode,kind in itertools.product(cfg['seeds'],cfg['selection']['conditions'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cards=[c for r in rr for c in r['conditions'][mode]['cards']];tt=[c['truth'] for c in cards if 'truth'in c];orders=[o['state'] for r in rr for o in r['conditions'][mode]['order']];summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cards),missing=sum(c['chosen'] is None for c in cards),known_groups=len(tt),all_witness=sum(t['all_witness'] for t in tt),wrong_points=sum(t['wrong_points'] for t in tt),order={k:orders.count(k) for k in set(orders)},hand_qualified_points=sum(len(c['chosen']['hand_qualified']) if c['chosen'] else 0 for c in cards)))
 write(V/'COMPARISON.json',dict(at=stamp(),rows=out,summary=summary,predictions=predictions,timing=timing,old_candidate_scores_reused=reused,new_normal_head_inputs=new_normal,upstream_inference_calls=0,web_changed=False))
 for s in summary:
  if s['seed']==42 and s['kind'] in ['full_normal','dynamic_complete_blocks','main_reversed','closed_hand_and_reversed']:print(s)
if __name__=='__main__':
 try:main()
 except BaseException:(V/f'COMPARE_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
