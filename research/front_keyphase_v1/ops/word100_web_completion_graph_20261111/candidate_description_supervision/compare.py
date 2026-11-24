from core import *
from collections import defaultdict
import itertools,time,traceback
B=U/'episode_stage_contrast';H=U/'hierarchical_relation_learning/parent_consistency_repair'
def main():
 assert not (W/'COMPARISON.json').exists();cfg=read(W/'FROZEN.json');assert all(sha(p)==h for p,h in cfg['pins'].items());torch.set_num_threads(4);torch.set_num_interop_threads(1)
 base=read(B/'BINDING_V2_SCORES.json')['rows'];groups=defaultdict(list)
 for r in base:groups[r['seed'],r['clip_id'],r['kind']].append(r)
 hp={(r['seed'],r['clip_id'],r['kind']):r for r in read(H/'COMPARISON.json')['predictions']};models={};out=[];predictions=[];timing=[]
 for t in read(W/'TRAINING_COMPLETE.json')['rows']:
  assert sha(t['model'])==t['model_sha256'];st=torch.load(t['model'],map_location='cpu',weights_only=True);n=CandidateHead(st['mean'].numpy(),st['std'].numpy());n.load_state_dict(st);n.eval();models[t['seed']]=n
 for e in read(B/'EVALUATION_COMPLETE.json')['rows']:
  key=e['seed'],e['clip_id'],e['kind'];rs=groups[key];assert sha(e['source'])==e['source_sha256'] and sha(e['path'])==e['sha256'];z=np.load(e['source'])['z'];stage=np.load(e['path'])['p'];hr=hp[key];assert sha(hr['path'])==hr['sha256'];hd=np.load(hr['path']);lookup={tuple(ab):float(p[0]) for ab,p in zip(hd['pairs'],hd['p'])};start=time.perf_counter()
  candidates=[];spec=[];xx=[]
  for ri,r in enumerate(rs):
   cs=[]
   for ci,c in enumerate(r['candidates']):
    if not c['has_motion']:continue
    ix=[i for i in c['body_observed'] if stage[i,F.index(r['family']),1]>=.5];reps=sorted(set(ix[k] for k in [0,len(ix)//2,len(ix)-1])) if ix else []
    cs.append(dict(c,index=ci,representatives=reps));xx.append(features(z,*c['span']));spec.append((ri,len(cs)-1,F.index(r['family'])))
   candidates.append(cs)
  with torch.inference_mode():p=models[e['seed']](torch.tensor(np.array(xx))).sigmoid().numpy() if xx else np.empty((0,5,3),np.float32)
  for (ri,ci,j),pp in zip(spec,p):candidates[ri][ci]['probabilities']=pp[j].tolist()
  linked=[(i,j) for i,j in itertools.combinations(range(len(rs)),2) if sorted([rs[i]['family'],rs[j]['family']]) in cfg['selection']['relation_contract']]
  def cross(a,b):
   vals=[]
   for i,j in itertools.product(a['representatives'],b['representatives']):
    same=1. if i==j else lookup[tuple(sorted([i,j]))];vals.append(np.log(np.clip(1-same,1e-6,1-1e-6)))
   return float(np.mean(vals)) if vals else 0.
  result={}
  for cond in cfg['selection']['conditions']:
   best=None
   for combo in itertools.product(*[cs or [None] for cs in candidates]):
    score=0.
    for c in combo:
     if c is None:continue
     score+=c['score'] if cond=='baseline' else float(np.log(np.clip(c['probabilities'][:2 if cond=='presence_purity' else 3],1e-6,1-1e-6)).sum())
    if cond=='coverage_joint':score+=sum(cross(combo[i],combo[j]) for i,j in linked if combo[i] and combo[j])
    if best is None or score>best[0]:best=(score,combo)
   selected=best[1];cards=[]
   for r,c in zip(rs,selected):
    if cond=='baseline':assert (c['span'] if c else None)==(r['chosen']['span'] if r['chosen'] else None)
    row=dict(card=r['card'],family=r['family'],chosen=c,semantic_completion=None)
    if 'truth' in r:
     t=r['truth'];observed=set(c['body_observed']) if c else set();row['truth']=dict(witness=t['witness'],wrong_witness=t['wrong_witness'],all_witness=set(t['witness'])<=observed,wrong_points=len(set(t['wrong_witness'])&observed))
    cards.append(row)
   orders=[]
   for i,j in linked:
    a,b=selected[i],selected[j];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse';orders.append(dict(cards=[rs[i]['card'],rs[j]['card']],state=state,semantic_completion=None))
   result[cond]=dict(score=best[0],cards=cards,order=orders)
  elapsed=time.perf_counter()-start;dst=W/'predictions'/str(e['seed'])/f"{e['clip_id']}_{e['kind']}.npz";dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();np.savez_compressed(dst,p=p,spec=np.array(spec,dtype=np.int32).reshape(-1,3))
  predictions.append(dict(seed=e['seed'],clip_id=e['clip_id'],kind=e['kind'],path=str(dst),sha256=sha(dst),source=e['source'],source_sha256=e['source_sha256'],stage=e['path'],stage_sha256=e['sha256'],parent=hr['path'],parent_sha256=hr['sha256']));timing.append(dict(seed=e['seed'],clip_id=e['clip_id'],kind=e['kind'],seconds=elapsed,candidates=len(spec)));out.append(dict(seed=e['seed'],clip_id=e['clip_id'],kind=e['kind'],conditions=result))
  if len(out)%60==0:print('C evaluated',len(out),flush=True)
 summary=[]
 for seed,cond,kind in itertools.product(cfg['seeds'],cfg['selection']['conditions'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cards=[c for r in rr for c in r['conditions'][cond]['cards']];truth=[c['truth'] for c in cards if 'truth' in c];order=[o['state'] for r in rr for o in r['conditions'][cond]['order']];summary.append(dict(seed=seed,condition=cond,kind=kind,cases=len(rr),cards=len(cards),missing=sum(c['chosen'] is None for c in cards),known_groups=len(truth),all_witness=sum(t['all_witness'] for t in truth),wrong_points=sum(t['wrong_points'] for t in truth),order={k:order.count(k) for k in set(order)},hand_qualified_points=sum(len(c['chosen']['hand_qualified']) if c['chosen'] else 0 for c in cards)))
 assert all(s['hand_qualified_points']==0 for s in summary if s['kind']=='all_hand_missing')
 write(W/'COMPARISON.json',dict(at=stamp(),rows=out,summary=summary,predictions=predictions,timing=timing,source_cases=82,upstream_calls=0,parent_model_calls=0,web_changed=False,code_sha256=sha(__file__),limitations=['Exposed development and synthetic transformations, not new-user accuracy.','Source witness coverage is not semantic completion.','Some head cells have few positive/negative source clips.','Shape/direction downstream regression still needed even if candidate binding improves.','Cached candidate feature/head/selection timing only; capture/RTMW/end-to-end excluded.']))
 for s in summary:
  if s['kind'] in ['full_normal','dynamic_complete_blocks']:print(s,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:(W/f'COMPARE_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
