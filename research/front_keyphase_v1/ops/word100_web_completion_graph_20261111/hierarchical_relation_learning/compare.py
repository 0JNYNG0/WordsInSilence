from core import *
from collections import defaultdict
import itertools,time,traceback
B=U/'episode_stage_contrast'

def freeze():
 assert not (W/'C_FROZEN.json').exists()
 cfg=read(W/'FROZEN_V2.json');witness=defaultdict(list);train_people=read(W/'PRETRAIN_VERIFICATION.json')['people']['train']
 for r in read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows']:
  if r['person'] not in train_people:continue
  for a,b in itertools.combinations(r['groups'],2):
   if a['family']!=b['family'] and a['group_id']!=b['group_id']:witness[tuple(sorted([a['family'],b['family']]))].append(r['clip_id'])
 relations=[dict(families=list(k),source_clips=sorted(set(v)),relation='distinct_parent_occurrences',order_constraint=False) for k,v in witness.items() if len(set(v))>=3]
 assert relations
 write(W/'C_CONTRACT_REPAIR.json',dict(at=stamp(),original_contract=cfg['linkage']['relation_contract'],repaired=relations,reason='Initial compiler required non-touching uncertainty bands and emitted no pairs. Distinct reviewed group membership is the actual target, not exact disjoint boundaries. This repair uses FIT source group IDs, never evaluation witnesses; no hard overlap ban. Training unchanged; first C evaluation not yet run.'))
 cfg['linkage']['relation_contract']=relations;cfg['at']=stamp();cfg['pins'].update({str(p):sha(p) for p in [Path(__file__),W/'C_CONTRACT_REPAIR.json',W/'TRAINING_COMPLETE.json',U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json']});write(W/'C_FROZEN.json',cfg)

def representatives(c,p,family):
 ix=[i for i in c['body_observed'] if p[i,F.index(family),1]>=.5]
 return sorted(set(ix[k] for k in [0,len(ix)//2,len(ix)-1])) if ix else []

def compare():
 assert not (W/'COMPARISON.json').exists();cfg=read(W/'C_FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 base=read(B/'BINDING_V2_SCORES.json')['rows'];groups=defaultdict(list)
 for r in base:groups[r['seed'],r['clip_id'],r['kind']].append(r)
 es=read(B/'EVALUATION_COMPLETE.json')['rows'];models={};out=[];predictions=[];bench=[];contracts={tuple(r['families']) for r in cfg['linkage']['relation_contract']}
 for seed in cfg['seeds']:
  state=torch.load(W/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu');net=RelationHead(state['mean'].numpy(),state['std'].numpy());net.load_state_dict(state);net.eval();models[seed]=net
 for e in es:
  assert sha(e['path'])==e['sha256'] and sha(e['source'])==e['source_sha256'];rs=groups[e['seed'],e['clip_id'],e['kind']];src=np.load(e['source']);z=src['z'];p=np.load(e['path'])['p'];start=time.perf_counter();cands=[];pairs=set()
  for r in rs:
   cs=[]
   for idx,c in enumerate(r['candidates']):
    if not c['has_motion']:continue
    cc=dict(c,index=idx,representatives=representatives(c,p,r['family']));cs.append(cc)
    pairs.update(itertools.combinations(cc['representatives'],2))
   cands.append(cs)
  linked=[]
  for i,j in itertools.combinations(range(len(rs)),2):
   if tuple(sorted([rs[i]['family'],rs[j]['family']])) not in contracts:continue
   linked.append([i,j])
   for a,b in itertools.product(cands[i],cands[j]):
    pairs.update(tuple(sorted([x,y])) for x,y in itertools.product(a['representatives'],b['representatives']) if x!=y)
  pairs=sorted(pairs);ps=np.empty((0,3),np.float32)
  if pairs:
   x=torch.tensor(np.stack([pair_features(z,a,b) for a,b in pairs]))
   with torch.inference_mode():ps=models[e['seed']].probabilities(x).numpy()
  lookup={pair:float(v[0]) for pair,v in zip(pairs,ps)}
  def same(a,b):return 1. if a==b else lookup[tuple(sorted([a,b]))]
  for cs in cands:
   for c in cs:
    vals=[np.log(np.clip(same(a,b),1e-6,1-1e-6)) for a,b in itertools.combinations(c['representatives'],2)];c['internal_log']=float(np.mean(vals)) if vals else 0.
  cross={}
  for i,j in linked:
   for a,b in itertools.product(cands[i],cands[j]):
    vals=[np.log(np.clip(1-same(x,y),1e-6,1-1e-6)) for x,y in itertools.product(a['representatives'],b['representatives'])];cross[i,j,a['index'],b['index']]=float(np.mean(vals)) if vals else 0.
  result={}
  for condition in cfg['linkage']['conditions']:
   internal=condition in ['internal','combined'];joint=condition in ['joint','combined'];best=None
   for combo in itertools.product(*[cs or [None] for cs in cands]):
    score=sum(c['score']+(c['internal_log'] if internal else 0) for c in combo if c)
    if joint:score+=sum(cross[i,j,combo[i]['index'],combo[j]['index']] for i,j in linked if combo[i] and combo[j])
    if best is None or score>best[0]:best=(score,combo)
   # Baseline is equivalent independent maxima; sanity against saved decisions.
   selected=best[1];cr=[]
   for r,c in zip(rs,selected):
    if condition=='baseline':assert (c['span'] if c else None)==(r['chosen']['span'] if r['chosen'] else None)
    row=dict(card=r['card'],family=r['family'],chosen=c,semantic_completion=None)
    if 'truth' in r:
     t=r['truth'];observed=set(c['body_observed']) if c else set();row['truth']=dict(witness=t['witness'],wrong_witness=t['wrong_witness'],all_witness=set(t['witness'])<=observed,wrong_points=len(set(t['wrong_witness'])&observed))
    cr.append(row)
   order=[]
   for i,j in linked:
    a,b=selected[i],selected[j];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse'
    order.append(dict(cards=[rs[i]['card'],rs[j]['card']],state=state,semantic_completion=None))
   result[condition]=dict(score=best[0],cards=cr,order=order)
  dst=W/'predictions'/str(e['seed'])/f"{e['clip_id']}_{e['kind']}.npz";dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();np.savez_compressed(dst,pairs=np.array(pairs,dtype=np.int32).reshape(-1,2),p=ps)
  predictions.append(dict(seed=e['seed'],clip_id=e['clip_id'],kind=e['kind'],path=str(dst),sha256=sha(dst),source=e['source'],source_sha256=e['source_sha256'],stage=e['path'],stage_sha256=e['sha256']))
  elapsed=time.perf_counter()-start;bench.append(dict(seed=e['seed'],clip_id=e['clip_id'],kind=e['kind'],seconds=elapsed,pairs=len(pairs)));out.append(dict(seed=e['seed'],clip_id=e['clip_id'],kind=e['kind'],conditions=result))
  if len(out)%40==0:print('C compared',len(out),'of',len(es),flush=True)
 summary=[]
 for seed,cond,kind in itertools.product(cfg['seeds'],cfg['linkage']['conditions'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cards=[c for r in rr for c in r['conditions'][cond]['cards']];truth=[c['truth'] for c in cards if 'truth'in c];order=[p['state'] for r in rr for p in r['conditions'][cond]['order']]
  summary.append(dict(seed=seed,condition=cond,kind=kind,cases=len(rr),cards=len(cards),missing=sum(c['chosen'] is None for c in cards),known_groups=len(truth),all_witness=sum(t['all_witness'] for t in truth),wrong_points=sum(t['wrong_points'] for t in truth),order={k:order.count(k) for k in set(order)},hand_qualified_points=sum(len(c['chosen']['hand_qualified']) if c['chosen'] else 0 for c in cards)))
 assert all(s['hand_qualified_points']==0 for s in summary if s['kind']=='all_hand_missing')
 write(W/'COMPARISON.json',dict(at=stamp(),rows=out,summary=summary,predictions=predictions,timing=bench,source_cases=len(es)//3,model_runs=len(es),old_encoder_calls=0,web_changed=False,limitations=['Exposed DEV and synthetic transformations; not new-user correctness.','Only source-reviewed distinct-action contract is used; unknown/concurrent relations get no joint exclusion.','Atomic conditional prediction not used; repetition generalization remains unverified.','Candidate presence is not completion. No shape/semantic score is promoted.','Timing covers pair features, relation inference and selection on cached observations, not video capture/pose/end-to-end.']))
 for s in summary:
  if s['kind'] in ['full_normal','dynamic_complete_blocks']:print(s,flush=True)

if __name__=='__main__':
 try:{'freeze':freeze,'compare':compare}[sys.argv[1]]()
 except BaseException:(W/f'C_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
