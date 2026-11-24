from split_runtime import *
import time,copy
sys.path[:0]=[str(D),str(O/'word100_guide_requirement_learning_20261111')]
import requirement_core as rc
from direction_adapter import DirectionAdapter
torch.set_num_threads(2);model=SplitObserver();direction=DirectionAdapter(D,rc)
manifest=read(D/'INTEGRATED_RESULTS.json')['rows'];sources={r['clip_id']:r for r in read(O/'word100_public_g4_20261114/EXTRACTED.json')['rows']};labels=read(O/'word100_public_g4_20261114/LABELS_FROZEN.json')['actions'];previous=read(P/'REPLAY_RESULTS.json')['rows'];reports=[]
def input_arrays(r):
 if r['dataset']=='development82':
  z=np.load(r['source']);return z['x'].copy(),z['ts'].copy()
 ch=np.load(sources[r['clip_id']]['channel_path']);x=ch['tcn280'];n=len(x);ids=np.arange(n);kind=r['kind']
 if kind=='prefix_before_goal':n=next(a['partial_end']for a in labels if a['clip_id']==r['clip_id']and a['card']=='close_chest')+1;ids=np.arange(n)
 if kind=='continuous_full_reverse':ids=ids[::-1].copy()
 rate={'speed075':.75,'speed125':1.25}.get(kind,1);x=x[ids].copy();ts=np.round(ch['timestamp_us'][:n]/rate).astype(np.int64);x[:,279]=np.r_[0,np.diff(ts)/1e6]
 if kind=='all_hands_missing':x[:,18:186]=0;x[:,195:279]=0
 return x,ts
for n,(r,pr)in enumerate(zip(manifest,previous)):
 assert(r['clip_id'],r['kind'])==(pr['clip_id'],pr['kind']);x,ts=input_arrays(r);src=read(r['result']);before=src['before'];start=time.perf_counter();m=direction.apply(before,x,ts,rc.FAMILIES[r['clip_id'].split('_')[2]]);allobs={h:model.all(x,h)for h in ['L','R']};old=read(pr['output'])['after'];modes={}
 for mode in ['compatible_shape','combined']:
  obs={h:allobs[h]|dict(states=allobs[h][mode])for h in ['L','R']};after=refine(m,obs,ts)
  assert after['original_candidates']==old['original_candidates']and after['original_orders']==old['original_orders']
  assert[c['state']for c in after['joint_cards']]==[c['state']for c in old['joint_cards']]
  cues=[dict(card=c['card'],**e)for c in after['joint_cards']for e in c['finger_details']];retained=0;old_count=0
  for a,b in zip(after['joint_cards'],old['joint_cards']):
   assert a['observations']==b['observations']and a['differences']==b['differences']
   for e in b['finger_details']:
    old_count+=1
    if any(e['finger']==q['finger']and e['hand']==q['hand']and min(e['span'][1],q['span'][1])>=max(e['span'][0],q['span'][0])for q in a['finger_details']):retained+=1
  modes[mode]=dict(details=len(cues),retained_old_details=retained,old_details=old_count,normal_warning=bool(r['normal']and cues),cues=cues)
 reports.append(dict(clip_id=r['clip_id'],kind=r['kind'],dataset=r['dataset'],normal=r['normal'],modes=modes,seconds=time.perf_counter()-start,coarse_direction_order_unchanged=True))
 if n%25==0:print('replay',n,flush=True)
summary={mode:dict(cases=len(reports),normal_cases=sum(r['normal']for r in reports),normal_new_warnings=sum(r['modes'][mode]['normal_warning']for r in reports),old_details=sum(r['modes'][mode]['old_details']for r in reports),new_details=sum(r['modes'][mode]['details']for r in reports),retained_old_details=sum(r['modes'][mode]['retained_old_details']for r in reports))for mode in ['compatible_shape','combined']}
write(U/'REPLAY_RESULTS.json',dict(summary=summary,rows=reports,scope='Public122 archived scenarios,64normal; synthetic errors not real learner accuracy; all coarse/direction/order outputs asserted equal'))
print(json.dumps(summary))
