"""Re-run archived public122conditions with the frozen quality_only candidate in shadow only."""
from quality_common import *
import torch,time
from quality_runtime_v2 import GuardedObserver,refine
sys.path[:0]=[str(D),str(O/'word100_guide_requirement_learning_20261111')]
import requirement_core as rc
from direction_adapter import DirectionAdapter
torch.set_num_threads(2);finger=GuardedObserver('quality_only');direction=DirectionAdapter(D,rc)
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
for r,pr in zip(manifest,previous):
 assert (r['clip_id'],r['kind'])==(pr['clip_id'],pr['kind']);x,ts=input_arrays(r);src=read(r['result']);before=src['before'];start=time.perf_counter();m=direction.apply(before,x,ts,rc.FAMILIES[r['clip_id'].split('_')[2]]);obs={h:finger.infer(x,h)for h in ['L','R']};after=refine(m,obs,ts);elapsed=time.perf_counter()-start;old=read(pr['output'])['after']
 assert after['original_candidates']==old['original_candidates'] and after['original_orders']==old['original_orders']
 assert [c['state']for c in after['joint_cards']]==[c['state']for c in old['joint_cards']]
 cues=[dict(card=c['card'],**e)for c in after['joint_cards']for e in c['finger_details']]
 assert not r['normal'] or not cues
 for a,b in zip(after['joint_cards'],old['joint_cards']):
  assert a['observations']==b['observations'] and a['differences']==b['differences']
  # Every retained detail must be inside previous support, never newly invented.
  for e in a['finger_details']:
   assert any(e['finger']==q['finger'] and e['hand']==q['hand'] and q['span'][0]<=e['span'][0]<=e['span'][1]<=q['span'][1] for q in b['finger_details'])
 reports.append(dict(clip_id=r['clip_id'],kind=r['kind'],dataset=r['dataset'],normal=r['normal'],old_details=len(pr['details']),new_details=len(cues),seconds=elapsed,verdicts_unchanged=True))
summary=dict(cases=len(reports),normal_cases=sum(r['normal']for r in reports),normal_new_warnings=0,old_details=sum(r['old_details']for r in reports),new_details=sum(r['new_details']for r in reports),seconds_median=float(np.median([r['seconds']for r in reports])),seconds_p95=float(np.percentile([r['seconds']for r in reports],95)),scope='Supplementary motion and all finger models only, excludesRTMW/base/DTW/web; synthetic errors are not real learner validation',web_changed=False)
write(U/'SHADOW_REPLAY.json',dict(summary=summary,rows=reports));print(json.dumps(summary))
