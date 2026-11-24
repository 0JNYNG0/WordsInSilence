from common import *
import torch,time,copy
sys.path[:0]=[str(P),str(O/'word100_guide_requirement_learning_20261111')]
import requirement_core as rc
from direction_adapter import DirectionAdapter
from finger_runtime import FingerObserver,refine
torch.set_num_threads(2);finger=FingerObserver();direction=DirectionAdapter(P,rc)
manifest=read(P/'INTEGRATED_RESULTS.json')['rows'];sources={r['clip_id']:r for r in read(O/'word100_public_g4_20261114/EXTRACTED.json')['rows']};labels=read(O/'word100_public_g4_20261114/LABELS_FROZEN.json')['actions'];reports=[]
def input_arrays(r):
 if r['dataset']=='development82':
  z=np.load(r['source']);return z['x'].copy(),z['ts'].copy()
 ch=np.load(sources[r['clip_id']]['channel_path']);x=ch['tcn280'];n=len(x);ids=np.arange(n);kind=r['kind']
 if kind=='prefix_before_goal':n=next(a['partial_end']for a in labels if a['clip_id']==r['clip_id']and a['card']=='close_chest')+1;ids=np.arange(n)
 if kind=='continuous_full_reverse':ids=ids[::-1].copy()
 rate={'speed075':.75,'speed125':1.25}.get(kind,1);x=x[ids].copy();ts=np.round(ch['timestamp_us'][:n]/rate).astype(np.int64);x[:,279]=np.r_[0,np.diff(ts)/1e6]
 if kind=='all_hands_missing':x[:,18:186]=0;x[:,195:279]=0
 return x,ts
for r in manifest:
 x,ts=input_arrays(r);src=read(r['result']);before=src['before'];t=time.perf_counter();m=direction.apply(before,x,ts,rc.FAMILIES[r['clip_id'].split('_')[2]])
 obs={h:finger.infer(x,h)for h in ['L','R']};after=refine(m,obs,ts);elapsed=time.perf_counter()-t
 assert after['original_candidates']==before['original_candidates']and after['original_orders']==before['original_orders']
 # Compare against the previously verified direction supplement, not artificial labels.
 assert [c['state']for c in after['joint_cards']]==[c['state']for c in src['after']['joint_cards']]
 cues=[dict(card=c['card'],**e)for c in after['joint_cards']for e in c['finger_details']]
 if r['normal']:assert not cues,(r['clip_id'],r['kind'],cues)
 if r['kind'].startswith('all_')and'missing'in r['kind']:assert not cues
 for bc,ac in zip(before['joint_cards'],after['joint_cards']):
  assert [e for e in bc['observations']if e['task']!='direction']==[e for e in ac['observations']if e['task']!='direction']
  for e in ac['finger_details']:
   a,b=e['span'];ca,cb=e['coarse_evidence']['span'];assert ca<=a<=b<=cb and b-a>=2 and ts[b]-ts[a]>=100000
 path=U/'replay'/Path(r['result']).name;path.parent.mkdir(exist_ok=True);write(path,dict(before=before,after=after,source=r['source'],source_sha256=r['source_sha256']))
 reports.append(dict(dataset=r['dataset'],clip_id=r['clip_id'],kind=r['kind'],normal=r['normal'],details=cues,seconds=elapsed,output=str(path)))
summary=dict(cases=len(reports),normal_cases=sum(r['normal']for r in reports),new_normal_finger_warnings=sum(bool(r['details'])for r in reports if r['normal']),detail_cases=sum(bool(r['details'])for r in reports),details=sum(len(r['details'])for r in reports),seconds_median=float(np.median([r['seconds']for r in reports])),seconds_p95=float(np.percentile([r['seconds']for r in reports],95)),scope='additional motion+finger processing after coordinates and baseline evidence; not end-to-end capture latency')
write(U/'REPLAY_RESULTS.json',dict(summary=summary,rows=reports,shape_order_candidates_preserved=True,web_changed=False,normal_finger_gate=True))
print(json.dumps(summary,ensure_ascii=False),flush=True)
