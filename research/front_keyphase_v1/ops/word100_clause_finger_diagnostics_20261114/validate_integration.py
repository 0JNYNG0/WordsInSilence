from pathlib import Path
import sys,json,collections,copy,time
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_public_g4_20261114';G6=O/'word100_live_g6_20261114'
sys.path[:0]=[str(O/'word100_guide_requirement_learning_20261111'),str(P)]
import requirement_core as rc
from conflict_guard import qualify
from direction_adapter import DirectionAdapter
torch.set_num_threads(2)
cfg=rc.read(U/'INTEGRATION_FROZEN.json')
for p,h in cfg['pins'].items():assert rc.sha(p)==h
assert not(U/'INTEGRATED_RESULTS.json').exists()
adapter=DirectionAdapter(U,rc);cases=[]
saved=rc.read(O/'word100_joint_completion_block_20261114/JOINT_FINAL_V2.json')['rows'];bags=collections.defaultdict(list)
for r in saved:bags[r['clip_id'],r['kind']].append(r)
pm={(r['clip_id'],r['kind']):r for r in rc.read(O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/null_candidate_learning/evaluation/COMPARISON.json')['predictions']if r['seed']==42}
for (cid,kind),rs in bags.items():
 r=pm[cid,kind];assert rc.sha(r['source'])==r['source_sha256'];z=np.load(r['source']);cases.append(dict(dataset='development82',clip_id=cid,kind=kind,x=z['x'],ts=z['ts'],before=qualify(rs),source=r['source'],source_sha256=r['source_sha256']))
sources={r['clip_id']:r for r in rc.read(P/'EXTRACTED.json')['rows']};labels=rc.read(P/'LABELS_FROZEN.json')['actions']
for p in sorted((G6/'equivalence').glob('*.json')):
 d=rc.read(p);r=d['check'];cid=r['clip_id'];kind=r['kind'];sr=sources[cid];ch=np.load(sr['channel_path']);x=ch['tcn280'];n=len(x);ids=np.arange(n)
 if kind=='prefix_before_goal':n=next(a['partial_end']for a in labels if a['clip_id']==cid and a['card']=='close_chest')+1;ids=np.arange(n)
 if kind=='continuous_full_reverse':ids=ids[::-1].copy()
 rate={'speed075':.75,'speed125':1.25}.get(kind,1);x=x[ids].copy();ts=np.round(ch['timestamp_us'][:n]/rate).astype(np.int64);x[:,279]=np.r_[0,np.diff(ts)/1e6]
 if kind=='all_hands_missing':x[:,18:186]=0;x[:,195:279]=0
 cases.append(dict(dataset='public8_40conditions',clip_id=cid,kind=kind,x=x,ts=ts,before=d['result'],source=str(p),source_sha256=rc.sha(p)))
out=[];dest=U/'integrated';dest.mkdir(exist_ok=True)
for r in cases:
 word=r['clip_id'].split('_')[2];b=r['before'];a=adapter.apply(b,r['x'],r['ts'],rc.FAMILIES[word]);changes=[];lost=[];newwarnings=[];retained=[]
 assert a['original_candidates']==b['original_candidates']and a['original_orders']==b['original_orders']
 for bc,ac in zip(b['joint_cards'],a['joint_cards']):
  bs=[e for e in bc['observations']if e['task']!='direction'];ss=[e for e in ac['observations']if e['task']!='direction'];assert bs==ss
  bd=[e for e in bc['observations']if e['task']=='direction'];ad=[e for e in ac['observations']if e['task']=='direction'];assert len(bd)==len(ad)
  for be,ae in zip(bd,ad):
   if be['state']!=ae['state']:changes.append(dict(card=bc['card'],before=be['state'],after=ae['state'],span=ae.get('span')))
   if be['state']=='supported'and ae['state']!='supported':lost.append(bc['card'])
   if be['state']=='opposite_direction':retained.append(ae['state']=='opposite_direction')
  if ac['state']in ['attributable_feature_difference','evidence_conflict']and bc['state']not in ['attributable_feature_difference','evidence_conflict']:newwarnings.append(dict(card=ac['card'],before=bc['state'],after=ac['state']))
 normal=r['kind']in['normal','full_normal','speed075','speed125','dynamic_complete_blocks']
 def both(q):return any(any(e['task']=='shape'for e in c['differences'])and any(e['task']=='direction'for e in c['differences'])and c['state']=='attributable_feature_difference'for c in q['joint_cards'])
 rec=dict(dataset=r['dataset'],clip_id=r['clip_id'],kind=r['kind'],normal=normal,changes=changes,lost_supported=lost,new_warnings=newwarnings,old_errors_retained=all(retained),both_before=both(b),both_after=both(a),seconds=a['supplementary_motion_seconds'],source=r['source'],source_sha256=r['source_sha256'])
 result=dest/f"{r['dataset']}__{r['clip_id']}__{r['kind']}.json";rc.write(result,dict(check=rec,before=b,after=a));rec['result']=str(result);out.append(rec)
summary=dict(cases=len(out),public_videos=len({r['clip_id']for r in out}),normal_cases=sum(r['normal']for r in out),normal_new_warnings=sum(bool(r['new_warnings'])for r in out if r['normal']),lost_normal_supported=sum(bool(r['lost_supported'])for r in out if r['normal']),lost_old_errors=sum(not r['old_errors_retained']for r in out),combined_before=sum(r['both_before']for r in out if r['kind']=='closed_hand_and_reversed'),combined_after=sum(r['both_after']for r in out if r['kind']=='closed_hand_and_reversed'),combined_n=sum(r['kind']=='closed_hand_and_reversed'for r in out),changes=[{k:r[k]for k in['dataset','clip_id','kind','changes','new_warnings']}for r in out if r['changes']or r['new_warnings']],seconds_median=float(np.median([r['seconds']for r in out])),seconds_p95=float(np.percentile([r['seconds']for r in out],95)))
rc.write(U/'INTEGRATED_RESULTS.json',dict(rows=out,summary=summary,baseline='Existing admitted G3/G4 evidence; original outputs reused; no retraining or ground-truth intervals in runtime adapter',web_changed=False,candidates_preserved=True,order_preserved=True,new_finger_head_shadow_only=True))
print(json.dumps(summary,ensure_ascii=False),flush=True)
