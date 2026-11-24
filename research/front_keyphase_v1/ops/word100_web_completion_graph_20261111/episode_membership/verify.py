"""Recalculate stored comparisons; no inference/training."""
from pathlib import Path
import json,hashlib,collections,datetime
W=Path(__file__).resolve().parent;U=W.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen=read(W/'FROZEN.json')
for p,h in frozen['pins'].items():assert sha(p)==h,p
results=read(W/'RESULTS.json');done=read(W/'EVALUATION_COMPLETE.json');base={(r['clip_id'],r['kind'],r['seed']):r for r in read(U/'context_role_training/RESULTS.json') if r['condition']=='source_context_control'}
assert len(results)==492
details=[]
for group in done['summary']:
 rows=[r for r in results if r['condition']==group['condition'] and r['seed']==group['seed']];assert len(rows)==82
 changed=0;normal_total=0;missing_new=0
 for r in rows:
  before=base[r['clip_id'],r['kind'],r['seed']]
  for old,new in zip(before['cards'],r['cards']):
   assert old['id']==new['id']
   observed=sorted(set(i for a,b in new['evidence_intervals'] for i in range(a,b+1)));assert observed==new['observed_indices']
   if r['kind'] in ('full_normal','speed075','speed125'):
    normal_total+=1;changed+=old['span']!=new['span'] or old['observed_indices']!=new['observed_indices']
   if r['kind']=='all_hand_missing':missing_new+=new['span'] is not None
   if old['parent'] or old['family'] not in [2,3,4,5,7]:assert old==new
   else:
    for g in new['groups']:
     ok=[c for c in g['candidates'] if c['eligible'] and c['anchor_indices'] and c['membership']>=.5]
     ranked=sorted(ok,key=lambda c:c['score']*c['membership'],reverse=True)
     assert g['span']==(ranked[0]['span'] if ranked else None)
 dynamic=[p for r in rows if r['kind']=='dynamic_complete_blocks' for p in r['pairs'] if p.get('limited_dynamic_target')]
 score=sum(p['state']=='reversed' and p['spans_within_target_occurrences'] is True and not p['selected_span_crosses_splice'] for p in dynamic)
 assert changed==len(group['normal_speed_changed']) and score==group['limited_dynamic']['correct']
 assert missing_new==0
 details.append(dict(condition=group['condition'],seed=group['seed'],normal_windows_changed=changed,normal_windows_total=normal_total,correct_mapped_dynamic=score,dynamic_total=len(dynamic),new_missing_positive=missing_new))
probes=[]
for r in read(W/'TRAINING_COMPLETE.json')['runs']:
 assert sha(r['path'])==r['sha256'];pp=read(Path(r['path']).parent/'PROBE.json')['rows'];n=correct=0
 for row in pp:
  for t in row['targets']:
   n+=1;correct+=(row['probability'][[2,3,4,5,7].index(t['family'])]>=.5)==bool(t['value'])
 probes.append(dict(condition=r['condition'],seed=r['seed'],correct=correct,total=n))
report=dict(status='REJECTED_NORMAL_EVIDENCE_REGRESSION_VERIFIED',at=datetime.datetime.now().isoformat(),verification='Root separate saved arithmetic pass, not independent agent/human verification; agent quota ended before this new experiment',rows=details,probe=probes,model_calls=0,training_replayed=False,web_changed=False,common100_preserved=True,scope='Normal/speed changes are candidate evidence changes, not per-word recognition accuracy. Synthetic dynamic truth is source correspondence of B/C movement fragments, not whole-sign correctness.',next='No extra epochs or threshold lowering. Expand source-verified guide-event/candidate supervision across remaining public FIT people and mixed movement+hold/transition contexts, with unknown kept explicit. Audit source exposure before new training.')
(U/'reviews/episode_membership_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(details,ensure_ascii=False));print('probe',probes)
