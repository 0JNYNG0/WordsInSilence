"""Separate saved-array arithmetic audit; no training or inference."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_feedback_recovery_20261113/motion_recovery'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=read(U/'MOTION_FROZEN.json');z=np.load(U/'MOTION_INPUTS.npz');rows=read(OLD/'PHYSICAL_INDEX.json')['rows'];a=read(U/'MOTION_COMPARISON.json');d=read(U/'MOTION_PREDICTIONS.json')
for p,h in c['pins'].items():assert sha(p)==h,p
assert len(read(U/'MOTION_TRAINED.json')['runs'])==18
for row in read(U/'MOTION_TRAINED.json')['runs']:
 p=U/'motion_runs'/f"{row['person']}_{row['seed']}"
 assert sha(p/'model.pt')==row['model_sha256'] and sha(p/'predictions.npz')==row['prediction_sha256']
for s in a['statistics']:
 part=[r for r in d if r['seed']==s['seed'] and r['view']==s['view'] and r['reverse']==s['reverse'] and r['truth']>=0]
 if s['group']=='all':
  assert len(part)==s['total']==228
  for k,t in [('prediction','candidate'),('previous','previous'),('rule','rule')]:assert sum(r[k]==r['truth'] for r in part)==s[t]
paired=collections.defaultdict(dict)
for r in d:paired[(r['seed'],r['source_key'],r['view'])][r['reverse']]=r
new_reverse_conflicts=[];old_reverse_conflicts=[]
for key,pair in paired.items():
 assert len(pair)==2
 if pair[False]['prediction']!=pair[True]['prediction']:new_reverse_conflicts.append(key)
 if pair[False]['previous']!=pair[True]['previous']:old_reverse_conflicts.append(key)
assert not new_reverse_conflicts
loss=sorted(set(r['source_key'] for r in d if r['truth']>=0 and r['previous']==r['truth'] and r['prediction']!=r['truth']))
gain=sorted(set(r['source_key'] for r in d if r['truth']>=0 and r['previous']!=r['truth'] and r['prediction']==r['truth']))
out=dict(passed=True,scope='Separate saved hashes/arrays/counts; same Codex; not fresh or human validation',
 source_hand_cells=230,known=228,unknown=2,paired_time_conditions=len(paired),
 previous_reverse_disagreements=len(old_reverse_conflicts),candidate_reverse_disagreements=0,
 source_cells_with_gain=gain,source_cells_with_loss=loss,
 adoption_gates_passed=a['adoption_gates_passed'],web_applied=False,
 conclusion='Time-order-invariant observation restored, but one short real movement is newly missed. Preserve existing model; do not call physical motion semantic completion.')
(U/'MOTION_SAVED_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
