from pathlib import Path
import json,hashlib,collections
from ownership_evidence import inspect_ownership
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
write('OWNERSHIP_FROZEN.json',dict(code_sha256=sha(U/'ownership_evidence.py'),
 input_sha256=sha(U/'FULL_GRAPH_COMPARISON.json'),
 target='Explicit mutual ownership conflict between independently detected roles and full candidate windows',
 no_expected_order_or_word_id=True,no_window_deletion=True,no_neural_calls=True,
 gates=['No new conflict on normal/speed','Keep actual reversal witnesses and all original full-span component data','Observed conflicting assignment is not evidence of a learner mistake']))
inp=read(U/'FULL_GRAPH_COMPARISON.json');rows=[]
for r in inp['rows']:
 rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],ownership=inspect_ownership(r['cards'])))
summary=[]
for seed in [42,43,44]:
 for kind in sorted({r['kind']for r in rows}):
  rr=[r for r in rows if r['seed']==seed and r['kind']==kind]
  summary.append(dict(seed=seed,kind=kind,cases=len(rr),conflict_cases=sum(bool(r['ownership']['conflicts'])for r in rr),
   conflict_pairs=sum(len(r['ownership']['conflicts'])for r in rr)))
write('OWNERSHIP_COMPARISON.json',dict(rows=rows,summary=summary,new_neural_calls=0,web_changed=False,
 normal_conflicts=[r for r in rows if r['kind']in ['full_normal','speed075','speed125']and r['ownership']['conflicts']],
 scope='Attribution uncertainty, not completed action recognition or semantic error accuracy'))
print(json.dumps([s for s in summary if s['seed']==42],ensure_ascii=False))
