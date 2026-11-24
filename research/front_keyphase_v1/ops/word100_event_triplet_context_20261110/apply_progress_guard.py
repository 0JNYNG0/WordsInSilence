from pathlib import Path
import json,hashlib,copy
from progress_guard import assess,apply
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
assert not (U/'GUARDED_COMPLETE.json').exists()
# Pure logical counterexamples: partial != error, core appearance != completed movement.
c=dict(display_status='similar');e=dict(span=[2,8],eligible=True,confidence=.9,prediction=2,complete=False)
b=dict(contexts=[4],movement_links=[e]);p=dict(a=2,b=8,observed=True,context=4,confidence=.9,core_state='progress',core_probabilities=[.05,.9,.05])
assert assess(c,b,[p])['withdraw_positive']
assert not assess(dict(display_status='difference'),b,[p])['withdraw_positive']
assert not assess(c,dict(contexts=[4],movement_links=[dict(e,complete=True)]),[p])['withdraw_positive']
assert not assess(c,b,[dict(p,observed=False)])['withdraw_positive']
assert not assess(c,b,[dict(p,core_state='core_appearance',core_probabilities=[.05,.05,.9])])['withdraw_positive']
assert not assess(c,b,[dict(p,context=5)])['withdraw_positive']
assert not assess(c,dict(contexts=[4],movement_links=[]),[])['withdraw_positive']
write(U/'PROGRESS_GUARD_PROTOCOL.json',dict(reason='Saved audit found an unfinished preparatory/partial positive-direction span being used as enough support. If independent center-only core head explicitly says progress and no complete movement is observed, similarity must remain unconfirmed. Does not claim omission or use the expected word order.',same_dev_post_result_fix=True,new_independent_validation=False,threshold_search=False,code_sha256=sha(U/'progress_guard.py'),tests=7,changes='Only similar -> uncertain; never add an error or positive; no retraining/inference.'))
rows=[]
for r in read(U/'FULLPATH_COMPLETE.json')['rows']:
 assert sha(r['result'])==r['sha256'];d=apply(read(r['result']));p=U/'guarded'/str(r['seed'])/(r['clip_id']+'_'+r['kind'])/'after.json';write(p,d);rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],baseline=r['baseline'],baseline_sha256=r['baseline_sha256'],candidate=r['result'],candidate_sha256=r['sha256'],result=str(p),sha256=sha(p),vetoes=d['triplet_completion_vetoes']))
write(U/'GUARDED_COMPLETE.json',dict(rows=rows,inference_rerun=False,training_rerun=False));print(json.dumps(dict(outputs=len(rows),vetoed=sum(bool(r['vetoes']) for r in rows))))
