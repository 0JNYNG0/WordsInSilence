from pathlib import Path
import json,hashlib,time,copy
from semantic_guard import guard,context_conflict
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (U/'GUARDED_COMPLETE.json').exists()
# Pure counterexamples, including a clearly conflicting but high-confidence prediction.
def p(k):return [1. if j==k else 0. for j in range(14)]
assert context_conflict(dict(a=0,b=3,context=3),[p(1)]*4)['conflict']
assert not context_conflict(dict(a=0,b=3,context=3),[p(3)]*4)['conflict']
assert not context_conflict(dict(a=0,b=3,context=3),[[1/14]*14]*4)['conflict']
write(U/'GUARD_PROTOCOL.json',dict(reason='Post-result RGB audit found front face pose falsely linked as later upright-forearm action in omission fixture. Conflict between saved original semantic observation and new body-context branch must veto a new correction, not be counted as omission success.',rule='Veto new direction/order only if a relevant context span has majority confident(>=existing0.65) saved old context disagreement. Keep all candidates/evidence; no new positive judgment.',code_sha256=sha(U/'semantic_guard.py'),model_unchanged=True,dev_causal_fix_not_independent_validation=True,threshold_search=False,tests=3))
rows=[]
for r in read(U/'FULLPATH_COMPLETE.json')['rows']:
 assert sha(r['result'])==r['sha256'] and sha(r['baseline'])==r['baseline_sha256'];candidate=read(r['result']);base=read(r['baseline']);after=guard(candidate,base)
 path=U/'guarded'/str(r['seed'])/(r['clip_id']+'_'+r['kind'])/'after.json';write(path,after)
 rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],before=r['baseline'],before_sha256=r['baseline_sha256'],candidate=r['result'],candidate_sha256=r['sha256'],result=str(path),sha256=sha(path),vetoes=after['guide_link_vetoes']))
write(U/'GUARDED_COMPLETE.json',dict(rows=rows,model_rerun=False,training_rerun=False,scope='Saved evidence conflict veto only; first results preserved.'))
print(json.dumps(dict(outputs=len(rows),vetoed=sum(bool(r['vetoes']) for r in rows))))
