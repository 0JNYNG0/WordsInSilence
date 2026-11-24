"""Independent arithmetic path over saved runs; never re-trains/re-infers."""
from pathlib import Path
import collections, hashlib, json
import numpy as np
U=Path(__file__).resolve().parent
P=U.parent/'word100_feedback_recovery_20261113/motion_recovery'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(U/'ARM_FROZEN.json'); data=read(U/'ARM_PREDICTIONS.json'); stats=read(U/'ARM_COMPARISON.json')
z=np.load(U/'ARM_INPUTS.npz'); rows=read(P/'PHYSICAL_INDEX.json')['rows']
for p,h in cfg['pins'].items(): assert sha(p)==h,p
trained=read(U/'ARM_TRAINED.json'); assert len(trained['runs'])==18
byseed={}
for r in trained['runs']:
    d=U/'arm_runs'/f"{r['person']}_{r['seed']}"
    assert sha(d/'model.pt')==r['model_sha256'] and sha(d/'predictions.npz')==r['prediction_sha256']
    q=np.load(d/'predictions.npz'); assert np.array_equal(q['held'],z['person']==r['person'])
    out=byseed.setdefault(r['seed'],np.full(len(rows),-1,dtype=int)); held=q['held']&z['eligible']
    out[held]=(q['logits'][held]>=0).astype(int)
for seed,out in byseed.items():
    saved=[r for r in data if r['seed']==seed]
    assert len(saved)==len(rows) and np.array_equal(out,[r['candidate'] for r in saved])
short=set(read(P/'PHYSICAL_ADOPTION_CONTRACT.json')['short_change']['source_hand_keys'])
for s in stats['statistics']:
    part=[r for r in data if r['seed']==s['seed'] and r['view']==s['view'] and r['reverse']==s['reverse'] and r['truth']>=0]
    g=s['group']
    if g in ('M','H'): part=[r for r in part if r['truth']==(g=='M')]
    elif g in ('L','R','LM','LH','RM','RH'):
        part=[r for r in part if r['side']==g[0] and (len(g)==1 or r['truth']==(g[1]=='M'))]
    elif g.startswith('cycle'):
        part=[r for r in part if r['word_id']=='WORD1567' and r['kind'].startswith('main_') and (g=='cycle' or r['side']==g[-1])]
    elif g=='short': part=[r for r in part if r['source_key'] in short]
    else: assert g=='all'
    assert len(part)==s['total']
    for name in ('candidate','previous','invariant','rule'): assert sum(r[name]==r['truth'] for r in part)==s[name]
    assert s['passed']==(s['candidate']>=max(s['previous'],s['rule']))
pairs=collections.defaultdict(dict)
for r in data: pairs[(r['seed'],r['source_key'],r['view'])][r['reverse']]=r
assert all(len(p)==2 and p[False]['candidate']==p[True]['candidate'] for p in pairs.values())
loss=sorted({r['source_key'] for r in data if r['truth']>=0 and r['previous']==r['truth'] and r['candidate']!=r['truth']})
gain=sorted({r['source_key'] for r in data if r['truth']>=0 and r['previous']!=r['truth'] and r['candidate']==r['truth']})
decision=all(s['passed'] for s in stats['statistics']) and not loss
assert decision==stats['adoption_gates_passed']==False
result=dict(passed=True,arithmetic_rows=len(data),strata=len(stats['statistics']),runs=18,
    source_hand_windows=230,known=228,unknown=2,paired_conditions=len(pairs),
    time_reverse_disagreements=0,source_cells_with_gain=gain,source_cells_with_loss=loss,
    adoption_gates_passed=False,web_applied=False,model_rerun=False,
    scope='Same Codex separate saved-array/hash arithmetic, not independent human truth or fresh evaluation')
with (U/'ARM_SAVED_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
