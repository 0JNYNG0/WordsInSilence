"""Separate NumPy recomputation directly from stored logits; never invokes models."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;LAB=U.parents[1]
def read(p):return json.loads(p.read_text('utf-8-sig'))
scope=read(U/'SCOPE.json');analysis=read(U/'ANALYSIS.json')
cfg=read(LAB/'config/word100_practice.json');count={k:[0,0] for k in ['C','T3F']};outcomes=Counter();ranks={}
for tid in scope['trial_ids']:
    d=read(LAB/'runs'/scope['session_id']/'trials'/f'{tid}.result.json')
    target=cfg['word_codes'].index(d['requested_word_id']);ranks[tid]={}
    rr=next(r for r in analysis['rows'] if r['trial_id']==tid)
    correct=[]
    for key,o in [('C',d['comparison']['baseline']),('T3F',d['practice100'])]:
        a=np.asarray(o['logits'],dtype=np.float64);order=np.argsort(-a,kind='stable')
        rank=int(np.flatnonzero(order==target)[0])+1
        soft=np.exp(a-a.max());soft/=soft.sum()
        assert rank==rr['models'][key]['rank']
        assert abs(100*soft[target]-rr['models'][key]['target_model_score_percent'])<1e-10
        assert cfg['labels'][order[0]]==rr['models'][key]['top1_label']
        assert abs(100*(soft[order[0]]-soft[order[1]])-rr['models'][key]['first_second_gap_score_points'])<1e-10
        count[key][0]+=int(rank==1);count[key][1]+=int(rank<=5);correct.append(rank==1)
        ranks[tid][key]=rank
    outcomes['both_correct' if all(correct) else 'gain' if correct==[False,True] else 'regression' if correct==[True,False] else 'both_wrong']+=1
assert count=={'C':[10,12],'T3F':[12,12]}
assert dict(outcomes)==analysis['paired_outcomes']
assert abs((count['T3F'][0]-count['C'][0])*100/len(scope['trial_ids'])-analysis['delta_top1_pp'])<1e-12
for relative,pin in scope['source_hashes'].items():
    assert hashlib.sha256((LAB/relative).read_bytes()).hexdigest()==pin,relative
o=dict(status='PASS',methods='independent NumPy stable sort/softmax versus original Python math implementation',
    pairs=len(ranks),logit_values_checked=2400,counts=count,paired_outcomes=dict(outcomes),
    gain_labels=[r['label'] for r in analysis['rows'] if r['outcome']=='gain'],
    same_input_pairs=sum(bool(r['same_input_sha256']) for r in analysis['rows']),
    source_files_unchanged=len(scope['source_hashes']),ranks=ranks,model_inference=0,training=0,
    reviewer='same Codex, separate arithmetic code; no independent human/agent review')
(U/'INDEPENDENT_SCORES.json').write_text(json.dumps(o,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps({k:v for k,v in o.items() if k!='ranks'},ensure_ascii=False))
