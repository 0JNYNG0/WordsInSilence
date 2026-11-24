"""Separate arithmetic/provenance audit; no repeated learning or scoring."""
from pathlib import Path
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_positive_direction_20261109'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=read(U/'LOCAL_DIRECTION_COMPARISON_V2.json');assert v['passed']
assert v['candidate_sha256']==sha(U/'localized_direction.py')
contracts=read(D/'GUIDE_CONTRACTS.json')['contracts'];cfg=read(D/'FROZEN.json')['direction']
changed=[];unique=set();events=0
for row in v['cases']:
    path=D/'predictions'/(row['case']+'.npz');assert sha(path)==row['prediction_sha256'];z=np.load(path)
    unique.add('_'.join(row['case'].split('_')[:5]))
    if row['normal']:assert row['previous_states']==row['current_states'] and not row['current_cues']
    for c in row['motion']:
        m=c['evidence'];contract=contracts[c['card']]
        active=np.flatnonzero(z['phase'][:,np.array(contract['phase_groups'])+1].sum(1)>=cfg['phase_support']).tolist()
        actual=[i for e in m['events'] for i in e['source_phase_indices']]
        assert actual==active,'An island was lost or an unassociated frame inserted'
        for e in m['events']:
            events+=1;a,b=e['span'];assert e['source_phase_indices']==list(range(a,b+1))
            assert e['whole_action_complete'] is None
            if e['state']=='contradicted':assert e['signed_change'] < -e['deadband'] and e['trend']<0
            if e['state']=='supported':assert e['signed_change'] > e['deadband'] and e['trend']>0
        assert m['contradicted_spans']==[e['span'] for e in m['events'] if e['state']=='contradicted']
    a=[c['focus_indices'] for c in row['previous_cues'] if c['kind']=='direction'];b=[c['focus_indices'] for c in row['current_cues'] if c['kind']=='direction']
    if a!=b:changed.append(dict(case=row['case'],before=a,after=b))
for p,h in read(O/'web_tutorial_live_integration_20261113/PROTECTED.json').items():assert sha(Path(p))==h,p
out=dict(passed=True,public_conditions=len(v['cases']),normal_conditions=sum(r['normal'] for r in v['cases']),
    unique_source_ids=len(unique),source_ids=sorted(unique),events=events,changed_direction_focus=changed,
    source_scope='Repeated conditions of 12 public video IDs; not independent 93 videos',
    all_active_indices_preserved=True,protected_models_unchanged=True,educational_correctness_validated=False,
    review='Separate arithmetic/source-code check by same Codex, not independent KSL expert')
with (U/'DIRECTION_SAVED_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
