"""Reuse fixed evaluation/audit logic, then independently compare checkpoint tensors."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve()
OPS=HERE.parent
WEB=OPS.parents[1]
old_validator=WEB/'ops/visibility_native_pair_20260919/validate_completed_pair.py'
old_source=old_validator.read_text(encoding='utf-8')
assert old_source.count("'runs/visibility_native_pair_v1'")==1
source=old_source.replace("'runs/visibility_native_pair_v1'", "'runs/head_only_native_pair_v1'")
# __file__ resolves the reused validator's OPS to this new output directory.
scope={'__file__':str(HERE),'__name__':'reused_fixed_pair_auditor'}
exec(compile(source,str(old_validator),'exec'),scope)

import torch
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
plan=read(OPS/'plan.json')
run=WEB/'runs/head_only_native_pair_v1'
parent=torch.load(plan['checkpoint'],weights_only=True,map_location='cpu')['state_dict']
result=read(run/'results.json')
checks=[]
def check(name,ok):checks.append({'name':name,'passed':bool(ok)})
for arm in ('control','treatment'):
    payload=torch.load(run/f'checkpoint_{arm}_final.pt',weights_only=True,map_location='cpu')
    child=payload['state_dict']
    changed=[k for k in parent if not torch.equal(parent[k],child[k])]
    check(arm+': identical state keys and shapes',list(parent)==list(child) and all(parent[k].shape==child[k].shape and parent[k].dtype==child[k].dtype for k in parent))
    check(arm+': only classifier tensors changed',set(changed)=={'head.weight','head.bias'})
    for setting_key in ('fixed_setting','settings'):
        cfg=payload[setting_key]
        check(arm+': '+setting_key+' truthful effective configuration',cfg['lr']=={'encoder':0.0,'head':0.001} and cfg['training_scope']=='head_only_frozen_encoder_and_buffers' and cfg['encoder_buffers_frozen'] is True)
    proof=result['arms'][arm]['train']['freeze_proof']
    check(arm+': reported freeze proof agrees with independent tensors',proof['all_non_head_state_identical'] and proof['all_non_head_gradients_none'] and set(proof['changed_head_keys'])==set(changed) and proof['non_head_state_tensor_count']==len(parent)-2)
check('saved metadata effective zero encoder lr',result['metadata']['settings']['lr']['encoder']==0 and result['metadata']['settings']['encoder_buffers_frozen'] is True)
report={'status':'PASS_REAL_CHECKPOINT_ENCODER_FREEZE' if all(c['passed'] for c in checks) else 'FAIL',
        'checks':checks,'passed':sum(c['passed'] for c in checks),'total':len(checks),
        'old_evaluator_sha256':hashlib.sha256(old_validator.read_bytes()).hexdigest(),
        'reused_evaluator_adaptation':'Only run directory changed; same whole development cohorts and scoring.',
        'parent_checkpoint_sha256':plan['checkpoint_sha256'],'new_training_in_audit':0,'heldout_read':False}
with (OPS/'real_checkpoint_freeze_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total']}))
if report['status']=='FAIL':raise SystemExit(1)
