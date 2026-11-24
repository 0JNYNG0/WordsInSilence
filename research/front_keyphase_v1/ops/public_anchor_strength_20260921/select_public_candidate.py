"""Independent public-output recount and preregistered selection; no inference/training."""
from pathlib import Path
import hashlib
import json
import math
from collections import Counter
import numpy as np
import torch

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
policy=read(OPS/'comparison_policy.json');report=read(OPS/'PUBLIC_EVALUATION.json');train=read(OPS/'REAL_TRAINING_AUDIT.json')
assert policy['coefficients']==[.1,.01]
checks=[];candidates={};native_changes={}
def check(name,value):checks.append(dict(name=name,passed=bool(value)))
check('same frozen selection policy',report['policy_sha256']==sha(OPS/'comparison_policy.json'))
check('four saved-model tensor audits passed',train['status']=='PASS' and train['passed']==36)
check('exact public input bytes retained',all(sha(p)==d for p,d in report['public_inputs'].items()))
check('exact4 frozen models retained',report['models']==train['models'] and all(sha(v['path'])==v['sha256'] for v in report['models'].values()))
check('only23 new encoder calls, public9 features reused',report['encoder_forwards']==23 and report['public_train_features_reused']==9 and report['linear_head_outputs']==128)
check('no user scoring/training/heldout in evaluation',report['optimizer_steps']==report['user_records_or_scores_read']==report['reserved_arrays_read']==0)
check('feature cache hash',report['features_scores_sha256']==sha(OPS/'public_features_scores.npz'))
old_public=read(WEB/'ops/parent_anchor_pair_20260921/public_reference_result.json')['results']['control']['rows']
old_fit=read(WEB/'ops/public_cached_domain_probe_20260921/TRAIN_FIT_RESULT.json')['rows']
check('exact public23 IDs labels conditions',[(r['id'],r['true'],r['condition']) for r in report['rows'] if r['domain']=='public_reference']==[(r['id'],r['true'],r['condition']) for r in old_public])
check('exact additional9 input IDs labels signers',[(r['id'],r['true'],r['signer_id']) for r in report['rows'] if r['condition']=='additional_train']==[(r['id'],r['reference_label'],r['signer_id']) for r in old_fit])
payload={name:torch.load(v['path'],map_location='cpu',weights_only=True) for name,v in report['models'].items()}
maxerror=0.;count=0;scopes_count=0
targets={r['true'] for r in old_public if r['condition']=='intact'};assert len(targets)==18
with np.load(OPS/'public_features_scores.npz',allow_pickle=False) as z:
    features=z['features'];scores=z['scores'];classes=list(z['classes']);names=list(z['models'])
    check('fixed32 inputs4heads93classes',features.shape[0]==32 and scores.shape==(32,4,93) and list(z['ids'])==[r['id'] for r in report['rows']])
    for j,row in enumerate(report['rows']):
        for k,name in enumerate(names):
            state=payload[name]['state_dict'];W=state['head.weight'].double().numpy();bias=state['head.bias'].double().numpy()
            values=np.array([math.fsum(float(a)*float(b) for a,b in zip(weights,features[j]))+float(offset) for weights,offset in zip(W,bias)])
            maxerror=max(maxerror,float(np.max(np.abs(values-scores[j,k]))));count+=93
            for scope,ids in [('project18',[i for i,c in enumerate(classes) if c in targets]),('full93',list(range(93)))]:
                order=sorted(ids,key=lambda i:-values[i]);saved=row['models'][name][scope];yi=classes.index(row['true'])
                assert saved['pred']==classes[order[0]] and saved['rank']==order.index(yi)+1 and saved['correct']==(order[0]==yi)
                scopes_count+=1
    for name in names:
        for scope in ['project18','full93']:
            check(name+' '+scope+' independent condition totals',all(report['summary'][name][scope][condition]['n']==sum(r['condition']==condition for r in report['rows']) and report['summary'][name][scope][condition]['correct']==sum(r['condition']==condition and r['models'][name][scope]['pred']==r['true'] for r in report['rows']) for condition in ['intact','reviewed_masked','additional_train']))
check('all11904 scalar scores independently agree',count==11904 and maxerror<1e-10)
check('all256 scoped rankings agree',scopes_count==256)
baseline=read(WEB/'runs/head_only_native_pair_v1/results.json')
for key in ['lambda_0p1','lambda_0p01']:
    audit=read(OPS/key/'native_pair_result_audit.json');check(key+' native output integrity passed',audit['status']=='PASS_OUTPUT_INTEGRITY' and audit['passed']==45)
    ready=read(OPS/key/'execution_ready.json');native=read(Path(ready['out'])/'results.json')
    name=key+'_treatment';pub=report['summary'][name];ev=native['arms']['treatment']['evaluations']
    # Recount the exact eligibility predicates from individual predictions.
    train9=all(r['models'][name]['full93']['pred']==r['true'] for r in report['rows'] if r['condition']=='additional_train')
    dev3=all(r['pred']==r['true'] for r in ev['additional3']['rows'])
    source_correct=sum(r['pred']==r['true'] for r in ev['source696']['rows'])
    retention=all(all(r['models'][name][scope]['pred']==r['true'] for r in report['rows'] if r['condition'] in ['intact','reviewed_masked']) for scope in ['project18','full93'])
    gates={'train9':train9,'dev3':dev3,'source696_at_least686':source_correct>=686,'public18_missing5_both_scopes':retention}
    full_correct=sum(r['pred']==r['true'] for r in ev['full']['rows']);truth=Counter(r['true'] for r in ev['full']['rows']);pred=Counter(r['pred'] for r in ev['full']['rows']);tp=Counter(r['true'] for r in ev['full']['rows'] if r['pred']==r['true']);macro=math.fsum(2*tp[x]/(truth[x]+pred[x]) for x in truth)/len(truth)
    check(key+' selection scores match independent counts',full_correct==ev['full']['correct'] and abs(macro-ev['full']['macro_f1'])<1e-12)
    candidates[name]={'eligible':all(gates.values()),'gates':gates,'full699_correct':full_correct,'supported_gt_macro_f1':macro,'squared_distance':train['details'][name]['squared_distance'],'source696_correct':source_correct,'checkpoint':report['models'][name]}
    native_changes[key]={}
    for scope in ['full','source696','additional3','target_nikl18']:
        now=ev[scope]['rows'];old={r['id']:r for r in baseline['arms']['treatment']['evaluations'][scope]['rows']};assert set(old)=={r['id'] for r in now}
        changed=[{'id':r['id'],'true':r['true'],'before':old[r['id']]['pred'],'after':r['pred']} for r in now if old[r['id']]['pred']!=r['pred']]
        native_changes[key][scope]={'gains':[r for r in changed if r['after']==r['true']],'regressions':[r for r in changed if r['before']==r['true']],'prediction_changes':changed}
eligible=[name for name,c in candidates.items() if c['eligible']]
order=sorted(eligible,key=lambda name:(-candidates[name]['full699_correct'],-candidates[name]['supported_gt_macro_f1'],candidates[name]['squared_distance']))
winner=order[0] if order else None
result={'status':'PASS_PUBLIC_ONLY_SELECTION' if all(x['passed'] for x in checks) else 'FAIL','policy_sha256':sha(OPS/'comparison_policy.json'),'passed':sum(x['passed'] for x in checks),'total':len(checks),'checks':checks,'candidates':candidates,'eligible_order':order,'selected':winner,'native_changes_vs_previous_unanchored_treatment':native_changes,'scalar_scores_recounted':count,'scoped_rankings_recounted':scopes_count,'max_scalar_error':maxerror,'new_inference':0,'user_records_or_scores_used_for_selection':False,'model_promoted':False,'limits':'Public development selection only. User/wider-person webcam generalization and automatic word boundaries remain unverified.'}
with (OPS/'SELECTION_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
assert result['status']=='PASS_PUBLIC_ONLY_SELECTION'
with (OPS/'SELECTED_CANDIDATE.json').open('x',encoding='utf-8') as f:
    json.dump({'status':'FROZEN_FOR_PROSPECTIVE_VALIDATION' if winner else 'NO_ELIGIBLE_CANDIDATE','selected':winner,'checkpoint':candidates[winner]['checkpoint'] if winner else None,'selection_policy_sha256':result['policy_sha256'],'selection_audit_sha256':sha(OPS/'SELECTION_AUDIT.json'),'selected_without_user_scores':True,'default_changed':False,'do_not_tune_from_followup_result':True},f,indent=2)
print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'candidates':candidates,'selected':winner},ensure_ascii=False))
