"""Independent saved-tensor and loss decomposition audit; no training/forwards."""
from pathlib import Path
import json
import math
import torch

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
RUN=WEB/'runs/parent_anchor_pair_v1_recovery'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
parent=load(Path(read(OPS/'plan.json')['checkpoint']))
results=read(RUN/'results.json'); oldmeta=read(WEB/'runs/head_only_native_pair_v1/metadata.json')
checks=[];detail={}
def check(name,value):checks.append(dict(name=name,passed=bool(value)))
check('new pair completed normally',read(OPS/'training_exit_recovery.json')['exit_code']==0)
check('same input counts and uniform schedule',results['metadata']['counts']==oldmeta['counts'] and results['metadata']['schedule_digest']==oldmeta['schedule_digest'])
for arm in ('control','treatment'):
    ck=load(RUN/f'checkpoint_{arm}_final.pt'); old=load(WEB/f'runs/head_only_native_pair_v1/checkpoint_{arm}_final.pt')
    train=results['arms'][arm]['train'];a=train['parent_anchor'];setting=ck['fixed_setting']
    nonhead=[k for k in parent['state_dict'] if not k.startswith('head.')]
    check(arm+' exact58 encoder/buffer tensors preserved',len(nonhead)==58 and all(torch.equal(parent['state_dict'][k],ck['state_dict'][k]) for k in nonhead))
    check(arm+' original optimizer/sampling settings unchanged',{k:v for k,v in setting.items() if k!='parent_anchor'}==old['fixed_setting'])
    check(arm+' 93classes/final660 only',ck['classes']==parent['classes'] and len(ck['classes'])==93 and ck['step']==660 and train['intermediate_checkpoints_kept']==0)
    check(arm+' zero initial distance',a['squared_distance'][0]==0 and a['penalty'][0]==0)
    check(arm+' finite660 component observations',all(len(a[k])==660 and all(math.isfinite(v) for v in a[k]) for k in ['cross_entropy','penalty','total_loss','squared_distance']))
    residual=max(abs(t-ce-p) for t,ce,p in zip(a['total_loss'],a['cross_entropy'],a['penalty']))
    check(arm+' total CE plus half squared distance',residual<1e-5 and all(p==.5*d for p,d in zip(a['penalty'],a['squared_distance'])))
    check(arm+' legacy loss is correctly labeled total',a['legacy_losses_are_total_including_anchor'] and not a['legacy_losses_are_pure_cross_entropy'] and all(abs(x-y)<5.1e-7 for x,y in zip(train['losses'],a['total_loss'])))
    distance=lambda model:sum(float(((model['state_dict'][k].double()-parent['state_dict'][k].double())**2).sum()) for k in ['head.weight','head.bias'])
    d=distance(ck);old_d=distance(old)
    check(arm+' saved final distance agrees with post-update metadata',abs(d-a['final_squared_distance'])<1e-12 and d==setting['parent_anchor']['final_squared_distance'])
    check(arm+' fixed uniform coefficient across all head parameters',a['coefficient']==1 and a['half_factor']==.5 and a['reduction']=='sum' and a['parameters']==['head.weight','head.bias'])
    detail[arm]={'new_squared_distance':d,'previous_no_anchor_squared_distance':old_d,'last_CE':a['cross_entropy'][-1],'last_total':a['total_loss'][-1],'last_penalty':a['penalty'][-1],'max_loss_sum_roundoff':residual}
first=load(WEB/'runs/parent_anchor_pair_v1/checkpoint_control_final.pt');again=load(RUN/'checkpoint_control_final.pt')
check('recovery control exactly matches preserved first control tensors',all(torch.equal(v,again['state_dict'][k]) for k,v in first['state_dict'].items()))
report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'details':detail,'new_training_or_forward_calls':0,'completed_control_training_executions':2,'completed_treatment_training_executions':1,'recovery_is_not_model_selection':True}
with (OPS/'REAL_TRAINING_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps(report,ensure_ascii=False))
raise SystemExit(0 if report['status']=='PASS' else 1)
