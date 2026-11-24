"""Audit saved public-study tensors/losses without reading any participant data."""
from pathlib import Path
import json
import math
import hashlib
import torch

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
parent=load(Path(read(OPS/'lambda_0p1_plan.json')['checkpoint']))
baseline=read(WEB/'runs/head_only_native_pair_v1/metadata.json')
checks=[];details={};models={}
def check(name,value):checks.append(dict(name=name,passed=bool(value)))
for key in ['lambda_0p1','lambda_0p01']:
    folder=OPS/key;plan=read(folder/'plan.json');coefficient=plan['parent_anchor']['coefficient']
    run=Path(read(folder/'execution_ready.json')['out']);result=read(run/'results.json')
    check(key+' completed process',read(folder/'training_exit.json')['exit_code']==0)
    check(key+' same complete public inputs and schedule',result['metadata']['counts']==baseline['counts'] and result['metadata']['schedule_digest']==baseline['schedule_digest'])
    for arm in ['control','treatment']:
        name=key+'_'+arm;path=run/f'checkpoint_{arm}_final.pt';ck=load(path);a=result['arms'][arm]['train']['parent_anchor'];train=result['arms'][arm]['train'];setting=ck['fixed_setting']
        check(name+' exact58 nonhead parameters/buffers',sum(not k.startswith('head.') for k in parent['state_dict'])==58 and all(torch.equal(v,ck['state_dict'][k]) for k,v in parent['state_dict'].items() if not k.startswith('head.')))
        check(name+' original optimizer settings unchanged',{k:v for k,v in setting.items() if k!='parent_anchor'}==baseline['settings'])
        check(name+'93class660steps',ck['classes']==parent['classes'] and len(ck['classes'])==93 and ck['step']==train['steps']==660)
        check(name+' same captured coefficient in stats and checkpoint',a['coefficient']==setting['parent_anchor']['coefficient']==coefficient and a['half_factor']==.5 and a['reduction']=='sum')
        check(name+' detached initial reference starts at zero',a['squared_distance'][0]==a['penalty'][0]==0)
        check(name+' all660 losses finite',all(len(a[k])==660 and all(math.isfinite(v) for v in a[k]) for k in ['cross_entropy','penalty','total_loss','squared_distance']))
        residual=max(abs(t-ce-p) for t,ce,p in zip(a['total_loss'],a['cross_entropy'],a['penalty']))
        penalty_err=max(abs(p-.5*coefficient*d) for p,d in zip(a['penalty'],a['squared_distance']))
        check(name+' coefficient loss arithmetic and rounded history',residual<1e-5 and penalty_err<1e-7 and all(abs(x-y)<5.1e-7 for x,y in zip(train['losses'],a['total_loss'])))
        distance=sum(float(((ck['state_dict'][k].double()-parent['state_dict'][k].double())**2).sum()) for k in ['head.weight','head.bias'])
        check(name+' post-update distance independently matches',abs(distance-a['final_squared_distance'])<1e-12 and distance==setting['parent_anchor']['final_squared_distance'])
        details[name]={'coefficient':coefficient,'squared_distance':distance,'loss_sum_error':residual,'penalty_error':penalty_err,'last_CE':a['cross_entropy'][-1]}
        models[name]={'path':str(path),'sha256':sha(path),'coefficient':coefficient,'arm':arm,'pair':key}
report={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','passed':sum(x['passed'] for x in checks),'total':len(checks),'checks':checks,'details':details,'models':models,'optimizer_steps_in_audit':0,'new_forwards':0,'user_records_read':0}
with (OPS/'REAL_TRAINING_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({k:report[k] for k in ['status','passed','total','details']}))
raise SystemExit(0 if report['status']=='PASS' else 1)
