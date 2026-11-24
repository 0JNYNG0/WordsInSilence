"""Affected-case checks only; reuse28 already passed loss/gradient checks."""
import verify_anchor as v
import copy
import json
import hashlib
import traceback
import numpy as np
import torch

error=None;details={}
try:
    plan=json.loads((v.OPS/'plan.json').read_text(encoding='utf-8'))
    torch.manual_seed(191); torch.set_num_threads(2)
    parent=v.MaskedSignTCN(width=8,dilations=(1,),dropout=.5,n_classes=3)
    rng=np.random.default_rng(191); classes=['a','b','c']
    pool=[dict(sample_id='test%d'%i,class_index=i,label=classes[i],x=rng.normal(size=(5,280)).astype(np.float32)) for i in range(3)]
    config=v.setting(); oldF=v.h.F;oldtrainer=v.h.run_head_only_training;oldsample=v.h._sample_batch
    wrapper=v.subject.make_trainer_wrapper(oldtrainer,plan)
    values=[]
    for _ in range(2):
        model=copy.deepcopy(parent)
        stats=wrapper(model,pool,[],classes,config,np.random.default_rng(72),log=lambda _:None)
        values.append((model,stats))
    v.check('shared setting supports both completed arms',len(values)==2 and all(s['steps']==3 for _,s in values))
    v.check('new arm anchors are independent initial models',all(s['parent_anchor']['squared_distance'][0]==0. for _,s in values) and all(torch.equal(t,values[1][0].state_dict()[k]) for k,t in values[0][0].state_dict().items()))
    v.check('audit metadata records successful last arm only',config['parent_anchor']['final_squared_distance']==values[-1][1]['parent_anchor']['final_squared_distance'])
    v.check('CE/sampler/trainer identities remain unchanged',v.h.F is oldF and v.h._sample_batch is oldsample and v.h.run_head_only_training is oldtrainer)
    for key,value in [('coefficient',True),('schema','unexpected'),('origin','zero')]:
        bad=copy.deepcopy(config);bad['parent_anchor'][key]=value; called=[]
        def sentinel(*args,**kwargs): called.append(True);raise RuntimeError('must never enter trainer')
        guarded=v.subject.make_trainer_wrapper(sentinel,plan)
        v.rejects('incompatible previous audit '+key,lambda:guarded(copy.deepcopy(parent),pool,[],classes,bad,None))
        v.check('invalid previous '+key+' did not enter trainer',not called)
    details={'steps_each':3,'initial_anchor_distances':[s['parent_anchor']['squared_distance'][0] for _,s in values]}
except Exception:
    error=traceback.format_exc()
report=dict(status='PASS_SHARED_SETTING' if error is None else 'FAIL',passed=sum(c['passed'] for c in v.checks),total=len(v.checks),checks=v.checks,error=error,details=details,reused_unchanged_checks=28,helper_sha256=hashlib.sha256((v.WEB/'tools_parent_anchor_pair.py').read_bytes()).hexdigest())
with (v.OPS/'verification_shared_setting.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({k:report[k] for k in ('status','passed','total','error')}))
raise SystemExit(0 if error is None else 1)
