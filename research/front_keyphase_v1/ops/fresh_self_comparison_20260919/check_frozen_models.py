"""Verify the three planned checkpoints can use the same unchanged web adapter, no inference."""
from pathlib import Path
import copy
import hashlib
import json
import sys

OPS=Path(__file__).resolve().parent
WEB=OPS.parents[1]
sys.path[:0]=[str(WEB/'src'),str(WEB.parent/'sign_model_pilot_v1/src')]
from wis_lab.service import neural
plan=json.loads((OPS/'protocol_LOCAL_ONLY.json').read_text(encoding='utf-8'))
registry=copy.deepcopy(neural.MODELS)
default=neural.DEFAULT_MODEL
classes=[]
report={}
for role,model in plan['models'].items():
    key='offline_fresh_fixed_'+role
    assert key not in neural.MODELS
    assert hashlib.sha256(Path(model['path']).read_bytes()).hexdigest()==model['sha256']
    neural.MODELS[key]={**model,'project_words':dict(neural.PROJECT_18),'model_id':key,'role':'offline verification only','status_label':'오프라인 준비'}
    loaded=neural._load(key)
    assert loaded['status']=='READY' and loaded['checkpoint_sha256']==model['sha256']
    classes.append(loaded['classes'])
    report[role]={'status':loaded['status'],'checkpoint_sha256':loaded['checkpoint_sha256'],'checkpoint_step':loaded['checkpoint_step'],'class_count':len(loaded['classes'])}
assert all(c==classes[0] and len(c)==93 for c in classes)
assert set(neural.PROJECT_18)=={r['word_id'] for r in plan['words']}
assert neural.DEFAULT_MODEL==default and all(neural.MODELS[k]==v for k,v in registry.items())
for role in plan['models']:
    neural.MODELS.pop('offline_fresh_fixed_'+role)
    neural._CACHES.pop('offline_fresh_fixed_'+role,None)
assert neural.MODELS==registry
out={'status':'PASS_THREE_FROZEN_MODELS_READY','models':report,'same_class_order':True,
     'inference_calls':0,'training_steps':0,'permanent_registration_changed':False,'camera_opened':False}
with (OPS/'model_readiness.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
