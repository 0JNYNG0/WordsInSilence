"""Freeze model/input contracts, without reading active session captures or loading models."""
import ast,hashlib,json
from datetime import datetime
from pathlib import Path
here=Path(__file__).resolve().parent
web=here.parents[1]
source=web/'ops/fresh_self_comparison_20260919/protocol_LOCAL_ONLY.json'
plan=json.loads(source.read_text(encoding='utf-8'))
tree=ast.parse((web/'src/wis_lab/service/neural.py').read_text(encoding='utf-8'))
node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROJECT_18' for t in n.targets))
labels=ast.literal_eval(node)
assert set(labels)=={w['word_id'] for w in plan['words']} and len(labels)==18
models={role:dict(spec,step=220 if role=='parent' else 660) for role,spec in plan['models'].items()}
assert set(models)=={'parent','control','treatment'}
config={'schema':'existing_ui_three_model_config_v1','prepared_at_kst':datetime.now().isoformat(),
        'web_root':str(web),'pilot_root':str(web.parent/'sign_model_pilot_v1'),
        'session_id':'ses_e1e9d0a79bc6','participant_id':'self',
        'minimum_created_epoch':datetime.fromisoformat(plan['prepared_at_kst']).timestamp(),
        'reference_set_id':plan['reference_set_id'],
        'words':{w['word_id']:{'label':labels[w['word_id']],'clip_id':w['clip_id']} for w in plan['words']},
        'models':models,'protected_sha256':dict(plan['protected_sha256']),
        'data_role':'VISIBLE_FEEDBACK_SINGLE_PERSON_DEVELOPMENT; NOT_BLINDED; NOT_EXECUTION_CORRECTNESS',
        'source_model_plan':str(source),'source_model_plan_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'capture_profile':'Existing user-preferred UI, arbitrary attempted count; no54-slot completion claim',
        'cohort_definition':'Every server session trial, fixed only after CLOSED and before offline inference; no performance-based selection',
        'training_uses_user_recordings':False,'automatic_promotion':False}
config['protected_sha256'][str(source)]=config['source_model_plan_sha256']
with (here/'config_LOCAL_ONLY.json').open('x',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'CONFIG_PREPARED_NO_CAPTURE_READ','session_id':config['session_id'],'models':list(models),'words':len(labels),'model_loads':0,'inference':0}))
