"""Fix models, all target words and future capture order before new observations exist."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
import hashlib
import json
import random
import urllib.request

OPS=Path(__file__).resolve().parent
WEB=OPS.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
    with p.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
def get(route):
    with urllib.request.urlopen('http://127.0.0.1:8765/api/v1/'+route,timeout=20) as r:return json.load(r)
health,refs=get('health'),get('references')
assert health['camera'] is None and health['camera_session_id'] is None
old=WEB/'ops/head_only_native_pair_20260919'
audit=read(old/'native_pair_result_audit.json')
assert audit['status']=='PASS_OUTPUT_INTEGRITY'
assert read(old/'real_checkpoint_freeze_audit.json')['status']=='PASS_REAL_CHECKPOINT_ENCODER_FREEZE'
training_plan=read(old/'plan.json')
models={'parent':{'path':training_plan['checkpoint'],'sha256':training_plan['checkpoint_sha256'],
                  'role':'Existing visibility parent; reference for preservation'},
        **{k:dict(v,role='Matched head-only '+k) for k,v in audit['checkpoints'].items()}}
assert all(sha(Path(v['path']))==v['sha256'] for v in models.values())
words=[{k:r[k] for k in ('word_id','clip_id','display_name')} for r in refs['items']]
assert len(words)==18 and len({r['word_id'] for r in words})==18
expected={r['word_id'] for r in read(WEB/'runs/union93_candidate_v1/inputs.json')['nikl']}
assert {r['word_id'] for r in words}==expected
slots=[]
rng=random.Random(20260919)
for round_number in range(1,4):
    order=[w['word_id'] for w in words]
    rng.shuffle(order)
    offset=len(slots)
    slots.extend({'slot':offset+i+1,'round':round_number,'word_id':wid} for i,wid in enumerate(order))
assert len(slots)==54 and set(Counter(r['word_id'] for r in slots).values())=={3}
public={'schema':'wis_fresh_self_capture_plan_v1','protocol_id':'self_frozen_20260919_v1',
        'reference_set_id':refs['set_id'],'rounds':3,'words':words,'slots':slots}
public_path=WEB/'web/frozen-self-plan.json'
write(public_path,public)
pins=[Path(v['path']) for v in models.values()]+[
      WEB/'src/wis_lab/service/app.py',WEB/'src/wis_lab/service/neural.py',WEB/'src/wis_lab/core/observation.py',
      WEB.parent/'sign_model_pilot_v1/src/pilot/web_input.py',WEB.parent/'sign_model_pilot_v1/src/pilot/data.py',
      WEB.parent/'sign_model_pilot_v1/src/pilot/model_masked_v1.py',WEB/'src/wis_lab/service/review_log.py',
      WEB/'web/index.html',WEB/'web/app.js',public_path]
plan={**public,'prepared_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(),'models':models,
      'scopes':{'primary':'project18','secondary':'full93'},'participant_id':'self','purpose':'frozen_retest',
      'participant_recordings_in_training':False,'population_scope':'One training-excluded but development-exposed participant; NEW takes only. Not new-person population proof.',
      'capture':{'backend':'Existing API unchanged','store_video':False,'start':'Participant explicit readiness and camera-open button','preview_mirror':'display only','research_model_key':None,
                 'feedback':'No predictions, distance, accuracy or correctness shown during capture; existing backend default output remains stored but is not used to label truth.',
                 'repetitions':3,'planned_slots':54,'stop_early':'Preserve incomplete slots and every failure. No replacement takes in the primary cohort.'},
      'label_source':'Explicit after-take participant self report in original server trial_reviews.jsonl, not requested prompt or model prediction.',
      'analysis':{'same_observations_for_all_models':True,'primary_pair':'treatment versus control (approved nine additional public examples)',
                  'preservation_pair':'treatment and control versus frozen visibility parent',
                  'keep_all_slots':True,'missing_cancelled_failed_or_unconfirmed':'Never silently excluded. Report counts over all54 and separately classification among known actual labels, with each denominator explicit.',
                  'outputs':['per-word counts and recall','macro F1 on actual supported classes with denominator','full confusion','paired gains and losses','observation/inference failures','all missing/unconfirmed slots','stored server latency and offline inference timing separately'],
                  'no_model_change_during_capture_or_analysis':True,'no_final_promotion_threshold':'Development comparison; no automatic promotion or correctness assessment from a class match.'},
      'protected_sha256':{str(p):sha(p) for p in pins},'training_runs':0,'new_capture_trials':0,'heldout_used':False}
write(OPS/'protocol_LOCAL_ONLY.json',plan)
write(OPS/'pre_capture_health.json',{'camera':health['camera'],'camera_session_id':health['camera_session_id'],
      'reference_set_id':refs['set_id'],'prepared_count':refs['prepared_count']})
print(json.dumps({'status':'FROZEN_BEFORE_CAPTURE','words':len(words),'slots':len(slots),'models':list(models),'camera_open':False}))
