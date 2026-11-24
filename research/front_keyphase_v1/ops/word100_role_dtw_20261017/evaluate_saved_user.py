import sys,json,hashlib
from pathlib import Path
U=Path(__file__).resolve().parent;B=U.parents[1]
sys.path[:0]=[str(B/'src'),str(B.parents[1]/'research/sign_model_pilot_v1/src')]
from wis_lab.service import word100_web_analysis as W
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text('utf-8'))
assert not (U/'USER_REPLAY.json').exists(),'Reuse saved replay; do not tune/retry from outcome'
frozen=read(U/'FROZEN_READY.json');assert frozen['status']=='PUBLIC_VALIDATION_PASS_BEFORE_USER_REPLAY'
for p,h in frozen['code_sha256'].items():assert sha(p)==h
assert sha(U/'ROLE_BANK.json')==frozen['bank_sha256']
prior=read(B/'ops/word100_c_user_paired_20261017/RESULT.json')
results=[]
for x in prior['rows']:
    r=W.analysis(prior['scope']['session'],x['trial_id']);new=r['analysis']['training_comparison'];old=r['analysis']['legacy_training_comparison']
    assert r['practice100']['requested_rank_all100']==x['C']['rank']
    if len(new.get('required_sides',[]))==2:
        for a,b in zip(new['metrics'],old['metrics']):assert a['value']==b['value'] and a['ratio']==b['ratio']
    results.append({'trial_id':x['trial_id'],'label':x['label'],'C_rank':x['C']['rank'],'old_status':old['status'],'new_status':new['status'],
                    'required_sides':new.get('required_sides'),'metrics':new['metrics'],'same_TCN_saved_rank':True})
    print(x['label'],old['status'],'->',new['status'],flush=True)
    if x['label']=='멀다':(U/'WEB_FIXTURE.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),'utf-8')
for p,h in prior['source_hashes'].items():assert sha(B/p)==h,p
out={'status':'COMPLETE_FIXED_RULE_EVALUATION','bank_sha256':frozen['bank_sha256'],'trials':results,
     'old_OK':sum(x['old_status']=='OK' for x in results),'new_OK':sum(x['new_status']=='OK' for x in results),
     'original_sources_and_models_unchanged':True,'training':0,'tuning':0,'TCN_inference':0,
     'meaning':'calculability and changed measured differences, not classification accuracy or validated sign correctness'}
(U/'USER_REPLAY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps({'old_OK':out['old_OK'],'new_OK':out['new_OK']},ensure_ascii=False))
