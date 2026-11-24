"""LOCAL ONLY: one frozen post-selection replay, never training or selection."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
import sys
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1]
sys.path[:0]=[str(L/'src'),str(L.parent/'sign_model_pilot_v1/src')]
from wis_lab.service.neural import PROJECT_18
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    out=H/'frozen_user_replay';out.mkdir(exist_ok=False)
    lock=read(H/'PUBLIC_SELECTION_CLOSED.json');assert lock['status']=='PUBLIC_SELECTION_CLOSED_CANDIDATE_FROZEN'
    olddir=H.parent/'frozen_public_candidate_validation_20260921'
    oldplan=read(olddir/'plan_LOCAL_ONLY.json');oldreport=read(olddir/'result_LOCAL_ONLY.json')
    assert oldreport['status']=='PASS_FROZEN_POST_SELECTION_DEVELOPMENT_REPLAY'
    assert sha(olddir/'plan_LOCAL_ONLY.json')==oldreport['plan_sha256']
    assert oldreport['passed']==oldreport['total']==60
    oldrows={(r['cohort'],r['id']):r for r in oldreport['rows']}
    proof=read(H/'RESULT_AUDIT.json');assert proof['status']=='PASS_PUBLIC_SELECTION_INTEGRITY'
    train=read(L/'runs/output_retention_candidate_v1/training.json')
    assert oldplan['checkpoint']['sha256']==train['parent']['sha256'] and proof['candidate']==lock['candidate']
    source=oldplan['feature_source'];pins={str(H/'PUBLIC_SELECTION_CLOSED.json'):sha(H/'PUBLIC_SELECTION_CLOSED.json'),str(H/'RESULT_AUDIT.json'):sha(H/'RESULT_AUDIT.json'),str(olddir/'plan_LOCAL_ONLY.json'):sha(olddir/'plan_LOCAL_ONLY.json'),str(olddir/'result_LOCAL_ONLY.json'):sha(olddir/'result_LOCAL_ONLY.json')}
    for key in ['plan','result','features']:
        assert sha(source[key])==source[key+'_sha256'];pins[source[key]]=source[key+'_sha256']
    for path,digest in oldplan['input_and_inference_pins'].items():assert sha(path)==digest;pins[path]=digest
    assert sha(oldplan['encoder_proof']['path'])==oldplan['encoder_proof']['sha256']
    pins[oldplan['encoder_proof']['path']]=oldplan['encoder_proof']['sha256']
    for records in oldplan['cohorts'].values():
        for row in records:assert sha(row['observation'])==row['observation_sha256'];pins[row['observation']]=row['observation_sha256']
    plan=dict(status='FROZEN_BEFORE_NEW_USER_SCORES',at=datetime.now(timezone(timedelta(hours=9))).isoformat(),candidate=lock['candidate'],baseline=oldplan['checkpoint'],
        cohorts=oldplan['cohorts'],feature_source=source,source_pins=pins,new_training=0,new_encoder_forwards=0,new_human_trials=0,candidate_reselection=False,
        limits='Same development-exposed user at two prior sessions; not36 people or a new unseen-person test. User data remain evaluation-only. No tuning/promotion from these results.')
    write(out/'plan_LOCAL_ONLY.json',plan)
    ck=torch.load(lock['candidate']['path'],map_location='cpu',weights_only=True);assert sha(lock['candidate']['path'])==lock['candidate']['sha256']
    parent=torch.load(oldplan['checkpoint']['path'],map_location='cpu',weights_only=True)
    assert sha(oldplan['checkpoint']['path'])==oldplan['checkpoint']['sha256']
    assert all(torch.equal(v,ck['state_dict'][k]) for k,v in parent['state_dict'].items() if not k.startswith('head.'))
    classes=ck['classes'];assert classes==parent['classes'] and len(classes)==93
    target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())];assert len(target)==18
    w=ck['state_dict']['head.weight'].numpy().astype(float);b=ck['state_dict']['head.bias'].numpy().astype(float)
    rows=[];all_scores=[];feature_indices=[];maxerr=0.;checks=0
    with np.load(source['features'],allow_pickle=False) as z:
        for cohort,specs in plan['cohorts'].items():
            assert len(specs)==18
            for spec in specs:
                prior=oldrows[cohort,spec['trial_id']]
                assert prior['truth']==spec['actual_label'] and prior['input_sha256']==spec['observation_sha256']
                feature=z['features'][prior['feature_index']].astype(float)
                score=w@feature+b
                direct=np.array([math.fsum(float(v)*float(ww) for v,ww in zip(feature,wc))+float(bb) for wc,bb in zip(w,b)])
                maxerr=max(maxerr,float(np.abs(score-direct).max()));yi=classes.index(prior['truth'])
                row=dict(cohort=cohort,id=spec['trial_id'],truth=prior['truth'],word_id=spec['actual_word_id'],input_sha256=prior['input_sha256'],feature_index=prior['feature_index'],baseline=prior['scopes'],candidate={})
                for scope,ids in [('full93',list(range(93))),('project18',target)]:
                    order=sorted(ids,key=lambda i:-score[i]);independent_order=sorted(ids,key=lambda i:-direct[i]);assert order==independent_order
                    rival=next(j for j in order if j!=yi)
                    row['candidate'][scope]=dict(prediction=classes[order[0]],correct=order[0]==yi,true_rank=order.index(yi)+1,true_margin=float(score[yi]-score[rival]))
                    checks+=1
                rows.append(row);all_scores.append(score);feature_indices.append(prior['feature_index'])
    assert len(rows)==len(set(feature_indices))==36 and checks==72 and maxerr<1e-10
    summary={}
    for cohort in plan['cohorts']:
        rs=[r for r in rows if r['cohort']==cohort];summary[cohort]={}
        for scope in ['project18','full93']:
            summary[cohort][scope]=dict(n=len(rs),baseline_correct=sum(r['baseline'][scope]['correct'] for r in rs),candidate_correct=sum(r['candidate'][scope]['correct'] for r in rs),
                errors=[dict(truth=r['truth'],prediction=r['candidate'][scope]['prediction']) for r in rs if not r['candidate'][scope]['correct']],
                regressions=[r['truth'] for r in rs if r['baseline'][scope]['correct'] and not r['candidate'][scope]['correct']],
                improvements=[r['truth'] for r in rs if not r['baseline'][scope]['correct'] and r['candidate'][scope]['correct']])
    with (out/'scores_LOCAL_ONLY.npz').open('xb') as f:np.savez_compressed(f,scores=np.stack(all_scores),classes=np.array(classes),feature_indices=np.array(feature_indices))
    for p,digest in pins.items():assert sha(p)==digest
    assert sha(lock['candidate']['path'])==lock['candidate']['sha256']
    result=dict(status='POST_SELECTION_REPLAY_VERIFIED',candidate=lock['candidate'],plan_sha256=sha(out/'plan_LOCAL_ONLY.json'),scores_sha256=sha(out/'scores_LOCAL_ONLY.npz'),summary=summary,rows=rows,
        scalar_scores_checked=3348,scoped_rankings_checked=72,max_scalar_error=maxerr,encoder_forwards=0,training_runs=0,new_human_trials=0,candidate_reselected=False,default_promoted=False,
        interpretation=plan['limits'])
    write(out/'result_LOCAL_ONLY.json',result)
    print(json.dumps(dict(status=result['status'],summary=summary,scalar_scores_checked=3348),ensure_ascii=False))

if __name__=='__main__':main()
