"""Post-selection one-model validation via proven unchanged web encoder features."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import torch

OPS=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(OPS/'plan_LOCAL_ONLY.json');assert not (OPS/'result_LOCAL_ONLY.json').exists()
checks=[]
def check(name,value):
    checks.append(dict(name=name,passed=bool(value)))
    if not value:raise AssertionError(name)
for key in ['selection_closed','selection_lock','encoder_proof']:
    check(key+' frozen evidence unchanged',sha(plan[key]['path'])==plan[key]['sha256'])
for p,digest in plan['input_and_inference_pins'].items():check('original input contract/baseline '+Path(p).name,sha(p)==digest)
source=plan['feature_source']
for name in ['plan','result','features']:check('cached '+name+' unchanged',sha(source[name])==source[name+'_sha256'])
check('selected model immutable',sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256'])
ck=torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=True);classes=ck['classes'];W=ck['state_dict']['head.weight'].double().numpy();bias=ck['state_dict']['head.bias'].double().numpy()
original=read(source['plan']);records=read(source['result'])['records'];expected=original['models']['parent']
check('shared encoder proof is for this selected model',read(plan['encoder_proof']['path'])['models']['lambda_0p01_treatment']['sha256']==plan['checkpoint']['sha256'] and read(plan['encoder_proof']['path'])['status']=='PASS')
targets={v['label'] for v in original['words'].values()};check('complete18 target space',len(targets)==18 and len(classes)==93)
scopeids={'project18':[i for i,c in enumerate(classes) if c in targets],'full93':list(range(93))}
rows=[];summaries={};saved_scores=[];indices=[];maxerror=0.;scalar_count=0
with np.load(source['features'],allow_pickle=False) as z:
    for cohort,specs in plan['cohorts'].items():
        matching=[r for r in records if r['cohort']==cohort];byid={r['id']:r for r in matching}
        check(cohort+' exactly18 attested inputs',len(matching)==len(specs)==18 and set(byid)=={s['trial_id'] for s in specs})
        for spec in specs:
            r=byid[spec['trial_id']]
            check(cohort+' observation '+r['id'],sha(spec['observation'])==spec['observation_sha256']==r['input_sha256'] and r['truth']==spec['actual_label'])
            feature=z['features'][r['feature_index']].astype(np.float64);scores=W@feature+bias
            independent=np.array([math.fsum(float(a)*float(b) for a,b in zip(weight,feature))+float(offset) for weight,offset in zip(W,bias)])
            maxerror=max(maxerror,float(np.max(np.abs(scores-independent))));scalar_count+=93
            out={'cohort':cohort,'id':r['id'],'actual_word_id':spec['actual_word_id'],'truth':r['truth'],'input_sha256':r['input_sha256'],'feature_index':r['feature_index'],'scopes':{},'saved_baselines':{k:r['models'][k] for k in ['parent','treatment']}}
            for scope,ids in scopeids.items():
                order=sorted(ids,key=lambda i:-scores[i]);yi=classes.index(r['truth']);other=next(i for i in order if i!=yi)
                out['scopes'][scope]={'prediction':classes[order[0]],'correct':order[0]==yi,'true_rank':order.index(yi)+1,'true_margin':float(scores[yi]-scores[other])}
            saved_scores.append(scores);indices.append(r['feature_index']);rows.append(out)
        summaries[cohort]={}
        for scope in scopeids:
            cohort_rows=[r for r in rows if r['cohort']==cohort]
            value={'n':18,'selected_correct':sum(r['scopes'][scope]['correct'] for r in cohort_rows),'errors':[{'id':r['id'],'truth':r['truth'],'pred':r['scopes'][scope]['prediction']} for r in cohort_rows if not r['scopes'][scope]['correct']]}
            for role in ['parent','treatment']:
                value[role+'_saved_correct']=sum(r['saved_baselines'][role][scope]['correct'] for r in cohort_rows)
                value['gains_vs_'+role]=[r['actual_word_id'] for r in cohort_rows if r['scopes'][scope]['correct'] and not r['saved_baselines'][role][scope]['correct']]
                value['regressions_vs_'+role]=[r['actual_word_id'] for r in cohort_rows if not r['scopes'][scope]['correct'] and r['saved_baselines'][role][scope]['correct']]
            summaries[cohort][scope]=value
check('all36 feature rows unique',len(rows)==len(set(indices))==36)
check('all3348 scores independently agree',scalar_count==3348 and maxerror<1e-10)
check('selection remains unchanged after validation',sha(plan['selection_lock']['path'])==plan['selection_lock']['sha256'] and sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256'])
np.savez_compressed(OPS/'scores_LOCAL_ONLY.npz',scores=np.stack(saved_scores),classes=np.array(classes),feature_indices=np.array(indices))
report={'status':'PASS_FROZEN_POST_SELECTION_DEVELOPMENT_REPLAY','plan_sha256':sha(OPS/'plan_LOCAL_ONLY.json'),'checkpoint':plan['checkpoint'],'summary':summaries,'rows':rows,'passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'scalar_scores_recounted':scalar_count,'max_recount_error':maxerror,'new_encoder_forwards':0,'new_linear_head_evaluations':36,'new_human_trials':0,'optimizer_steps':0,'candidate_reselected':False,'default_promoted':False,'interpretation':'Existing same-user development validation only; separate from completed public-only selection, not a fresh or untouched trial.'}
with (OPS/'result_LOCAL_ONLY.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({k:report[k] for k in ['status','passed','total','summary']},ensure_ascii=False))
