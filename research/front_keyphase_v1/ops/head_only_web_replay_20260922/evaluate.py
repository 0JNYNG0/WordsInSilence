"""Local frozen head replay using already verified features, never training."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1]
sys.path[:0]=[str(L/'src'),str(L.parent/'sign_model_pilot_v1/src')]
from wis_lab.service.neural import PROJECT_18
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def normal(scopes):
    return {s:dict(prediction=v['prediction'],correct=v['correct'],true_rank=v['true_rank'],
             true_margin=v.get('true_margin',v.get('true_minus_best_other'))) for s,v in scopes.items()}

def main():
    lock_path=L/'ops/motion_position_pair_20260922/NEXT_SINGLE_TASK.json';lock=read(lock_path)
    upath=Path(lock['reuse']['user_existing_results']);kpath=Path(lock['reuse']['public_two_results'])
    uplan=read(upath/'plan_LOCAL_ONLY.json');ures=read(upath/'result_LOCAL_ONLY.json')
    source=uplan['feature_source'];kplan=read(kpath/'plan.json');kres=read(kpath/'result.json')
    assert ures['status']=='POST_SELECTION_REPLAY_VERIFIED'
    assert ures['plan_sha256']==sha(upath/'plan_LOCAL_ONLY.json')
    assert kres['plan_sha256']==sha(kpath/'plan.json')
    assert kres['arrays_sha256']==sha(kpath/'features_and_scores.npz')
    assert sha(source['features'])==source['features_sha256']
    for key in ['plan','result']:assert sha(source[key])==source[key+'_sha256']
    specs=dict(candidate=lock['candidate'],b8=uplan['baseline'],d1=uplan['candidate'])
    assert specs['b8']['sha256']==kplan['models']['pre_context_parent']['sha256']
    assert specs['d1']['sha256']==kplan['models']['output_retention_candidate']['sha256']
    checkpoints={}
    for name,spec in specs.items():
        assert sha(spec['path'])==spec['sha256'];checkpoints[name]=torch.load(spec['path'],map_location='cpu',weights_only=True)
    candidate=checkpoints['candidate'];classes=candidate['classes'];assert len(classes)==93
    nonhead=[k for k in candidate['state_dict'] if not k.startswith('head.')];assert len(nonhead)==58
    for ck in checkpoints.values():
        assert ck['classes']==classes and all(torch.equal(candidate['state_dict'][k],ck['state_dict'][k]) for k in nonhead)
    target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())];assert len(target)==18
    cohort_records=uplan['cohorts'];assert all(len(rs)==18 for rs in cohort_records.values()) and len(cohort_records)==2
    assert len(kplan['rows'])==2 and {r['id'] for r in kplan['rows']}=={'KSL77_00_48','KSL77_00_16'}
    paths=[lock_path,upath/'plan_LOCAL_ONLY.json',upath/'result_LOCAL_ONLY.json',kpath/'plan.json',kpath/'result.json',kpath/'INDEPENDENT_AUDIT.json',kpath/'SEMANTIC_CONFIRMATION.json',kpath/'features_and_scores.npz',H/'evaluate.py']+[Path(source[k]) for k in ['plan','result','features']]
    plan=dict(status='FROZEN_BEFORE_CANDIDATE_WEB_SCORES',at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
       models=specs,encoder_tensors_identical=58,source_pins={str(q):sha(q) for q in paths},
       user_features=source['features'],public_features=str(kpath/'features_and_scores.npz'),
       user_result=str(upath/'result_LOCAL_ONLY.json'),public_result=str(kpath/'result.json'),public_plan=str(kpath/'plan.json'),
       scopes=['full93','project18'],user_records=36,public_records=2,
       new_training=0,new_extraction=0,new_encoder_forwards=0,
       rule='Fixed post-selection diagnostic only; no score-driven tuning,reselection,default promotion or new-person claim.')
    write(H/'plan_LOCAL_ONLY.json',plan)
    # First feature reads/new candidate scores occur only below the written lock.
    features=[];rows=[];uspec={(cohort,r['trial_id']):r for cohort,rs in cohort_records.items() for r in rs}
    with np.load(source['features'],allow_pickle=False) as z:
        for r in ures['rows']:
            spec=uspec[r['cohort'],r['id']]
            assert spec['actual_label']==r['truth'] and spec['observation_sha256']==r['input_sha256']
            features.append(z['features'][r['feature_index']].astype(float))
            rows.append(dict(cohort=r['cohort'],id=r['id'],truth=r['truth'],input_sha256=r['input_sha256'],
                feature_source='user',feature_index=r['feature_index'],models=dict(b8=normal(r['baseline']),d1=normal(r['candidate']))))
    with np.load(kpath/'features_and_scores.npz',allow_pickle=False) as z:
        assert z['classes'].tolist()==classes
        for i,spec in enumerate(kplan['rows']):
            assert z['ids'][i]==spec['id'];features.append(z['features'][i].astype(float));models={}
            for name,oldname in [('b8','pre_context_parent'),('d1','output_retention_candidate')]:
                prior=next(r for r in kres['rows'] if r['id']==spec['id'] and r['model']==oldname)
                assert prior['label']==spec['label'];models[name]=normal(prior['scopes'])
            rows.append(dict(cohort='public_confirmed2',id=spec['id'],truth=spec['label'],input_sha256=spec['input_sha256'],feature_source='public',feature_index=i,models=models))
    features=np.stack(features);assert features.shape==(38,64)
    w=candidate['state_dict']['head.weight'].numpy().astype(float);b=candidate['state_dict']['head.bias'].numpy().astype(float)
    scores=features@w.T+b
    for r,score in zip(rows,scores):
        yi=classes.index(r['truth']);r['models']['candidate']={}
        for scope,indices in [('full93',list(range(93))),('project18',target)]:
            order=sorted(indices,key=lambda j:(-float(score[j]),j));rival=next(j for j in order if j!=yi)
            r['models']['candidate'][scope]=dict(prediction=classes[order[0]],correct=bool(order[0]==yi),true_rank=order.index(yi)+1,true_margin=float(score[yi]-score[rival]))
    summary={}
    for cohort in dict.fromkeys(r['cohort'] for r in rows):
        rs=[r for r in rows if r['cohort']==cohort];summary[cohort]={}
        for scope in ['full93','project18']:
            value=dict(n=len(rs),correct={name:sum(r['models'][name][scope]['correct'] for r in rs) for name in specs},
              candidate_errors=[dict(truth=r['truth'],prediction=r['models']['candidate'][scope]['prediction']) for r in rs if not r['models']['candidate'][scope]['correct']],changes={})
            for name in ['b8','d1']:
                value['changes'][name]=dict(gains=[r['truth'] for r in rs if r['models']['candidate'][scope]['correct'] and not r['models'][name][scope]['correct']],
                   regressions=[r['truth'] for r in rs if not r['models']['candidate'][scope]['correct'] and r['models'][name][scope]['correct']])
            summary[cohort][scope]=value
    for q,d in plan['source_pins'].items():assert sha(q)==d
    with (H/'scores_LOCAL_ONLY.npz').open('xb') as f:np.savez_compressed(f,scores=scores,classes=np.array(classes),ids=np.array([r['id'] for r in rows]))
    result=dict(status='COMPLETE_PENDING_INDEPENDENT_AUDIT',at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
       candidate=specs['candidate'],plan_sha256=sha(H/'plan_LOCAL_ONLY.json'),scores_sha256=sha(H/'scores_LOCAL_ONLY.npz'),summary=summary,rows=rows,
       user_feature_records_read=36,user_raw_observations_read=0,public_feature_records_read=2,new_encoder_forwards=0,
       new_training=0,reserved_arrays_read=0,other34_ksl_read=0,default_promoted=False,candidate_reselected=False,
       limits=['same development-exposed user,2 sessions,not36 people','public2 signer identities unresolved','no execution-correctness or final heldout generalization claim'])
    write(H/'result_LOCAL_ONLY.json',result)
    print(json.dumps(dict(status=result['status'],summary=summary),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
