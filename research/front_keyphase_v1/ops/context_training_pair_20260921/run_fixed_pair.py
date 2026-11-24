"""Operational execution and evaluation of the one predeclared public pair."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import hashlib
import json
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;LAB=HERE.parents[1]
sys.path[:0]=[str(HERE),str(LAB/'src'),str(LAB.parent/'sign_model_pilot_v1/src')]
from generated_v2 import run_pair
from wis_lab.service.neural import PROJECT_18
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def write(path,v):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(plain(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    policy=read(HERE/'comparison_policy.json');feature=read(HERE/'FEATURE_PREPARATION.json')
    fa=read(HERE/'FEATURE_AUDIT.json');code=read(HERE/'generated_v2_AUDIT.json')
    assert fa['status']==code['status']=='PASS'
    assert sha(HERE/'public_features.npz')==fa['feature_cache_sha256']==feature['arrays_sha256']
    assert sha(HERE/'generated_v2.py')==code['module_sha256']
    assert sha(policy['parent']['path'])==policy['parent']['sha256']
    parent=torch.load(policy['parent']['path'],map_location='cpu',weights_only=True)
    run=LAB/'runs/context_head_pair_v1';run.mkdir(exist_ok=False)
    pins={str(HERE/n):sha(HERE/n) for n in ['comparison_policy.json','feature_plan.json','feature_metadata.json','public_features.npz','FEATURE_AUDIT.json','generated_v2.py','generated_v2_AUDIT.json','run_fixed_pair.py']}
    write(run/'preflight.json',dict(status='READY_FOR_SINGLE_PAIR',pins=pins,parent=policy['parent'],user_inputs=0,reserved_inputs=0))
    with np.load(HERE/'public_features.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    meta=read(HERE/'feature_metadata.json');classes=a['classes'].tolist();assert classes==parent['classes']
    w0=parent['state_dict']['head.weight'].numpy().copy();b0=parent['state_dict']['head.bias'].numpy().copy()
    kw={k:policy['settings'][k] for k in ['steps','batch_size','seed','learning_rate','weight_decay','anchor_coefficient']}
    torch.set_num_threads(2);assert torch.get_default_device()==torch.device('cpu')
    started=time.time();print('Starting one control/treatment pair, 660 updates each',flush=True)
    result=run_pair(a['base_x'],a['base_y'],a['pair_annotation_x'],a['pair_context_x'],a['pair_context_y'],w0,b0,**kw)
    elapsed=time.time()-started;print('Both arms finished',round(elapsed,2),'seconds',flush=True)
    arrays={**result['schedule']};model_specs={};settings=result['settings']
    for arm in ['control','treatment']:
        val=result[arm]
        for k in ['weight','bias','pre_update_losses']:arrays[arm+'_'+k]=val[k]
        ck=copy.deepcopy(parent);ck['state_dict']['head.weight']=torch.from_numpy(val['weight'].copy());ck['state_dict']['head.bias']=torch.from_numpy(val['bias'].copy())
        ck['context_pair_provenance']=dict(parent_sha256=policy['parent']['sha256'],arm=arm,policy_sha256=sha(HERE/'comparison_policy.json'),settings=plain(settings),encoder_frozen=True)
        path=run/('checkpoint_'+arm+'_final.pt');torch.save(ck,path)
        model_specs[arm]=dict(path=str(path),sha256=sha(path),final_anchor_distance_squared=val['final_anchor_distance_squared'])
    with (run/'training_arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    write(run/'training.json',dict(status='FINISHED_PENDING_INDEPENDENT_AUDIT',at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
          models=model_specs,settings=settings,seconds=elapsed,training_array_sha256=sha(run/'training_arrays.npz'),
          paired_updates=660,actual_optimizer_updates=1320,project_training_runs=2,retries=0,user_training_records=0))
    models={'parent':dict(weight=w0,bias=b0),'control':result['control'],'treatment':result['treatment']}
    target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())]
    rows=[];stats={};scores={}
    for group in ['annotation_dev','whole_dev','public','mp','pair_context','pair_annotation']:
        source='pair_context' if group=='pair_annotation' else group
        xs=a[group+'_x'];ys=a[source+'_y'];specs=meta[source]
        stats[group]={}
        for name,model in models.items():
            scores[group+'__'+name]=xs.astype(np.float64)@model['weight'].astype(np.float64).T+model['bias'].astype(np.float64)
        for i,(row,yi) in enumerate(zip(specs,ys)):
            condition=row.get('condition','intact')
            entry=dict(group=group,id=row['id'],condition=condition,true=classes[int(yi)],models={})
            if 'signer_id' in row:entry['signer_id']=row['signer_id']
            for name in models:
                output={};logit=scores[group+'__'+name][i]
                for scope,ids in [('full93',list(range(93))),('project18',target)]:
                    if int(yi) not in ids:continue
                    order=sorted(ids,key=lambda j:-float(logit[j]));rival=next(j for j in order if j!=yi)
                    output[scope]=dict(pred=classes[order[0]],correct=order[0]==yi,rank=order.index(int(yi))+1,margin=float(logit[yi]-logit[rival]))
                entry['models'][name]=output
            rows.append(entry)
        group_rows=[r for r in rows if r['group']==group]
        for name in models:
            stats[group][name]={}
            for cond in sorted({r['condition'] for r in group_rows}):
                stats[group][name][cond]={}
                for scope in ['full93','project18']:
                    rs=[r for r in group_rows if r['condition']==cond and scope in r['models'][name]]
                    stats[group][name][cond][scope]=dict(n=len(rs),correct=sum(r['models'][name][scope]['correct'] for r in rs),ground_truth_classes=len({r['true'] for r in rs}))
    def cnt(group,name,condition='intact',scope='full93'):return stats[group][name][condition][scope]['correct']
    gates={
        'full_context74_improves_parent_and_control':cnt('whole_dev','treatment')>max(cnt('whole_dev','parent'),cnt('whole_dev','control')),
        'native_annotation699_at_least690':cnt('annotation_dev','treatment')>=690,
        'native_target33_at_least32':cnt('annotation_dev','treatment',scope='project18')>=32,
    }
    ap=[r for r in rows if r['group']=='annotation_dev' and r['id'].startswith('NIA_SL_SEN0356_')]
    assert len(ap)==3
    gates['apology_development3_retained']=all(r['models']['treatment']['full93']['correct'] for r in ap)
    for condition,n in [('intact',18),('reviewed_masked',5),('additional_train',9)]:
        gates['public_'+condition+'_retained']=all(cnt('public','treatment',condition,scope)==n for scope in ['full93','project18'])
    gates['mp_annotation7_retained']=all(cnt('mp','treatment','provided_annotation',scope)==7 for scope in ['full93','project18'])
    gates['mp_whole_at_least5']=all(cnt('mp','treatment','whole_clip',scope)>=5 for scope in ['full93','project18'])
    decision='ELIGIBLE_PENDING_INDEPENDENT_AUDIT' if all(gates.values()) else 'REJECTED_BY_PREDECLARED_GATES'
    with (run/'evaluation_scores.npz').open('xb') as f:np.savez_compressed(f,classes=np.array(classes),**scores)
    report=dict(status='COMPLETE_PENDING_INDEPENDENT_AUDIT',decision=decision,gates=gates,summary=stats,rows=rows,
                parent=policy['parent'],models=model_specs,evaluation_scores_sha256=sha(run/'evaluation_scores.npz'),
                user_records_or_scores_read=0,reserved_arrays_read=0,new_encoder_forwards_during_training_or_evaluation=0,
                default_model_promoted=False,policy_sha256=sha(HERE/'comparison_policy.json'))
    write(run/'evaluation.json',report)
    for path,h in pins.items():assert sha(path)==h,path
    assert sha(policy['parent']['path'])==policy['parent']['sha256']
    print(json.dumps(dict(decision=decision,gates=gates,summary=stats),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
