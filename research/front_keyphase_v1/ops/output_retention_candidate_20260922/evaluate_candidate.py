"""Evaluate only after the final candidate is frozen; reuse all baseline scores."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1];R=L/'runs/output_retention_candidate_v1'
sys.path[:0]=[str(H),str(L/'src'),str(L.parent/'sign_model_pilot_v1/src'),str(H.parent/'context_training_pair_20260921')]
from execute_candidate import plain,write
from wis_lab.service.neural import PROJECT_18
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    train=read(R/'training.json');assert train['status']=='TRAINING_COMPLETE_CHECKPOINT_FROZEN_BEFORE_DEVELOPMENT_EVALUATION'
    assert not (R/'evaluation.json').exists()
    ep=read(H/'execution_plan.json');policy=read(ep['policy'])
    for p,d in ep['source_pins'].items():assert sha(p)==d,p
    candidate=train['candidate'];assert sha(candidate['path'])==candidate['sha256']
    ck=torch.load(candidate['path'],map_location='cpu',weights_only=True);classes=ck['classes']
    with np.load(policy['data']['feature_path'],allow_pickle=False) as z:
        keys=['base_x','base_y','annotation_dev_x','annotation_dev_y','whole_dev_x','whole_dev_y','public_x','public_y','mp_x','mp_y','pair_context_x','pair_context_y','classes']
        a={k:z[k].copy() for k in keys}
    assert a['classes'].tolist()==classes
    meta=read(H.parent/'context_training_pair_20260921/feature_metadata.json')
    old_eval=read(L/'runs/context_head_pair_v1/evaluation.json')
    oldmap={(r['group'],r['id'],r['condition']):r for r in old_eval['rows']}
    old_train=read(H.parent/'public_prediction_retention_20260922/result.json')
    trainmap={r['id']:r for r in old_train['rows']}
    target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())]
    weight=ck['state_dict']['head.weight'].numpy().astype(float);bias=ck['state_dict']['head.bias'].numpy().astype(float)
    rows=[];scores={};summaries={}
    for group in ['base','annotation_dev','whole_dev','public','mp','pair_context']:
        xs=a[group+'_x'].astype(float);ys=a[group+'_y'];scores[group]=xs@weight.T+bias
        for i,(spec,y) in enumerate(zip(meta[group],ys)):
            cond=spec.get('condition','intact');ident=spec['id'];label=classes[int(y)]
            assert label==spec.get('label',spec.get('true',spec.get('reference_label')))
            row=dict(group=group,id=ident,true=label,condition=cond,models={})
            if 'signer_id' in spec:row['signer_id']=spec['signer_id']
            if group=='base':
                oldrow=trainmap[ident];assert oldrow['true']==label
                for newname,oldname in [('parent','parent'),('context_baseline','treatment')]:
                    row['models'][newname]={'full93':{k:oldrow['models'][oldname][k] for k in ['pred','correct']}}
            else:
                oldrow=oldmap[group,ident,cond];assert oldrow['true']==label
                row['models']['parent']=oldrow['models']['parent'];row['models']['context_baseline']=oldrow['models']['treatment']
            row['models']['candidate']={}
            for scope,inds in [('full93',list(range(93))),('project18',target)]:
                if int(y) not in inds:continue
                order=sorted(inds,key=lambda j:-float(scores[group][i,j]));rival=next(j for j in order if j!=y)
                row['models']['candidate'][scope]=dict(pred=classes[order[0]],correct=bool(order[0]==y),rank=order.index(int(y))+1,margin=float(scores[group][i,y]-scores[group][i,rival]))
            rows.append(row)
        rs=[r for r in rows if r['group']==group];summaries[group]={}
        for name in ['parent','context_baseline','candidate']:
            summaries[group][name]={}
            for cond in sorted({r['condition'] for r in rs}):
                summaries[group][name][cond]={}
                for scope in ['full93','project18']:
                    ss=[r for r in rs if r['condition']==cond and scope in r['models'][name]]
                    if not ss:continue
                    summaries[group][name][cond][scope]=dict(n=len(ss),correct=sum(r['models'][name][scope]['correct'] for r in ss),ground_truth_classes=len({r['true'] for r in ss}))
    with np.load(policy['data']['gram_path'],allow_pickle=False) as z:
        delta=scores['base']-z['parent'];delta-=delta.mean(axis=1,keepdims=True)
        preservation=.5/93*float(np.sum(z['row_weights'][:,None]*delta**2))
    assert abs(preservation-train['final_output_loss_raw'])<1e-10
    def count(group,cond='intact',scope='full93'):return summaries[group]['candidate'][cond][scope]['correct']
    ap=[r for r in rows if r['group']=='annotation_dev' and r['id'].startswith('NIA_SL_SEN0356_')];assert len(ap)==3
    gates=dict(original_train2122=count('base')==2122,
        native_annotation699=count('annotation_dev')>=690,native_target33=count('annotation_dev',scope='project18')>=32,
        native_whole74=count('whole_dev')==74,native_whole_target30=count('whole_dev',scope='project18')==30,
        apology_development3=all(r['models']['candidate'][s]['correct'] for r in ap for s in ['full93','project18']),
        score_drift_reduced=preservation<.7895602850906952)
    for cond,n in [('intact',18),('reviewed_masked',5),('additional_train',9)]:gates['public_'+cond]=all(count('public',cond,s)==n for s in ['full93','project18'])
    for cond in ['whole_clip','provided_annotation']:gates['mp_'+cond]=all(count('mp',cond,s)==7 for s in ['full93','project18'])
    decision='ELIGIBLE_PENDING_INDEPENDENT_AUDIT' if all(gates.values()) else 'REJECTED_BY_PREDECLARED_GATES'
    with (R/'evaluation_scores.npz').open('xb') as f:np.savez_compressed(f,**scores,classes=np.array(classes))
    result=dict(status='PUBLIC_EVALUATION_COMPLETE_PENDING_AUDIT',decision=decision,candidate=candidate,gates=gates,
        summary=summaries,rows=rows,final_output_preservation_loss=preservation,baseline_output_preservation_loss=.7895602850906952,
        scores_sha256=sha(R/'evaluation_scores.npz'),policy_sha256=sha(ep['policy']),training_record_sha256=sha(R/'training.json'),
        baseline_predictions_reused=True,new_encoder_forwards=0,user_arrays_read=0,reserved_arrays_read=0,default_promoted=False)
    write(R/'evaluation.json',result)
    for p,d in ep['source_pins'].items():assert sha(p)==d,p
    assert sha(candidate['path'])==candidate['sha256']
    print(json.dumps(plain(dict(decision=decision,gates=gates,summary={g:v['candidate'] for g,v in summaries.items()},preservation_loss=preservation)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
