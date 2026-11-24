"""Read-only scientific diagnosis on the original public TRAINING features only."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
import re
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1];PRE=L/'ops/context_training_pair_20260921'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    with (H/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    assert not (H/'plan.json').exists()
    policy=read(PRE/'comparison_policy.json');closed=read(PRE/'CLOSEOUT.json')
    trainpath=L/'runs/context_head_pair_v1/training.json'
    assert sha(trainpath)==closed['artifact_pins'][str(trainpath)]
    trained=read(trainpath)
    specs={'parent':policy['parent'],**trained['models']}
    feature=PRE/'public_features.npz';metap=PRE/'feature_metadata.json'
    assert sha(feature)=='1f94183ef8d71cb1c50baf4770937c871fb1e53f652f81fe120b596c4f7cf386'
    assert sha(metap)=='815dff33a2d084b3fdf88e275b9d8b500dda1c70e363d8c219bb3df325b62525'
    for m in specs.values():assert sha(m['path'])==m['sha256']
    old_audit=read(PRE/'RESULT_AUDIT.json');assert old_audit['status']=='PASS_OUTPUT_INTEGRITY'
    metadata=read(metap)
    rows=metadata['base'];context_labels=sorted({r['label'] for r in metadata['pair_context']})
    assert len(rows)==2122 and len(context_labels)==16
    assert all(r['role']=='train' for r in rows)
    native=[r for r in rows if '_REAL' in r['id']]
    assert {int(re.search(r'_REAL(\d+)_',r['id']).group(1)) for r in native}==set(range(1,10))
    plan=dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),purpose='Diagnose original public training predictions and formulate an output-preservation objective; no new training.',
        models=specs,source_pins={str(p):sha(p) for p in [feature,metap,trainpath,PRE/'RESULT_AUDIT.json',PRE/'comparison_policy.json']},
        allowed_array_keys=['base_x','base_y','classes'],training_records=2122,training_classes=93,
        analysis=['Top1, true-versus-best-other margin, cross entropy, changes relative to parent',
                  'Whole-context16-label versus other77-label subsets; native2095/public18/approved9 domains',
                  'Class-balanced centered score drift and exact frozen-feature Gram-matrix form'],
        no_threshold_selection=True,training_runs=0,encoder_forwards=0,user_arrays_read=0,development_arrays_read=0,reserved_arrays_read=0,
        provenance_note='Existing metadata includes development records but only base and paired TRAIN metadata are used; only listed NPZ keys are decompressed.')
    save('plan.json',plan)
    with np.load(feature,allow_pickle=False) as z:
        x=z['base_x'].astype(np.float64);y=z['base_y'].copy();classes=z['classes'].tolist()
    assert x.shape==(2122,64) and np.isfinite(x).all() and set(y)==set(range(93))
    assert all(classes[int(yi)]==r['label'] for yi,r in zip(y,rows))
    models={name:torch.load(s['path'],map_location='cpu',weights_only=True) for name,s in specs.items()}
    assert all(m['classes']==classes for m in models.values())
    heads={k:np.concatenate((m['state_dict']['head.weight'].numpy().astype(float),m['state_dict']['head.bias'].numpy().astype(float)[:,None]),axis=1) for k,m in models.items()}
    features=np.column_stack((x,np.ones(len(x))))
    scores={name:features@head.T for name,head in heads.items()}
    stats={};metrics={};n=len(y);ci=np.arange(n)
    masks={'all':np.ones(n,bool),'context_labels16':np.isin(y,[classes.index(c) for c in context_labels]),'other_labels77':~np.isin(y,[classes.index(c) for c in context_labels]),
           'native_original2095':np.array([r.get('method')=='native_annotation_forward' for r in rows]),
           'nikl18':np.array([r['id'].startswith('NIKL_') for r in rows]),
           'approved_apology9':np.array([r['id'].startswith('NIA_SL_SEN0356_') for r in rows])}
    assert [int(masks[k].sum()) for k in ['native_original2095','nikl18','approved_apology9']]==[2095,18,9]
    counts=np.bincount(y,minlength=93);weights=1./(93*counts[y])
    assert abs(float(weights.sum())-1)<1e-14
    gram=features.T@(weights[:,None]*features)
    for name,s in scores.items():
        pred=s.argmax(axis=1);others=s.copy();others[ci,y]=-np.inf
        margin=s[ci,y]-others.max(axis=1)
        mx=s.max(axis=1);logp=s-mx[:,None]-np.log(np.exp(s-mx[:,None]).sum(axis=1))[:,None]
        ce=-logp[ci,y];correct=pred==y
        metrics[name]=dict(pred=pred,margin=margin,ce=ce,correct=correct,logp=logp)
        stats[name]={g:dict(n=int(mask.sum()),ground_truth_classes=len(set(y[mask].tolist())),correct=int(correct[mask].sum()),
                         true_margin_min=float(margin[mask].min()),true_margin_median=float(np.median(margin[mask])),
                         cross_entropy_mean=float(ce[mask].mean())) for g,mask in masks.items()}
    comparisons={};result_rows=[]
    for name in ['control','treatment']:
        shift=scores[name]-scores['parent'];shift-=shift.mean(axis=1,keepdims=True)
        row_mse=np.square(shift).mean(axis=1)
        delta=heads[name]-heads['parent'];delta-=delta.mean(axis=0,keepdims=True)
        direct=.5*float(np.dot(weights,row_mse));equivalent=.5/93*float(np.sum((delta@gram)*delta))
        assert abs(direct-equivalent)<1e-12
        kl=np.sum(np.exp(metrics['parent']['logp'])*(metrics['parent']['logp']-metrics[name]['logp']),axis=1)
        changed=metrics['parent']['pred']!=metrics[name]['pred']
        lost=metrics['parent']['correct']&~metrics[name]['correct'];gained=~metrics['parent']['correct']&metrics[name]['correct']
        decreased=metrics[name]['margin']<metrics['parent']['margin']
        comparisons[name]=dict(class_balanced_centered_score_preservation_loss=direct,gram_equivalent_loss=equivalent,
            current_parameter_penalty=.5*.01*float(np.square(heads[name]-heads['parent']).sum()),
            class_balanced_teacher_kl_t1=float(np.dot(weights,kl)),
            centered_score_rms=float(np.sqrt(row_mse.mean())),
            subsets={g:dict(n=int(mask.sum()),predictions_changed=int((changed&mask).sum()),right_to_wrong=int((lost&mask).sum()),wrong_to_right=int((gained&mask).sum()),
                          true_margin_decreased=int((decreased&mask).sum()),median_margin_change=float(np.median((metrics[name]['margin']-metrics['parent']['margin'])[mask]))) for g,mask in masks.items()})
    for i,r in enumerate(rows):
        result_rows.append(dict(id=r['id'],true=r['label'],true_in_context16=r['label'] in context_labels,
            models={name:dict(pred=classes[int(m['pred'][i])],correct=bool(m['correct'][i]),true_margin=float(m['margin'][i]),cross_entropy=float(m['ce'][i])) for name,m in metrics.items()}))
    byclass={label:{name:dict(n=int((y==j).sum()),correct=int((m['correct']&(y==j)).sum()),margin_median=float(np.median(m['margin'][y==j]))) for name,m in metrics.items()} for j,label in enumerate(classes)}
    npz=H/'training_scores_and_gram.npz'
    with npz.open('xb') as f:np.savez_compressed(f,**scores,classes=np.array(classes),ids=np.array([r['id'] for r in rows]),labels=y,row_weights=weights,gram=gram)
    for path,digest in plan['source_pins'].items():assert sha(path)==digest
    for m in specs.values():assert sha(m['path'])==m['sha256']
    result=dict(status='COMPLETE_PENDING_INDEPENDENT_AUDIT',plan_sha256=sha(H/'plan.json'),script_sha256=sha(__file__),arrays_sha256=sha(npz),
                summary=stats,comparisons=comparisons,rows=result_rows,by_class=byclass,context_labels=context_labels,
                affine_head_outputs=6366,encoder_forwards=0,training_runs=0,user_arrays_read=0,development_arrays_read=0,reserved_arrays_read=0,
                gram_identity='G = sum_i (1/(93*n_class_i)) f_i f_i.T, f=[frozen_feature,1]; Delta=A-A_parent centered across output classes; L=0.5/93*sum((Delta@G)*Delta)',
                interpretation_limits=['Training results cannot establish generalization.','Centered-score preservation is a proposed common objective, not a proven fix.','Source labels and public reference semantics are unchanged.'])
    save('result.json',result)
    print(json.dumps(dict(summary=stats,comparisons=comparisons),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
