"""Independent scalar score/recount and proposed quadratic identity audit."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

H=Path(__file__).resolve().parent;P=H.parent/'context_training_pair_20260921'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(H/'plan.json');result=read(H/'result.json');meta=read(P/'feature_metadata.json')['base']
checks=[]
def check(name,ok,**detail):
    checks.append(dict(name=name,passed=bool(ok),**detail))
    if not ok:raise AssertionError(name)
check('input_and_result_pins',all(sha(p)==h for p,h in plan['source_pins'].items()) and sha(H/'training_scores_and_gram.npz')==result['arrays_sha256'] and sha(H/'plan.json')==result['plan_sha256'])
maxerr=0.;scalar_count=0;maxmetric=0.
with np.load(P/'public_features.npz',allow_pickle=False) as a,np.load(H/'training_scores_and_gram.npz',allow_pickle=False) as z:
    x=a['base_x'].astype(float);y=a['base_y'];classes=a['classes'].tolist()
    check('training_only_alignment',len(meta)==2122 and all(r['role']=='train' for r in meta) and z['ids'].tolist()==[r['id'] for r in meta] and np.array_equal(y,z['labels']) and all(classes[int(yi)]==r['label'] for r,yi in zip(meta,y)))
    counts=np.bincount(y,minlength=93);omega=z['row_weights'];f=np.column_stack((x,np.ones(len(x))))
    check('93_classes_equal_preservation_weight',all(abs(float(omega[y==c].sum())-1/93)<1e-15 for c in range(93)))
    check('positive_weights_sum_one',bool((omega>0).all()) and abs(float(omega.sum())-1)<1e-14)
    gram=sum((f[y==c].T@f[y==c])/int(counts[c]) for c in range(93))/93
    check('gram_independent_class_average',np.max(np.abs(gram-z['gram']))<1e-12)
    eigen=np.linalg.eigvalsh(gram)
    check('gram_symmetric_psd',np.max(np.abs(gram-gram.T))<1e-12 and eigen[0]>-1e-10,min_eigenvalue=float(eigen[0]))
    measures={};heads={}
    for name,spec in plan['models'].items():
        check(name+'_checkpoint_unchanged',sha(spec['path'])==spec['sha256'])
        ck=torch.load(spec['path'],map_location='cpu',weights_only=True)
        check(name+'_class_order',classes==ck['classes'])
        w=ck['state_dict']['head.weight'].numpy().astype(float);b=ck['state_dict']['head.bias'].numpy().astype(float)
        heads[name]=np.column_stack((w,b));saved=z[name];pr=[];marg=[];ces=[]
        for i,(xx,yy,row) in enumerate(zip(x,y,result['rows'])):
            scores=np.array([math.fsum(float(v)*float(ww) for v,ww in zip(xx,wc))+float(bb) for wc,bb in zip(w,b)])
            maxerr=max(maxerr,float(np.max(np.abs(scores-saved[i]))));scalar_count+=93
            pred=max(range(93),key=lambda j:scores[j]);other=max(scores[j] for j in range(93) if j!=int(yy))
            margin=float(scores[yy]-other);mx=float(scores[pred]);ce=mx+math.log(math.fsum(math.exp(float(v)-mx) for v in scores))-float(scores[yy])
            got=row['models'][name]
            assert got['pred']==classes[pred] and got['correct']==(pred==int(yy))
            maxmetric=max(maxmetric,abs(got['true_margin']-margin),abs(got['cross_entropy']-ce))
            pr.append(pred);marg.append(margin);ces.append(ce)
        measures[name]=dict(pred=np.array(pr),margin=np.array(marg),ce=np.array(ces))
    check('592038_scalar_scores',scalar_count==592038 and maxerr<1e-10,max_error=maxerr)
    check('6366_training_predictions_and_margins',maxmetric<1e-10,max_error=maxmetric)
    context=np.array([r['label'] in result['context_labels'] for r in meta]);masks={'all':np.ones(len(y),bool),'context_labels16':context,'other_labels77':~context,
        'native_original2095':np.array([r.get('method')=='native_annotation_forward' for r in meta]),'nikl18':np.array([r['id'].startswith('NIKL_') for r in meta]),'approved_apology9':np.array([r['id'].startswith('NIA_SL_SEN0356_') for r in meta])}
    for name,m in measures.items():
        for group,mask in masks.items():
            s=result['summary'][name][group]
            assert s['n']==int(mask.sum()) and s['correct']==int(((m['pred']==y)&mask).sum())
            assert s['ground_truth_classes']==len(set(y[mask].tolist()))
            assert abs(s['true_margin_median']-float(np.median(m['margin'][mask])))<1e-10
            assert abs(s['true_margin_min']-float(m['margin'][mask].min()))<1e-10
            assert abs(s['cross_entropy_mean']-float(m['ce'][mask].mean()))<1e-10
        if name=='parent':continue
        for group,mask in masks.items():
            c=result['comparisons'][name]['subsets'][group];d=m['margin']-measures['parent']['margin']
            assert c['true_margin_decreased']==int(((d<0)&mask).sum())
            assert c['predictions_changed']==int(((m['pred']!=measures['parent']['pred'])&mask).sum())
            assert c['right_to_wrong']==int(((measures['parent']['pred']==y)&(m['pred']!=y)&mask).sum())
            assert c['wrong_to_right']==int(((measures['parent']['pred']!=y)&(m['pred']==y)&mask).sum())
            assert abs(c['median_margin_change']-float(np.median(d[mask])))<1e-10
        delta=heads[name]-heads['parent'];delta-=delta.mean(axis=0,keepdims=True)
        loss=math.fsum(float(omega[i])*math.fsum(float(v)**2 for v in (delta@f[i])) for i in range(len(y)))/(2*93)
        check(name+'_direct_centered_score_objective',abs(loss-result['comparisons'][name]['class_balanced_centered_score_preservation_loss'])<1e-12)
    check('all_group_recounts',True)
    # Analytic gradient and invariance of the PROPOSED objective, no optimization step.
    delta=heads['treatment']-heads['parent']
    def loss(d):
        centered=d-d.mean(axis=0,keepdims=True)
        return float(np.sum((centered@gram)*centered))/(2*93)
    analytic=((delta-delta.mean(axis=0,keepdims=True))@gram)/93
    errors=[]
    for row,col in [(0,0),(0,64),(10,12),(20,40),(45,63),(92,64)]:
        plus=delta.copy();minus=delta.copy();plus[row,col]+=1e-5;minus[row,col]-=1e-5
        errors.append(abs((loss(plus)-loss(minus))/(2e-5)-analytic[row,col]))
    check('proposed_gradient_finite_difference_six_entries',max(errors)<1e-8,max_error=max(errors))
    common=np.linspace(-.3,.3,65)[None,:]
    check('common_score_shift_invariance',abs(loss(delta+common)-loss(delta))<1e-12 and loss(np.repeat(common,93,axis=0))<1e-25)
    check('zero_at_parent',loss(np.zeros_like(delta))==0)
out=dict(status='PASS',checks=checks,scalar_scores_checked=scalar_count,predictions_checked=6366,max_scalar_error=maxerr,
         numerical_gradient_max_error=max(errors),new_training_runs=0,new_encoder_forwards=0,user_arrays_read=0,development_arrays_read=0,reserved_arrays_read=0,
         result_sha256=sha(H/'result.json'),arrays_sha256=sha(H/'training_scores_and_gram.npz'),source_plan_sha256=sha(H/'plan.json'))
with (H/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
