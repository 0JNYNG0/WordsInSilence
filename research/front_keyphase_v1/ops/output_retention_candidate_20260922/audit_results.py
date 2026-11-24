"""Independent checkpoint, score, baseline reuse and gate audit without retraining."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import sys
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1];R=L/'runs/output_retention_candidate_v1'
sys.path[:0]=[str(L/'src'),str(L.parent/'sign_model_pilot_v1/src')]
from wis_lab.service.neural import PROJECT_18
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,ok,**detail):
    checks.append(dict(name=name,passed=bool(ok),**detail))
    if not ok:raise AssertionError(name)
ep=read(H/'execution_plan.json');policy=read(ep['policy']);train=read(R/'training.json');report=read(R/'evaluation.json');pre=read(R/'preflight.json')
check('all_training_input_pins',all(sha(p)==d for p,d in pre['input_pins'].items()))
check('scores_training_arrays_checkpoint_pins',sha(R/'evaluation_scores.npz')==report['scores_sha256'] and sha(R/'training_arrays.npz')==train['training_arrays_sha256'] and sha(train['candidate']['path'])==train['candidate']['sha256'])
parent=torch.load(train['parent']['path'],map_location='cpu',weights_only=True);ck=torch.load(train['candidate']['path'],map_location='cpu',weights_only=True)
nonhead=[k for k in parent['state_dict'] if not k.startswith('head.')]
check('58_encoder_parameters_and_buffers_frozen',len(nonhead)==58 and all(torch.equal(parent['state_dict'][k],ck['state_dict'][k]) for k in nonhead))
check('architecture_class_order_unchanged',all(parent[k]==ck[k] for k in ['classes','width','dilations','kernel_size','dropout','n_classes']))
check('training_frozen_before_evaluation_record',sha(R/'training.json')==report['training_record_sha256'] and Path(train['candidate']['path']).stat().st_mtime_ns < (R/'evaluation.json').stat().st_mtime_ns)
meta=read(H.parent/'context_training_pair_20260921/feature_metadata.json')
old_eval=read(L/'runs/context_head_pair_v1/evaluation.json');oldmap={(r['group'],r['id'],r['condition']):r for r in old_eval['rows']}
old_training=read(H.parent/'public_prediction_retention_20260922/result.json');oldtrainmap={r['id']:r for r in old_training['rows']}
maxerr=0.;nscore=0;nranks=0;maxmargin=0.;summary={};changes={}
with np.load(policy['data']['feature_path'],allow_pickle=False) as a,np.load(R/'evaluation_scores.npz',allow_pickle=False) as z,np.load(R/'training_arrays.npz',allow_pickle=False) as t,np.load(policy['data']['sample_schedule_path'],allow_pickle=False) as old_t,np.load(policy['data']['gram_path'],allow_pickle=False) as g:
    classes=ck['classes'];target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())]
    check('21120_saved_draws_unchanged',all(np.array_equal(t[k],old_t[k]) and t[k].shape==(660,16) for k in ['base_indices','pair_indices']))
    w=ck['state_dict']['head.weight'].numpy().astype(float);b=ck['state_dict']['head.bias'].numpy().astype(float)
    pw=parent['state_dict']['head.weight'].numpy().astype(float);pb=parent['state_dict']['head.bias'].numpy().astype(float)
    check('head_matches_training_array',np.array_equal(w,t['weight']) and np.array_equal(b,t['bias']))
    loss=t['pre_update_losses'];check('660_finite_preupdate_loss_rows',loss.shape==(660,4) and np.isfinite(loss).all() and np.allclose(loss[:,:3].sum(axis=1),loss[:,3],rtol=2e-7,atol=1e-7))
    check('same_initial_context_loss_zero_new_terms',np.array_equal(loss[0,[0,1,3]],old_t['treatment_pre_update_losses'][0]) and loss[0,2]==0.)
    cfg=policy['settings'];setting=train['settings']
    check('fixed_optimization_settings',all(setting[k]==cfg[k] for k in ['steps','batch_size','learning_rate','weight_decay','anchor_coefficient']) and setting['output_coefficient']==1.0 and setting['optimizer_betas']==[.9,.999] and setting['optimizer_eps']==1e-8)
    distance=math.fsum(float(v)**2 for ar in [w-pw,b-pb] for v in ar.ravel())
    d=np.column_stack((w-pw,b-pb));d-=d.mean(axis=0,keepdims=True)
    raw=float(np.sum((d@g['gram'])*d))/(2*93)
    check('postfinal_reported_quantities',abs(distance-train['final_anchor_distance_squared'])<1e-12 and abs(raw-train['final_output_loss_raw'])<1e-12)
    for group in ['base','annotation_dev','whole_dev','public','mp','pair_context']:
        xs=a[group+'_x'].astype(float);ys=a[group+'_y'];rs=[r for r in report['rows'] if r['group']==group]
        check(group+'_row_identity',len(rs)==len(xs) and all(r['id']==m['id'] and r['true']==classes[int(y)] for r,m,y in zip(rs,meta[group],ys)))
        for i,(x,y,row) in enumerate(zip(xs,ys,rs)):
            scores=np.array([math.fsum(float(v)*float(ww) for v,ww in zip(x,wc))+float(bb) for wc,bb in zip(w,b)])
            maxerr=max(maxerr,float(np.max(np.abs(scores-z[group][i]))));nscore+=93
            for scope,ids in [('full93',range(93)),('project18',target)]:
                if int(y) not in ids:continue
                order=sorted(ids,key=lambda j:-scores[j]);rival=next(j for j in order if j!=int(y));got=row['models']['candidate'][scope]
                assert got['pred']==classes[order[0]] and got['correct']==(order[0]==int(y)) and got['rank']==order.index(int(y))+1
                maxmargin=max(maxmargin,abs(float(scores[y]-scores[rival])-got['margin']));nranks+=1
            if group=='base':
                previous=oldtrainmap[row['id']]
                for new,old in [('parent','parent'),('context_baseline','treatment')]:
                    assert row['models'][new]['full93']=={k:previous['models'][old][k] for k in ['pred','correct']}
            else:
                previous=oldmap[group,row['id'],row['condition']]
                assert row['models']['parent']==previous['models']['parent'] and row['models']['context_baseline']==previous['models']['treatment']
        summary[group]={}
        for name in ['parent','context_baseline','candidate']:
            summary[group][name]={}
            for cond in sorted({r['condition'] for r in rs}):
                summary[group][name][cond]={}
                for scope in ['full93','project18']:
                    subset=[r for r in rs if r['condition']==cond and scope in r['models'][name]]
                    if not subset:continue
                    summary[group][name][cond][scope]=dict(n=len(subset),correct=sum(r['models'][name][scope]['correct'] for r in subset),ground_truth_classes=len({r['true'] for r in subset}))
        changes[group]={}
        for name in ['parent','context_baseline']:
            lose=[r for r in rs if r['models'][name]['full93']['correct'] and not r['models']['candidate']['full93']['correct']]
            gain=[r for r in rs if not r['models'][name]['full93']['correct'] and r['models']['candidate']['full93']['correct']]
            def small(rr):return [dict(id=r['id'],true=r['true'],condition=r['condition'],before=r['models'][name]['full93']['pred'],after=r['models']['candidate']['full93']['pred']) for r in rr]
            changes[group][name]=dict(right_to_wrong=len(lose),wrong_to_right=len(gain),regressions=small(lose),improvements=small(gain))
    check('294345_scalar_scores',nscore==294345 and maxerr<1e-10,max_error=maxerr)
    check('all_candidate_rankings',maxmargin<1e-10,n=nranks,max_margin_error=maxmargin)
    check('all_reused_baseline_outputs_unchanged',True)
    check('all_summary_counts',summary==report['summary'])
    delta=z['base']-g['parent'];delta-=delta.mean(axis=1,keepdims=True)
    direct=math.fsum(float(ww)*math.fsum(float(s)**2 for s in r) for ww,r in zip(g['row_weights'],delta))/(2*93)
    check('full_original_training_output_preservation',abs(direct-raw)<1e-12 and abs(direct-report['final_output_preservation_loss'])<1e-12)
    def count(g,c='intact',s='full93'):return summary[g]['candidate'][c][s]['correct']
    gates=dict(original_train2122=count('base')==2122,native_annotation699=count('annotation_dev')>=690,native_target33=count('annotation_dev',s='project18')>=32,
        native_whole74=count('whole_dev')==74,native_whole_target30=count('whole_dev',s='project18')==30,
        apology_development3=all(r['models']['candidate'][s]['correct'] for r in report['rows'] if r['group']=='annotation_dev' and r['id'].startswith('NIA_SL_SEN0356_') for s in ['full93','project18']),score_drift_reduced=direct<.7895602850906952)
    for cond,n in [('intact',18),('reviewed_masked',5),('additional_train',9)]:gates['public_'+cond]=all(count('public',cond,s)==n for s in ['full93','project18'])
    for cond in ['whole_clip','provided_annotation']:gates['mp_'+cond]=all(count('mp',cond,s)==7 for s in ['full93','project18'])
    check('all12_predeclared_gates',gates==report['gates'] and all(gates.values()))
    check('public_eligible_without_default_promotion',report['decision']=='ELIGIBLE_PENDING_INDEPENDENT_AUDIT' and not report['default_promoted'])
out=dict(status='PASS_PUBLIC_SELECTION_INTEGRITY',candidate=train['candidate'],checks=checks,gates=gates,changes=changes,scalar_scores_checked=nscore,scoped_rankings_checked=nranks,max_scalar_error=maxerr,
    source_pins={str(p):sha(p) for p in R.iterdir() if p.is_file()},training_reruns=0,encoder_forwards=0,user_arrays_read=0,reserved_arrays_read=0)
with (H/'RESULT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=out['status'],checks=len(checks),scores=nscore,rankings=nranks,native_changes=changes['annotation_dev']),ensure_ascii=False))
