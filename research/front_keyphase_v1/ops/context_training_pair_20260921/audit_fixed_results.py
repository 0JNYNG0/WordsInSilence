"""Independent output, schedule, scalar score and decision audit; no retraining."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import sys
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1];R=L/'runs/context_head_pair_v1'
sys.path[:0]=[str(L/'src'),str(L.parent/'sign_model_pilot_v1/src')]
from wis_lab.service.neural import PROJECT_18
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,ok):
    checks.append(dict(name=name,pass_=bool(ok)))
    if not ok:raise AssertionError(name)
report=read(R/'evaluation.json');train=read(R/'training.json');pre=read(R/'preflight.json');meta=read(H/'feature_metadata.json');policy=read(H/'comparison_policy.json')
check('all_preflight_pins',all(sha(p)==h for p,h in pre['pins'].items()))
check('saved_array_pins',sha(R/'evaluation_scores.npz')==report['evaluation_scores_sha256'] and sha(R/'training_arrays.npz')==train['training_array_sha256'])
parent=torch.load(policy['parent']['path'],map_location='cpu',weights_only=True)
check('parent_file_unchanged',sha(policy['parent']['path'])==policy['parent']['sha256'])
models={'parent':parent}
for name,spec in train['models'].items():
    check(name+'_file_pin',sha(spec['path'])==spec['sha256'])
    models[name]=torch.load(spec['path'],map_location='cpu',weights_only=True)
    check(name+'_encoder_buffers_class_order_unchanged',models[name]['classes']==parent['classes'] and all(torch.equal(v,models[name]['state_dict'][k]) for k,v in parent['state_dict'].items() if not k.startswith('head.')))
    check(name+'_architecture_unchanged',all(models[name][k]==parent[k] for k in ['width','dilations','kernel_size','dropout','n_classes']))
scalar_count=0;ranking_count=0;max_error=0.;rows=report['rows'];summaries={};changes={}
with np.load(H/'public_features.npz',allow_pickle=False) as a,np.load(R/'training_arrays.npz',allow_pickle=False) as t,np.load(R/'evaluation_scores.npz',allow_pickle=False) as sc:
    classes=a['classes'].tolist();target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())]
    rng=np.random.default_rng(policy['settings']['seed'])
    groups=[(a['base_y'],t['base_indices']),(a['pair_context_y'],t['pair_indices'])]
    buckets=[[np.flatnonzero(ys==c) for c in sorted(set(ys.tolist()))] for ys,_ in groups]
    expected=[np.empty((660,16),np.int64),np.empty((660,16),np.int64)]
    for step in range(660):
        for g in range(2):
            for slot in range(16):
                choices=buckets[g][int(rng.integers(len(buckets[g])))];expected[g][step,slot]=choices[int(rng.integers(len(choices)))]
    check('all21120_shared_draws_match',all(np.array_equal(e,g[1]) for e,g in zip(expected,groups)))
    check('effective_optimizer_settings',all(train['settings'][k]==policy['settings'][k] for k in ['steps','batch_size','seed','learning_rate','weight_decay','anchor_coefficient']) and train['settings']['betas']==[.9,.999] and train['settings']['eps']==1e-8)
    for name in ['control','treatment']:
        check(name+'_two_saved_head_representations_match',all(np.array_equal(t[name+'_'+k],models[name]['state_dict']['head.'+k].numpy()) for k in ['weight','bias']))
        loss=t[name+'_pre_update_losses'];check(name+'_660_finite_preupdate_rows',loss.shape==(660,3) and np.isfinite(loss).all() and np.allclose(loss[:,0]+loss[:,1],loss[:,2],rtol=2e-7,atol=1e-7))
        f=a['pair_annotation_x'] if name=='control' else a['pair_context_x']
        bx=np.concatenate((a['base_x'][t['base_indices'][0]],f[t['pair_indices'][0]])).astype(float)
        by=np.concatenate((a['base_y'][t['base_indices'][0]],a['pair_context_y'][t['pair_indices'][0]]))
        logits=bx@parent['state_dict']['head.weight'].numpy().astype(float).T+parent['state_dict']['head.bias'].numpy().astype(float)
        mx=logits.max(axis=1);ce=np.mean(mx+np.log(np.exp(logits-mx[:,None]).sum(axis=1))-logits[np.arange(32),by])
        check(name+'_same_parent_initial_loss',abs(ce-loss[0,0])<1e-5 and loss[0,1]==0)
        dist=math.fsum(float(v)**2 for k in ['weight','bias'] for v in (t[name+'_'+k].astype(float)-parent['state_dict']['head.'+k].numpy().astype(float)).ravel())
        check(name+'_postfinal_distance',abs(dist-train['models'][name]['final_anchor_distance_squared'])<1e-12)
    for group in ['annotation_dev','whole_dev','public','mp','pair_context','pair_annotation']:
        source='pair_context' if group=='pair_annotation' else group
        rs=[r for r in rows if r['group']==group];xs=a[group+'_x'];ys=a[source+'_y']
        check(group+'_rows_align',len(rs)==len(xs) and all(r['id']==m['id'] and r['true']==classes[int(y)] for r,m,y in zip(rs,meta[source],ys)))
        summaries[group]={}
        for name,ck in models.items():
            w=ck['state_dict']['head.weight'].numpy().astype(float);b=ck['state_dict']['head.bias'].numpy().astype(float)
            saved=sc[group+'__'+name];counts={}
            for i,(x,y,row) in enumerate(zip(xs.astype(float),ys,rs)):
                logits=np.array([math.fsum(float(xj)*float(wj) for xj,wj in zip(x,wc))+float(bc) for wc,bc in zip(w,b)])
                err=float(np.abs(logits-saved[i]).max());max_error=max(max_error,err);scalar_count+=93
                assert err<1e-10
                cond=row['condition'];counts.setdefault(cond,{})
                for scope,ids in [('full93',range(93)),('project18',target)]:
                    if int(y) not in ids:continue
                    order=sorted(ids,key=lambda j:-logits[j]);got=row['models'][name][scope];rival=next(j for j in order if j!=int(y))
                    assert got['pred']==classes[order[0]] and got['correct']==(order[0]==int(y)) and got['rank']==order.index(int(y))+1 and abs(got['margin']-float(logits[y]-logits[rival]))<1e-10
                    ranking_count+=1;counts[cond].setdefault(scope,[]);counts[cond][scope].append((int(y),order[0]==int(y)))
            summaries[group][name]={cond:{scope:dict(n=len(vals),correct=sum(v for _,v in vals),ground_truth_classes=len({y for y,_ in vals})) for scope,vals in scopes.items()} for cond,scopes in counts.items()}
        changes[group]={}
        for comparator in ['parent','control']:
            lost=[r for r in rs if r['models'][comparator]['full93']['correct'] and not r['models']['treatment']['full93']['correct']]
            gained=[r for r in rs if not r['models'][comparator]['full93']['correct'] and r['models']['treatment']['full93']['correct']]
            changes[group][comparator]=dict(right_to_wrong=len(lost),wrong_to_right=len(gained),regressions=[dict(id=r['id'],true=r['true'],before=r['models'][comparator]['full93']['pred'],after=r['models']['treatment']['full93']['pred'],condition=r['condition']) for r in lost],improvements=[dict(id=r['id'],true=r['true'],before=r['models'][comparator]['full93']['pred'],after=r['models']['treatment']['full93']['pred'],condition=r['condition']) for r in gained])
    check('all_reported_counts',summaries==report['summary'])
    def count(g,m,c='intact',s='full93'):return summaries[g][m][c][s]['correct']
    gates={'full_context74_improves_parent_and_control':count('whole_dev','treatment')>max(count('whole_dev','parent'),count('whole_dev','control')),
           'native_annotation699_at_least690':count('annotation_dev','treatment')>=690,'native_target33_at_least32':count('annotation_dev','treatment',s='project18')>=32,
           'apology_development3_retained':sum(r['group']=='annotation_dev' and r['id'].startswith('NIA_SL_SEN0356_') and r['models']['treatment']['full93']['correct'] for r in rows)==3}
    for cond,n in [('intact',18),('reviewed_masked',5),('additional_train',9)]:gates['public_'+cond+'_retained']=all(count('public','treatment',cond,s)==n for s in ['full93','project18'])
    gates['mp_annotation7_retained']=all(count('mp','treatment','provided_annotation',s)==7 for s in ['full93','project18'])
    gates['mp_whole_at_least5']=all(count('mp','treatment','whole_clip',s)>=5 for s in ['full93','project18'])
    check('all_predeclared_gates',gates==report['gates'])
    check('reject_decision_no_promotion',not all(gates.values()) and report['decision']=='REJECTED_BY_PREDECLARED_GATES' and not report['default_model_promoted'])
out=dict(status='PASS_OUTPUT_INTEGRITY',decision='REJECTED_BY_PREDECLARED_GATES',checks=checks,scalar_scores_checked=scalar_count,scoped_rankings_checked=ranking_count,max_scalar_error=max_error,
         gates=gates,changes=changes,training_reruns=0,new_encoder_forwards=0,user_records_read=0,reserved_arrays_read=0,
         execution_exit_code=1,execution_exception='Terminal print only: json.dumps received numpy.bool_. Training, checkpoints, arrays, evaluation JSON, and final pin checks had already completed. No artifact rewrite or training rerun.',
         artifact_pins={str(p):sha(p) for p in sorted(R.iterdir()) if p.is_file()})
with (H/'RESULT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=out['status'],checks=len(checks),scalar_scores_checked=scalar_count,scoped_rankings_checked=ranking_count,max_scalar_error=max_error,decision=out['decision'],native_changes={k:{s:v for s,v in val.items() if s in ['right_to_wrong','wrong_to_right','regressions']} for k,val in changes['annotation_dev'].items()}),ensure_ascii=False))
