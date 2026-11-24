"""One public development evaluation after both matched candidates are frozen."""
from pair_io import *
from wis_lab.service.neural import PROJECT_18
from datetime import datetime,timezone,timedelta

def summarize(rows,arm,scope):
    rs=[r for r in rows if scope in r['models'][arm]];detail={}
    for label in sorted({r['true'] for r in rs}):
        n=sum(r['true']==label for r in rs)
        tp=sum(r['true']==label and r['models'][arm][scope]['pred']==label for r in rs)
        pred=sum(r['models'][arm][scope]['pred']==label for r in rs)
        detail[label]=dict(n=n,correct=tp,predicted=pred,recall=tp/n,precision=tp/pred if pred else 0.,f1=2*tp/(n+pred))
    people={str(s):dict(n=sum(r.get('signer_id')==s for r in rs),correct=sum(r.get('signer_id')==s and r['models'][arm][scope]['correct'] for r in rs)) for s in sorted({r['signer_id'] for r in rs if 'signer_id' in r})}
    return dict(n=len(rs),correct=sum(r['models'][arm][scope]['correct'] for r in rs),ground_truth_classes=len(detail),
       macro_recall=float(np.mean([d['recall'] for d in detail.values()])) if detail else None,
       macro_f1=float(np.mean([d['f1'] for d in detail.values()])) if detail else None,per_class=detail,per_signer=people)

def main():
    p=policy();impl=implementation();train=read(R/'training.json')
    assert train['status']=='TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION'
    assert train['policy_sha256']==sha(H/'comparison_policy.json')
    assert not (R/'evaluation.json').exists()
    models={};payloads={}
    for arm,spec in train['models'].items():payloads[arm],models[arm],classes=checkpoint_model(spec,impl,p)
    target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())];assert len(target)==18
    prior=read(L/'runs/partial_encoder_pair_v1/evaluation.json')
    prior_map={(r['group'],r['id'],r['condition']):r for r in prior['rows']}
    groups,meta=old.evaluation_inputs(old.policy());arrays={'classes':np.array(classes)};rows=[];summary={};forward_error=0.
    for group,xs in groups.items():
        for arm,model in models.items():
            pooled=[];logits=[]
            hook=model.base.head.register_forward_pre_hook(lambda module,args:pooled.append(args[0].detach().cpu().numpy().copy()))
            try:
                with torch.no_grad():
                    for x in xs:logits.append(model(torch.from_numpy(x[None]),torch.ones((1,len(x)),dtype=torch.float32)).numpy())
            finally:hook.remove()
            pooled=np.concatenate(pooled);logits=np.concatenate(logits)
            state=payloads[arm]['state_dict'];w=state['base.head.weight'].numpy().astype(float);b=state['base.head.bias'].numpy().astype(float)
            scores=pooled.astype(float)@w.T+b;err=float(np.max(np.abs(scores-logits)))
            assert err<1e-4 and np.array_equal(scores.argmax(1),logits.argmax(1));forward_error=max(forward_error,err)
            arrays[group+'__'+arm+'__pooled']=pooled;arrays[group+'__'+arm+'__scores']=scores
        for i,spec in enumerate(meta[group]):
            label=spec.get('label',spec.get('true',spec.get('reference_label')));yi=classes.index(label);cond=spec.get('condition','intact')
            row=dict(group=group,id=spec['id'],condition=cond,true=label,models={})
            if 'signer_id' in spec:row['signer_id']=spec['signer_id']
            ref=prior_map[group,spec['id'],cond];assert ref['true']==label
            row['models']['historical_reference']=ref['models']['control']
            for arm in models:
                scores=arrays[group+'__'+arm+'__scores'][i];row['models'][arm]={}
                for scope,indices in [('full93',list(range(93))),('project18',target)]:
                    if yi not in indices:continue
                    order=sorted(indices,key=lambda j:-float(scores[j]));rival=next(j for j in order if j!=yi)
                    row['models'][arm][scope]=dict(pred=classes[order[0]],correct=bool(order[0]==yi),rank=order.index(yi)+1,margin=float(scores[yi]-scores[rival]))
            rows.append(row)
        rs=[r for r in rows if r['group']==group]
        summary[group]={arm:{cond:{sc:summarize([r for r in rs if r['condition']==cond],arm,sc) for sc in ['full93','project18']} for cond in sorted({r['condition'] for r in rs})} for arm in ['historical_reference','control','treatment']}
        print(json.dumps(dict(evaluated_group=group,conditions=len(xs))),flush=True)
    def s(g,a='treatment',c='intact',scope='full93'):return summary[g][a][c][scope]
    ap=[r for r in rows if r['group']=='annotation_dev' and r['id'].startswith('NIA_SL_SEN0356_')];assert len(ap)==3
    gates={
      'annotation_count':s('annotation_dev')['correct']>=max(691,s('annotation_dev','control')['correct']),
      'annotation_macro_recall':s('annotation_dev')['macro_recall']>s('annotation_dev','control')['macro_recall']+1e-12 and s('annotation_dev')['macro_recall']>=p['historical_reference_counts']['native_macro_recall']-1e-12,
      'annotation_each_signer':all(s('annotation_dev')['per_signer'][str(k)]['correct']>=s('annotation_dev','control')['per_signer'][str(k)]['correct'] for k in (11,12,13)),
      'annotation_target':s('annotation_dev',scope='project18')['correct']>=max(32,s('annotation_dev','control',scope='project18')['correct']),
      'apology_dev':all(r['models']['treatment'][sc]['correct'] for r in ap for sc in ['full93','project18']),
      'whole':s('whole_dev')['correct']==74 and s('whole_dev',scope='project18')['correct']==30}
    for cond,n in [('intact',18),('reviewed_masked',5),('additional_train',9)]:gates['public_'+cond]=all(s('public',c=cond,scope=sc)['correct']==n for sc in ['full93','project18'])
    for cond in ['whole_clip','provided_annotation']:gates['mp_'+cond]=all(s('mp',c=cond,scope=sc)['correct']==7 for sc in ['full93','project18'])
    changes={}
    for group in groups:
        rs=[r for r in rows if r['group']==group];changes[group]={}
        for against in ['control','historical_reference']:
            changes[group][against]={
              'gains':[dict(id=r['id'],condition=r['condition'],true=r['true'],before=r['models'][against]['full93']['pred']) for r in rs if r['models']['treatment']['full93']['correct'] and not r['models'][against]['full93']['correct']],
              'regressions':[dict(id=r['id'],condition=r['condition'],true=r['true'],after=r['models']['treatment']['full93']['pred']) for r in rs if not r['models']['treatment']['full93']['correct'] and r['models'][against]['full93']['correct']]}
    with (R/'evaluation_arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    result=dict(status='PUBLIC_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT',at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
      decision='ELIGIBLE_PENDING_AUDIT' if all(gates.values()) else 'REJECTED_BY_PREDECLARED_GATES',gates=gates,summary=summary,changes=changes,rows=rows,models=train['models'],
      policy_sha256=sha(H/'comparison_policy.json'),training_record_sha256=sha(R/'training.json'),arrays_sha256=sha(R/'evaluation_arrays.npz'),
      max_float32_forward_affine_error=forward_error,new_candidate_forward_clips=2*sum(map(len,groups.values())),historical_predictions_reused=True,
      user_arrays_read=0,reserved_arrays_read=0,default_promoted=False,
      limits=['native699 and whole74 overlap source recordings','native target33 covers6 of18 target truths','MP7 clips use training signers01/02; two windows are not independent people','public reference includes training examples','no new-person web or execution-correctness claim'])
    write(R/'evaluation.json',result)
    print(json.dumps(dict(decision=result['decision'],gates=gates,annotation={a:{k:s('annotation_dev',a)[k] for k in ['correct','macro_recall','macro_f1','per_signer']} for a in ['historical_reference','control','treatment']})),flush=True)

if __name__=='__main__':main()
