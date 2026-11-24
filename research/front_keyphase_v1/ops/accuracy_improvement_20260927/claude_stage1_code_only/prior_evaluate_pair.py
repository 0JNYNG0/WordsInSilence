"""Public evaluation only after both checkpoints are frozen; no result-driven changes."""
from local_io import *
from generated_v1 import extract_prefix,predict_prefix
from wis_lab.service.neural import PROJECT_18

def summarize(rows,arm,scope):
    rs=[r for r in rows if scope in r['models'][arm]]
    labels=sorted({r['true'] for r in rs});detail={};recalls=[];f1s=[]
    for label in labels:
        support=sum(r['true']==label for r in rs)
        tp=sum(r['true']==label and r['models'][arm][scope]['pred']==label for r in rs)
        predicted=sum(r['models'][arm][scope]['pred']==label for r in rs)
        recall=tp/support;precision=tp/predicted if predicted else 0.;f1=2*tp/(support+predicted)
        detail[label]=dict(n=support,correct=tp,predicted=predicted,recall=recall,precision=precision,f1=f1)
        recalls.append(recall);f1s.append(f1)
    people={str(s):dict(n=sum(r.get('signer_id')==s for r in rs),correct=sum(r.get('signer_id')==s and r['models'][arm][scope]['correct'] for r in rs)) for s in sorted({r['signer_id'] for r in rs if 'signer_id' in r})}
    return dict(n=len(rs),correct=sum(r['models'][arm][scope]['correct'] for r in rs),ground_truth_classes=len(labels),
       macro_recall=float(np.mean(recalls)) if recalls else None,macro_f1=float(np.mean(f1s)) if f1s else None,per_class=detail,per_signer=people)

def main():
    p=policy();train=read(R/'training.json');assert train['status']=='TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION'
    assert sha(H/'comparison_policy.json')==train['policy_sha256']
    ck,parent,classes=parent_model(p);models={};payloads={}
    for arm,spec in train['models'].items():
        assert sha(spec['path'])==spec['sha256']
        payloads[arm]=torch.load(spec['path'],map_location='cpu',weights_only=True);models[arm],cl=u.rebuild_from_payload(payloads[arm]);assert cl==classes
    groups,meta=evaluation_inputs(p);target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())]
    old=read(p['paths']['old_eval']);oldmap={(r['group'],r['id'],r['condition']):r for r in old['rows']}
    arrays={'classes':np.array(classes)};rows=[];summary={};rebuild_error={};forwarderr=0.
    with np.load(p['paths']['old_features'],allow_pickle=False) as cache:
        for group,xs in groups.items():
            prefixes=extract_prefix(parent,xs)
            # This validates the new execution split, not a new estimate of old-model accuracy.
            recon=predict_prefix(parent,prefixes)['pooled'];prior=cache[group+'_x']
            error=float(np.max(np.abs(recon-prior)));assert np.allclose(recon,prior,atol=1e-4,rtol=1e-5),(group,error)
            rebuild_error[group]=error
            outputs={}
            for arm,model in models.items():
                outputs[arm]=predict_prefix(model,prefixes)
                pooled=outputs[arm]['pooled'];w=payloads[arm]['state_dict']['head.weight'].numpy().astype(float);b=payloads[arm]['state_dict']['head.bias'].numpy().astype(float)
                score=pooled.astype(float)@w.T+b
                err=float(np.max(np.abs(score-outputs[arm]['logits'])));forwarderr=max(forwarderr,err);assert err<1e-4
                assert np.array_equal(score.argmax(1),outputs[arm]['logits'].argmax(1))
                arrays[group+'__'+arm+'__pooled']=pooled;arrays[group+'__'+arm+'__scores']=score
            for i,spec in enumerate(meta[group]):
                label=spec.get('label',spec.get('true',spec.get('reference_label')));yi=classes.index(label);cond=spec.get('condition','intact')
                row=dict(group=group,id=spec['id'],condition=cond,true=label,models={})
                if 'signer_id' in spec:row['signer_id']=spec['signer_id']
                oldrow=oldmap[group,spec['id'],cond];assert oldrow['true']==label
                row['models']['historical_reference']=oldrow['models']['candidate']
                for arm in models:
                    score=arrays[group+'__'+arm+'__scores'][i];row['models'][arm]={}
                    for scope,inds in [('full93',list(range(93))),('project18',target)]:
                        if yi not in inds:continue
                        order=sorted(inds,key=lambda j:-float(score[j]));rival=next(j for j in order if j!=yi)
                        row['models'][arm][scope]=dict(pred=classes[order[0]],correct=bool(order[0]==yi),rank=order.index(yi)+1,margin=float(score[yi]-score[rival]))
                rows.append(row)
            rs=[r for r in rows if r['group']==group];summary[group]={}
            for arm in ['historical_reference','control','treatment']:
                summary[group][arm]={cond:{scope:summarize([r for r in rs if r['condition']==cond],arm,scope) for scope in ['full93','project18']} for cond in sorted({r['condition'] for r in rs})}
            print(json.dumps({'evaluated_group':group,'clips_or_conditions':len(xs)}),flush=True)
    def s(g,a='treatment',c='intact',scope='full93'):return summary[g][a][c][scope]
    ap=[r for r in rows if r['group']=='annotation_dev' and r['id'].startswith('NIA_SL_SEN0356_')];assert len(ap)==3
    gates={
      'annotation_count':s('annotation_dev')['correct']>=max(690,s('annotation_dev','control')['correct']),
      'annotation_macro_recall_improvement':s('annotation_dev')['macro_recall']>s('annotation_dev','control')['macro_recall']+1e-12,
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
            gains=[{'id':r['id'],'condition':r['condition'],'true':r['true'],'before':r['models'][against]['full93']['pred']} for r in rs if r['models']['treatment']['full93']['correct'] and not r['models'][against]['full93']['correct']]
            regressions=[{'id':r['id'],'condition':r['condition'],'true':r['true'],'after':r['models']['treatment']['full93']['pred']} for r in rs if not r['models']['treatment']['full93']['correct'] and r['models'][against]['full93']['correct']]
            changes[group][against]={'gains':gains,'regressions':regressions}
    with (R/'evaluation_arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    result={'status':'PUBLIC_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT','decision':'ELIGIBLE_PENDING_AUDIT' if all(gates.values()) else 'REJECTED_BY_PREDECLARED_GATES',
      'gates':gates,'summary':summary,'changes':changes,'rows':rows,'models':train['models'],'policy_sha256':sha(H/'comparison_policy.json'),
      'training_record_sha256':sha(R/'training.json'),'arrays_sha256':sha(R/'evaluation_arrays.npz'),'parent_cache_rebuild_error':rebuild_error,'max_float32_forward_affine_error':forwarderr,
      'new_evaluation_prefix_clips':sum(len(x) for x in groups.values()),'new_candidate_last_block_evaluations':2*sum(len(x) for x in groups.values()),
      'historical_predictions_reused':True,'user_arrays_or_scores_read':0,'reserved_arrays_read':0,'default_promoted':False,
      'limits':['annotation699 and whole74 share source recordings; do not add as independent cases','actual project target truths are6, not18, in native development','public MP7 has training signers01/02 and two windows each','public reference includes training examples and approved development stress cases','no new-person web generalization or execution-correctness claim']}
    write(R/'evaluation.json',result)
    print(json.dumps({'decision':result['decision'],'gates':gates,'annotation':{a:{k:s('annotation_dev',a)[k] for k in ['correct','macro_recall','macro_f1','per_signer']} for a in ['historical_reference','control','treatment']},'whole':{a:s('whole_dev',a)['correct'] for a in ['historical_reference','control','treatment']}}),flush=True)
if __name__=='__main__':main()
