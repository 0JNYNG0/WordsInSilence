"""Independent saved-result recount. No encoder or optimizer execution."""
import math
from collections import Counter
from local_io import *
from wis_lab.service.neural import PROJECT_18

def main():
    p=policy();tr=read(R/'training.json');ev=read(R/'evaluation.json');pf=read(R/'preflight.json');ia=read(H/'IMPLEMENTATION_AUDIT.json')
    checks={}
    def ck(n,v):
        checks[n]=bool(v)
        if not v:raise AssertionError(n)
    ck('implementation_pass',ia['status']=='PASS')
    for path,d in pf['pins'].items():ck('pin:'+Path(path).name,sha(path)==d)
    ck('policy_pin',tr['policy_sha256']==ev['policy_sha256']==sha(H/'comparison_policy.json'))
    ck('evaluation_training_pin',ev['training_record_sha256']==sha(R/'training.json'))
    ck('evaluation_array_pin',ev['arrays_sha256']==sha(R/'evaluation_arrays.npz'))
    ck('training_array_pin',tr['training_arrays_sha256']==sha(R/'training_arrays.npz'))
    parent=torch.load(p['parent']['path'],map_location='cpu',weights_only=True);classes=parent['classes'];states={}
    for arm,spec in tr['models'].items():
        ck(arm+'_checkpoint_pin',sha(spec['path'])==spec['sha256'])
        obj=torch.load(spec['path'],map_location='cpu',weights_only=True);state=obj['state_dict'];states[arm]=state
        changed={n for n in state if not torch.equal(state[n],parent['state_dict'][n])}
        allowed={n for n in spec['trainable_names']}
        ck(arm+'_changes_match_record',changed==set(spec['changed_tensors']))
        ck(arm+'_only_allowed_changes',changed<=allowed)
        ck(arm+'_all_buffers_fixed',all(torch.equal(v,parent['state_dict'][n]) for n,v in state.items() if any(k in n for k in ['running_mean','running_var','num_batches_tracked'])))
        ck(arm+'_prefix_fixed',all(torch.equal(v,parent['state_dict'][n]) for n,v in state.items() if not n.startswith(('head.','blocks.3.'))))
        ck(arm+'_class_order',obj['classes']==classes)
    ck('control_changes_two_head_tensors',set(tr['models']['control']['changed_tensors'])=={'head.weight','head.bias'})
    ck('treatment_last_block_changed',any(n.startswith('blocks.3.') for n in tr['models']['treatment']['changed_tensors']))
    with np.load(R/'training_arrays.npz',allow_pickle=False) as new,np.load(p['paths']['schedule'],allow_pickle=False) as old:
        ck('same_660_draw_schedule',new['base_indices'].shape==(660,16) and np.array_equal(new['base_indices'],old['base_indices']) and np.array_equal(new['context_indices'],old['pair_indices']))
        for arm in states:
            losses=new[arm+'_losses'];ck(arm+'_losses_finite',losses.shape==(660,4) and np.isfinite(losses).all())
            ck(arm+'_loss_total',np.allclose(losses[:,:3].sum(1),losses[:,3],rtol=1e-12,atol=1e-12))
        ck('initial_losses_match',np.allclose(new['control_losses'][0],new['treatment_losses'][0],atol=1e-10,rtol=1e-10))
    meta=read(p['paths']['metadata']);target=[i for i,c in enumerate(classes) if c in PROJECT_18.values()]
    maxerr=0.;nscore=0;nranks=0;newpred={}
    with np.load(R/'evaluation_arrays.npz',allow_pickle=False) as z:
        ck('scores_class_order',z['classes'].tolist()==classes)
        for group,n in p['evaluation_groups'].items():
            rows=[r for r in ev['rows'] if r['group']==group]
            ck(group+'_row_identity',len(rows)==n and [(r['id'],r['condition']) for r in rows]==[(r['id'],r.get('condition','intact')) for r in meta[group]])
            if group in ['annotation_dev','whole_dev']:ck(group+'_development_signers',set(r['signer_id'] for r in rows)=={11,12,13})
            for arm in states:
                pool=z[group+'__'+arm+'__pooled'];scores=z[group+'__'+arm+'__scores'];w=states[arm]['head.weight'].numpy();b=states[arm]['head.bias'].numpy()
                ck(group+'_'+arm+'_score_shape',scores.shape==(n,93))
                for i,row in enumerate(rows):
                    independent=[math.fsum(float(a)*float(v) for a,v in zip(w[j],pool[i]))+float(b[j]) for j in range(93)]
                    err=max(abs(a-float(v)) for a,v in zip(independent,scores[i]));maxerr=max(err,maxerr);nscore+=93
                    ckkey=(group,arm,row['id'],row['condition'])
                    newpred[ckkey]=classes[max(range(93),key=lambda j:independent[j])]
                    yi=classes.index(row['true'])
                    for scope,inds in [('full93',list(range(93))),('project18',target)]:
                        if yi not in inds:continue
                        order=sorted(inds,key=lambda j:-independent[j]);record=row['models'][arm][scope]
                        assert record['pred']==classes[order[0]] and record['correct']==(order[0]==yi) and record['rank']==order.index(yi)+1
                        rival=next(j for j in order if j!=yi);assert abs(record['margin']-(independent[yi]-independent[rival]))<1e-9
                        nranks+=1
                for cond in sorted({r['condition'] for r in rows}):
                    for scope in ['full93','project18']:
                        rs=[r for r in rows if r['condition']==cond and scope in r['models'][arm]]
                        expected=ev['summary'][group][arm][cond][scope]
                        truths=Counter(r['true'] for r in rs);preds=Counter(r['models'][arm][scope]['pred'] for r in rs);correct=Counter(r['true'] for r in rs if r['models'][arm][scope]['correct'])
                        assert expected['n']==len(rs) and expected['correct']==sum(correct.values()) and expected['ground_truth_classes']==len(truths)
                        recall=sum(correct[c]/truths[c] for c in truths)/len(truths)
                        f1=sum(2*correct[c]/(truths[c]+preds[c]) for c in truths)/len(truths)
                        assert abs(expected['macro_recall']-recall)<1e-12 and abs(expected['macro_f1']-f1)<1e-12
                        for who,counts in expected['per_signer'].items():
                            subset=[r for r in rs if r.get('signer_id')==int(who)]
                            assert counts==dict(n=len(subset),correct=sum(r['models'][arm][scope]['correct'] for r in subset))
            ck(group+'_affine_ranking_and_macro_recount',True)
    ck('independent_affine_error',maxerr<1e-9)
    # Independently reconstruct gates from rows, not evaluation helper functions.
    def subset(g,c='intact',scope='full93'):return [r for r in ev['rows'] if r['group']==g and r['condition']==c and scope in r['models']['treatment']]
    def count(rs,a,sc='full93'):return sum(r['models'][a][sc]['correct'] for r in rs)
    ad=subset('annotation_dev');tg=subset('annotation_dev',scope='project18');wd=subset('whole_dev');wt=subset('whole_dev',scope='project18')
    def recall(rs,a):
        n=Counter(r['true'] for r in rs);c=Counter(r['true'] for r in rs if r['models'][a]['full93']['correct'])
        return sum(c[k]/v for k,v in n.items())/len(n)
    gates={'annotation_count':count(ad,'treatment')>=max(690,count(ad,'control')),
      'annotation_macro_recall_improvement':recall(ad,'treatment')>recall(ad,'control')+1e-12,
      'annotation_each_signer':all(count([r for r in ad if r['signer_id']==s],'treatment')>=count([r for r in ad if r['signer_id']==s],'control') for s in (11,12,13)),
      'annotation_target':count(tg,'treatment','project18')>=max(32,count(tg,'control','project18')),
      'apology_dev':all(r['models']['treatment'][sc]['correct'] for r in ad if r['id'].startswith('NIA_SL_SEN0356_') for sc in ['full93','project18']),
      'whole':count(wd,'treatment')==74 and count(wt,'treatment','project18')==30}
    for cond,n in [('intact',18),('reviewed_masked',5),('additional_train',9)]:gates['public_'+cond]=all(count(subset('public',cond,sc),'treatment',sc)==n for sc in ['full93','project18'])
    for cond in ['whole_clip','provided_annotation']:gates['mp_'+cond]=all(count(subset('mp',cond,sc),'treatment',sc)==7 for sc in ['full93','project18'])
    ck('predeclared_gates_match',gates==ev['gates']);ck('selection_decision',ev['decision']==('ELIGIBLE_PENDING_AUDIT' if all(gates.values()) else 'REJECTED_BY_PREDECLARED_GATES'))
    result={'status':'PASS','checks':checks,'check_count':len(checks),'scores_recomputed':nscore,'rankings_recomputed':nranks,'max_score_difference':maxerr,
      'decision':ev['decision'],'failed_gates':[k for k,v in gates.items() if not v],'training_sha256':sha(R/'training.json'),'evaluation_sha256':sha(R/'evaluation.json'),'new_encoder_forwards':0,'new_training':0,'user_inputs':0,'reserved_inputs':0}
    write(H/'RESULT_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
if __name__=='__main__':main()
