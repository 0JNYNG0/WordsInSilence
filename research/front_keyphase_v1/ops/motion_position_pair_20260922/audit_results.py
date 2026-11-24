"""Independent saved-artifact audit; no further encoder inference or training."""
from pair_io import *
import math
from collections import Counter

def main():
    p=policy();t=read(R/'training.json');ev=read(R/'evaluation.json');pre=read(R/'preflight.json');checks={}
    def check(name,value):
        checks[name]=bool(value)
        assert checks[name],name
    check('both_frozen_before_evaluation',t['status']=='TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION' and t['at_kst']<ev['at_kst'])
    check('training_record_hash',sha(R/'training.json')==ev['training_record_sha256'])
    check('policy_hash',t['policy_sha256']==ev['policy_sha256']==sha(H/'comparison_policy.json'))
    check('preflight_hash',sha(R/'preflight.json')==t['preflight_sha256'])
    check('preflight_files_fixed',all(sha(q)==d for q,d in pre['files'].items()))
    check('arrays_hash',sha(R/'evaluation_arrays.npz')==ev['arrays_sha256'] and sha(R/'training_arrays.npz')==t['arrays_sha256'])
    check('one_pair_only',t['attempts']==1 and t['real_training_arms']==2)
    check('same_fixed_settings',all(t['settings'][k]==v for k,v in dict(steps=660,base_half=16,context_half=16,batch_size=32,learning_rate=.001,weight_decay=.001,anchor_coefficient=.01,output_coefficient=1.,channels=280,class_count=93).items()))
    check('no_user_reserved_fit_or_eval',all(t[k]==0 for k in ['user_arrays_read','reserved_arrays_read','development_arrays_read']) and ev['user_arrays_read']==ev['reserved_arrays_read']==0)
    _,parent_model,classes=parent(p);parent_state=parent_model.state_dict();payloads={}
    with np.load(p['cache']['path'],allow_pickle=False) as cache,np.load(R/'training_arrays.npz',allow_pickle=False) as z:
        check('exact_existing_schedule',all(np.array_equal(cache[k],z[k]) for k in ['base_indices','context_indices']))
        check('motion_scale_reused',np.array_equal(cache['velocity_scale'],z['velocity_scale']))
        check('position_scale_pinned',np.array_equal(z['position_scale'],np.array(read(H/'POSITION_SCALE.json')['scale'],dtype=np.float32)))
        for arm,mode in [('control','position'),('treatment','motion')]:
            spec=t['models'][arm];check(arm+'_checkpoint_hash',sha(spec['path'])==spec['sha256']==ev['models'][arm]['sha256'])
            ck=torch.load(spec['path'],map_location='cpu',weights_only=True);state=ck['state_dict'];payloads[arm]=ck
            check(arm+'_classes_mode_parent',ck['classes']==classes and ck['feature_mode']==mode and ck['parent']==p['parent'])
            check(arm+'_all_finite',all(bool(torch.isfinite(v).all()) for v in state.values()))
            changed={k for k,v in parent_state.items() if not torch.equal(v,state['base.'+k])}
            check(arm+'_only_head_changed_in_parent',changed=={'head.weight','head.bias'})
            check(arm+'_only_two_added_keys',set(state)=={'base.'+k for k in parent_state}|{'velocity_weight','velocity_scale'})
            check(arm+'_scale_frozen',np.array_equal(state['velocity_scale'].numpy(),z['position_scale' if mode=='position' else 'velocity_scale']))
            check(arm+'_adapter_nonzero',bool(torch.count_nonzero(state['velocity_weight'])) and tuple(state['velocity_weight'].shape)==(64,279,1))
            check(arm+'_matched_capacity',sum(state[k].numel() for k in ['velocity_weight','base.head.weight','base.head.bias'])==23901)
            losses=z[arm+'_losses'];check(arm+'_660_finite_losses',losses.shape==(660,4) and bool(np.isfinite(losses).all()) and bool((losses>=0).all()))
            expected=(losses[:,0].astype(np.float32)+losses[:,1].astype(np.float32)).astype(float)+losses[:,2]
            check(arm+'_loss_components',bool(np.allclose(losses[:,3],expected,atol=1e-12,rtol=0)))
    previous=read(L/'runs/partial_encoder_pair_v1/evaluation.json');lookup={(r['group'],r['id'],r['condition']):r for r in previous['rows']}
    check('historical_predictions_reused_exactly',all(r['models']['historical_reference']==lookup[r['group'],r['id'],r['condition']]['models']['control'] for r in ev['rows']))
    from wis_lab.service.neural import PROJECT_18
    target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())]
    max_error=0.;score_count=rank_count=0
    with np.load(R/'evaluation_arrays.npz',allow_pickle=False) as arrays:
        check('score_classes',arrays['classes'].tolist()==classes)
        for g in ['annotation_dev','whole_dev','public','mp']:
            rows=[r for r in ev['rows'] if r['group']==g]
            for arm,ck in payloads.items():
                pooled=arrays[g+'__'+arm+'__pooled'];saved=arrays[g+'__'+arm+'__scores']
                w=ck['state_dict']['base.head.weight'].numpy().astype(float);b=ck['state_dict']['base.head.bias'].numpy().astype(float)
                scores=np.array([[math.fsum([float(v)*float(q) for v,q in zip(x,wj)]+[float(bj)]) for wj,bj in zip(w,b)] for x in pooled])
                err=float(np.max(np.abs(scores-saved)));max_error=max(max_error,err);score_count+=scores.size
                check(g+'_'+arm+'_independent_scores',err<1e-10)
                good=True
                for row,score in zip(rows,scores):
                    yi=classes.index(row['true'])
                    for scope,inds in [('full93',list(range(93))),('project18',target)]:
                        if yi not in inds:
                            good=good and scope not in row['models'][arm];continue
                        ranking=sorted(inds,key=lambda j:(-score[j],j));rival=max(score[j] for j in inds if j!=yi);value=row['models'][arm][scope]
                        good=good and value['pred']==classes[ranking[0]] and value['correct']==(ranking[0]==yi) and value['rank']==ranking.index(yi)+1 and abs(value['margin']-(score[yi]-rival))<1e-10
                        rank_count+=1
                check(g+'_'+arm+'_independent_ranks',good)
    # Independent confusion counts for every reported condition/scope, including historical rows.
    for g,arms in ev['summary'].items():
        good=True
        for arm,conds in arms.items():
            for cond,scopes in conds.items():
                for sc,reported in scopes.items():
                    rows=[r for r in ev['rows'] if r['group']==g and r['condition']==cond and sc in r['models'][arm]]
                    truth=Counter(r['true'] for r in rows);pred=Counter(r['models'][arm][sc]['pred'] for r in rows)
                    hits=Counter(r['true'] for r in rows if r['true']==r['models'][arm][sc]['pred'])
                    recalls=[hits[c]/n for c,n in truth.items()];f1=[2*hits[c]/(n+pred[c]) for c,n in truth.items()]
                    good=good and reported['n']==len(rows) and reported['correct']==sum(hits.values()) and reported['ground_truth_classes']==len(truth)
                    good=good and abs(reported['macro_recall']-math.fsum(recalls)/len(recalls))<1e-12 and abs(reported['macro_f1']-math.fsum(f1)/len(f1))<1e-12
                    for c,n in truth.items():
                        v=reported['per_class'][c];good=good and v['n']==n and v['correct']==hits[c] and v['predicted']==pred[c]
                    for person,v in reported['per_signer'].items():
                        pr=[r for r in rows if str(r.get('signer_id'))==person]
                        good=good and v['n']==len(pr) and v['correct']==sum(r['true']==r['models'][arm][sc]['pred'] for r in pr)
        check(g+'_all_metrics_recomputed',good)
    rows=ev['rows']
    def subset(g,cond='intact',sc='full93',arm='treatment'):
        return [r for r in rows if r['group']==g and r['condition']==cond and sc in r['models'][arm]]
    def correct(rs,arm='treatment',sc='full93'):return sum(r['true']==r['models'][arm][sc]['pred'] for r in rs)
    native=subset('annotation_dev');cn=correct(native,'control');mn=correct(native)
    def macro(arm):
        labels=set(r['true'] for r in native)
        return math.fsum(correct([r for r in native if r['true']==c],arm)/sum(r['true']==c for r in native) for c in labels)/len(labels)
    gates=dict(annotation_count=mn>=max(691,cn),annotation_macro_recall=macro('treatment')>macro('control')+1e-12 and macro('treatment')>=.9861111111111112-1e-12,
      annotation_each_signer=all(correct([r for r in native if r['signer_id']==person])>=correct([r for r in native if r['signer_id']==person],'control') for person in (11,12,13)),
      annotation_target=correct(subset('annotation_dev',sc='project18'),sc='project18')>=max(32,correct(subset('annotation_dev',sc='project18'),'control','project18')),
      apology_dev=all(r['models']['treatment'][sc]['correct'] for r in native if r['id'].startswith('NIA_SL_SEN0356_') for sc in ['full93','project18']),
      whole=correct(subset('whole_dev'))==74 and correct(subset('whole_dev',sc='project18'),sc='project18')==30)
    for cond,n in [('intact',18),('reviewed_masked',5),('additional_train',9)]:gates['public_'+cond]=all(correct(subset('public',cond,sc),sc=sc)==n for sc in ['full93','project18'])
    for cond in ['whole_clip','provided_annotation']:gates['mp_'+cond]=all(correct(subset('mp',cond,sc),sc=sc)==7 for sc in ['full93','project18'])
    check('independent_all_gates',gates==ev['gates'])
    check('independent_rejection',not all(gates.values()) and ev['decision']=='REJECTED_BY_PREDECLARED_GATES')
    check('native_supported_truth_scope',len(native)==699 and len(set(r['true'] for r in native))==81 and len(subset('annotation_dev',sc='project18'))==33 and len({r['true'] for r in subset('annotation_dev',sc='project18')})==6)
    check('nothing_promoted',not t['default_promoted'] and not ev['default_promoted'])
    result=dict(status='PASS',checks=checks,count=len(checks),decision=ev['decision'],failed_gates=[k for k,v in gates.items() if not v],
       independent_affine_scores=score_count,independent_rankings=rank_count,max_affine_error=max_error,
       new_training=0,new_encoder_evaluation=0,user_arrays_read=0,reserved_arrays_read=0,
       evidence={str(q):sha(q) for q in [R/'training.json',R/'evaluation.json',R/'training_arrays.npz',R/'evaluation_arrays.npz',H/'audit_results.py']})
    write(H/'RESULT_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['checks','evidence']}),flush=True)

if __name__=='__main__':main()
