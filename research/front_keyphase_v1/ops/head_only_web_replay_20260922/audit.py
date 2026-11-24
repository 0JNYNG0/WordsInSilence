"""Independent local score/ranking/count audit of the one frozen replay."""
from evaluate import H,L,read,sha,write
import math
import numpy as np
import torch

def main():
    p=read(H/'plan_LOCAL_ONLY.json');r=read(H/'result_LOCAL_ONLY.json');checks={}
    def check(name,ok):checks[name]=bool(ok);assert checks[name],name
    check('plan_before_scores',p['at_kst']<r['at_kst'] and r['plan_sha256']==sha(H/'plan_LOCAL_ONLY.json'))
    check('source_pins',all(sha(q)==d for q,d in p['source_pins'].items()))
    check('score_hash',sha(H/'scores_LOCAL_ONLY.npz')==r['scores_sha256'])
    check('candidate_fixed',p['models']['candidate']==r['candidate'])
    ck=torch.load(p['models']['candidate']['path'],map_location='cpu',weights_only=True)
    check('candidate_hash',sha(p['models']['candidate']['path'])==p['models']['candidate']['sha256'])
    nonhead=[k for k in ck['state_dict'] if not k.startswith('head.')]
    for model in ['b8','d1']:
        prior=torch.load(p['models'][model]['path'],map_location='cpu',weights_only=True)
        check(model+'_encoder_exact',len(nonhead)==58 and all(torch.equal(ck['state_dict'][k],prior['state_dict'][k]) for k in nonhead))
    u=read(p['user_result']);k=read(p['public_result']);kp=read(p['public_plan'])
    with np.load(p['user_features'],allow_pickle=False) as uf,np.load(p['public_features'],allow_pickle=False) as kf,np.load(H/'scores_LOCAL_ONLY.npz',allow_pickle=False) as z:
        classes=z['classes'].tolist();check('classes',classes==ck['classes'])
        w=ck['state_dict']['head.weight'].numpy();b=ck['state_dict']['head.bias'].numpy()
        from wis_lab.service.neural import PROJECT_18
        target=[i for i,c in enumerate(classes) if c in set(PROJECT_18.values())]
        err=0.;ranking_count=0;score_count=0;old_exact=True;new_exact=True;bindings=True
        for i,row in enumerate(r['rows']):
            src=uf if row['feature_source']=='user' else kf
            feature=src['features'][row['feature_index']]
            if row['feature_source']=='user':
                old=next(x for x in u['rows'] if x['cohort']==row['cohort'] and x['id']==row['id'])
                bindings=bindings and old['truth']==row['truth'] and old['input_sha256']==row['input_sha256'] and old['feature_index']==row['feature_index']
                priors=dict(b8=old['baseline'],d1=old['candidate'])
            else:
                spec=kp['rows'][row['feature_index']]
                bindings=bindings and spec['id']==row['id'] and spec['label']==row['truth'] and spec['input_sha256']==row['input_sha256']
                priors={name:next(x for x in k['rows'] if x['id']==row['id'] and x['model']==older)['scopes'] for name,older in [('b8','pre_context_parent'),('d1','output_retention_candidate')]}
            direct=np.array([math.fsum(float(a)*float(bb) for a,bb in zip(feature,wc))+float(bc) for wc,bc in zip(w,b)])
            err=max(err,float(np.max(np.abs(direct-z['scores'][i]))));score_count+=len(direct)
            for scope,indices in [('full93',list(range(93))),('project18',target)]:
                yi=classes.index(row['truth']);order=sorted(indices,key=lambda j:(-direct[j],j));margin=direct[yi]-max(direct[j] for j in indices if j!=yi)
                val=row['models']['candidate'][scope]
                new_exact=new_exact and val['prediction']==classes[order[0]] and val['correct']==(order[0]==yi) and val['true_rank']==order.index(yi)+1 and abs(val['true_margin']-margin)<1e-10
                ranking_count+=1
                for name in ['b8','d1']:
                    a=row['models'][name][scope];bval=priors[name][scope]
                    old_exact=old_exact and all(a[field]==bval[field] for field in ['prediction','correct','true_rank']) and a['true_margin']==bval.get('true_margin',bval.get('true_minus_best_other'))
        check('feature_truth_bindings',bindings);check('saved_ids',z['ids'].tolist()==[row['id'] for row in r['rows']])
        check('independent_scores',err<1e-10);check('independent_new_ranks',new_exact);check('old_predictions_exact',old_exact)
    for cohort,scopes in r['summary'].items():
        rows=[row for row in r['rows'] if row['cohort']==cohort];ok=True
        for scope,v in scopes.items():
            ok=ok and v['n']==len(rows)
            for model,count in v['correct'].items():ok=ok and count==sum(x['models'][model][scope]['prediction']==x['truth'] for x in rows)
            errors=[dict(truth=x['truth'],prediction=x['models']['candidate'][scope]['prediction']) for x in rows if x['models']['candidate'][scope]['prediction']!=x['truth']]
            ok=ok and errors==v['candidate_errors']
            for model,c in v['changes'].items():
                gains=[x['truth'] for x in rows if x['models']['candidate'][scope]['prediction']==x['truth'] and x['models'][model][scope]['prediction']!=x['truth']]
                regress=[x['truth'] for x in rows if x['models']['candidate'][scope]['prediction']!=x['truth'] and x['models'][model][scope]['prediction']==x['truth']]
                ok=ok and gains==c['gains'] and regress==c['regressions']
        check(cohort+'_counts_changes',ok)
    check('38_unique_records',len(r['rows'])==len({(x['cohort'],x['id']) for x in r['rows']})==38)
    check('no_training_new_encoder_or_heldout',r['new_training']==r['new_encoder_forwards']==r['reserved_arrays_read']==r['other34_ksl_read']==r['user_raw_observations_read']==0)
    check('no_promotion_or_reselection',not r['default_promoted'] and not r['candidate_reselected'])
    result=dict(status='PASS',checks=checks,count=len(checks),independent_scores=score_count,independent_rankings=ranking_count,max_score_error=err,
       new_encoder_forwards=0,new_training=0,result_sha256=sha(H/'result_LOCAL_ONLY.json'),audit_script_sha256=sha(__file__),
       decision='WEB_REGRESSION_REMAINS_NO_DEFAULT_PROMOTION')
    write(H/'INDEPENDENT_AUDIT_LOCAL_ONLY.json',result)
    print({key:val for key,val in result.items() if key!='checks'})

if __name__=='__main__':main()
