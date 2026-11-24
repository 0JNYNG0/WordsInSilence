"""Verify saved canonical features/targets without any neural-network execution."""
from pathlib import Path
import collections,json,hashlib
import numpy as np
K=Path(__file__).resolve().parent;J=K.parent/'splice_rank_learning';O=K.parent.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pooled_local(x,spans):
    norm=read(O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json')
    hand=np.where(np.repeat(x[:,237:279]>.5,2,1),
        (x[:,102:186]-np.asarray(norm['mean'])[102:186])/np.asarray(norm['std'])[102:186],0).astype(np.float32)
    v=np.c_[hand,x[:,237:279]].astype(np.float64)
    cs=np.r_[np.zeros((1,126)),v.cumsum(0)];cs2=np.r_[np.zeros((1,126)),(v*v).cumsum(0)]
    a=spans[:,0];b=spans[:,1]+1;n=b-a;c1=a+(n+2)//3;c2=c1+(n+1)//3
    avg=lambda l,r:(cs[r]-cs[l])/(r-l)[:,None]
    mean=avg(a,b);variance=np.maximum((cs2[b]-cs2[a])/n[:,None]-mean*mean,0)
    return np.c_[avg(a,c1),avg(c1,c2),avg(c2,b),np.sqrt(variance)].astype(np.float32)

def main():
    assert not(K/'PREPARATION_VERIFICATION.json').exists()
    cfg=read(K/'FROZEN.json');plan=read(K/'PREPARATION_FROZEN.json');meta=read(K/'PREPARATION_INPUTS_COMPLETE.json')
    checks=collections.Counter();max_pool_error=0.
    def ck(ok,name):assert bool(ok),name;checks[name]+=1
    for table in [cfg['pins'],plan['pins'],read(K/'LAUNCH_CORRECTION.json')['pins']]:
        for p,h in table.items():ck(sha(p)==h,'frozen_inputs_code_unchanged')
    ck(meta['status']=='CANONICAL_RATE_FEATURES_READY_NOT_TRAINED','prepared_status')
    ck(meta['source_contract_sha256']==sha(K/'FROZEN.json')and meta['preparation_contract_sha256']==sha(K/'PREPARATION_FROZEN.json'),'prepared_contract_refs')
    sm={(r['clip_id'],r['view'],r['reverse']):r for r in cfg['splice_sequences']};canonical={}
    for r in meta['sequences']:
        ck(sha(r['canonical_path'])==r['canonical_sha256'],'canonical_pin');x=np.load(r['canonical_path']);q=np.load(r['path'])
        base=np.load(sm[r['clip_id'],r['view'],r['reverse']]['path'])
        xx=base['x'] if r['base_kind']=='splice'else base['x'][np.argsort(base['baseline_index'],kind='stable')]
        ck(np.array_equal(x['x'][:,:279],xx[q['base_query_index'],:279]),'x0_279_exact_mapping')
        ck(np.array_equal(x['x'][:,279],np.r_[0,np.diff(q['timestamp_us']).astype(np.float32)/1e6]),'dt_rebuilt_exact')
        ck(np.array_equal(x['timestamp_us'],q['timestamp_us'])and np.all(np.diff(x['timestamp_us'])>0),'query_pts_exact')
        ck(np.array_equal(x['baseline_index'],q['baseline_index'])and np.array_equal(x['source_index'],q['source_index']),'saved_provenance_exact')
        ck(np.isfinite(x['x']).all()and x['x'].shape[1]==280,'canonical_finite_shape')
        canonical[r['canonical_path']]=x['x']
    ck(len(meta['sequences'])==144 and len(meta['bags'])==288,'complete_case_denominator')
    summaries=collections.defaultdict(lambda:dict(cases=0,held=0,features=0,positive=0,negative=0))
    for r in meta['bags']:
        ss=summaries[r['rate'],r['base_kind'],r['family']];ss['cases']+=1
        if r['skipped']:
            ck('path'not in r and r['positive']==0 and r['held'],'held_has_no_features_or_forced_positive');ss['held']+=1;continue
        ck(sha(r['path'])==r['sha256'],'feature_bag_pin');z=np.load(r['path']);t=np.load(r['label_path']);ix=z['target_grid_index']
        ck(np.array_equal(z['spans'],t['spans'][ix])and np.array_equal(z['y'],t['positive'][ix]),'selected_target_alignment')
        ck(z['rank_mask'].all()and(t['positive'][ix]|t['negative'][ix]).all(),'no_unknown_rank_targets')
        ck(set(ix[z['y']])==set(np.flatnonzero(t['positive'])),'all_inherited_full_positives_kept')
        ck(int(z['y'].sum())==r['positive']and int((~z['y']).sum())==r['negative']and 0<r['negative']<=512,'sample_count_contract')
        ck(z['local_full'].shape==(len(ix),1160)and np.isfinite(z['local_full']).all(),'finite1160_features')
        if r['base_kind']=='splice':ck(t['crossing'][ix[~z['y']]].all(),'splice_negatives_known_joins_only')
        expected=pooled_local(canonical[r['canonical_path']],z['spans'])
        actual=np.concatenate([z['local_full'][:,p*290+164:p*290+290]for p in range(4)],axis=1)
        error=float(np.max(np.abs(expected-actual)));max_pool_error=max(max_pool_error,error)
        ck(error<=3e-5,'explicit_local_features_independent_pool')
        ss['features']+=1;ss['positive']+=r['positive'];ss['negative']+=r['negative']
    ck(sum(r['skipped']for r in meta['bags'])==78,'all78_held_retained')
    ck(sum(not r['skipped']for r in meta['bags'])==210,'all210_viable_encoded')
    for key,expected in [('replay_original',178),('replay_splice',72)]:
        ck(len(meta[key])==expected,'rate1_replay_count')
        for r in meta[key]:ck(r['rate']==1 and sha(r['path'])==r['sha256'],'rate1_old_file_reused')
    ck({r['seed']for r in meta['parent_models']}=={42,43,44},'matching_J_parents')
    for r in meta['parent_models']:ck(sha(r['model'])==r['model_sha256'],'J_parent_model_unchanged')
    ck(read(K/'LAUNCH_FINISHED.json')['status']=='COMPLETED','single_launch_completed')
    report=dict(status='PREPARED_INPUTS_SAVED_VERIFIED_NO_TRAINING_BY_VERIFIER',total_checks=sum(checks.values()),checks=dict(checks),
        summaries=[dict(rate=k[0],base_kind=k[1],family=k[2],**v)for k,v in sorted(summaries.items())],
        maximum_explicit_pool_absolute_error=max_pool_error,new_feature_bags=210,held_cases=78,new_positive_candidates=sum(r['positive']for r in meta['bags']if not r['skipped']),
        new_encoded_candidates=sum(r['encoded_candidates']for r in meta['sequences']),replay_original=178,replay_splice=72,
        no_model_loading_or_inference=True,old_code_data_models_preserved=True,manifest_sha256=sha(K/'PREPARATION_INPUTS_COMPLETE.json'),
        verifier='Codex subagent independent saved-array formulas; root code review distinct; no human labels',script_sha256=sha(__file__))
    (K/'PREPARATION_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Prepared input verification PASS',report['total_checks'],'pool error',max_pool_error,'new encoded',report['new_encoded_candidates'])
if __name__=='__main__':main()
