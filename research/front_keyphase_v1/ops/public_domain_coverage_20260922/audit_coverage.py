"""Independent inventory recount and spectral (not Cholesky) metric verification."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import numpy as np

H=Path(__file__).resolve().parent;L=H.parents[1];O=H.parent/'context_training_pair_20260921'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(H/'plan.json');report=read(H/'result.json');meta=read(O/'feature_metadata.json')
checks=[]
def check(name,passed,**extra):
    checks.append(dict(name=name,passed=bool(passed),**extra))
    if not passed:raise AssertionError(name)
check('pins_unchanged',all(sha(p)==s for p,s in plan['source_pins'].items()) and sha(H/'metric_and_queries.npz')==report['arrays_sha256'] and sha(H/'plan.json')==report['plan_sha256'])
with np.load(O/'public_features.npz',allow_pickle=False) as a,np.load(H/'metric_and_queries.npz',allow_pickle=False) as z,np.load(L/'runs/context_head_pair_v1/training_arrays.npz',allow_pickle=False) as schedule:
    x=a['base_x'].astype(float);y=a['base_y'];classes=a['classes'].tolist();counts=np.bincount(y,minlength=93);weight=1/(93*counts[y])
    mu=np.average(x,axis=0,weights=weight);delta=x-mu
    covariance=np.einsum('n,ni,nj->ij',weight,delta,delta)
    vals,vec=np.linalg.eigh(covariance);assert vals.min()>0
    inv=(vec/vals)@vec.T
    check('training_only_center_and_covariance',np.max(np.abs(mu-z['mu']))<1e-12 and np.max(np.abs(covariance-z['covariance']))<1e-11)
    check('cholesky_contract',np.max(np.abs(z['cholesky']@z['cholesky'].T-covariance))<1e-11)
    norms=np.sqrt(np.einsum('ni,ij,nj->n',delta,inv,delta)/64)
    check('training_radii_spectral_recount',np.max(np.abs(norms-z['training_center_distances']))<1e-10 and abs(float(weight@norms**2)-1)<1e-10)
    identifiers=[r['id'] for r in meta['base']];clipids=np.array([re.sub(r'_a\d+_\d+$','',i) for i in identifiers])
    check('original_source_clips1882',len(set(clipids))==report['inventory']['original_source_clip_count']==1882)
    known={r['signer_id'] for r in meta['base'] if 'signer_id' in r}
    check('train_signers_only01_09',known==set(range(1,10)) and all(r['role']=='train' for r in meta['base']))
    source=lambda r:'NIKL_MediaPipe_reference' if r['id'].startswith('NIKL_') else 'AIHub_native_approved_apology' if r['id'].startswith('NIA_SL_SEN0356_') else 'AIHub_native_original'
    for group,stats in report['inventory']['source_groups'].items():
        rr=[r for r in meta['base'] if source(r)==group]
        check(group+'_inventory',stats['rows']==len(rr) and stats['labels']==len({r['label'] for r in rr}) and stats['source_clips']==len({re.sub(r'_a\d+_\d+$','',r['id']) for r in rr}) and stats['known_signer_ids']==sorted({r['signer_id'] for r in rr if 'signer_id' in r}) and stats['actual_base_half_draws']==sum(source(meta['base'][int(i)])==group for i in schedule['base_indices'].ravel()))
    for r in report['per_class_inventory']:
        rr=[b for b in meta['base'] if b['label']==r['label']]
        assert len(rr)==r['original_rows'] and r['native_rows']==sum('signer_id' in b for b in rr) and r['nikl_source_clips']==sum(b['id'].startswith('NIKL_') for b in rr)
    check('all93_class_inventory_rows',len(report['per_class_inventory'])==93)
    reference_only=[r['label'] for r in report['per_class_inventory'] if r['target18'] and r['native_rows']==0]
    check('target18_reference_only12',len(reference_only)==12 and reference_only==report['inventory']['target18_reference_only'])
    check('same_sources_context_exposure',all(r['id'] in set(clipids) for r in meta['pair_context']) and len(meta['pair_context'])==224 and schedule['pair_indices'].size==10560)
    max_error=0.;query_lookup={};distances={};qscore=read(L/'runs/output_retention_candidate_v1/evaluation.json')
    old_scores={(r['group'],r['id'],r['condition']):r['models']['candidate'] for r in qscore['rows']}
    for i,(feature,row) in enumerate(zip(z['query_features'],report['queries'])):
        assert row['id']==z['query_ids'][i]
        spec=next(p for p in plan['pairs'] if p['clip_id']==row['clip_id']);pi=spec['native_train_row_index']
        if row['domain']=='native':
            feature0=a['pair_annotation_x' if row['window']=='annotation' else 'pair_context_x'][pi]
            key=('base',meta['base'][int(a['pair_base_indices'][pi])]['id'],'intact') if row['window']=='annotation' else ('pair_context',spec['clip_id'],'intact')
        else:
            condition='provided_annotation' if row['window']=='annotation' else 'whole_clip'
            mi=next(i for i,m in enumerate(meta['mp']) if m['id']==spec['clip_id'] and m['condition']==condition)
            feature0=a['mp_x'][mi];key=('mp',spec['clip_id'],condition)
        assert np.array_equal(feature,feature0) and row['reused_prediction']==old_scores[key]
        diff=x-feature;ds=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',diff,inv,diff),0)/64)
        nearest=int(np.argmin(ds));exclude=ds.copy();exclude[clipids==spec['clip_id']]=np.inf;other=int(np.argmin(exclude))
        assert row['nearest_original_training']['id']==identifiers[nearest] and row['nearest_other_source_clip']['id']==identifiers[other]
        center=float(np.sqrt((feature-mu)@inv@(feature-mu)/64));max_error=max(max_error,abs(center-row['center_distance']),abs(float(ds[nearest])-row['nearest_original_training']['distance']),abs(float(ds[other])-row['nearest_other_source_clip']['distance']))
        # At exact training-vector equality, distance rounding can shift an empirical CDF at that one point.
        expected=float(100*weight[norms<=center+1e-12].sum());lower=float(100*weight[norms<center-1e-12].sum())
        assert lower-1e-10<=row['weighted_training_percentile']<=expected+1e-10
        query_lookup[row['id']]=feature;distances[row['id']]=center
    mapping={'native_vs_mp_annotation':('native__annotation','mediapipe__annotation'),'native_vs_mp_whole':('native__whole','mediapipe__whole'),'native_annotation_vs_whole':('native__annotation','native__whole'),'mp_annotation_vs_whole':('mediapipe__annotation','mediapipe__whole')}
    for pair in report['pairs']:
        for field,(aa,bb) in mapping.items():
            d=query_lookup[pair['clip_id']+'__'+aa]-query_lookup[pair['clip_id']+'__'+bb]
            value=float(np.sqrt(d@inv@d/64));max_error=max(max_error,abs(value-pair[field]))
    check('all28_queries_and28_pair_distances',len(query_lookup)==28 and max_error<1e-9,max_error=max_error)
    for group,stats in report['summary'].items():
        rr=[r for r in report['queries'] if r['domain']+'_'+r['window']==group]
        assert stats['n']==7 and stats['reused_full93_correct']==sum(r['reused_prediction']['full93']['correct'] for r in rr)
        assert stats['nearest_other_source_label_agreement']==sum(r['nearest_other_source_clip']['label']==r['source_label'] for r in rr)
        assert abs(stats['center_distance_median']-float(np.median([distances[r['id']] for r in rr])))<1e-9
        assert abs(stats['nearest_other_source_distance_median']-float(np.median([r['nearest_other_source_clip']['distance'] for r in rr])))<1e-9
    check('summary_and_medians_recount',all(abs(report['pair_distance_medians'][k]-float(np.median([r[k] for r in report['pairs']])))<1e-12 for k in mapping))
    larger=sum(min(r['native_annotation_vs_whole'],r['mp_annotation_vs_whole'])>max(r['native_vs_mp_annotation'],r['native_vs_mp_whole']) for r in report['pairs'])
    check('observed_window_shift_exceeds_detector_shift_all7',larger==7)
out=dict(status='PASS',checks=checks,query_count=28,pair_distance_count=28,nearest_neighbors_checked=56,max_metric_difference=max_error,
    window_shift_larger_in_pairs=larger,result_sha256=sha(H/'result.json'),arrays_sha256=sha(H/'metric_and_queries.npz'),new_training=0,new_encoder_forwards=0,user_inputs_read=0,reserved_arrays_read=0)
with (H/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
