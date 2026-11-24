"""Public-only source inventory and frozen-feature distances; no model inference."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import sys
import numpy as np

H=Path(__file__).resolve().parent;L=H.parents[1];OLD=H.parent/'context_training_pair_20260921'
sys.path[:0]=[str(L/'src'),str(L.parent/'sign_model_pilot_v1/src')]
from wis_lab.service.neural import PROJECT_18
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):
    with (H/name).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def clip(ident):return re.sub(r'_a\d+_\d+$','',ident)
def source(row):
    if row['id'].startswith('NIKL_'):return 'NIKL_MediaPipe_reference'
    if row['id'].startswith('NIA_SL_SEN0356_'):return 'AIHub_native_approved_apology'
    return 'AIHub_native_original'

def main():
    plan=read(H/'plan.json');assert not (H/'result.json').exists()
    for p,d in plan['source_pins'].items():assert sha(p)==d,p
    metadata=read(OLD/'feature_metadata.json');base_meta=metadata['base'];pair_meta=metadata['pair_context'];mp_meta=metadata['mp']
    with np.load(OLD/'public_features.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in plan['allowed_feature_keys']}
    with np.load(H.parent/'public_prediction_retention_20260922/training_scores_and_gram.npz',allow_pickle=False) as z:moment=z['gram'].copy();weights=z['row_weights'].copy()
    with np.load(L/'runs/context_head_pair_v1/training_arrays.npz',allow_pickle=False) as z:draws={k:z[k].copy() for k in plan['allowed_schedule_keys']}
    classes=a['classes'].tolist();base=a['base_x'].astype(float);y=a['base_y'];mu=moment[:-1,-1]
    assert base.shape==(2122,64) and len(classes)==93 and abs(moment[-1,-1]-1)<1e-14
    covariance=moment[:-1,:-1]-np.outer(mu,mu)
    cholesky=np.linalg.cholesky(covariance)
    white=np.linalg.solve(cholesky,(base-mu).T).T
    norms=np.sqrt(np.mean(white**2,axis=1))
    assert abs(float(np.dot(weights,norms**2))-1)<1e-10
    base_clips=np.array([clip(r['id']) for r in base_meta])
    # Labels describe counts and returned neighbors; they never filter nearest-neighbor searches.
    inventories={}
    for kind in ['AIHub_native_original','AIHub_native_approved_apology','NIKL_MediaPipe_reference']:
        rs=[r for r in base_meta if source(r)==kind]
        inventories[kind]=dict(rows=len(rs),source_clips=len({clip(r['id']) for r in rs}),labels=len({r['label'] for r in rs}),
            known_signer_ids=sorted({r['signer_id'] for r in rs if 'signer_id' in r}),signer_identity_untracked_rows=sum('signer_id' not in r for r in rs),
            actual_base_half_draws=sum(source(base_meta[int(i)])==kind for i in draws['base_indices'].ravel()))
    perclass=[]
    for label in classes:
        rs=[r for r in base_meta if r['label']==label];native=[r for r in rs if 'signer_id' in r];mp=[r for r in rs if r['id'].startswith('NIKL_')]
        perclass.append(dict(label=label,original_rows=len(rs),native_rows=len(native),native_signer_ids=sorted({r['signer_id'] for r in native}),nikl_source_clips=len(mp),target18=label in set(PROJECT_18.values())))
    mp_only=[r['label'] for r in perclass if r['target18'] and r['native_rows']==0]
    multi=[r['label'] for r in perclass if r['target18'] and r['native_rows']>0]
    inventory=dict(base_rows=2122,classes=93,source_groups=inventories,original_source_clip_count=len(set(base_clips)),
        native_signer_ids=sorted({r['signer_id'] for r in base_meta if 'signer_id' in r}),
        target18_native_plus_reference=multi,target18_reference_only=mp_only,
        full_context_training=dict(rows=224,source_clips=len({r['id'] for r in pair_meta}),labels=len({r['label'] for r in pair_meta}),known_signer_ids=sorted({r['signer_id'] for r in pair_meta}),actual_draws=int(draws['pair_indices'].size),additional_independent_source_clips=0),
        public_mp_probe=dict(source_clips=7,labels=len({r['label'] for r in plan['pairs']}),known_signer_ids=sorted({r['signer_id'] for r in plan['pairs']}),source_signers_already_in_native_training=True,used_as_mp_training_rows_in_this_head_comparison=False),
        note='This inventory is the fixed-head adaptation data, not a claim about every historical encoder training step. Repeated draws and earlier augmentation do not add independently recorded sources or people.')
    old=read(L/'runs/output_retention_candidate_v1/evaluation.json')
    pred={(r['group'],r['id'],r['condition']):r['models']['candidate'] for r in old['rows']}
    query_rows=[];query_features=[];pairs=[]
    def add_query(pair,domain,window,feature,oldkey,base_index=None):
        q=(np.linalg.solve(cholesky,feature-mu));center_distance=float(np.sqrt(np.mean(q*q)))
        all_dist=np.sqrt(np.mean((white-q)**2,axis=1));nearest=int(np.argmin(all_dist))
        other=all_dist.copy();other[base_clips==pair['clip_id']]=np.inf;neighbor=int(np.argmin(other))
        assert np.isfinite(other[neighbor])
        def nn(index):return dict(id=base_meta[index]['id'],label=base_meta[index]['label'],source=source(base_meta[index]),distance=float(all_dist[index]),same_source_clip=bool(base_clips[index]==pair['clip_id']))
        qid=pair['clip_id']+'__'+domain+'__'+window
        row=dict(id=qid,clip_id=pair['clip_id'],source_label=pair['label'],source_signer_id=pair['signer_id'],domain=domain,window=window,
            feature_training_role='original_base_training' if domain=='native' and window=='annotation' else 'context_training' if domain=='native' else 'public_probe_only',
            center_distance=center_distance,weighted_training_percentile=float(100*weights[norms<=center_distance].sum()),
            nearest_original_training=nn(nearest),nearest_other_source_clip=nn(neighbor),reused_prediction=pred[oldkey],base_index=base_index)
        query_rows.append(row);query_features.append(feature.copy());return q
    for spec in plan['pairs']:
        index=spec['native_train_row_index'];native=pair_meta[index];assert native['id']==spec['clip_id'] and native['label']==spec['label']
        bi=int(a['pair_base_indices'][index]);assert base_meta[bi]['id']==native['original_id'] and np.array_equal(a['pair_annotation_x'][index],a['base_x'][bi])
        ma=next(i for i,r in enumerate(mp_meta) if r['id']==spec['clip_id'] and r['condition']=='provided_annotation')
        mw=next(i for i,r in enumerate(mp_meta) if r['id']==spec['clip_id'] and r['condition']=='whole_clip')
        assert classes[int(a['mp_y'][ma])]==classes[int(a['mp_y'][mw])]==spec['label']
        na=add_query(spec,'native','annotation',a['pair_annotation_x'][index].astype(float),('base',base_meta[bi]['id'],'intact'),bi)
        nw=add_query(spec,'native','whole',a['pair_context_x'][index].astype(float),('pair_context',native['id'],'intact'))
        pa=add_query(spec,'mediapipe','annotation',a['mp_x'][ma].astype(float),('mp',spec['clip_id'],'provided_annotation'))
        pw=add_query(spec,'mediapipe','whole',a['mp_x'][mw].astype(float),('mp',spec['clip_id'],'whole_clip'))
        dist=lambda x,y:float(np.sqrt(np.mean((x-y)**2)))
        pairs.append(dict(clip_id=spec['clip_id'],label=spec['label'],signer_id=spec['signer_id'],
            native_vs_mp_annotation=dist(na,pa),native_vs_mp_whole=dist(nw,pw),native_annotation_vs_whole=dist(na,nw),mp_annotation_vs_whole=dist(pa,pw)))
    metrics={}
    for domain in ['native','mediapipe']:
        for window in ['annotation','whole']:
            rs=[r for r in query_rows if r['domain']==domain and r['window']==window]
            metrics[domain+'_'+window]=dict(n=len(rs),center_distance_median=float(np.median([r['center_distance'] for r in rs])),
                center_distance_min=float(min(r['center_distance'] for r in rs)),center_distance_max=float(max(r['center_distance'] for r in rs)),
                nearest_other_source_distance_median=float(np.median([r['nearest_other_source_clip']['distance'] for r in rs])),
                nearest_other_source_label_agreement=sum(r['nearest_other_source_clip']['label']==r['source_label'] for r in rs),
                reused_full93_correct=sum(r['reused_prediction']['full93']['correct'] for r in rs))
    for p,d in plan['source_pins'].items():assert sha(p)==d,p
    with (H/'metric_and_queries.npz').open('xb') as f:np.savez_compressed(f,mu=mu,covariance=covariance,cholesky=cholesky,query_features=np.stack(query_features),query_ids=np.array([r['id'] for r in query_rows]),training_center_distances=norms)
    result=dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',plan_sha256=sha(H/'plan.json'),script_sha256=sha(__file__),arrays_sha256=sha(H/'metric_and_queries.npz'),
        inventory=inventory,per_class_inventory=perclass,queries=query_rows,pairs=pairs,summary=metrics,
        pair_distance_medians={k:float(np.median([r[k] for r in pairs])) for k in ['native_vs_mp_annotation','native_vs_mp_whole','native_annotation_vs_whole','mp_annotation_vs_whole']},
        metric_dimension=64,weighted_training_squared_radius=float(np.dot(weights,norms**2)),new_training=0,encoder_forwards=0,new_classifier_forwards=0,user_arrays_or_scores_read=0,development_arrays_read=0,reserved_arrays_read=0,
        limits=plan['limits'])
    write('result.json',result)
    print(json.dumps({k:result[k] for k in ['inventory','summary','pair_distance_medians']},ensure_ascii=False))

if __name__=='__main__':main()
