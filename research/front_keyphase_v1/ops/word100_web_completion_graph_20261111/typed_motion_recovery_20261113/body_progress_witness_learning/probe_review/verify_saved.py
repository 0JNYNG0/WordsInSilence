"""Independent PROBE labels/weights/saved-logit arithmetic. No model imports."""
from pathlib import Path
import collections, copy, datetime, hashlib, json, time, traceback
import numpy as np

U=Path(__file__).resolve().parent;BP=U.parent;G=BP.parent.parent;W=G/'candidate_description_supervision'
F=[2,3,4,5,7];AX={2:0,3:1,4:2,5:4,7:3}
NAMES={1:'original_direct_motion',2:'old_non_target_outside_bands',3:'own_terminal_body_hold',4:'other_family_terminal_body_hold'}
checks=collections.Counter()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(label,value):
    checks[label]+=1
    if not value:raise AssertionError(label)
def pins(p):
    for path,h in p.items():ck('hash',sha(path)==h)
def metrics(a,b,y,take):
    oc=(a>=0)==(y>0);nc=(b>=0)==(y>0)
    return dict(count=int(take.sum()),old_correct=int((take&oc).sum()),new_correct=int((take&nc).sum()),old_failures=int((take&~oc).sum()),
                new_failures=int((take&~nc).sum()),regressions=int((take&oc&~nc).sum()),recoveries=int((take&~oc&nc).sum()))
def weighted(a,b,y,w):
    out=dict(old_accuracy=float((((a>=0)==(y>0))*w).sum()),new_accuracy=float((((b>=0)==(y>0))*w).sum()))
    for key,z in [('old_BCE',a),('new_BCE',b)]:
        z=z.astype(np.float64);out[key]=float(((np.maximum(z,0)-z*y+np.log1p(np.exp(-np.abs(z))))*w).sum())
    return dict(**out,total_weight=float(w.sum()))

def main():
    dst=U/'INDEPENDENT_SAVED_VERIFICATION.json';ck('exclusive_output',not dst.exists())
    cfg=read(U/'EVALUATION_CONTRACT.json');meta=read(U/'INPUTS_COMPLETE.json');rgb=read(U/'RGB_REVIEW.json');res=read(U/'RESULTS.json')
    approval=read(U/'ROOT_REVIEW.json');started=read(U/'EVALUATION_STARTED.json');fit=read(BP/'INPUTS_COMPLETE.json');training=read(BP/'TRAINING_COMPLETE.json')
    pins(cfg['pins']);pins(meta['pins']);pins(rgb['pins']);pins(approval['pins'])
    protected=read(G/'PROTECTIONS.json');ck('477_manifest',len(protected)==477 and protected==cfg['protected_pins']==meta['protected_pins']==fit['protected_pins']);pins(protected)
    ck('review_contract_links',res['root_review_sha256']==sha(U/'ROOT_REVIEW.json') and res['contract_sha256']==sha(U/'EVALUATION_CONTRACT.json'))
    ck('started_review_links',started['root_review_sha256']==res['root_review_sha256'] and started['contract_sha256']==res['contract_sha256'])
    ck('frozen_rule',cfg['seeds']==[42,43,44] and cfg['threshold_logit']==0 and cfg['new_forwards_per_seed']==48)
    ck('input_rgb_audit_links',meta['source_review_sha256']==sha(U/'RGB_REVIEW.json') and meta['label_audit_sha256']==sha(U/'LABEL_AUDIT.json'))
    base=read(G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')
    ck('original_false',base['training_allowed'] is False and rgb['training_allowed'] is False and rgb['original_annotations_preserved'] is True)
    sources={r['clip_id']:copy.deepcopy(r) for r in base['rows'] if r['person'] in [8,9]};review={r['clip_id']:r for r in rgb['rows']}
    ck('8sources_8reviews',len(sources)==len(review)==8 and set(sources)==set(review))
    contrast={r['clip_id']:r for r in read(G/'episode_stage_contrast/RGB_CONTRAST_LABELS.json')['rows']}
    supplements=read(W/'short_action_review/REVIEW.json')['rows']+read(W/'short_action_review/FOREARM_REVIEW.json')['rows']
    for r in supplements:
        if r['clip_id'] in sources:
            event=next(e for e in sources[r['clip_id']]['events'] if e['event_id']==r['original_event']['event_id'])
            event['progression_witness_frames']=sorted(set(event['progression_witness_frames']+r['additional_progression_frames']))
    for cid,s in sources.items():
        rr=review[cid];bands={t for e in s['events'] for k in ['onset','offset'] for t in range(e[k][0],e[k][1]+1)}
        ck('RGB_review_source',rr['source_sha256']==s['source_sha256'] and rr['person']==s['person'])
        seen=set()
        for sheet in rr['sheets']:
            ck('RGB_sheet_hash_and_view_record',sheet['actual_viewed_this_review'] is True and sha(sheet['path'])==sheet['sha256']);seen.update(sheet['frames'])
        ck('qualified_holds_exact',set(rr['qualified_hold_sample_frames'])==set(s['hold_sample_frames'])-bands)
        ck('qualified_holds_seen',set(rr['qualified_hold_sample_frames'])<=seen)
        ck('excluded_boundary_holds',set(rr['excluded_inclusive_boundary_hold_frames'])==set(s['hold_sample_frames'])&bands)
    ck('24qualified_source_hold_frames',sum(len(r['qualified_hold_sample_frames']) for r in review.values())==24)
    rows=meta['sequences'];ck('48probe',len(rows)==48 and {r['person'] for r in rows}=={8,9} and all(r['split']=='probe' and r['rate']==1 for r in rows))
    for key in ['person','clip_id','source_sha256']:ck('fit_probe_disjoint',not({r[key] for r in rows}&{r[key] for r in fit['sequences']}))
    ck('probe_not_training_admitted',not(set(sources)&set(read(BP/'SOURCE_ADMISSION.json')['approved_clip_ids'])))
    original_meta=read(W/'source_rank_learning/INPUTS_COMPLETE.json')['sequences']
    raw={r['clip_id']:r for r in read(G.parent/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['clip_id'] in sources}
    times={}
    for cid,r in raw.items():
        ck('raw_time_hash',sha(r['path'])==r['sha256'])
        with np.load(r['path'],allow_pickle=False) as z:times[cid]=(z['timestamp_us'].copy(),z['frame_id'].copy())
    data=[];codes=collections.Counter();removed=0
    for r in rows:
        ck('prepared_original_hash',sha(r['path'])==r['sha256'] and sha(r['original_path'])==r['original_sha256'])
        prior=original_meta[r['original_sequence_index']]
        ck('original_probe_mapping',prior['split']=='probe' and all(r[k]==prior[k] for k in ['clip_id','person','view','reverse','source_sha256']) and r['original_path']==prior['path'])
        with np.load(r['path'],allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        with np.load(r['original_path'],allow_pickle=False) as z:old={k:z[k].copy() for k in ['x','y','mask']}
        s=sources[r['clip_id']];n=r['length'];x=d['x'];ck('x150_exact',x.shape==(n,150) and np.array_equal(x,old['x']) and np.isfinite(x).all())
        obs=np.stack([x[:,50:100].reshape(n,10,5)[:,AX[f]*2,4]>.5 for f in F],axis=1);source_obs=obs[::-1] if r['reverse'] else obs
        yy=np.zeros((n,5),np.float32);mm=np.zeros_like(yy,dtype=bool)
        groups={g['family']:{f for e in s['events'] if e['event_id'] in g['event_ids'] for f in e['progression_witness_frames']} for g in s['groups']}
        last=max(s['groups'],key=lambda g:g['offset'][1])['family'];co=contrast[r['clip_id']];bg=set(co['entry']+co['other_held_presentation']+co['returning'])
        for f,motion in groups.items():
            j=F.index(f);other=bg|set().union(*(v|(set(s['hold_sample_frames']) if k==last else set()) for k,v in groups.items() if k!=f))
            for t in motion:
                if source_obs[t,j]:yy[t,j]=1;mm[t,j]=True
            for t in other:
                if source_obs[t,j]:mm[t,j]=True
        if r['reverse']:yy=yy[::-1].copy();mm=mm[::-1].copy()
        ck('old_labels_rebuilt_from_source_and_supplements',np.array_equal(yy,old['y']) and np.array_equal(mm,old['mask']))
        ck('saved_original_labels_exact',np.array_equal(d['original_y'],old['y']) and np.array_equal(d['original_mask'],old['mask']))
        bands={t for e in s['events'] for k in ['onset','offset'] for t in range(e[k][0],e[k][1]+1)}
        bm=np.array([(n-1-i if r['reverse'] else i) in bands for i in range(n)],bool)
        removed+=int((mm&(yy==0)&bm[:,None]).sum());mm[(yy==0)&bm[:,None]]=False
        pp=np.where(mm,np.where(yy>0,1,2),0).astype(np.int8)
        for t in review[r['clip_id']]['qualified_hold_sample_frames']:
            i=n-1-t if r['reverse'] else t
            for f in groups:
                j=F.index(f)
                if obs[i,j]:
                    ck('no_hold_motion_contradiction',not(old['mask'][i,j] and old['y'][i,j]));yy[i,j]=0;mm[i,j]=True;pp[i,j]=3 if f==last else 4
        ck('target_mask_provenance_exact',np.array_equal(yy,d['y']) and np.array_equal(mm,d['mask']) and np.array_equal(pp,d['label_provenance']))
        ck('observed_valid_boundary_exact',np.array_equal(obs,d['observed']) and np.array_equal(obs,d['valid']) and np.array_equal(bm,d['inclusive_boundary_mask']))
        ck('unknown_and_unobserved_ignored',not(mm&~obs).any() and np.all(d['weights'][~mm]==0) and np.all(d['weights'][mm]>0))
        ck('no_boundary_negatives',not(mm&(yy==0)&bm[:,None]).any())
        ts,fi=times[r['clip_id']]
        ck('PTS_original_and_reverse',np.array_equal(fi,np.arange(n)) and np.array_equal(d['timestamp_us'],ts[-1]-ts[::-1] if r['reverse'] else ts) and (np.diff(d['timestamp_us'])>0).all())
        ck('source_frame_mapping',np.array_equal(d['source_frame_id'],fi[::-1] if r['reverse'] else fi))
        codes.update(map(int,pp.flat));data.append(d)
    for cid in sources:ck('3views2directions',[(r['view'],r['reverse']) for r in rows if r['clip_id']==cid]==[(v,b) for v in ['full','upper','actual_crop'] for b in [False,True]])
    for j in range(5):
        for v in [0,1]:
            eligible={r['clip_id'] for r,d in zip(rows,data) if ((d['mask'][:,j]>0)&(d['y'][:,j]==v)).any()};total=0.
            for cid in eligible:
                ids=[i for i,(r,d) in enumerate(zip(rows,data)) if r['clip_id']==cid and ((d['mask'][:,j]>0)&(d['y'][:,j]==v)).any()]
                for i in ids:
                    d=data[i];take=(d['mask'][:,j]>0)&(d['y'][:,j]==v);expected=1/(10*len(eligible)*len(ids)*int(take.sum()))
                    ck('every_equal_scope_weight',np.allclose(d['weights'][take,j],expected,rtol=0,atol=1e-15));total+=float(d['weights'][take,j].sum())
            ck('family_class_weight_point1',np.isclose(total,.1,rtol=0,atol=1e-12))
    ck('label_counts',dict(codes)=={0:30046,1:240,2:344,3:144,4:36} and removed==12)
    ck('reported_counts',{str(k):v for k,v in codes.items()}==meta['counts']['label_codes'] and res['counts']==meta['counts'])
    flat={k:np.concatenate([d[k] for d in data]) for k in ['y','mask','weights','label_provenance']};allfail=[];runs=[]
    ck('3seed_results',{r['seed'] for r in res['runs']}=={42,43,44} and len(res['runs'])==3)
    for run in res['runs']:
        seed=run['seed'];parent=next(r for r in fit['parent_models'] if r['seed']==seed);trained=next(r for r in training['rows'] if r['seed']==seed)
        ck('new_model_hash_and_lineage',sha(trained['model'])==run['model_sha256']==trained['model_sha256'] and trained['initial_model_sha256']==parent['model_sha256'])
        ck('old_prediction_hash',sha(parent['predictions'])==parent['predictions_sha256'])
        ck('new_saved_files_hashes',sha(run['predictions'])==run['predictions_sha256'] and sha(run['prediction_index'])==run['prediction_index_sha256'])
        index=read(run['prediction_index']);ck('prediction_index48',len(index['rows'])==48 and index['input_sha256']==sha(U/'INPUTS_COMPLETE.json'))
        aa=[];bb=[]
        with np.load(parent['predictions'],allow_pickle=False) as old,np.load(run['predictions'],allow_pickle=False) as new:
            ck('saved_new48keys',set(new.files)=={f'logits_{i}' for i in range(48)})
            for i,(r,d) in enumerate(zip(rows,data)):
                a=old[f"logits_{r['original_sequence_index']}"][:,:,2].copy();b=new[f'logits_{i}'].copy();aa.append(a);bb.append(b)
                ck('finite_old_new_shapes',a.shape==b.shape==(r['length'],5) and np.isfinite(a).all() and np.isfinite(b).all())
                expected=dict(input_index=i,original_sequence_index=r['original_sequence_index'],clip_id=r['clip_id'],person=r['person'],view=r['view'],reverse=r['reverse'],by_provenance={NAMES[c]:metrics(a,b,d['y'],d['label_provenance']==c) for c in NAMES})
                ck('sequence_metrics_exact',expected==index['rows'][i])
                oc=(a>=0)==(d['y']>0);nc=(b>=0)==(d['y']>0)
                for t,j in np.argwhere((d['mask']>0)&~nc):allfail.append(dict(seed=seed,clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],source_frame_id=int(d['source_frame_id'][t]),query_frame_id=int(t),family=F[j],label_code=int(d['label_provenance'][t,j]),kind='regression' if oc[t,j] else 'persisting_failure',old_logit=float(a[t,j]),new_logit=float(b[t,j])))
        a=np.concatenate(aa);b=np.concatenate(bb);y=flat['y'];bycode={NAMES[c]:metrics(a,b,y,flat['label_provenance']==c) for c in NAMES}
        ck('seed_provenance_metrics_exact',bycode==run['by_provenance'])
        family=[dict(family=f,label=v,**metrics(a[:,j],b[:,j],y[:,j],(flat['mask'][:,j]>0)&(y[:,j]==v))) for j,f in enumerate(F) for v in [0,1]]
        ck('family_class_metrics_exact',family==run['by_family_class'])
        weight=weighted(a,b,y,flat['weights']);ck('weighted_metrics_exact',all(np.isclose(v,run['equal_scope_weighted'][k],rtol=0,atol=1e-12) for k,v in weight.items()))
        grouped={}
        for groupby in ['clip_id','person','view','reverse']:
            grouped[groupby]=[]
            for value in sorted({r[groupby] for r in rows}):
                ids=[i for i,r in enumerate(rows) if r[groupby]==value];ga=np.concatenate([aa[i] for i in ids]);gb=np.concatenate([bb[i] for i in ids]);gy=np.concatenate([data[i]['y'] for i in ids]);gp=np.concatenate([data[i]['label_provenance'] for i in ids])
                grouped[groupby].append(dict(value=value,by_provenance={NAMES[c]:metrics(ga,gb,gy,gp==c) for c in NAMES}))
        ck('all_source_person_view_reverse_metrics_exact',grouped==run['by_group'])
        runs.append(dict(seed=seed,by_provenance=bycode,by_family_class=family,equal_scope_weighted=weight,by_group=grouped))
    ck('every_failure_exact',allfail==res['failures'])
    ck('recorded_forward_schedule',res['new_forward_calls']==144 and res['parent_forward_calls']==res['encoder_forward_calls']==0 and all(r['new_forward_calls']==48 and r['parent_forward_calls']==0 for r in res['runs']))
    ck('no_completion_or_web_change',res['semantic_completion'] is None and res['whole_sign_correctness'] is None and res['web_changed'] is False)
    pins(cfg['pins']);pins(protected)
    report=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='PROBE_SAVED_LABELS_AND_METRICS_SEPARATELY_VERIFIED_NOT_ADOPTED',
        verifier='Codex progress_input_review_resume; separate saved-array arithmetic from progress_probe_review implementation and root evaluation; not independent human/KSL gold',
        checks=dict(checks),total_checks=sum(checks.values()),probe_sources=8,probe_people=2,probe_sequences=48,qualified_source_hold_frames=24,
        own_hold_observations=144,original_motion_observations=240,label_codes=dict(codes),old_boundary_negatives_withheld=removed,
        source_labels_plus_prior_supplements_reconstructed=True,all_equal_scope_weights_verified=True,unknown_zero_weight=True,original_PTS_and_frame_mapping=True,
        protected_hashes_verified=477,model_forward_calls_by_verifier=0,training_reruns=0,runs=runs,failures=allfail,
        observation_counts_include_views_and_reversals=True,previously_exposed_probe_not_fresh_final=True,completion=None,web_changed=False,
        result_sha256=sha(U/'RESULTS.json'),input_sha256=sha(U/'INPUTS_COMPLETE.json'),verifier_sha256=sha(__file__))
    with dst.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report['total_checks'],runs=[dict(seed=r['seed'],by_provenance=r['by_provenance'],equal_scope_weighted=r['equal_scope_weighted']) for r in runs]),indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        with (U/f'VERIFICATION_FIRST_FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
