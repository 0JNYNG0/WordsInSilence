"""Offline public replay: separate observation, alignment, source guide, feedback.

No training, new detector, language model, or educational correctness oracle.
Controls are numerical invariants/perturbations, not labeled sign errors.
"""
from pathlib import Path
import json,hashlib,time,copy
from collections import Counter
import numpy as np
from motion_observer import read_observation,extract_features,summarize_motion
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x
def save(name,x):(U/name).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def subset(f,select):
    g=dict(f);n=len(f['x'])
    for k,v in f.items():
        if isinstance(v,np.ndarray) and v.ndim and len(v)==n:g[k]=v[select].copy()
    return g
def selected_groups(f):
    # Speed differences are reported, never part of alignment cost.
    # Estimator disagreement is a diagnostic, not an action feature.
    return {k:v for k,v in f['groups'].items() if not k.endswith(('_speed','_diagnostic'))}
def local_cost(q,r):
    total=np.zeros((len(q['x']),len(r['x'])))
    groups=selected_groups(q)
    for name,cols in groups.items():
        a,b=q['x'][:,cols],r['x'][:,cols]
        valid=q['mask'][:,None,cols] & r['mask'][None,:,cols]
        err=np.minimum(np.abs(a[:,None,:]-b[None,:,:]),2.)
        expected=r['mask'][None,:,cols]
        missing=expected & ~q['mask'][:,None,cols]
        denominator=expected.sum(axis=2)
        # Reference requirements are independent of query visibility. Missing
        # reference evidence is unknown, never itself a measured disagreement.
        numerator=np.where(valid,err,0.).sum(axis=2)+missing.sum(axis=2)
        group_cost=np.divide(numerator,denominator,out=np.zeros(numerator.shape),where=denominator>0)
        total+=group_cost
    return total/len(groups)
def dtw(q,r):
    cost=local_cost(q,r);n,m=cost.shape
    d=np.full((n+1,m+1),np.inf);d[0,0]=0.; prev=np.zeros((n,m),np.int8)
    for i in range(n):
        for j in range(m):
            opts=(d[i,j],d[i,j+1],d[i+1,j]);k=int(np.argmin(opts))
            d[i+1,j+1]=cost[i,j]+opts[k];prev[i,j]=k
    i,j=n-1,m-1;path=[]
    while i>=0 and j>=0:
        path.append((i,j));k=prev[i,j]
        if k==0:i-=1;j-=1
        elif k==1:i-=1
        else:j-=1
    path.reverse()
    assert path[0]==(0,0) and path[-1]==(n-1,m-1)
    return path,float(d[n,m]/len(path))
def compare(q,r):
    path,cost=dtw(q,r);qr={i:[] for i in range(len(q['x']))};rq={j:[] for j in range(len(r['x']))}
    for i,j in path:qr[i].append(j);rq[j].append(i)
    # One query representative per reference frame. Repeated path cells do not
    # inflate coverage, and every original reference frame remains in scope.
    mapping=np.array([min(v,key=lambda i:abs(i/(len(q['x']) or 1)-j/(len(r['x']) or 1))) for j,v in rq.items()])
    groups={}
    for name,cols in selected_groups(q).items():
        expect=r['mask'][:,cols];actual=q['mask'][mapping][:,cols];common=expect&actual
        delta=np.abs(q['x'][mapping][:,cols]-r['x'][:,cols])
        known=int(common.sum());expected=int(expect.sum());total=expect.size
        gaps=np.any(expect&~actual,axis=1)
        query_direct_gaps=~q['mask'][:,cols].all(axis=1)
        groups[name]={'common_cells':known,'reference_observed_cells':expected,'reference_total_cells':total,
            'reference_unobserved_cells':total-expected,'query_unobserved_at_reference_cells':expected-known,
            'coverage_of_observed_reference':known/expected if expected else None,
            'mean_absolute_difference':float(delta[common].mean()) if known else None,
            'max_absolute_difference':float(delta[common].max()) if known else None,
            'missing_reference_times':r['timestamp_us'][gaps]/1e6,
            'query_direct_unobserved_cells':int((~q['mask'][:,cols]).sum()),
            'query_direct_total_cells':int(q['mask'][:,cols].size),
            'query_direct_unobserved_times':q['timestamp_us'][query_direct_gaps]/1e6,
            'interpretation':'2D observation difference, not sign error or success'}
    missing=sum(g['query_unobserved_at_reference_cells'] for g in groups.values())
    known=sum(g['common_cells'] for g in groups.values())
    expected=sum(g['reference_observed_cells'] for g in groups.values())
    ref_unknown=sum(g['reference_unobserved_cells'] for g in groups.values())
    query_unknown=sum(g['query_direct_unobserved_cells'] for g in groups.values())
    status='NO_COMPARISON_EVIDENCE' if known==0 else ('PARTIAL_EVIDENCE' if missing or ref_unknown or query_unknown else 'OBSERVED_DIFFERENCE_ONLY')
    mapped=np.array([int(np.median(v)) for v in qr.values()])
    repeated_q=max(Counter(i for i,j in path).values());repeated_r=max(Counter(j for i,j in path).values())
    return {'path':path,'query_to_reference':mapped,'reference_to_query':mapping,'alignment_cost':cost,
        'query_frames':len(q['x']),'reference_frames':len(r['x']), 'reference_scope_retained':len(rq)==len(r['x']),
        'common_cells':known,'reference_observed_cells':expected,'query_missing_cells':missing,
        'reference_unknown_cells':ref_unknown,'query_direct_unknown_cells':query_unknown,'status':status,'groups':groups,
        'max_reference_frames_per_query':repeated_q,'max_query_frames_per_reference':repeated_r,
        'compression_note':'Reusing frames is DTW behavior; magnitude alone cannot certify omission or a sign error.',
        'educational_verdict':None,'important_omission_verdict':None}
def clip_export(row,full,f):
    return {'alias':row['alias'],'word_id':row['word_id'],'label':row['label'],'person':row['recorded_person_id'],
        'role':row['role'],'video':Path(row['source']).as_uri(),'annotation_intervals':row['annotation_intervals'],
        'scope':'Original annotation interval for comparison; full observation source retained separately.',
        'full_video_frames_retained':len(full['x']),'comparison_frames':len(f['x']),
        'timestamp_seconds':f['timestamp_us']/1e6,'frame_id':f['frame_id'],
        'feature_names':f['feature_names'],'units':f['units'],'groups':f['groups'],'x':f['x'],'mask':f['mask'],
        'motion_summary':summarize_motion(f),'sample_images':[{k:z[k] for k in ['ordinal','pts_us','file']} for z in row['frames']]}
def main():
    t0=time.perf_counter();frozen=read(U/'FROZEN.json')
    for p,h in frozen['input_pins'].items():assert sha(p)==h,p
    rows=read(O/'word100_shared_feature_bundle_20261027/visual_inputs/INDEX.json')['rows']
    records={r['clip_id']:r for r in read(O/'word100_shared_feature_bundle_20261027/FROZEN.json')['rows']}
    cache={};raw={};export=[]
    for row in rows:
        rec=read(records[row['clip_id']]['record']);a=read_observation(rec['views']['masked']['observation'])
        full=extract_features(a);sel=np.zeros(len(full['x']),bool)
        for start,end in row['annotation_intervals']:sel|=(full['timestamp_us']>=round(start*1e6))&(full['timestamp_us']<=round(end*1e6))
        assert sel.any(),row['alias']
        f=subset(full,sel);cache[row['alias']]=f;raw[row['alias']]=(a,sel)
        export.append(clip_export(row,full,f))
    save('OBSERVATIONS.json',{'status':'GEOMETRY_FROM_SAVED_PRETRAINED_MEDIAPIPE_NOT_SEMANTIC_AI','clips':export,
        'word_guide_used_by_observer':False,'model_inference_this_run':False})
    pairs=[]
    for qrow in rows:
        if qrow['recorded_person_id']!=3:continue
        for rrow in rows:
            if rrow['word_id']!=qrow['word_id'] or rrow['recorded_person_id'] not in [1,2]:continue
            t=time.perf_counter();result=compare(cache[qrow['alias']],cache[rrow['alias']])
            pairs.append({'query':qrow['alias'],'reference':rrow['alias'],'word_id':qrow['word_id'],'label':qrow['label'],
                'comparison_seconds':time.perf_counter()-t,**result})
    save('PAIRED_REPLAY.json',{'status':'PUBLIC_DEVELOPMENT_REPLAY_NOT_ACCURACY','pairs':pairs})
    controls=[]
    for row in rows:
        if row['recorded_person_id']!=3:continue
        f=cache[row['alias']];n=len(f['x']);base=compare(f,f)
        for factor in [0.75,1.5]:
            g=copy.deepcopy(f);g['timestamp_us']=np.rint(g['timestamp_us']/factor).astype(np.int64)
            # No video-frame removal or change; a pure time-parameter control.
            r=compare(g,f)
            controls.append({'clip':row['alias'],'case':f'timestamp_speed_{factor}','pass':bool(r['path']==base['path'] and abs(r['alignment_cost']-base['alignment_cost'])<1e-10),
                'scope':'same observations, changed timestamps only; not a real faster performance','cost_delta':r['alignment_cost']-base['alignment_cost']})
        a,sel=raw[row['alias']];b={k:v.copy() for k,v in a.items()};b['xy_px']=b['xy_px'].astype(np.float64)*1.3+np.array([120.,-80.])
        g=subset(extract_features(b),sel)
        invariant=np.array_equal(f['mask'],g['mask']) and np.allclose(f['x'],g['x'],atol=1e-5,equal_nan=True)
        controls.append({'clip':row['alias'],'case':'global_translate_scale','pass':bool(invariant),'scope':'same skeleton affine transform; not different body morphology'})
        controls.append({'clip':row['alias'],'case':'same_observations_missing_not_difference',
            'pass':bool(abs(base['alignment_cost'])<1e-10 and all(g['mean_absolute_difference'] in (0.,None) for g in base['groups'].values())),
            'reference_unknown_cells':base['reference_unknown_cells'],'status':base['status'],
            'scope':'Missing evidence is retained as unknown; a zero observed difference is not a correct sign verdict.'})
        for side in ['L','R']:
            g=copy.deepcopy(f);off=33 if side=='L' else 54
            cols=[i for i,name in enumerate(g['feature_names']) if any(off<=j<off+21 for j in g['dependencies'][name])]
            g['mask'][:,cols]=False;g['x'][:,cols]=np.nan;r=compare(g,f)
            required=sum(f['mask'][:,i].sum() for i in cols)
            controls.append({'clip':row['alias'],'case':side+'_hand_observation_removed',
                'pass':bool(r['educational_verdict'] is None and r['reference_scope_retained'] and (r['query_missing_cells']>0 or required==0)),
                'scope':'sensor mask deletion, not a wrong sign','new_missing_cells':r['query_missing_cells'],
                'hand_observed_cells_before':int(required),'status':r['status'],'reference_frames':r['reference_frames']})
        a,b=n*44//100,max(n*44//100+1,n*56//100);sel=np.r_[np.arange(a),np.arange(b,n)]
        g=subset(f,sel);g['timestamp_us']=np.arange(len(sel))*33333;g['frame_id']=np.arange(len(sel))
        deleted=compare(g,f)
        controls.append({'clip':row['alias'],'case':'middle_interval_removed','contract_pass':deleted['reference_scope_retained'] and deleted['important_omission_verdict'] is None,
            'deleted_frames':b-a,'deleted_original_seconds':[float(f['timestamp_us'][a]/1e6),float(f['timestamp_us'][b-1]/1e6)],
            'reference_frames':n,'cost_before':base['alignment_cost'],'cost_after':deleted['alignment_cost'],
            'max_reference_frames_per_query':deleted['max_reference_frames_per_query'],'important_omission_detected':None,
            'scope':'technical frame removal; no ground-truth linguistic necessity'})
        g=subset(f,np.arange(n-1,-1,-1));g['timestamp_us']=f['timestamp_us'].copy();g['frame_id']=f['frame_id'].copy()
        rev=compare(g,f)
        controls.append({'clip':row['alias'],'case':'reverse_time','contract_pass':rev['reference_scope_retained'] and rev['educational_verdict'] is None,
            'cost_before':base['alignment_cost'],'cost_after':rev['alignment_cost'],'important_order_error_detected':None,
            'scope':'reversed saved feature vectors; no re-extraction or linguistic error truth'})
    save('CONTROLS.json',{'status':'TECHNICAL_CONTROLS_NOT_SIGN_ERROR_ACCURACY','cases':controls})
    # Static draft check: count availability at its actual source PTS, without
    # converting projected 2D angles into unvalidated handshape class labels.
    labels=read(O/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json')['rows'];draft=[]
    for row in labels:
        full=extract_features(raw[row['alias']][0]);loc=np.flatnonzero(full['timestamp_us']==row['pts_us'])
        assert len(loc)==1;idx=int(loc[0])
        for side in ['R','L']:
            cols=full['groups'][side+'_finger_angle'];draft.append({'alias':row['alias'],'pts_us':row['pts_us'],'hand':side,
                'geometry_angles_available':int(full['mask'][idx,cols].sum()),'total_angles':len(cols),
                'visual_index_only_draft':row['labels'][side+'_index_only'],'visual_open_four_draft':row['labels'][side+'_open_four'],
                'message':'Angles being present does not imply image-visible handshape is identifiable.'})
    save('OBSERVATION_VS_DRAFT.json',{'rows':draft,'accuracy':None,'reason':'Geometry availability and RGB semantic visibility are different kinds of evidence.'})
    for p,h in frozen['protected'].items():assert sha(p)==h,p
    checks=[r['pass'] for r in controls if 'pass'in r];contracts=[r['contract_pass'] for r in controls if 'contract_pass'in r]
    summary={'status':'GEOMETRY_REPLAY_COMPLETE_SEMANTIC_MODEL_AND_EDUCATIONAL_TRUTH_PENDING',
        'clips':len(rows),'expressions':len(set(r['word_id'] for r in rows)),'recorded_people':3,
        'full_saved_frames':sum(r['full_video_frames_retained'] for r in export),'annotation_frames':sum(r['comparison_frames'] for r in export),
        'paired_public_replays':len(pairs),'technical_invariance_cases':len(checks),'technical_invariance_pass':sum(checks),
        'mutation_contract_cases':len(contracts),'mutation_contract_pass':sum(contracts),
        'important_omission_accuracy':None,'normal_false_correction_rate':None,'new_model_training_runs':0,'new_vlm_inference_runs':0,
        'educational_verdicts':0,'protected_files_unchanged':len(frozen['protected']),
        'elapsed_seconds_local_replay':time.perf_counter()-t0,
        'latency_scope':'cached observations, excludes video decoding, MediaPipe, any VLM, and UI; not production latency',
        'first_errors':[{'stage':'guide card build','error':'KeyError text in missing-description claim','resolution':'Preserved missing description as null; no fabricated text'},
            {'stage':'reference missing cost','error':'Both-missing evidence contributed to apparent difference, including identical one-hand video',
             'resolution':'CORRECTION_01.json; pre-fix implementation/results preserved, reference/query missing displayed separately from observed differences.'},
            {'stage':'affine invariant control','error':'float32 multiplication rounded transformed input; 12/12 strict invariance controls failed',
             'resolution':'Preserved first outputs. Cast source coordinates to float64 before mathematical affine transform. No matcher or tolerance changed.',
             'example_max_error_before':0.0001605128202923911,'example_max_error_after':5.541123115904156e-13}],
        'limitations':['No independently labeled genuine execution errors or tolerances.','No guaranteed semantic phase alignment.','No confirmed guide/video variant link.','Saved MediaPipe estimates and RGB drafts are not truth.','No VLM test until specific download confirmation.']}
    save('SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
