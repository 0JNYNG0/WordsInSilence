"""Independent saved DEV arithmetic. Never imports an evaluator or model."""
from pathlib import Path
import collections, datetime, hashlib, itertools, json, traceback, time
import numpy as np

E=Path(__file__).resolve().parent; M=E.parent; H=M.parent; G=H.parent; L=H/'full_bag_rate_rank_learning'
F=[2,3,4,5,7]; AX={2:0,3:1,4:2,5:4,7:3}
MODES=['full_bag_rate_rank','old_local_full_plus_fullbag_half'];SEEDS=[42,43,44]
checks=collections.Counter()
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r): return r['seed'],r['clip_id'],r['kind']
def ck(label,value):
    checks[label]+=1
    if not value: raise AssertionError(label)
def verify_pins(pins):
    for p,h in pins.items(): ck('immutable_hash',sha(p)==h)
def null_completions(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ['completion','semantic_completion','educational_completion']:ck('null_completion',v is None)
            null_completions(v)
    elif isinstance(value,list):
        for v in value:null_completions(v)

def body_context(q,ts):
    """Direct timestamp-neighborhood reduction, independent of production AST."""
    current=np.concatenate([q[...,:4],q[...,8:9]],axis=-1).reshape(len(ts),50)
    before=[];after=[]
    for i,t in enumerate(ts):
        before.append(current[(ts>=t-133333)&(ts<=t)].mean(axis=0))
        after.append(current[(ts>=t)&(ts<=t+133333)].mean(axis=0))
    return np.concatenate([np.asarray(before),current,np.asarray(after)],axis=1).astype(np.float32)

def relation(a,b):
    if a is None or b is None:return dict(state='unobserved',overlap_duration_us=None)
    if a[1]<=b[0]:return dict(state='forward',overlap_duration_us=0)
    if b[1]<=a[0]:return dict(state='reverse',overlap_duration_us=0)
    return dict(state='ambiguous_overlap',overlap_duration_us=int(min(a[1],b[1])-max(a[0],b[0])))

def support(card,ts,logits,mask):
    selected=card['chosen'];prefix=dict(card=card['card'],family=card['family'])
    if selected is None:
        return dict(**prefix,candidate_span=None,candidate_extent_us=None,support_indices=[],support_timestamp_us=[],support_extent_us=None,
                    state='association_unknown',reason='no_candidate',completion=None)
    a,b=selected['span'];j=F.index(card['family'])
    ck('inclusive_original_span_bounds',0<=a<=b<len(ts))
    observed=[i for i in range(a,b+1) if mask[i,j]]
    ck('original_selected_body_mask',observed==selected['body_observed'])
    points=[i for i in observed if logits[i,j]>=0]
    times=[int(ts[i]) for i in points];enough=len(points)>=2
    state='observed_progression' if enough else 'body_unknown' if len(observed)<2 else 'insufficient_progression_support'
    reason=None if enough else 'fewer_than_two_body_observations' if len(observed)<2 else 'fewer_than_two_observed_progression_points'
    return dict(**prefix,candidate_span=[a,b],candidate_extent_us=[int(ts[a]),int(ts[b])],support_indices=points,support_timestamp_us=times,
                support_extent_us=[times[0],times[-1]] if enough else None,observed_body_count=len(observed),state=state,reason=reason,completion=None)

def old_orders(cards,ts):
    raw=[];pts=[]
    for a,b in itertools.combinations(cards,2):
        if sorted([a['family'],b['family']])!=[2,3]:continue
        spans=[c['chosen']['span'] if c['chosen'] else None for c in [a,b]];names=[a['card'],b['card']]
        extents=[None if s is None else [int(ts[s[0]]),int(ts[s[1]])] for s in spans]
        rel=relation(*extents)
        if any(s is None for s in spans):rawstate=detail='unobserved'
        else:
            x,y=spans;p,q=extents
            rawstate='ambiguous_overlap' if max(x[0],y[0])<=min(x[1],y[1]) else 'forward' if x[1]<y[0] else 'reverse'
            detail=('forward_shared_boundary' if p[1]==q[0] else 'forward_separated') if rel['state']=='forward' else (
                ('reverse_shared_boundary' if q[1]==p[0] else 'reverse_separated') if rel['state']=='reverse' else 'positive_duration_overlap')
        raw.append(dict(cards=names,state=rawstate));pts.append(dict(cards=names,state=rel['state'],detail=detail,
            overlap_duration_us=rel['overlap_duration_us'],spans=spans,time_extents_us=extents))
    return raw,pts

def main():
    dst=E/'INDEPENDENT_SAVED_VERIFICATION.json';ck('exclusive_verification',not dst.exists())
    cfg=read(E/'FROZEN.json');execution=read(E/'EXECUTION_FROZEN.json');result=read(E/'COMPARISON.json')
    prior=read(L/'evaluation/COMPARISON.json');sem=read(L/'changed_span_shadow/RESULTS.json');training=read(M/'TRAINING_COMPLETE.json')
    ck('frozen_digest',result['frozen_sha256']==sha(E/'FROZEN.json'))
    ck('training_digest',result['training_sha256']==sha(M/'TRAINING_COMPLETE.json'))
    ck('original_comparison_digest',result['original_comparison_sha256']==sha(L/'evaluation/COMPARISON.json'))
    ck('original_semantics_digest',result['original_semantics_sha256']==sha(L/'changed_span_shadow/RESULTS.json'))
    ck('frozen_modes_seeds',cfg['conditions']==MODES and cfg['seeds']==SEEDS and cfg['family_order']==F)
    ck('same82scenarios',cfg['scenarios']==read(L/'evaluation/FROZEN.json')['scenarios'] and len(cfg['scenarios'])==82)
    verify_pins(cfg['pins']);verify_pins(execution['pins'])
    protected=read(G/'PROTECTIONS.json')
    ck('protected477_manifest',len(protected)==477 and protected==cfg['protected_pins']==read(M/'INPUTS_COMPLETE.json')['protected_pins'])
    verify_pins(protected)
    ck('execution_three_fixed_models',execution['pins']=={str(M/'TRAINING_COMPLETE.json'):sha(M/'TRAINING_COMPLETE.json'),**{r['model']:r['model_sha256'] for r in training['rows']}})
    ck('freeze_before_comparison',cfg['at']<execution['at']<result['at'])
    ck('unchanged_saved_FIT_review',result['saved_FIT_review']==read(M/'SAVED_FIT_VERIFICATION.json'))
    parity=read(E/'INPUT_PARITY.json')
    ck('parity_digest',sha(E/'INPUT_PARITY.json')==cfg['parity_sha256'])
    ck('parity144_96_48',parity['sequences']==144 and parity['original_cache_sequences']==96 and parity['added_source_saved_context_sequences']==48)
    expected={(seed,s['clip_id'],s['kind']) for s in cfg['scenarios'] for seed in SEEDS}
    mappings=[]
    for rows in [result['rows'],result['predictions'],prior['rows'],prior['predictions'],sem['rows'],result['timing']]:
        mapping={key(r):r for r in rows};ck('exact_unique246keys',len(rows)==len(mapping)==246 and set(mapping)==expected);mappings.append(mapping)
    rm,pm,old,oldpred,sm,tm=mappings
    inputs={(r['clip_id'],r['kind']):r for r in result['inputs']}
    ck('unique82inputs',len(result['inputs'])==len(inputs)==82)
    source_cache={}
    for s in cfg['scenarios']:
        k=s['clip_id'],s['kind'];r=inputs[k]
        ck('input_source_lineage',r['source']==s['source'] and r['source_sha256']==s['source_sha256'] and r['family_order']==F)
        ck('source_hash',sha(s['source'])==s['source_sha256']);ck('input_hash',sha(r['path'])==r['sha256'])
        with np.load(s['source'],allow_pickle=False) as z:q=z['z'][:,292:392].reshape(-1,10,10).copy();ts=z['ts'].copy()
        ck('original_integer_PTS',np.issubdtype(ts.dtype,np.integer) and (np.diff(ts)>0).all())
        expected_x=body_context(q,ts);mask=np.stack([q[:,AX[f]*2,8]>.5 for f in F],axis=1)
        with np.load(r['path'],allow_pickle=False) as z:
            ck('cached150_array_keys',set(z.files)=={'x','timestamp_us','body_observed'})
            ck('cached150_exact',z['x'].dtype==np.float32 and z['x'].shape==(len(ts),150) and np.array_equal(z['x'],expected_x))
            ck('cached_PTS_exact',np.array_equal(z['timestamp_us'],ts))
            ck('cached_body_mask_exact',z['body_observed'].dtype==np.bool_ and np.array_equal(z['body_observed'],mask))
        ck('cache_frames',r['frames']==len(ts));source_cache[k]=(ts,mask)
    island_rows=[];unknown_shape=0;all_points=0;preserved_fields=0
    for k,row in rm.items():
        baseline=old[k];meta=pm[k];ts,mask=source_cache[k[1:]];inp=inputs[k[1:]]
        for field,value in baseline.items():ck('old_row_field_exact',row[field]==value);preserved_fields+=1
        ck('only_separate_new_fields',set(row)==set(baseline)|{'saved_semantic_conditions','observed_progression','observed_progression_order'})
        ck('all10semantic_conditions_exact',row['saved_semantic_conditions']==sm[k]['conditions'] and len(row['saved_semantic_conditions'])==10)
        ck('prediction_source_lineage',meta['source']==inp['source'] and meta['source_sha256']==inp['source_sha256'])
        ck('prediction_input_lineage',meta['input_path']==inp['path'] and meta['input_sha256']==inp['sha256'])
        ck('old_prediction_lineage',meta['previous_prediction']==oldpred[k]['path'] and meta['previous_prediction_sha256']==oldpred[k]['sha256'])
        ck('prediction_shared_by_conditions',meta['shared_by_conditions']==MODES and meta['forward_calls']==1)
        ck('point_prediction_hash',sha(meta['path'])==meta['sha256'])
        with np.load(meta['path'],allow_pickle=False) as z:
            ck('point_prediction_keys',set(z.files)=={'progression_logits','timestamp_us','body_observed','family_order'})
            logits=z['progression_logits'].copy();ck('finite_point_logits',logits.shape==(len(ts),5) and np.isfinite(logits).all())
            ck('prediction_PTS_masks_families',np.array_equal(z['timestamp_us'],ts) and np.array_equal(z['body_observed'],mask) and np.array_equal(z['family_order'],F))
        for mode in MODES:
            cards=baseline['conditions'][mode]['cards'];supports=[support(c,ts,logits,mask) for c in cards]
            ck('every_support_point_and_island_exact',row['observed_progression'][mode]==supports)
            raw,pts=old_orders(cards,ts)
            ck('old_raw_discrete_order_recomputed',raw==baseline['conditions'][mode]['order'])
            ck('old_candidate_PTS_order_recomputed',pts==baseline['pts_order'][mode])
            pairs=[]
            for a,b in itertools.combinations(supports,2):
                if sorted([a['family'],b['family']])!=[2,3]:continue
                pairs.append(dict(cards=[a['card'],b['card']],original_candidate_extent_order=relation(a['candidate_extent_us'],b['candidate_extent_us']),
                    observed_body_progression_order=relation(a['support_extent_us'],b['support_extent_us']),
                    unresolved_reasons=[s['reason'] for s in [a,b] if s['reason'] is not None],expected_order_used=False,candidate_spans_unchanged=True,completion=None))
            ck('first_last_actual_PTS_order_exact',pairs==row['observed_progression_order'][mode])
            for s in supports:
                points=s['support_indices'];islands=sum(i==0 or t!=points[i-1]+1 for i,t in enumerate(points));all_points+=len(points)
                if islands>1:island_rows.append(dict(seed=k[0],clip_id=k[1],kind=k[2],condition=mode,card=s['card'],islands=islands,points=len(points)))
        if row['kind']=='all_hand_missing':
            shape=[e for cards in row['saved_semantic_conditions'].values() for c in cards for e in c['evidence'] if e['task']=='shape']
            ck('allmissing_shape_unknown',all(e['state']=='observation_unknown' for e in shape));unknown_shape+=len(shape)
        null_completions(row)
        ck('one_timing_record',tm[k]['forward_calls']==1 and np.isfinite(tm[k]['seconds']) and tm[k]['seconds']>=0)
    summary=[];gates=[]
    for seed,mode,kind in itertools.product(SEEDS,MODES,sorted({r['kind'] for r in result['rows']})):
        subset=[r for r in result['rows'] if r['seed']==seed and r['kind']==kind]
        npairs=[p for r in subset for p in r['observed_progression_order'][mode]];opairs=[p for r in subset for p in r['pts_order'][mode]]
        supports=[s for r in subset for s in r['observed_progression'][mode]]
        summary.append(dict(seed=seed,condition=mode,kind=kind,scenarios=len(subset),original_candidate_PTS_order=dict(collections.Counter(p['state'] for p in opairs)),
            observed_body_progression_order=dict(collections.Counter(p['observed_body_progression_order']['state'] for p in npairs)),support_states=dict(collections.Counter(s['state'] for s in supports)),metrics_are_not_equivalent=True))
        if kind in ['full_normal','speed075','speed125']:
            ck('normal_speed_denominator3',len(npairs)==len(opairs)==3)
            bad=[dict(clip_id=r['clip_id'],pair=p) for r in subset for p in r['observed_progression_order'][mode] if p['observed_body_progression_order']['state']!='forward']
            gates.append(dict(seed=seed,condition=mode,gate=kind+'_observed_order',required=3,passed=3-len(bad),pass_all=not bad,failures=bad))
    for seed,mode in itertools.product(SEEDS,MODES):
        cases=[]
        for r in result['rows']:
            if r['seed']!=seed or r['kind'] in ['full_normal','speed075','speed125']:continue
            for oldpair,newpair in zip(r['pts_order'][mode],r['observed_progression_order'][mode]):
                if oldpair['state']=='reverse':cases.append(dict(clip_id=r['clip_id'],kind=r['kind'],cards=oldpair['cards'],state=newpair['observed_body_progression_order']['state']))
        ck('nonempty_edited_reverse_gate',bool(cases));bad=[c for c in cases if c['state']!='reverse']
        gates.append(dict(seed=seed,condition=mode,gate='edited_reverse_retained',required=len(cases),passed=len(cases)-len(bad),pass_all=not bad,failures=bad))
    ck('all_summaries_exact',summary==result['summary']);ck('all_gates_exact',gates==result['observed_order_gates'])
    ck('old_extent_summary_exact',result['original_extent_summary']==prior['summary'])
    ck('old_semantic_summary_exact',result['original_semantic_summary']==sem['summary'])
    ck('old_L_decision_exact',result['original_L_decision']==read(L/'DECISION.json') and result['original_L_extent_G2_pass'] is False)
    ck('no_G2_G3_claim',result['G2_pass'] is None and result['G3_pass'] is None)
    ck('246_recorded_forward_calls',result['forward_calls']==sum(r['forward_calls'] for r in result['timing'])==246)
    ck('null_result_completion',result['semantic_completion'] is None and result['educational_completion'] is None)
    verify_pins(cfg['pins']);verify_pins(execution['pins']);verify_pins(protected)
    aggregate=[]
    for mode,kind in itertools.product(MODES,sorted({r['kind'] for r in result['rows']})):
        ss=[s for s in summary if s['condition']==mode and s['kind']==kind];new=collections.Counter();oldc=collections.Counter()
        for s in ss:new.update(s['observed_body_progression_order']);oldc.update(s['original_candidate_PTS_order'])
        aggregate.append(dict(condition=mode,kind=kind,old_extent=dict(oldc),observed_progression=dict(new)))
    report=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='SAVED_BODY_PROGRESS_ARITHMETIC_AND_PRESERVATION_VERIFIED_ROOT_DECISION_PENDING',
        checks=dict(checks),total_checks=sum(checks.values()),scenarios=82,rows=246,conditions=MODES,preserved_old_row_fields=preserved_fields,
        semantic_conditions_exact=10,allmissing_shape_unknown=unknown_shape,all_retained_support_points=all_points,multiple_support_islands=island_rows,
        observed_order_gates=gates,aggregate=aggregate,protected_files_verified=477,cache150_reconstructed_from_original_source=True,
        original_extent_G2_decision_preserved=True,metrics_are_not_equivalent=True,new_model_forward_calls_by_verifier=0,training_reruns=0,
        completion=None,G2_pass=None,G3_pass=None,web_changed=False,independent_human_truth=False,
        comparison_sha256=sha(E/'COMPARISON.json'),frozen_sha256=sha(E/'FROZEN.json'),verifier_sha256=sha(__file__))
    with dst.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report['total_checks'],allmissing_shape_unknown=unknown_shape,failed_gates=[g for g in gates if not g['pass_all']],aggregate=aggregate),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        with (E/f'VERIFICATION_FIRST_FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
