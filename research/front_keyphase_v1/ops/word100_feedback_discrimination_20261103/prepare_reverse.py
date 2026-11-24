"""Freeze exact existing reversed RGB inputs before a new AH readout, without inference."""
import ast
import hashlib
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from reverse_data import AN,O,AJ,AH,G,AA,M,R,V,WORDS,MODEL_SHA,read,sha,require,write_new,prepare_existing

def main():
    require(not (AN/'reverse_inputs.json').exists(),'Do not overwrite a completed input preparation')
    checks=[];pins={};historical={}
    def check(name,ok):
        require(ok,name);checks.append(name)
    saved=read(AA/'SAVED_CASES_MANIFEST.json');oldpins=saved['pins']
    ah=read(AH/'DATASET_FROZEN.json');train=read(AH/'TRAIN_RESULT.json')
    trusted={**ah['pins'],**oldpins}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);value=sha(path)
        if expected is not None:check(key+':declared_pin',value==expected)
        if key in trusted:
            check(key+':historical_pin',trusted[key]==value);historical[key]=value
        if key in pins:check(key+':unchanged',pins[key]==value)
        pins[key]=value;return value
    check('fixed_AH_seed42_final80',train['seed']==42 and train['epochs']==80 and train['steps']==320 and not train['backbone_updated'])
    check('exact_AH_checkpoint',train['model_sha256']==MODEL_SHA)
    pin(train['model'],MODEL_SHA)
    for p in [AN/'reverse_data.py',AN/'prepare_reverse.py',AA/'SAVED_CASES_MANIFEST.json',
              AH/'TRAIN_RESULT.json',AH/'DATASET_FROZEN.json',AH/'local_events.py',AH/'evaluate_local.py',
              AJ/'aj_data.py',AJ/'INPUT_FROZEN.json',G/'FROZEN.json',G/'test_main_order.py',G/'common.py',
              V/'demo_adapter.py',V/'relation_runtime.py',M/'motion_model.py',M/'MOTION_GUIDE_SUPPORT.json',
              M/'NORMALIZATION.json',R/'rgb_model.py',R/'RGB_NORMALIZATION.json',
              O/'word100_process_contract_alignment_20261102/process_rgb_support.py',
              O/'word100_learned_guide_feedback_20261102/live_adapter.py',
              O/'word100_seated_web_framing_20261102/adapter.py',
              O/'word100_signrep_feasibility_20261029/common.py']:
        pin(p)
    plan=read(G/'FROZEN.json');sources={r['word_id']:r for r in plan['sources']}
    check('four_previously_fixed_G_sources',set(sources)==set(WORDS))
    normal={r['word_id']:r for r in read(AJ/'INPUT_FROZEN.json')['rows'] if r['kind']=='reencoded_normal'}
    geo=read(M/'NORMALIZATION.json');rgb=read(R/'RGB_NORMALIZATION.json');guide=read(M/'MOTION_GUIDE_SUPPORT.json')
    check('normalization_unmodified_FIT_only',geo['fit_people']==list(range(1,10)) and rgb['fit_only'] is True and ah['normalization']['refit'] is False)
    # Independent old arithmetic functions only: no old native runtime imports.
    pure={'np':np}
    for path,names in [(M/'motion_model.py',{'features','normalize'}),(R/'rgb_model.py',{'rgb_input'})]:
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        require(len(nodes)==len(names),'Required pure arithmetic absent')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),pure)
    rows=[]
    for word in WORDS:
        rec=saved['cases'][word]['order'];source=sources[word];nr=normal[word]
        check(word+':reverse_not_AJ_completed24',rec['saved_kind']=='actual_reverse_main' and rec['case_kind']=='order')
        check(word+':public_registered12',source['person']==12 and source['clip_id']==nr['original_clip_id'] and nr['person']==12)
        video=G/'order_videos'/f'{word}_reverse_main.mp4';meta_path=video.with_suffix('.json')
        meta=read(meta_path);pin(meta_path)
        ap=G/'order_inputs'/word/'analysis.npz';bp=ap.parent/'RESULT.json';mp=G/'order_feedback'/f'{word}.json'
        check(word+':exact_AA_paths',Path(rec['analysis']).resolve()==ap.resolve() and Path(rec['video']).resolve()==video.resolve() and Path(rec['y_base_result']).resolve()==mp.resolve())
        video_sha=pin(video,rec['video_sha256']);input_sha=pin(ap,rec['analysis_sha256'])
        check(word+':AA_historical_cache_pin',str(ap.resolve()) in oldpins)
        base_sha=pin(bp);motion_sha=pin(mp,rec['y_base_result_sha256']);aa_sha=pin(rec['result'],rec['result_sha256'])
        pin(rec['base_result'],rec['base_result_sha256'])
        base,motion,aaresult=read(bp),read(mp),read(rec['result'])
        check(word+':all_results_same_source_word',all(z['word_id']==word and z['source_sha256']==video_sha for z in [base,motion,aaresult]))
        check(word+':same_original_prototype_path',base['path']==motion['path']==aaresult['path'] and base['path_axes']==['query_native_index','learned_canonical_index'])
        check(word+':original_single_DTW_no_individual_comparisons',all(z['learned_prototype_comparisons']==1 and z['individual_reference_comparisons']==0 and z['reference_video_reads']==0 for z in [base,motion,aaresult]))
        check(word+':exact_V_relation_origin',base['relation_condition']=='rgb_geometry' and base['relation_seed']==42 and base['relation_head']=='public_supervised_goal_blind_3_relations')
        p=prepare_existing(ap,geo,rgb,ah['geometry_valid']);n=len(p['ids'])
        check(word+':native_AA_axis',rec['native_frame_ids']==p['ids'].tolist() and np.array_equal(np.rint(np.asarray(rec['native_times_seconds'])*1e6).astype(np.int64),p['ts']))
        check(word+':native_30fps',np.array_equal(p['ts'],np.rint(np.arange(n)*1e6/30).astype(np.int64)))
        check(word+':saved_counts',all(z['frames']==n for z in [base,motion,aaresult]) and rec['frames']==n)
        lo,hi=source['reverse_frames'];mapping=np.arange(n);mapping[lo:hi+1]=mapping[lo:hi+1][::-1]
        check(word+':exact_actual_pixel_reversal',meta['source_frame_indices']==mapping.tolist()==rec['source_frame_mapping'] and meta['reversed_frames']==[lo,hi])
        check(word+':pixel_generator_source',meta['sha256']==video_sha and meta['source_sha256']==source['source_sha256']==nr['video']['sha256'] and Path(meta['source']).resolve()==Path(source['source']).resolve()==Path(nr['video']['path']).resolve())
        pin(source['source'],source['source_sha256'])
        check(word+':original_source_ledger_count',len(nr['native_frame_ids'])==n and len(nr['source_ledger']['original_source_timestamp_us'])==n)
        with np.load(ap,allow_pickle=False) as a:
            z,mask=pure['features'](a,a['retained_probability'],a['relation_probability'])
            mask &= np.isfinite(z)&a['image_available'][:,None]
            direct=np.concatenate([pure['normalize'](np.where(mask,z,0),mask,geo),pure['rgb_input'](a,rgb)],1)
            check(word+':exact915_existing_V_relation_alias',np.array_equal(direct,p['x']))
            check(word+':no_saved_RGB_window_vectors_invented',not any(k in a for k in ['window_features','window_starts','window_center_us']))
        check(word+':AA_image_availability',p['ids'][~p['image_available']].tolist()==rec['image_unavailable_frames'])
        path=np.asarray(base['path'],dtype=np.int64)
        check(word+':path_bounds_monotonic',path.ndim==2 and path.shape[1]==2 and path[:,0].min()==0 and path[:,0].max()==n-1 and bool((np.diff(path,axis=0)>=0).all()))
        scopes=[]
        for old in motion['motion_observations']:
            target=old['target'];canon=next(g for g in guide if g['word_id']==word and g['target']==target)
            check(word+':fixed_canonical_target',old['canonical_frames']==canon['canonical_frames'] and old['guide_id']==canon['guide_id'])
            low,high=old['canonical_frames'];query=np.unique(path[(path[:,1]>=low)&(path[:,1]<=high),0]);require(len(query)>0,'Empty old fixed scope')
            ix=np.arange(query.min(),query.max()+1)
            check(word+':saved_scope_exact',old['frame_ids']==p['ids'][ix].tolist() and old['start_seconds']==p['ts'][ix[0]]/1e6 and old['end_seconds']==p['ts'][ix[-1]]/1e6)
            scopes.append(dict(target_index=int(target),guide_id=old['guide_id'],name=old['name'],source_phrases=old['source_phrases'],
                canonical_frames=old['canonical_frames'],indices=[int(ix[0]),int(ix[-1])],frame_ids=p['ids'][ix].tolist(),timestamp_us=p['ts'][ix].tolist(),
                proposal_source=str(mp.resolve()),proposal_source_sha256=motion_sha,
                selection='Exact saved original prototype-path scope before reading new AH scores',
                occurrence_truth=None,educational_error=None,whole_clause_truth=None,complete_action=None,
                source_binding='RELATED_PHYSICAL_COMPONENT_ONLY_NOT_FULL_CLAUSE',scope_family='legacy_R_canonical_motion_DTW'))
        rows.append(dict(case_id=word+'_reverse_main',word_id=word,kind='reverse_main',person=12,role='previously_exposed_development_controls',
            original_clip_id=source['clip_id'],video=dict(path=str(video.resolve()),sha256=video_sha),
            analysis=dict(path=str(ap.resolve()),sha256=input_sha,historical_pin_source='AA/SAVED_CASES_MANIFEST.json'),
            relation=dict(storage='same_analysis_npz',field='relation_probability',condition='rgb_geometry',seed=42,
                alias='In-memory NPZ probability key only; values/frame_id/timestamp_us unchanged',new_inference=False),
            saved_base_result=dict(path=str(bp.resolve()),sha256=base_sha),saved_motion_result=dict(path=str(mp.resolve()),sha256=motion_sha),
            saved_aa_result=dict(path=rec['result'],sha256=aa_sha,flow_key=rec['flow_key'],original_order_result_untouched=True),
            native_frame_ids=p['ids'].tolist(),timestamp_us=p['ts'].tolist(),image_available=p['image_available'].tolist(),
            evidence_available=p['evidence_available'].tolist(),motion_anchor=p['motion_anchor'].tolist(),observed_fragments=p['observed_fragments'],
            input_shape=list(p['x'].shape),normalized_915_sha256=hashlib.sha256(p['x'].tobytes()).hexdigest(),prototype_path=base['path'],scopes=scopes,
            source_ledger=dict(source_frame_indices=mapping.tolist(),reversed_source_frames=[lo,hi],
                immediate_source_video=dict(path=source['source'],sha256=source['source_sha256']),
                original_public_source=nr['original_source'],crop_source=nr['crop_source'],
                original_timestamp_us=[nr['source_ledger']['original_source_timestamp_us'][i] for i in mapping],
                output_timestamp_us=p['ts'].tolist(),runtime_input=False,diagnostic_provenance_only=True,
                actual_RGB_reencoded_then_extracted=True,not_feature_rows_reversed=True),
            rgb_lineage=dict(status=p['rgb_lineage'],window_rows=16,stride_rows=2,window_vectors_saved=False,
                numerical_reinterpolation_verified=False,structural_support=p['support'],
                limitation='Actual reverse-video cache has no saved window vectors; structural16/stride2 recipe only, not vector-value reconstruction'),
            occurrence_truth=None,whole_sign_correctness=None,educational_error=None,complete_action=None))
    check('four_sources_five_fixed_queries',len(rows)==4 and sum(len(r['scopes']) for r in rows)==5)
    check('all_occurrence_truth_unknown',all(s['occurrence_truth'] is None for r in rows for s in r['scopes']))
    for p,h in pins.items():check(p+':final_pin_unchanged',sha(p)==h)
    check('no_native_model_media_modules',not any(k in sys.modules for k in ['torch','openvino','av','cv2','mediapipe']))
    result=dict(status='FROZEN_FOUR_EXISTING_ACTUAL_REVERSE_INPUTS',rows=rows,pins=pins,historical_pins_verified=historical,
        input_dim=915,geometry_valid=ah['geometry_valid'],targets=ah['targets'],supported_targets=ah['supported_targets'],
        model=dict(path=train['model'],sha256=MODEL_SHA,seed=42,epochs=80,training_steps=320),
        thresholds=dict(high=.8,low=.2,raw=.5),fixed_denominator=dict(clips=4,registered_people=1,person=12,scopes=5,full_native_diagnostics=5,known_occurrence_gold=0),
        execution=dict(new_AH_full_native_forwards=4,new_training=0,new_DTWs=0,new_RGB=0,new_sensor_extraction=0,AJ_completed24_reruns=0),
        interpretation='Physical main-range reversal is not educational error gold. Direction occurrence and AA partial temporal order are separate existing/new fields; no threshold/scope selection after new scores.',
        scope_family='legacy_R_canonical_motion_DTW',
        current_AM_fidelity_claim=False,
        AM_scope_limitation='Current AM target0 uses source-S nominal scope and may differ from this legacy R scope. This comparison retains the AJ legacy scope policy; no current-AM fidelity or exact source-event boundary claim.',
        rgb_support_limitation='Recipe provenance only; saved per-window vectors unavailable',whole_goal_complete=False)
    write_new(AN/'reverse_inputs.json',result)
    write_new(AN/'REVERSE_PREPARATION.json',dict(status='PASS_CACHE_ONLY_PREPARATION',checks_count=len(checks),checks=checks,
        input_sha256=sha(AN/'reverse_inputs.json'),native_frames=sum(len(r['native_frame_ids']) for r in rows),
        clips=4,scopes=5,registered_people=[12],independent_new_people=0,model_loads=0,model_forwards=0,training=0,
        media_decodes=0,old_outputs_changed=False,old_results_regraded=False,AJ24_reexecuted=False))
    print({'status':'PASS_CACHE_ONLY_PREPARATION','checks':len(checks),'inputs':4,'scopes':5,'input_sha256':sha(AN/'reverse_inputs.json')})

if __name__=='__main__':
    try:main()
    except BaseException:
        path=AN/'PREPARATION_FIRST_FAILURE.txt'
        if not path.exists():path.write_text(traceback.format_exc(),encoding='utf-8')
        raise
