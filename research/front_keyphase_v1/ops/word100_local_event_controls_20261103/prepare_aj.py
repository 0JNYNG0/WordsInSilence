"""Freeze24 existing pixel-control caches. No video decode or model execution."""
import ast, json, time
from pathlib import Path
from collections import Counter
import numpy as np
from aj_data import AJ,O,AH,U,V,R,M,AC,WORDS,KINDS,read,sha,require,prepare_arrays,load_inputs

def write(name,value):
    with (AJ/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def pure_reference_functions():
    """AST extract arithmetic functions only; no old module/model imports."""
    namespace={'np':np}
    for path,names in [(M/'motion_model.py',{'features','normalize'}),(R/'rgb_model.py',{'rgb_input'})]:
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        require(len(nodes)==len(names),'Missing frozen pure arithmetic function')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def main():
    require(not any((AJ/n).exists() for n in ['INPUT_FROZEN.json','PREP_CHECKS.json','PLAN_FROZEN.json']),'Do not overwrite completed preparation')
    started=time.perf_counter();checks=[];pins={};historical={}
    def check(name,value):
        checks.append({'name':name,'passed':bool(value)})
        require(value,name)
    ah=read(AH/'DATASET_FROZEN.json');train=read(AH/'TRAIN_RESULT.json');verified=read(AH/'SAVED_VERIFIED.json')
    oldpins=read(O/'word100_direct_occurrence_order_20261102/FROZEN.json')['pins']
    trusted={**oldpins,**ah['pins']}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=sha(path)
        if expected is not None:check(key+':declared_hash',h==expected)
        if key in trusted:check(key+':historical_hash',h==trusted[key]);historical[key]=trusted[key]
        if key in pins:check(key+':unchanged',pins[key]==h)
        pins[key]=h;return h
    check('AH_saved_verification_PASS',verified['status']=='PASS_SAVED_ARITHMETIC_AND_LINEAGE' and verified['failed_checks']==[])
    check('AH_single_completed_model',train['seed']==42 and train['epochs']==80 and train['steps']==320 and not train['backbone_updated'])
    model_hash=pin(train['model'],train['model_sha256'])
    for f in ['DATASET_FROZEN.json','TRAIN_RESULT.json','SAVED_VERIFIED.json','EXECUTION_FROZEN.json','local_events.py','data.py']:
        pin(AH/f)
    SEAT=O/'word100_seated_web_framing_20261102';P=O/'word100_learned_guide_feedback_20261102';SIGN=O/'word100_signrep_feasibility_20261029'
    initial=[AJ/'aj_data.py',AJ/'prepare_aj.py',U/'VIDEO_CONTROLS.json',U/'prepare_video_controls.py',U/'run_video_controls.py',U/'retained_live.py',U/'CONTROL_VERIFICATION.json',P/'live_adapter.py',SEAT/'adapter.py',SEAT/'INPUT_VALIDATION.json',SEAT/'validate_prepare.py',SIGN/'common.py',SIGN/'EXPORT.json',V/'evaluate_relations.py',R/'rgb_runtime.py',M/'motion_runtime.py',M/'MOTION_GUIDE_SUPPORT.json',M/'MOTION_TARGETS.json',M/'motion_model.py',M/'NORMALIZATION.json',R/'rgb_model.py',R/'RGB_NORMALIZATION.json',AC/'process_rgb_support.py',O/'word100_direct_occurrence_order_20261102/FROZEN.json']
    for path in initial:pin(path)
    manifest=read(U/'VIDEO_CONTROLS.json');crops={r['word_id']:r for r in read(SEAT/'INPUT_VALIDATION.json')['public_crops']}
    controls=manifest['rows'];check('exact24word_kind_pairs',len(controls)==24 and {(r['word_id'],r['kind']) for r in controls}=={(w,k) for w in WORDS for k in KINDS})
    originals={r['word_id']:r for r in ah['rows'] if r['person']==12}
    normal_stats=read(M/'NORMALIZATION.json');rgb_stats=read(R/'RGB_NORMALIZATION.json');geometry_valid=ah['geometry_valid']
    check('normalization_FIT_only_unchanged',normal_stats['fit_people']==list(range(1,10)) and rgb_stats['fit_only'] is True and ah['normalization']['refit'] is False)
    pure=pure_reference_functions();guide=read(M/'MOTION_GUIDE_SUPPORT.json')
    positives=[r for r in read(M/'MOTION_TARGETS.json')['rows'] if r['kind']=='directed' and r['person']==12]
    rows=[];scope_count=0;cut_notes=[]
    for c in controls:
        word,kind=c['word_id'],c['kind'];case=word+'_'+kind;crop=crops[word];original=originals[word]
        check(case+':public_registered12',c['person_id']==12 and c['role']=='development_controls_not_training' and crop['clip_id']==original['clip_id'] and original['role']=='development')
        check(case+':crop_source_identity',c['source_sha256']==crop['video_sha256'] and Path(c['source']).resolve()==Path(crop['video']).resolve() and crop['source_sha256']==original['source_sha256'])
        pin(c['path'],c['sha256']);pin(c['source'],c['source_sha256']);pin(crop['source'],crop['source_sha256'])
        ap=U/'actual_controls'/case/'analysis.npz';bp=ap.parent/'RESULT.json';rp=V/'control_predictions'/('rgb_geometry_42_'+case+'.npz');vp=V/'control_feedback'/('rgb_geometry_42_'+case+'.json');mp=R/'feedback'/('rgb_residual_42_'+case+'.json')
        analysis_sha=pin(ap);base_sha=pin(bp);relation_sha=pin(rp);vsha=pin(vp);msha=pin(mp)
        check(case+':analysis_preexisting_Y_pin',str(ap.resolve()) in oldpins and analysis_sha==oldpins[str(ap.resolve())])
        base,body,motion=read(bp),read(vp),read(mp)
        check(case+':saved_result_source_identity',all(b['source_sha256']==c['sha256'] and b['word_id']==word for b in [base,body,motion]))
        check(case+':same_single_prototype_path',base['path']==body['path']==motion['path'] and body['path_axes']==['query_native_index','learned_canonical_index'] and all(b['learned_prototype_comparisons']==1 and b['individual_reference_comparisons']==0 and b['reference_video_reads']==0 for b in [base,body,motion]))
        p=prepare_arrays(ap,rp,normal_stats,rgb_stats,geometry_valid);n=len(p['ids']);ledger=np.asarray(c['source_frame_indices'],dtype=np.int64)
        check(case+':result_array_count',all(b['frames']==n for b in [base,body,motion]) and len(ledger)==n and n>=16)
        check(case+':source_ledger_in_original',bool(((ledger>=0)&(ledger<crop['frames'])).all()) and bool((np.diff(ledger)>=0).all()) and crop['frames']==len(original['native_frame_ids']))
        src_n=crop['frames']
        if kind in ['reencoded_normal','main_blackout']:expected=np.arange(src_n)
        elif kind=='speed_0.8':expected=np.minimum(np.arange(0,src_n,.8).astype(int),src_n-1)
        elif kind=='speed_1.25':expected=np.minimum(np.arange(0,src_n,1.25).astype(int),src_n-1)
        elif kind=='trim_initial_rest':expected=np.arange(30,src_n)
        else:
            lo,hi=c['removed_source_frames'];expected=np.array([i for i in range(src_n) if not lo<=i<=hi])
        check(case+':actual_generator_pixel_map',np.array_equal(ledger,expected) and c['output_fps']==30)
        expected_pts=np.rint(np.arange(n,dtype=np.float64)*1e6/30).astype(np.int64)
        check(case+':saved_control_pts_30fps',np.array_equal(p['ts'],expected_pts))
        blank=np.zeros(n,bool)
        if kind=='main_blackout':
            lo,hi=c['blackout_source_frames'];blank=(ledger>=lo)&(ledger<=hi)
        check(case+':actual_image_availability_preserved',np.array_equal(~p['image_available'],blank))
        check(case+':blank_not_neural_evidence',bool((~p['evidence_available'][blank]).all()) and bool((~p['motion_anchor'][blank]).all()) and bool((p['x'][blank]==0).all()))
        with np.load(ap,allow_pickle=False) as a,np.load(rp,allow_pickle=False) as rel:
            check(case+':window_vectors_not_invented',not any(k in a for k in ['window_features','window_starts','window_center_us']))
            z,mask=pure['features'](a,a['retained_probability'],rel['probability']);mask=mask&np.isfinite(z)&a['image_available'][:,None]
            reference=np.concatenate([pure['normalize'](np.where(mask,z,0),mask,normal_stats),pure['rgb_input'](a,rgb_stats)],1)
            check(case+':exact_AH915_arithmetic',np.array_equal(reference,p['x']))
        path=np.asarray(body['path'],dtype=np.int64)
        check(case+':path_indices',path.ndim==2 and path.shape[1]==2 and path[:,0].min()==0 and path[:,0].max()==n-1 and bool((np.diff(path,axis=0)>=0).all()))
        scopes=[]
        for old in motion['motion_observations']:
            target=old['target'];lo,hi=old['canonical_frames'];query=np.unique(path[(path[:,1]>=lo)&(path[:,1]<=hi),0]);check(case+':projected_scope_nonempty',len(query)>0)
            ix=np.arange(query.min(),query.max()+1);source_scope=next(s for s in guide if s['word_id']==word and s['target']==target)
            check(case+':canonical_support_exact',old['canonical_frames']==source_scope['canonical_frames'] and old['guide_id']==source_scope['guide_id'])
            check(case+':reuse_old_scope_exact',old['frame_ids']==p['ids'][ix].tolist() and old['start_seconds']==float(p['ts'][ix[0]])/1e6 and old['end_seconds']==float(p['ts'][ix[-1]])/1e6)
            source_positive=next(r for r in positives if r['word_id']==word and r['mask'][target] and r['y'][target]==1)
            pa,pb=source_positive['frames'];kept=np.flatnonzero((ledger>=pa)&(ledger<=pb));visible=kept[~blank[kept]]
            preservation=kind in KINDS[:4]
            scope=dict(target_index=target,canonical_frames=old['canonical_frames'],indices=[int(ix[0]),int(ix[-1])],frame_ids=p['ids'][ix].tolist(),timestamp_us=p['ts'][ix].tolist(),guide_id=old['guide_id'],name=old['name'],source_phrases=old['source_phrases'],source_binding='EXISTING_RELATED_PHYSICAL_COMPONENT_NOT_FULL_CLAUSE',proposal_source=str(mp),proposal_source_sha256=msha,
                selection='Exact saved learned-prototype path hull of pre-existing canonical motion bracket; no AH score selection',
                expectation='motion_preservation_under_existing_pixel_transform' if preservation else 'unknown_occurrence_after_edit_or_observation_loss',expected_occurrence=1 if preservation else None,expectation_is_new_independent_gold=False,
                original_positive_evidence=dict(path=str(M/'MOTION_TARGETS.json'),id=source_positive['id'],frames=source_positive['frames'],original_source_sha256=source_positive['source_sha256'],mapped_control_rows=kept.tolist(),visible_mapped_control_rows=visible.tolist(),source_frames_kept_in_bracket=ledger[kept].tolist(),skipped_original_bracket_frames=sorted(set(range(pa,pb+1))-set(ledger[kept].tolist())),not_exact_dense_truth=True),
                unavailable_scope_rows=ix[~p['image_available'][ix]].tolist(),whole_clause_truth=None,educational_error=None,complete_action=None)
            scopes.append(scope);scope_count+=1
            if kind=='action_cut':cut_notes.append(dict(word_id=word,target_index=target,removed_source_frames=c['removed_source_frames'],old_positive_bracket=source_positive['frames'],remaining_positive_bracket_source_frames=ledger[kept].tolist(),occurrence_truth=None,reason_ko='삭제된 부분은 기존 주요 구간과 관계가 있지만 남은 방향 움직임의 부재를 직접 확인한 새 정답은 아니다. 기존 구간 전체의 이동방향/과정 판정을 새 발생0으로 옮기지 않는다.'))
        row=dict(case_id=case,word_id=word,kind=kind,person=12,role='development_controls_not_training',original_clip_id=original['clip_id'],
            video=dict(path=c['path'],sha256=c['sha256']),crop_source=dict(path=c['source'],sha256=c['source_sha256'],crop=crop['crop'],preserves_frame_count=True,original_pts_max_recorded_crop_tolerance_us=2),original_source=dict(path=crop['source'],sha256=crop['source_sha256']),
            analysis=dict(path=str(ap),sha256=analysis_sha,historical_pin_source='Y/FROZEN.json'),relation=dict(path=str(rp),sha256=relation_sha,historical_pin_source=None,provenance='Existing V evaluate_relations output on this exact U cache; native axes joined; frozen now, no re-inference'),base_result=dict(path=str(bp),sha256=base_sha),relation_feedback=dict(path=str(vp),sha256=vsha),old_motion_feedback=dict(path=str(mp),sha256=msha),
            native_frame_ids=p['ids'].tolist(),timestamp_us=p['ts'].tolist(),input_shape=list(p['x'].shape),evidence_available=p['evidence_available'].tolist(),image_available=p['image_available'].tolist(),rgb_available=p['available'].tolist(),motion_anchor=p['motion_anchor'].tolist(),observed_fragments=p['observed_fragments'],prototype_path=body['path'],scopes=scopes,
            source_ledger=dict(source_frame_indices=ledger.tolist(),original_source_timestamp_us=[original['timestamp_us'][i] for i in ledger],control_timestamp_us=p['ts'].tolist(),blackout_source_frames=c['blackout_source_frames'],removed_source_frames=c['removed_source_frames'],source_discontinuity_output_rows=(np.flatnonzero(np.diff(ledger)>1)+1).tolist(),source_repeated_output_rows=(np.flatnonzero(np.diff(ledger)==0)+1).tolist(),skipped_source_frames=sorted(set(range(src_n))-set(ledger.tolist())),runtime_input=False,oracle_diagnostic_only=True,source_times_are_original_cache_not_new_crop_decode=True),
            rgb_lineage=dict(status=p['rgb_lineage'],window_rows=16,stride_rows=2,window_starts=[w['slot_rows'][0] for w in p['support']['windows']],center_twice_us=p['support']['center_twice_us'],recipe_paths=[str(U/'retained_live.py'),str(P/'live_adapter.py'),str(SEAT/'adapter.py'),str(SIGN/'common.py')],saved_window_vectors=False,reinterpolation_values_independently_reconstructed=False,limitation_ko='원시 캐시의window vectors가 없어 벡터수치 재보간 검증은 하지 않았다. 검증된 실제 생성코드·source SHA·원시간으로 structural support만 정확히 재구성한다.'),
            whole_clause_truth=None,educational_error=None,new_original_decode=False)
        rows.append(row)
    check('24inputs30frozenmotion_scopes',len(rows)==24 and scope_count==30)
    check('20preservation10unknown_queries',Counter(s['expected_occurrence'] for r in rows for s in r['scopes'])=={1:20,None:10})
    for path,h in pins.items():check(path+':final_preserved',sha(path)==h)
    contract=dict(status='FROZEN_EXISTING_PIXEL_CONTROL_INPUTS',rows=rows,pins=pins,historical_pins_verified=historical,input_dim=915,geometry_valid=geometry_valid,targets=ah['targets'],supported_targets=ah['supported_targets'],model=dict(path=train['model'],sha256=model_hash,seed=42,epochs=80,training_steps=320),normalization=dict(geometry=str(M/'NORMALIZATION.json'),rgb=str(R/'RGB_NORMALIZATION.json'),refit=False),rgb_recipe_scope='retained_live full-frame -> P live_adapter.extract_rgb -> seated adapter._encode -> common.starts;16stride2 center-time interpolation; no saved window vectors',new_model_loads=0,new_model_forwards=0,new_training=0,new_video_decodes=0,new_feature_extractions=0,old_result_regrades=0,whole_goal_complete=False)
    write('INPUT_FROZEN.json',contract)
    # Exercise the exported pure-cache API once; this performs no model call.
    loaded=load_inputs()
    check('exported_cache_loader24',len(loaded)==24 and all(v['x'].shape==tuple(next(r['input_shape'] for r in rows if r['case_id']==k)) for k,v in loaded.items()))
    check('no_model_media_modules_imported',not any(k in __import__('sys').modules for k in ['torch','openvino','av','cv2','mediapipe']))
    prep=dict(status='PASS_CACHED_INPUT_PREPARATION_ONLY',checks_count=len(checks),checks=checks,inputs=24,source_people=[12],original_sources=4,source_dependent_control_families=6,motion_scopes=30,preservation_expectations=20,unknown_edit_expectations=10,actual_blank_frames=sum(not x for r in rows for x in r['image_available']),recipe_only_RGB_caches=24,window_vectors_reconstructed=0,exact_AH915_arithmetic_cases=24,source_videos_sha_only=True,source_video_decodes=0,model_imports=0,model_loads=0,model_forwards=0,new_training=0,old_scores_regraded=0,input_sha256=sha(AJ/'INPUT_FROZEN.json'),elapsed_seconds=time.perf_counter()-started,cut_semantics=cut_notes,limitations=['구현자 자체 구조/계보검사이며 별도 코드 검토가 필요하다.','24변형은4원본에 종속되며 새로운24사람/독립 정답이 아니다.','원본 crop시간은 기존 생성 검증상2us 이내 보존; 이번에는source SHA와cache 시간만 읽고 디코드하지 않았다.'])
    write('PREP_CHECKS.json',prep)
    plan=dict(status='FROZEN_BEFORE_NEW_AH_CONTROL_INFERENCE',goal_ko='이미학습된 AH 후보가 기존 실제pixel변형 입력에서 정상움직임을 보존하고 관측누락을 과장하지 않는지 확인하는 단일 후속이다. 새모델선택·재학습이 아니다.',inputs=24,source_people=[12],role='previously_exposed_development_controls',model=contract['model'],input_manifest=str(AJ/'INPUT_FROZEN.json'),thresholds={'raw':.5,'high':.8,'low':.2},execution={'full_native_head_forwards':24,'head_checkpoints':1,'seeds':1,'training_steps':0,'old_model_forwards':0,'new_DTW':0,'runtime_reference_video_comparisons':0,'feature_extractions':0,'source_video_decodes':0,'fixed_prototype_scope_readouts':30,'full_native_diagnostic_readouts':30},
        inquiry_ko='정상/속도0.8/1.25/시작휴지제거의20방향query는 발생근거 보존 기대다. 기존path고정구간과fullnative 점수를 별도로 보고한다. 가림5query/삭제5query의 방향 발생정답은null이며 낮은 점수를자동정상/완료로 만들지 않는다.',
        gates={'input_and_preservation':'24 native axes/masks/support and old fields/model pins exact; full sequence evaluated once per case; no output time removal','preservation_expectation':'Report each of20 normal/speed/trim fixed-scope high confirmations; all20 expected. Full-native high20 reported separately to separate proposal versus observer limitations. This is a metamorphic research expectation, not new human gold.','blackout':'Every unavailable native frame retained and never an anchor; any high evidence is only attributed to available fragment/context, missing affected scope never fully verified or absence-assessable; no whole action completion.','action_cut':'No new occurrence0 truth or error-accuracy denominator. Preserve remaining motion evidence, report alongside partial/order status; no success/completion inference.','failure':'Do not change thresholds, scope, model seed or weights after scores. Retain failed and unknown cases, stop promotion claims and report cause.'},
        source_scope_limits_ko='관측은5물리방향이며 문구 전체·번갈아완료·반복횟수·접촉·학습자오류 정답이 아니다. 기존AE source조건/시간binding 한계를 유지한다.1/0/newcut의 임의정답을 만들지 않는다.',
        cut_semantics=cut_notes,loader_api='aj_data.load_inputs() -> {case_id: x[n,915], ids, ts, mask[n,73], available, image_available, evidence_available, motion_anchor, observed_fragments, support, prototype_path, scopes, source_ledger, cache_binding, video}; no model imports',
        pins={str(AJ/n):sha(AJ/n) for n in ['INPUT_FROZEN.json','PREP_CHECKS.json','aj_data.py','prepare_aj.py']},code_review_required_before_execution=True,no_training_or_execution_approval_inferred_from_preparation=True,whole_goal_complete=False)
    write('PLAN_FROZEN.json',plan)
    print(json.dumps({'status':prep['status'],'checks':len(checks),'inputs':24,'scopes':30,'expectations':20,'unknown':10,'input_sha':sha(AJ/'INPUT_FROZEN.json'),'plan_sha':sha(AJ/'PLAN_FROZEN.json')},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        path=AJ/'PREP_FIRST_FAILURE.txt'
        if not path.exists():path.write_text(__import__('traceback').format_exc(),encoding='utf-8')
        raise
