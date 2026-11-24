"""Once-only new evidence composition; original evaluations remain immutable."""
from collections import Counter
import json
from pathlib import Path
import time
import traceback
import numpy as np
from compose_feedback_v2 import attach, digest, high_confirmed
from reverse_data import read,sha,write_new,require,verify_pins

U=Path(__file__).resolve().parent;O=U.parent
AE=O/'word100_guide_feature_supervision_20261102'
AA=O/'word100_partial_order_live_20261102'
AJ=O/'word100_local_event_controls_20261103'


def main():
    require((U/'COMPOSITION_STARTED.json').exists() and not (U/'COMPOSITION_FINISHED.json').exists(), 'Only the known interrupted composition can resume')
    recovery=read(U/'COMPOSITION_RECOVERY.json')
    verify_pins(recovery['preserved_output_pins'])
    original_freeze=read(U/'COMPOSITION_FROZEN.json')
    verify_pins(original_freeze['pins'])
    review=read(U/'COMPOSER_REVIEW_v2.json')
    require(review['approved_for_saved_composition'] is True,'Composer review not passed')
    verify_pins(review['code_pins'])
    roots=[AE/'SAVED_PUBLIC_COMPARISON.json',AA/'SAVED_CASES_MANIFEST.json',AJ/'INPUT_FROZEN.json',
        AJ/'RESULT.json',U/'reverse_inputs.json',U/'REVERSE_RESULT.json',U/'REVERSE_RUN_FINISHED.json',
        U/'compose_feedback_v2.py',U/'assemble_resume.py',U/'COMPOSER_REVIEW_v2.json']
    pins={str(p.resolve()):sha(p) for p in roots}
    def pin(path,expected):
        require(sha(path)==expected,'Changed input '+str(path));pins[str(Path(path).resolve())]=expected
    sr=read(roots[0]); aa=read(roots[1]);aj_input=read(roots[2]);aj_result=read(roots[3])
    reverse_input=read(roots[4]);reverse_result=read(roots[5])
    locals_={r['case_id']:r for r in aj_result['rows']+reverse_result['rows']}
    inputs={r['case_id']:r for r in aj_input['rows']+reverse_input['rows']}
    kinds=dict(normal='reencoded_normal',cut='action_cut',gap='main_blackout',order='reverse_main')
    queue=[]
    for row in sr['rows']:
        word=row['word_id'];kind=row['condition'];cid=word+'_'+kinds[kind]
        original=aa['cases'][word][kind];local=locals_[cid];inp=inputs[cid]
        for key,hkey in [('result','result_sha256'),('base_path','base_sha256'),('arrays_path','arrays_sha256')]:pin(row[key],row[hkey])
        require(row['base_path']==original['result'] and row['base_sha256']==original['result_sha256'],'Wrong AA base')
        require(row['arrays_path']==original['analysis']==inp['analysis']['path']==local['cache_binding']['path'],'Input cache identity differs')
        require(row['arrays_sha256']==original['analysis_sha256']==inp['analysis']['sha256']==local['cache_binding']['sha256'],'Input cache hash differs')
        require(original['video_sha256']==inp['video']['sha256'],'Local video hash differs')
        pin(original['video'],original['video_sha256']);pin(local['prediction']['path'],local['prediction']['sha256'])
        base=read(row['result']);previous=read(row['base_path'])
        require({k:v for k,v in base.items() if k!='guide_feature_comparison'}==previous,'AE did not preserve AA')
        ids=inp['native_frame_ids'];ts=inp['timestamp_us']
        require(ids==original['native_frame_ids'] and ts==np.rint(np.asarray(original['native_times_seconds'])*1e6).astype('int64').tolist(),'AA/native axis differs')
        with np.load(row['arrays_path'],allow_pickle=False) as a:
            require(a['frame_id'].tolist()==ids and a['timestamp_us'].tolist()==ts,'Cache/native axis differs')
            require(a['frame_id'][~a['image_available']].tolist()==local['image_unavailable_frames'],'Image gaps differ')
        with np.load(local['prediction']['path'],allow_pickle=False) as a:
            require(a['frame_id'].tolist()==ids and a['timestamp_us'].tolist()==ts,'Local prediction/native axis differs')
        identity=dict(word_id=word,video_sha256=original['video_sha256'],native_frame_ids=ids,timestamp_us=ts,
            local_case_id=cid,analysis_path=row['arrays_path'],analysis_sha256=row['arrays_sha256'],
            local_timestamp_key='output_timestamp_us' if kind=='order' else 'control_timestamp_us',
            base_value_sha256=digest(base),local_value_sha256=digest(local),
            motion_scope_family='legacy_R_canonical_motion_DTW',AM_target0_scope_fidelity=False,
            registered_id=12,role='already_exposed_public_development',
            direct_educational_gold=False,base_file=dict(path=row['result'],sha256=row['result_sha256']))
        queue.append(dict(case_id=cid,kind=kind,base=base,local=local,identity=identity))
    require(len(queue)==16 and len({q['case_id'] for q in queue})==16,'Fixed sixteen cases')
    require([dict(case_id=q['case_id'],kind=q['kind'],identity=q['identity']) for q in queue]==original_freeze['cases'], 'Resumed sixteen inputs differ from original freeze')
    write_new(U/'COMPOSITION_RESUME_FROZEN.json',dict(pins=pins,cases=[dict(case_id=q['case_id'],kind=q['kind'],identity=q['identity']) for q in queue],
        new_model_forwards=0,new_scoped_scores=0,new_DTW=0,new_training=0,
        source_extents='Each old source/native/partial-order extent preserved separately',
        completion_labels_added=0,important_wrong_handshape_gold_count=0,
        speed_controls='AJ unchanged summary reused separately, no fabricated missing order outputs'))
    write_new(U/'COMPOSITION_RESUME_STARTED.json',dict(time=time.time(),cases=16))
    dest=U/'combined_results';dest.mkdir(exist_ok=True);rows=[]
    for q in queue:
        path=dest/(q['case_id']+'.json')
        if path.exists():
            require(str(path.resolve()) in recovery['preserved_output_pins'],'Unregistered existing output')
            out=read(path);seconds=None
            require({k:v for k,v in out.items() if k!='combined_evidence_feedback'}==q['base'],'Preserved base changed')
            require(out['combined_evidence_feedback']['identity']==q['identity'],'Preserved identity changed')
        else:
            start=time.perf_counter();out=attach(q['base'],q['local'],q['identity']);seconds=time.perf_counter()-start
            write_new(path,out)
        p=out['combined_evidence_feedback']
        rows.append(dict(case_id=q['case_id'],word_id=out['word_id'],kind=q['kind'],
            result=dict(path=str(path),sha256=sha(path)),native_frames=p['native_denominator'],
            source_point_states=dict(Counter(x['observation'] for item in p['source_comparisons'] for x in item.get('native_points',[]))),
            pair_observations=dict(Counter(x['observation_status'] for x in p['partial_order_items'])),
            local_high=sum(high_confirmed(e['motion']) for e in p['motion_events']),local_scopes=len(p['motion_events']),
            local_scopes_with_gaps=sum(bool(e['motion']['unavailable_native_frames']) for e in p['motion_events']),
            global_image_unavailable_frames=len(p['global_image_unavailable_frames']),
            factor_flags=dict(Counter(flag for e in p['motion_events'] for flag in e['flags'])),
            composition_seconds=seconds,accuracy_gain=None,whole_sign_correctness=None))
    verify_pins(pins);verify_pins(recovery['preserved_output_pins'])
    summaries=[]
    for kind in kinds:
        selected=[r for r in rows if r['kind']==kind];c=Counter()
        for r in selected:c.update(r['pair_observations'])
        summaries.append(dict(kind=kind,clips=len(selected),local_high=sum(r['local_high'] for r in selected),
            local_scopes=sum(r['local_scopes'] for r in selected),order_counts=dict(c),
            local_scopes_with_gaps=sum(r['local_scopes_with_gaps'] for r in selected),
            whole_sign_error_truth=None))
    write_new(U/'COMPOSITION_RESULT.json',dict(status='SIXTEEN_SAVED_FACTORS_COMPOSED',rows=rows,summary=summaries,
        unchanged_AJ_speed_summary=[s for s in aj_result['summary'] if s['kind'] in ['speed_0.8','speed_1.25','trim_initial_rest']],
        old_scores_recomputed=False,model_forwards=0,new_training=0,video_decodes=0,
        educational_accuracy_gain_established=False,current_AM_scope_fidelity=False,
        input_and_old_scope_preservation=True,whole_goal_complete=False))
    write_new(U/'COMPOSITION_FINISHED.json',dict(status='COMPLETE',result_sha256=sha(U/'COMPOSITION_RESULT.json')))
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/('COMPOSITION_FIRST_FAILURE_'+str(time.time_ns())+'.txt');p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
