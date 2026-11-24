"""Separate saved-evidence audit; no pipeline/model imports or inference."""
from pathlib import Path
import ast
import hashlib
import json
import math
import statistics
import numpy as np

U=Path(__file__).resolve().parent
OLD=U.parent/'word100_stage_feedback_cycle_20261031'
STREAM=U.parent/'word100_feature_support_streaming_20261031/streaming'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    count=0;failures=[];array_checks=0
    def check(value,name,detail=None):
        nonlocal count
        count+=1
        if not bool(value):failures.append(dict(check=name,detail=detail))
    def close(actual,expected,name):check(math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-11),name,dict(actual=actual,expected=expected))
    def same_arrays(left,right,label):
        nonlocal array_checks
        with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
            check(set(a.files)==set(b.files),label+'/keys')
            for name in a.files:
                array_checks+=1
                check(a[name].shape==b[name].shape and a[name].dtype==b[name].dtype and a[name].tobytes()==b[name].tobytes(),label+'/'+name+'/bitwise')
    frozen=read(U/'FROZEN.json');result=read(U/'RESULT.json');completion=read(U/'EXECUTION_COMPLETE.json')
    expected_keys=[f'{alias}_{repeat}_{mode}' for repeat in [1,2] for alias in ['P005','P043','P079'] for mode in ['after_end','during_recording']]
    check([r['key'] for r in result['rows']]==expected_keys,'exact12_order')
    check(completion['status']=='COMPLETE' and completion['active'] is None and completion['server_closed'] and completion['chrome_closed'],'execution_shutdown')
    check(not (U/'RUNNING.json').exists(),'no_active_lock')
    for category in ['pins','protected']:
        for p,h in frozen[category].items():check(sha(p)==h,category+'/'+Path(p).name)
    fixed_run=read(OLD/'runs/pose_capacity_matched_42/RUN.json')
    for name,key in [('head.pt','checkpoint_sha256'),('probabilities.npy','prediction_sha256')]:check(sha(OLD/'runs/pose_capacity_matched_42'/name)==fixed_run[key],'original_fixed_head/'+name)
    for name in ['pipeline.py','server.py','browser_test.py']:ast.parse((U/name).read_text(encoding='utf-8-sig'))
    browser=(U/'browser_test.py').read_text(encoding='utf8')
    check('--remote-debugging-port=0' in browser and 'DevToolsActivePort' in browser and 'profile.mkdir(exist_ok=False)' in browser and 'proc.poll() is None' in browser,'isolated_browser_ownership_fix')
    check((U/'browser_test_PRE_REVIEW.py').exists() and (U/'FROZEN_PRE_REVIEW.json').exists(),'pre_review_evidence_preserved')
    source_rows={r['alias']:r for r in frozen['rows']};reference_rows={r['alias']:r for r in frozen['references']}
    refs_expected={'P005':['P001','P002','P003'],'P043':['P039','P040','P041'],'P079':['P075','P076','P077']}
    jobs={};denominators={};timestamp_checks=0;max_component_gap=0.
    for row in result['rows']:
        key=row['key'];alias=row['alias'];mode=row['mode'];d=U/'runs'/key;source=source_rows[alias]
        saved=read(d/'COMPLETE.json');backend=read(d/'BACKEND_COMPLETE.json');trace=read(d/'TRACE.json');payload=read(d/'feedback.json')
        jobs[key]=payload
        check(row==saved,'result_complete_identity/'+key)
        check(all(saved[k]==v for k,v in backend.items()),'backend_complete_identity/'+key)
        for name,digest in backend['output_sha256'].items():check(sha(d/name)==digest,'output_pin/'+key+'/'+name)
        check(saved['source_sha256']==source['source_sha256'],'original_source_hash/'+key)
        with np.load(source['input'],allow_pickle=False) as a:frame_ids=a['frame_id'];times=a['timestamp_us']
        with np.load(d/'candidate.npz',allow_pickle=False) as a:
            check(np.array_equal(a['frame_id'],frame_ids) and np.array_equal(a['timestamp_us'],times),'all_original_frame_pts/'+key)
            xy=a['xy_px'];usable=a['usable'];expected_skeleton=np.where(usable[:,:,None]&np.isfinite(xy),xy,np.nan)
        got_skeleton=np.array(payload['skeleton'],dtype=np.float64)
        check(np.array_equal(got_skeleton,expected_skeleton,equal_nan=True),'ui_skeleton_no_filled_missing/'+key)
        check(payload['frame_id']==frame_ids.tolist() and payload['seconds']==(times/1e6).tolist(),'ui_original_frame_time/'+key)
        check(len(trace['rows'])==len(trace['arrivals'])==len(frame_ids)==row['frames']==source['frames'],'all_frames_count/'+key)
        prior=STREAM/'captures'/f"{alias}_{row['repeat']}_single_decode_batch"
        original_trace=read(prior/'TIMING_TRACE.json')
        for actual,original in zip(trace['arrivals'],original_trace['arrivals']):
            check(all(actual[k]==original[k] for k in ['frame_id','source_pts','timestamp_us']),'original_pts_metadata/'+key)
            timestamp_checks+=1
        for i,(record,arrival) in enumerate(zip(trace['rows'],trace['arrivals'])):
            check(all(record[k]==arrival[k] for k in ['frame_id','source_pts','timestamp_us','available_ns','due_ns']),'arrival_record_identity/'+key)
            check(record['available_ns']<=record['process_start_ns']<=record['detector_end_ns']<=record['assignment_end_ns'],'clock_order/'+key)
            close(record['queue_wait_seconds'],(record['process_start_ns']-record['available_ns'])/1e9,'queue_wait/'+key)
            if i:check(trace['rows'][i-1]['assignment_end_ns']<=record['process_start_ns'],'sequential_detector_owner/'+key)
        end=trace['capture_end_ns'];ready=trace['ready_ns'];ack=saved['render_ack_received_ns']
        check(trace['epoch_ns']<=end<=trace['saved_ns']<=trace['feature_end_ns']<=trace['head_end_ns']<=trace['compare_end_ns']<=ready<=ack,'timing_order/'+key)
        if mode=='during_recording':
            check(end==trace['arrivals'][-1]['available_ns'],'end_is_last_frame_arrival/'+key)
            for arrival in trace['arrivals']:check(arrival['due_ns']==trace['epoch_ns']+arrival['timestamp_us']*1000,'original_pts_pacing/'+key)
        else:
            check(end>=trace['epoch_ns']+int(times[-1])*1000,'after_end_full_recording_span/'+key)
            check(all(x['available_ns']>=end for x in trace['arrivals']),'after_end_no_early_processing/'+key)
        close(saved['backend_after_end_seconds'],(ready-end)/1e9,'backend_tail/'+key)
        close(saved['capture_tail_seconds'],(trace['saved_ns']-end)/1e9,'capture_tail/'+key)
        close(saved['head_seconds'],(trace['head_end_ns']-trace['feature_end_ns'])/1e9,'head_component/'+key)
        close(saved['dtw_three_refs_seconds'],(trace['compare_end_ns']-trace['head_end_ns'])/1e9,'comparison_component/'+key)
        close(saved['end_to_render_ack_seconds'],(ack-end)/1e9,'end_to_ack/'+key)
        close(saved['backend_ready_to_render_ack_seconds'],(ack-ready)/1e9,'ready_to_ack/'+key)
        close(saved['end_to_render_ack_seconds'],saved['backend_after_end_seconds']+saved['backend_ready_to_render_ack_seconds'],'tail_component_sum/'+key)
        check(saved['remaining_at_end']==sum(x['assignment_end_ns']>end for x in trace['rows']),'remaining_frames/'+key)
        close(saved['queue_wait_max_seconds'],max(x['queue_wait_seconds'] for x in trace['rows']),'queue_max/'+key)
        gap=(trace['feature_end_ns']-trace['saved_ns'])/1e9-saved['feature_seconds'];max_component_gap=max(max_component_gap,gap)
        check(gap>=-1e-11,'feature_start_gap_nonnegative/'+key)
        body=saved['browser'];check(body['rendered_key']==key and body['visible_scope'] and body['visible_caution'] and body['reference_rows']==3 and body['paint_callbacks']==2,'ui_ack_identity/'+key)
        pairs=payload['result']['pairs'];check([p['reference'] for p in pairs]==refs_expected[alias],'same_three_references/'+key)
        valid=[]
        for p in pairs:
            pairlabel=key+'/'+p['reference'];path=p['path'];nq=p['query_frames'];nr=p['reference_frames']
            check({x[0] for x in path}==set(range(nq)) and {x[1] for x in path}==set(range(nr)) and p['reference_scope_retained'],'full_dtw_sample_scope/'+pairlabel)
            check(p['compared_cells']+p['missing_cells']==p['reference_expected_cells'],'reference_denominator_partition/'+pairlabel)
            denom_key=(alias,p['reference']);value=p['reference_expected_cells']
            if denom_key in denominators:check(value==denominators[denom_key],'reference_denominator_fixed/'+pairlabel)
            else:denominators[denom_key]=value
            if value:close(p['coverage'],p['compared_cells']/value,'reference_coverage/'+pairlabel)
            else:check(p['coverage'] is None,'zero_expected_coverage_null/'+pairlabel)
            check(p['educational_verdict'] is None and p['mandatory_omission_verdict'] is None,'no_educational_verdict/'+pairlabel)
            check(set(p['query_seconds']).issubset(payload['seconds']),'query_seek_uses_source_pts/'+pairlabel)
            with np.load(reference_rows[p['reference']]['input'],allow_pickle=False) as a:reference_times=set((a['timestamp_us']/1e6).tolist())
            check(set(p['reference_seconds']).issubset(reference_times),'reference_seek_uses_source_pts/'+pairlabel)
            check(len(p['query_seconds'])==nq and len(p['reference_seconds'])==nr and len(p['query_to_reference'])==nq,'seek_mapping_lengths/'+pairlabel)
            if p['distance'] is not None and p['compared_cells']>0:valid.append(p)
        aggregate=payload['result'];check(aggregate['comparison_is_not_correctness'],'aggregate_not_correctness/'+key)
        check(aggregate['compared_reference_count']==len(valid) and aggregate['excluded_reference_count']==3-len(valid),'aggregate_evidence_counts/'+key)
        if valid:close(aggregate['distance'],statistics.median(p['distance'] for p in valid),'aggregate_median/'+key)
        else:check(aggregate['distance'] is None and not aggregate['events'],'no_evidence_withheld/'+key)
        for event in aggregate['events']:check(event['educational_verdict'] is None and len(set(event['supporting_references']))>=2,'event_consensus_not_correctness/'+key)
        for filename in ['baseline.npz','candidate.npz','raw.npz','assignment.npz']:same_arrays(d/filename,prior/filename,'prior_saved_arrays/'+key+'/'+filename)
    for repeat in [1,2]:
        for alias in ['P005','P043','P079']:
            left=f'{alias}_{repeat}_after_end';right=f'{alias}_{repeat}_during_recording'
            check(jobs[left]==jobs[right],'identical_complete_ui_payload/'+left)
            for filename in ['baseline.npz','candidate.npz','raw.npz','assignment.npz']:same_arrays(U/'runs'/left/filename,U/'runs'/right/filename,'paired_saved_arrays/'+left+'/'+filename)
            a=np.load(U/'runs'/left/'head_probabilities.npy',allow_pickle=False);b=np.load(U/'runs'/right/'head_probabilities.npy',allow_pickle=False);array_checks+=1
            check(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'head_bitwise/'+left)
    summary=[]
    for mode in ['after_end','during_recording']:
        rr=[r for r in result['rows'] if r['mode']==mode];summary.append(dict(mode=mode,n=len(rr),metrics={key:dict(median=statistics.median(r[key] for r in rr),p95=float(np.quantile([r[key] for r in rr],.95)),range=[min(r[key] for r in rr),max(r[key] for r in rr)]) for key in ['backend_after_end_seconds','end_to_render_ack_seconds','backend_ready_to_render_ack_seconds','capture_tail_seconds','head_seconds','dtw_three_refs_seconds','detector_setup_seconds','queue_wait_max_seconds']}))
    check(result['js_errors']==[] and result['narrow_no_overflow'],'browser_recorded_checks')
    check(len(result['media_seek'])==2 and all(x['width']>0 and x['time']>0 for x in result['media_seek']),'last_screen_recorded_media_seek')
    review=dict(status='PASS_WITH_SCOPE_LIMITATIONS' if not failures else 'FAIL',reviewer='Codex rgb_labels_b, separate agent from root implementation; previous stream_capture implementation was by this reviewer and separately reviewed in the previous unit',model_inference_calls=0,training_calls=0,project_model_imports=False,implementation_files_modified=False,checks=count,failures=failures,array_bitwise_checks=array_checks,original_pts_frame_checks=timestamp_checks,executed_jobs=len(result['rows']),independent_summary=summary,maximum_unlabelled_pre_feature_gap_seconds=max_component_gap,
      findings=[dict(priority=1,status='RESOLVED_BEFORE_EXECUTION',topic='Fixed CDP port could attach to unrelated browser',before='Hard-coded9233 endpoint with first available page and Browser.close.',after='Fresh exclusive profile, ephemeral debugger port from its DevToolsActivePort and child-process liveness check; original code/freeze preserved.')],
      static_checks=['Every decoded source frame retained, original PTS monotonic contract, one detector owner, no queue drops; queue fails above2GiB rather than skipping frames.','Both conditions use the same single-decode path. The streamed end uses last_frame_available_ns, not feed return after consumer join.','Fixed original pose_capacity_matched_42 checkpoint and stored fit-reference predictions; reference IDs1/2/3 remain separate and never include the query ID5.','Full query support remains separate from96sample DTW alignment. Expected support is derived from each reference and cannot shrink due to query absence.','The UI receives structured missing-cell counts, original query/reference sampled timestamps and noninterpolated usable skeleton coordinates. No educational verdict is generated.'],
      limitations=['This is public-file PTS replay with supplied target word and fixed ROI, not live-camera measurement, word recognition or semantic stage discovery.','End is the availability of the last source frame at its PTS, not HTML video ended or the end of a terminal exposure interval.','End-to-render acknowledgement is one server monotonic-clock interval ending after browser two requestAnimationFrame callbacks and return HTTP transport. It bounds acknowledged text/table/disclaimer rendering; headless callbacks are not physical-display latency.','The acknowledgement does not wait for video loading/seek readiness. Media readiness and seek were tested separately only on the final screen.','The frozen legacy physical support gate remains10/14 angles; it is not the stricter12non-thumb gate from the previous exploratory adapter. UI cautions about observational uncertainty are necessary.','dtw_three_refs_seconds also includes saving head probabilities and prepare_case; backend ready timestamp precedes TRACE writing, output/source hashing and publishing status, whose cost is included in ready-to-ack. Feature-start timestamp is not saved, so its tiny gap after saved_ns is bounded rather than exactly reconstructed.','App/reference cache and per-job detector initialization are excluded from end-tail and reported separately. SignRep GPU, C/T3F recognition and production routes are absent.','Three public clips, six samples per mode, fixed after_end then during_recording order; results cannot establish general latency percentiles.','Saved outputs demonstrate denominator conservation on these runs. Missing-query protection additionally rests on reviewed frozen code and previous synthetic controls, not new real hand-disappearance trials.'],
      reviewed_pins={str(U/name):sha(U/name) for name in ['pipeline.py','server.py','browser_test.py','index.html','FROZEN.json','RESULT.json','EXECUTION_COMPLETE.json']},verifier_sha256=sha(__file__))
    with (U/'INDEPENDENT_REVIEW.json').open('x',encoding='utf8') as f:json.dump(review,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=review['status'],checks=count,failures=len(failures),array_bitwise_checks=array_checks),ensure_ascii=False))
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
