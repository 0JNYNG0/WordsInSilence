"""Frozen public timing/equivalence study; all writes stay in this directory."""
from stream_capture import *
import argparse
import os
import traceback
from datetime import datetime, timezone

OLD = OPS/'word100_stage_feedback_cycle_20261031'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def event(**value):
    value.update(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid())
    with (HERE/'EVENTS.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def prepare():
    auth=read(UNIT/'AUTHORIZATION.json');assert auth['approved']
    plan=read(HERE/'PLAN.json');info=read(OLD/'LEARNING_INPUTS.json')
    rows=[next(r for r in info['rows'] if r['alias']==alias) for alias in plan['aliases']]
    files=[HERE/'stream_capture.py',HERE/'benchmark.py',HERE/'selftest.py',HERE/'PLAN.json',HERE/'gpu_probe.py',HERE/'GPU_PROBE.json',UNIT/'AUTHORIZATION.json',OLD/'LEARNING_INPUTS.json',LAB/'config/contract.json',LAB/'src/wis_lab/capture/video_source.py',LAB/'src/wis_lab/capture/landmarker.py',LAB/'src/wis_lab/core/observation.py',OPS/'word100_hand_assignment_20261016/capture_raw.py',OPS/'word100_hand_assignment_joint_20261016/assignment.py',OPS/'word100_hand_assignment_joint_20261016/SCOPE.json',OPS/'word100_hand_assignment_joint_20261016/POINT_CONTRACT.json',OPS/'word100_fullframe_roi_pair_20261016/run_pair.py',ROOT/'Tools/models/hand_landmarker.task',ROOT/'Tools/models/pose_landmarker_lite.task']
    manifest=[]
    for row in rows:
        assert sha256_file(row['source'])==row['source_sha256']
        files.append(Path(row['source']))
        r={key:row[key] for key in ['alias','clip_id','source','source_sha256','frames','box','input','input_sha256']}
        r['old_captures']={str(rep):str(OLD/'cpu_runtime'/f"{row['alias']}_{rep}") for rep in plan['repetitions']}
        for path in r['old_captures'].values():
            for filename in ['baseline.npz','candidate.npz','raw.npz','assignment.npz']:
                assert (Path(path)/filename).is_file()
                files.append(Path(path)/filename)
        manifest.append(r)
    frozen=dict(status='FROZEN_BEFORE_NEW_TIMING_OR_CAPTURE',rows=manifest,conditions=plan['conditions'],repetitions=plan['repetitions'],protected=read(OLD/'PROTOCOL.json')['protected'],pins={str(p):sha256_file(p) for p in files},model_initialization='New original CPU detector pair before timed source start',execution_order='For each repetition and clip, legacy batch -> single batch -> single paced; fixed order and warm-up limitations retained',scope='Previously consumed public test clips used only for timing/equivalence, never quality model choice')
    write_new(HERE/'FROZEN.json',frozen)
    event(stage='prepared',captures=len(rows)*len(plan['conditions'])*len(plan['repetitions']))


def check_pins(frozen):
    for group in ['pins','protected']:
        for path,expected in frozen[group].items():
            assert sha256_file(path)==expected,(group,path)


class PixelLedger:
    """No model: exact source pixels are hashed outside the timing experiment."""
    def __init__(self,contract):
        self.contract=contract;self.rows=[];self.handedness_disagreements=0;self.handedness_comparisons=0;self.unassigned_hands=0
    def extractor_id(self):
        return 'pixel-ledger-no-model'
    def process(self,rgb,timestamp_us,out,row):
        self.rows.append(dict(frame_id=int(out['frame_id'][row]),timestamp_us=int(timestamp_us),shape=list(rgb.shape),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))


def decode_audit(frozen,contract):
    rows=[]
    for row in frozen['rows']:
        ledger=PixelLedger(contract)
        video_source.extract(row['source'],ledger,row['clip_id'])
        single=[];raw_pts=[]
        for ordinal,pts,us,rgb,metadata in decode_once(row['source']):
            raw_pts.append(pts)
            single.append(dict(frame_id=ordinal,timestamp_us=us,shape=list(rgb.shape),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        assert ledger.rows==single,row['alias']
        assert len(single)==row['frames'] and [r['frame_id'] for r in single]==list(range(row['frames']))
        with np.load(row['input'],allow_pickle=False) as old:
            assert [r['timestamp_us'] for r in single]==old['timestamp_us'].tolist()
            assert [r['frame_id'] for r in single]==old['frame_id'].tolist()
        rows.append(dict(alias=row['alias'],frames=len(single),exact_pixel_pts_match=True,source_raw_pts=raw_pts,time_base=metadata['time_base'],source_origin_pts=metadata['origin_pts'],ledger=single))
    result=dict(status='PASS',model_calls=0,rows=rows,decodes_per_source=3,note='Legacy no-model extract uses probe+decode, single iterator uses one decode. This audit is outside measured timing.')
    write_new(HERE/'DECODE_EQUIVALENCE.json',result)
    return result


def observation_comparison(a,b):
    return {name:equal_array(getattr(a,name),getattr(b,name)) for name in ARRAY_NAMES}


def compare_saved(left,right):
    comparisons={}
    for filename in ['baseline.npz','candidate.npz','raw.npz','assignment.npz']:
        with np.load(Path(left)/filename,allow_pickle=False) as a,np.load(Path(right)/filename,allow_pickle=False) as b:
            keys=set(a.files)-{'schema'}
            comparisons[filename]={key:equal_array(a[key],b[key]) for key in keys if key in b.files}
            comparisons[filename]['same_keys']=keys==(set(b.files)-{'schema'})
    return dict(exact=all(v for group in comparisons.values() for v in group.values()),arrays=comparisons)


def capture_one(row,rep,condition,contract,capture,assignment,base):
    directory=HERE/'captures'/f"{row['alias']}_{rep}_{condition}"
    directory.mkdir(parents=True,exist_ok=False)
    init=time.perf_counter_ns()
    lm=capture.RawCaptureLandmarker(contract)
    processor=FrameProcessor(lm,assignment.HandAssigner(),contract)
    setup_seconds=(time.perf_counter_ns()-init)/1e9
    source_open_count=0
    real_open=av.open
    def counted_open(path,*args,**kwargs):
        nonlocal source_open_count
        if str(path)==str(row['source']):source_open_count+=1
        return real_open(path,*args,**kwargs)
    av.open=counted_open
    epoch=time.perf_counter_ns()
    try:
        if condition=='legacy_double_decode_batch':
            original=video_source.extract(row['source'],processor,row['clip_id'])
            metadata={key:original.sidecar[key] for key in ['width','height','time_base']}
            feed=dict(metadata=metadata,arrivals=[],maximum_queued_frames=0,last_frame_available_ns=epoch,decode_passes=2)
        else:
            original=None
            feed=feed_single_decode(row['source'],processor,condition,epoch)
            assert sha256_file(row['source'])==row['source_sha256']  # Same source-hash work as legacy extract.
        baseline,candidate,raw,sides=make_observations(processor,row,feed['metadata'],processor.extractor_id())
        candidate.sidecar.update(research_capture_mode=condition,source_raw_pts=[r['source_pts'] for r in processor.timing],production_code_changed=False)
        baseline.save(str(directory/'baseline.npz'))
        candidate.save(str(directory/'candidate.npz'))
        np.savez_compressed(directory/'raw.npz',**raw)
        np.savez_compressed(directory/'assignment.npz',sides=sides)
        write_new(directory/'assignment_trace.json',processor.diagnostics)
        saved=time.perf_counter_ns()
    finally:
        av.open=real_open
        lm.close()
    capture_end=feed['last_frame_available_ns'] if condition=='single_decode_paced' else epoch
    result=trace_summary(processor.timing,epoch,capture_end,saved,condition=='single_decode_paced')
    result.update(alias=row['alias'],repeat=rep,condition=condition,directory=str(directory),model_setup_seconds=setup_seconds,epoch_ns=epoch,capture_end_ns=capture_end,saved_artifacts_ns=saved,maximum_queued_frames=feed['maximum_queued_frames'],source_open_decode_passes=source_open_count,metadata=feed['metadata'],original_pixels_resized=False,source_frames_dropped=0,source_frames=row['frames'])
    write_new(directory/'TIMING_TRACE.json',dict(epoch_ns=epoch,capture_end_ns=capture_end,saved_ns=saved,rows=processor.timing,arrivals=feed['arrivals']))
    assert source_open_count==(2 if condition=='legacy_double_decode_batch' else 1)
    assert baseline.n_frames==row['frames']
    capture.verify_capture(raw,baseline,contract)
    replay,replay_sides,replay_diag=base.associate(raw,baseline,assignment.HandAssigner,contract)
    replay_check=observation_comparison(candidate,replay)
    replay_check['assignment_sides']=equal_array(sides,replay_sides)
    replay_check['diagnostics']=processor.diagnostics==replay_diag
    result['online_vs_frozen_batch_assignment_exact']=all(replay_check.values())
    result['assignment_replay_checks']=replay_check
    if original is not None:
        result['legacy_extract_vs_collected_rows']=observation_comparison(original,baseline)
    result['old_completed_cpu_capture_comparison']=compare_saved(directory,row['old_captures'][str(rep)])
    result['output_sha256']={str(p):sha256_file(p) for p in directory.iterdir() if p.is_file()}
    write_new(directory/'SUCCESS.json',result)
    return result


def run():
    frozen=read(HERE/'FROZEN.json');check_pins(frozen)
    assert not (HERE/'RESULT.json').exists()
    write_new(HERE/'RUN_LOCK.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat()))
    contract=read(LAB/'config/contract.json')
    t=time.perf_counter();capture,assignment,base=dependencies();dependency_setup=time.perf_counter()-t
    decode_audit(frozen,contract)
    output=[]
    for rep in frozen['repetitions']:
        for row in frozen['rows']:
            for condition in frozen['conditions']:
                event(stage='capture_start',alias=row['alias'],repeat=rep,condition=condition)
                result=capture_one(row,rep,condition,contract,capture,assignment,base)
                output.append(result)
                event(stage='capture_complete',alias=row['alias'],repeat=rep,condition=condition,whole_seconds=result['entire_wall_seconds'],tail_seconds=result['after_capture_saved_tail_seconds'],frames=result['frames'],assignment_exact=result['online_vs_frozen_batch_assignment_exact'])
    equivalence=[]
    for rep in frozen['repetitions']:
        for row in frozen['rows']:
            records=[r for r in output if r['repeat']==rep and r['alias']==row['alias']]
            for alternate in records[1:]:
                comp=compare_saved(records[0]['directory'],alternate['directory'])
                equivalence.append(dict(alias=row['alias'],repeat=rep,left=records[0]['condition'],right=alternate['condition'],**comp))
    summary=[]
    for condition in frozen['conditions']:
        records=[r for r in output if r['condition']==condition]
        s=dict(condition=condition,n=len(records))
        for field in ['entire_wall_seconds','after_capture_saved_tail_seconds','after_capture_detector_assignment_tail_seconds','model_setup_seconds','queue_wait_max_seconds']:
            vals=[r[field] for r in records]
            s[field]=dict(median=float(np.median(vals)),p95=float(np.quantile(vals,.95)),range=[min(vals),max(vals)])
        summary.append(s)
    passed=all(r['exact'] for r in equivalence) and all(r['online_vs_frozen_batch_assignment_exact'] for r in output)
    result=dict(status='COMPLETE_EXACT_EQUIVALENCE' if passed else 'COMPLETE_EQUIVALENCE_DIFFERENCES_REQUIRING_REVIEW',rows=output,summary=summary,paired_equivalence=equivalence,dependency_import_setup_seconds=dependency_setup,gpu_probe=read(HERE/'GPU_PROBE.json'),source_model_and_protected_pins_unchanged=True,scope=read(HERE/'PLAN.json')['scope_limit'],no_real_camera=True,no_training=True,no_frame_dropping=True,production_changed=False,quality_score_evaluation=False)
    check_pins(frozen)
    write_new(HERE/'RESULT.json',result)
    write_new(HERE/'EXECUTION_COMPLETE.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),status=result['status']))
    event(stage='complete',status=result['status'],captures=len(output))


def verify():
    frozen=read(HERE/'FROZEN.json');check_pins(frozen)
    result=read(HERE/'RESULT.json');checks=[]
    for record in result['rows']:
        t=read(Path(record['directory'])/'TIMING_TRACE.json');r=t['rows']
        expected=trace_summary(r,t['epoch_ns'],t['capture_end_ns'],t['saved_ns'],record['condition']=='single_decode_paced')
        checks.append(dict(alias=record['alias'],condition=record['condition'],repeat=record['repeat'],timing_recalculated=all(expected[k]==record[k] for k in expected),all_frames_and_pts=record['frames']==record['source_frames'] and [x['frame_id'] for x in r]==list(range(record['frames'])) and all(y['timestamp_us']>x['timestamp_us'] for x,y in zip(r,r[1:])),all_queue_waits_nonnegative=all(x['queue_wait_seconds']>=0 for x in r),all_output_hashes=all(sha256_file(p)==h for p,h in record['output_sha256'].items()),assignment_replay=record['online_vs_frozen_batch_assignment_exact']))
    names=['timing_recalculated','all_frames_and_pts','all_queue_waits_nonnegative','all_output_hashes','assignment_replay']
    passed=all(c[k] for c in checks for k in names) and len(checks)==18 and all(r['exact'] for r in result['paired_equivalence'])
    write_new(HERE/'VERIFICATION.json',dict(status='PASS' if passed else 'FAIL',reviewer='Same implementing Codex subagent; parent independent review still needed',checks=checks,paired_equivalence=result['paired_equivalence'],source_model_and_protected_pins_unchanged=True))
    print(json.dumps(dict(status='PASS' if passed else 'FAIL',captures=len(checks))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','verify']);args=parser.parse_args()
    try:
        globals()[args.action]()
    except BaseException as error:
        failure=HERE/(args.action.upper()+'_FIRST_FAILURE.json')
        if not failure.exists():write_new(failure,dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat()))
        raise
