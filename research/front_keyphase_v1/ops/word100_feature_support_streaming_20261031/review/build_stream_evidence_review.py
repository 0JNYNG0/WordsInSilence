"""Independent saved-artifact audit. No detector, model, video decode, or scores run."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import json, hashlib, importlib.util
import numpy as np

U=Path(__file__).resolve().parents[1]
OLD=U.parent/'word100_stage_feedback_cycle_20261031'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name, ok): checks.append(dict(name=name,passed=bool(ok)))
def close(a,b): return bool(np.isclose(a,b,rtol=0,atol=1e-10))
result=read(U/'streaming/RESULT.json'); frozen=read(U/'streaming/FROZEN.json')
rows=result['rows']; source_rows={r['alias']:r for r in frozen['rows']}
timing=[]; hash_count=0; frame_count=0; pair_count=0; array_count=0
for r in rows:
    tag=f"{r['alias']}/{r['repeat']}/{r['condition']}"
    trace=read(Path(r['directory'])/'TIMING_TRACE.json'); tr=trace['rows']; frame_count+=len(tr)
    epoch,cap,saved=trace['epoch_ns'],trace['capture_end_ns'],trace['saved_ns']
    check(tag+'/clock_roots', (epoch,cap,saved)==(r['epoch_ns'],r['capture_end_ns'],r['saved_artifacts_ns']))
    check(tag+'/all_frame_ids_once', [t['frame_id'] for t in tr]==list(range(r['source_frames'])))
    ts=[t['timestamp_us'] for t in tr]
    check(tag+'/timestamps_ordered', ts[0]==0 and all(a<b for a,b in zip(ts,ts[1:])))
    waits=[(t['process_start_ns']-t['available_ns'])/1e9 for t in tr]
    late=[max(0,(t['available_ns']-t['due_ns'])/1e9) for t in tr]
    detectors=[(t['detector_end_ns']-t['process_start_ns'])/1e9 for t in tr]
    assignments=[(t['assignment_end_ns']-t['detector_end_ns'])/1e9 for t in tr]
    for i,t in enumerate(tr):
        check(tag+f'/frame_{i}_timing', waits[i]>=0 and detectors[i]>=0 and assignments[i]>=0 and close(waits[i],t['queue_wait_seconds']) and close(detectors[i],t['detector_seconds']) and close(assignments[i],t['assignment_seconds']))
    paced=r['condition']=='single_decode_paced'
    actual=dict(frames=len(tr),source_pts_span_seconds=ts[-1]/1e6,
        entire_wall_seconds=(saved-epoch)/1e9,capture_end_seconds=(cap-epoch)/1e9 if paced else 0,
        after_capture_detector_assignment_tail_seconds=max(0,(max(t['assignment_end_ns'] for t in tr)-cap)/1e9),
        after_capture_saved_tail_seconds=max(0,(saved-cap)/1e9),
        processed_frames_at_capture_end=sum(t['assignment_end_ns']<=cap for t in tr),
        unfinished_frames_at_capture_end=sum(t['assignment_end_ns']>cap for t in tr),
        queue_wait_mean_seconds=float(np.mean(waits)),queue_wait_p95_seconds=float(np.quantile(waits,.95)),queue_wait_max_seconds=max(waits),
        detector_service_seconds=sum(detectors),assignment_service_seconds=sum(assignments))
    if paced:
        actual.update(arrival_due_lateness_mean_seconds=float(np.mean(late)),arrival_due_lateness_max_seconds=max(late))
        check(tag+'/paced_schedule', all(t['due_ns']==epoch+t['timestamp_us']*1000 and t['available_ns']>=t['due_ns'] for t in tr))
        check(tag+'/last_arrival_endpoint', cap==tr[-1]['available_ns'])
    else: check(tag+'/batch_endpoint',cap==epoch)
    if r['condition']!='legacy_double_decode_batch':
        tb=Fraction(r['metadata']['time_base']); origin=r['metadata']['origin_pts']
        check(tag+'/source_pts_to_us',all(t['timestamp_us']==round(float((t['source_pts']-origin)*tb)*1e6) for t in tr))
    check(tag+'/summary_arithmetic',all(close(v,r[k]) for k,v in actual.items()))
    for p,h in r['output_sha256'].items():
        check(tag+'/hash/'+Path(p).name,sha(p)==h);hash_count+=1
    timing.append(dict(alias=r['alias'],repeat=r['repeat'],condition=r['condition'],**actual))

summary=[]
for s in result['summary']:
    group=[r for r in rows if r['condition']==s['condition']]; out=dict(condition=s['condition'],n=len(group))
    for key in ['entire_wall_seconds','after_capture_saved_tail_seconds','after_capture_detector_assignment_tail_seconds','model_setup_seconds','queue_wait_max_seconds']:
        a=[r[key] for r in group]; out[key]=dict(median=float(np.median(a)),p95=float(np.quantile(a,.95)),range=[min(a),max(a)])
        check(s['condition']+'/'+key+'/aggregate', close(out[key]['median'],s[key]['median']) and close(out[key]['p95'],s[key]['p95']) and out[key]['range']==s[key]['range'])
    summary.append(out)

def compare_arrays(left,right,label):
    global pair_count,array_count
    pair_count+=1
    for filename in ['baseline.npz','candidate.npz','raw.npz','assignment.npz']:
        with np.load(Path(left)/filename,allow_pickle=False) as a,np.load(Path(right)/filename,allow_pickle=False) as b:
            ak=set(a.files)-{'schema'};bk=set(b.files)-{'schema'};check(label+'/'+filename+'/keys',ak==bk)
            for k in sorted(ak&bk):
                array_count+=1;x,y=a[k],b[k]
                check(label+'/'+filename+'/'+k,x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes())
for r in rows:
    src=source_rows[r['alias']]
    compare_arrays(r['directory'],src['old_captures'][str(r['repeat'])],f"old/{r['alias']}/{r['repeat']}/{r['condition']}")
for r in rows:
    if r['condition']=='legacy_double_decode_batch':
        for other in [x for x in rows if x['alias']==r['alias'] and x['repeat']==r['repeat'] and x['condition']!=r['condition']]:
            compare_arrays(r['directory'],other['directory'],f"paired/{r['alias']}/{r['repeat']}/{other['condition']}")
pin_counts={}
for group in ['pins','protected']:
    pin_counts[group]=len(frozen[group])
    for p,h in frozen[group].items(): check(group+'/'+p,sha(p)==h)
for r in frozen['rows']:
    check(r['alias']+'/source_hash',sha(r['source'])==r['source_sha256'])
    check(r['alias']+'/input_hash',sha(r['input'])==r['input_sha256'])

# Observation-mask arithmetic only. Never load targets, prediction files, or model weights.
spec=importlib.util.spec_from_file_location('review_evidence_only',U/'evidence_adapter.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
adapter_guards=adapter.selftest()
gate_cases=[]
for name,missing,expected in [('all_observed',[],True),('all_missing',list(range(14)),False),('index_missing',[2,3,4],False),('thumb_missing',[0,1],True),('one_non_thumb_angle_missing',[13],False)]:
    m=np.ones((1,14),bool);m[:,missing]=False;s=adapter.shape_support(m)
    supported=bool(s['comparison_supported'][0]);check('shape_guard/'+name,supported==expected)
    gate_cases.append(dict(case=name,observed_angles=int(m.sum()),expected_angles=14,supported=supported))
evidence_sets=[]; evidence_input_pins=[]
for label,manifest,index in [('development',OLD/'LEARNING_INPUTS.json',OLD/'VISUAL_INDEX.json'),('fresh',U/'fresh/INPUTS.json',U/'fresh/VISUAL_INDEX.json')]:
    selected=read(manifest)['rows']; selected=[r for r in selected if label=='fresh' or r['role']=='development']
    visuals={r['alias']:r for r in read(index)['rows']}; intervals=[];shape_total=shape_supported=old_supported=partial_non_thumb=0
    for r in selected:
        evidence_input_pins.append(dict(alias=r['alias'],scope=label,path=r['input'],sha256=sha(r['input'])))
        with np.load(r['input'],allow_pickle=False) as a:
            for side in ['R','L']:
                names=r['feature_names'];ix=[i for i,n in enumerate(names) if n.startswith(side+'_') and n.endswith('_angle') and n!=side+'_elbow_angle']
                expected_order=[side+'_'+d+'_'+joint+'_angle' for d,joints in [('thumb',['mcp','ip']),('index',['mcp','pip','dip']),('middle',['mcp','pip','dip']),('ring',['mcp','pip','dip']),('little',['mcp','pip','dip'])] for joint in joints]
                check(label+'/'+r['alias']+'/'+side+'/angle_order',[names[i] for i in ix]==expected_order)
                mask=a['mask'][:,ix]; support=mask[:,2:14].all(1);old_gate=mask.sum(1)>=10
                shape_total+=len(mask);shape_supported+=int(support.sum());old_supported+=int(old_gate.sum());partial_non_thumb+=int((old_gate&~support).sum())
                check(label+'/'+r['alias']+'/'+side+'/adapter_saved_masks',np.array_equal(adapter.shape_support(mask)['comparison_supported'],support))
                for loidx,hiidx in [(0,3),(3,6),(6,9),(9,12)]:
                    lo,hi=[visuals[r['alias']]['frames'][i]['frame_id'] for i in [loidx,hiidx]]
                    ids=np.flatnonzero((a['frame_id']>=lo)&(a['frame_id']<=hi));w=a['mask'][ids][:,[names.index(side+'_hand_wrist_x'),names.index(side+'_hand_wrist_y')]];v=a['mask'][ids][:,[names.index(side+'_wrist_velocity_x'),names.index(side+'_wrist_velocity_y')]]
                    known=w.all(1);continuous=v.all(1);continuous[0]=True;valid=known&continuous
                    actual=adapter.interval_support(w,v);hidden=adapter.interval_support(np.zeros_like(w),np.zeros_like(v))
                    check(label+'/'+r['alias']+'/'+side+f'/{lo}-{hi}/support',actual['expected_frames']==len(ids) and actual['observed_frames']==int(valid.sum()) and actual['comparison_supported']==bool(valid.all()) and actual['educational_verdict'] is None)
                    check(label+'/'+r['alias']+'/'+side+f'/{lo}-{hi}/hidden_fixed_interval',hidden['expected_frames']==len(ids) and hidden['observed_frames']==0 and not hidden['comparison_supported'] and hidden['educational_verdict'] is None)
                    intervals.append(dict(alias=r['alias'],side=side,start=lo,end=hi,**actual))
    evidence_sets.append(dict(scope=label,clips=len(selected),shape_expected_hand_frames=shape_total,shape_supported_hand_frames=shape_supported,prior_count_gate_supported_hand_frames=old_supported,prior_count_gate_pass_now_rejected=partial_non_thumb,interval_count=len(intervals),unsupported_intervals=sum(not x['comparison_supported'] for x in intervals),minimum_interval_frames=min(x['expected_frames'] for x in intervals),maximum_interval_frames=max(x['expected_frames'] for x in intervals),intervals=intervals))
controls=read(U/'CONTROLS.json');dev=evidence_sets[0]
check('controls/interval_count',controls['full_interval_missing_tests']==dev['interval_count']==96)
check('controls/unknown_interval_count',controls['unknown_intervals']==dev['unsupported_intervals']==45)
speed=[{k:r[k] for k in ['seed','time_factor','known_cells','label_flips','max_probability_change']} for r in controls['speed_rows']]
for r in speed:r['flip_fraction']=r['label_flips']/r['known_cells']
paths=['controls.py','evidence_adapter.py','quality.py','summarize_quality.py','CONTROLS.json','streaming/stream_capture.py','streaming/benchmark.py','streaming/selftest.py','streaming/gpu_probe.py','streaming/RESULT.json','streaming/RESULT_KO.txt','streaming/FROZEN.json','streaming/BYTE_EQUIVALENCE.json','streaming/DECODE_EQUIVALENCE.json','streaming/SELFTEST.json','streaming/GPU_PROBE.json','streaming/VERIFICATION.json','fresh/INPUTS.json','fresh/VISUAL_INDEX.json']
evidence_pins=[dict(path=str((U/p).resolve()),sha256=sha(U/p)) for p in paths]
evidence_pins.extend(dict(path=str(p.resolve()),sha256=sha(p)) for p in [OLD/'LEARNING_INPUTS.json',OLD/'VISUAL_INDEX.json',Path(__file__)])
paced=[r for r in rows if r['condition']=='single_decode_paced']
report=dict(schema='wis_stream_and_evidence_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),reviewer='Codex separate subagent guide_stage_map; same model family, not independent human or linguistic gold review',status='PASS_WITH_EXPLICIT_SCOPE_AND_LIMITATIONS' if all(c['passed'] for c in checks) else 'FAIL',
    permitted_actions=dict(static_source_review=True,saved_array_arithmetic=True,pure_boolean_evidence_selftest=True,video_decode=0,detector_calls=0,training_calls=0,model_score_inference=0,operational_changes=0),
    conclusion_ko='저장된 18회 실행의 시간 산술·30쌍 관측 배열 일치와 고정 해시를 독립 확인했다. 1.134초는 공개 짧은 영상의 PTS 재생에서 마지막 원본 프레임 도착 뒤 관절·좌우할당 산출물 저장까지의 지연이다. 언어적 정확도·전체 제품 응답 시간·실카메라 지속 처리 보장이 아니다. 관측 부족은 정답/오답과 분리되어 있으며 제안 adapter는 운영에 연결되지 않았다.',
    verification=dict(checks=len(checks),failures=[c for c in checks if not c['passed']],capture_runs=len(rows),timing_trace_frames=frame_count,unique_source_frames=sum(r['frames'] for r in frozen['rows']),artifact_hash_checks=hash_count,npz_comparison_pairs=pair_count,npz_array_byte_comparisons=array_count,pin_counts=pin_counts,byte_rule='Exact dtype, shape, tobytes equality, including NaN payload; container ZIP bytes, schema marker and sidecar runtime metadata are not coordinate comparisons.',summary=summary),
    timing_interpretation=dict(allowed_comparison='6.08260265 s legacy batch after recording vs 1.1340049 s paced after last frame arrival, same isolated detector/assignment/source-hash/observation-save workload. Much of paced work runs during the source span. This is a tail comparison, not a 5.36x throughput claim.',paced_entire_wall_median_seconds=4.63209775,single_decode_batch_median_seconds=5.8416961,
        timer_includes=['source decoding (legacy probe plus decode; new single decode)','full original-resolution detector and unchanged causal side assignment','source SHA check','observation construction and validation','baseline/candidate/raw/assignment/assignment_trace persistence'],
        timer_excludes=['model and assigner initialization and extractor identity setup','imports, pre-run pixel/PTS audit','model shutdown after saved marker','TIMING_TRACE/SUCCESS report writes, post-run verification, replay and output hashing','SignRep RGB extraction, quality heads, DTW, browser/UI and network/camera pipeline'],
        initialization='Measured separately for each run before epoch; paced median 0.13620535 s and maximum 0.3783026 s. Reported tail is neither cold-start end-to-end latency nor live-camera latency.',
        endpoint='Paced capture end = actual availability timestamp of last original PTS frame, with no terminal exposure interval. Batch capture end = processing epoch because complete file already exists.',
        queue=dict(policy='Single sequential detector owner, unbounded FIFO for frozen short clips; fail after sampled queued RGB exceeds 2 GiB. This is not a production backpressure policy.',max_queued_frames_range=[min(r['maximum_queued_frames'] for r in paced),max(r['maximum_queued_frames'] for r in paced)],unfinished_at_last_arrival_range=[min(r['unfinished_frames_at_capture_end'] for r in paced),max(r['unfinished_frames_at_capture_end'] for r in paced)],max_arrival_lateness_seconds=max(r['arrival_due_lateness_max_seconds'] for r in paced),claim='No dropped frame in these saved successful runs. Longer/slower/live sources can accumulate backlog or fail; actual arrivals can be late. qsize measurements are samples, not a hard maximum-memory guarantee.'),
        sampling='3 public clips, 2 repeats, fixed legacy→single-batch→paced order; six observations per condition. Medians and interpolated p95 describe this sample, not population confidence or an unbiased cache-controlled benchmark.'),
    preservation=dict(verified='All stored baseline/candidate coordinates, confidence/observed/usable masks, raw detections, frame IDs/timestamps and assignment sides match in 12 new-condition pairs and 18 old-capture comparisons.',allowed_normal_cross_claim='For normal crossing that is present in these tested recordings, unchanged assigned coordinates preserve the existing crossing trajectory; online processing does not itself remove or change it.',not_established=['whether the frozen detector/side assignment was anatomically correct in every crossing','all normal crossings or all sign variants preserved','3D hand shape, contact, repetition, semantic necessity or linguistic correctness','masked/invisible pixels recovered']),
    evidence_adapter=dict(pre_review_finding='Generic count>=10/14 could mark support with all three index angles missing (11/14 present).',resolution='Root changed gate to require every index/middle/ring/little angle (positions 2:14, 12 values); two thumb angles remain diagnostic. Saved feature-name order independently checked. Added missing-index guard passes.',adapter_selftest=adapter_guards,adversarial_mask_cases=gate_cases,
        expected_scope='Shape expected_angle_count stays 14 and required non-thumb count is separately reported. Intervals use all original frame IDs in the fixed inclusive endpoints, including unobserved frames. No supported-only crop shrinks expected_frames. Every all-missing interval stays unsupported with educational_verdict=None.',
        motion_boundary='First frame velocity is waived to avoid requiring the pre-interval transition; its wrist position is still mandatory. A hypothetical one-frame interval could support coordinates without proving direction. Current audited intervals are multi-frame.',
        operational_status='Proposed observation-support adapter only. controls.py validates it; summarize_quality.py records support separately from prediction correctness. It does not currently gate a product lesson pass/fail. Generated classifier probabilities alone are not observation evidence.',
        residual_limit='Coordinate existence is not a certificate of actual RGB visibility, detector correctness, 3D pose, finger orientation or language meaning; existing explicit scope text correctly preserves this distinction.',
        old_control_guard_note='CONTROLS.json was generated before the missing-index guard addition; its stored speed predictions are unchanged. This review runs only the amended pure mask guard and stored-mask arithmetic, without re-running controls/model inference.',datasets=evidence_sets),
    speed_perturbation=dict(saved_rows=speed,provenance='Read existing CONTROLS.json only; did not recalculate model probabilities or balanced accuracy.',finding='Every seed/factor retains nonzero threshold flips: 2–9 of 172 known labels (1.16%–5.23%). Speed-invariance is not established.',definition='Only observed wrist velocity channels 9/10 are rescaled by inverse time factor (.75 means 1.333x velocity; 1.25 means .8x). Sequence, frame count/order, position and displacement inputs remain fixed.',limits='This is a partial input sensitivity probe, not an actual timestamp/interval resampling, accelerated camera, motion blur, occlusion or human speed validation. Max probability change is over all output cells, including unknown labels; flips are known-mask only. Seeds/factors share examples and are not independent samples.',remaining_action='Keep speed robustness unresolved; any future remedy must be designed with fit/development data and a fresh locked evaluation, not tune on consumed old held or fresh evaluation outcomes.'),
    gpu_claim=dict(allowed='Installed MediaPipe 0.10.33 Windows build rejected GPU delegate construction for both pose and hand with GPU processing disabled in build flags; CPU detector was retained.',not_allowed='This does not prove GPU detection impossible on all environments. No GPU detector inference, download or new runtime implementation was tested.'),
    evidence_sha256=evidence_pins,input_mask_sha256=evidence_input_pins,
    independent_scope_note='Source audit and saved data verification are independent of the streaming implementer, but same Codex family. Existing pixel-decode ledger and synthetic streaming selftest were read rather than re-executed. No fresh training quality score audit here; that belongs to reviewer B.')
out=U/'review/STREAM_AND_EVIDENCE_REVIEW.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(path=str(out),sha256=sha(out),status=report['status'],checks=len(checks),failures=report['verification']['failures'],arrays=array_count,frames=frame_count,evidence_summary=[{k:v for k,v in x.items() if k!='intervals'} for x in evidence_sets]),ensure_ascii=False))
