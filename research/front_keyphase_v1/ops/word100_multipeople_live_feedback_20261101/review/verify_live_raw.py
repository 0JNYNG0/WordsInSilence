"""Read-only saved raw-upload audit. No producer imports, decoding or inference."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import re
import numpy as np

U = Path(__file__).resolve().parents[1]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def run():
    destination=U/'review/LIVE_RAW_VERIFICATION.json'
    if destination.exists():raise FileExistsError(destination)
    checks=[];pins={}
    def check(name, value):checks.append(dict(name=name,passed=bool(value)))
    def record(path):pins[str(Path(path).resolve())]=sha(path)
    ui=read(U/'LIVE_UI_VERIFICATION.json'); upload=read(U/'PUBLIC_UPLOAD_STARTED.json')
    matches=list((U/'sessions').glob('*/'+ui['job_id']))
    if len(matches)!=1:raise ValueError('Exact reported public job must have one session directory')
    job=matches[0]; session=job.parent; folder=job/'adapter'
    request=read(job/'request.json'); timing=read(job/'REQUEST_TIMING.json')
    startup=read(session/'STARTUP.json'); complete=read(folder/'COMPLETE.json')
    result=read(job/'server_result.json'); feedback=read(job/'FEEDBACK.json')
    bank=read(U/'REFERENCE_BANK.json'); public=read(U/'PUBLIC_INPUTS.json')
    public_row=next(r for r in public['rows'] if r['clip_id']==ui['public_source'])
    source=Path(upload['source']); submitted=job/'submission.mp4'
    for p in [Path(__file__),U/'LIVE_UI_VERIFICATION.json',U/'PUBLIC_UPLOAD_STARTED.json',
              U/'PUBLIC_INPUTS.json',U/'REFERENCE_BANK.json',job/'request.json',job/'REQUEST_TIMING.json',
              session/'STARTUP.json',session/'FIRST_REQUEST_TIMING.json',folder/'COMPLETE.json',
              job/'server_result.json',job/'FEEDBACK.json',U/'adapter.py',U/'server.py',U/'engine.py',
              U/'check_live_browser.py',U/'index.html',U/'app.js',source,submitted]:record(p)
    check('saved UI check report PASS with nine checks',ui['status']=='PASS' and len(ui['checks'])==9)
    check('UI HTML and JS pins still match',sha(U/'index.html')==ui['html_sha256'] and sha(U/'app.js')==ui['js_sha256'])
    check('exact public REAL12 row',public_row['person']==12 and public_row['word_id']=='WORD0739')
    expected=public_row['source_sha256']
    check('public source/upload/adapter/UI SHA exact',all(h==expected for h in
        [sha(source),sha(submitted),upload['sha256'],complete['source_sha256'],ui['source_sha256']]))
    check('upload request byte count exact',submitted.stat().st_size==request['bytes']==source.stat().st_size)
    check('job identity exact',ui['job_id']==upload['job_id']==timing['job_id'])
    check('request word retained',request['word']==result['word']==public_row['word_id'])
    check('no submitted training or bank addition',request['training_allowed'] is False
          and request['reference_bank_addition_allowed'] is False and result['training_used_submission'] is False)
    check('saved feedback and server result identical',result==feedback)
    check('first-request timing preserved in session and UI',timing==read(session/'FIRST_REQUEST_TIMING.json')==ui['request_timing'])
    check('job completed with no adapter/server failure record',timing['status']=='COMPLETE'
          and not (job/'SERVER_FIRST_FAILURE.txt').exists() and not (folder/'FIRST_FAILURE.txt').exists())
    check('startup READY before request accepted',startup['status']=='READY'
          and datetime.fromisoformat(startup['ready_at'])<datetime.fromisoformat(request['created_at']))
    check('startup excluded from first actual request',timing['is_first_actual_analysis_request'] is True
          and timing['startup_seconds_included'] is False and startup['first_actual_request_included'] is False)
    check('startup times linked exactly',timing['separately_recorded_startup_seconds']==startup['total_startup_seconds']
          and timing['separately_recorded_warmup_seconds']==startup['warmup_seconds'])
    warm=startup['warmup']; device=complete['device']
    check('startup uses synthetic-only warmup',warm['real_video_inference']==0
          and warm['synthetic_rgb_windows']==1 and warm['synthetic_head_windows']==1)
    check('recorded GPU route matches startup',device==warm['device'] and device['requested']=='GPU'
          and device['actual']==['GPU.0'] and device['precision']=='f16' and device['failed_attempts']==[])
    reuse=complete['runtime_reuse']
    check('initialized runtimes reused but no video predictions cached',reuse['signrep_initialized_before_video'] is True
          and reuse['heads_initialized_before_video'] is True and reuse['predictions_cached'] is False
          and complete['cached_predictions_used'] is False)
    check('one signer/postcapture/no training semantics retained',complete['one_signer_per_video'] is True
          and complete['postcapture_future_context'] is True and complete['training'] is False
          and complete['no_educational_or_sign_correctness_verdict'] is True)
    check('all reported latency fields agree',all(result['latency'][k]==v for k,v in complete['latency'].items())
          and ui['latency']==result['latency'])
    check('server processing exceeds engine which exceeds adapter duration',timing['analysis_seconds']>=result['latency']['total_analysis_seconds']>=complete['latency']['total_adapter_seconds'])
    check('startup and stage durations nonnegative',all(v>=0 for v in complete['latency'].values())
          and 0<=startup['warmup_seconds']<=startup['total_startup_seconds'])
    for name,h in complete['artifacts'].items():
        path=folder/name;check('adapter artifact SHA '+name,sha(path)==h);record(path)
    a=npz(folder/'analysis.npz'); ob=npz(folder/'observations.npz'); raw=npz(folder/'raw.npz')
    face=npz(folder/'face.npz'); rgb=npz(folder/'rgb_windows.npz')
    n=complete['frames']; ids=a['frame_id']; ts=a['timestamp_us']; names=complete['feature_names']
    check('123 full native frames retained',n==123 and a['x'].shape==(n,69) and a['mask'].shape==(n,69)
          and a['mask'].dtype==bool and a['rgb'].shape==(n,768) and a['probabilities'].shape==(n,7))
    check('strict source frame and real PTS axes',np.all(np.diff(ids)>0) and np.all(np.diff(ts)>0)
          and ts[0]==0 and ts[-1]<=20_000_000 and n<=1200)
    for name,z in [('observation',ob),('raw',raw),('face',face)]:
        check(name+' exact source axes',np.array_equal(ids,z['frame_id']) and np.array_equal(ts,z['timestamp_us']))
    check('supported geometry and RGB finite',np.isfinite(a['x'][a['mask']]).all() and np.isfinite(a['rgb']).all())
    centers=np.array([i for i in range(6,n-6) if np.array_equal(ids[i-6:i+7],np.arange(ids[i]-6,ids[i]+7))])
    availability=np.zeros((n,7),bool);availability[:,:3]=True;availability[centers,3:]=True
    check('native3/full and new4/111 exact centers only',len(centers)==111
          and complete['new_four_predicted_frames']==111 and np.array_equal(a['new_four_center_indices'],centers)
          and np.array_equal(np.isfinite(a['probabilities']),availability)
          and np.array_equal(a['posterior_available'],availability))
    check('posteriors in binary probability range',((a['probabilities'][availability]>=0)&(a['probabilities'][availability]<=1)).all())
    support=np.zeros((n,7),bool)
    for j,target in enumerate(complete['targets']):
        if target.endswith('higher'):
            columns=[names.index(s+'_palm_center_'+d) for s in ['L','R'] for d in ['x','y']]
            support[:,j]=a['mask'][:,columns].all(1)
        else:
            s=target[0];wrist=[names.index(s+'_hand_wrist_'+d) for d in ['x','y']]
            angles=[i for i,name in enumerate(names) if name.startswith(s+'_') and '_angle' in name and 'elbow' not in name]
            check('fourteen angle schema '+target,len(angles)==14)
            support[:,j]=a['mask'][:,wrist].all(1)&(a['mask'][:,angles].sum(1)>=10)
    check('support proxies preserved independently of posterior availability',np.array_equal(support,a['input_support']))
    starts=list(range(0,n-15,2))
    if starts[-1]!=n-16:starts.append(n-16)
    check('every stride2 16frame RGB window plus final window retained',np.array_equal(rgb['starts'],starts)
          and rgb['features'].shape==(len(starts),768))
    midpoint=(ts[np.asarray(starts)]+ts[np.asarray(starts)+15])/2
    check('RGB window centers are original PTS midpoint',np.array_equal(rgb['center_timestamp_us'],midpoint))
    interpolated=np.stack([np.interp(ts,midpoint,rgb['features'][:,j]) for j in range(768)],1).astype(np.float32)
    check('saved per-frame RGB equals declared frozen interpolation',np.array_equal(interpolated,a['rgb']))
    check('face missing stays separate and not a classifier input',face['xyz_normalized'].shape==(n,478,3)
          and np.array_equal(face['found'],np.isfinite(face['xyz_normalized']).all((1,2)))
          and complete['face_used_by_attribute_head'] is False)
    reference_rows={r['clip_id']:r for r in bank['rows'] if r['word_id']==result['word']}
    check('nine exact FIT references preserved',len(result['references'])==9
          and {r['id'] for r in result['references']}==set(reference_rows)
          and {r['person'] for r in result['references']}==set(range(1,10)))
    paths={}
    for comparison in result['references']:
        row=reference_rows[comparison['id']]; r=npz(row['input']); p=np.asarray(comparison['path'],int)
        d=np.diff(p,axis=0); paths[comparison['id']]={tuple(v) for v in p}
        check('full returned path '+comparison['id'],np.array_equal(p[0],[0,0])
              and np.array_equal(p[-1],[n-1,len(r['frame_id'])-1])
              and set(p[:,0])==set(range(n)) and set(p[:,1])==set(range(len(r['frame_id'])))
              and all(tuple(v) in [(1,0),(0,1),(1,1)] for v in d))
        check('returned playback PTS '+comparison['id'],np.array_equal(comparison['query_frame_times'],ts/1e6)
              and np.array_equal(comparison['reference_frame_times'],r['timestamp_us']/1e6))
    refs={r['id']:r for r in result['references']};peak_count=0
    for stage in result['stages']:
        for feature in stage['features']:
            for rid,peak in zip(feature['reference_ids'],feature['peak_at']):
                if peak is None:continue
                peak_count+=1; qi,ri=peak['query_frame'],peak['reference_frame']
                check('saved integrated peak link '+str(peak_count),(qi,ri) in paths[rid]
                      and peak['query_seconds']==refs[rid]['query_frame_times'][qi]
                      and peak['reference_seconds']==refs[rid]['reference_frame_times'][ri])
    check('chosen whole-reference IDs are two returned sequences',len(result['selected_whole_references'])==2
          and set(result['selected_whole_references'])<=set(refs))
    cold=read(U/'raw_public_probe/adapter/COMPLETE.json');record(U/'raw_public_probe/adapter/COMPLETE.json')
    cold_row=next(r for r in public['rows'] if r['source_sha256']==cold['source_sha256'])
    check('cold and warm use different public videos and frame counts',cold_row['person']==11 and cold['frames']==141
          and cold['source_sha256']!=complete['source_sha256'] and cold_row['clip_id']!=public_row['clip_id'])
    stderr_path=U/'server_stderr.txt';stderr=stderr_path.read_text(encoding='utf8',errors='replace')
    count=stderr.count('ConnectionAbortedError:'); tracebacks=stderr.count('Traceback (most recent call last):')
    media=stderr.count('self.file(media)');submission=stderr.count("self.file(source, job['_mime'])")
    keepalive=stderr.count('self.raw_requestline = self.rfile.readline')
    first_model=stderr.find('Created TensorFlow Lite');earlier=stderr[:first_model].count('ConnectionAbortedError:') if first_model>=0 else None
    check('stderr exceptions classified as media/HTTP disconnects',count==tracebacks==media+submission+keepalive
          and 'in analyze_submission' not in stderr and 'in _capture' not in stderr and 'in _heads' not in stderr)
    failures=[c['name'] for c in checks if not c['passed']]
    out=dict(status='PASS_SAVED_RAW_INTEGRATION_WITH_LIMITS' if not failures else 'FAIL',
        checks=len(checks),failures=failures,check_results=checks,source_pins=pins,
        reviewed_utc=datetime.now(timezone.utc).isoformat(),job_id=ui['job_id'],job_directory=str(job.resolve()),
        public_clip=public_row['clip_id'],source_sha256=expected,frames=n,duration_last_minus_first_seconds=float((ts[-1]-ts[0])/1e6),
        native_old3_known_posterior_cells=int(availability[:,:3].sum()),new4_known_posterior_cells=int(availability[:,3:].sum()),
        unavailable_new4_boundary_cells=int((~availability[:,3:]).sum()),face_frames_with_result=int(face['found'].sum()),
        RGB_windows=len(starts),all_integrated_peak_links_checked=peak_count,device=device,runtime_reuse=reuse,
        timings=dict(startup_seconds=startup['total_startup_seconds'],warmup_wrapper_seconds=startup['warmup_seconds'],
            warmup_internal_seconds=warm['total_seconds'],warmup_wrapper_includes_import_overhead=True,
            adapter_seconds=complete['latency']['total_adapter_seconds'],engine_analysis_seconds=result['latency']['total_analysis_seconds'],
            server_processing_seconds=timing['analysis_seconds'],upload_seconds=timing['upload_seconds'],queue_seconds=timing['queue_seconds'],
            browser_render_and_poll_delay_not_in_engine_or_server_times=True),
        cold_probe=dict(public_clip=cold_row['clip_id'],frames=cold['frames'],adapter_seconds=cold['latency']['total_adapter_seconds'],
                        scope='Adapter only, different video and older code before explicit limits/cache; not a matched end-to-end timing baseline'),
        speed_improvement_percentage=None,
        stderr_snapshot=dict(path=str(stderr_path.resolve()),sha256=sha(stderr_path),bytes=stderr_path.stat().st_size,
            connection_aborted=count,media_GET_writes=media,submission_GET_writes=submission,keepalive_request_reads=keepalive,
            errors_logged_before_first_mediapipe_initialization=earlier,individual_exception_timestamps_available=False,
            all_after_job_complete_claim_supported=False,
            interpretation='HTTP/media disconnect stacks are consistent with browser resource switches/closure; first errors precede MediaPipe initialization in log order. Exact per-error timing/causality is not recorded. Completed analysis artifacts remain valid.'),
        limits=['Saved-artifact verification by the adapter/server author; no independent KSL correctness annotation or new browser replay.',
                'UI nine checks are a saved integration report, not rerun here; stored paths, all integrated peaks and source/time identities were independently checked.',
                'GPU route is recorded runtime provenance; GPU hardware counters were not independently sampled.',
                'Face observation absence is preserved; no facial-expression/3D/whole-sign accuracy claim.',
                'Different videos and adapter-versus-engine timing scopes forbid a direct cold-to-warm speedup percentage.'],
        producer_module_imports=0,new_model_inference=0,video_decode=0,server_restarts=0,camera_access=0)
    with destination.open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=out['status'],checks=len(checks),failures=failures,report=str(destination),sha256=sha(destination)),ensure_ascii=False))
    if failures:raise AssertionError(failures)

if __name__=='__main__':run()
