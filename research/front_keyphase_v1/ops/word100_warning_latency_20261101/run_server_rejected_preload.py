"""Isolated8803 candidate demo. Import does not start a server or load models.

Preserved old server/engine/guard files are read-only. Only adapter routing and
the server's asset/session ROOT are redirected, in this dedicated process.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,importlib,importlib.util,json,sys,time
import numpy as np

U=Path(__file__).resolve().parent
OLD=U.parent/'word100_multipeople_live_feedback_20261101'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_new(p,value):
    text=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    with Path(p).open('x',encoding='utf8') as f:f.write(text)
    return len(text.encode('utf8'))

def preserve_original(original,augmented,pointer=''):
    """Every pre-existing value remains exact; only new dictionary keys allowed."""
    if isinstance(original,dict):
        if not isinstance(augmented,dict):raise AssertionError(pointer)
        for key,value in original.items():
            if key not in augmented:raise AssertionError(pointer+'/'+key)
            preserve_original(value,augmented[key],pointer+'/'+key)
    elif isinstance(original,list):
        if not isinstance(augmented,list) or len(original)!=len(augmented):raise AssertionError(pointer)
        for i,(a,b) in enumerate(zip(original,augmented)):preserve_original(a,b,pointer+'/'+str(i))
    elif original!=augmented:raise AssertionError('Original evidence changed: '+pointer)

def compact_details(enriched):
    """Drop only NEW verbose traces; full witness sets stay in local full file."""
    out=deepcopy(enriched)
    for ci,ref in enumerate(out.get('references',[])):
        for si,stage in enumerate(ref.get('stages',[])):
            for family,full in stage.get('detail_enrichment',{}).get('families',{}).items():
                keep={key:value for key,value in full.items() if key not in ['trace','exceedances']}
                keep['full_detail_json_pointer']=f'/references/{ci}/stages/{si}/detail_enrichment/families/{family}'
                keep['full_trace_saved_locally']=True
                criteria={}
                for key,value in full['exceedances'].items():
                    reduced={k:v for k,v in value.items() if k!='episodes'};episodes=[]
                    for episode in value['episodes']:
                        ep={k:deepcopy(episode[k]) for k in ['start','end','sample_count','source_pts_span_us','singleton','dimensions','max_abs_residual']}
                        witnesses=episode['max_witnesses']
                        ep['representative_max_witness']=deepcopy(witnesses[0]) if witnesses else None
                        ep['max_witness_count']=len(witnesses)
                        ep['representative_only_all_witnesses_saved_locally']=True
                        episodes.append(ep)
                    reduced['episodes']=episodes;criteria[key]=reduced
                keep['exceedances']=criteria
                stage['detail_enrichment']['families'][family]=keep
    return out

class ResearchEngine:
    def __init__(self,base,live,detail):self.base=base;self.live=live;self.detail=detail
    def catalog(self):
        value=deepcopy(self.live.catalog())
        value['research_demo']=dict(candidate_adapter=True,extra_evidence_only=True,educational_accuracy_verified=False)
        return value
    def analyze_submission(self,path,jobdir,progress=None):
        begin=time.perf_counter();jobdir=Path(jobdir)
        def relay(*args,**kwargs):
            if len(args)>1 and args[1]==100:return
            if progress:progress(*args,**kwargs)
        guarded=self.live.analyze_submission(path,jobdir,relay)
        if progress:progress('차이의 근거 정리',98,'관측 종류와 원래 영상 시점을 연결합니다.')
        prepared=time.perf_counter();a={};rows=[];bank_arrays={};ranges={}
        if guarded.get('references'):
            directory=jobdir/'adapter'
            with np.load(directory/'analysis.npz',allow_pickle=False) as z:
                a={k:z[k].copy() for k in ['x','mask','probabilities','frame_id','timestamp_us']}
            a['feature_names']=read(directory/'COMPLETE.json')['feature_names']
            bank,bank_arrays=self.base.bank()
            if bank['observer']!='protected_baseline':raise ValueError('Protected baseline observer required')
            rows=[r for r in bank['rows'] if r['word_id']==guarded['word']]
            ranges=read(self.base.U/'NORMAL_RANGES.json')['ranges']
        prep_seconds=time.perf_counter()-prepared
        t=time.perf_counter();full=self.detail.enrich(a,rows,bank_arrays,guarded,ranges)
        enrich_seconds=time.perf_counter()-t
        preserve_original(guarded,full)
        t=time.perf_counter();full_bytes=save_new(jobdir/'DETAIL_FEEDBACK.json',full)
        full_save_seconds=time.perf_counter()-t
        t=time.perf_counter();response=compact_details(full);preserve_original(guarded,response)
        compact_seconds=time.perf_counter()-t
        timings=dict(input_read_seconds=prep_seconds,enrichment_seconds=enrich_seconds,full_detail_serialize_save_seconds=full_save_seconds,
            compact_projection_seconds=compact_seconds,full_detail_bytes=full_bytes,
            total_through_details_seconds=time.perf_counter()-begin,
            original_latency_fields_unchanged=True,server_request_timing_includes_final_response_serialization=True)
        response['research_pipeline']=dict(status='ISOLATED_CANDIDATE_DEMO',detail_timings=timings,
            full_detail_saved_locally=True,original_values_preserved=True,classification_changed=False,
            warning='Extra detail explains existing comparisons; no educational correctness or3D rotation verdict.')
        save_new(jobdir/'DETAIL_TIMING.json',timings)
        t=time.perf_counter();response_bytes=save_new(jobdir/'RESEARCH_FEEDBACK.json',response)
        save_new(jobdir/'RESPONSE_SAVE_TIMING.json',dict(compact_response_serialize_save_seconds=time.perf_counter()-t,
            compact_response_bytes=response_bytes,total_through_response_save_seconds=time.perf_counter()-begin,
            HTTP_server_serialization_and_transfer_not_included=True))
        if progress:progress('분석 완료',100,'원래 비교 결과와 관측 근거의 종류·시점을 함께 표시합니다.')
        return response

def load_file(name,path):
    existing=sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing,'__file__','')).resolve()!=Path(path).resolve():raise RuntimeError('Module routing collision: '+name)
        return existing
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:sys.modules.pop(name,None);raise
    return module

def preload_capture_modules(candidate):
    """Import the exact capture dependencies; never construct a detector.

    VIDEO trackers, face engines, observations and assignment state are still
    created afresh inside each request's unchanged candidate._capture.
    """
    begin=time.perf_counter();src=candidate.LAB/'src'
    if str(src) not in sys.path:sys.path.insert(0,str(src))
    expected={
        'wis_lab.capture.face_landmarks':src/'wis_lab/capture/face_landmarks.py',
        'wis_lab.core.observation':src/'wis_lab/core/observation.py'}
    loaded={}
    for name,path in expected.items():
        module=importlib.import_module(name)
        if Path(module.__file__).resolve()!=path.resolve():raise ValueError('Capture import route differs: '+name)
        loaded[name]=dict(path=str(path),sha256=sha(path))
    for name,path in [('_live_raw_capture',candidate.OPS/'word100_hand_assignment_20261016/capture_raw.py'),
                      ('_live_joint_assignment',candidate.OPS/'word100_hand_assignment_joint_20261016/assignment.py')]:
        module=candidate._module(name,path)
        if Path(module.__file__).resolve()!=path.resolve():raise ValueError('Capture import route differs: '+name)
        loaded[name]=dict(path=str(path),sha256=sha(path))
    return dict(status='CAPTURE_MODULES_IMPORTED_ONLY',seconds=time.perf_counter()-begin,modules=loaded,
        detector_instances_created=0,detector_process_calls=0,real_video_reads=0,camera_calls=0,
        request_VIDEO_state_shared=False,meaning='Import preparation only; each request still constructs its own detector and assignment state.')

def startup_warmup(candidate):
    """Preserve frozen synthetic observer warmup, then import capture modules."""
    begin=time.perf_counter();observer=candidate.warmup()
    preload=preload_capture_modules(candidate)
    return dict(observer,capture_import_preload=preload,
        startup_warmup_wrapper_seconds=time.perf_counter()-begin,
        observer_total_seconds_scope='Original candidate.warmup total_seconds; capture import cost is separate.',
        historical_gap_note='The earlier capture-envelope minus nested timers (~2.7s) was an unattributed first-request gap, not an exact import measurement.')

def components():
    # Protect all historical modules from import-time bytecode writes too.
    sys.dont_write_bytecode=True
    candidate=load_file('adapter',U/'candidate_adapter.py')
    engine=load_file('engine',OLD/'engine.py')
    load_file('evidence_guard',OLD/'evidence_guard.py')
    live=load_file('_candidate_preserved_live_engine',OLD/'live_engine.py')
    detail=load_file('_candidate_detail_enrichment',U/'detail_enrichment.py')
    server=load_file('_candidate_preserved_server',OLD/'server.py')
    server.ROOT=U
    return server,ResearchEngine(engine,live,detail),candidate

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8803)
    args=parser.parse_args()
    if args.port==8802:raise ValueError('Preserve existing8802 server; use a separate research port')
    server_module,engine,candidate=components()
    print('Preparing candidate observers; no requests accepted before warmup completes.',flush=True)
    server=server_module.create_server(engine,args.port,storage_root=U/'sessions',warmup=lambda:startup_warmup(candidate))
    startup=dict(adapter=str(U/'candidate_adapter.py'),adapter_sha256=sha(U/'candidate_adapter.py'),
        old_engine_sha256=sha(OLD/'engine.py'),old_guard_sha256=sha(OLD/'evidence_guard.py'),old_server_sha256=sha(OLD/'server.py'),
        detail_sha256=sha(U/'detail_enrichment.py'),wrapper_sha256=sha(__file__),
        UI_sha256={n:sha(U/n) for n in ['index.html','app.js']},port=args.port,
        model_or_bank_replacement=False,output_root=str(U/'sessions'),
        capture_import_preload_enabled=True,detector_instances_remain_request_local=True)
    save_new(server.session_dir/'RESEARCH_ROUTING.json',startup)
    print(f'Isolated research demo: http://127.0.0.1:{server.server_port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
