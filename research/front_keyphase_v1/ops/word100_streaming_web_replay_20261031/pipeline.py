"""Isolated public-file replay -> frozen physical head -> three-reference DTW.

No training, new accuracy scoring, production route, or personal capture.
"""
from pathlib import Path
import sys, json, time, hashlib, threading, traceback, copy
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
OPS = U.parent
OLD = OPS/'word100_stage_feedback_cycle_20261031'
STREAM = OPS/'word100_feature_support_streaming_20261031/streaming'
sys.path.insert(0, str(STREAM))
import stream_capture as sc
sys.path.insert(0, str(OLD))
import feedback as fb
import numpy as np
import torch
sys.path.insert(0, str(OPS/'word100_signrep_temporal_head_20261029'))
from prepare_inputs import extract_features

ALIASES = ['P005','P043','P079']
MODES = ['after_end','during_recording']
CONDITION = 'pose_capacity_matched'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, v): sc.write_new(p, v)
def sha(p): return sc.sha256_file(p)

def freeze():
    info=read(OLD/'LEARNING_INPUTS.json')
    selected=[r for r in info['rows'] if r['alias'] in ALIASES]
    words={r['word_id'] for r in selected}
    refs=[r for r in info['rows'] if r['role']=='fit' and r['word_id'] in words]
    assert len(selected)==3 and len(refs)==9
    files=[Path(__file__), U/'server.py', U/'index.html', U/'browser_test.py', OLD/'learning.py', OLD/'feedback.py', OLD/'LEARNING_INPUTS.json', OLD/'NORMALIZATION.json', OLD/'runs/pose_capacity_matched_42/head.pt', OLD/'runs/pose_capacity_matched_42/probabilities.npy', STREAM/'stream_capture.py', sc.LAB/'config/contract.json', OPS/'word100_signrep_temporal_head_20261029/prepare_inputs.py', U/'AUTHORIZATION.json']
    for r in selected+refs:
        assert sha(r['source'])==r['source_sha256']
        files.extend([Path(r['source']),Path(r['input']),Path(r['observation'])])
    # Carry detector/assignment pins and protected operational assets forward.
    prior=read(STREAM/'FROZEN.json')
    pins=dict(prior['pins']); pins.update({str(p):sha(p) for p in files})
    spec=dict(status='FROZEN_BEFORE_WEB_RUNS', aliases=ALIASES, repetitions=[1,2], modes=MODES,
      route='CPU MediaPipe full frame + frozen joint assignment + fixed ROI + frozen pose head seed42 + three separate fit references + DTW + HTTP + browser paint acknowledgement',
      selection='No candidate selection or quality tuning; old pose head is a fixed research observer, not adopted educational model',
      rows=selected,references=refs,pins=pins,protected=prior['protected'],
      timing='Warm app/reference cache; new detector pair initialized before each recording. End=last source frame available at original PTS. Server receipt of browser acknowledgement after two requestAnimationFrame callbacks is a conservative rendering acknowledgement bound, not physical display latency.',
      comparison='Both modes use identical single decode and full frames. Batch begins after simulated recording ends; stream processes during original PTS arrival. Same head/DTW/evidence gates.',
      scope='3 consumed public timing clips, 2 repeats. No accuracy re-score, camera, SignRep GPU branch, C/T3F recognition, uploads or training. Fixed ROI and known target word supplied. No semantic stage discovery or educational correctness.',
      pass_conditions=['all 12 jobs complete or preserve first failure', 'all frames/PTS retained', 'paired observation arrays, head outputs and structured DTW result identical', 'reference denominator retained with missing query controls', 'current result and insufficient-evidence disclaimer painted before acknowledgement', 'protected hashes unchanged'],
      order='repeat -> alias -> after_end then during_recording; fixed-order cache limitation reported')
    write(U/'FROZEN.json',spec)
    return spec

class Engine:
    def __init__(self):
        start=time.perf_counter()
        self.frozen=read(U/'FROZEN.json'); self.check_pins()
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        self.info=read(OLD/'LEARNING_INPUTS.json'); self.stats=read(OLD/'NORMALIZATION.json')
        self.rows={r['alias']:r for r in self.frozen['rows']}
        self.head=fb.Head(len(self.info['targets']))
        self.head.load_state_dict(torch.load(OLD/'runs/pose_capacity_matched_42/head.pt',weights_only=True)); self.head.eval()
        self.projection=torch.from_numpy(fb.projection())
        self.refs={}; self.refrows={}
        stored=np.load(OLD/'runs/pose_capacity_matched_42/probabilities.npy',mmap_mode='r')
        for r in self.frozen['references']:
            i=next(i for i,x in enumerate(self.info['rows']) if x['alias']==r['alias'])
            a=dict(np.load(r['input'],allow_pickle=False))
            self.refs[r['alias']]=fb.prepare_case(r,a,stored[i,:len(a['x'])])
            self.refrows[r['alias']]=r
        self.contract=read(sc.LAB/'config/contract.json')
        self.capture,self.assignment,_=sc.dependencies()
        self.jobs={}; self.lock=threading.Lock(); self.active=None
        self.setup_seconds=time.perf_counter()-start

    def check_pins(self):
        for group in ['pins','protected']:
            for p,h in self.frozen[group].items(): assert sha(p)==h,p

    def start(self,alias,mode,repeat):
        assert alias in ALIASES and mode in MODES and repeat in [1,2]
        key=f'{alias}_{repeat}_{mode}'
        with self.lock:
            if self.active is not None: raise ValueError('A replay is already running')
            if key in self.jobs or (U/'runs'/key).exists(): raise ValueError('Completed/partial run is immutable; no silent retry')
            job=dict(key=key,alias=alias,mode=mode,repeat=repeat,status='preparing',processed=0,expected=self.rows[alias]['frames'])
            self.jobs[key]=job; self.active=key
        threading.Thread(target=self.run,args=(job,),daemon=True).start()
        return key

    def public(self,key):
        with self.lock:
            return copy.deepcopy(self.jobs[key])

    def update(self,job,**kw):
        with self.lock: job.update(kw)

    def run(self,job):
        d=U/'runs'/job['key']; d.mkdir(parents=True,exist_ok=False)
        row=self.rows[job['alias']]; lm=None
        try:
            setup=time.perf_counter_ns(); lm=self.capture.RawCaptureLandmarker(self.contract)
            processor=sc.FrameProcessor(lm,self.assignment.HandAssigner(),self.contract)
            detector_setup=(time.perf_counter_ns()-setup)/1e9
            original_process=processor.process_frame
            def process(*args):
                original_process(*args)
                self.update(job,processed=len(processor.timing))
            processor.process_frame=process
            with np.load(row['input'],allow_pickle=False) as old:
                duration=float(old['timestamp_us'][-1]/1e6)
            epoch=time.perf_counter_ns(); self.update(job,status='recording',source_duration=duration)
            if job['mode']=='after_end':
                time.sleep(duration)
                capture_end=time.perf_counter_ns(); self.update(job,status='processing')
                feed=sc.feed_single_decode(row['source'],processor,'single_decode_batch',time.perf_counter_ns())
            else:
                feed=sc.feed_single_decode(row['source'],processor,'single_decode_paced',epoch)
                capture_end=feed['last_frame_available_ns']
            self.update(job,status='comparing')
            baseline,obs,raw,sides=sc.make_observations(processor,row,feed['metadata'],processor.extractor_id())
            assert obs.n_frames==row['frames']
            baseline.save(str(d/'baseline.npz')); obs.save(str(d/'candidate.npz'))
            np.savez_compressed(d/'raw.npz',**raw); np.savez_compressed(d/'assignment.npz',sides=sides)
            lm.close(); lm=None
            saved=time.perf_counter_ns()
            # Frozen full-frame detection first, ROI only when building comparison input.
            t=time.perf_counter_ns(); ob=fb.roi_observation(d/'candidate.npz',row['box'])
            raw_ob={k:getattr(obs,k).copy() for k in sc.ARRAY_NAMES}
            xy=raw_ob['xy_px']; b=row['box']; raw_ob['usable']&=(xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3])
            feat=extract_features(raw_ob)
            arr=dict(x=np.nan_to_num(feat['x']).astype('float32'),mask=feat['mask'],frame_id=obs.frame_id,timestamp_us=obs.timestamp_us)
            px=np.where(arr['mask'],(arr['x']-self.stats['pose_mean'])/np.array(self.stats['pose_std']),0)
            x=torch.tensor(np.concatenate([px,arr['mask']],1)[None],dtype=torch.float32); v=torch.ones((1,len(px)))
            feature_end=time.perf_counter_ns()
            with torch.no_grad(): prob=self.head(x,x@self.projection,v).sigmoid().numpy()[0]
            head_end=time.perf_counter_ns(); np.save(d/'head_probabilities.npy',prob,allow_pickle=False)
            q=fb.prepare_case(dict(row,observation=str(d/'candidate.npz')),arr,prob,ob)
            refrows=[r for r in self.frozen['references'] if r['word_id']==row['word_id']]
            pairs=[dict(reference=r['alias'],**fb.pair(q,self.refs[r['alias']],CONDITION)) for r in refrows]
            for p in pairs:
                p['query_seconds']=(q['seq']['timestamp_us']/1e6).tolist()
                p['reference_seconds']=(self.refs[p['reference']]['seq']['timestamp_us']/1e6).tolist()
            result=fb.aggregate(row['alias'],pairs,CONDITION)
            compare_end=time.perf_counter_ns()
            payload=dict(result=result, label=row['label'], targets=self.info['targets'],
                scope='일부 손 모양·이동 방향·겹침의 연구 비교. 전체 수어의 정답/오답 판정이 아닙니다.',
                caution='관측이 부족한 부분은 비교 근거 부족입니다. 차이 후보도 틀린 수어라는 뜻은 아닙니다. 이번에는 계산 동등성을 위해 기존 관측 기준을 고정했습니다. 좌표 존재가 실제 손 모양의 정확성을 보증하지 않습니다.',
                source_frames=row['frames'],frame_id=obs.frame_id.tolist(),seconds=(obs.timestamp_us/1e6).tolist(),
                # 2D overlay is for inspection only; never fabricated across a missing joint.
                skeleton=np.where(np.isfinite(obs.xy_px)&obs.usable[:,:,None],obs.xy_px,np.nan).tolist(),
                width=feed['metadata']['width'],height=feed['metadata']['height'],
                references=[dict(alias=r['alias'],label=r['label']) for r in refrows])
            # JSON null conveys unobserved points; no zero-valued pseudo-coordinate.
            payload['skeleton']=[[[None if not np.isfinite(c) else c for c in p] for p in f] for f in payload['skeleton']]
            write(d/'feedback.json',payload)
            ready=time.perf_counter_ns()
            trace=dict(epoch_ns=epoch,capture_end_ns=capture_end,saved_ns=saved,feature_end_ns=feature_end,head_end_ns=head_end,compare_end_ns=compare_end,ready_ns=ready,rows=processor.timing,arrivals=feed['arrivals'])
            write(d/'TRACE.json',trace)
            metrics=dict(key=job['key'],alias=row['alias'],mode=job['mode'],repeat=job['repeat'],frames=obs.n_frames,
              detector_setup_seconds=detector_setup,capture_end_ns=capture_end,ready_ns=ready,
              backend_after_end_seconds=(ready-capture_end)/1e9, capture_tail_seconds=(saved-capture_end)/1e9,
              feature_seconds=(feature_end-t)/1e9,head_seconds=(head_end-feature_end)/1e9,dtw_three_refs_seconds=(compare_end-head_end)/1e9,
              maximum_queued_frames=feed['maximum_queued_frames'],remaining_at_end=sum(r['assignment_end_ns']>capture_end for r in processor.timing),
              queue_wait_max_seconds=max(r['queue_wait_seconds'] for r in processor.timing),
              arrival_lateness_max_seconds=max(max(0,(a['available_ns']-a['due_ns'])/1e9) for a in feed['arrivals']) if job['mode']=='during_recording' else None,
              source_sha256=sha(row['source']), output_sha256={p.name:sha(p) for p in d.iterdir() if p.is_file()})
            write(d/'BACKEND_COMPLETE.json',metrics)
            self.update(job,status='ready',payload=payload,metrics=metrics)
        except BaseException as e:
            failure=dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
            write(d/'FIRST_FAILURE.json',failure); self.update(job,status='failed',error=str(e))
            with self.lock:self.active=None
        finally:
            if lm is not None:lm.close()

    def acknowledge(self,key,body):
        received=time.perf_counter_ns()
        with self.lock:
            job=self.jobs[key]
            assert job['status']=='ready' and self.active==key
            assert body['rendered_key']==key and body['visible_scope'] and body['visible_caution'] and body['paint_callbacks']==2
            assert body['reference_rows']==3
            metrics=dict(job['metrics'],render_ack_received_ns=received,
              end_to_render_ack_seconds=(received-job['metrics']['capture_end_ns'])/1e9,
              backend_ready_to_render_ack_seconds=(received-job['metrics']['ready_ns'])/1e9,
              browser=body)
            write(U/'runs'/key/'COMPLETE.json',metrics)
            job['status']='complete';job['metrics']=metrics;self.active=None
        return metrics

if __name__=='__main__':
    if sys.argv[1:] == ['freeze']: freeze();print('FROZEN')
