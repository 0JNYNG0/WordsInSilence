from pathlib import Path
import time,json,hashlib,traceback,sys
import numpy as np
import av
from stream_runtime import Engine,StreamSession,reuse_observations,U
from freeze import sha

def write_new(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def event(**kw):
    kw['at']=time.time();s=json.dumps(kw,ensure_ascii=False)
    with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(s+'\n')
    print(s,flush=True)
def main():
    if (U/'COMPLETE.json').exists():raise RuntimeError('Completed replay must not rerun')
    plan=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))
    for p,h in plan['protected'].items():assert sha(p)==h,p
    e=Engine();event(kind='engine_ready',compile_s=e.compile_seconds,warm_s=e.warm_seconds,devices=e.devices)
    summaries=[]
    for row in plan['rows']:
        assert sha(row['source'])==row['source_sha256'];assert sha(row['saved_baseline'])==row['saved_baseline_sha256']
        old=np.load(row['saved_baseline'],allow_pickle=False);frames=[];ts=[]
        with av.open(row['source']) as c:
            st=c.streams.video[0];start=None
            for f in c.decode(st):
                if start is None:start=f.pts
                frames.append(f.to_ndarray(format='bgr24'));ts.append(round(float((f.pts-start)*st.time_base)*1e6))
        assert np.array_equal(ts,old['timestamp_us'])
        for i,im in enumerate(frames):assert hashlib.sha256(im.tobytes()).hexdigest()==str(old['frame_sha256'][i])
        event(kind='prepared_source',clip=row['clip_id'],frames=len(frames),predecode_outside_stream_timer=True)
        for rep in range(plan['repetitions']):
            out=U/'runs'/row['clip_id']/str(rep)
            if (out/'RESULT.json').exists():
                summaries.append(json.loads((out/'RESULT.json').read_text(encoding='utf-8')));continue
            if out.exists():raise RuntimeError('Inspect partial run: '+str(out))
            h,w=frames[0].shape[:2];s=StreamSession(e,w,h,plan['queue_capacity']);t0=time.perf_counter()
            for i,(t,im) in enumerate(zip(ts,frames)):
                due=t0+t/1e6;wait=due-time.perf_counter()
                if wait>0:time.sleep(wait)
                s.submit(i,int(t),im)
            last=time.perf_counter();session=s.finish(out)
            before=e.calls;features=reuse_observations(out);assert before==e.calls
            np.savez_compressed(out/'analysis_input.npz',**{k:v for k,v in features.items() if isinstance(v,np.ndarray)})
            final=time.perf_counter();new=np.load(out/'observations.npz',allow_pickle=False)
            ids=new['frame_id'];a=new['xy_px'];b=old['xy_px'][ids]
            valid=np.isfinite(a).all(2)&np.isfinite(b).all(2);delta=np.linalg.norm(a[valid]-b[valid],axis=1)
            times=session['records'];lag=np.array([(r['completed_s']-r['received_s'])*1000 for r in times])
            # Source pacing slip is separate from inference lag.
            slip=np.array([(r['received_s']-(t0-s.t0)-r['timestamp_us']/1e6)*1000 for r in session['receipts']])
            outrow=dict(clip_id=row['clip_id'],rep=rep,source_frames=len(frames),completed=session['completed'],dropped=session['dropped'],
                input_complete=session['status']=='COMPLETE',finite_pattern_preserved=bool(np.array_equal(np.isfinite(a),np.isfinite(b))),
                pixel_hashes_preserved=bool(np.array_equal(new['frame_sha256'],old['frame_sha256'][ids])),
                time_axis_preserved=bool(np.array_equal(new['timestamp_us'],np.array(ts)[ids])),
                max_coordinate_delta_px=float(delta.max()) if len(delta) else None,
                p95_coordinate_delta_px=float(np.percentile(delta,95)) if len(delta) else None,
                coordinate_preserved=bool(np.allclose(a,b,atol=1e-4,rtol=1e-5,equal_nan=True)),
                lag_p50_ms=float(np.median(lag)),lag_p95_ms=float(np.percentile(lag,95)),lag_max_ms=float(lag.max()),
                compute_ms_median=float(np.median([r['compute_ms'] for r in times])),
                source_pacing_slip_max_ms=float(slip.max()),max_queue=session['max_queue'],
                post_capture_to_reused_input_ms=(final-last)*1000,detector_calls=session['detector_calls'],reuse_detector_calls=e.calls-before,
                represented_duration_seconds=ts[-1]/1e6,stream_wall_seconds=final-t0,
                visibility_and_feedback_accuracy='not assessed; this is input scheduling/reuse validation')
            outrow['engineering_targets_pass']=bool(outrow['input_complete'] and outrow['lag_p95_ms']<=plan['lag_p95_budget_ms'] and outrow['post_capture_to_reused_input_ms']<=plan['post_capture_budget_ms'])
            new.close()
            write_new(out/'RESULT.json',outrow);summaries.append(outrow);event(kind='run_complete',**outrow)
        old.close()
    for p,h in plan['protected'].items():assert sha(p)==h,p
    write_new(U/'COMPLETE.json',dict(runs=summaries,training=0,camera_used=False,protected_unchanged=True,
        compile_seconds=e.compile_seconds,warm_seconds=e.warm_seconds,actual_devices=e.devices,
        boundary='predecoded public frames, no browser/network/codec/game rendering; no end-to-end feedback accuracy or latency claim'))
if __name__=='__main__':
    try:main()
    except BaseException:
        (U/('FIRST_FAILURE_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
