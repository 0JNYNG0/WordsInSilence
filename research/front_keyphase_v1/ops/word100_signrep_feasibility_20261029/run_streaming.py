from common import *
import traceback
def main():
    import numpy as np, openvino as ov, av, cv2, statistics, queue, threading
    condition='ov_gpu_fp16_b1'
    ver=read(ROOT/'VERIFICATION.json')
    assert next(r for r in ver['conditions'] if r['condition']==condition)['numerical_pass']
    core=ov.Core();ir=core.read_model(ROOT/'converted/signrep_features_fp32.xml');ir.reshape([1,3,16,224,224])
    t=time.perf_counter();comp=core.compile_model(ir,'GPU',{'INFERENCE_PRECISION_HINT':'f16','PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1'});compile_sec=time.perf_counter()-t
    assert all('GPU' in x for x in comp.get_property('EXECUTION_DEVICES'))
    x,_=batch(norm(np.load(ROOT/'inputs/V01/rgb224.npy',allow_pickle=False)),[0],1)
    for _ in range(2):comp([x])
    results=[];out=ROOT/'streaming';out.mkdir(exist_ok=False)
    for rep in range(2):
        for row in read(ROOT/'INPUTS.json')['rows']:
            work=queue.Queue(maxsize=4); ys=[]; errors=[]; stopped=threading.Event()
            def consume():
                try:
                    while True:
                        item=work.get()
                        if item is None:break
                        idx,arr=item; ys.append((idx,np.array(comp([arr])[0],copy=True)))
                except BaseException:errors.append(traceback.format_exc())
                finally:stopped.set()
            def put(item):
                while True:
                    if stopped.is_set():raise RuntimeError(errors or 'Unexpected infer worker exit')
                    try:work.put(item,timeout=.5);return
                    except queue.Full:pass
            t=time.perf_counter();worker=threading.Thread(target=consume,daemon=True);worker.start()
            from collections import deque
            buf=deque(maxlen=16);seq=row['window_starts']; ends={s+15:s for s in seq};n=0;pts=[]
            with av.open(row['source']) as c:
                for i,f in enumerate(c.decode(video=0)):
                    im=cv2.resize(f.to_ndarray(format='rgb24'),(224,224),interpolation=cv2.INTER_LINEAR)
                    buf.append(norm(im[None])[0]);n+=1;pts.append(float(f.pts*f.time_base))
                    if i in ends:
                        arr=np.ascontiguousarray(np.stack(buf).transpose(1,0,2,3)[None]);put((ends[i],arr))
            put(None);worker.join(timeout=120)
            if worker.is_alive() or errors:raise RuntimeError(errors or 'Worker timeout')
            starts_out=[s for s,_ in ys];features=np.concatenate([y for _,y in ys]);seconds=time.perf_counter()-t
            assert starts_out==seq and n==row['frames'] and pts==row['pts']
            ref=np.load(ROOT/'features'/f"torch_cpu_fp32_b1_{row['alias']}_r0.npy",allow_pickle=False)
            prev=np.load(ROOT/'features'/f"{condition}_{row['alias']}_r0.npy",allow_pickle=False)
            d=diff(ref,features);identical=bool(np.array_equal(prev,features))
            assert d['finite'] and d['max_relative_l2']<=.01 and d['min_cosine']>=.999
            # Serial and overlapped paths must compute the same operation on same pixels.
            assert identical, diff(prev,features)
            np.save(out/f"{row['alias']}_r{rep}.npy",features,allow_pickle=False)
            rec=dict(alias=row['alias'],repeat=rep,seconds=seconds,frames=n,windows=len(ys),all_pts_same=True,bitwise_equal_to_serial_gpu=identical,parity=d)
            results.append(rec);event('streaming_timing',**rec)
    save('STREAMING.json',dict(status='PASS',condition=condition,compile_seconds=compile_sec,execution_devices=list(comp.get_property('EXECUTION_DEVICES')),rows=results,median_seconds=statistics.median(r['seconds'] for r in results),range_seconds=[min(r['seconds'] for r in results),max(r['seconds'] for r in results)],scope='original-file decode overlapped with single GPU worker to features; warmed; no feedback head/DTW/UI',protocol_sha256=sha(ROOT/'STREAMING_PROTOCOL.json')))
if __name__=='__main__':
    try:main()
    except Exception:
        p=ROOT/'STREAM_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
