from common import *
import traceback
def main():
    import numpy as np, openvino as ov, av, cv2, statistics
    verified=read(ROOT/'VERIFICATION.json')
    candidates=[r for r in verified['conditions'] if r['condition'].startswith('ov_gpu') and r['numerical_pass']]
    if not candidates:raise RuntimeError('No numerical-qualified GPU condition')
    chosen=min(candidates,key=lambda r:r['inference_median_seconds'])['condition']
    bs=4 if chosen.endswith('b4') else 1; prec='f16' if 'fp16' in chosen else 'f32'
    core=ov.Core(); ir=core.read_model(ROOT/'converted/signrep_features_fp32.xml');ir.reshape([bs,3,16,224,224])
    t=time.perf_counter(); comp=core.compile_model(ir,'GPU',{'INFERENCE_PRECISION_HINT':prec,'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1'});compile_sec=time.perf_counter()-t
    assert all('GPU' in x for x in comp.get_property('EXECUTION_DEVICES'))
    x,_=batch(norm(np.load(ROOT/'inputs/V01/rgb224.npy',allow_pickle=False)),[0]*bs,bs)
    for _ in range(2):comp([x])
    out=ROOT/'e2e';out.mkdir(exist_ok=False);results=[]
    for rep in range(2):
        for row in read(ROOT/'INPUTS.json')['rows']:
            t=time.perf_counter(); images=[]
            with av.open(row['source']) as c:
                for frame in c.decode(video=0):images.append(cv2.resize(frame.to_ndarray(format='rgb24'),(224,224),interpolation=cv2.INTER_LINEAR))
            frames=norm(np.stack(images)); seq=starts(len(frames)); ys=[]
            for i in range(0,len(seq),bs):
                x,n=batch(frames,seq[i:i+bs],bs);ys.append(np.array(comp([x])[0],copy=True)[:n])
            y=np.concatenate(ys);seconds=time.perf_counter()-t
            assert seq==row['window_starts'] and len(frames)==row['frames']
            reference=np.load(ROOT/'features'/f"torch_cpu_fp32_b1_{row['alias']}_r0.npy",allow_pickle=False)
            d=diff(reference,y); assert d['finite'] and d['max_relative_l2']<=.01 and d['min_cosine']>=.999,d
            np.save(out/f"{row['alias']}_r{rep}.npy",y,allow_pickle=False)
            rec=dict(alias=row['alias'],repeat=rep,seconds=seconds,frames=len(frames),windows=len(seq),parity=d)
            results.append(rec);event('e2e_timing',**rec)
    save('END_TO_END.json',dict(status='PASS',condition=chosen,compile_seconds=compile_sec,execution_devices=list(comp.get_property('EXECUTION_DEVICES')),rows=results,median_seconds=statistics.median(x['seconds'] for x in results),range_seconds=[min(x['seconds'] for x in results),max(x['seconds'] for x in results)],scope='public original file to features; model loaded/warm; no feedback head, DTW, UI or camera',protocol_sha256=sha(ROOT/'END_TO_END_PROTOCOL.json')))
if __name__=='__main__':
    try:main()
    except Exception:
        p=ROOT/'E2E_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
