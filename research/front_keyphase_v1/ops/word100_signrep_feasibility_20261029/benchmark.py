from common import *
import traceback
def main():
    import numpy as np, torch, openvino as ov, av, cv2, gc
    assert read(ROOT/'EXPORT.json')['status']=='PASS'
    rows=read(ROOT/'INPUTS.json')['rows']; core=ov.Core()
    assert 'GPU' in core.available_devices, 'No GPU: no CPU fallback'
    torch.set_num_threads(4)
    t=time.perf_counter(); model=feature_wrapper(load_model()); loading=time.perf_counter()-t
    conditions=[('torch_cpu_fp32_b1',1,None),('ov_gpu_fp32_b1',1,'f32'),('ov_gpu_fp16_b1',1,'f16'),('ov_gpu_fp16_b4',4,'f16')]
    runners={}; compilation=[]
    runners['torch_cpu_fp32_b1']=lambda x:model(torch.from_numpy(x)).numpy()
    frames_by_alias={r['alias']:norm(np.load(ROOT/'inputs'/r['alias']/'rgb224.npy',allow_pickle=False)) for r in rows}
    with torch.inference_mode():
        for name,bs,prec in conditions:
            if prec:
                event('compile_start',condition=name)
                ir=core.read_model(ROOT/'converted/signrep_features_fp32.xml'); ir.reshape([bs,3,16,224,224])
                t=time.perf_counter()
                comp=core.compile_model(ir,'GPU',{'INFERENCE_PRECISION_HINT':prec,'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1'})
                rec=dict(condition=name,seconds=time.perf_counter()-t,devices=list(comp.get_property('EXECUTION_DEVICES')),precision_hint=str(comp.get_property('INFERENCE_PRECISION_HINT')))
                assert all('GPU' in d for d in rec['devices']),rec
                runners[name]=lambda x,c=comp:np.array(c([x])[0],copy=True)
                compilation.append(rec); event('compile_done',**rec)
            x,_=batch(frames_by_alias['V01'],[0]*bs,bs)
            for warm in range(2): runners[name](x)
        save('COMPILES.json',dict(torch_load_seconds=loading,rows=compilation))
        output=ROOT/'features';output.mkdir(exist_ok=False)
        results=[]; parity=[]; preproc=[]
        for repeat in range(2):
            for name,bs,prec in (conditions if repeat==0 else conditions[::-1]):
                for row in rows:
                    alias=row['alias']; frames=frames_by_alias[alias]; seq=row['window_starts']; ys=[]
                    t=time.perf_counter()
                    for i in range(0,len(seq),bs):
                        x,n=batch(frames,seq[i:i+bs],bs)
                        ys.append(runners[name](x)[:n])
                    features=np.concatenate(ys); sec=time.perf_counter()-t
                    assert features.shape==(len(seq),768) and np.isfinite(features).all()
                    np.save(output/f'{name}_{alias}_r{repeat}.npy',features,allow_pickle=False)
                    rec=dict(condition=name,alias=alias,repeat=repeat,seconds=sec,frames=row['frames'],windows=len(seq),valid_outputs=len(features),padded_compute_only=(-len(seq))%bs)
                    results.append(rec);event('timing',**rec)
                    ref=np.load(output/f'torch_cpu_fp32_b1_{alias}_r0.npy',allow_pickle=False)
                    check=dict(condition=name,alias=alias,repeat=repeat,**diff(ref,features))
                    check['numerical_pass']=check['finite'] and check['max_relative_l2']<=.01 and check['min_cosine']>=.999
                    parity.append(check)
                    # Persist each completed timing without replay if a later condition fails.
                    with (ROOT/'TIMINGS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(**rec,parity=check))+'\n')
        for repeat in range(2):
            for row in rows:
                t=time.perf_counter(); imgs=[]
                with av.open(row['source']) as c:
                    for f in c.decode(video=0):imgs.append(cv2.resize(f.to_ndarray(format='rgb24'),(224,224),interpolation=cv2.INTER_LINEAR))
                frames=norm(np.stack(imgs)); sec=time.perf_counter()-t
                assert np.array_equal(frames,frames_by_alias[row['alias']])
                preproc.append(dict(alias=row['alias'],repeat=repeat,seconds=sec,frames=len(frames)))
        save('BENCHMARK.json',dict(status='COMPLETED',rows=results,parity=parity,preprocessing=preproc,freeze_sha256=sha(ROOT/'FROZEN.json'),scope='cached normalized RGB to768 features; preprocessing separate; not total feedback latency'))
    event('benchmark_complete',timings=len(results))
if __name__=='__main__':
    try: main()
    except Exception:
        p=ROOT/'BENCH_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
