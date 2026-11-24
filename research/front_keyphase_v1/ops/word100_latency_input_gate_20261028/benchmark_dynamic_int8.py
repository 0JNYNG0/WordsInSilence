"""Bounded same-input CPU experiment. Never edits source models or cached features."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import torch

U=Path(__file__).resolve().parent
P=U.parent/'word100_shubert_feasibility_20261028'
sys.path.insert(0,str(P))
from dino_cpu import load_teacher,extract
from cpu_shubert_adapter import CPUShubertAdapter,CPUShubertConfig,stack_official_features

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(name,obj):
    p=U/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def event(obj):
    obj={'at':now(),**obj}
    with (U/'BENCH_EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    print(json.dumps(obj,ensure_ascii=False),flush=True)
def encode(model,streams):
    with torch.inference_mode():
        r=model.extract_features([{k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in streams.items()}],mask=False)
        return stack_official_features(r)[-1].cpu().numpy()
def drift(a,b):
    assert a.shape==b.shape and a.ndim==2
    a=a.astype(np.float64);b=b.astype(np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1);valid=(na>1e-12)&(nb>1e-12)
    cos=np.sum(a[valid]*b[valid],axis=1)/(na[valid]*nb[valid])
    rel=np.linalg.norm(a[valid]-b[valid],axis=1)/na[valid]
    return {'rows':len(a),'zero_norm_rows':int((~valid).sum()),'max_abs':float(np.max(np.abs(a-b))),
            'cosine_min':float(cos.min()) if len(cos) else None,'cosine_p05':float(np.quantile(cos,.05)) if len(cos) else None,
            'cosine_median':float(np.median(cos)) if len(cos) else None,'relative_l2_median':float(np.median(rel)) if len(rel) else None,
            'relative_l2_p95':float(np.quantile(rel,.95)) if len(rel) else None,'relative_l2_max':float(rel.max()) if len(rel) else None}

def main():
    freeze=json.loads((U/'FROZEN.json').read_text());aliases=freeze['fixed_aliases']
    if (U/'BENCH_RESULT.json').exists() or (U/'BENCH_FIRST_FAILURE.json').exists():raise RuntimeError('Existing outcome; preserve and inspect, do not overwrite')
    lock=U/'BENCH_RUNNING.json'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'at':now()},f)
    out=U/'benchmark_arrays';out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(freeze['threads'])
        protected={str(p):sha(p) for alias in aliases for p in (P/'features'/alias).iterdir() if p.is_file()}
        for p in [P/'inputs/MANIFEST.json',P/'weights/checkpoint_836_400000.pt',P/'weights/hands_dinov2_checkpoint.pth',P/'weights/face_dinov2_checkpoint.pth']:
            protected[str(p)]=sha(p)
        for alias in aliases:
            done=json.loads((P/'inputs'/alias/'DONE.json').read_text())
            for s,h in done['output_pins'].items():
                assert sha(s)==h,('input mismatch',s)
                protected[s]=h
        event({'stage':'input_hashes_verified','protected_files':len(protected),'engine':torch.backends.quantized.engine})
        load={};original={}
        for kind in ['hands','face']:
            t=time.perf_counter();m,audit=load_teacher(kind);load[kind]={'seconds':time.perf_counter()-t,'strict_load':audit};original[kind]=m
        t=time.perf_counter();ckpt=torch.load(P/'weights/checkpoint_836_400000.pt',map_location='cpu',weights_only=True,mmap=True)
        model=CPUShubertAdapter(CPUShubertConfig.from_mapping(ckpt['cfg']['model']));model.load_exact_state_dict(ckpt['model']);model.eval().requires_grad_(False);del ckpt
        load['shubert_seconds']=time.perf_counter()-t
        quantized={};qa={}
        for kind,m in original.items():
            t=time.perf_counter()
            q=torch.ao.quantization.quantize_dynamic(m,{torch.nn.Linear},dtype=torch.qint8,inplace=False)
            q.eval().requires_grad_(False)
            modules=[(n,v) for n,v in q.named_modules() if isinstance(v,torch.ao.nn.quantized.dynamic.Linear)]
            assert len(modules)>0
            assert all(v.weight().dtype==torch.qint8 for _,v in modules)
            qa[kind]={'prepare_seconds':time.perf_counter()-t,'original_linear_count':sum(isinstance(v,torch.nn.Linear) for v in m.modules()),
                'quantized_linear_count':len(modules),'weight_dtype':str(modules[0][1].weight().dtype),'module_names':[n for n,_ in modules]}
            quantized[kind]=q
        assert all(p.dtype==torch.float32 for p in model.parameters())
        write('QUANTIZATION_AUDIT.json',{'at':now(),'engine':torch.backends.quantized.engine,'supported_engines':torch.backends.quantized.supported_engines,'teachers':qa,'load':load,'shubert_unchanged_fp32':True})
        variants={'fp32':original,'int8_linear':quantized}
        warm={}
        for variant,teachers in variants.items():
            streams={};t=time.perf_counter()
            for name,kind in [('face','face'),('left_hand','hands'),('right_hand','hands')]:
                rgb=np.load(P/'inputs/V01'/f'{name}_rgb.npy',mmap_mode='r')[:8]
                streams[name]=extract(teachers[kind],rgb,batch_size=8)
            streams['body_posture']=np.load(P/'inputs/V01/body14.npy')[:8]
            y=encode(model,streams);assert y.shape==(8,768) and np.isfinite(y).all()
            warm[variant]=time.perf_counter()-t;event({'stage':'warmup','variant':variant,'seconds':warm[variant]})
        rows=[]
        for repeat in range(2):
            for alias in (aliases if repeat==0 else list(reversed(aliases))):
                for variant in (['fp32','int8_linear'] if repeat==0 else ['int8_linear','fp32']):
                    streams={};timing={};t0=time.perf_counter()
                    for name,kind in [('face','face'),('left_hand','hands'),('right_hand','hands')]:
                        t=time.perf_counter();rgb=np.load(P/'inputs'/alias/f'{name}_rgb.npy',mmap_mode='r')
                        streams[name]=extract(variants[variant][kind],rgb,batch_size=8)
                        timing[name+'_seconds']=time.perf_counter()-t
                    t=time.perf_counter();streams['body_posture']=np.load(P/'inputs'/alias/'body14.npy');timing['body_read_seconds']=time.perf_counter()-t
                    t=time.perf_counter();y=encode(model,streams);timing['shubert_seconds']=time.perf_counter()-t
                    timing['cached_rgb_to_features_seconds']=time.perf_counter()-t0
                    assert y.shape==(len(streams['face']),768) and np.isfinite(y).all()
                    # I/O for result preservation is outside measured region.
                    dest=out/f'{alias}_{variant}_r{repeat}.npz';np.savez(dest,**streams,last_ffn=y)
                    row={'alias':alias,'variant':variant,'repeat':repeat,'frames':len(y),'timing':timing,'array':str(dest),'sha256':sha(dest)}
                    rows.append(row);write('BENCH_STATUS.json',{'stage':'RUNNING','completed':len(rows),'total':12,'last':row});event({'stage':'clip_complete',**{k:row[k] for k in ['alias','variant','repeat','frames','timing']}})
        differences=[];repro=[];cached=[]
        for alias in aliases:
            for repeat in range(2):
                with np.load(out/f'{alias}_fp32_r{repeat}.npz') as a,np.load(out/f'{alias}_int8_linear_r{repeat}.npz') as b:
                    differences.append({'alias':alias,'repeat':repeat,'fields':{k:drift(a[k],b[k]) for k in a.files}})
            for variant in variants:
                with np.load(out/f'{alias}_{variant}_r0.npz') as a,np.load(out/f'{alias}_{variant}_r1.npz') as b:
                    repro.append({'alias':alias,'variant':variant,'fields':{k:drift(a[k],b[k]) for k in a.files}})
            with np.load(out/f'{alias}_fp32_r0.npz') as a,np.load(P/'features'/alias/'streams.npz') as b:
                fields={k:drift(a[k],b[k]) for k in b.files};fields['last_ffn']=drift(a['last_ffn'],np.load(P/'features'/alias/'last_ffn.npy'));cached.append({'alias':alias,'fields':fields})
        assert all(sha(p)==h for p,h in protected.items()),'protected artifact modified'
        summary={}
        for variant in variants:
            times=[r['timing']['cached_rgb_to_features_seconds'] for r in rows if r['variant']==variant]
            summary[variant]={'min':min(times),'median':float(np.median(times)),'max':max(times),'n':len(times)}
        ratio=summary['fp32']['median']/summary['int8_linear']['median']
        result={'status':'COMPLETE_CPU_SPEED_AND_NUMERICAL_AUDIT_NOT_SIGN_ACCURACY','at':now(),'torch':torch.__version__,'engine':torch.backends.quantized.engine,
                'frozen_sha256':sha(U/'FROZEN.json'),'script_sha256':sha(__file__),'load':load,'quantization':qa,'warmup_seconds':warm,
                'rows':rows,'summary':summary,'median_speedup':ratio,'differences':differences,'repeat_differences':repro,'previous_cache_differences':cached,
                'protected_pins':protected,'protected_unchanged':True,'same_all_original_frames':True,'face_removed':False,'fps_reduced':False,
                'source_kind_input_to_shubert':False,'observation_side_input_work_separate':True,'training_runs':0,'private_access':False,'operating_changes':0,
                'limitations':['Only3publicclips and2repeats pervariant; no latencySLA/p95claim','cachedRGB→features excludesdecode/MediaPipe/crops/DTW/feedback/UI','numericaldrift is nothandshape/stage accuracy; noKSLtrainedhead exists','OSbackground/power/thermal conditions notfullycontrolled','quantizedmodels live only inmemory; pretrained filesunmodified']}
        write('BENCH_RESULT.json',result);write('BENCH_STATUS.json',{'stage':'COMPLETE','completed':12,'total':12});event({'stage':'complete','summary':summary,'median_speedup':ratio})
        lock.rename(U/'BENCH_FINISHED.json')
    except BaseException:
        obj={'at':now(),'status':'FAILED_PRESERVED','traceback':traceback.format_exc()}
        with (U/'BENCH_FIRST_FAILURE.json').open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
        write('BENCH_STATUS.json',obj);event({'stage':'failure','traceback':obj['traceback']});raise

if __name__=='__main__':main()
