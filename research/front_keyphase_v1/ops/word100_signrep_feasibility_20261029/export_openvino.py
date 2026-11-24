from common import *
import traceback
def main():
    import torch, numpy as np, openvino as ov
    assert read(ROOT/'WRAPPER_PARITY.json')['status']=='PASS'
    model=load_model(); wrapper=feature_wrapper(model)
    frames=norm(np.load(ROOT/'inputs/V01/rgb224.npy',allow_pickle=False))
    x=batch(frames,[0],1)[0]
    with torch.inference_mode(): reference=wrapper(torch.from_numpy(x)).numpy()
    # OV shape inference mishandles empty class-token branches (num_cls_tokens=0).
    # The same official repository supplies the mathematically identical no-CLS
    # attention. Reuse its forward with all original weights intact, then verify.
    from models.hiera.hiera import MaskUnitAttention
    from types import MethodType
    import copy
    export_wrapper=copy.deepcopy(wrapper)
    for block in export_wrapper.backbone.blocks:
        assert block.attn.num_cls_tokens==0
        block.attn.forward=MethodType(MaskUnitAttention.forward,block.attn)
    checks=[]
    with torch.inference_mode():
        for row in read(ROOT/'INPUTS.json')['rows']:
            fr=norm(np.load(ROOT/'inputs'/row['alias']/'rgb224.npy',allow_pickle=False)); seq=row['window_starts']
            for s in [seq[0],seq[len(seq)//2],seq[-1]]:
                xx=torch.from_numpy(batch(fr,[s],1)[0]); a=wrapper(xx); b=export_wrapper(xx)
                assert torch.allclose(a,b,rtol=1e-5,atol=1e-5)
                checks.append(dict(alias=row['alias'],start=s,**diff(a.numpy(),b.numpy())))
    if not (ROOT/'EXPORT_COMPAT_PARITY.json').exists():
        save('EXPORT_COMPAT_PARITY.json',dict(status='PASS',checks=checks,change='reuse official no-CLS attention forward only; weights/inputs unchanged',source_sha256=sha(ROOT/'vendor/SignRep/models/hiera/hiera.py')))
    event('conversion_start')
    t=time.perf_counter()
    with torch.no_grad():
        converted=ov.convert_model(export_wrapper,example_input=torch.from_numpy(x),input=[(-1,3,16,224,224)])
    conversion=time.perf_counter()-t
    path=ROOT/'converted/signrep_features_fp32.xml';path.parent.mkdir(exist_ok=False)
    ov.save_model(converted,path,compress_to_fp16=False)
    event('conversion_done',seconds=conversion)
    core=ov.Core(); cpu_model=core.read_model(path); cpu_model.reshape([1,3,16,224,224])
    compiled=core.compile_model(cpu_model,'CPU',{'INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':4,'PERFORMANCE_HINT':'LATENCY'})
    y=compiled([x])[0]
    d=diff(reference,y)
    if d['max_relative_l2']>0.0001 or d['min_cosine']<0.99999: raise RuntimeError(d)
    save('EXPORT.json',dict(status='PASS',conversion_seconds=conversion,cpu_parity=d,xml_sha256=sha(path),bin_sha256=sha(path.with_suffix('.bin')),compression='none FP32',telemetry='built-in no-op stub',output='768 pooled features, not coordinates or semantic phases'))
    event('export_validated',**d)
if __name__=='__main__':
    try: main()
    except Exception:
        p=ROOT/'EXPORT_FIRST_FAILURE.txt'
        if not p.exists(): p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
