"""One owned worker for frozen order and new local-event heads; no camera."""
from pathlib import Path
import sys, json, traceback, hashlib, time

U = Path(__file__).resolve().parent
O = U.parent
Z = O/'word100_partial_order_feedback_20261102'
sys.path.insert(0, str(Z))
from order_runtime import PartialOrderEngine, read, sha, np
from order_feedback import attach_partial_order
sys.path.insert(0, str(U))
from local_runtime import LocalEventRuntime

def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)

def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def value_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()

def allowed(path):
    d = Path(path).resolve()
    parts = d.relative_to(U.resolve()).parts
    assert len(parts)==3 and parts[0]=='public_runtime_runs' and parts[2]=='analysis'
    assert len(parts[1])==32 and all(c in '0123456789abcdef' for c in parts[1])
    return d

def process(order, local, request):
    d = allowed(request['directory'])
    contract = read(d/'RUNTIME_INPUT_CONTRACT.json')
    assert contract['job_id'] == d.parent.name == request['job_id']
    assert contract['word_id'] == request['word_id']
    assert contract['source_role'] == 'public_runtime_inference'
    assert all(contract[k] is False for k in ['training','tuning','candidate_selection'])
    assert contract['input'] == str(d/'analysis.npz') and contract['base'] == str(d/'SOURCE_FEEDBACK.json')
    assert sha(contract['input'])==contract['input_sha256'] and sha(contract['base'])==contract['base_sha256']
    a=dict(np.load(contract['input'],allow_pickle=False)); base=read(contract['base'])
    assert base['word_id']==contract['word_id'] and base['source_sha256']==contract['source_sha256']
    assert hashlib.sha256(a['frame_id'].tobytes()+a['timestamp_us'].tobytes()).hexdigest()==contract['native_axis_sha256']
    assert not (d/'WORKER_INFERENCE_STARTED.json').exists(), 'Do not repeat a partially scored job'
    write(d/'WORKER_INFERENCE_STARTED.json', dict(at=time.time(), job_id=contract['job_id']))
    value=order.infer(a,contract['word_id'],base)
    ordered=attach_partial_order(base,value['pairs'],order.families,order.graphs[contract['word_id']])
    np.savez_compressed(d/'order_arrays.npz',**{k:v for k,v in value.items() if isinstance(v,np.ndarray)})
    write(d/'ORDER_PAIRS.json',value['pairs'])
    ordered['partial_order_runtime']=dict(head_seconds=value['seconds'],model_sha256=value['model_sha256'],
        additional_DTW=0, individual_reference_video_comparisons=0, job_id=contract['job_id'],
        input_sha256=contract['input_sha256'], base_sha256=contract['base_sha256'], training=False,
        model_use='Preserved frozen research order head; not verified linguistic error judgement')
    assert all(ordered[k]==v for k,v in base.items())
    write(d/'ORDER_FEEDBACK.json', ordered)
    local_contract=dict(contract,base=str(d/'ORDER_FEEDBACK.json'),base_sha256=sha(d/'ORDER_FEEDBACK.json'),
        base_value_sha256=value_sha(ordered))
    write(d/'LOCAL_INPUT_CONTRACT.json',local_contract)
    out=local.analyze(ordered,d/'analysis.npz',local_contract,d)
    assert all(out[k]==v for k,v in ordered.items())
    dest=d/'LOCAL_EVENT_FEEDBACK.json'
    if dest.exists():
        assert read(dest)==out
    else:
        write(dest,out)
    return dict(status='complete',id=request['id'],job_id=contract['job_id'],path=str(dest),result_sha256=sha(dest))

def main():
    for p,h in read(U/'EXECUTION_FROZEN.json')['pins'].items():
        assert sha(p)==h,p
    start=time.perf_counter()
    order=PartialOrderEngine()
    local=LocalEventRuntime()
    local.load()
    emit(dict(status='ready',seconds=time.perf_counter()-start,local_model_loaded=True))
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            assert request['command']=='infer'
            emit(process(order,local,request))
        except Exception as exc:
            failure=U/'WORKER_FIRST_FAILURE.txt'
            if not failure.exists(): failure.write_text(traceback.format_exc(),encoding='utf-8')
            emit(dict(status='failed',id=request.get('id'),job_id=request.get('job_id'),error=str(exc)))

if __name__=='__main__':
    try:
        main()
    except Exception:
        failure=U/'WORKER_BOOT_FIRST_FAILURE.txt'
        if not failure.exists(): failure.write_text(traceback.format_exc(),encoding='utf-8')
        raise
