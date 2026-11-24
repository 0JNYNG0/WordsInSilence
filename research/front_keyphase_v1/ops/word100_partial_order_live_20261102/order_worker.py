"""Persistent isolated neural worker; no camera, training, or reference bank."""
from pathlib import Path
import sys,json,hashlib,traceback
AA=Path(__file__).resolve().parent;O=AA.parent;Z=O/'word100_partial_order_feedback_20261102'
sys.path.insert(0,str(Z))
from order_runtime import PartialOrderEngine,read,sha,np
from order_feedback import attach_partial_order

def emit(value):print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)

def allowed_directory(path):
    p=Path(path).resolve()
    relative=p.relative_to(AA.resolve())
    assert len(relative.parts)==3 and relative.parts[0] in ['public_runtime_runs','private_sessions']
    assert len(relative.parts[1])==32 and all(c in '0123456789abcdef' for c in relative.parts[1])
    assert relative.parts[2]=='analysis'
    return p

def process(engine,request):
    d=allowed_directory(request['directory']);contract=read(d/'ORDER_INPUT_CONTRACT.json')
    assert contract['job_id']==d.parent.name==request['job_id']
    assert contract['word_id']==request['word_id'] and not contract['training']
    assert contract['input']==str(d/'analysis.npz') and contract['base']==str(d/'SOURCE_FEEDBACK.json')
    assert sha(contract['input'])==contract['input_sha256'] and sha(contract['base'])==contract['base_sha256']
    a=dict(np.load(contract['input'],allow_pickle=False));base=read(contract['base'])
    assert base['word_id']==contract['word_id'] and base['source_sha256']==contract['source_sha256']
    assert hashlib.sha256(a['frame_id'].tobytes()+a['timestamp_us'].tobytes()).hexdigest()==contract['native_axis_sha256']
    dest=d/'ORDER_FEEDBACK.json';assert not dest.exists()
    value=engine.infer(a,contract['word_id'],base)
    out=attach_partial_order(base,value['pairs'],engine.families,engine.graphs[contract['word_id']])
    # Native indices and times are stored independently of the display summary.
    np.savez_compressed(d/'order_arrays.npz',**{k:v for k,v in value.items() if isinstance(v,np.ndarray)})
    (d/'ORDER_PAIRS.json').write_text(json.dumps(value['pairs'],ensure_ascii=False,indent=2),encoding='utf8')
    out['partial_order_runtime']=dict(head_seconds=value['seconds'],model_sha256=value['model_sha256'],
        additional_DTW=0,individual_reference_video_comparisons=0,job_id=contract['job_id'],
        input_sha256=contract['input_sha256'],base_sha256=contract['base_sha256'],training=False,
        model_use='post-comparison research component; not independent generalization validation')
    for k,v in base.items():assert out[k]==v,k
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return dict(id=request['id'],status='complete',path=str(dest),result_sha256=sha(dest),
        head_seconds=value['seconds'],job_id=contract['job_id'])

def main():
    for p,h in read(AA/'EXECUTION_FROZEN.json')['pins'].items():assert sha(p)==h,p
    engine=PartialOrderEngine();emit(dict(status='ready'))
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line);assert request['command']=='infer'
            emit(process(engine,request))
        except BaseException as exc:
            error=traceback.format_exc();p=AA/'ORDER_WORKER_FIRST_FAILURE.txt'
            if not p.exists():p.write_text(error,encoding='utf8')
            emit(dict(id=request.get('id'),status='failed',error=str(exc)))

if __name__=='__main__':
    try:main()
    except BaseException:
        p=AA/'ORDER_WORKER_BOOT_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
