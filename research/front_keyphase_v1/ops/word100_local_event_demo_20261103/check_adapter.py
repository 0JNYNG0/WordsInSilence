"""Exercise new isolation/serialization with synthetic inputs, without models."""
from pathlib import Path
from functools import wraps
import ast, json, hashlib, threading, time, types, sys
U=Path(__file__).resolve().parent
def load_function(name,env,path=None):
    path=path or U/'learner_adapter.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[name]
def main():
    checks=[]
    def check(label,value):
        assert value,label
        checks.append(label)
    def rejects(label,fn):
        try: fn()
        except (AssertionError,ValueError): check(label,True)
        else: check(label,False)
    work=U/'contract_tests'; work.mkdir(exist_ok=False)
    lock=threading.RLock()
    env=dict(Path=Path,U=work,WORDS=['WORD1133'],threading=threading,wraps=wraps,ANALYZE_LOCK=lock)
    env['sha']=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    validate=load_function('validate_request',env)
    for folder,role in [('private_sessions','user_inference_only'),('public_runtime_runs','public_runtime_inference')]:
        d=work/folder/('a'*32)/'analysis';d.parent.mkdir(parents=True)
        video=d.parent/'input.video';video.write_bytes(b'EXPLICIT SYNTHETIC CONTRACT TEST, NOT A VIDEO')
        p=dict(source_role=role,source_sha256=env['sha'](video),word_id='WORD1133',training=False,tuning=False,candidate_selection=False)
        check('matching role:'+folder,validate(video,d,'WORD1133',p)==(d.resolve(),p['source_sha256']))
        wrong=dict(p,source_role='public_runtime_inference' if folder=='private_sessions' else 'user_inference_only')
        rejects('no role relabel:'+folder,lambda:validate(video,d,'WORD1133',wrong))
        for field in ['training','tuning','candidate_selection']:
            rejects('inference only:'+folder+':'+field,lambda field=field:validate(video,d,'WORD1133',dict(p,**{field:True})))
        rejects('hash binding:'+folder,lambda:validate(video,d,'WORD1133',dict(p,source_sha256='0'*64)))
        rejects('target binding:'+folder,lambda:validate(video,d,'WORD0000',p))
        if folder=='private_sessions':
            other=work/'unrelated.video';other.write_bytes(video.read_bytes())
            rejects('private source belongs to own job',lambda:validate(other,d,'WORD1133',p))
        d.mkdir();rejects('never overwrite previous analysis:'+folder,lambda:validate(video,d,'WORD1133',p))
    allowed=load_function('allowed',dict(Path=Path,U=work),U/'order_event_worker.py')
    for folder in ['private_sessions','public_runtime_runs']:
        p=work/folder/('b'*32)/'analysis';check('worker isolated root:'+folder,allowed(p)==p.resolve())
    rejects('worker path escape',lambda:allowed(work.parent/'private_sessions'/('b'*32)/'analysis'))
    rejects('worker nonhex job',lambda:allowed(work/'private_sessions'/'not-a-job'/'analysis'))
    serialized=load_function('serialized',env)
    entered=threading.Event(); release=threading.Event(); second=threading.Event(); sequence=[]
    @serialized
    def first():
        sequence.append('first-start'); entered.set(); assert release.wait(3); sequence.append('first-end')
    @serialized
    def next_call():
        sequence.append('next-start'); second.set()
    a=threading.Thread(target=first);b=threading.Thread(target=next_call)
    a.start();assert entered.wait(3);b.start()
    check('whole call waits during active analysis',not second.wait(.05))
    release.set();a.join(3);b.join(3)
    check('whole call serialization',sequence==['first-start','first-end','next-start'])
    # Run only the AST-loaded wrapper with a failing fake body. No native imports.
    fake_extract=object();fake_capture=object();replacement=object()
    shared=types.SimpleNamespace(extract_rgb=fake_extract,base=types.SimpleNamespace(_capture=fake_capture))
    def failing(*args):
        check('new parallel capture installed inside lock',shared.base._capture is replacement)
        check('rgb support tap installed inside lock',shared.extract_rgb is not fake_extract)
        raise ValueError('synthetic body failure')
    source=types.SimpleNamespace(old=types.SimpleNamespace(shared_capture=types.SimpleNamespace(shared=shared)),analyze=failing)
    d=work/'private_sessions'/('c'*32)/'analysis';d.parent.mkdir(parents=True)
    video=d.parent/'input.video';video.write_bytes(b'CONTRACT TEST')
    provenance=dict(source_role='user_inference_only',source_sha256=env['sha'](video),word_id='WORD1133',training=False,tuning=False,candidate_selection=False)
    env.update(worker=object(),source=source,time=time,extract_with_support=lambda *a:object())
    old_module=sys.modules.get('parallel_capture');sys.modules['parallel_capture']=types.SimpleNamespace(capture_parallel=replacement)
    try:
        analyze=load_function('analyze',env)
        rejects('body exception preserved',lambda:analyze(video,d,'WORD1133',provenance=provenance))
        check('original RGB function restored on error',shared.extract_rgb is fake_extract)
        check('original sensor function restored on error',shared.base._capture is fake_capture)
    finally:
        if old_module is None:sys.modules.pop('parallel_capture',None)
        else:sys.modules['parallel_capture']=old_module
    for name in ['warmup','close','analyze']:
        node=next(n for n in ast.parse((U/'learner_adapter.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        check('same full-lock decorator:'+name,any(isinstance(v,ast.Name) and v.id=='serialized' for v in node.decorator_list))
    ak=U.parent/'word100_local_event_runtime_20261103'
    for name in ['extract_with_support']:
        def body(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        check('frozen RGB extraction body unchanged',ast.dump(body(U/'learner_adapter.py'))==ast.dump(body(ak/'learner_adapter.py')))
    report=dict(status='PASS_SYNTHETIC_DEMO_ADAPTER_CONTRACT',check_count=len(checks),checks=checks,models_loaded=0,video_decodes=0,neural_forwards=0,
        private_recordings_read=0,synthetic_bytes_only=True,
        pins={str(U/n):env['sha'](U/n) for n in ['learner_adapter.py','order_event_worker.py','local_runtime.py','parallel_capture.py','check_adapter.py']})
    with (U/'ADAPTER_CHECKS.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(report['status'],len(checks))
if __name__=='__main__':main()
