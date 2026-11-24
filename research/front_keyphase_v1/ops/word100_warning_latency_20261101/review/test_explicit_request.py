"""Fake API checks only: no OpenVINO import, model, video or GPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib, importlib.util, json, sys, threading

U=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('_explicit_request_tested',U/'explicit_request.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    pins={str(U/n):sha(U/n) for n in ['candidate_adapter.py','run_server.py']}
    class Request:
        def __init__(self):self.calls=[];self.outputs=[];self.fail=False
        def infer(self,items,**kwargs):
            if self.fail:raise ValueError('fake inference failure')
            self.calls.append((threading.current_thread(),items,kwargs))
            out={0:list(items[0])};self.outputs.append(out);return out
    class Compiled:
        def __init__(self):self.created=[];self.request=Request();self.fail=False
        def __call__(self,*args):raise AssertionError('CompiledModel convenience call must not be used')
        def create_infer_request(self):
            self.created.append(threading.current_thread())
            if self.fail:raise ValueError('fake creation failure')
            return self.request
    with TemporaryDirectory(prefix='explicit_fake_',dir=U/'review') as tmp:
        path=Path(tmp)/'windows.jsonl';model=Compiled();profile=[]
        wrapper=m.ExplicitRequest(model,profile,path)
        check('constructor creates no request',model.created==[])
        check('constructor reserves empty new ledger',path.read_text()=='')
        inputs=[[[1.,2.]],[[3.,4.]]];outputs=[];errors=[]
        def worker():
            try:
                for values in inputs:outputs.append(wrapper(values))
            except BaseException as exc:errors.append(exc)
        thread=threading.Thread(target=worker,name='fake-signrep-consumer');thread.start();thread.join(2)
        check('fake worker completed without error',not thread.is_alive() and not errors)
        check('one request created by consumer',model.created==[thread])
        check('all infer calls same consumer',all(c[0] is thread for c in model.request.calls))
        check('input objects passed unchanged',all(c[1] is x for c,x in zip(model.request.calls,inputs)))
        check('explicit sharing matches convenience defaults',all(c[2]=={'share_inputs':True,'share_outputs':False} for c in model.request.calls))
        check('returned output objects unmodified',all(a is b for a,b in zip(outputs,model.request.outputs)))
        check('earlier independent output retained',outputs[0][0]==[1.,2.])
        check('profile list is supplied object',wrapper.profile is profile)
        check('all calls get ordered durations',len(profile)==2 and [r['window_index'] for r in profile]==[0,1] and all(r['infer_seconds']>=0 for r in profile))
        events=[json.loads(line) for line in path.read_text().splitlines()]
        check('begins and ends persisted around each call',[r['event'] for r in events]==['CALL_BEGIN','REQUEST_CREATED','CALL_END','CALL_BEGIN','CALL_END'])
        try:wrapper(inputs[0])
        except RuntimeError:check('main thread cannot reuse consumer request',len(model.request.calls)==2)
        else:raise AssertionError('wrong thread accepted')
        try:m.ExplicitRequest(model,jsonl_path=path)
        except FileExistsError:check('existing ledger protected',True)
        else:raise AssertionError('overwrite accepted')
        other=m.ExplicitRequest(model);other(inputs[0])
        check('new wrapper can reuse compiled model',len(model.created)==2 and model.created[-1] is threading.current_thread())
        for phase in ['request_create','infer']:
            failed=Compiled();failed.fail=phase=='request_create';failed.request.fail=phase=='infer'
            w=m.ExplicitRequest(failed,jsonl_path=Path(tmp)/(phase+'.jsonl'))
            try:w(inputs[0])
            except ValueError:check(phase+' error propagates',True)
            else:raise AssertionError('failure swallowed')
            check(phase+' timing record retains failure',w.profile[0]['status']=='FAILED' and w.profile[0]['failed_phase']==phase)
            try:w(inputs[0])
            except RuntimeError:check(phase+' automatic retry blocked',len(failed.created)==1)
            else:raise AssertionError('failure retry accepted')
        reentrant=m.ExplicitRequest(Compiled());reentrant._lock.acquire()
        try:
            try:reentrant(inputs[0])
            except RuntimeError:check('busy call rejected without waiting',True)
            else:raise AssertionError('reentrant accepted')
        finally:reentrant._lock.release()
    check('adapter and server untouched',all(sha(p)==s for p,s in pins.items()))
    return dict(status='PASS_FAKE_API_ONLY',checks=len(checks),results=checks,
                code_sha256=sha(U/'explicit_request.py'),test_sha256=sha(__file__),protected_pins=pins,
                real_model_loads=0,real_inferences=0,real_video_reads=0,
                limitation='API/lifetime/logging tests only. No claim that this resolves the observed latency regression.')

if __name__=='__main__':
    result=run()
    with (U/'review/EXPLICIT_REQUEST_SELFTEST.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ['status','checks','code_sha256']}))
