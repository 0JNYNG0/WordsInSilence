"""Source-linked live feedback plus frozen partial-order neural inference."""
from pathlib import Path
import sys,importlib.util,time,json,subprocess,threading,queue,uuid,hashlib
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;X=O/'word100_learned_component_extent_20261102'
sys.path.insert(0,str(X))
spec=importlib.util.spec_from_file_location('_partial_order_live_source',X/'learner_adapter.py')
source=importlib.util.module_from_spec(spec);sys.modules[spec.name]=source;spec.loader.exec_module(source)
read=source.read;write=source.write;WORDS=source.WORDS
PY=O/'word100_signrep_feasibility_20261029/runtime/Scripts/python.exe'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Worker:
    def __init__(self):
        self.lock=threading.Lock();self.messages=queue.Queue();self.broken=False
        self.log=(U/'ORDER_WORKER.stderr.log').open('a',encoding='utf8')
        self.proc=subprocess.Popen([str(PY),'-B','-u',str(U/'order_worker.py')],cwd=str(U),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf8',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        threading.Thread(target=self._read,daemon=True).start()
        reply=self._next(60);assert reply['status']=='ready',reply
        write(U/'ORDER_WORKER_STATE.json',dict(pid=self.proc.pid,status='ready',training=False))
    def _read(self):
        try:
            for line in self.proc.stdout:self.messages.put(json.loads(line))
        except Exception as exc:self.messages.put(dict(status='failed',error=str(exc)))
        finally:self.messages.put(dict(status='terminated',exit_code=self.proc.poll()))
    def _next(self,timeout):
        try:return self.messages.get(timeout=timeout)
        except queue.Empty:
            self.broken=True
            raise RuntimeError('순서 분석기의 응답을 확인하지 못했습니다. 자동으로 중복 실행하지 않습니다.')
    def infer(self,d,word):
        with self.lock:
            if self.broken or self.proc.poll() is not None:raise RuntimeError('순서 분석기가 준비되지 않았습니다.')
            rid=uuid.uuid4().hex;job=d.parent.name
            self.proc.stdin.write(json.dumps(dict(command='infer',id=rid,directory=str(d),word_id=word,job_id=job))+'\n');self.proc.stdin.flush()
            reply=self._next(60)
            if reply.get('id')!=rid or reply.get('job_id',job)!=job:
                self.broken=True;raise RuntimeError('분석 요청과 결과가 일치하지 않습니다.')
            if reply.get('status')!='complete':raise RuntimeError(reply.get('error','순서 분석 실패'))
            assert Path(reply['path']).resolve()==(d/'ORDER_FEEDBACK.json').resolve()
            assert sha(reply['path'])==reply['result_sha256']
            return read(reply['path'])

worker=None
def warmup():
    global worker
    source.U=U;out=source.warmup()
    if worker is None:worker=Worker()
    return dict(base=out,partial_order_worker_ready=True)

def analyze(video,directory,word,condition='learned_motion',progress=None):
    assert worker is not None and condition=='learned_motion'
    d=Path(directory).resolve();rel=d.relative_to(U.resolve())
    assert len(rel.parts)==3 and rel.parts[0] in ['public_runtime_runs','private_sessions'] and rel.parts[2]=='analysis'
    start=time.perf_counter();source.U=U
    result=source.analyze(video,d,word,condition,progress)
    arrays=dict(np.load(d/'analysis.npz',allow_pickle=False))
    assert sha(video)==result['source_sha256'] and result['word_id']==word
    contract=dict(job_id=d.parent.name,word_id=word,source_sha256=result['source_sha256'],
        input=str(d/'analysis.npz'),input_sha256=sha(d/'analysis.npz'),
        base=str(d/'SOURCE_FEEDBACK.json'),base_sha256=sha(d/'SOURCE_FEEDBACK.json'),
        native_axis_sha256=hashlib.sha256(arrays['frame_id'].tobytes()+arrays['timestamp_us'].tobytes()).hexdigest(),
        training=False,tuning=False,candidate_selection=False)
    write(d/'ORDER_INPUT_CONTRACT.json',contract)
    if progress:progress(dict(stage='checking_observed_order'))
    out=worker.infer(d,word)
    # Keep the base latency intact; new total is explicitly named for this
    # combined pipeline and includes the order worker round trip.
    out['combined_feedback_latency_seconds']=time.perf_counter()-start
    out['combined_latency_scope']='server analysis after request begins, not browser upload or model boot'
    write(d/'LIVE_FEEDBACK.json',out)
    if progress:progress(dict(stage='complete'))
    return out
