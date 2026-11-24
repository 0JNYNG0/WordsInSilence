"""Separate seated-webcam research demo, original8802 preserved."""
from pathlib import Path
import importlib.util,sys,json,argparse,hashlib
U=Path(__file__).resolve().parent;B=U.parent/'word100_multipeople_live_feedback_20261101'
sys.dont_write_bytecode=True
def load(name,path):
    if name in sys.modules:
        assert Path(sys.modules[name].__file__).resolve()==path.resolve(),name
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
class Engine:
    def __init__(self,live):self.live=live
    def catalog(self):
        c=self.live.catalog();c['input_policy']='live_sensor_extent_v1';return c
    def analyze_submission(self,path,jobdir,progress=None):
        result=self.live.analyze_submission(path,jobdir,progress)
        complete=json.loads((Path(jobdir)/'adapter/COMPLETE.json').read_text(encoding='utf8'))
        result['input_policy']=complete['calibration']
        if complete['status']=='UNSUPPORTED_INPUT':
            from sensor_boundary import diagnostic
            reasons=complete['unavailable_reasons']
            result['stages'][0]['message']=diagnostic(reasons)
            result['input_failure_reasons']=reasons
            result['summary']='입력에서 확인 가능한 근거가 부족합니다. 준비 자세의 손내림 때문에 거절한 것은 아닙니다.'
        result['training_used_submission']=False
        return result
def components():
    a=load('adapter',U/'adapter.py');load('engine',B/'engine.py');load('evidence_guard',B/'evidence_guard.py')
    live=load('_seated_live',B/'live_engine.py');s=load('_seated_server',B/'server.py');s.ROOT=U
    return s,Engine(live),a
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8804);args=p.parse_args()
    assert args.port==8804,'Preserve existing servers'
    s,e,a=components();print('Preparing unchanged models; no camera access.',flush=True)
    srv=s.create_server(e,args.port,storage_root=U/'sessions',warmup=a.warmup)
    (srv.session_dir/'ROUTING.json').write_text(json.dumps(dict(port=args.port,policy='live_sensor_extent_v1',
        adapter_sha256=hashlib.sha256((U/'adapter.py').read_bytes()).hexdigest(),model_changed=False)),encoding='utf8')
    print('READY http://127.0.0.1:8804/',flush=True)
    try:srv.serve_forever()
    finally:srv.server_close()
