"""One frozen TCN pair on the saved public observation, no re-extraction."""
from pathlib import Path
import json,sys,hashlib,traceback
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(U))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
for p,h in read(U/'EXECUTION_FROZEN.json')['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
assert not (U/'public_tcn.json').exists()
try:
 import integrated_adapter as adapter
 run=read(O/'word100_local_event_runtime_20261103/PUBLIC_RUN_RESULT.json')
 path=Path(run['result']);assert hashlib.sha256(path.read_bytes()).hexdigest()==run['result_sha256']
 r=read(path)
 tcn=adapter.classify(path.parent,r['overlay'])
 assert all(tcn[k]['model_status']=='READY' for k in ['C','T3F']),tcn
 out=dict(source_sha256=r['source_sha256'],tcn=tcn,role='public saved observation integration check',training=False)
 with (U/'public_tcn.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
 print({k:tcn[k]['ranking'][:3] for k in ['C','T3F']})
except BaseException:
 with (U/'PUBLIC_TCN_FIRST_FAILURE.txt').open('x',encoding='utf8') as f:f.write(traceback.format_exc())
 raise
