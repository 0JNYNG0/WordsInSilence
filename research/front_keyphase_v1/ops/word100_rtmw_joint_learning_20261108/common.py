from pathlib import Path
import sys,json,hashlib,time,os
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;LAB=O.parent;ROOT=LAB.parents[1]
sys.path[:0]=[str(LAB/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def event(kind,**kw):
 d=dict(time=time.time(),pid=os.getpid(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
