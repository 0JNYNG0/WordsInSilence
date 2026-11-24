from pathlib import Path
import json,hashlib,sys,time
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
J=O/'word100_rtmw_joint_learning_20261108';V=J/'feedback_viewfix';D=O/'word100_action_feedback_demo_20261108';L=O/'word100_learned_guide_feedback_20261102'
sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event(kind,**kw):
 r=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)
