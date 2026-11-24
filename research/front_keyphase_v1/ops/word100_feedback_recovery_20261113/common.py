from pathlib import Path
import json,hashlib,sys,time,importlib.util,os
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
P=O/'word100_feedback_recovery_plan_20261113';PREV=O/'word100_feature_expansion_20261113'
B=O/'word100_shared_motion_learning_20261109';J=O/'word100_rtmw_joint_learning_20261108';A=O/'demo_handshape_learning_20261113'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');q.replace(p)
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def event(kind,**kw):
    x=dict(time=time.time(),pid=os.getpid(),kind=kind,**kw)
    with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print(json.dumps(x,ensure_ascii=False),flush=True)
