from pathlib import Path
import sys,json,hashlib,time
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
PREV=O/'word100_gru_semantic_finetune_20261109';J=O/'word100_rtmw_joint_learning_20261108';V=J/'feedback_viewfix';L=O/'word100_learned_guide_feedback_20261102';D=O/'word100_action_feedback_demo_20261108'
sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def event(kind,**kwargs):
 row=dict(time=time.time(),kind=kind,**kwargs)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)
