"""Isolated common observations experiment. Never imports a target into motion()."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
B=O/'word100_shared_motion_learning_20261109'; M=O/'word100_motion_components_20261109'
J=O/'word100_rtmw_joint_learning_20261108'; V=J/'feedback_viewfix'
D=O/'word100_action_feedback_demo_20261108'; E=O/'word100_event_completion_20261109'
F=O/'word100_completion_fullpath_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');t.replace(p)
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def event(kind,**kw):
 r=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)
shared=load('_common_action_shared',B/'core.py'); motion=load('_common_action_motion',M/'motion.py')
STATS=shared.STATS; BASE=shared.BASE
def base_model():
 m=shared.PoseGuideNet(STATS['guide_vectors'],STATS['guide_words'],STATS['prototype_lengths'])
 m.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True));return m
def physical(z):
 x=z.copy();x[:,:186]=x[:,:186]*np.asarray(STATS['std'])+np.asarray(STATS['mean']);return x.astype(np.float32)
class CommonActionNet(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.base=base_model()
  self.register_buffer('motion_mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('motion_std',torch.tensor(std,dtype=torch.float32))
  self.motion_head=nn.Sequential(nn.Linear(58+192,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,3))
 def movement(self,h,geometry,indices):
  i,a,b=indices.unbind(1);ha=h[i,a];hb=h[i,b]
  return self.motion_head(torch.cat([(geometry-self.motion_mean)/self.motion_std,ha,hb,hb-ha],-1))
def rows():
 out=[]
 for r in read(U/'READY.json')['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path'],allow_pickle=False)
  out.append(dict(r,**{k:z[k].copy() for k in z.files}))
 return out
def build_batch(rr,view='full',reverse_pairs=True):
 b=shared.batch(rr,view);gg=[];ii=[];yy=[]
 for i,r in enumerate(rr):
  raw=physical(r['x_'+view])
  for p in r['pairs']:
   for rev in ([False,True] if reverse_pairs else [False]):
    g,ok,_=motion.features(raw,r['timestamp_us'],p['a'],p['b'],motion.KINDS[p['axis']],reverse=rev)
    if ok:
     a,c=(p['b'],p['a']) if rev else (p['a'],p['b']);gg.append(g);ii.append([i,a,c]);yy.append(2-p['target'] if rev else p['target'])
 b.update(geometry=torch.tensor(np.asarray(gg,np.float32).reshape(-1,58)),indices=torch.tensor(ii,dtype=torch.long).reshape(-1,3),motion_y=torch.tensor(yy,dtype=torch.long))
 return b
