from pathlib import Path
import sys,json,time,hashlib,importlib.util,copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];J=O/'word100_episode_joint_linkage_20261111';V=J/'proposal_adapter'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
jc=load('_short_joint',J/'joint_core.py');prior,old,bc,lc,le=jc.prior,jc.old,jc.bc,jc.lc,jc.le
read,write,sha=jc.read,jc.write,jc.sha
def event(k,**kw):
 d=dict(time=time.time(),kind=k,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
class MotionNet(prior.Network):
 def __init__(self):
  super().__init__('factorized',read(old.P/'NORMALIZATION.json'));self.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',map_location='cpu',weights_only=True));self.requires_grad_(False)
  for m in [self.motion_conv,self.motion_gru,self.motion_head]:m.requires_grad_(True)
 def predict(self,g):
  # Semantic/word context is removed only for this local movement decision.
  h=torch.zeros((len(g),1,64),device=g.device);ii=torch.stack([torch.arange(len(g),device=g.device),torch.zeros(len(g),device=g.device,dtype=torch.long),torch.zeros(len(g),device=g.device,dtype=torch.long)],1)
  return self.movement(h,g,ii)
 def heads(self):return {k:v for k,v in self.state_dict().items() if k.startswith(('motion_conv.','motion_gru.','motion_head.'))}
def direction(net,x,ts,a,b,axis,moving='R',anchor='L'):
 g,ok,info=prior.geometry(x,ts,a,b,axis,moving,anchor)
 if not ok:return dict(span=[a,b],axis=axis,moving=moving,prediction=-1,confidence=0.,eligible=False,geometry=info)
 with torch.inference_mode():p=net.predict(torch.tensor(g[None])).softmax(-1)[0].numpy()
 k=int(p.argmax());agree=k==1 or np.sign(info['signed_change'])==k-1
 return dict(span=[a,b],axis=axis,moving=moving,prediction=k,confidence=float(p.max()),eligible=bool(ok and agree),geometry=info)
def fused(old_result,new_result):
 out=copy.deepcopy(old_result);out['short_motion']=new_result;out['rescue_used']=False
 # Existing non-hold judgments are not replaced. Unknown observations stay unknown.
 if old_result['eligible'] and old_result['prediction']==1 and new_result['eligible'] and new_result['prediction'] in [0,2] and new_result['confidence']>=.90:
  out.update(prediction=new_result['prediction'],confidence=new_result['confidence'],rescue_used=True)
 return out
def completion_guard(net,e,x,ts):
 a,b,k=e['a'],e['b'],e['axis'];side=e['moving'];anchor='L' if side=='R' else 'R';inside=direction(net,x,ts,a,b,k,side,anchor)
 # About120ms of actual timestamps, not a word-specific frame count.
 end=int(np.searchsorted(ts,ts[b]+120000));end=min(len(ts)-1,max(b+3,end))
 if end-b<3:return dict(state='unobserved_after',semantic_complete=False,inside=inside,after=None)
 after=direction(net,x,ts,b,end,k,side,anchor)
 if not inside['eligible'] or not after['eligible']:state='unobserved_after'
 elif inside['prediction']!=1 and inside['confidence']>=.9 and after['prediction']==inside['prediction'] and after['confidence']>=.9:state='continues_same_direction'
 elif after['prediction']==1 and after['confidence']>=.9:state='local_stop_observed'
 elif inside['prediction']!=1 and after['prediction']==2-inside['prediction'] and after['confidence']>=.9:state='direction_turn_observed'
 else:state='uncertain_boundary'
 # A local stop/turn is evidence about motion, not proof the whole guide was done.
 return dict(state=state,semantic_complete=False,inside=inside,after=after)
