"""Frozen common observation learner + separately supervised temporal evidence heads."""
from pathlib import Path
import sys,importlib.util,json,time,hashlib,types
import numpy as np
import torch
from torch import nn
P=Path(__file__).resolve().parent;O=P.parent;B=O/'word100_shared_motion_learning_20261109';D=O/'word100_action_feedback_demo_20261108';S=O/'word100_gru_semantic_finetune_20261109';M=O/'word100_motion_components_20261109'
sys.dont_write_bytecode=True
sys.path.insert(0,str(B))
import core as common
read=common.read;write=common.write;sha=common.sha

def load_module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
motion=load_module('_separate_motion',M/'motion.py')
positive=load_module('_separate_positive',O/'word100_positive_direction_20261109/positive_evidence.py')
coverage=load_module('_separate_coverage',M/'component_coverage.py')
old_task=sys.modules.get('task');stub=types.ModuleType('task');stub.U=D;stub.read=read;sys.modules['task']=stub
evidence=load_module('_separate_evidence',D/'evidence.py')
if old_task is None:del sys.modules['task']
else:sys.modules['task']=old_task
G=read(common.J/'feedback_viewfix/FEEDBACK_NORMALIZATION.json');GROUPS={k:i for i,ls in enumerate(read(S/'FROZEN.json')['semantic_stage_groups']) for k in ls}
CHECKPOINT=B/'runs/shared100_42/model.pt'

def event(kind,**kw):
 d=dict(time=time.time(),kind=kind,**kw)
 with (P/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)

def geometry(raw):
 # Existing body-normalized points, 9 body +2 palm centres; no word/person/time position.
 p=raw[:,:102].reshape(-1,51,2);v=raw[:,186:237]>.5
 coords=[p[:,:9]];valid=[v[:,:9]]
 for base in [9,30]:
  ids=[base,base+5,base+9,base+13,base+17];coords.append(p[:,ids].mean(1)[:,None]);valid.append(v[:,ids].all(1)[:,None])
 q=np.concatenate(coords,1);vm=np.concatenate(valid,1);q=np.where(vm[:,:,None],q,0);vel=np.zeros_like(q);vel[1:]=q[1:]-q[:-1];vv=vm.copy();vv[0]=False;vv[1:]&=vm[:-1];vel=np.where(vv[:,:,None],vel,0)
 # Values are never reconstructed across a missing interval.
 return np.concatenate([q.reshape(len(q),22),vm.astype(float),vel.reshape(len(q),22),vv.astype(float)],1).astype(np.float32)

class Heads(nn.Module):
 def __init__(self,relations):
  super().__init__();self.relations=relations;self.register_buffer('guide_words',torch.tensor([-1]+G['guide_words']))
  self.temporal=nn.Sequential(nn.Conv1d(137 if relations else 71,48,3,padding=1),nn.GELU(),nn.Conv1d(48,48,3,padding=1),nn.GELU())
  self.phase=nn.Linear(48,14);self.difference=nn.Sequential(nn.Linear(78,48),nn.GELU(),nn.Linear(48,7))
 def forward(self,h,p,g,word,valid,expected):
  z=torch.cat([h,p],-1);t=torch.cat([z,g],-1) if self.relations else z;t=t.transpose(1,2)*valid[:,None,:]
  for layer in self.temporal:t=layer(t)*valid[:,None,:]
  phase=self.phase(t.transpose(1,2));allowed=(self.guide_words[None,:]==word[:,None])|(self.guide_words[None,:]<0);phase=phase.masked_fill(~allowed[:,None,:],-1e4)
  dif=self.difference(torch.cat([z,expected],-1));return phase,dif

def batch(rows,view):
 T=max(len(r['frame_id']) for r in rows);n=len(rows);out={}
 for key,width in [('h',64),('p',7),('g',66),('y',7),('ym',7)]:
  out[key]=np.zeros((n,T,width),np.float32)
  for i,r in enumerate(rows):out[key][i,:len(r['frame_id'])]=r[key+'_'+view] if key in ('h','p','g') else r[key]
 out['sy']=np.full((n,T),-1,np.int64);out['sw']=np.zeros((n,T),np.float32);out['valid']=np.zeros((n,T),bool)
 for i,r in enumerate(rows):
  l=len(r['frame_id']);out['sy'][i,:l]=r['stage_y'];out['sw'][i,:l]=r['stage_w'];out['valid'][i,:l]=True
 out={k:torch.from_numpy(v) for k,v in out.items()};out['word']=torch.tensor([r['word'] for r in rows]);return out

def load_rows():
 out=[]
 for r in read(P/'PREPARED.json')['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path'],allow_pickle=False);out.append(dict(r,**{k:z[k].copy() for k in z.files}))
 return out

def decide(word,phase,d0,d1,attrs,raw,ts,fid):
 out=evidence.assess(word,phase,d0,d1,raw,ts,fid);out=positive.refine(out,attrs,raw,ts,fid,phase);return coverage.apply_coverage(out)
