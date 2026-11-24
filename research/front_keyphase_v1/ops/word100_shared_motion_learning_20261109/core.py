from pathlib import Path
import json,hashlib,sys,time,importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;J=O/'word100_rtmw_joint_learning_20261108';V=J/'feedback_viewfix';S=O/'word100_gru_semantic_finetune_20261109';D=O/'word100_action_feedback_demo_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def event(kind,**kw):
 v=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
 print(json.dumps(v,ensure_ascii=False),flush=True)
spec=importlib.util.spec_from_file_location('_shared_original_model',J/'feedback_model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
PoseGuideNet=module.PoseGuideNet
STATS=read(V/'FEEDBACK_NORMALIZATION.json');WORDS=read(J/'FROZEN.json')['feedback']['words'];CLASSES=read(J/'FROZEN.json')['classes'];BASE=V/'feedback_runs/pose_transfer_42/model.pt'
def normalize(x):
 z=x.copy();m=np.concatenate([np.repeat(x[:,186:237]>.5,2,1),np.repeat(x[:,237:279]>.5,2,1)],1)
 z[:,:186]=np.where(m,(x[:,:186]-np.asarray(STATS['mean']))/np.asarray(STATS['std']),0);return z.astype(np.float32)
def available(x):
 v=x[:,186:237]>.5;cols=[]
 for j in range(7):
  pts=[9,30] if j in [5,6] else ([9]+list(range(14,30)) if j in [2,4] else [30]+list(range(35,51)))
  cols.append(v[:,pts].all(1))
 return np.stack(cols,1)
def mask_hand(x,start,stop,side):
 z=x.copy();a,b=(9,30) if side==0 else (30,51);l0,l1=(102,144) if side==0 else (144,186);m0,m1=(237,258) if side==0 else (258,279)
 z[start:stop,a*2:b*2]=0;z[start:stop,l0:l1]=0;z[start:stop,186+a:186+b]=0;z[start:stop,m0:m1]=0;return z
class SharedNet(nn.Module):
 def __init__(self):
  super().__init__();self.base=PoseGuideNet(STATS['guide_vectors'],STATS['guide_words'],STATS['prototype_lengths'])
  self.word_head=nn.Linear(128,100)
 def forward(self,x,lengths,word):
  o=self.base(x,lengths,word);h=o['latent'];valid=o['valid'];avg=(h*valid[:,:,None]).sum(1)/lengths[:,None];mx=h.masked_fill(~valid[:,:,None],-1e4).max(1).values
  o['word_logits']=self.word_head(torch.cat([avg,mx],-1));return o
def load_rows():
 rows=[]
 for m in read(U/'READY.json')['rows']:
  assert sha(m['path'])==m['sha256'];z=np.load(m['path'],allow_pickle=False);rows.append(dict(m,**{k:z[k].copy() for k in z.files}))
 return rows
def batch(rows,view='full',mask=False,rng=None):
 T=max(len(r['x_full']) for r in rows);B=len(rows);out={}
 for k,d in [('x',280),('y',7),('ym',7)]:out[k]=np.zeros((B,T,d),np.float32)
 out['sy']=np.full((B,T),-1,np.int64);out['sw']=np.zeros((B,T),np.float32);out['ci']=np.full((B,T),-1,np.int64)
 for i,r in enumerate(rows):
  x=r['x_'+view].copy();n=len(x)
  # Missing sensor samples, never a new normal movement or a synthetic pose target.
  if mask and n>8:
   start=int(rng.integers(2,n-3));x=mask_hand(x,start,start+2,int(rng.integers(0,2)))
  out['x'][i,:n]=x
  for k in ['y','ym','sy','sw','ci']:out[k][i,:n]=r[k]
  out['ym'][i,:n]*=available(r['x_'+view])
 out={k:torch.from_numpy(v) for k,v in out.items()};out['lengths']=torch.tensor([len(r['x_full']) for r in rows]);out['word']=torch.tensor([max(0,r['word']) for r in rows]);out['class']=torch.tensor([r['class_index'] for r in rows]);return out
def supervised_loss(model,b,pw):
 o=model(b['x'],b['lengths'],b['word']);m=b['ym'];a=(F.binary_cross_entropy_with_logits(o['attributes'],b['y'],reduction='none',pos_weight=pw)*m).sum()/m.sum().clamp_min(1)
 ce=F.cross_entropy(o['stage'].transpose(1,2),b['sy'],ignore_index=-1,reduction='none');st=(ce*b['sw']).sum()/b['sw'].sum().clamp_min(1)
 valid=(b['ci']>=0)&o['valid'];proto=model.base.prototype[b['word'][:,None],b['ci'].clamp_min(0)]
 al=((o['latent']-proto).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
 return a+.35*st+.1*al,dict(attribute=float(a.detach()),stage=float(st.detach()),prototype=float(al.detach()))
