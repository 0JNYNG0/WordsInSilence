"""Isolated, descriptor-conditioned evidence comparator. No word IDs in the model.

Outputs feature support/contradiction, never whole-sign educational correctness.
The frozen 100-word Conv+GRU observer and deployed models are not modified.
"""
from pathlib import Path
import sys, json, hashlib, importlib.util
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode = True
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
B=O/'word100_broad_factorized_learning_20261110'
S=O/'word100_shared_motion_learning_20261109'
R=O/'word100_candidate_role_learning_20261111'
V=O/'word100_relative_position_binding_20261111'
Q=O/'word100_completion_boundary_targets_20261111'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); t.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
prior=load('_requirement_broad',B/'core.py')
CARDS=read(prior.D/'CARDS.json')['words']
FAMILIES=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
VECTORS=np.asarray(prior.STATS['guide_vectors'],np.float32)
FEATURE_NAMES=['right_index_only','right_four_open','left_four_open','right_four_closed','left_four_closed','hands_near_projection','hands_overlap_projection']

def requirement(task,feature=0,expected=1,axis=0,direction=1,guides=None):
 q=np.zeros(76,np.float32); q[{'shape':0,'direction':1,'order':2}[task]]=1
 if task=='shape':q[3+feature]=1;q[10+expected]=1
 if task=='direction':q[12+axis]=1;q[17+direction]=1
 if task=='order':q[20:48]=VECTORS[guides[0]];q[48:76]=VECTORS[guides[1]]
 return q

def observation(h,a,b,g=None):
 a,b=sorted((int(a),int(b))); middle=h[a:b+1].mean(0)
 return np.r_[h[a],middle,h[b],np.zeros(58) if g is None else g].astype(np.float32)

def logit(p):
 p=np.clip(p,1e-5,1-1e-5);return float(np.log(p/(1-p)))
def decide(p,eligible=True,association=True):
 if not eligible:return 'observation_unknown'
 if not association:return 'association_unknown'
 return 'supported' if p>=.65 else 'difference' if p<=.35 else 'uncertain'

def direction_verdict(probabilities,desired,eligible=True):
 """A hold or weak motion is not proof of moving in the wrong direction.

 Three independent compatibility probabilities need not sum to one. Require
 explicit opposing evidence, not merely a low desired-direction score.
 """
 if not eligible:return 'observation_unknown'
 p=np.asarray(probabilities)
 if desired==1:
  if p[1]>=.65 and max(p[0],p[2])<.65:return 'supported'
  return 'uncertain' # bodily sway is not automatically a wrong sign
 opposite=2-desired
 if p[desired]>=.65 and p[opposite]<.65 and p[1]<.65:return 'supported'
 if p[opposite]>=.65 and p[desired]<.35 and p[1]<.65:return 'opposite_direction'
 if p[1]>=.65:return 'movement_not_established'
 return 'uncertain'

class RequirementAdapter(nn.Module):
 def __init__(self,mean,std):
  super().__init__(); self.register_buffer('mean',torch.tensor(mean));self.register_buffer('std',torch.tensor(std))
  self.observe=nn.Sequential(nn.Linear(250,64),nn.GELU(),nn.Linear(64,32),nn.GELU())
  self.request=nn.Sequential(nn.Linear(76,32),nn.GELU())
  self.compare=nn.Sequential(nn.Linear(97,48),nn.GELU(),nn.Linear(48,1))
  nn.init.zeros_(self.compare[-1].weight);nn.init.zeros_(self.compare[-1].bias)
 def forward(self,ob,req,base):
  z=self.observe((ob-self.mean)/self.std);q=self.request(req)
  delta=self.compare(torch.cat([z,q,z*q,base[:,None]],-1)).squeeze(-1)
  return base+2*torch.tanh(delta) # bounded residual; keep established strong evidence

def public_rows():
 """Read only label-bearing public entries, not 899 arrays or any private roots."""
 extra=read(B/'ADDITIONAL_LABELS.json'); movement={r['clip_id']:r['pairs'] for r in read(prior.M/'READY.json')['rows']}
 rows=[]
 for r in read(S/'READY.json')['rows']:
  if not r['has_labels']:continue
  assert r['clip_id'].startswith('NIA_SL_WORD') and '_REAL' in r['clip_id']
  assert sha(r['path'])==r['sha256']
  z=np.load(r['path'],allow_pickle=False);r=dict(r,**{k:z[k].copy() for k in z.files})
  r['pairs']=list(movement.get(r['clip_id'],r['pairs']))+extra['motion'].get(r['clip_id'],[])
  r['orders']=list(r['orders'])+extra.get('orders',{}).get(r['clip_id'],[])
  for a in extra['shape'].get(r['clip_id'],[]):
   idx=int(np.flatnonzero(r['frame_id']==a['frame'])[0])
   for j,y in a['features'].items():
    j=int(j);assert not r['ym'][idx,j] or r['y'][idx,j]==y
    r['y'][idx,j]=y;r['ym'][idx,j]=1;r['direct_ym'][idx,j]=1
  rows.append(r)
 return rows

def cached_h(r,view='full'):
 p=O/'word100_observed_event_boundaries_20261110/cache'/f"{r['clip_id']}.npz"
 if p.exists():
  z=np.load(p);return z['frozen_h_'+view].copy(),p
 key=f"{r['clip_id']}_{view}_normal.npz";p=R/'train_cache'/key
 z=np.load(p);return z['z'][:,392:456].copy(),p

def frozen_observer():
 n=prior.Network('coupled',read(B/'NORMALIZATION.json'));n.load_state_dict(torch.load(B/'runs/coupled_42/model.pt',map_location='cpu',weights_only=True));n.eval();n.requires_grad_(False);return n

def guide_links(r,task,a,b,feature=None,axis=None):
 stage=set(int(s) for s in r['stage_gold'][a:b+1] if s>=0)
 hits=[]
 for c in CARDS.get(r['word_id'],[]):
  gs={s for g in c['groups'] for s in g}
  if not stage.intersection(gs):continue
  if task=='shape' and str(feature) not in c.get('features',{}):continue
  if task=='direction':
   f=next(f for f in FAMILIES[r['word_id']] if f['id']==c['id'])
   if f['axis']!=axis:continue
  hits.append(c['id'])
 return hits
