"""Bounded component learner; originals are read-only, no word inputs to forward."""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
U=Path(__file__).resolve().parent;O=U.parent
S=O/'word100_source_component_connector_20261102'
Q=O/'word100_predicate_supervision_20261102'
V=O/'word100_learned_body_relation_20261102'
M=O/'word100_learned_motion_order_20261102'
sys.path.insert(0,str(V));sys.path.insert(0,str(M))
from relation_model import RelationNet,normalized
from motion_model import features

POINT=['both_visible_thumbs_up_with_closed_finger_masses','bilateral_oblique_long_finger_axes']
TEMP=['right_elbow_lowers_hand_remains_raised','right_hand_traverses_left_arm_toward_wrist','bilateral_height_alternation']
CRITICAL=[[30,31,32,33],[2,3,4,5,36,37],[2,3,30,31]]
PROTECTED_POINT={'right_non_thumb_closed':('shape',3),'right_hand_at_left_arm_projection':('relation',2),'bilateral_upper_ends_converged':('relation',1)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 p=U/n;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def load():
 rows=read(S/'DIRECT_TARGETS.json')['rows'];sources=read(Q/'SUPERVISION.json')['sources']
 arrays={};meta={r['clip_id']:r for r in sources}
 for r in sources:
  assert r['person'] in list(range(1,10))+[11,12,13]
  assert sha(r['input'])==r['input_sha256']
  with np.load(r['input'],allow_pickle=False) as a:arrays[r['clip_id']]={k:a[k].copy() for k in a.files}
 return rows,meta,arrays

def current_geometry(a):
 # features() adds arm-relative projections at61/62. Pass no contextual
 # shape/relation features and keep only the63 current-frame quantities.
 n=len(a['x']);g,m=features(a,np.empty((n,0),np.float32),np.empty((n,0),np.float32))
 assert g.shape==m.shape==(n,63)
 return g,m

def temporal_sample(r,a,stats,variant='native'):
 lo,hi=min(r['native_frames']),max(r['native_frames'])
 # Dense native envelope is context, not dense labels. Retain interior gaps.
 ids=np.flatnonzero((a['frame_id']>=lo)&(a['frame_id']<=hi))
 assert np.array_equal(a['frame_id'][ids],np.arange(lo,hi+1))
 g,m=current_geometry(a);g=g[ids].copy();m=m[ids].copy()
 ts=a['timestamp_us'][ids].astype(np.int64);source_frames=a['frame_id'][ids].copy()
 image=np.asarray(a.get('image_available',np.ones(len(a['x']),bool)))[ids].copy()
 order=np.arange(len(ids));play=ts-ts[0];y=r['y'];known=r['mask']
 if variant=='reverse':
  order=order[::-1];play=play[-1]-play[::-1];y=1 if r['target']==TEMP[2] else 0
 elif variant=='still':order[:]=len(ids)//2;y=0
 elif variant=='image_gap':image[:]=False;known=False;y=None
 elif variant in ['speed_075','speed_125']:
  speed=.75 if variant=='speed_075' else 1.25;play=np.rint(play/speed).astype(np.int64)
 elif variant!='native':raise ValueError(variant)
 g=g[order];m=m[order]&image[order,None];image=image[order]
 source_frames=source_frames[order];source_ts=ts[order]
 z=np.where(m,(g-np.asarray(stats['mean']))/np.asarray(stats['std']),0).astype(np.float32)
 x=np.concatenate([z,m],-1).astype(np.float32)
 j=TEMP.index(r['target']);coverage=float(m[:,CRITICAL[j]].all(1).mean())
 usable=bool(len(ids)>=3 and image.all() and coverage>=.8)
 return dict(x=x,record=r,variant=variant,y=y,mask=known,
  native_extent=[int(lo),int(hi)],source_frames=source_frames.tolist(),source_timestamps_us=source_ts.tolist(),
  playback_timestamps_us=play.tolist(),geometry_coverage=coverage,image_coverage=float(image.mean()),
  usable=usable,input_mode='current_frame_geometry_control',actual_transformed_rgb=False,
  independent_case=False)

def build_temporal(rows,arrays,stats):
 out=[]
 for r in rows:
  if r['target'] not in TEMP:continue
  variants=['native']
  # Controls have reviewed direction/alternation support. Never synthesize
  # negative truth from an unknown, interval endpoint or a word identity.
  if r['kind']=='sequence' and r['mask'] and r['y']==1:variants+=['reverse','still','image_gap','speed_075','speed_125']
  for v in variants:out.append(temporal_sample(r,arrays[r['clip_id']],stats,v))
 return out

def point_samples(rows,arrays):
 stats=read(V/'NORMALIZATION.json');out=[]
 normalized_cache={cid:normalized(a,stats) for cid,a in arrays.items()}
 for r in rows:
  if r['target'] not in POINT:continue
  assert r['kind']=='point' and len(r['native_frames'])==1
  a=arrays[r['clip_id']];f=r['native_frames'][0];ids=a['frame_id'];where=np.flatnonzero(ids==f);assert len(where)==1
  center=int(where[0]);ix=np.arange(center-6,center+7);valid=bool(ix[0]>=0 and ix[-1]<len(ids))
  if valid:valid=bool(np.array_equal(ids[ix],np.arange(f-6,f+7)))
  p,z=normalized_cache[r['clip_id']]
  av=np.asarray(a.get('image_available',np.ones(len(ids),bool)))
  valid=valid and bool(av[center])
  out.append(dict(record=r,pose=p[ix] if valid else np.zeros((13,138),np.float32),
   rgb=z[ix] if valid else np.zeros((13,768),np.float32),usable=valid,y=r['y'],mask=r['mask']))
 return out

class PointNet(RelationNet):
 def __init__(self):
  super().__init__(True)
  self.load_state_dict(torch.load(V/'runs/rgb_geometry_42/model.pt',map_location='cpu',weights_only=True))
  for p in self.parameters():p.requires_grad_(False)
  self.out=nn.Linear(32,2)
  # Bounded finetuning of the copied final convolution; originals never mutate.
  for p in self.temporal[2].parameters():p.requires_grad_(True)

class TemporalNet(nn.Module):
 def __init__(self,condition):
  super().__init__();assert condition in ['pooled','ordered'];self.condition=condition
  self.embed=nn.Sequential(nn.Linear(126,32),nn.GELU())
  self.out=nn.Linear(64,3)
  if condition=='ordered':self.temporal=nn.GRU(32,32,batch_first=True,bidirectional=True)
  else:self.temporal=nn.Sequential(nn.Linear(32,64),nn.GELU())
 def forward(self,x,lengths):
  h=self.embed(x)
  if self.condition=='ordered':
   _,last=self.temporal(pack_padded_sequence(h,lengths.cpu(),batch_first=True,enforce_sorted=False));v=torch.cat([last[0],last[1]],-1)
  else:
   mask=torch.arange(x.shape[1])[None,:]<lengths[:,None];v=self.temporal((h*mask[:,:,None]).sum(1)/lengths[:,None])
  return self.out(v)

def batch(samples):
 l=torch.tensor([len(s['x']) for s in samples]);x=torch.zeros(len(samples),int(l.max()),126)
 for i,s in enumerate(samples):x[i,:l[i]]=torch.from_numpy(s['x'])
 return x,l

def grouped_weights(samples,targets):
 # Equal target -> clip -> class -> representations, preventing duplicate
 # point/interval views and synthetic variants from becoming new people.
 groups={}
 for i,s in enumerate(samples):
  r=s['record'];key=(r['target'],r['clip_id'],int(s['y']))
  groups.setdefault(key,[]).append(i)
 weights=np.zeros(len(samples),np.float32)
 for target in targets:
  clips=sorted({c for t,c,_ in groups if t==target})
  for cid in clips:
   classes=[v for t,c,v in groups if t==target and c==cid]
   for v in classes:
    ix=groups[(target,cid,v)];weights[ix]=1/(len(targets)*len(clips)*len(classes)*len(ix))
 assert np.isclose(weights.sum(),1),weights.sum()
 return torch.from_numpy(weights)

def metadata(s):
 return {k:v for k,v in s.items() if k not in ['x','pose','rgb']}
