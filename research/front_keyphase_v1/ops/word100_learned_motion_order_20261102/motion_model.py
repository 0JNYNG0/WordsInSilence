from pathlib import Path
import json,numpy as np,torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
M=Path(__file__).resolve().parent;O=M.parent;P=O/'word100_learned_guide_feedback_20261102';V=O/'word100_learned_body_relation_20261102';U=O/'word100_learned_feature_retention_20261102';OLD=O/'word100_multipeople_live_feedback_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def features(a,shape,relation):
 x=np.asarray(a['x']);mask=np.asarray(a['mask'],bool);n=len(x)
 # All current-frame quantities, not precomputed velocity/direction (indices61+).
 static=x[:,:61];sm=mask[:,:61]&np.isfinite(static)
 elbow=x[:,4:6];wrist=x[:,2:4];palm=x[:,36:38];v=wrist-elbow;d=palm-elbow;norm=(v*v).sum(1)
 good=mask[:,[2,3,4,5,36,37]].all(1)&np.isfinite(norm)&(norm>.0025)
 along=(d*v).sum(1)/np.maximum(norm,1e-6);side=(d[:,0]*v[:,1]-d[:,1]*v[:,0])/np.maximum(norm,1e-6)
 z=np.column_stack([static,along,side,shape,relation]);valid=np.column_stack([sm,good,good,np.isfinite(shape),np.isfinite(relation)])
 available=np.asarray(a.get('image_available',np.ones(n,bool)),bool);valid&=available[:,None]
 return np.where(valid,z,0).astype('float32'),valid
def public_arrays(row):
 a=dict(np.load(row['input']));cid=row['clip_id'];d=OLD/('bank_predictions' if row['role']=='fit' else 'baseline_probabilities')
 s=np.load(d/f'{cid}.npz');r=np.load(V/'runs/rgb_geometry_42/predictions'/f'{cid}.npz')
 assert np.array_equal(a['frame_id'],s['frame_id']) and np.array_equal(a['frame_id'],r['frame_id'])
 assert np.array_equal(a['timestamp_us'],r['timestamp_us'])
 z,m=features(a,s['probabilities'],r['probability']);return a,z,m
def normalize(x,m,stats):
 z=np.where(m,(x-np.array(stats['mean']))/np.array(stats['std']),0)
 return np.concatenate([z,m],-1).astype('float32')
class MotionNet(nn.Module):
 def __init__(self,condition,dim=146):
  super().__init__();self.condition=condition;self.embed=nn.Sequential(nn.Linear(dim,32),nn.GELU())
  if condition=='ordered_gru':self.temporal=nn.GRU(32,32,batch_first=True,bidirectional=True)
  else:self.temporal=nn.Sequential(nn.Linear(32,64),nn.GELU())
  self.out=nn.Linear(64,6)
 def forward(self,x,lengths):
  h=self.embed(x)
  if self.condition=='ordered_gru':
   _,last=self.temporal(pack_padded_sequence(h,lengths.cpu(),batch_first=True,enforce_sorted=False));v=torch.cat([last[0],last[1]],-1)
  else:
   mask=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None];v=self.temporal((h*mask[:,:,None]).sum(1)/lengths[:,None])
  return self.out(v)
def batch(sequences):
 lengths=torch.tensor([len(s) for s in sequences]);x=torch.zeros(len(sequences),int(lengths.max()),sequences[0].shape[1])
 for i,s in enumerate(sequences):x[i,:len(s)]=torch.from_numpy(np.asarray(s,dtype='float32'))
 return x,lengths
def predict(model,sequences):
 result=[]
 with torch.inference_mode():
  for i in range(0,len(sequences),64):
   x,l=batch(sequences[i:i+64]);result.append(model(x,l).sigmoid().numpy())
 return np.concatenate(result)
