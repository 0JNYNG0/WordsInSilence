"""Reuse the learned common-100 pose encoder inside each candidate, with no outside context."""
from local_sequence_core import *
TQ=W/'transfer_local_timefix'

def encoder():
 rc=load('_transfer_local_req',O/'word100_guide_requirement_learning_20261111/requirement_core.py')
 base=rc.frozen_observer().shared.base
 base.eval();base.requires_grad_(False)
 return base

def normalized_pose(x):
 norm=read(O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json')
 masks=np.c_[np.repeat(x[:,186:237]>.5,2,1),np.repeat(x[:,237:279]>.5,2,1)]
 out=x.copy();out[:,:186]=np.where(masks,(x[:,:186]-np.asarray(norm['mean']))/np.asarray(norm['std']),0)
 return out.astype(np.float32)

def encode_windows(base,pose,q,spans,reverse=False):
 seq=[];local=[];lengths=[]
 for a,b in spans:
  px=pose[a:b+1].copy();qq=q[a:b+1].copy();intervals=px[1:,279].copy()
  if reverse:px=px[::-1].copy();qq=qq[::-1].copy();intervals=intervals[::-1].copy()
  # Channel279 is elapsed seconds since the preceding frame, NOT progress.
  px[:,279]=np.r_[0.,intervals]
  assert np.isfinite(intervals).all() and (intervals>=0).all()
  seq.append(px);local.append(np.c_[qq,px[:,102:186],px[:,237:279]]);lengths.append(len(px))
 lens=torch.tensor(lengths);x=np.zeros((len(seq),max(lengths),280),np.float32)
 for i,s in enumerate(seq):x[i,:len(s)]=s
 with torch.inference_mode():
  # Only the shared observation encoder. No stage/guide/target-word query.
  valid=torch.arange(x.shape[1])[None]<lens[:,None];h=torch.nn.functional.gelu(base.input(torch.tensor(x)))*valid[:,:,None];h=h.transpose(1,2)
  for layer in base.temporal:h=layer(h)*valid[:,None]
  packed=nn.utils.rnn.pack_padded_sequence(h.transpose(1,2),lens,batch_first=True,enforce_sorted=False);out,_=base.context(packed);out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=x.shape[1]);out=out.numpy()
 feats=[]
 for i,n in enumerate(lengths):
  z=np.c_[out[i,:n],local[i]];parts=np.array_split(np.arange(n),3)
  if any(len(p)==0 for p in parts):parts=[np.array([0]),np.arange(n),np.array([n-1])]
  feats.append(np.r_[*[z[p].mean(0) for p in parts],z.std(0)])
 return np.asarray(feats,np.float32)

class TransferHead(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32));self.head=nn.Sequential(nn.Linear(1160,96),nn.GELU(),nn.Linear(96,15))
 def forward(self,x):return self.head((x-self.mean)/self.std).reshape(-1,5,3)
