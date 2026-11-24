"""Bounded context around an observed candidate. No expected word sequence input."""
from transfer_local_timefix_core import *
R=W/'framing_boundary_candidate'
CONDITIONS=['crop_local','boundary_only','crop_boundary']
HALO_US=133333 # Existing temporal feature window, not fitted on this comparison.

def encode_context(base,pose,q,ts,spans,reverse=False,halo_us=HALO_US):
 seq=[];local=[];lengths=[];cores=[]
 for a,b in spans:
  lo=int(np.searchsorted(ts,ts[a]-halo_us));hi=int(np.searchsorted(ts,ts[b]+halo_us,side='right')-1)
  px=pose[lo:hi+1].copy();qq=q[lo:hi+1].copy();intervals=px[1:,279].copy();ca,cb=a-lo,b-lo
  if reverse:
   px=px[::-1].copy();qq=qq[::-1].copy();intervals=intervals[::-1].copy();ca,cb=hi-b,hi-a
  px[:,279]=np.r_[0.,intervals]
  assert np.isfinite(intervals).all() and (intervals>=0).all()
  seq.append(px);local.append(np.c_[qq,px[:,102:186],px[:,237:279]]);lengths.append(len(px));cores.append((ca,cb))
 lens=torch.tensor(lengths);x=np.zeros((len(seq),max(lengths),280),np.float32)
 for i,s in enumerate(seq):x[i,:len(s)]=s
 with torch.inference_mode():
  valid=torch.arange(x.shape[1])[None]<lens[:,None];h=torch.nn.functional.gelu(base.input(torch.tensor(x)))*valid[:,:,None];h=h.transpose(1,2)
  for layer in base.temporal:h=layer(h)*valid[:,None]
  packed=nn.utils.rnn.pack_padded_sequence(h.transpose(1,2),lens,batch_first=True,enforce_sorted=False);out,_=base.context(packed);out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=x.shape[1]);out=out.numpy()
 feats=[]
 for i,n in enumerate(lengths):
  a,b=cores[i];z=np.c_[out[i,:n],local[i]][a:b+1];parts=np.array_split(np.arange(len(z)),3)
  if any(len(p)==0 for p in parts):parts=[np.array([0]),np.arange(len(z)),np.array([len(z)-1])]
  feats.append(np.r_[*[z[p].mean(0) for p in parts],z.std(0)])
 return np.asarray(feats,np.float32)

def balanced_weights(y,mask,rows,train):
 yy=y[train];mm=mask[train];rr=[r for r,t in zip(rows,train) if t];w=np.zeros_like(mm)
 old=np.load(Q/'TRAIN_WEIGHTS.npz')['weights'];base_n=len(old);w[:base_n]=old
 assert all(r['view']!='actual_crop' for r in rr[:base_n])
 for j in range(5):
  for h in range(3):
   for v in [0,1]:
    take=(mm[:,j,h]>0)&(yy[:,j,h]==v);clips=sorted({r['clip_id'] for r,t in zip(rr,take) if t})
    for cid in clips:
     orig=take&np.array([r['clip_id']==cid and r['view']!='actual_crop' for r in rr]);added=take&np.array([r['clip_id']==cid and r['view']=='actual_crop' for r in rr])
     if added.any():
      mass=w[orig,j,h].sum();assert mass>0
      w[orig,j,h]*=2/3;w[added,j,h]=mass/(3*added.sum())
 assert np.isclose(w.sum(),1)
 return w
