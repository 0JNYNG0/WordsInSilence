"""Explicit-axis DTW utilities; preserve every original time point."""
import numpy as np
def dtw(query,reference):
 q=np.asarray(query,np.float64);r=np.asarray(reference,np.float64)
 assert q.ndim==r.ndim==2 and q.shape[1]==r.shape[1] and np.isfinite(q).all() and np.isfinite(r).all()
 cost=((q[:,None,:]-r[None,:,:])**2).mean(-1)
 n,m=cost.shape;D=np.full((n+1,m+1),np.inf);D[0,0]=0;prev=np.zeros((n,m),np.int8)
 for i in range(n):
  for j in range(m):
   # Deterministic diagonal priority. No free skipping of missing intervals.
   a,b,c=D[i,j],D[i,j+1],D[i+1,j]
   if a<=b and a<=c:k=0;best=a
   elif b<=c:k=1;best=b
   else:k=2;best=c
   D[i+1,j+1]=cost[i,j]+best;prev[i,j]=k
 path=[];i=n-1;j=m-1
 while True:
  path.append((i,j))
  if i==0 and j==0:break
  k=prev[i,j]
  if k==0:i-=1;j-=1
  elif k==1:i-=1
  else:j-=1
  assert i>=0 and j>=0
 path=np.array(path[::-1],np.int64)
 assert set(path[:,0])==set(range(n)) and set(path[:,1])==set(range(m))
 return dict(path=path,cost=cost,mean_cost=float(cost[path[:,0],path[:,1]].mean()))
def representative(path,index):
 values=np.asarray(path)[np.asarray(path)[:,0]==index,1]
 return None if not len(values) else int(np.floor(np.median(values)+.5))
def stage_at(frame,guides,word):
 matches=[g['canonical_stage_id'] for g in guides if g['word_id']==word and g['frames'][0]<=frame<=g['frames'][1]]
 return matches[0] if len(matches)==1 else None
