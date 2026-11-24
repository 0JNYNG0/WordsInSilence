"""Shared, word-blind geometry and evidence-preserving sequence comparison."""
import numpy as np
from collections import defaultdict
GROUPS=['L_shape','R_shape','L_location','R_location','L_axis','R_axis','L_arm','R_arm','hands_relation','L_to_R_forearm','R_to_L_forearm','L_direction','R_direction']
NAMES={'L_shape':'왼손 손가락 투영 각도','R_shape':'오른손 손가락 투영 각도','L_location':'왼손의 몸 기준 위치','R_location':'오른손의 몸 기준 위치','L_axis':'왼손의 화면 방향','R_axis':'오른손의 화면 방향','L_arm':'왼팔 배치·구부림','R_arm':'오른팔 배치·구부림','hands_relation':'두 손 사이의 배치','L_to_R_forearm':'왼손과 오른쪽 전완의 관계','R_to_L_forearm':'오른손과 왼쪽 전완의 관계','L_direction':'왼손 이동 방향','R_direction':'오른손 이동 방향'}
SUPPORT=.75
def features(a):
 x=np.asarray(a['x'],float);m=np.asarray(a['mask'],bool)&np.isfinite(x);out={}
 for hand,o in [('L',0),('R',28)]:
  for name,ids,div in [('shape',list(range(o+11,o+25)),1),('location',[o,o+1,o+8,o+9],1),('axis',[o+25,o+26],2)]:
   out[hand+'_'+name]=(np.where(m[:,ids],x[:,ids]/div,0),m[:,ids].copy())
  v=x[:,o+2:o+4]-x[:,o+4:o+6];length=np.linalg.norm(v,axis=1);ok=m[:,o+2:o+6].all(1)&(length>1e-8)
  axis=np.divide(v,length[:,None],out=np.zeros_like(v),where=ok[:,None])/2
  ids=[o+4,o+5,o+10];out[hand+'_arm']=(np.concatenate([np.where(m[:,ids],x[:,ids],0),axis,np.where(ok,length,0)[:,None]],1),np.concatenate([m[:,ids],np.tile(ok[:,None],(1,3))],1))
  other=28 if o==0 else 0;origin=x[:,other+4:other+6];direction=x[:,other+2:other+4]-origin;disp=x[:,o:o+2]-origin;norm2=(direction*direction).sum(1)
  good=m[:,o:o+2].all(1)&m[:,other+2:other+6].all(1)&(norm2>1e-8)
  projection=np.divide((disp*direction).sum(1),norm2,out=np.zeros(len(x)),where=good)
  cross=np.divide(direction[:,0]*disp[:,1]-direction[:,1]*disp[:,0],norm2,out=np.zeros(len(x)),where=good)
  out[hand+'_to_'+('R' if o==0 else 'L')+'_forearm']=(np.stack([projection,cross],1),np.tile(good[:,None],(1,2)))
  ids=[63,64] if o==0 else [67,68];out[hand+'_direction']=(np.where(m[:,ids],x[:,ids]/2,0),m[:,ids].copy())
 ids=[56,57,58];out['hands_relation']=(np.where(m[:,ids],x[:,ids],0),m[:,ids].copy())
 return out
def alignment_cost(q,r):
 costs=[]
 for group in GROUPS:
  x,qm=q[group];y,rm=r[group];expected=rm[None];shared=qm[:,None]&expected;den=rm.sum(1)[None]
  delta=np.minimum(np.abs(x[:,None]-y[None]),1)
  cost=np.divide(np.where(shared,delta,0).sum(2)+(expected&~qm[:,None]).sum(2),den,out=np.full((len(x),len(y)),.5),where=den>0)
  costs.append(cost)
 return np.mean(costs,axis=0)
def dtw(cost):
 n,m=cost.shape;dist=np.full((n+1,m+1),np.inf);dist[0,0]=0;back=np.full((n,m),-1,np.int8)
 for i in range(n):
  for j in range(m):
   if abs(i/max(n-1,1)-j/max(m-1,1))>.25:continue
   vals=(dist[i,j],dist[i,j+1],dist[i+1,j]);k=min(range(3),key=vals.__getitem__);dist[i+1,j+1]=vals[k]+cost[i,j];back[i,j]=k
 assert np.isfinite(dist[n,m])
 path=[];i=n-1;j=m-1
 while i>=0 and j>=0:
  path.append((i,j));k=back[i,j]
  if k==0:i-=1;j-=1
  elif k==1:i-=1
  elif k==2:j-=1
  else:raise ValueError('broken path')
 path=np.asarray(path[::-1],int);assert set(path[:,0])==set(range(n)) and set(path[:,1])==set(range(m))
 return path
def observed_summary(q,r,path,ref_ids=None):
 if ref_ids is not None:path=path[np.isin(path[:,1],ref_ids)]
 qi,ri=path.T;result={}
 for group in GROUPS:
  x,qm=q[group];y,rm=r[group];ref_scope=np.arange(len(y)) if ref_ids is None else np.asarray(ref_ids)
  expected=int(rm[ref_scope].sum());compared=np.zeros_like(rm);maxima=np.zeros_like(y);missing=0
  for i,j in path:
   shared=qm[i]&rm[j];maxima[j]=np.maximum(maxima[j],np.where(shared,np.minimum(np.abs(x[i]-y[j]),1),0));compared[j]|=shared;missing+=int((rm[j]&~qm[i]).sum())
  count=int(compared[ref_scope].sum());nref=len(ref_scope);available_frames=int(rm[ref_scope].any(1).sum());query_frames=np.unique(qi)
  distinct_shared_query=len({int(i) for i,j in path if (qm[i]&rm[j]).any()})
  result[group]=dict(expected=expected,compared=count,reference_possible=nref*y.shape[1],reference_unobserved=nref*y.shape[1]-expected,path_missing_cells=missing,coverage=count/expected if expected else None,mean_difference=float(maxima[ref_scope].sum()/count) if count else None,peak_difference=float(maxima[ref_scope].max()) if count else None,sufficient=bool(expected and count/expected>=SUPPORT and missing==0 and available_frames>=2 and distinct_shared_query>=2),distinct_shared_query_frames=distinct_shared_query,query_unobserved_frames=query_frames[~qm[query_frames].any(1)].tolist())
 return result
def temporal_changes(q,r,path,segments):
 out=[]
 for s in segments:
  thirds=np.array_split(np.arange(s['core'][0],s['core'][1]+1),3);entry={'segment_id':s['segment_id'],'role':s['role'],'core':s['core'],'groups':{}}
  if any(len(t)==0 for t in thirds):out.append(entry);continue
  for g in GROUPS:
   x,qm=q[g];y,rm=r[g];rp=[];qp=[];oks=[]
   for third in thirds:
    pp=path[np.isin(path[:,1],third)];a,b=pp.T;shared=qm[a]&rm[b];den=shared.sum(0);good=den>0
    qp.append(np.divide(np.where(shared,x[a],0).sum(0),den,out=np.zeros(x.shape[1]),where=good));rp.append(np.divide(np.where(shared,y[b],0).sum(0),den,out=np.zeros(x.shape[1]),where=good));oks.append(good)
   valid=np.logical_and.reduce(oks)
   diff=np.abs(np.diff(np.array(qp),axis=0)-np.diff(np.array(rp),axis=0))
   entry['groups'][g]=dict(dimensions=int(valid.sum()),difference=float(np.minimum(diff[:,valid],1).mean()) if valid.any() else None,scope='change under given DTW path; not proof of semantic order')
  out.append(entry)
 return out
def free_order(q,r,segments):
 """Unconstrained original-query window retrieval; does not impose reference order."""
 n=len(next(iter(q.values()))[0]);anchors=[]
 for s in segments:
  ids=np.arange(s['core'][0],s['core'][1]+1);template={}
  for g in GROUPS:
   y,m=r[g];count=m[ids].sum(0);ok=count>=max(1,int(np.ceil(.75*len(ids))))
   template[g]=(np.divide(np.where(m[ids],y[ids],0).sum(0),count,out=np.zeros(y.shape[1]),where=count>0),ok)
  costs=[];supports=[]
  for i in range(n):
   window=np.arange(max(0,i-2),min(n,i+3));gs=[];countall=0;exall=0
   for g in GROUPS:
    x,m=q[g];y,expected=template[g]
    if not expected.any():continue
    shared=m[window]&expected;diff=np.minimum(np.abs(x[window]-y),1);ex=int(expected.sum()*len(window));cnt=int(shared.sum());gs.append((np.where(shared,diff,0).sum()+(ex-cnt))/ex);countall+=cnt;exall+=ex
   costs.append(float(np.mean(gs)) if gs else 1.);supports.append(countall/exall if exall else 0.)
  best=int(np.argmin(costs));ties=np.flatnonzero(np.isclose(costs,costs[best],atol=1e-9,rtol=0)).tolist()
  ranked=np.argsort(costs,kind='stable');second=int(ranked[1]) if len(ranked)>1 else best
  anchors.append(dict(segment_id=s['segment_id'],reference_core=s['core'],query_center=best,query_window=[max(0,best-2),min(n-1,best+2)],cost=costs[best],coverage=supports[best],ties=ties,unique=len(ties)==1,supported=supports[best]>=SUPPORT,second_candidate=dict(center=second,cost=costs[second],gap=costs[second]-costs[best]),five_nearest_candidates=[dict(center=int(j),cost=costs[j],coverage=supports[j]) for j in ranked[:5]]))
 pairs=[]
 for a,b in zip(anchors,anchors[1:]):
  state='insufficient' if not a['supported'] or not b['supported'] else 'ambiguous' if not a['unique'] or not b['unique'] or a['query_window'][1]>=b['query_window'][0] and a['query_window'][0]<=b['query_window'][1] else 'forward' if a['query_center']<b['query_center'] else 'reverse'
  pairs.append(dict(first=a['segment_id'],second=b['segment_id'],state=state))
 return dict(anchors=anchors,relations=pairs,semantic_order_verified=False,scope='unconstrained numerical retrieval candidates; repeated/indistinguishable postures can remain ambiguous')
def compare(q,r,segments,path=None):
 if path is None:path=dtw(alignment_cost(q,r))
 n=len(next(iter(q.values()))[0]);m=len(next(iter(r.values()))[0]);assert set(path[:,0])==set(range(n)) and set(path[:,1])==set(range(m))
 core=[s for s in segments if s['role']=='core_candidate']
 per=[dict(segment_id=s['segment_id'],reference_core=s['core'],query_frames=np.unique(path[np.isin(path[:,1],np.arange(s['core'][0],s['core'][1]+1)),0]).tolist(),groups=observed_summary(q,r,path,np.arange(s['core'][0],s['core'][1]+1))) for s in core]
 return dict(path=path.tolist(),query_frames=n,reference_frames=m,query_scope_retained=n,reference_scope_retained=m,groups=observed_summary(q,r,path),stages=per,changes=temporal_changes(q,r,path,core),free_order=free_order(q,r,core),max_query_reuse=int(np.bincount(path[:,0]).max()),max_reference_reuse=int(np.bincount(path[:,1]).max()))
