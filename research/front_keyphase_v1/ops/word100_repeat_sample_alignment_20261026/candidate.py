"""Collapse only exactly repeated observation rows for path search, never for evidence.
All original reference frames and all source-observation windows remain in the output.
"""
import numpy as np
KEYS=('features','present','core','values','available','quality_eligible','quality_after')

def groups(z):
 n=len(z['core']);equal=np.ones(max(0,n-1),bool)
 for k in KEYS:
  a=z[k][1:];b=z[k][:-1];same=(a==b)
  if np.issubdtype(a.dtype,np.floating):same|=np.isnan(a)&np.isnan(b)
  equal&=same.reshape(n-1,-1).all(1)
 starts=np.r_[0,np.flatnonzero(~equal)+1];ends=np.r_[starts[1:]-1,n-1];ids=np.repeat(np.arange(len(starts)),ends-starts+1)
 return starts,ends,ids

def compressed(z,start,end):
 n=len(z['core']);ix=(start+end)//2
 return {k:(v[ix].copy() if v.ndim and v.shape[0]==n else v.copy()) for k,v in z.items()}

def expand_phase(p,start,end):
 p=np.asarray(p,float);lo=np.floor(p).astype(int);hi=np.ceil(p).astype(int);w=p-lo
 # Interpolate only across the transition between two runs, not through their dwell time.
 return np.where(lo==hi,(start[lo]+end[lo])/2,end[lo]*(1-w)+start[hi]*w)

def pair(q,r,M):
 qs,qe,qg=groups(q);rs,re,rg=groups(r)
 cq=compressed(q,qs,qe);cr=compressed(r,rs,re);cp=M.pair(cq,cr)
 base=expand_phase(cp['phase_old'][rg],qs,qe);new=expand_phase(cp['phase_new'][rg],qs,qe);first=qs[cp['span_first'][rg]];last=qe[cp['span_last'][rg]];n=len(r['core']);domain=np.broadcast_to(r['core'][:,None,None],(n,2,23)).copy()
 om=M.map_values(q,base,first,last);nm=M.map_values(q,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)));rm=M.map_values(r,np.arange(n,dtype=float),np.arange(n),np.arange(n))
 od,ov,oq=M.describe(om,domain);nd,nv,nq=M.describe(nm,domain);rd,rv,rq=M.describe(rm,domain);source=np.where(nv,1,np.where(ov,2,0)).astype(np.uint8);oldquality=oq&rq;keep=oldquality&~(np.where(source==1,nq,oq)&rq);source=np.where(keep,2,source);ok=(source>0)&rv;point=domain&om['valid']&r['available']
 return dict(domain=domain,phase_old=base,phase_new=new,span_first=first,span_last=last,point_ok=point,point_delta=np.where(point,abs(om['values']-r['values']),np.nan).astype(np.float32),sequence_ok=ok,sequence_delta=np.where(ok[...,None],abs(np.where((source==1)[...,None],nd,od)-rd),np.nan).astype(np.float32),source=source,sequence_quality=np.where(source==1,nq,oq)&rq&ok,query_group_start=qs,query_group_end=qe,reference_group_start=rs,reference_group_end=re,reference_group_index=rg,compressed_path=cp['path'],compressed_phase_old=cp['phase_old'],compressed_phase_new=cp['phase_new'],compressed_span_first=cp['span_first'],compressed_span_last=cp['span_last'])
