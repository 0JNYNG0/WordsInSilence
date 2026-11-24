"""Separate scalar location/convex-polygon audit, no candidate imports."""
from common import *
import numpy as np, math
from collections import Counter

def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def hull(points):
 pts=sorted(set(tuple(map(float,p)) for p in points));a=[];b=[]
 for seq,out in ((pts,a),(pts[::-1],b)):
  for p in seq:
   while len(out)>1 and cross(out[-2],out[-1],p)<=0:out.pop()
   out.append(p)
 return a[:-1]+b[:-1]
def area(p):return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1])))/2 if len(p)>2 else 0.
def clip(p,q):
 out=p[:]
 for a,b in zip(q,q[1:]+q[:1]):
  src=out;out=[]
  if not src:break
  prev=src[-1];pd=cross(a,b,prev)
  for v in src:
   vd=cross(a,b,v)
   if (vd>=0)!=(pd>=0):
    f=pd/(pd-vd);out.append((prev[0]+f*(v[0]-prev[0]),prev[1]+f*(v[1]-prev[1])))
   if vd>=0:out.append(v)
   prev,pd=v,vd
 return out
def independent(obs,raw,ass,view,w,h,indices):
 hist={};answer=[]
 for f in indices:
  xy=obs['xy_px'][f].astype(float).copy();mask=obs['usable'][f].copy();conf=raw['pose_conf'][f].copy();t=obs['timestamp_us'][f]/1e6
  inside=lambda p,m=0:bool(np.isfinite(p).all() and m<=p[0]<=w-m and m<=p[1]<=h-m)
  if view=='masked':
   allowed=np.array([mask[i] and inside(xy[i]) for i in range(75)]);xy[~allowed]=np.nan;mask &= allowed;conf[~allowed[:33]]=0
   assigned=[bool(mask[33+21*s:54+21*s].any()) for s in (0,1)];unassigned=False
  else:
   xy[:33]=raw['pose_xy_px'][f];ss=ass['sides'][f,:int(raw['raw_count'][f])];assigned=[s in ss for s in (0,1)];unassigned=bool((ss<0).any())
  trusted=lambda i,require=True:bool(np.isfinite(xy[i]).all() and np.isfinite(conf[i]) and conf[i]>=.5 and (not require or (mask[i] and inside(xy[i]))))
  hist={s:z for s,z in hist.items() if t-z[1]<=.5}
  palmids=[33+21*s+np.array([0,5,9,13,17]) for s in (0,1)];good=[bool(mask[ids].all() and np.isfinite(xy[ids]).all()) for ids in palmids];palms=[xy[ids].mean(0) for ids in palmids]
  shoulder=trusted(11) and trusted(12);scale=float(np.linalg.norm(xy[11]-xy[12])) if shoulder else 0;shoulder &= scale>1e-6
  positions=[];proximity=[]
  for s in (0,1):
   pos=5;prox=0;sh,el,wr=11+s,13+s,15+s
   if good[s]:pos=0
   elif assigned[s]:pos=1
   elif unassigned:pos=4
   elif shoulder and trusted(sh) and trusted(el) and trusted(wr,False):
    up=np.linalg.norm(xy[sh]-xy[el])/scale;lo=np.linalg.norm(xy[el]-xy[wr])/scale
    if .1<=up<=1.5 and .1<=lo<=1.5:
     p=xy[wr];dist=np.array([p[0],w-p[0],p[1],h-p[1]])/scale;k=int(dist.argmin());out=np.dot(p-xy[el],[(-1,0),(1,0),(0,-1),(0,1)][k])/scale
     if dist[k]<=.15 and out>.15:pos=3
     elif inside(p,.15*scale):
      pos=2
      if good[1-s] and s in hist:prox=2 if np.linalg.norm(p-palms[1-s])/scale<=.35 and np.linalg.norm(p-hist[s][0])/scale<=.4 else 1
   positions.append(pos);proximity.append(prox)
  states=[]
  for s,pos in enumerate(positions):
   ext=ov=0;pts=xy[np.array([15,17,19,21])+s];cs=conf[np.array([15,17,19,21])+s];sc2=float(np.sum((xy[11]-xy[12])**2))
   if pos in (2,3) and np.isfinite(pts).all() and np.isfinite(cs).all() and min(cs)>=.5 and math.isfinite(sc2) and sc2>0:
    p=hull(pts)
    if area(p)/sc2>1e-9:
     ext=1 if all(inside(z) for z in pts) else 2
     if ext==1 and good[1-s]:
      q=hull([xy[j] for j in range(33+21*(1-s),54+21*(1-s)) if mask[j] and inside(xy[j])])
      if area(q)/sc2>1e-9:ov=2 if area(clip(p,q))/sc2>1e-9 else 1
   states.append([pos,ext,ov,proximity[s]])
  for s in (0,1):
   if good[s]:hist[s]=(palms[s].copy(),t)
  answer.append(states)
 return np.asarray(answer,np.int8)

def main():
 assert not (U/'PREP_RUNNING.json').exists();c=read(U/'PREP_CONTEXT.json');prep=read(U/'PREP_RESULT.json');checkpins(prep['record_pins']);checkpins(c['pins'])
 basevalues=states_checked=0;counts=Counter();known_controls=0;differences=[];control_states=[]
 labels=read(CURRENT/'OUTPUT.json')['comparisons'];controls={(x['clip_id'],x['frame'],x['side']) for x in labels if x['visual_category']=='observed_control'}
 for n,r in enumerate(c['records']):
  rec=read(U/'inputs_v2'/f"{r['clip_id']}.json");checkpins(rec['outputs']);w,h=r['box'][2:]
  for v,paths in r['views'].items():
   new=loadnp(rec['views'][v]);old=loadnp(paths['channels']);obs,raw,ass=(loadnp(paths[k]) for k in ['observation','raw','assignment'])
   lookup={int(f):i for i,f in enumerate(obs['frame_id'])}
   for temporal in ('whole','context') if r['role']=='train_candidate' else ('whole',):
    x=new['x_'+temporal];assert np.array_equal(x[:,:280],old['x_'+temporal]);basevalues+=x[:,:280].size
    assert np.array_equal(new[temporal+'_frame_id'],old[temporal+'_frame_id']) and np.array_equal(new[temporal+'_ts'],old[temporal+'_ts'])
    expected=independent(obs,raw,ass,v,w,h,[lookup[int(f)] for f in new[temporal+'_frame_id']]);actual=new['states_'+temporal]
    if not np.array_equal(expected,actual):
     for f,s,k in np.argwhere(expected!=actual)[:30]:differences.append({'clip':r['clip_id'],'view':v,'temporal':temporal,'frame':int(f),'side':int(s),'field':int(k),'expected':int(expected[f,s,k]),'actual':int(actual[f,s,k])})
    for s in (0,1):
     for j,(off,sz) in enumerate(((0,6),(6,3),(9,3),(12,3))):
      block=x[:,280+15*s+off:280+15*s+off+sz];assert np.array_equal(block,np.eye(sz,dtype=np.float32)[actual[:,s,j]])
    states_checked+=actual.shape[0]*2;counts.update(f'{v}:{k}' for k in actual[:,:,0].ravel())
   if v=='sensor':
    for cid,f,s in controls:
     if cid==r['clip_id']:
      # Compare against C's actual admissible observation, not the earlier
      # diagnostic's un-clipped detector mask. Do not relax the image edge.
      ids=33+21*s+np.array([0,5,9,13,17]);expected_pos=0 if obs['usable'][f,ids].all() else 1
      assert new['full_states'][f,s,0]==expected_pos,(cid,f,s,'C observation control changed')
      known_controls+=1;control_states.append({'clip_id':cid,'frame':f,'side':s,'actual_C_control_state':expected_pos,'legacy_diagnostic_state':0})
  if (n+1)%200==0:print('Independent',n+1,flush=True)
 checkpins(c['pins'])
 result={'status':'PASS' if not differences else 'FAIL','base_channel_values_exact':basevalues,'hand_states_scalar_checked':states_checked,'known_C_observation_controls_preserved':known_controls,'control_states':control_states,'counts':dict(counts),'semantic_mismatches':differences,'verifier_imports_candidate':False,'same_operator_separate_code':True,'training_allowed':False}
 write(U/'INPUTS_INDEPENDENT.json',result,new=True);assert not differences,differences[:5];print(json.dumps({k:v for k,v in result.items() if k!='counts'}))
if __name__=='__main__':main()
