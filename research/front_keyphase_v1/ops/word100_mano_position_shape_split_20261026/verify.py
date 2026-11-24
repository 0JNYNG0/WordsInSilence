"""Separate scalar/geometry checker; does not import the new candidate."""
from pathlib import Path
import json,hashlib,math,statistics
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_mano_identity_recheck_20261026';H=O/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def distance(a,b):return math.hypot(float(a[0]-b[0]),float(a[1]-b[1]))
def convex(points):
 p=sorted(set((float(x),float(y)) for x,y in points))
 def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
 lo=[];hi=[]
 for q in p:
  while len(lo)>=2 and cross(lo[-2],lo[-1],q)<=0:lo.pop()
  lo.append(q)
 for q in reversed(p):
  while len(hi)>=2 and cross(hi[-2],hi[-1],q)<=0:hi.pop()
  hi.append(q)
 return lo[:-1]+hi[:-1]
def outside_dist(p,h):
 inside=True;ds=[]
 for a,b in zip(h,h[1:]+h[:1]):
  x=b[0]-a[0];y=b[1]-a[1];dx=float(p[0])-a[0];dy=float(p[1])-a[1]
  if x*dy-y*dx < -1e-6:inside=False
  t=max(0,min(1,(dx*x+dy*y)/(x*x+y*y)))
  ds.append(math.hypot(dx-t*x,dy-t*y))
 return 0. if inside else min(ds)
def main():
 f=read(U/'FROZEN.json');r=read(U/'RESULT.json');p=f['params'];idx=read(U/'TRACK_INDEX.json');e=read(U/'EVIDENCE.json');steps=0;cache={}
 for path,key in [(U/'candidate.py','candidate_sha256'),(U/'run.py','runner_sha256')]:assert sha(path)==f[key]
 for pin in f['prior_pins']:assert sha(pin['path'])==pin['sha256'],pin['path']
 for path,h in f['protected'].items():assert sha(path)==h,path
 for tr in idx:
  assert sha(tr['path'])==tr['sha256'];a=dict(np.load(tr['path']));valid=[True]*len(a['initial']);qual=len(valid)>=6;finished=tr['anchor']
  for i,st in enumerate(tr['steps']):
   steps+=1
   for j in range(len(valid)):
    q=a['forward'][i,j];b=a['backward'][i,j];old=a['previous'][i,j];err=float(a['error'][i,j])
    ok=a['status'][i,j]==1 and a['back_status'][i,j]==1 and all(math.isfinite(float(v)) for v in q) and math.isfinite(err) and err<=20 and distance(b,old)<=1.5 and 0<=q[0]<640 and 0<=q[1]<360
    valid[j]=bool(valid[j] and ok)
   assert np.array_equal(valid,a['valid'][i])
   shifts=[a['forward'][i,j]-a['previous'][i,j] for j in range(len(valid)) if valid[j]]
   if shifts:
    med=[statistics.median(float(q[k]) for q in shifts) for k in (0,1)]
    ds=sorted(distance(q,med) for q in shifts);q=(len(ds)-1)*.9;low=math.floor(q);high=math.ceil(q);res=(ds[low]*(high-q)+ds[high]*(q-low)) if high!=low else ds[low]
    res/=tr['scale']/3
    assert math.isclose(res,st['residual'],rel_tol=1e-4,abs_tol=1e-6),(tr['path'],res,st['residual'])
   else:res=None
   qual=sum(valid)>=6 and sum(valid)/len(valid)>=.60 and res is not None and res<=.10
   assert qual==st['qualified'];finished=st['frame_to']
  if not tr['steps']:valid=[False]*len(valid)
  assert np.array_equal(valid,a['final_valid'])
  assert tr['qualified']==bool(qual and finished==tr['target'])
  cache[(tr['clip'],tr['anchor'],tr['target'],tr['side'])]=a
 prepared={m['clip_id']:dict(np.load(m['prepared'])) for m in f['manifest']};geometry=0;proposals=0
 for row in e:
  aa=row['assessment'];k=row['clip_index'];t=row['frame'];s=row['side'];cid=row['clip']
  if k<7:
   z=prepared[cid];hand=z['full_xy'][t,s] if row['source']=='original' else np.load(H/f'raw_predictions/{k:02d}_{t:04d}_{s}.npz')['xy_px'];pose=z['pose'][t];pv=z['pose_valid'][t]
  else:
   m=next(m for m in f['controls'] if m['clip_id']==cid);zz=np.load(m['path']);pose=zz['xy_px'][t,:33];pv=zz['usable'][t,:33];hand=zz['xy_px'][t,33+21*s:54+21*s]
  scale=distance(pose[11],pose[12]);ds=[distance(hand[0],pose[15+q])/scale if all(pv[[11+q,13+q,15+q]]) else None for q in (0,1)]
  arm=-1
  if None not in ds:
   w=0 if ds[0]<=ds[1] else 1
   if ds[w]<=.25 and ds[1-w]-ds[w]>=.10:arm=w
  assert arm==aa['arm_side'];h=convex(np.asarray(hand,np.float32))
  region=[]
  for q in (0,1):
   details=aa['regions'][q]['details'];good=[]
   for d in details:
    arr=cache[(cid,d['anchor'],t,q)];dist=[outside_dist(point,h)/scale for point in arr['xy'][arr['final_valid']]]
    med=statistics.median(dist) if dist else None
    if med is not None:assert abs(med-d['distance_to_region'])<1e-6,(cid,t,s,q,med,d['distance_to_region'])
    else:assert d['distance_to_region'] is None
    geometry+=1;good.append(d['flow_qualified'] and med is not None and med<=.12)
    assert bool(good[-1])==d['region_match']
   both=len(details)==2 and details[0]['anchor']<t<details[1]['anchor'];passed=both and all(good)
   assert passed==aa['regions'][q]['qualified'];region.append(passed)
  rs=region.index(True) if sum(region)==1 else -1;cs=[q for q in (0,1) if aa['contour'][q]['pass']];ct=cs[0] if len(cs)==1 else -1
  identity=arm if arm>=0 and rs==arm and ct in [-1,arm] else -1
  assert identity==aa['identity_proposal'];assert not aa['finger_shape_confirmed'] and not aa['precise_palm_center_confirmed'] and not aa['coordinates_changed'];proposals+=identity>=0
 points=crossframes=crosspoints=0
 for m in f['manifest']:
  old=dict(np.load(P/'outputs_v3'/(m['clip_id']+'.npz')));new=dict(np.load(U/'outputs'/(m['clip_id']+'.npz')))
  assert old.keys()==new.keys()
  for name in old:assert np.array_equal(old[name],new[name],equal_nan=True),(m['clip_id'],name)
  points+=int(old['R_observed'].sum());crossframes+=len(m['crossing_frames']);crosspoints+=int(old['R_valid'][m['crossing_frames']].sum())
 out=dict(status='PASS',operator='Codex separate scalar verification, not independent human GT',flow_paths=len(idx),flow_steps=steps,assessment_rows=len(e),region_geometry_checks=geometry,identity_diagnostic_proposals=proposals,original_points_preserved=points,crossing_frames_preserved=crossframes,crossing_points_preserved=crosspoints,normal_control_frames_preserved=94,protected_files=len(f['protected']),prior_files=len(f['prior_pins']),all_baseline_arrays_identical=True,contour='Reused pinned previous scalar-verified implementation, not a newly independent RGB contour truth audit',adoption_requires_visual_review=True)
 write(U/'VERIFICATION.json',out);print(json.dumps(out))
if __name__=='__main__':main()
