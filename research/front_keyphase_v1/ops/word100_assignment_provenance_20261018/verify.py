"""Separate scalar provenance/geometry reconstruction and saved-output audit."""
import json,math,sys,time,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;P=U.parent/'word100_assignment_conflict_20261018'
from run import read,write,load,sha,pinscheck
PALM=(0,5,9,13,17)
def length(x,y):return math.sqrt(sum((float(a)-float(b))**2 for a,b in zip(x,y)))
def match(cost,i,s,gate):
 q=cost[i][s]
 if not math.isfinite(q) or q>gate:return False
 return all(not math.isfinite(r) or q<.7*r for r in [cost[i][1-s]]+[cost[j][s] for j in range(len(cost)) if j!=i])
def turn(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def hull(points):
 points=sorted(set(tuple(float(t) for t in p) for p in np.asarray(points,np.float32)))
 if len(points)<3:return points
 lo=[];hi=[]
 for p in points:
  while len(lo)>1 and turn(lo[-2],lo[-1],p)<=0:lo.pop()
  lo.append(p)
 for p in points[::-1]:
  while len(hi)>1 and turn(hi[-2],hi[-1],p)<=0:hi.pop()
  hi.append(p)
 return lo[:-1]+hi[:-1]
def area(poly):return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1])))/2 if len(poly)>2 else 0.
def overlap(hands):
 if len(hands)!=2 or not np.isfinite(hands).all():return 0.
 a,b=map(hull,hands);den=min(area(a),area(b))
 if den<=1e-6:return 0.
 poly=a
 for p,q in zip(b,b[1:]+b[:1]):
  if not poly:break
  clipped=[];prev=poly[-1];pv=turn(p,q,prev)
  for cur in poly:
   cv=turn(p,q,cur)
   if (cv>=0)!=(pv>=0):
    alpha=pv/(pv-cv);clipped.append((prev[0]+alpha*(cur[0]-prev[0]),prev[1]+alpha*(cur[1]-prev[1])))
   if cv>=0:clipped.append(cur)
   prev,pv=cur,cv
  poly=clipped
 return max(0.,min(1.,area(poly)/den))

def main():
 c=read(U/'CONTEXT.json');pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'));counts=Counter();changes=[];start=time.monotonic();max_geom_error=0
 for rn,r in enumerate(c['rows'],1):
  for v in ('masked','sensor'):
   raw=load(r['views'][v]['raw']);out=load(U/'results'/f"{r['clip_id']}_{v}.npz");old=load(P/'results'/f"{r['clip_id']}_{v}.npz");traces=read(Path(r['views'][v]['assignment']).with_name('trace.json'))
   confirmed={};scale=None;scale_at=None;last_time=None
   assert np.array_equal(raw['frame_id'],out['frame_id']) and np.array_equal(raw['timestamp_us'],out['timestamp_us'])
   for f in range(len(raw['raw_count'])):
    n=int(raw['raw_count'][f]);hh=raw['hand_xy_px'][f,:n];p=raw['pose_xy_px'][f];pu=raw['pose_usable'][f];t=int(raw['timestamp_us'][f])/1e6
    if last_time is not None:assert t>last_time
    last_time=t;confirmed={s:q for s,q in confirmed.items() if t-q['time']<=.5}
    if pu[11] and pu[12] and np.isfinite(p[[11,12]]).all():
     z=length(p[11],p[12])
     if z>1e-6:scale=z;scale_at=t
    sw=scale if scale_at is not None and t-scale_at<=.5 else None
    assert (sw is None and math.isnan(out['scale'][f])) or abs(sw-out['scale'][f])<1e-9
    pc=[[math.inf]*2 for _ in range(n)];ct=[[math.inf]*2 for _ in range(n)];di=[False]*n;hi=[False]*n
    w=float(raw['width'][f]);h=float(raw['height'][f])
    for i in range(n):
     hand=np.asarray(hh[i],np.float32);sc=float(raw['raw_score'][f,i])
     valid=np.isfinite(hand).all() and math.isfinite(sc) and sc>=.5 and all(-.05*w<=float(hand[k,0])<=1.05*w and -.05*h<=float(hand[k,1])<=1.05*h for k in PALM)
     for s,j in enumerate((15,16)):
      if sw is not None:
       if valid and pu[j] and np.isfinite(p[j]).all():pc[i][s]=length(hh[i,0],p[j])/sw
       if s in confirmed and np.isfinite(hh[i]).all():
        q=confirmed[s];dt=t-q['time'];ct[i][s]=sum(length(hh[i,k],[q['palm'][z][a]+q['velocity'][a]*dt for a in range(2)]) for z,k in enumerate(PALM))/5/sw
      for actual,recorded in [(pc[i][s],out['pose_distance'][f,i,s]),(ct[i][s],out['confirmed_track_distance'][f,i,s])]:
       assert (not math.isfinite(actual) and math.isnan(recorded)) or abs(actual-recorded)<1e-8,(r['clip_id'],v,f,i,s,actual,recorded)
     side=int(old['sides'][f,i]);cur=int(out['sides'][f,i]);st=int(out['state'][f,i])
     di[i]=side>=0 and match(pc,i,side,.35) # recomputed after full rows below
    for i in range(n):
     s=int(old['sides'][f,i]);di[i]=s>=0 and match(pc,i,s,.35);hi[i]=s>=0 and match(ct,i,s,.4)
     assert di[i]==bool(out['current_pose'][f,i]) and hi[i]==bool(out['confirmed_history'][f,i])
    ov=overlap(hh);err=abs(ov-out['region_overlap'][f]);max_geom_error=max(max_geom_error,err);assert err<2e-4,(r['clip_id'],v,f,'geometry',ov,out['region_overlap'][f])
    proposed_states=[0 if int(old['sides'][f,i])<0 else 2 if di[i] or hi[i] else 1 for i in range(n)];expected_sides=old['sides'][f,:n].tolist();expected_guard=[0]*n
    for i in range(n):
     s=expected_sides[i]
     if s<0 or proposed_states[i]!=1:continue
     q=confirmed.get(s);reason=traces[f]['reasons'][i]
     if q is not None and (reason=='label_without_close_pose' or reason.startswith('temporal')):
      dt=t-q['time']
      if 0<dt<=.1+1e-12 and ct[i][s]>max(.4,12*dt):expected_sides[i]=-1;expected_guard[i]=1
    if n==2 and ov>=.8:
     for i in range(n):
      s=expected_sides[i];j=1-i
      if s<0 or proposed_states[i]!=1 or expected_sides[j]!=1-s or proposed_states[j]!=2:continue
      q=confirmed.get(s);reachable=q is not None and ct[i][s]<=max(.4,12*(t-q['time']))
      if not reachable and ct[i][1-s]<=.4 and (not math.isfinite(ct[i][s]) or ct[i][1-s]<.7*ct[i][s]):expected_sides[i]=-1;expected_guard[i]=2
    assert expected_sides==out['sides'][f,:n].tolist() and expected_guard==out['guard'][f,:n].tolist(),(r['clip_id'],v,f,'guard_mismatch')
    for i in range(n):
     s=int(out['sides'][f,i]);oldside=int(old['sides'][f,i]);st=int(out['state'][f,i]);expected=3 if expected_guard[i] else proposed_states[i]
     assert st==expected
     if st==2:
      assert s>=0 and (di[i] or hi[i])
      root=f if di[i] else confirmed[s]['root'];parent=-1 if di[i] else confirmed[s]['frame']
      assert int(out['origin_frame'][f,i])==root and int(out['parent_frame'][f,i])==parent
      counts['confirmed_provenance_checked']+=1
     if st==1:assert not di[i] and not hi[i] and 0<=int(out['origin_frame'][f,i])<=f
     if oldside!=s:
      assert oldside>=0 and s==-1 and st==3
      changes.append((r['clip_id'],v,f,i,oldside,s))
     counts['state_'+['unassigned','provisional','confirmed','withheld_conflict'][st]]+=1
    # Independent commit after every frame's decisions, not per-detection.
    updates={}
    for i in range(n):
     if int(out['state'][f,i])!=2:continue
     s=int(out['sides'][f,i]);q=confirmed.get(s);palms=[[float(v) for v in hh[i,k]] for k in PALM];vel=[0.,0.]
     if q is not None and t>q['time'] and sw is not None:
      vel=[(sum(p[a] for p in palms)-sum(p[a] for p in q['palm']))/5/(t-q['time']) for a in (0,1)];speed=length(vel,[0,0]);cap=12*sw
      if speed>cap:vel=[z*cap/speed for z in vel]
     updates[s]={'palm':palms,'time':t,'frame':f,'velocity':vel,'root':int(out['origin_frame'][f,i])}
    confirmed.update(updates);counts['frames']+=1;counts['raw_candidates']+=n
   assert (out['state'][raw['raw_count']==0]==0).all()
  if rn%200==0:print('Independent scalar checked',rn,'/899',flush=True)
 expected=read(U/'CHANGES.json');assert sorted(changes)==sorted((x['clip_id'],x['view'],x['index'],x['raw_index'],x['old_side'],x['new_side']) for x in expected)
 summary=read(U/'SUMMARY.json')
 for k in ('frames','raw_candidates','state_unassigned','state_provisional','state_confirmed','state_withheld_conflict'):assert counts[k]==summary['counts'].get(k,0)
 ann=read(U/'ANNOTATED_COMPARISON.json');assert len(ann)==456 and len({x['id'] for x in ann})==456
 cross=read(U/'CROSSING_COMPARISON.json');assert len(cross)==315
 # Replay changed clips and earlier eight crossing controls in reversed raw
 # order. Prefix outputs are inherently causal; compare standalone prefixes.
 sys.path.insert(0,str(U));from candidate import HandAssigner
 selection={x['clip_id'] for x in expected}|{x['clip_id'] for x in read(P/'CROSSING_SELECTION.json')}
 for r in c['rows']:
  if r['clip_id'] not in selection:continue
  for v in ('masked','sensor'):
   raw=load(r['views'][v]['raw']);saved=load(U/'results'/f"{r['clip_id']}_{v}.npz");a=HandAssigner();b=HandAssigner();cut=len(raw['raw_count'])//2
   for f in range(len(raw['raw_count'])):
    n=int(raw['raw_count'][f]);args=[raw['pose_xy_px'][f],raw['pose_usable'][f],int(raw['timestamp_us'][f]),int(raw['width'][f]),int(raw['height'][f])]
    out,d=a.assign(raw['hand_xy_px'][f,:n][::-1],raw['raw_label'][f,:n][::-1],raw['raw_score'][f,:n][::-1],*args)
    assert [-1 if out[i] is None else out[i] for i in range(n)][::-1]==saved['sides'][f,:n].tolist()
    assert [0 if x=='unassigned' else 1 if x=='provisional' else 2 if x=='confirmed' else 3 for x in d['identity_state']][::-1]==saved['state'][f,:n].tolist()
    counts['permutation_frames']+=1
    if f<cut:
     out,d=b.assign(raw['hand_xy_px'][f,:n],raw['raw_label'][f,:n],raw['raw_score'][f,:n],*args)
     assert [-1 if out[i] is None else out[i] for i in range(n)]==saved['sides'][f,:n].tolist();counts['prefix_frames']+=1
 pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'))
 write(U/'INDEPENDENT.json',{'status':'PASS','counts':dict(counts),'changed_associations_matched':len(changes),'max_independent_polygon_overlap_error':max_geom_error,'contracts_reused':read(U/'CONTRACTS.json'),'protected_pins':len(c['pins']),'seconds':time.monotonic()-start,'limits':'Same operator separate scalar and polygon implementation; confirms computations, not independent anatomical truth. Known TRAIN development audit, not unseen accuracy.','training':0,'production_changed':False})
 print('Independent PASS',dict(counts),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  if not (U/'VERIFY_FIRST_ERROR.json').exists():write(U/'VERIFY_FIRST_ERROR.json',{'traceback':traceback.format_exc()})
  raise
