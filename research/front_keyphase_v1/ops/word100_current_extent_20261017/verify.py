"""Independent saved-output audit: scalar monotone hull / polygon clipping.
Does not import spatial.py or use OpenCV geometry.
"""
import json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def hull(points):
 p=sorted(set(tuple(map(float,x)) for x in points));lo=[];hi=[]
 for seq,out in ((p,lo),(p[::-1],hi)):
  for v in seq:
   while len(out)>1 and cross(out[-2],out[-1],v)<=0:out.pop()
   out.append(v)
 return lo[:-1]+hi[:-1]
def area(poly):return abs(sum(p[0]*q[1]-p[1]*q[0] for p,q in zip(poly,poly[1:]+poly[:1])))/2 if len(poly)>2 else 0.
def clip(subject,mask):
 out=subject[:]
 for a,b in zip(mask,mask[1:]+mask[:1]):
  inp=out;out=[]
  if not inp:break
  prev=inp[-1];pd=cross(a,b,prev)
  for v in inp:
   vd=cross(a,b,v)
   if (vd>=0)!=(pd>=0):
    k=pd/(pd-vd);out.append((prev[0]+k*(v[0]-prev[0]),prev[1]+k*(v[1]-prev[1])))
   if vd>=0:out.append(v)
   prev,pd=v,vd
 return out
s=read(U/'SCOPE.json');o=read(U/'OUTPUT.json')
assert o['scope_sha256']==sha(U/'SCOPE.json') and o['new_labels_sha256']==sha(U/'VISUAL_LABELS.json')
for p,h in {**s['source_pins'],**s['protected_pins'],**s['dependencies']}.items():assert sha(p)==h,p
old={r['clip_id']:r for r in read(O/'word100_hand_evidence_split_20261017/OUTPUT.json')['records']}
rows={r['clip_id']:r for r in o['records']};checked=preserved=polygon_count=0;max_area_diff=0.
for r in s['records']:
 raw=np.load(r['paths']['raw']);obs=np.load(r['paths']['candidate'])
 for f,fr in enumerate(rows[r['clip_id']]['frames']):
  xy=obs['xy_px'][f].astype(float);mask=obs['usable'][f];conf=raw['pose_conf'][f];w,h=int(raw['width'][f]),int(raw['height'][f]);scale2=sum((xy[11]-xy[12])**2)
  for side,row in enumerate(fr['hands']):
   checked+=1
   if r['clip_id'] in old:
    prior=old[r['clip_id']]['frames'][f]['hands'][side]
    assert all(row[k]==v for k,v in prior.items());preserved+=1
   assert row['coordinate_repair'] is False and row['eligible_for_automatic_exclusion'] is False
   ext,ov=row['current_extent'],row['current_overlap']
   assert ext['whole_hand_visibility_proven'] is False and ov['occlusion_cause_confirmed'] is False
   if row['position']['state'] not in ('wrist_interior','wrist_boundary_or_outside'):
    assert ext['status']==ov['status']=='not_assessed';continue
   ids=[15+side,17+side,19+side,21+side];points=xy[ids]
   if not (np.isfinite(points).all() and np.isfinite(conf[ids]).all() and min(conf[ids])>=.5 and math.isfinite(scale2) and scale2>0):
    assert ext['status']==ov['status']=='coarse_pose_unavailable';continue
   p=hull(points)
   if len(p)<3 or area(p)/scale2<=1e-9:
    assert ext['status']=='degenerate_pose_region';continue
   edges=[e for e,v in zip(('left','top','right','bottom'),(min(points[:,0])<0,min(points[:,1])<0,max(points[:,0])>w,max(points[:,1])>h)) if v]
   assert ext['crossed_edges']==edges and ext['status']==('estimated_extent_crosses_frame' if edges else 'estimated_extent_inside_frame')
   if edges:assert ov['status']=='boundary_competes_with_overlap';continue
   other=1-side
   if fr['hands'][other]['shape']['state']!='usable_observed_points':assert ov['status']=='other_hand_not_observed';continue
   pts=[xy[j] for j in range(33+other*21,54+other*21) if mask[j] and np.isfinite(xy[j]).all() and 0<=xy[j,0]<=w and 0<=xy[j,1]<=h]
   q=hull(pts)
   if len(q)<3 or area(q)/scale2<=1e-9:assert ov['status']=='other_hand_region_unavailable';continue
   a=area(clip(p,q));polygon_count+=1
   assert ov['status']==('current_2d_intersection_suspected' if a/scale2>1e-9 else 'no_current_2d_intersection'),(r['clip_id'],f,side,a,ov)
   diff=abs(a-ov['intersection_pixels2']);max_area_diff=max(max_area_diff,diff);assert diff<.1,(diff,r['clip_id'],f)
tables={};errors=[];misses=[];target=[]
for x in o['comparisons']:
 c=x['candidate'];e=c['current_extent']['status'];v=x['visual_category'];g=x['origin'];d=tables.setdefault(g,Counter())
 if v=='observed_control':d['controls_preserved']+=int(c['shape']['state']=='usable_observed_points');continue
 if v=='boundary_or_outside':
  d['visual_boundary']+=1
  if e=='estimated_extent_crosses_frame':d['boundary_cue']+=1
  elif e=='estimated_extent_inside_frame':d['boundary_miss']+=1;misses.append(x)
  else:d['boundary_abstain']+=1
 else:
  d['visual_inside']+=1
  if e=='estimated_extent_crosses_frame':d['false_boundary_cue']+=1;errors.append(x)
  elif e=='estimated_extent_inside_frame':d['inside_estimate']+=1
  else:d['inside_abstain']+=1
 positive=c['current_overlap']['status']=='current_2d_intersection_suspected'
 if v=='inside_overlap':
  d['visual_overlap']+=1;d['overlap_cue_matches' if positive else 'overlap_cue_misses']+=1
 elif positive:d['false_overlap_cue']+=1;errors.append(x)
 if x['review_id'] in ('R016','N001'):target.append({'review_id':x['review_id'],'clip_id':x['clip_id'],'frame':x['frame'],'side':x['side'],'extent':c['current_extent']})
assert len(target)==2 and all(x['extent']['status']=='estimated_extent_crosses_frame' for x in target)
res={'status':'PASS','contract_hand_states_checked':checked,'previous_hand_states_preserved':preserved,'scalar_polygon_pairs':polygon_count,'max_intersection_area_difference_pixels2':max_area_diff,
 'tables':tables,'known_two_boundary_cases':target,'visual_cue_contradictions':errors,'visual_boundary_misses':misses,
 'note':'PASS is arithmetic/contract, not all-case accuracy. Same Codex operator separate code; known regression + new8 visual review. No inference of no overlap from absent intersection.'}
with (U/'INDEPENDENT_CHECK.json').open('x',encoding='utf-8') as f:json.dump(res,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in res.items() if k not in ('known_two_boundary_cases','visual_cue_contradictions','visual_boundary_misses')},ensure_ascii=False,indent=2))
