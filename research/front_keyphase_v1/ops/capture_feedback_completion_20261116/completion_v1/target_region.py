"""Visible hand-edge / other-hand region; never certify physical 3D contact."""
from plane_model import *

def region_distance(ch,active='R'):
 a,b=(112,91)if active=='R'else(91,112)
 p=ch['body_relative_xy'];available=ch['point_available']
 edge=p[:,a+np.array([17,18,19,20])];palm=p[:,b+np.array([0,5,9,13,17])]
 # Distance from the visible little-finger edge to the other palm's convex hull.
 # The palm scale avoids dependence on camera distance or the actor's size.
 import cv2
 out=np.full(len(p),np.nan)
 for i in range(len(p)):
  if not available[i,np.r_[a+np.array([17,18,19,20]),b+np.array([0,5,9,13,17])]].all():continue
  if not np.isfinite(edge[i]).all()or not np.isfinite(palm[i]).all():continue
  scale=np.linalg.norm(p[i,b+9]-p[i,b])
  if scale<=1e-5:continue
  hull=cv2.convexHull(palm[i].astype(np.float32))
  out[i]=min(max(0.,-cv2.pointPolygonTest(hull,tuple(v.astype(float)),True))for v in edge[i])/scale
 return out

class TargetRegion:
 def __init__(self):self.limit=read(V/'target_region/FROZEN.json')['fit_max_distance']
 def criterion(self,ch,events,active='R'):
  d=region_distance(ch,active);spans=[];unconfirmed=[]
  for e in events:
   # The approach itself need not overlap the target. Inspect its observed end.
   a=max(e['start'],e.get('arrival_index',e['end'])-2);b=e['end'];ids=np.arange(a,b+1)
   valid=ids[np.isfinite(d[ids])&(d[ids]<=self.limit)]
   if len(valid)>=2:spans.append([int(valid[0]),int(valid[-1])])
   else:unconfirmed.append([a,b])
  return dict(id='relative_target',kind='projected_hand_region',status='supported'if spans and not unconfirmed else'unknown',
   source_spans=spans,unconfirmed_events=unconfirmed,physical_contact=None,
   reason='Observed hand edge approaches the other hand region; combined with separately checked shape and visible plane.')

def main():
 out=V/'target_region';out.mkdir(exist_ok=True);assert not(out/'FROZEN.json').exists()
 rows=read(B/'INPUTS.json')['rows'];source=read(B/'source_corrected/SOURCE_TRUTH.json')['rows'];values=[];truth=[]
 for r in rows:
  if r['role']!='fit':continue
  ch=dict(np.load(r['channels']));d=region_distance(ch)
  for label in source:
   if label['clip_id']!=r['clip_id']or label['state']!=2:continue
   ix=int(np.flatnonzero(ch['frame_id']==label['frame'])[0])
   if np.isfinite(d[ix]):values.append(float(d[ix]));truth.append(dict(**label,distance=float(d[ix])))
 write(out/'SOURCE_TRUTH.json',dict(rows=truth,scope='Original reviewed near-hand frames. No 3D contact annotation.'))
 write(out/'FROZEN.json',dict(fit_max_distance=max(values),fit_count=len(values),fit_people=list(range(1,10)),
  calibration='Maximum observed normalized edge-to-other-palm distance from reviewed FIT near frames; no DEV tuning.',
  decision='At least two observed endpoint frames in FIT envelope plus independent learned arrival/shape/plane evidence.',
  physical_contact=None,source_sha=sha(out/'SOURCE_TRUTH.json')))
 print('REGION',len(values),min(values),max(values))
if __name__=='__main__':main()
