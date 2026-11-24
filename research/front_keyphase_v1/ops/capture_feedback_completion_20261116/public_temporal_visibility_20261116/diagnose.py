from pathlib import Path
import sys
L=Path(__file__).resolve().parent;K=L.parent/'public_view_arrival_support_20261116'
sys.path.insert(0,str(K))
from preserved_arrival import *
from target_region import region_distance

def current_sources():return sources()+read(J/'confirmation/INPUTS.json')['rows']+read(K/'confirmation/INPUTS.json')['rows']
def palm_diameter_distance(ch,active='R'):
 a,b=(112,91)if active=='R'else(91,112);p=ch['body_relative_xy'];av=ch['point_available'];out=np.full(len(p),np.nan)
 for i in range(len(p)):
  ids=np.r_[a+np.array([17,18,19,20]),b+np.array([0,5,9,13,17])]
  if not av[i,ids].all()or not np.isfinite(p[i,ids]).all():continue
  edge=p[i,a+np.array([17,18,19,20])];palm=p[i,b+np.array([0,5,9,13,17])];scale=np.linalg.norm(palm[:,None]-palm[None,:],axis=-1).max()
  if scale<=1e-5:continue
  hull=cv2.convexHull(palm.astype(np.float32));out[i]=min(max(0.,-cv2.pointPolygonTest(hull,tuple(v.astype(float)),True))for v in edge)/scale
 return out

def main():
 torch.set_num_threads(2);net=PreservedArrivalLesson();out=L/'audit';out.mkdir(exist_ok=True);rows=[]
 write(L/'CONTRACT.json',dict(goal='Separate hidden/foreshortened finger evidence from missed repetition terminal evidence. Preserve existing normals, errors, event bounds and speed.',graph=['A source/keypoint/temporal-head diagnosis','B public FIT annotation/evidence repair as cause warrants','C normal/error/partial/occlusion and unselected public verification','D admitted web, auto-stop and timing; document remaining observability limits'],rules=['No word/person exception or expected-count-driven completion','Do not fabricate hidden fingers or physicalcontact','Private recordings excluded from training/tuning/selection','C-T3F and100-class backbone frozen','Previously admitted paths immutable'],baseline=str(K/'RESULT.json')))
 for r in current_sources():
  z=dict(np.load(r['raw']));ch=channels(z);q=net.analyze(z);detail={}
  for hand,side in[('L','left'),('R','right')]:
   ob=net.shape.observations[hand];s=np.array(q['hand_form_evidence'][hand]['states']);detail[hand]=dict(states=s.tolist(),form=q['hand_form_evidence'][hand]['form'],all_available=ob['available'].all(1).tolist(),projection_conflict=ob['projection_conflict'].tolist(),available=ob['available'].tolist(),fingers=q['shape_fingers'][hand])
  write(out/(r['id']+'.json'),dict(source=r,output=q,hand=detail,motion=net.motion.sequence(ch).tolist(),distance=region_distance(ch).tolist(),diameter_distance=palm_diameter_distance(ch).tolist()))
  row=dict(id=r['id'],events=len(q['events']),unknown=[c['id']for c in q['criteria']if c['status']=='unknown'],warning=[c['id']for c in q['criteria']if c['status']=='difference']);rows.append(row)
 write(out/'INDEX.json',dict(rows=rows));print('AUDITED',len(rows),flush=True)
 for r in rows:
  if r['unknown']or r['warning']:print(r,flush=True)
if __name__=='__main__':main()
