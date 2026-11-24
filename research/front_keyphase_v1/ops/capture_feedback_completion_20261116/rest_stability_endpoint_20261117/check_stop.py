from stable_rest import *
import importlib.util
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def simulate(cls,z,kind):
 obj=cls();xy=z['xy_px'].copy();w=int(z['source_width']);h=int(z['source_height']);n=len(xy)
 for _ in range(12):obj.calibrate(xy[0],w,h)
 obj.begin();stops=[]
 for i,p in enumerate(xy):
  if obj.update(p,w,h,int(z['timestamp_us'][i])):stops.append(i)
 return dict(calibrated=obj.rest is not None,first_stop=stops[0]if stops else None,trace=obj.trace if hasattr(obj,'trace')else None)
def main():
 rows=[]
 for p in sorted((P/'probe').glob('*.npz')):
  z=dict(np.load(p));a=simulate(PreviousRest,z,'normal');b=simulate(StableRest,z,'normal')
  cid=p.stem.split('_0p')[0];baseline=read(G/'public_frontal_evidence_repair_20261117/web'/('normal_'+cid+'.json'));last_action=max(e['end']for e in baseline['lesson_acceptance']['events'])
  rows.append(dict(id=p.stem,kind='public_crop',passed=b['first_stop']is not None and b['first_stop']>last_action,before=a['first_stop'],after=b['first_stop'],last_action=last_action))
  # Reproduce isolated wrong hand-height samples, not actual hand movements.
  # Keep the public arm/body track and add an explicit synthetic detection fault.
  for width,period in [(1,10),(2,15)]:
   end=simulate(PreviousRest,z,'normal')['first_stop'];start=end-10;base=z['xy_px'][end];post=np.repeat(base[None],100,axis=0);xy=np.concatenate([z['xy_px'][:start],post])
   for i in range(start,len(xy)):
    if (i-start)%period>=width:continue
    for shoulder,elbow,wrist,hand in [(5,7,9,91),(6,8,10,112)]:
     scale=np.linalg.norm(xy[i,5]-xy[i,6]);xy[i,wrist]=xy[i,elbow]+[0,.1*scale]
     xy[i,hand:hand+21]=xy[i,wrist]+[0,2]
   zz=dict(z,xy_px=xy,timestamp_us=np.arange(len(xy))*33333)
   a=simulate(PreviousRest,zz,'jitter');b=simulate(StableRest,zz,'jitter')
   rows.append(dict(id=p.stem,kind=f'injected_height_jitter_{width}',before=a['first_stop'],after=b['first_stop'],passed=b['first_stop']is not None and b['first_stop']>=start,synthetic_observation_fault=True))
 oldcheck=load('prior_stop_checks',P/'check_stop.py');oldcheck.After=StableRest;controls=oldcheck.controls()
 # A genuine raised-hand return resets the clock immediately, even one frame.
 for length in [1,2,8]:
  obj=StableRest();pose=oldcheck.pose
  for _ in range(12):obj.calibrate(pose(),640,480)
  obj.begin();first=None
  for i in range(150):
   p=pose(('R',))if i<40 or 57<=i<57+length else pose()
   if obj.update(p,640,480,i*33333)and first is None:first=i
  controls.append(dict(kind='genuine_hand_reentry',frames=length,first_stop=first,passed=first is not None and first>=57+length+23))
 write(D/'checks/STOP_CHECK.json',dict(passed=all(r['passed']for r in rows+controls),rows=rows,controls=controls,private_data=False,scope='Existing public frontal cropped RTMW observations and separately marked synthetic detection-fault controls. No user data used.'))
 print('STOP',len(rows),len(controls),'failures',[(r.get('id'),r['kind'],r.get('before'),r.get('after'))for r in rows+controls if not r['passed']],flush=True)
if __name__=='__main__':main()
