"""Read-only replay of the SAME frozen observers for requested diagnosis.
No training, alternative-model comparison, thresholds or production changes.
"""
from pathlib import Path
import sys,json,hashlib,collections
N=Path(__file__).resolve().parent;G=N.parent.parent;O=G.parent
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server
sys.path[:0]=[str(G/'public_visibility_roles_20261116'),str(G/'apple_v1')]
from apple_candidate import RoleEvidenceCandidate
from guard import quarantine
from auto_end import ReturnToRest
import numpy as np,torch
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
 torch.set_num_threads(2);admission=read(N.parent/'web/ADMISSION.json');pins=admission['pins'];assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 rows=read(N/'DISCOVERY.json')['rows'];net=RoleEvidenceCandidate();checks=[]
 for j in [0,3]:
  p=Path(rows[j]['directory']);raw=dict(np.load(p/'rtmw/observations.npz'));saved=read(p/'inference_result.json')['result'];obs=quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']));z=dict(raw);z['xy_px']=obs['xy_px'];q=net.analyze(z)
  before={c['id']:c['status']for c in saved['lesson_acceptance']['criteria']};after={c['id']:c['status']for c in q['criteria']};assert before==after
  mouth=next(e for e in q['events']if e['region']==1);a,b=mouth['start'],mouth['end'];s=q['observations']['shape']['R'];f=np.array(s['fingers']);av=np.array(s['available']);count=collections.Counter(tuple(x)for x in f[a:b+1]);checks.append(dict(attempt=j+1,same_original_criteria=True,interval=[a,b],finger_order=['index','middle','ring','little'],shape_states=[dict(values=list(k),frames=v)for k,v in count.items()],availability_all=int(av[a:b+1].all(1).sum()),interval_frames=b-a+1,criteria=q['criteria']))
 auto=[];ending=ReturnToRest()
 for j,r in enumerate(rows,1):
  p=Path(r['directory']);raw=dict(np.load(p/'rtmw/observations.npz'));q=read(p/'inference_result.json')['result'];meta=read(p/'rtmw/SESSION.json');width=int(raw['source_width']);height=int(raw['source_height']);facts=[]
  for i in list(range(min(35,len(raw['frame_id']))))+list(range(max(35,len(raw['frame_id'])-90),len(raw['frame_id']))):
   g=ending.geometry(raw['xy_px'][i],width,height);facts.append(dict(frame=i,time=float(raw['timestamp_us'][i])/1e6,geometry_valid=g is not None,hand_high=g[1].tolist()if g is not None else None,elbows_below_shoulders=bool((g[0][:,1]>.12).all())if g is not None else None))
  auto.append(dict(attempt=j,stop=q['stop_reason'],calibrated=q['auto_end_calibrated'],observation_rows=len(raw['frame_id']),stream_received=meta['received'],stream_completed=meta['completed'],stream_dropped=meta['dropped'],stream_errors=meta['errors'],missing_ranges=meta['missing_input_id_ranges'],return_geometry=facts))
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 result=dict(scope='User-requested diagnosis only, same admitted weights',private_data=True,training_use=False,tuning_use=False,candidate_selection_use=False,checks=checks,auto_end=auto,pins_unchanged=len(pins),preview_calibration_frames_not_saved=True)
 (N/'diagnostic_only/FROZEN_DIAGNOSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
 for r in checks:print('DIAGNOSIS',r['attempt'],r['same_original_criteria'],r['availability_all'],r['interval_frames'],r['shape_states'])
 for r in auto:print('CAPTURE',r['attempt'],r['stop'],r['calibrated'],r['stream_completed'],r['stream_dropped'])
if __name__=='__main__':main()
