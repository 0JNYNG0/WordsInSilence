"""Public regression for a vertical-only candidate; no personal records opened."""
from pathlib import Path
import sys,json,copy,importlib.util
import numpy as np
import torch
import cv2
V=Path(__file__).resolve().parent;U=V.parent;O=U.parent;sys.path.insert(0,str(U))
from action_runtime import ActionObserver
from coach import Coach
from public_probe import run_case,write
spec=importlib.util.spec_from_file_location('prior_public_cases',U/'usability_repair_20261116/public_compare.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class VerticalObserver(ActionObserver):
 def __init__(self):
  super().__init__();self.vertical=copy.deepcopy(self.net.motion_head)
  ck=torch.load(V/'MOTION_CANDIDATE.pt',map_location='cpu',weights_only=True)
  self.vertical.load_state_dict(ck['state']);self.vertical.eval();self.vertical.requires_grad_(False)
 def movement(self,start_us,end_us,axis,moving='R',anchor='L',min_samples=None):
  if int(axis)!=0 or not hasattr(self,'vertical'):return super().movement(start_us,end_us,axis,moving,anchor,min_samples)
  original=self.net.motion_head
  try:
   self.net.motion_head=self.vertical
   return super().movement(start_us,end_us,axis,moving,anchor,min_samples)
  finally:self.net.motion_head=original

def main():
 assert not (V/'CANDIDATE_COMPARISON.json').exists()
 write(V/'AXIS_SCOPE_DECISION.json',dict(status='Fixed revised candidate before regression',
  first_attempt='MOTION_RESULTS.json: unrestricted motion-head replacement regressed one historical forearm hold. Preserve that failure; do not deploy unrestricted head.',
  design='The reproduced problem is vertical up/down. Route only axis0 to copied updated head; forearm/other axes retain byte-identical prior head. Both hands share vertical head. No word IDs or source frame branches.',
  gate='All24 frozen prior conditions; both public pixel views; delayed repeat and held-out crop; oldDEV32 native no regression. No confidence/distance changes.',private_inputs=False,training='No extra run; same fixed candidate checkpoint',new_person_accuracy=None))
 m=Coach(action=VerticalObserver()); results=[]
 with np.load(U/'source/reference.npz')as z:raw={k:z[k].copy()for k in z.files}
 for name,a,expected,_ in old.cases(raw):
  row=run_case(m,'candidate_'+name,a);row['expected']=expected;row['passed']=row['completed']==expected;results.append(row)
 for name in ['public_video_640_letterbox','public_video_upper_body_640']:
  with np.load(V/(name+'.npz'))as z:a={k:z[k].copy()for k in z.files}
  row=run_case(m,'candidate_'+name,a);row['expected']=[True,True];row['passed']=row['completed']==row['expected'];results.append(row)
 # Wait condition is still controlled source augmentation, not an independent person.
 ix=np.r_[np.arange(52),np.repeat(51,480),np.arange(52,len(raw['timestamp_us']))];a={k:v.copy()for k,v in raw.items()}
 a['xy_px']=a['xy_px'][ix];a['score']=a['score'][ix]
 for j in range(52,532):a['xy_px'][j,:,1]+=np.sin(j*.17)*.5
 a['frame_id']=np.arange(len(ix));a['timestamp_us']=np.rint(a['frame_id']*1e6/30).astype(np.int64)
 row=run_case(m,'candidate_delayed_repeat',a);row['expected']=[True,True];row['passed']=row['completed']==row['expected'];results.append(row)
 sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
 from stream_runtime import Engine
 engine=Engine();cap=cv2.VideoCapture(str(O.parent/'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4')); pp=[];ss=[]
 while True:
  ok,bgr=cap.read()
  if not ok:break
  im=cv2.resize(bgr[0:780,440:1480],(640,480));r=engine.infer(im);pp.append(r['xy']);ss.append(r['score'])
 cap.release();a=dict(raw);a.update(xy_px=np.array(pp),score=np.array(ss),source_width=np.int64(640),source_height=np.int64(480));np.savez_compressed(V/'heldout_crop.npz',**a)
 before=run_case(Coach(model=m.model),'before_heldout_crop',a)
 row=run_case(m,'candidate_heldout_crop',a);row['before']=before;row['expected']=[True,True];row['passed']=row['completed']==row['expected'];results.append(row)
 # No DEV fitting: original head retained on all nonvertical axes.
 d=json.loads((V/'MOTION_RESULTS.json').read_text('utf-8')); dev=d['results']['old_dev']['rows']; rr=[]
 for r in dev:
  q=r['after']if r['axis']==0 else r['before'];rr.append(dict(target=r['target'],pred=int(np.argmax(q)),reverse=r['reverse'],axis=r['axis'],old=int(np.argmax(r['before']))))
 native=[r for r in rr if not r['reverse']];dev_ok=all(r['pred']==r['target'] for r in native)
 write(V/'CANDIDATE_COMPARISON.json',dict(passed=all(r['passed']for r in results)and dev_ok,rows=results,public_passes=sum(r['passed']for r in results),public_total=len(results),old_dev_normal_correct=sum(r['pred']==r['target']for r in native),old_dev_normal_total=len(native),dev=rr,training_private_data=False,production_applied=False))
 print('FINAL',sum(r['passed']for r in results),len(results),'DEV',dev_ok,flush=True)

if __name__=='__main__':main()
