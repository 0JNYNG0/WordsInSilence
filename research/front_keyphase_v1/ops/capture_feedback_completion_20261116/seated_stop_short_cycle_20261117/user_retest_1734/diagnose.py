"""Saved inference and capture diagnostics only; no model or parameter selection."""
from pathlib import Path
import json,sys,hashlib
from collections import Counter
import numpy as np,av,cv2
D=Path(__file__).resolve().parent;R=D.parent;O=R.parent.parent
sys.path[:0]=[str(R),str(O/'word100_action_feedback_demo_20261108')]
from return_rest import ReturnToRest
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 pins=read(R/'web/ADMISSION.json')['pins'];assert all(sha(f)==h for f,h in pins.items());rows=[]
 for number,name in enumerate(['y8lllx6b','kgrc181c','_gjkghpw'],1):
  p=Path('C:/Users/local-user/AppData/Local/Temp')/('wis_action_demo_'+name);record=read(p/'inference_result.json');q=record['result'];ack=read(p/'browser_ack.json');diag=read(p/'auto_end_diagnostic.json');session=read(p/'rtmw/SESSION.json');raw=dict(np.load(p/'rtmw/observations.npz'));ts=raw['timestamp_us'];n=len(ts)
  assert q['stop_short_cycle_version']=='seated_rest_dual_window_v1'
  hashes={str(f):sha(f)for f in [p/'recorded.mp4',p/'inference_result.json',p/'browser_ack.json',p/'auto_end_diagnostic.json',p/'rtmw/observations.npz',p/'rtmw/SESSION.json']}
  tail=[f for f in diag['frames']if f['timestamp_us']>=ts[-1]-3000000];facts=[];rest=ReturnToRest();baseline=np.array(diag['rest_baseline'])
  for i,xy in enumerate(raw['xy_px']):
   g=rest.geometry(xy,int(raw['source_width']),int(raw['source_height']));facts.append(dict(frame=i,time=float(ts[i])/1e6,valid=g is not None,high=g[1].tolist()if g is not None else None,distance=np.linalg.norm(g[0]-baseline,axis=1).tolist()if g is not None else None))
  needed=sorted(set(np.linspace(30,min(n-1,90),16).astype(int).tolist()+np.linspace(max(0,n-150),n-1,16).astype(int).tolist()));images={}
  with av.open(str(p/'recorded.mp4'))as source:
   for i,frame in enumerate(source.decode(video=0)):
    if i in needed:images[i]=frame.to_ndarray(format='bgr24')
  sheets={}
  for section,ids in [('action',np.linspace(30,min(n-1,90),16).astype(int)),('end',np.linspace(max(0,n-150),n-1,16).astype(int))]:
   sheet=np.full((4*265,4*320,3),245,np.uint8)
   for j,i in enumerate(ids):
    if i not in images:continue
    x=j%4*320;y=j//4*265;sheet[y+25:y+265,x:x+320]=cv2.resize(images[i],(320,240));cv2.putText(sheet,f'{number}: f{i} {ts[i]/1e6:.2f}s',(x+5,y+18),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
   path=D/f'{number}_{section}.jpg';cv2.imwrite(str(path),sheet);sheets[section]=str(path)
  row=dict(number=number,source=str(p),recorded_at=record['recorded_at'],word=q['word'],label=q['label'],stop_reason=q['stop_reason'],overall_pass=q['overall_pass'],criteria=q['lesson_acceptance']['criteria'],events=q['lesson_acceptance'].get('events'),movement_evidence=q['lesson_acceptance'].get('movement_evidence'),tcn_top3=q['tcn_top3'],duration=float(ts[-1])/1e6,frames=n,stop_to_result_seconds=q['stop_to_result_seconds'],stop_to_browser_ack_seconds=ack['stop_to_browser_ack_seconds'],selective_retry=q.get('selective_retry'),session=session,diagnostic=diag,geometry=facts,sheets=sheets,source_hashes=hashes,tail_reasons=dict(Counter(f['reason']for f in tail)),longest_rest_dwell_seconds=max([0]+[(f['timestamp_us']-f['rest_since_us'])/1e6 for f in diag['frames']if f['rest_since_us']is not None]))
  rows.append(row);print(number,q['word'],q['stop_reason'],q['overall_pass'],row['tail_reasons'],row['longest_rest_dwell_seconds'],flush=True)
 assert all(sha(f)==h for r in rows for f,h in r['source_hashes'].items())and all(sha(f)==h for f,h in pins.items())
 write(D/'DIAGNOSIS.json',dict(rows=rows,private_diagnosis_only=True,training=False,tuning=False,candidate_selection=False,new_inference=False,runtime_changed=False,pins_unchanged=len(pins)))
if __name__=='__main__':main()
