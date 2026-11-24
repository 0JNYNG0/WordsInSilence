from bootstrap import *
from auto_end import ReturnToRest
import numpy as np,json
def pose(up=()):
 p=np.full((133,2),np.nan);p[5]=[220,150];p[6]=[420,150];p[7]=[220,300];p[8]=[420,300]
 for h in up:
  shoulder,elbow,wrist,hand=(5,7,9,91) if h=='L' else (6,8,10,112)
  p[elbow]=p[shoulder]+[30,25];p[wrist]=p[shoulder]+[15,-60];p[hand:hand+21]=p[wrist]
 return p
rows=[]
for hands in [('L',),('R',),('L','R')]:
 r=ReturnToRest()
 for _ in range(12):r.calibrate(pose(),640,480)
 r.begin();stops=[]
 for f in range(120):
  p=pose(hands) if f<80 else pose()
  # Temporary hand-only occlusion while arm remains raised.
  if 40<=f<45:p[9:11]=np.nan;p[91:]=np.nan
  if r.update(p,640,480,round(f/30*1e6)):stops.append(f)
 rows.append(dict(hands=hands,first_stop=stops[0] if stops else None,temporary_occlusion_safe=not any(f<80 for f in stops),returned_stops=bool(stops) and 102<=stops[0]<=104))
r=ReturnToRest()
for _ in range(12):r.calibrate(pose(),640,480)
r.begin();r.update(pose(('R',)),640,480,1000000)
missing=[]
for f in range(60):missing.append(r.update(np.full((133,2),np.nan),640,480,1100000+f*33333))
checks=dict(one_and_two_hand_stop=all(x['returned_stops'] for x in rows),short_occlusion_safe=all(x['temporary_occlusion_safe'] for x in rows),all_points_missing_not_stop=not any(missing))
out=dict(passed=all(checks.values()),checks=checks,rows=rows,scope='Synthetic stop-state regression; not sign correctness or physical camera accuracy')
(U/'AUTO_END_CONTRACT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out));assert out['passed']
