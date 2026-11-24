from probe_stop import *

def pose(up=()):
 p=np.full((133,2),np.nan);p[5]=[220,150];p[6]=[420,150];p[7]=[220,300];p[8]=[420,300]
 for h in up:
  sh,el,wr,hand=(5,7,9,91)if h=='L'else(6,8,10,112)
  p[el]=p[sh]+[30,25];p[wr]=p[sh]+[15,-60];p[hand:hand+21]=p[wr]
 return p

def controls():
 rows=[]
 for hands in [('L',),('R',),('L','R')]:
  for kind in ['crossing','short_dropout','raised_hold','low_visible_hold']:
   r=After()
   for _ in range(12):r.calibrate(pose(),640,480)
   r.begin();stops=[]
   for i in range(150):
    p=pose(hands)if i<100 else pose()
    if kind=='crossing'and 40<=i<85:p[9]=p[10]=[320,160];p[91:]=[320,160]
    if kind=='short_dropout'and 40<=i<45:p[9:11]=np.nan;p[91:]=np.nan
    if kind=='low_visible_hold'and 40<=i<100:
     p=pose()
     for el,wr,hand in [(7,9,91),(8,10,112)]:
      p[wr]=p[el]+[0,15];p[hand:hand+21]=p[wr];p[hand+5]=p[wr]+[-18,10];p[hand+9]=p[wr]+[0,20];p[hand+17]=p[wr]+[18,10]
    if r.update(p,640,480,i*33333):stops.append(i)
   passed=bool(stops)and 122<=stops[0]<=124
   rows.append(dict(kind=kind,hands=hands,first_stop=stops[0]if stops else None,passed=passed))
 for kind in ['all_points_missing','stale_stream','no_action']:
  r=After()
  for _ in range(12):r.calibrate(pose(),640,480)
  r.begin();r.update(pose(('R',))if kind!='no_action'else pose(),640,480,1000000)
  stop=[]
  for i in range(70):stop.append(r.update(np.full((133,2),np.nan)if kind=='all_points_missing'else pose(),640,480,1100000+i*(300000 if kind=='stale_stream' else 33333)))
  rows.append(dict(kind=kind,passed=not any(stop)))
 return rows

def main():
 torch.set_num_threads(2);cv2.setNumThreads(2);engine=Engine();rows=[]
 for r in sources():
  if r['person']not in [1,2,11]:continue
  baseline=read(G/'public_frontal_evidence_repair_20261117/web'/('normal_'+r['id']+'.json'))
  last=max(e['end']for e in baseline['lesson_acceptance']['events'])
  for border in [.18,.35]:
   z=extract(r,engine,border);a=run_stop(Before,z);b=run_stop(After,z)
   passed=b['calibrated']and b['first_stop']is not None and b['first_stop']>last
   rows.append(dict(id=r['id'],border=border,passed=bool(passed),last_action=last,before=a,after=b))
   print('STOP',r['id'],border,a['first_stop'],b['first_stop'],bool(passed),flush=True)
  if r['person']in [2,11]:
   z=extract(r,engine,.18,aspect=True);a=run_stop(Before,z);b=run_stop(After,z)
   passed=b['calibrated']and b['first_stop']is not None and b['first_stop']>last
   rows.append(dict(id=r['id'],border=.18,aspect_preserved=True,passed=bool(passed),last_action=last,before=a,after=b))
   print('STOP ASPECT',r['id'],a['first_stop'],b['first_stop'],bool(passed),flush=True)
 cs=controls();write(D/'checks/STOP_CHECK.json',dict(rows=rows,controls=cs,passed=all(r['passed']for r in rows+cs),physical_camera=False,private_data=False,scope='RTMW rerun on public frontal upper-body crops; synthetic safety controls separately identified'))
 print('STOP CONTROLS',[(r['kind'],r.get('hands'),r.get('first_stop'))for r in cs if not r['passed']],flush=True)
if __name__=='__main__':main()
