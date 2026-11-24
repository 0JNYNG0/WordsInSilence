from final_cycle import *
from verify_scope import verify
import importlib.util
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 if '--regression'in sys.argv:
  m=load('cycle_final_public_regression',J/'check_regression.py');m.J=M;m.main(LearnedCycleLesson,'final_regression');return
 m=load('cycle_final_episodes',M/'check_episode_controls.py');m.main(LearnedCycleLesson,'FINAL_EPISODE_CONTROLS.json')
 old,new=TemporalVisibilityLesson(),LearnedCycleLesson();out=M/'final_extra';out.mkdir(exist_ok=True);rows=[];controls=[];missing=[]
 for r in read(M/'confirmation/INPUTS.json')['rows']:
  raw=dict(np.load(r['raw']));a,b=old.analyze(raw),new.analyze(raw);rows.append(dict(id=r['id'],**verify(a,b),before={c['id']:c['status']for c in a['criteria']},after={c['id']:c['status']for c in b['criteria']},events_before=len(a['events']),events_after=len(b['events'])));write(out/(r['id']+'.json'),dict(before=a,after=b))
 for r in all_sources():
  if r['id']in[f'NIA_SL_WORD1290_REAL{p:02d}_{v}'for p in[12,18]for v in['L','R']]:
   z=dict(np.load(r['raw']));cut=48 if 'REAL12_'in r['id']else 62;hold=48 if 'REAL12_'in r['id']else 53;n=len(z['frame_id'])
   for kind in ['partial','constant']:
    raw={k:v.copy()for k,v in z.items()}
    if kind=='partial':
     for k in ['xy_px','score','timestamp_us','frame_id']:raw[k]=raw[k][:cut]
    else:raw['xy_px']=np.repeat(raw['xy_px'][hold:hold+1],n,axis=0)
    q=new.analyze(raw);controls.append(dict(id=r['id'],kind=kind,events=len(q['events']),passed=len(q['events'])<2))
  if r['id']in['NIA_SL_WORD1290_REAL11_L','NIA_SL_WORD1290_REAL15_R']:
   z=dict(np.load(r['raw']));start=baseline(r['id'])['events'][0]['start']
   for hand,offset in [('L',91),('R',112)]:
    for size in [1,3,8]:
     raw={k:v.copy()for k,v in z.items()};ix=np.arange(start,start+size);raw['xy_px'][start:start+size,offset+13:offset+21]=np.nan;raw['score'][start:start+size,offset+13:offset+21]=0;a=old.analyze(raw);b=new.analyze(raw);sa=np.array(a['hand_form_evidence'][hand]['states']);sb=np.array(b['hand_form_evidence'][hand]['states']);ob=new.shape.observations[hand];missing.append(dict(id=r['id'],hand=hand,frames=size,passed=bool(not np.any((sb[ix]==0)&(sa[ix]!=0))and not ob['available'][ix,3:].any())))
 write(out/'RESULT.json',dict(rows=rows,controls=controls,missing=missing,passed=all(r['passed']for r in rows+controls+missing),scope='Known-source regression; confirmation8LR inspected before initial candidate output, then reused for final composite, not untouched holdout.',private_data=False));print('FINAL_EXTRA',read(out/'RESULT.json')['passed'],flush=True)
if __name__=='__main__':main()
