"""Public controlled regression of diagnostics/role/gap repairs, not signer accuracy."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
V=Path(__file__).resolve().parent;U=V.parent;sys.path.insert(0,str(U))
from coach import Coach

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cases(raw):
 def cp():return {k:v.copy()for k,v in raw.items()}
 out=[]
 out.append(('native_reference',cp(),[True,True],False))
 a=cp()
 for k in ['xy_px','score','timestamp_us','frame_id']:a[k]=a[k][:49]
 out.append(('first_stage_only',a,[True,False],False))
 a=cp();a['xy_px'][34:46]=a['xy_px'][34:46][::-1];a['score'][34:46]=a['score'][34:46][::-1]
 out.append(('reverse_sweep_frames',a,[False,False],False))
 for name,frame in [('hold_two_fists',51),('one_downstroke_then_hold',56)]:
  a=cp();a['xy_px'][frame+1:]=a['xy_px'][frame];a['score'][frame+1:]=a['score'][frame];out.append((name,a,[True,False],False))
 a=cp();a['xy_px'][37:44,91:]=np.nan;out.append(('missing_sweep_hands',a,[False,False],False))
 a=cp();a['xy_px'][:]=np.nan;out.append(('all_missing',a,[False,False],False))
 for name,factor in [('fast_timestamps',.65),('slow_timestamps',1.5)]:
  a=cp();a['timestamp_us']=(a['timestamp_us']*factor).astype(np.int64);out.append((name,a,[True,True],False))
 a=cp();a['xy_px']=a['xy_px']*.75+np.array([90,40],np.float32);out.append(('scale_translation',a,[True,True],False))
 a=cp()
 for i in range(49,80):
  for start in [91,112]:a['xy_px'][i,start:start+21]=a['xy_px'][i,start]+raw['xy_px'][38,start:start+21]-raw['xy_px'][38,start]
 out.append(('open_hands_in_fists_motion',a,[True,False],False))
 a=cp()
 for i in range(34,46):a['xy_px'][i,112:]=a['xy_px'][i,112]+raw['xy_px'][55,112:]-raw['xy_px'][55,112]
 out.append(('closed_sweep_hand',a,[False,False],False))
 for side,pose in [(91,9),(112,10)]:
  a=cp();a['xy_px'][49:80,side:side+21]=raw['xy_px'][51,side:side+21];a['xy_px'][49:80,pose]=raw['xy_px'][51,pose]
  out.append((f'held_hand_{side}',a,[True,False],False))
 for step in [2,3]:
  a=cp()
  for k in ['xy_px','score','timestamp_us','frame_id']:a[k]=a[k][::step]
  out.append((f'subsample_{30//step}fps',a,[True,True]if step==2 else[False,False],False))
 a=cp();a['xy_px'][30:47,92:112]=np.nan;out.append(('support_fingers_unavailable',a,[True,True],True))
 a=cp()
 for i in range(30,47):a['xy_px'][i,91:112]=a['xy_px'][i,91]+raw['xy_px'][55,91:112]-raw['xy_px'][55,91]
 out.append(('support_hand_closed',a,[True,True],True))
 a=cp();a['xy_px'][30:47,[7,9]]=np.nan;out.append(('support_forearm_unavailable',a,[False,False],True))
 a=cp();a['xy_px'][30:47,112:]=np.nan;out.append(('moving_hand_unavailable',a,[False,False],True))
 # Public held frame insertion separates brief stationary occlusion from unseen motion.
 def hold_bottom():
  a=cp();idx=np.r_[np.arange(57),np.repeat(56,12),np.arange(57,len(raw['timestamp_us']))]
  for k in ['xy_px','score']:a[k]=a[k][idx]
  a['frame_id']=np.arange(len(idx));a['timestamp_us']=np.rint(a['frame_id']*1e6/30).astype(np.int64);return a
 a=hold_bottom();out.append(('normal_bottom_hold',a,[True,True],True))
 a=hold_bottom();a['xy_px'][62,112:]=np.nan;out.append(('brief_missing_at_observed_hold',a,[True,True],True))
 a=hold_bottom();a['xy_px'][57:69,112:]=np.nan;out.append(('long_missing_after_first_down',a,[True,False],True))
 a=cp();a['xy_px'][57:61,91:]=np.nan;out.append(('hidden_upward_return',a,[True,False],True))
 return out

def main():
 out=V/('PUBLIC_COMPARISON'+(sys.argv[1]if len(sys.argv)>1 else'')+'.json');assert not out.exists()
 spec=importlib.util.spec_from_file_location('usability_frozen_before',V/'before/coach.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 before=old.Coach();after=Coach(model=before.model)
 with np.load(U/'source/reference.npz')as z:raw={k:z[k].copy()for k in z.files}
 historical=json.loads((U/'action_integration/PUBLIC_COMPARISON_FT.json').read_text(encoding='utf-8'))
 historical={r['case']:r['after']for r in historical['rows']}
 pins={str(p.relative_to(U)):digest(p)for p in [U/'coach.py',U/'action_runtime.py']};rows=[]
 for name,a,expected,new in cases(raw):
  row=dict(case=name,expected=expected,frames=len(a['timestamp_us']),derived_from='one approved public SEN0354 reference')
  models=[('before',before),('after',after)]if new else[('after',after)]
  if not new:row['before']=dict(completed=historical[name]['completed'],evidence='frozen previous comparison; same source and condition')
  for side,m in models:
   m.reset();outputs=[];times=[]
   for i,t in enumerate(a['timestamp_us']):
    st=time.perf_counter();ob=m.update(a['xy_px'][i],a['score'][i],int(t),int(a['source_width']),int(a['source_height']));times.append((time.perf_counter()-st)*1000)
    ob.pop('assessed_xy',None);outputs.append(ob)
   cats={}
   for ob in outputs:
    cat=(ob.get('diagnostic')or{}).get('category','absent');cats[cat]=cats.get(cat,0)+1
   row[side]=dict(completed=m.completed.copy(),events=m.events,category_counts=cats,
    first_completion_received_frame=[next((i for i,o in enumerate(outputs)if o['completed'][stage]),None)for stage in [0,1]],
    median_ms=float(np.median(times[6:])),p95_ms=float(np.quantile(times[6:],.95)),
    held_updates=sum((o.get('diagnostic')or{}).get('held_progress')is True for o in outputs))
   if new and side=='after':(V/f'TRACE_{name}{sys.argv[1]if len(sys.argv)>1 else""}.json').write_text(json.dumps(outputs,ensure_ascii=False),encoding='utf-8')
  row['passed']=row['after']['completed']==expected;rows.append(row)
  print(json.dumps(dict(case=name,before=row['before']['completed'],after=row['after']['completed'],passed=row['passed']),ensure_ascii=False),flush=True)
 report=dict(passed=all(r['passed']for r in rows),passes=sum(r['passed']for r in rows),total=len(rows),rows=rows,code_hashes=pins,
  code_unchanged=all(digest(U/p)==h for p,h in pins.items()),source_sha256=digest(U/'source/reference.npz'),
  scope=__doc__,training=False,private_data=False,camera=False,new_person_accuracy=None,
  timing='Cached RTMW -> learned hand/action/controller only; excludes webcam,RTMW,network,render')
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],passes=report['passes'],total=report['total'])))
if __name__=='__main__':main()
