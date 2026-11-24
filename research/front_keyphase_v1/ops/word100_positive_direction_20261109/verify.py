"""Separate saved-result checks and invariance/counterexample tests; no camera."""
from setup import *
import sys,time,numpy as np
sys.path.insert(0,str(D))
from positive_evidence import along_forearm,CONTRACTS,shape_evidence,projected_shape_support

def main():
 assert not (U/'VERIFICATION.json').exists();checks=[]
 def ck(name,condition):
  checks.append(dict(name=name,pass_=bool(condition)))
  if not condition:raise AssertionError(name)
 r=read(U/'RECHECK.json');rows=r['rows'];ck('93_conditions',len(rows)==93)
 for row in rows:
  ck(row['clip_id']+row['kind']+'prediction_preserved',sha(row['prediction'])==row['sha256'])
  d=read(row['decision'])['after'];ck('no_educational_claim',not d['educational_correctness_validated'])
  for c in d['cards']:
   if c['display_status']=='similar':
    ck('green_has_positive_components',bool(c['checked_components']))
    ck('green_has_all_shapes',all(s['state']=='supported' for s in c['shape_evidence']))
    ck('green_has_direction_if_required',not c['motion_evidence'] or c['motion_evidence']['state']=='supported')
   ck('uncertainty_not_wrong',c['educational_wrong'] is False)
 for kind in ['full_normal','crop_normal','speed0.75','speed1.25','translation_scale','gap2']:
  rr=[x for x in rows if x['kind']==kind];ck(kind+'_no_orange',sum(x['orange_review_cues'] for x in rr)==0)
 for kind in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']:
  rr=[x for x in rows if x['kind']==kind];ck(kind+'_not_green',all(x['after'][0]!='supported' for x in rr));ck(kind+'_visible_review',all(x['orange_review_cues']>0 for x in rr))
 for row in [x for x in rows if x['kind']=='translation_scale']:ck('invariant_channels',row['max_channel_delta']<1e-4)
 # Generic movement engine: both hands, body translation/rotation/scale, normal
 # shorter excursion, steady/jitter, reverse, and true observation gaps.
 ts=np.arange(30)*33333;phase=np.zeros((30,14));phase[:,4]=1
 x=np.zeros((30,280),np.float32);x[:,186:279]=1;q=x[:,:102].reshape(-1,51,2)
 q[:,3]=[0,0];q[:,5]=[1,0]
 for i in [30,35,39,43,47]:q[:,i,0]=np.linspace(.2,.9,30)
 con=CONTRACTS['brush_arm'];baseline=along_forearm(x,ts,phase,con);ck('forward',baseline['state']=='supported')
 for scale in [.5,1.5]:
  for angle in [0,.5,1.3]:
   y=x.copy();p=y[:,:102].reshape(-1,51,2);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);p[:]=(p@rot)*scale+[3,-2]
   v=along_forearm(y,ts,phase,con);ck('rigid_body_invariance',v['state']=='supported' and abs(v['signed_change']-baseline['signed_change'])<1e-5)
 for length in [.7,.5]:
  y=x.copy();p=y[:,:102].reshape(-1,51,2)
  for i in [30,35,39,43,47]:p[:,i,0]=.2+(p[:,i,0]-.2)*length
  ck('shorter_same_direction',along_forearm(y,ts,phase,con)['state']=='supported')
 ck('reverse',along_forearm(x[::-1].copy(),ts,phase,con)['state']=='contradicted')
 y=x.copy();p=y[:,:102].reshape(-1,51,2)
 for i in [30,35,39,43,47]:p[:,i,0]=.5+.01*np.sin(np.arange(30))
 ck('jitter_not_direction',along_forearm(y,ts,phase,con)['state']=='unknown')
 y=x.copy();y[10:25,186+5]=0;ck('missing_anchor_unknown',along_forearm(y,ts,phase,con)['state']=='unknown')
 y=x.copy();p=y[:,:102].reshape(-1,51,2);p[:,4]=p[:,3];p[:,6]=p[:,5]
 for l,rr in zip([9,14,18,22,26],[30,35,39,43,47]):p[:,l]=p[:,rr]
 c={**con,'moving_hand':'L','anchor_arm':'R'};ck('left_right_symmetric',along_forearm(y,ts,phase,c)['state']=='supported')
 # Class probabilities alone cannot turn missing points or a contradictory
 # finger configuration into a confident correction.
 attrs=np.zeros((30,7));attrs[:,1]=.99;attrs[:,3]=.01
 y=x.copy();y[:,186+30:186+51]=0
 ck('missing_points_not_positive',shape_evidence(attrs,y,ts,[0,29],{'1':1})[0]['state']=='unknown')
 attrs[:,3]=.99;ck('mutually_conflicting_unknown',shape_evidence(attrs,x,ts,[0,29],{'1':1})[0]['state']=='unknown')
 # Protection hashes are verified independently of the implementation.
 protected=read(U/'FROZEN.json')['protected']
 for path,pin in protected.items():
  expected=pin['sha256'] if isinstance(pin,dict) else pin
  ck('protected:'+path,sha(path)==expected)
 # Repeated benchmark is computation timing only, not repeated model scoring.
 sample=next(z for z in rows if z['kind']=='crop_normal' and z['word']=='WORD1133');z=np.load(sample['prediction']);dur=[]
 for _ in range(50):
  t=time.perf_counter();along_forearm(z['x'],z['timestamp_us'],z['phase'],con);dur.append((time.perf_counter()-t)*1000)
 write(U/'VERIFICATION.json',dict(status='PASS',checks=len(checks),detail=checks,protected=len(protected),implementation_and_verifier='Same Codex, separate checks; not an independent human annotator',
  direction_median_ms=float(np.median(dur)),combined_refinement_median_ms=float(np.median([z['milliseconds'] for z in rows])),public_condition_count=len(rows)))
 print(json.dumps({'checks':len(checks),'protected':len(protected),'direction_ms':float(np.median(dur))}))
if __name__=='__main__':main()
