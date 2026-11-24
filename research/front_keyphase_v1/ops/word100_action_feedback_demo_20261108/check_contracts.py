from task import *
import numpy as np
from auto_end import ReturnToRest
def pose(high=(),raised_elbows=False):
 p=np.full((133,2),np.nan);p[5:9]=[[200,150],[400,150],[220,330],[380,330]];p[9:11]=[[220,520],[380,520]]
 for side in high:
  wrist,start=(9,91) if side=='L' else (10,112);p[wrist]=[250 if side=='L' else 350,230];p[start:start+21]=p[wrist]
 if raised_elbows:p[7:9,1]=200
 return p
def init():
 r=ReturnToRest()
 for _ in range(10):r.calibrate(pose(),640,480)
 r.begin(0);return r
def main():
 write(U/'CAPTURE_CONTRACT_FROZEN.json',dict(source_sha256=sha(U/'auto_end.py'),checks=['one_hand','two_hand','short_occlusion','raised_arms_occluded','partial_action_then_rest','stale_gap','no_baseline','fresh_timestamps'],capture_stop_not_semantic_completion=True))
 results=[]
 for hands in [('R',),('L','R')]:
  r=init();states=[]
  for i in range(70):states.append(r.update(pose(hands if i<35 else ()),640,480,i*33333))
  results.append(dict(test='one_hand' if len(hands)==1 else 'two_hand',pass_=not any(states[:35]) and any(states[35:])))
 r=init();s=[]
 for i in range(90):
  p=pose(('R',)) if not 40<=i<46 else pose();s.append(r.update(p,640,480,i*33333))
 results.append(dict(test='short_occlusion_6_frames',pass_=not any(s)))
 r=init();s=[]
 for i in range(90):
  p=pose(('R',),True)
  if i>25:p[9:]=np.nan
  s.append(r.update(p,640,480,i*33333))
 results.append(dict(test='arms_raised_hands_occluded',pass_=not any(s)))
 r=init()
 for i in range(35):r.update(pose(('R',)),640,480,i*33333)
 result=[r.update(pose(),640,480,(35+i)*33333) for i in range(30)]
 results.append(dict(test='one_of_two_actions_then_lower_arms_still_stops',pass_=any(result),semantic_completion_checked_here=False))
 r=init()
 for i in range(35):r.update(pose(('R',)),640,480,i*33333)
 r.update(pose(),640,480,1200000);z=r.update(pose(),640,480,3000000);results.append(dict(test='stale_gap_does_not_complete_rest_timer',pass_=not z))
 r=ReturnToRest();r.begin(0);results.append(dict(test='no_baseline_no_automatic_stop',pass_=not any(r.update(pose(),640,480,i*33333) for i in range(100))))
 a=read(U/'RELEASE_DECISION.json');results.append(dict(test='no_unvalidated_correct_wrong_release',pass_=not a['automatic_educational_correction_enabled'] and not a['automatic_pass_enabled']))
 # Ordinary DTW necessarily reaches the last reference index. This is not evidence of completion.
 ex=read(U/'examples/WORD0739_edited/result.json');c=next(x for x in ex['cards'] if x['status']=='missing_candidate');results.append(dict(test='missing_card_has_no_manufactured_query_match',pass_=c['query_indices'] is None and c['reference_only'] and ex['path'][-1][1]==ex['prototype_length']-1,dtw_reaches_reference_end=True))
 write(U/'CAPTURE_CONTRACT_TEST.json',dict(pass_=all(r['pass_'] for r in results),rows=results,real_camera=False,synthetic_geometry_not_accuracy_test=True));assert all(r['pass_'] for r in results),results;print('PASS',len(results))
if __name__=='__main__':main()
