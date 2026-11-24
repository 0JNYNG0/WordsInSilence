"""Interpret frozen public controls without selecting thresholds or models."""
import prepare as R
import numpy as np,sys
from collections import Counter,defaultdict

def run():
 assert R.read(R.U/'CONTROLS_VERIFIED.json')['pairs']==5593
 done=R.read(R.U/'CONTROLS_DONE.json');normal=R.read(R.U/'RANGE_DONE.json');R.check(done['pins']);R.check(normal['pins']);counts=defaultdict(Counter);words=defaultdict(lambda:defaultdict(Counter));drifts=defaultdict(list)
 for row in done['rows']:
  kind=row['kind'];before=R.load(row['original_range']);after=R.load(row['output']);old=R.load(row['original_pair']);q=row['query'];d=before['domain'];cut=np.array(row['target_original_indices']);target=(old['phase_old']>=cut.min()-2)&(old['phase_old']<=cut.max()+2)&d.any((1,2));drifts[kind].append(row['alignment_drift_mean']);counts[kind]['pairs']+=1
  for mode in ['point','sequence']:
   for group,sl in [('angle',slice(0,15)),('position',slice(15,23))]:
    name=mode+'_'+group;b=before[mode+'_status'][:,:,sl];a=after[mode+'_status'][:,:,sl];grid=d[:,:,sl];both=(b>=4)&(a>=4);hold=(b>=4)&(a>0)&(a<4);new=(b==4)&(a==5)
    counts[kind][name+'_domain']+=int(grid.sum());counts[kind][name+'_common_supported']+=int(both.sum());counts[kind][name+'_supported_lost']+=int(hold.sum());counts[kind][name+'_within_to_beyond']+=int(new.sum());counts[kind][name+'_beyond_to_within']+=int(((b==5)&(a==4)).sum())
    words[kind][q][name+'_baseline_target_supported']+=int((b[target]>=4).sum());words[kind][q][name+'_target_new_beyond']+=int(new[target].sum());words[kind][q][name+'_target_lost_support']+=int(hold[target].sum());words[kind][q][name+'_target_frames']+=int(target.sum())
 totals={}
 for kind,qq in words.items():
  summary=Counter(queries=len(qq),pairs=counts[kind]['pairs'])
  for q,c in qq.items():
   for mode in ['point','sequence']:
    for group in ['angle','position']:
     name=mode+'_'+group
     if c[name+'_target_frames']==0:state='no_original_target_correspondence'
     elif c[name+'_target_new_beyond']>0:state='new_local_excess'
     elif c[name+'_target_lost_support']>0:state='new_local_uncertainty_only'
     elif c[name+'_baseline_target_supported']>0:state='no_new_local_signal_with_some_baseline_support'
     else:state='baseline_target_already_insufficient'
     summary[name+'_'+state]+=1
  totals[kind]=dict(summary=summary,cell_transitions=dict(counts[kind]),mean_alignment_drift_inputframes=float(np.mean(drifts[kind])),max_mean_alignment_drift_inputframes=float(np.max(drifts[kind])),perquery={q:dict(c) for q,c in qq.items()})
 nc=normal['counts'];coverage={}
 for mode in ['point','sequence']:
  for group in ['angle','position']:
   key=mode+'_'+group;state={str(k):nc[key+'_state'+str(k)] for k in range(1,6)};total=sum(state.values());coverage[key]=dict(states=state,domain=total,sufficient=state['4']+state['5'],sufficient_percent=round(100*(state['4']+state['5'])/total,2),within_observed_percent_of_supported=round(100*state['4']/max(1,state['4']+state['5']),2))
 result=dict(at=R.now(),normal=coverage,controls=totals,interpretation='599new+300reused publicTRAIN. Original899 observed examples, not gold-correctness. Each original query withheld from band calibration. Body position uses raw observation, finger angles use same-MediaPipe full/crop image agreement proxy. Distinct person IDs not independent identity certification. Cell rates are not word recognition accuracy, all controls use already extracted observations. New uncertainty is never improvement. Three removed time samples not necessarily linguistically required action.',source_pins={str(p):R.sha(p) for p in [R.U/'RANGE_DONE.json',R.U/'RANGE_VERIFIED.json',R.U/'CONTROLS_DONE.json',R.U/'CONTROLS_VERIFIED.json',R.U/'INPUTS_VERIFIED.json',R.U/'VERIFIED_IMAGES.json']},whole_action_correct=None,validated_correction=None,recognition_change=None,training=0,operating_changed=False)
 R.write(R.U/'ANALYSIS.json',result)
 print('EMPIRICAL_REFERENCE_COVERAGE',coverage,flush=True)
 for k,v in totals.items():print('CONTROL_SUMMARY',k,dict(v['summary']),flush=True)
if __name__=='__main__':run()
