"""Recount saved status transitions independently of the reporting aggregator."""
import prepare as R
import numpy as np
from collections import defaultdict, Counter

def main():
 a=R.read(R.U/'ANALYSIS.json');R.check(a['source_pins'])
 counts=defaultdict(Counter); flags=defaultdict(lambda:defaultdict(lambda:np.zeros(4,bool)));drifts=defaultdict(list)
 for i,row in enumerate(R.read(R.U/'CONTROLS_DONE.json')['rows']):
  kind=row['kind'];b=R.load(row['original_range']);z=R.load(row['output']);p=R.load(row['original_pair']);cut=row['target_original_indices']
  chosen=(p['phase_old']>=min(cut)-2)&(p['phase_old']<=max(cut)+2)&b['domain'].any(axis=(1,2));counts[kind]['pairs']+=1;drifts[kind].append(row['alignment_drift_mean'])
  for mode in ('point','sequence'):
   for g,start,end in [('angle',0,15),('position',15,23)]:
    name=mode+'_'+g;old=b[mode+'_status'][:,:,start:end];new=z[mode+'_status'][:,:,start:end]
    table=np.bincount((old.astype(int)*6+new).ravel(),minlength=36).reshape(6,6)
    counts[kind][name+'_domain']+=int(table[1:].sum());counts[kind][name+'_common_supported']+=int(table[4:,4:].sum());counts[kind][name+'_supported_lost']+=int(table[4:,1:4].sum());counts[kind][name+'_within_to_beyond']+=int(table[4,5]);counts[kind][name+'_beyond_to_within']+=int(table[5,4])
    local=np.bincount((old[chosen].astype(int)*6+new[chosen]).ravel(),minlength=36).reshape(6,6)
    key=(row['query'],name);flags[kind][key]|=np.array([chosen.any(),local[4:].sum()>0,local[4,5]>0,local[4:,1:4].sum()>0])
  if (i+1)%1000==0:print('REPORT_TRANSITIONS_VERIFIED',i+1,flush=True)
 for kind,c in counts.items():
  wanted=a['controls'][kind];assert dict(c)==wanted['cell_transitions'],kind
  summary=Counter(queries=len({k[0] for k in flags[kind]}),pairs=c['pairs'])
  for (_,name),(has_target,had_support,excess,lost) in flags[kind].items():
   category=('no_original_target_correspondence' if not has_target else 'new_local_excess' if excess else 'new_local_uncertainty_only' if lost else 'no_new_local_signal_with_some_baseline_support' if had_support else 'baseline_target_already_insufficient')
   summary[name+'_'+category]+=1
  assert dict(summary)==wanted['summary'],kind
  assert abs(sum(drifts[kind])/len(drifts[kind])-wanted['mean_alignment_drift_inputframes'])<1e-9
  assert max(drifts[kind])==wanted['max_mean_alignment_drift_inputframes']
 nc=R.read(R.U/'RANGE_VERIFIED.json')['counts']
 for key,v in a['normal'].items():
  states={str(i):nc[key+'_state'+str(i)] for i in range(1,6)};assert states==v['states'];assert sum(states.values())==v['domain'];assert states['4']+states['5']==v['sufficient'];assert round(v['sufficient']*100/v['domain'],2)==v['sufficient_percent']
 R.write(R.U/'ANALYSIS_VERIFIED.json',dict(at=R.now(),status='PASS_INDEPENDENT_STATUS_TRANSITION_HISTOGRAMS',pairs=5593,query_conditions=700,reference_pairs=7184,analysis_sha256=R.sha(R.U/'ANALYSIS.json'),verifier_sha256=R.sha(__file__),method='Same Codex separate saved-status histogram recount, not independent human groundtruth. No inference/alignment/calibration rerun.'))
 print('ANALYSIS_INDEPENDENT_RECOUNT_PASS',flush=True)
if __name__=='__main__':main()
