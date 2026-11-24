"""Admitted supplemental observations. Existing recognition/DTW are unchanged."""
from pathlib import Path
import sys,time
U=Path(__file__).resolve().parent;P=U.parent/'word100_clause_finger_diagnostics_20261114'
sys.path.insert(0,str(P))
from direction_adapter import DirectionAdapter
from finger_runtime import FingerObserver,refine
class Extension:
    def __init__(self,rc):
        self.rc=rc;self.direction=DirectionAdapter(P,rc);self.fingers=FingerObserver()
    def apply(self,result,x,ts,word):
        started=time.perf_counter();q=self.direction.apply(result,x,ts,self.rc.FAMILIES[word]);obs={h:self.fingers.infer(x,h)for h in ['L','R']};q=refine(q,obs,ts)
        q['supplement_seconds']=time.perf_counter()-started
        # Do not misattribute an updated direction verdict to the baseline scores.
        for c in q['joint_cards']:
            for e in c['observations']:
                if e['task']=='direction'and 'baseline_state'in e:
                    e['baseline_probabilities']=e.pop('probabilities',None);e['resolution_source']='supplementary_motion';e['resolved_probabilities_by_seed']=e['supplementary_motion']
        return q
