"""Isolated evidence supplement, not a replacement for current word recognition.

Preserve existing supported observations when new evidence is merely unknown.
Report a conflict when two confident observations oppose each other. Recover
an undecided direction only from all three fixed compatible motion readings.
"""
import copy,time
import numpy as np,torch
from motion_core import SymmetricMotion
from stable_geometry import geometry
STRONG={'supported','opposite_direction'}
def merge_state(old,new_states):
    strong=set(new_states)&STRONG
    if len(strong)>1:return 'evidence_conflict'
    if len(new_states)==3 and len(strong)==1 and all(s in STRONG for s in new_states):
        proposed=next(iter(strong))
        if old in STRONG and old!=proposed:return 'evidence_conflict'
        return proposed
    return old
class DirectionAdapter:
    def __init__(self,root,rc):
        self.root=root;self.rc=rc;self.models={};base=rc.frozen_observer()
        for seed in [42,43,44]:
            m=SymmetricMotion(base);m.load_state_dict(torch.load(root/'motion_models_v2'/f'{seed}.pt',weights_only=True));m.eval();self.models[seed]=m
    def apply(self,qualified,x,ts,specs):
        start=time.perf_counter();out=copy.deepcopy(qualified);audit=[]
        for c in out['joint_cards']:
            if c['state']in ['action_link_uncertain','evidence_conflict']:continue
            spec=next(s for s in specs if s['id']==c['card'])
            if spec['axis']is None or spec['sign']is None:continue
            for e in c['observations']:
                if e['task']!='direction' or 'span'not in e or e['state']in ['observation_unknown','association_unknown']:continue
                evidence=[]
                for seed,m in self.models.items():
                    witness=next((r for r in c.get('cross_run_evidence',[])if r['seed']==seed),None)
                    obs=next((z for z in witness['observations']if z['task']=='direction'and'span'in z),None)if witness else e
                    if obs is None or obs['state']in ['observation_unknown','association_unknown']:
                        evidence.append(dict(seed=seed,state='observation_unknown'));continue
                    a,b=obs['span'];g,ok,info=geometry(self.rc.prior,x,ts,a,b,spec['axis'],spec['hands'][0],'L'if spec['hands'][0]=='R'else'R')
                    with torch.no_grad():p=m(torch.tensor(g[None])).softmax(-1)[0].numpy()
                    evidence.append(dict(seed=seed,span=[a,b],state=self.rc.direction_verdict(p,spec['sign']+1,ok),probabilities=p.tolist(),measurement=info))
                before=e['state'];after=merge_state(before,[v['state']for v in evidence]);audit.append(dict(card=c['card'],before=before,after=after,candidate_evidence=evidence))
                e['supplementary_motion']=evidence
                if after==before:continue
                if after=='evidence_conflict':
                    c['state_before_supplement']=c['state'];c['state']='evidence_conflict';c['conflicts']=c.get('conflicts',[])+[['supplementary_direction']];continue
                e['baseline_state']=before;e['state']=after
                c['differences']=[d for d in c['differences']if d['task']!='direction']
                if after=='opposite_direction':
                    c['differences'].append(copy.deepcopy(e));c['state']='attributable_feature_difference'
            c['whole_action_complete']=None;c['show_as_correct_sign']=False
        out['supplementary_motion_audit']=audit
        out['supplementary_motion_seconds']=time.perf_counter()-start
        out['new_finger_model_used']=False
        return out
