"""Research-only causal recheck of a provisional anatomical identity.

Raw data and the previous candidate are immutable. Current visible arm support
can challenge an unconfirmed identity; repetition of a raw label cannot confirm
it. Corrected memory has an evidence expiry independent of its position updates.
"""
import sys,importlib.util
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent.parent/'word100_visible_arm_duplicate_20261018'
spec=importlib.util.spec_from_file_location('_track_recheck_prior',P/'candidate.py')
prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior)
PARAM=prior.prior.PARAMS
PALM=np.array([0,5,9,13,17])
GAP=PARAM['bootstrap_max_gap_s'];GATE=PARAM['track_gate'];RATIO=PARAM['ambiguity_ratio']
COUNT=PARAM['conflict_confirmation_frames']

def distance(hand,memory,scale):
    return float(np.linalg.norm(hand[PALM]-memory['palm'],axis=1).mean()/scale) if scale>1e-6 else float('inf')

def unique_hand(hands,i,memory,scale):
    value=distance(hands[i],memory,scale)
    return value<=GATE and all(value<RATIO*distance(hands[j],memory,scale) for j in range(len(hands)) if j!=i)

def observed_target(i,sides,states,g,scores):
    if sides[i] is None or states[i]!='provisional' or not g['valid'][i] or scores[i]<.5 or not np.isfinite(scores[i]):return -1
    s=g['preferred'][i]
    if s<0 or not all(g['arms_visible']):return -1
    value=g['arm_cost'][i,s]
    if not all(value<RATIO*g['arm_cost'][j,s] for j in range(len(sides)) if j!=i):return -1
    # A confirmed other hand in the proposed slot cannot be displaced by an
    # uncertain observation. This protects real crossings and duplicates.
    if any(j!=i and sides[j]==s and states[j]=='confirmed' for j in range(len(sides))):return -1
    return s

class Revalidator:
    def __init__(self):self.reset()
    def reset(self):self.pending={};self.corrected={};self.last_time=None;self.frame=-1
    def step(self,hands,scores,timestamp_us,sides,states,g):
        h=np.asarray(hands,float);t=int(timestamp_us)/1e6;n=len(h);sw=g['scale']
        if self.last_time is not None and t<=self.last_time:self.reset()
        self.last_time=t;self.frame+=1
        self.corrected={s:q for s,q in self.corrected.items() if 0<=t-q['arm_time']<=GAP+1e-12}
        oldpending=self.pending;self.pending={}
        targets=[observed_target(i,sides,states,g,scores) for i in range(n)]
        out=list(sides);reason=['previous_candidate']*n;proposals={};matched={};roots=[None]*n;counts=[0]*n
        for i,old in enumerate(sides):
            if old is None or states[i]!='provisional':continue
            s=targets[i]
            if s>=0 and s!=old:
                if s in self.corrected and unique_hand(h,i,self.corrected[s],sw):
                    proposals[i]=s;reason[i]='renewed_current_arm';roots[i]=self.corrected[s]['root'];counts[i]=COUNT
                else:
                    q=oldpending.get(s)
                    linked=q is not None and q['old_side']==old and 0<t-q['time']<=GAP+1e-12 and unique_hand(h,i,q,sw)
                    cnt=q['count']+1 if linked else 1;root=q['root'] if linked else self.frame
                    counts[i]=cnt;roots[i]=root
                    if cnt>=COUNT:proposals[i]=s;reason[i]='revalidated_current_arm'
                    else:
                        out[i]=None;reason[i]='pending_arm_conflict'
                        self.pending[s]={'palm':h[i,PALM].copy(),'time':t,'old_side':old,'count':cnt,'root':root}
            elif g['valid'][i] and np.isfinite(scores[i]) and scores[i]>=.5:
                # A bounded carry is permitted only when current arm evidence
                # is absent/ambiguous. Contradictory evidence cancels the carry.
                hits=[s for s,q in self.corrected.items() if s!=old and unique_hand(h,i,q,sw)]
                if len(hits)==1 and g['preferred'][i] in (-1,hits[0]):
                    s=hits[0]
                    if not any(j!=i and sides[j]==s and states[j]=='confirmed' for j in range(n)):
                        proposals[i]=s;reason[i]='bounded_revalidated_history';roots[i]=self.corrected[s]['root'];counts[i]=COUNT
        # Resolve proposed destinations together. Never put two raw candidates
        # in one anatomical slot or remove an unrelated current assignment.
        proposed=out.copy()
        for i,s in proposals.items():proposed[i]=s
        for i,s in proposals.items():
            if sum(x==s for x in proposed)>1:
                out[i]=None;reason[i]='destination_conflict';continue
            out[i]=s;matched[i]=s
        for i,s in matched.items():
            old=self.corrected.get(s)
            arm_time=t if targets[i]==s else old['arm_time']
            self.corrected[s]={'palm':h[i,PALM].copy(),'time':t,'arm_time':arm_time,'root':roots[i]}
        # A confirmed observed occupant closes any stale alternate hypothesis.
        for i,s in enumerate(sides):
            if s is not None and states[i]=='confirmed':self.corrected.pop(s,None);self.pending.pop(s,None)
        assert len([s for s in out if s is not None])==len(set(s for s in out if s is not None))
        return out,{'recheck_reason':reason,'current_arm_target':targets,'confirmation_count':counts,'evidence_origin_frame':roots,
            'pending_sides':sorted(self.pending),'corrected_sides':sorted(self.corrected),'corrected_arm_time':{str(s):q['arm_time'] for s,q in self.corrected.items()},
            'coordinates_generated':False,'future_frames_used':False}

class HandAssigner:
    def __init__(self):self.previous=prior.HandAssigner();self.recheck=Revalidator()
    def assign(self,hands,labels,scores,pose,pose_usable,timestamp_us,width,height):
        old,d=self.previous.assign(hands,labels,scores,pose,pose_usable,timestamp_us,width,height)
        g=prior.geometry(hands,pose,pose_usable,width,height)
        out,trace=self.recheck.step(hands,scores,timestamp_us,[old[i] for i in range(len(hands))],d['previous']['identity_state'],g)
        return dict(enumerate(out)),{'previous':d,**trace,'assigned_sides':out}
