"""Research-only causal association with explicit identity provenance.

Raw detections are never removed or synthesized. `assign` returns effective
anatomical associations and a trace; baseline proposals remain in that trace.
Confirmed means independent pose/confirmed-history evidence, not ground truth.
No word IDs, manual annotations, future frames or outside-frame pixels are inputs.
"""
import sys, importlib.util
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent.parent/'word100_assignment_conflict_20261018'
spec=importlib.util.spec_from_file_location('_frozen_provenance_baseline',P/'assignment.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
PARAMS=dict(base.PARAMS)
REGION_CONTAINMENT=0.80  # >=80% of smaller full21 convex hull, geometric conflict screen.
PALM=base.PALM

@dataclass
class EvidenceTrack:
    palm: np.ndarray
    time: float
    frame: int
    velocity: np.ndarray
    origin_frame: int
    origin_basis: str
    last_pose_frame: int

def distance_matrix(hands, tracks, t, scale):
    out=np.full((len(hands),2),np.inf)
    if scale is not None:
        for i,h in enumerate(hands):
            if not np.isfinite(h).all():continue
            for s,tr in tracks.items():
                out[i,s]=np.linalg.norm(h[PALM]-(tr.palm+tr.velocity*(t-tr.time)),axis=1).mean()/scale
    return out

def unique(cost,i,s,gate):
    if not np.isfinite(cost[i,s]) or cost[i,s]>gate:return False
    rivals=[cost[i,1-s]]+[cost[j,s] for j in range(len(cost)) if j!=i]
    return all(not np.isfinite(v) or cost[i,s]<PARAMS['ambiguity_ratio']*v for v in rivals)

def region_overlap(hands):
    if len(hands)!=2 or not np.isfinite(hands).all():return 0.0
    hull=[cv2.convexHull(np.asarray(h,dtype=np.float32)) for h in hands]
    areas=[cv2.contourArea(h) for h in hull]
    if min(areas)<=1e-6:return 0.0
    area,_=cv2.intersectConvexConvex(hull[0],hull[1])
    return float(max(0.0,min(1.0,area/min(areas))))

class HandAssigner:
    def __init__(self):self.reset()
    def reset(self):
        self.proposer=base.HandAssigner()
        self.confirmed={}
        self.provisional={}
        self.last_time=None
        self.frame=-1

    def assign(self,hands,labels,scores,pose,pose_usable,timestamp_us,width,height):
        h=np.asarray(hands,dtype=np.float64);t=int(timestamp_us)/1e6
        if self.last_time is not None and t<=self.last_time:self.reset()
        self.last_time=t;self.frame+=1
        for tracks in (self.confirmed,self.provisional):
            for s in list(tracks):
                if t-tracks[s].time>PARAMS['max_gap_s']:del tracks[s]
        proposal,d=self.proposer.assign(h,labels,scores,pose,pose_usable,timestamp_us,width,height)
        n=len(h);scale=d['scale'];result=dict(proposal)
        pc=np.array([[np.inf if v is None else v for v in row] for row in d['pose_distance']],float).reshape(n,2)
        ct=distance_matrix(h,self.confirmed,t,scale)
        pt=distance_matrix(h,self.provisional,t,scale)
        direct=[];history=[];states=[];roots=[];parents=[]
        for i in range(n):
            s=result[i]
            po=s is not None and unique(pc,i,s,PARAMS['pose_gate'])
            hi=s is not None and unique(ct,i,s,PARAMS['track_gate'])
            direct.append(po);history.append(hi)
            if s is None:
                states.append('unassigned');roots.append(None);parents.append(None)
            elif po:
                states.append('confirmed');roots.append({'frame':self.frame,'basis':'current_unique_pose'});parents.append(None)
            elif hi:
                tr=self.confirmed[s]
                states.append('confirmed');roots.append({'frame':tr.origin_frame,'basis':tr.origin_basis});parents.append(tr.frame)
            else:
                states.append('provisional')
                tr=self.provisional.get(s)
                linked=tr is not None and unique(pt,i,s,PARAMS['track_gate'])
                roots.append({'frame':tr.origin_frame if linked else self.frame,'basis':'unconfirmed_proposal'})
                parents.append(tr.frame if linked else None)
        overlap=region_overlap(h)
        guards=[]
        # A label-only jump must not overwrite a recently independently located
        # same-side track. Reachability uses the existing speed/time bound.
        for i in range(n):
            s=result[i]
            if s is None or states[i]!='provisional':continue
            tr=self.confirmed.get(s)
            why=d['reasons'][i]
            if tr is not None and (why=='label_without_close_pose' or why.startswith('temporal')):
                dt=t-tr.time
                gate=max(PARAMS['track_gate'],PARAMS['max_speed_shoulders_s']*dt)
                if 0<dt<=PARAMS['bootstrap_max_gap_s']+1e-12 and ct[i,s]>gate:
                    result[i]=None;states[i]='withheld_conflict'
                    guards.append({'raw_index':i,'side':s,'kind':'unconfirmed_jump_from_recent_confirmed_track','distance':float(ct[i,s]),'gate':gate,'parent_frame':tr.frame})
        # Only an unresolved side with a confirmed opposite anchor can be
        # withheld as a duplicate risk. A reachable own confirmed history
        # protects genuine crossing even if the opposite hand is closer.
        if n==2 and overlap>=REGION_CONTAINMENT:
            for i in range(2):
                s=result[i];j=1-i
                if s is None or states[i]!='provisional':continue
                if result[j]!=1-s or states[j]!='confirmed':continue
                own=self.confirmed.get(s)
                own_gate=max(PARAMS['track_gate'],PARAMS['max_speed_shoulders_s']*(t-own.time)) if own is not None else PARAMS['track_gate']
                own_reachable=own is not None and ct[i,s]<=own_gate
                opposite_reachable=ct[i,1-s]<=PARAMS['track_gate']
                opposite_preferred=not np.isfinite(ct[i,s]) or ct[i,1-s]<PARAMS['ambiguity_ratio']*ct[i,s]
                if own_reachable or not (opposite_reachable and opposite_preferred):continue
                result[i]=None;states[i]='withheld_conflict'
                guards.append({'raw_index':i,'side':s,'kind':'overlapping_region_with_confirmed_opposite_anchor','anchor_raw_index':j,'overlap_min_area':overlap,'opposite_track_distance':float(ct[i,1-s]),'own_confirmed_track_reachable':False})
        # Commit only current observed palms. Provisional observations never
        # update confirmed tracks or create independent confirmation by repetition.
        for i,s in result.items():
            if s is None:continue
            dst=self.confirmed if states[i]=='confirmed' else self.provisional
            old=dst.get(s);v=np.zeros(2)
            if old is not None and t>old.time and scale is not None:
                v=(h[i,PALM].mean(0)-old.palm.mean(0))/(t-old.time)
                speed=np.linalg.norm(v);cap=PARAMS['max_speed_shoulders_s']*scale
                if speed>cap:v*=cap/speed
            root=roots[i]
            dst[s]=EvidenceTrack(h[i,PALM].copy(),t,self.frame,v,root['frame'],root['basis'],self.frame if direct[i] else old.last_pose_frame if old is not None else -1)
            if states[i]=='confirmed':self.provisional.pop(s,None)
        def serial(a):return [[float(v) if np.isfinite(v) else None for v in row] for row in a]
        return result,{'baseline':d,'proposal_sides':[proposal[i] for i in range(n)],'assigned_sides':[result[i] for i in range(n)],'identity_state':states,'origin':roots,'parent_frame':parents,'current_pose_confirmation':direct,'confirmed_history_match':history,'confirmed_track_distance':serial(ct),'provisional_track_distance':serial(pt),'region_overlap_min_area':overlap,'guards':guards,'coordinates_generated':False,'raw_observations_preserved':True}
