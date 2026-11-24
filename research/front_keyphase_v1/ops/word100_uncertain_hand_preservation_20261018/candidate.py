"""Isolated observation/identity separation plus evidence-backed duplicate guard.

Coordinates are retained without an anatomical assignment. A low-quality Pose
wrist never erases a current Hand observation. Possible temporal links are not
anatomical proof. No coordinates, side labels, or prior evidence are fabricated.
"""
import importlib.util,sys
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_single_arm_recheck_20261018'
s=importlib.util.spec_from_file_location('_uncertain_hand_prior',P/'candidate.py');prior=importlib.util.module_from_spec(s);sys.modules[s.name]=prior;s.loader.exec_module(prior)
PALM=[0,5,9,13,17];GAP_US=100000;TRACK_GATE=.4;COVERAGE=.8

def measured_mask(hands,scores,width,height):
    h=np.asarray(hands,float);sc=np.asarray(scores,float)
    return np.isfinite(h).all(axis=-1)&(h[:,:,0]>=0)&(h[:,:,0]<width)&(h[:,:,1]>=0)&(h[:,:,1]<height)&(np.isfinite(sc)&(sc>=.5))[:,None]

def duplicate_guard(sides,states,current_pose,origin_us,timestamp_us,confirmed_distance,coverage,valid):
    out=list(sides);blocked=[];anchor=[-1]*len(sides)
    if len(sides)!=2:return out,blocked,anchor
    for i,side in enumerate(sides):
        j=1-i
        if side is None or states[i]!=1 or sides[j]!=1-side or states[j]!=2:continue
        if not (valid[i] and valid[j]) or coverage[i,j]<COVERAGE:continue
        # A current Pose-confirmed anchor or an independently located anchor no
        # older than100ms. Repeating a provisional label cannot refresh this.
        age=int(timestamp_us)-int(origin_us[j])
        if not current_pose[j] and not (origin_us[j]>=0 and 0<=age<=GAP_US):continue
        own=float(confirmed_distance[i,side]);other=float(confirmed_distance[i,1-side])
        # Protect real crossings with any existing own confirmed history.
        # Never select between two candidates with no anatomical anchor.
        if np.isfinite(own) or not np.isfinite(other) or other>TRACK_GATE:continue
        out[i]=None;blocked.append(i);anchor[i]=j
    return out,blocked,anchor

class ObservationLedger:
    def __init__(self):self.previous=None
    def step(self,hands,scores,pose,pose_usable,timestamp_us,width,height,suggested,effective,states,actions,blocked,scale):
        h=np.asarray(hands,float);n=len(h);mask=measured_mask(h,scores,width,height);us=int(timestamp_us)
        p=np.asarray(pose,float);pu=np.asarray(pose_usable,bool)
        wrist=np.array([bool(pu[k] and np.isfinite(p[k]).all() and 0<=p[k,0]<width and 0<=p[k,1]<height) for k in (15,16)])
        side=np.full(n,-1,np.int8);identity=np.zeros(n,np.int8)
        for i in range(n):
            if effective[i] is not None and (states[i]==2 or actions[i] in (2,3,4)):
                side[i]=effective[i];identity[i]=1
            if i in blocked:identity[i]=2  # Conflict, raw observation still present.
        points=np.where(mask[:,:,None],h,np.nan)
        local=np.full_like(h,np.nan);local_mask=np.zeros_like(mask)
        for i in range(n):
            if mask[i,0]:local_mask[i]=mask[i];local[i,mask[i]]=h[i,mask[i]]-h[i,0]
        links=np.zeros((n,2),bool);link_cost=np.full((n,2),np.nan);common=np.zeros((n,2),np.int8)
        old=self.previous
        if old is not None and 0<us-old['us']<=GAP_US and np.isfinite(scale) and scale>1e-6:
            for i in range(n):
                for j in range(len(old['hands'])):
                    ids=[k for k in PALM if mask[i,k] and old['mask'][j,k]];common[i,j]=len(ids)
                    # At least three corresponding visible palm nodes. This is
                    # only a possible physical association, never a side label.
                    if len(ids)>=3:
                        v=float(np.linalg.norm(h[i,ids]-old['hands'][j,ids],axis=1).mean()/scale);link_cost[i,j]=v;links[i,j]=v<=TRACK_GATE
        self.previous={'hands':h.copy(),'mask':mask.copy(),'us':us}
        return {'observed_xy':points,'point_mask':mask,'local_xy':local,'local_mask':local_mask,
            'identity_side':side,'identity_state':identity,'pose_wrist_usable':wrist,
            'possible_links':links,'link_cost':link_cost,'common_palm_points':common,
            'suggested_side':np.array([-1 if s is None else s for s in suggested],np.int8),
            'effective_side':np.array([-1 if s is None else s for s in effective],np.int8)}

class HandAssigner:
    def __init__(self):self.old=prior.HandAssigner();self.times=[];self.ledger=ObservationLedger()
    def assign(self,hands,labels,scores,pose,pose_usable,timestamp_us,width,height):
        if self.times and timestamp_us<=self.times[-1]:self.times=[];self.ledger=ObservationLedger()
        self.times.append(int(timestamp_us));sides,d=self.old.assign(hands,labels,scores,pose,pose_usable,timestamp_us,width,height)
        n=len(hands);pr=d['previous']['previous'];state_map={'unassigned':0,'provisional':1,'confirmed':2,'withheld_conflict':3};st=[state_map[x] for x in pr['identity_state']]
        g=prior.prior.geometry(hands,pose,pose_usable,width,height)
        orig=[-1 if x is None else self.times[x['frame']] for x in pr['origin']]
        cd=np.array([[np.nan if x is None else x for x in rr] for rr in pr['confirmed_track_distance']]).reshape(n,2)
        previous=[sides[i] for i in range(n)]
        out,blocked,anchor=duplicate_guard(previous,st,pr['current_pose_confirmation'],orig,timestamp_us,cd,g['coverage'],g['valid'])
        code={'previous_candidate':0,'pending_arm_conflict':1,'revalidated_current_arm':2,'renewed_current_arm':3,'bounded_revalidated_history':4,'destination_conflict':5}
        obs=self.ledger.step(hands,scores,pose,pose_usable,timestamp_us,width,height,previous,out,st,[code[x] for x in d['recheck_reason']],blocked,g['scale'])
        return dict(enumerate(out)),{'previous':d,'new_duplicate_conflicts':blocked,'anchor_raw':anchor,'observation':obs}
