"""Isolated causal research candidate. No training / production dependencies.

Visible arm support is a geometric observation, not anatomical ground truth.
Elbow-only evidence cannot decide identity. Current, usable, in-frame wrists
are required for a hard duplicate conflict. No projected/off-screen wrist is used.
"""
import sys, importlib.util
from pathlib import Path
import numpy as np
import cv2
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent.parent/'word100_assignment_provenance_20261018'
spec=importlib.util.spec_from_file_location('_visible_arm_prior',P/'candidate.py')
prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior)
PALM=np.array([0,5,9,13,17])
GATE=prior.PARAMS['pose_gate']
RATIO=prior.PARAMS['ambiguity_ratio']
MARGIN=prior.PARAMS['duplicate_palm_fraction']
COVERAGE=prior.REGION_CONTAINMENT

def visible(p, usable, ids, w, h):
    a=p[ids]
    return bool(np.asarray(usable)[ids].all() and np.isfinite(a).all() and (a[:,0]>=0).all() and (a[:,0]<w).all() and (a[:,1]>=0).all() and (a[:,1]<h).all())

def segment_distance(x,a,b):
    d=b-a;den=float(d@d)
    q=a if den<=1e-12 else a+np.clip(float((x-a)@d)/den,0,1)*d
    return float(np.linalg.norm(x-q))

def region_distance(x, pts):
    hull=cv2.convexHull(np.asarray(pts,np.float32))
    if cv2.contourArea(hull)<=1e-6:return float('inf')
    return max(0.,-cv2.pointPolygonTest(hull,tuple(map(float,x)),True))

def preferred(cost):
    return [next((s for s in (0,1) if np.isfinite(row[s]) and row[s]<=GATE and row[s]<RATIO*row[1-s]),-1) for row in cost]

def geometry(hands,pose,usable,width,height):
    h=np.asarray(hands,float);p=np.asarray(pose,float);n=len(h)
    elbow=np.full((n,2),np.inf);cost=elbow.copy();distal=elbow.copy()
    arms=[visible(p,usable,[11+s,13+s,15+s],width,height) for s in (0,1)]
    shoulders=visible(p,usable,[11,12],width,height)
    sw=float(np.linalg.norm(p[11]-p[12])) if shoulders else 0.
    coverage=np.zeros((n,n));spans=np.zeros(n)
    valid=[np.isfinite(hand).all() and bool(((hand[PALM,0]>=0)&(hand[PALM,0]<width)&(hand[PALM,1]>=0)&(hand[PALM,1]<height)).all()) for hand in h]
    for i in range(n):
        if not valid[i]:continue
        spans[i]=float(np.linalg.norm(h[i,5]-h[i,17]))
        if sw<=1e-6:continue
        for s in (0,1):
            if visible(p,usable,[11+s,13+s],width,height):elbow[i,s]=np.linalg.norm(h[i,0]-p[13+s])/sw
            if arms[s]:
                # Two parts: observed wrist lies in/near hand palm; Hand wrist
                # attaches to the observed elbow-wrist segment. A shifted wrist
                # alone does not discard the rest of an observed palm.
                cost[i,s]=max(segment_distance(h[i,0],p[13+s],p[15+s]),region_distance(p[15+s],h[i,PALM]))/sw
            if visible(p,usable,[11+s,13+s,17+s,19+s,21+s],width,height):
                distal[i,s]=np.linalg.norm(h[i,[20,8,4]]-p[[17+s,19+s,21+s]],axis=1).mean()/sw
        for j in range(n):
            if i==j or not valid[j]:continue
            # Directed landmark coverage instead of whole bounding/hull area:
            # a warped wrist can enlarge a duplicate's hull along the forearm.
            span=float(np.linalg.norm(h[j,5]-h[j,17]))
            if span>1e-6:
                coverage[i,j]=sum(region_distance(point,h[j])<=MARGIN*span for point in h[i,1:])/20
    return {'arm_cost':cost,'elbow_distance':elbow,'distal_cost':distal,'coverage':coverage,'arms_visible':arms,'preferred':preferred(cost),'valid':valid,'scale':sw}

def decide(sides,states,g):
    out=list(sides);blocked=[]
    if len(out)==2 and all(g['arms_visible']):
        for i,s in enumerate(sides):
            j=1-i
            if s is None or states[i]!='provisional' or sides[j]!=1-s or states[j]!='confirmed':continue
            if g['preferred'][i]==1-s and g['preferred'][j]==1-s and g['coverage'][i,j]>=COVERAGE:
                out[i]=None;blocked.append(i)
    return out,blocked

class HandAssigner:
    def __init__(self):self.previous=prior.HandAssigner()
    def assign(self,hands,labels,scores,pose,pose_usable,timestamp_us,width,height):
        old,d=self.previous.assign(hands,labels,scores,pose,pose_usable,timestamp_us,width,height)
        g=geometry(hands,pose,pose_usable,width,height)
        sides,blocked=decide([old[i] for i in range(len(hands))],d['identity_state'],g)
        def serial(a):return [[float(v) if np.isfinite(v) else None for v in row] for row in a]
        return dict(enumerate(sides)),{'previous':d,'prior_sides':[old[i] for i in range(len(hands))],
            'assigned_sides':sides,'arm_preference':g['preferred'],'arm_cost':serial(g['arm_cost']),
            'elbow_distance':serial(g['elbow_distance']),'distal_cost':serial(g['distal_cost']),
            'finger_region_coverage':g['coverage'].tolist(),'arms_visible':g['arms_visible'],'strict_hand_inside':g['valid'],
            'new_withheld':blocked,'scale':g['scale'],'history_policy':'Previous proposals remain separate. New conflict never seeds a track or creates a hand.',
            'elbow_only_decides_identity':False,'generated_coordinates':False}
