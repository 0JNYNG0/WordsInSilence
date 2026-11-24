"""Research-only ambiguity guard. Preserve raw points, never invent a wrist/side.

A collapsed 2-D palm is NOT an error by itself. Suppress only a provisional
side when overlapping hypotheses, discrepant wrists and a recent single-hand
observation jointly make a two-hand interpretation unsupported. This is a
duplicate suspicion, not proof that the other hypothesis is anatomically right.
"""
import cv2
import numpy as np

PARAMS = dict(min_points=16, palm=[0,5,9,13,17], mcp=[5,9,13,17],
              collapse_ratio=.25, reference_ratio=.50, overlap_smaller=.75,
              wrist_separation=.50, max_gap_us=100000, previous_distance=.40)

def geometry(points, mask):
    q=np.asarray(points,dtype=float); m=np.asarray(mask,bool)&np.isfinite(q).all(axis=1)
    if m.sum()<PARAMS['min_points'] or not m[PARAMS['palm']].all():
        return {'usable':False,'reason':'insufficient_observed_palm'}
    p=q[PARAMS['mcp']]; span=float(np.linalg.norm(p[:,None]-p[None,:],axis=2).max())
    if span<=1e-6:return {'usable':False,'reason':'zero_palm_span'}
    center=p.mean(axis=0); ratio=float(np.linalg.norm(center-q[0])/span)
    hull=cv2.convexHull(q[m].astype(np.float32)); area=float(cv2.contourArea(hull))
    return {'usable':area>1e-6,'ratio':ratio,'span':span,'area':area,
            'hull':hull[:,0,:].tolist(),'reason':'collapsed_projection' if ratio<PARAMS['collapse_ratio'] else 'not_collapsed'}

def decision(g, overlap, separation, previous, sides, identity):
    out=list(map(int,sides)); reasons=['unchanged']*len(out)
    if len(out)!=2:return out,reasons
    for i in range(2):
        j=1-i
        if identity[i]>=0:reasons[i]='keep_confirmed';continue
        if sides[i]<0:reasons[i]='keep_withheld';continue
        if not (g[i]['usable'] and g[j]['usable']):continue
        if (g[i]['ratio']<PARAMS['collapse_ratio'] and g[j]['ratio']>=PARAMS['reference_ratio']
            and overlap>=PARAMS['overlap_smaller'] and separation>=PARAMS['wrist_separation']
            and previous[i] and previous[j]):
            out[i]=-1;reasons[i]='withhold_duplicate_wrist_conflict'
    return out,reasons

def propose(points, masks, sides, identity, scale, previous_points=None,
            previous_masks=None, delta_us=None):
    q=np.asarray(points,float);m=np.asarray(masks,bool);g=[geometry(a,b) for a,b in zip(q,m)]
    overlap=0.;separation=0.;prev=[False]*len(q);distances=[None]*len(q)
    if len(q)==2 and all(a['usable'] for a in g):
        area,_=cv2.intersectConvexConvex(np.asarray(g[0]['hull'],np.float32),np.asarray(g[1]['hull'],np.float32))
        overlap=float(area/min(a['area'] for a in g))
        separation=float(np.linalg.norm(q[0,0]-q[1,0])/max(a['span'] for a in g))
    if (previous_points is not None and len(previous_points)==1 and
        delta_us is not None and 0<delta_us<=PARAMS['max_gap_us'] and np.isfinite(scale) and scale>0):
        pg=geometry(previous_points[0],previous_masks[0])
        if pg['usable'] and pg['ratio']>=PARAMS['reference_ratio']:
            for i,a in enumerate(g):
                if a['usable']:
                    d=float(np.median(np.linalg.norm(q[i,PARAMS['mcp']]-previous_points[0,PARAMS['mcp']],axis=1))/scale)
                    distances[i]=d;prev[i]=d<=PARAMS['previous_distance']
    out,reasons=decision(g,overlap,separation,prev,sides,identity)
    return {'sides':out,'reason':reasons,'geometry':g,'overlap_smaller':overlap,
            'wrist_separation':separation,'previous_support':prev,'previous_distance':distances,
            'identity_side':list(map(int,identity)), 'coordinates_modified':False}
