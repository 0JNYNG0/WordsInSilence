"""Research observation adapter. No word/target label influences recovery.

Coordinates are retained independently of feedback eligibility. Detector agreement
is a quality proxy, never anatomical truth. Inferred HaMeR and region proposals
stay in separate arrays and cannot contribute measured finger comparisons.
"""
import numpy as np
PALM=np.array([0,5,9,13,17])
TRIPLES=np.array([(c[i],c[i+1],c[i+2]) for c in
    ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)) for i in range(3)])
GROUPS={'angle':(0,15),'palm':(15,17),'elbow':(17,19),'pose_wrist':(19,21),'hand_wrist':(21,23)}
def inside(x,box):
    return np.isfinite(x).all(-1)&(x[...,0]>=box[0])&(x[...,1]>=box[1])&(x[...,0]<box[2])&(x[...,1]<box[3])
def roi(x,v,box):
    v=v&inside(x,box);return np.where(v[...,None],x,np.nan).astype(np.float32),v
def pair(frame,side,old_present,mask,box):
    if not mask.any():return None,None
    if not old_present:
        qs=[q for q in frame['decisions'] if q['accepted'] and q['side']==side]
        assert len(qs)<=1
        if not qs:return None,None
        b=np.asarray(frame['crop'][qs[0]['crop_index']]['xy']);return b,mask&inside(b,box)
    qs=[q for q in frame['decisions'] if q['reason']=='confirm_existing_hand' and q['side']==side]
    if len(qs)!=1:return None,None
    q=qs[0];assert q['matched_old'] and all(frame['baseline_sides'][j]==side for j in q['matched_old'])
    b=np.asarray(frame['full'][q['index']]['xy']);return b,inside(b,box)
def angles(x,v):
    a=x[...,TRIPLES[:,0],:]-x[...,TRIPLES[:,1],:]
    b=x[...,TRIPLES[:,2],:]-x[...,TRIPLES[:,1],:]
    present=v[...,TRIPLES].all(-1)&np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
    good=present&(np.linalg.norm(a,axis=-1)>1e-6)&(np.linalg.norm(b,axis=-1)>1e-6)
    degree=np.degrees(np.arctan2(abs(a[...,0]*b[...,1]-a[...,1]*b[...,0]),(a*b).sum(-1)))
    return np.where(good,degree,np.nan),good,present
def channel_view(x,v,states,pose,pose_valid,conflict):
    """23 channels/hand; fixed body normalization from unchanged pose only."""
    t=len(x);values=np.full((t,2,23),np.nan);available=np.zeros(values.shape,bool)
    sh=pose_valid[:,11]&pose_valid[:,12]
    widths=np.linalg.norm(pose[:,11]-pose[:,12],axis=-1)
    widths=widths[sh&np.isfinite(widths)&(widths>1e-6)]
    if not len(widths):return values,available,available.copy(),np.nan
    scale=float(np.median(widths));origin=(pose[:,11]+pose[:,12])/2
    deg,good,_=angles(x.astype(float),v)
    values[:,:,:15]=deg;available[:,:,:15]=good
    pv=v[:,:,PALM].all(-1)
    centre=np.mean(x[:,:,PALM],axis=2)
    values[:,:,15:17]=(centre-origin[:,None])/scale
    available[:,:,15:17]=(pv&sh[:,None])[:,:,None]
    for s,(elbow,wrist) in enumerate(((13,15),(14,16))):
        for j,p in ((17,elbow),(19,wrist)):
            values[:,s,j:j+2]=(pose[:,p]-origin)/scale
            available[:,s,j:j+2]=(pose_valid[:,p]&sh)[:,None]
    values[:,:,21:23]=(x[:,:,0]-origin[:,None])/scale
    available[:,:,21:23]=(v[:,:,0]&sh[:,None])[:,:,None]
    eligible=available.copy();eligible[:,:,:15]&=states==4
    # Identity holds apply to hand measurements, not independent arm observations.
    eligible[:,:,:17]&=~conflict[:,:,None];eligible[:,:,21:]&=~conflict[:,:,None]
    return np.where(available,values,np.nan),available,eligible,scale
def on_path(values,valid,path,n):
    """One reference clock. Require every contributing source sample, no gap fill."""
    out=np.full((n,)+values.shape[1:],np.nan);ok=np.zeros(out.shape,bool)
    spans=np.full((n,2),-1,int)
    for j in range(n):
        ids=path[path[:,1]==j,0]
        if not len(ids):continue
        lo,hi=int(ids.min()),int(ids.max());spans[j]=[lo,hi]
        ok[j]=valid[lo:hi+1].all(0)
        out[j]=np.where(ok[j],np.mean(values[lo:hi+1],axis=0),np.nan)
    return out,ok,spans
def compared(q,ref,domain,path):
    x,ok,span=on_path(q['values'],q['eligible'],path,len(ref['values']))
    mask=domain&ok&ref['eligible']
    return dict(compared=mask,difference=np.where(mask,abs(x-ref['values']),np.nan),
                expected=domain.copy(),unknown=domain&~mask,query_span=span)
def selftest():
    x=np.arange(4*2*23,dtype=float).reshape(4,2,23);v=np.ones(x.shape,bool)
    path=np.array([[0,0],[1,1],[2,1],[3,2]])
    domain=np.ones((3,2,23),bool);r={'values':np.zeros(domain.shape),'eligible':domain}
    a=compared({'values':x,'eligible':v},r,domain,path)
    v[2,0,1]=False;b=compared({'values':x,'eligible':v},r,domain,path)
    assert np.array_equal(a['expected'],b['expected']) and b['unknown'][1,0,1]
    assert np.isnan(b['difference'][1,0,1]) and b['compared'][1,0,2]
    wrist=np.full((2,2,21,2),np.nan);wv=np.zeros((2,2,21),bool)
    wrist[0,0,0]=[1,2];wv[0,0,0]=True
    assert not angles(wrist,wv)[1].any()
    return dict(status='PASS',checks=['fixed_reference_scope_under_absence','no_bridging_contributing_gap','unknown_not_zero','independent_neighbor_channels','wrist_does_not_create_finger_angle'])
