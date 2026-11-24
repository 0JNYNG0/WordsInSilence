"""Pure measurement contract: palm root/MCP centre along opposite forearm.

Excludes fingertips and joint-flexion labels. An available coordinate is not
verified visibility/contact. This file does not load a model or make a verdict.
Axes other than2 retain the existing measurement exactly.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from common import old,np

PALM_OFFSETS=(0,5,9,13,17)
SCHEMA='axis2_palm0_mcp_mean_shoulderwidth_current_anchor_unit_v1'

def geometry(x,ts,a,b,axis,moving='R',anchor='L',reverse=False):
    if axis!=2:
        return old.geometry(x,ts,a,b,axis,moving,anchor,reverse)
    assert moving in ('L','R') and anchor in ('L','R') and moving!=anchor
    assert 0<=a<=b<len(x) and len(ts)==len(x)
    p,valid=old.prior.motion.body_points(x)
    m=old.prior.motion.SIDES[moving];r=old.prior.motion.SIDES[anchor]
    p[:,:,0]*=(np.sign(p[:,2,0]-p[:,1,0])*(1 if moving=='R'else -1))[:,None]
    palm_ids=[m['hand']+j for j in PALM_OFFSETS]
    palm=p[:,palm_ids].mean(1)
    other=p[:,r['wrist']];arm=other-p[:,r['elbow']]
    length=np.linalg.norm(arm,axis=1);rel=palm-p[:,r['elbow']]
    values=np.column_stack([(rel*arm).sum(1)/np.maximum(length,1e-5),rel[:,0],rel[:,1],np.linalg.norm(palm-other,axis=1)])
    good=valid[:,palm_ids+[r['elbow'],r['wrist']]].all(1)&np.isfinite(values).all(1)&(length>.05)
    vv=values[a:b+1];gg=good[a:b+1];tt=np.asarray(ts[a:b+1],float);ll=length[a:b+1]
    if reverse:vv=vv[::-1];gg=gg[::-1];ll=ll[::-1];tt=tt[-1]-tt[::-1]
    cov=float(gg.mean());count=int(gg.sum())
    ok=cov>=.8 and count>=4 and bool(gg[0]and gg[-1])
    info=dict(schema=SCHEMA,coverage=cov,observed=count,frames=b-a+1,
        moving_point='mean of hand wrist0 and MCP5/9/13/17',palm_ids=palm_ids,
        anchor_points=[r['elbow'],r['wrist']],units='shoulder_widths_along_current_anchor_direction',
        endpoint_available=[bool(gg[0]),bool(gg[-1])],retained_coordinates_not_verified_visibility=True,
        contact_verified=False,body_wrist_only=False,fingertips_used=False,
        semantic_direction_verified=False)
    if not ok:
        info['state']='unavailable_measurement'
        return np.zeros(58,np.float32),False,info
    q=vv[gg];tg=tt[gg]
    assert np.all(np.diff(tg)>0)
    sampled=np.stack([np.interp(np.linspace(tg[0],tg[-1],9),tg,q[:,j])for j in range(4)],1)
    delta=sampled[-1]-sampled[0]
    g=np.r_[(sampled-sampled[0]).ravel(),sampled[0],delta,np.abs(np.diff(sampled,axis=0)).mean(0),np.ptp(sampled,axis=0),np.eye(5)[axis],cov]
    assert g.shape==(58,) and np.isfinite(g).all()
    info.update(state='observed_geometry_only',signed_change=float(delta[0]),trajectory=sampled[:,0].tolist(),anchor_length_range=[float(ll[gg].min()),float(ll[gg].max())])
    return g.astype(np.float32),True,info
