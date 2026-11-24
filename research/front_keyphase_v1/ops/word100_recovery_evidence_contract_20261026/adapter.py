"""Research receipt consumer. No detector/model forward, labels or thresholds.

Original recognition input is separate. Raw observed coordinates, computable
angles and corroborated angles are different fields. State 4 is NOT ground truth.
Intervals are inclusive sample centres, not inferred continuous visibility.
"""
import numpy as np
GROUPS={'angle':(0,15),'palm':(15,17),'elbow':(17,19),'pose_wrist':(19,21),'hand_wrist':(21,23)}

def intervals(mask,time):
    ids=np.flatnonzero(mask)
    if not len(ids):return []
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return [dict(start_index=int(g[0]),end_index=int(g[-1]),start_seconds=float(time[g[0]]/1e6),
                 end_seconds=float(time[g[-1]]/1e6),samples=len(g)) for g in groups]

def pairing(rr,s,k,p,base_adapter):
    """Return saved second image prediction, never rerun or invent coordinates."""
    i=rr['i'];sl=slice(33+21*s,54+21*s);v=p['usable'][k,i,sl];box=p['roi_box']
    if not v.any():return None,None,0
    native=p['full_usable'][0,i,sl].any()
    if native:
        ds=[d for d in rr['mp_decisions'] if d['reason']=='confirm_existing_hand' and d['side']==s]
        if len(ds)!=1:return None,None,0
        d=ds[0]
        assert d['matched_old'] and all(rr['old_sides'][j]==s for j in d['matched_old'])
        b=np.asarray(rr['mp_full'][d['index']]['xy'],float)
        return b,base_adapter.inside(b,box),1
    stage='yolo' if k==2 and np.any(v&~p['usable'][1,i,sl]) else 'mp'
    if k==0:return None,None,0
    ds=[d for d in rr[stage+'_decisions'] if d['accepted'] and d['side']==s]
    if not ds:return None,None,0
    assert len(ds)==1
    b=np.asarray(rr[stage+'_crop'][ds[0]['crop_index']]['xy'],float)
    return b,v&base_adapter.inside(b,box),2

def identity_flags(records,p):
    """Disagreement sidecar only; no forced reassignment or coordinate deletion."""
    n=len(p['frame_id']);state=np.zeros((n,2),np.uint8)
    for rr in records:
        i=rr['i']
        for d in rr['mp_decisions']:
            s=d['side']
            if d['reason']=='confirm_existing_hand' and s in (0,1):state[i,s]=1
            if d['accepted'] and s in (0,1):state[i,s]=2
        for d in rr['yolo_decisions']:
            if d['accepted'] and d['side'] in (0,1):state[i,d['side']]=2
        for d in rr['mp_decisions']:
            if d['reason'] not in ('existing_identity_conflict','unresolved_existing_overlap'):continue
            implicated={rr['old_sides'][j] for j in d.get('matched_old',[])}|{d['side']}
            for s in implicated:
                if s in (0,1) and p['usable'][2,i,33+21*s:54+21*s].any():state[i,s]=3
    return state

def build(p,records,A,Q,E):
    n=len(p['frame_id']);x=p['xy_px'][:,:,33:].reshape(3,n,2,21,2).copy()
    valid=p['usable'][:,:,33:].reshape(3,n,2,21).copy()
    states=[];degrees=[];pairs=[];p_available=[];p_unresolved=[]
    pair_xy=np.full_like(x,np.nan,dtype=float);pair_valid=np.zeros_like(valid)
    identity=identity_flags(records,p);conflict=identity==3
    for k in range(3):
        pair=np.zeros((n,2),bool);av=np.zeros((n,2,15),bool);au=av.copy()
        for rr in records:
            i=rr['i']
            for s in (0,1):
                b,bv,_=pairing(rr,s,k,p,A)
                if b is None:continue
                pair_xy[k,i,s]=b;pair_valid[k,i,s]=bv
                q=Q.compare_shape(x[k,i,s],valid[k,i,s],b,bv)
                pair[i,s]=q['pair_available'];av[i,s]=q['angle_available'];au[i,s]=q['angle_unresolved']
        e=E.angle_evidence(x[k],valid[k],pair,av,au)
        states.append(e['angle_state']);degrees.append(e['angle_deg']);pairs.append(pair);p_available.append(av);p_unresolved.append(au)
    pose=p['xy_px'][0,:,:33];pv=p['usable'][0,:,:33]
    channels=[A.channel_view(x[k],valid[k],states[k],pose,pv,conflict) for k in range(3)]
    return dict(xy=x,valid=valid,point_origin=p['point_origin'][:,:,33:].reshape(3,n,2,21),
        angle_state=np.stack(states),angle_degree=np.stack(degrees),quality_pair=np.stack(pairs),
        pair_xy=pair_xy,pair_valid=pair_valid,quality_available=np.stack(p_available),quality_unresolved=np.stack(p_unresolved),
        values=np.stack([c[0] for c in channels]),available=np.stack([c[1] for c in channels]),eligible=np.stack([c[2] for c in channels]),
        body_scale=np.asarray([c[3] for c in channels]),identity_state=identity,identity_conflict=conflict,
        pose_xy=pose,pose_valid=pv,position_only=p['position_only'],timestamp_us=p['timestamp_us'],frame_id=p['frame_id'],roi_box=p['roi_box'])

def observation_receipt(z):
    """No expected-motion domain can be inferred just from current missing hands."""
    out=[];time=z['timestamp_us'];n=len(time)
    for k in range(3):
        hands=[]
        for s in (0,1):
            counts=np.bincount(z['angle_state'][k,:,s].ravel(),minlength=5).tolist()
            hands.append(dict(side=s,angle_state_counts=counts,groups={g:dict(computable_cells=int(z['available'][k,:,s,a:b].sum()),
                quality_eligible_cells=int(z['eligible'][k,:,s,a:b].sum()),total_cells=n*(b-a)) for g,(a,b) in GROUPS.items()},
                wrist_not_observed=intervals(~z['valid'][k,:,s,0],time),
                finger_quality_incomplete=intervals(~z['eligible'][k,:,s,:15].all(-1),time),
                identity_disagreement=intervals(z['identity_conflict'][:,s],time)))
        out.append(dict(condition=k,hands=hands))
    return dict(conditions=out,expected_motion_scope=None,whole_action_correct=None,
        explanation='No reference domain in DEV receipt: absent hand is not automatically required, wrong or correct. Angle state4 is geometry corroboration, not anatomical validation.')

def comparison_receipt(p):
    """Consume frozen DTW paths/differences without re-alignment or new scoring."""
    out=[];time=p['timestamp_us'];domain=p['expected']
    for k in range(3):
        hands=[]
        for s in (0,1):
            groups={}
            for g,(a,b) in GROUPS.items():
                expected=domain[:,s,a:b];ok=p['compared'][k,:,s,a:b];diff=p['difference'][k,:,s,a:b]
                assert not (ok&~expected).any() and np.isnan(diff[~ok]).all()
                missing=expected&~ok;den=int(expected.sum());num=int(ok.sum())
                groups[g]=dict(expected_cells=den,compared_cells=num,missing_cells=den-num,
                    coverage=num/den if den else None,mean_difference=float(diff[ok].mean()) if num else None,
                    units='projected_degrees' if g=='angle' else 'shoulder_width',
                    evidence_status='not_required_by_frozen_reference' if not den else 'no_comparable_evidence' if not num else 'partial_evidence' if num<den else 'all_expected_samples_comparable',
                    missing_intervals=intervals(missing.any(-1),time))
            # Keep all15 angles separately; no neighbor suppression or merged low score.
            angles=[]
            for a in range(15):
                ex=domain[:,s,a];ok=p['compared'][k,:,s,a];d=p['difference'][k,:,s,a]
                angles.append(dict(angle=a,expected=int(ex.sum()),compared=int(ok.sum()),
                    mean_degrees=float(d[ok].mean()) if ok.any() else None,missing_intervals=intervals(ex&~ok,time)))
            hands.append(dict(side=s,groups=groups,angles=angles))
        out.append(dict(condition=k,hands=hands))
    return dict(conditions=out,whole_action_correct=None,alignment_verified=False,correction_direction=None,
        warning='Mean difference covers comparable samples only. Low difference cannot certify the entire action; unknown samples remain in the reference denominator.')
