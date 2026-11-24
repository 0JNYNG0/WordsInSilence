"""Same FIT candidate windows: body wrist vs palm0+MCP5/9/13/17.
No point chosen from model scores, no training and no changed fixture/labels.
"""
from audit import *

def measure(x,ts,a,b,point):
    p,v=old.prior.motion.body_points(x)
    p[:,:,0]*=np.sign(p[:,2,0]-p[:,1,0])[:,None]
    ids=[6] if point=='body_wrist' else [30,35,39,43,47]
    q=p[:,ids].mean(1);arm=p[:,5]-p[:,3];length=np.linalg.norm(arm,axis=1);rel=q-p[:,3];unit=arm/np.maximum(length[:,None],1e-5)
    val=(rel*unit).sum(1);ratio=val/np.maximum(length,1e-5);rad=np.linalg.norm(q-p[:,5],axis=1)
    good=v[:,ids+[3,5]].all(1)&(length>.05)&np.isfinite(val)&np.isfinite(ratio)
    gg=good[a:b+1];cov=float(gg.mean());ok=cov>=.8 and int(gg.sum())>=4 and bool(gg[0]and gg[-1])
    out=dict(point=point,points=ids,eligible=bool(ok),coverage=cov,observed=int(gg.sum()),frames=b-a+1,first_observed=bool(gg[0]),last_observed=bool(gg[-1]),mask_is_retained_coordinate_not_RGB_visibility=True)
    if not ok:return out
    ii=np.flatnonzero(gg)+a;r=rel[ii];u=unit[ii];dd=np.diff(val[ii]);z=val[ii]
    trans=float(((r[-1]-r[0])*(u[-1]+u[0])/2).sum());rot=float(((r[-1]+r[0])/2*(u[-1]-u[0])).sum())
    assert abs(trans+rot-float(z[-1]-z[0]))<1e-10
    out.update(projected_delta=float(z[-1]-z[0]),legacy_forearm_fraction_delta=float(ratio[ii[-1]]-ratio[ii[0]]),radial_delta=float(rad[ii[-1]]-rad[ii[0]]),
        relative_point_motion=trans,anchor_rotation=rot,projected_path=float(np.abs(dd).sum()),
        anchor_min_length=float(length[ii].min()),anchor_max_length=float(length[ii].max()),query_path=float(np.linalg.norm(np.diff(q[ii],axis=0),axis=1).sum()))
    return out

def main():
    assert not(U/'POINT_MEASURE_CONTRACT.json').exists() and not(U/'POINT_MEASURE_AUDIT.json').exists()
    prior=read(U/'RESULTS.json');ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'and r['person']in range(2,8)}
    crops={r['clip_id']:r for p in[old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json']for r in read(p)['rows']if r['clip_id']in ready}
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'))
    guard=old.load('_point_guard',OPS/'word100_observation_guard_20261108/guard.py');ad=old.load('_point_ad',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    cfg=dict(at=stamp(),rows='Exactly all1757 frozen FIT02..07 original positive full/directpartial/reviewedhold candidates; unsupported masks remain denominator',
        points={'body_wrist':[6],'palm0_and_MCP':[30,35,39,43,47]},normalization='Same shoulderwidth body_points; axis opposite elbow3 to body wrist5; no fingertip',
        formulas='Current unit-axis projection vs older forearm-fraction projection, both using same point. Report separately, no choose-best fitting.',
        eligibility='Original >=.8 coverage, >=4 samples, endpoints present, anchor length>.05; retained coordinate mask does not prove physical visibility.',
        intervention='No donor-hand intervention, no old fixture scores, no generated new shape test. Existing aligned_fist is not palm-coordinate-invariant.',
        labels='Original progression/hold annotations, no coordinate-derived relabel, no semantic completion or contact labels.',
        pins={str(p):sha(p)for p in[Path(__file__),U/'RESULTS.json',OPS/'word100_motion_components_20261109/motion.py',OPS/'word100_broad_factorized_learning_20261110/core.py',PRIOR/'projected_direction_core.py',OPS/'word100_shape_direction_audit_20261109/audit.py',OPS/'word100_positive_direction_20261109/compare.py']},private_access=False,final8=False,model_inference=False)
    write(U/'POINT_MEASURE_CONTRACT.json',cfg)
    cache={};pins={}
    def raw(row):
        key=row['clip_id'],row['view']
        if key not in cache:
            sr=ready[row['clip_id']]
            for p,h in[(sr['path'],sr['sha256']),(sr['channel'],sr['channel_sha256'])]:assert sha(p)==h;pins[p]=h
            zz=np.load(sr['path']);ch=np.load(sr['channel']);tt=zz['timestamp_us']
            if row['view']=='actual_crop':
                cr=crops[row['clip_id']];assert sha(cr['path'])==cr['sha256'];pins[cr['path']]=cr['sha256'];cz=np.load(cr['path']);assert np.array_equal(tt,cz['timestamp_us']);cl=guard.quarantine(cz['xy_px'],cz['score'],tt,640,480);x=ad.convert(cl['xy_px'],cz['score'],tt,cz['frame_id'],640,480)['tcn280']
            else:x=ch['full']if row['view']=='full'else ch['feedback']if'feedback'in ch.files else ch['upper_mask']
            cache[key]=(x.copy(),tt.copy())
        x,tt=cache[key];x=x.copy();tt=tt.copy()
        if row['reverse']:x=x[::-1].copy();tt=tt[-1]-tt[::-1];x[:,279]=np.r_[0,np.diff(tt)/1e6]
        if row['label_kind']=='hold':
            aa,bb=row['source_span'];x=x[aa:bb+1].copy();tt=tt[aa:bb+1];x[0,279]=0
        return x,tt
    out=[]
    for row in prior['rows']:
        assert row['person']in range(2,8)
        x,tt=raw(row);aa,bb=row['span'];a=measure(x,tt,aa,bb,'body_wrist');p=measure(x,tt,aa,bb,'palm0_and_MCP')
        assert a['eligible']==row['measurement']['eligible']
        if a['eligible']:assert abs(a['projected_delta']-row['measurement']['delta'])<1e-6
        out.append(dict(bag_index=row['bag_index'],candidate_index=row['candidate_index'],clip_id=row['clip_id'],view=row['view'],reverse=row['reverse'],label_kind=row['label_kind'],span=row['span'],source_span=row['source_span'],body_wrist=a,palm=p))
    summary=[]
    buckets=defaultdict(list)
    for r in out:buckets[r['label_kind'],r['view'],r['reverse']].append(r)
    for key,rr in sorted(buckets.items()):
        for pt in ['body_wrist','palm']:
            gg=[r[pt]for r in rr if r[pt]['eligible']];sg=-1 if key[2] else 1
            summary.append(dict(label_kind=key[0],view=key[1],reverse=key[2],point=pt,candidates=len(rr),eligible=len(gg),unavailable=len(rr)-len(gg),
                projected_delta=numbers([v['projected_delta']for v in gg]),legacy_delta=numbers([v['legacy_forearm_fraction_delta']for v in gg]),
                projected_sign_conflicts=sum(sg*v['projected_delta']<=0 for v in gg) if key[0]!='hold'else None,
                legacy_sign_conflicts=sum(sg*v['legacy_forearm_fraction_delta']<=0 for v in gg) if key[0]!='hold'else None,
                radial_approach_sign_conflicts=sum(sg*v['radial_delta']>=0 for v in gg) if key[0]!='hold'else None,
                rotation_dominates=sum(abs(v['anchor_rotation'])>abs(v['relative_point_motion'])for v in gg)))
    raw_cases={}
    for r in prior['sign_conflicts']:
        raw_cases[r['clip_id'],tuple(r['original_span'])]=1
    matched=[]
    for(cid,span)in sorted(raw_cases):
        for view in['full','actual_crop']:
            x,tt=raw(dict(clip_id=cid,view=view,reverse=False,label_kind='partial'))
            matched.append(dict(clip_id=cid,span=span,view=view,body_wrist=measure(x,tt,*span,'body_wrist'),palm=measure(x,tt,*span,'palm0_and_MCP')))
    for p,h in {**pins,**cfg['pins']}.items():assert sha(p)==h
    result=dict(at=stamp(),rows=out,summary=summary,matched_original_sign_conflicts=matched,pins=pins,contract_sha256=sha(U/'POINT_MEASURE_CONTRACT.json'),no_model_inference=True,no_fixture_change=True,no_relabel=True,scope='FIT measurement agreement, not independent direction accuracy or cross-word generalization')
    write(U/'POINT_MEASURE_AUDIT.json',result)
    for r in summary:
        if not r['reverse']:print(r['label_kind'],r['view'],r['point'],'n',r['candidates'],'available',r['eligible'],'proj conflicts',r['projected_sign_conflicts'],'legacy conflicts',r['legacy_sign_conflicts'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'POINT_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
