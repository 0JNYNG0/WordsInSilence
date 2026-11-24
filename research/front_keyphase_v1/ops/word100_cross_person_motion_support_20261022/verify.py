"""Independent saved arithmetic; known-time angle/DP verifier reused read-only."""
import traceback,copy
import numpy as np
import study as d
U=d.U
V=d.module('cross_person_independent_angle',d.K/'verify.py')
eq=V.eq

def scalar_geometry(q,r,path,req):
    n=len(r['core']);shape=(n,2,3)
    wanted=np.zeros(shape,bool);unknown=np.zeros(shape,bool);strict=np.zeros(shape,bool)
    dist=np.full(shape,np.nan);gap=dist.copy();frac=dist.copy();sel=np.full(shape,-1,int)
    for j in range(n):
        matched=sorted(set(int(i) for i,jj in path if jj==j))
        for h,side in enumerate(('L','R')):
            if side not in req or not r['core'][j] or r['source_rest'][j,h]:continue
            for k in range(3):
                wanted[j,h,k]=True
                rv=r['palm_valid'][j,h] if k==0 else r['arm_valid'][j,h,k-1]
                if not rv:unknown[j,h,k]=True;continue
                valid=q['palm_valid'][:,h] if k==0 else q['arm_valid'][:,h,k-1]
                validbase=[i for i in matched if valid[i]]
                strict[j,h,k]=len(validbase)==len(matched);frac[j,h,k]=len(validbase)/len(matched)
                eligible=set(validbase)
                for i in validbase:
                    eligible.update(t for t in (i-1,i+1) if 0<=t<len(valid) and valid[t])
                if not eligible:unknown[j,h,k]=True;continue
                values=[]
                for i in sorted(eligible):
                    p=q['centre'][i,h] if k==0 else q['arm_xy'][i,h,k-1]
                    target=r['centre'][j,h] if k==0 else r['arm_xy'][j,h,k-1]
                    raw=float(np.sqrt(np.square(p-target).sum()))
                    outside=max(raw-float(q['radius'][i,h])-float(r['radius'][j,h]),0.) if k==0 else raw
                    values.append((outside,i,raw))
                outside,i,raw=min(values)
                gap[j,h,k]=outside;dist[j,h,k]=raw;sel[j,h,k]=i
    return dict(geom_expected=wanted,geom_unknown=unknown,geom_strict=strict,geom_distance=dist,geom_gap=gap,geom_selected=sel,geom_fraction=frac)

def validate(z,q,r,req):
    path=z['path'];a=V.independent(q,r,path,req)
    for k in a:eq(z[k],a[k])
    g=scalar_geometry(q,r,path,req)
    for k in g:
        if g[k].dtype.kind=='f':np.testing.assert_allclose(z[k],g[k],rtol=1e-12,atol=1e-12,equal_nan=True)
        else:eq(z[k],g[k])
    for j in range(len(r['core'])):
        ids=path[path[:,1]==j,0];valid=q['angle_valid'][ids]
        eq(z['angle_strict'][j],a['expected'][j]&r['angle_valid'][j]&valid.all(0))
        np.testing.assert_allclose(z['angle_fraction'][j],valid.mean(0),rtol=0,atol=0)
    assert not np.any(z['angle_strict']&z['unknown'])
    assert not np.any(z['geom_strict']&z['geom_unknown'])
    assert np.isnan(z['distance'][z['unknown']]).all()
    assert np.isnan(z['geom_distance'][z['geom_unknown']]).all()

def run():
    f=d.read(U/'FROZEN.json');d.check(f['pins']);d.check(f['protected'])
    assert not (U/'RECOVERY_RUNNING.json').exists();res=d.read(U/'RESULT.json')
    rows={r['clip_id']:r for r in f['rows']};samples={c:d.data(r) for c,r in rows.items()};seen=set();checked=0
    refs={}
    for row in res['rows']:
        cid,rid,kind=row['query'],row['reference'],row['kind'];assert rows[cid]['signer_id']!=rows[rid]['signer_id']
        assert rows[cid]['word_code']==rows[rid]['word_code'] and rows[cid]['role']=='train_candidate'
        req=f['roles'].get(rows[cid]['word_code'],{}).get('required_sides',['L','R'])
        q,e=samples[cid];r,re=samples[rid];qq,qe,ix=d.transform(q,e,kind);z=d.load(U/'cases'/(row['name']+'.npz'))
        eq(ix,z['source_index']);lo=np.floor(ix).astype(int);hi=np.ceil(ix).astype(int)
        for k in ('angle_valid','palm_valid','arm_valid'):eq(qe[k],e[k][lo]&e[k][hi])
        path=z['path'];assert tuple(path[0])==(0,0) and tuple(path[-1])==(len(ix)-1,len(re['core'])-1)
        ds=np.diff(path,axis=0);assert (ds>=0).all() and (ds<=1).all() and (ds.sum(1)>0).all()
        np.testing.assert_allclose(z['cost'][tuple(path.T)].sum(),V.optimum(z['cost']),rtol=1e-12,atol=1e-10)
        validate(z,qe,re,req)
        for k,v in d.stats(z).items():assert row[k]==v,(row['name'],k)
        assert row['whole_action_correct'] is None
        key=(cid,rid)
        if key not in refs:refs[key]=(z['expected'],z['geom_expected'])
        else:eq(refs[key][0],z['expected']);eq(refs[key][1],z['geom_expected'])
        seen.add((cid,rid,kind));checked+=1
        if checked%180==0:print('VERIFIED',checked,flush=True)
    assert len(seen)==1080 and len(refs)==216
    counts={}
    for row in res['controls']:
        parent=next(t for t in res['rows'] if t['name']==row['parent']);cid=parent['query'];rid=parent['reference']
        q,e=samples[cid];r,re=samples[rid];zz={k:v.copy() for k,v in e.items()};h=('L','R').index(row['side']);mode=row['mode']
        co=d.load(U/'controls'/(row['name']+'.npz'));original=d.load(U/'cases'/(row['parent']+'.npz'));ids=co['removed']
        if mode=='wholearm':zz['arm_valid'][ids,h]=False;zz['arm_xy'][ids,h]=np.nan
        else:
            for k in ('angle_valid','palm_valid'):zz[k][ids,h]=False
            for k in ('angle_low','angle_high','centre','radius'):zz[k][ids,h]=np.nan
        req=f['roles'].get(rows[cid]['word_code'],{}).get('required_sides',['L','R'])
        validate(co,zz,re,req)
        for k in ('expected','geom_expected','path'):eq(co[k],original[k])
        if mode=='wholehand':
            eq(co['unknown'][:,h],co['expected'][:,h]);eq(co['geom_unknown'][:,h,0],co['geom_expected'][:,h,0])
            eq(co['geom_distance'][:,:,1:],original['geom_distance'][:,:,1:])
        elif mode=='wholearm':
            eq(co['geom_unknown'][:,h,1:],co['geom_expected'][:,h,1:]);eq(co['unknown'],original['unknown']);eq(co['distance'],original['distance'])
        else:
            for j in range(len(re['core'])):
                matched=co['path'][co['path'][:,1]==j,0]
                if set(matched)<=set(ids):
                    assert co['unknown'][j,h][co['expected'][j,h]].all()
                    assert co['geom_unknown'][j,h,0] or not co['geom_expected'][j,h,0]
        counts[mode]=counts.get(mode,0)+1
    # Source inputs retain exact arms and masks; diagnostics never reassign sides.
    for cid,(q,e) in samples.items():
        saved=d.load(U/'inputs'/(cid+'.npz'))
        for k in e:eq(e[k],saved[k])
        eq(e['arm_xy'],q['features'][:,:8].reshape(-1,2,2,2).astype(float))
    d.check(f['pins']);d.check(f['protected']);d.check(d.read(U/'RECOVERY_AUTHORIZATION.json')['preserved'])
    d.write(U/'VERIFY.json',dict(status='PASS',at=d.now(),cases=1080,ordered_people_pairs=216,controls=counts,
      verifier='Codex separate scalar point/angle reconstruction plus DP optimum; not independent human semantics review.',
      expected_domain_fixed=True,all_current_missing_not_good=True,partial_evidence_separate=True,
      protected=len(f['protected']),recovery_preserved80_exact=True,
      limits=['Different-person exact corresponding times are unknown.','Saved-cost DP checked; prior frame-cost math verification reused.','Fixed-path missing controls do not test missing-feature realignment.','No new rawRGB tracking/crossing identity accuracy measurement.']))
    print('VERIFY PASS',counts,flush=True)

if __name__=='__main__':
    try:run()
    except BaseException:
        p=U/'VERIFY_FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
