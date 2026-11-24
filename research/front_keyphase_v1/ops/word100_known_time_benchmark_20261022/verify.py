"""Independent reconstruction of saved paths, masks, envelopes and timing."""
import traceback,copy
from collections import defaultdict
import numpy as np
import study as d
U=d.U
def eq(a,b):assert np.array_equal(a,b,equal_nan=True),'Saved array mismatch'

def optimum(cost):
    n,m=cost.shape;band=max(1,round(.2*max(n,m)));a=np.full((n,m),np.inf)
    for i in range(n):
        centre=i*(m-1)/(n-1)
        for j in range(max(0,int(np.floor(centre-band))),min(m-1,int(np.ceil(centre+band)))+1):
            if i==0 and j==0:best=0.
            else:best=min(a[i-1,j-1] if i and j else np.inf,a[i-1,j] if i else np.inf,a[i,j-1] if j else np.inf)
            a[i,j]=cost[i,j]+best
    return float(a[-1,-1])

def independent(q,r,path,req):
    wanted=np.zeros_like(r['angle_valid']);unknown=np.zeros_like(wanted);bad=np.zeros_like(wanted)
    lo=np.full(wanted.shape,np.nan);hi=lo.copy();dist=lo.copy()
    for h,s in enumerate(('L','R')):
        if s in req:wanted[:,h]=np.broadcast_to((r['core']&~r['source_rest'][:,h])[:,None],(len(r['core']),15))
    for j,h,a in np.argwhere(wanted):
        if not r['angle_valid'][j,h,a]:unknown[j,h,a]=True;continue
        base=set(path[path[:,1]==j,0].tolist());good=sorted(i for i in base if q['angle_valid'][i,h,a]);extended=set(good)
        for i in good:
            for nxt in (i-1,i+1):
                if 0<=nxt<len(q['core']) and q['angle_valid'][nxt,h,a]:extended.add(nxt)
        if not extended:unknown[j,h,a]=True;continue
        lo[j,h,a]=min(float(q['angle_low'][i,h,a]) for i in extended);hi[j,h,a]=max(float(q['angle_high'][i,h,a]) for i in extended)
        dist[j,h,a]=max(float(r['angle_low'][j,h,a])-hi[j,h,a],lo[j,h,a]-float(r['angle_high'][j,h,a]),0.)
        bad[j,h,a]=dist[j,h,a]>1e-6
    persistent=np.zeros_like(wanted)
    for h in (0,1):
        for a in range(15):
            start=None
            for j in range(len(wanted)+1):
                hit=j<len(wanted) and bad[j,h,a]
                if hit and start is None:start=j
                if not hit and start is not None:
                    if j-start>=3:persistent[start:j,h,a]=True
                    start=None
    return dict(expected=wanted,unknown=unknown,mismatch=bad,persistent=persistent,local_low=lo,local_high=hi,distance=dist)

def fixtures():
    n=12;r=dict(core=np.ones(n,bool),source_rest=np.zeros((n,2),bool),angle_valid=np.ones((n,2,15),bool),angle_low=np.zeros((n,2,15)),angle_high=np.zeros((n,2,15)))
    ident=np.column_stack([np.arange(n),np.arange(n)]);res={}
    x=d.C.diagnose(r,r,ident,['R']);assert not x['mismatch'].any();res['identity']=True
    rr=copy.deepcopy(r);rr['angle_low'][4:8,1,0]=90;rr['angle_high'][4:8,1,0]=90
    x=d.C.diagnose(r,rr,ident,['R']);assert x['persistent'][4:8,1,0].all() and x['persistent'].sum()==4;res['missing_observed_excursion']=True
    q=copy.deepcopy(r);q['angle_valid'][:]=False;x=d.C.diagnose(q,rr,ident,['R']);assert np.array_equal(x['unknown'],x['expected']) and not x['mismatch'].any();res['no_evidence_separate_not_match']=True
    q=copy.deepcopy(r);q['angle_valid'][:,1,0]=False;x=d.C.diagnose(q,rr,ident,['R']);assert x['unknown'][:,1,0].all() and not x['unknown'][:,1,1:].any() and not x['expected'][:,0].any();res['independent_angles_optional_hand']=True
    rr['source_rest'][4:8,1]=True;x=d.C.diagnose(r,rr,ident,['R']);assert not x['expected'][4:8,1].any();res['positive_rest_preserved']=True
    # This limitation is expected: endpoints of a monotonic change span omitted middle.
    rr=copy.deepcopy(r);rr['angle_low'][:,1,0]=np.arange(n)*5;rr['angle_high'][:,1,0]=np.arange(n)*5
    ix=np.array([0,1,2,3,8,9,10,11]);q={k:v[ix].copy() for k,v in rr.items()}
    path=np.array([(int(np.argmin(abs(ix-j))),j) for j in range(n)])
    x=d.C.diagnose(q,rr,path,['R']);assert not x['mismatch'][4:8,1,0].any();res['monotonic_deletion_not_distinguishable']=True
    return res

def run():
    f=d.read(U/'FROZEN.json');assert not (U/'RUNNING.json').exists();d.prev.prev.prev.pipe.check(f['pins']);d.prev.prev.prev.pipe.check(f['protected'])
    result=d.read(U/'RESULT.json')['rows'];assert len(result)==len(f['cases'])==297
    rows={r['clip_id']:r for r in f['rows']};es={c:d.load(d.P/'evidence'/(c+'.npz')) for c in rows};qs={c:d.prev.prev.sample(r) for c,r in rows.items()}
    counters=defaultdict(int);groups=defaultdict(list);timingerrors=defaultdict(list);table={}
    for spec in f['cases']:
        cid,kind=spec['clip'],spec['kind'];row=rows[cid];req=f['roles'].get(row['word_code'],{}).get('required_sides',['L','R']);e=es[cid];q=qs[cid];name=kind+'__'+cid
        z=d.load(U/'cases'/(name+'.npz'));m=d.read(U/'cases'/(name+'.json'));ix=np.array(spec['source_index']);deleted=np.array(spec['deleted'],int)
        eq(z['source_index'],ix);eq(z['deleted'],deleted);assert m==next(r for r in result if r['clip']==cid and r['kind']==kind)
        planned,dels=d.mapping(len(e['core']),e['core'],kind);eq(planned,ix);eq(dels,deleted)
        a=np.floor(ix).astype(int);b=np.ceil(ix).astype(int);fraction=ix-a
        qq,qz=d.resample(q,e,ix)
        for k in ('angle_valid','observed','palm_valid','source_rest','core'):eq(qz[k],e[k][a]&e[k][b])
        for k in ('features',):
            w=fraction[:,None];expected=q[k][a]*(1-w)+q[k][b]*w;expected[~(np.isfinite(q[k][a])&np.isfinite(q[k][b]))]=np.nan;eq(qq[k],expected)
        for k in ('angle_low','angle_high'):
            w=fraction[:,None,None];expected=e[k][a]*(1-w)+e[k][b]*w;expected[~(np.isfinite(e[k][a])&np.isfinite(e[k][b]))]=np.nan;eq(z['query_'+k],expected)
        eq(z['query_angle_valid'],qz['angle_valid']);eq(z['query_core'],qz['core']);eq(qz['frame_id'],np.arange(len(ix)))
        if len(deleted):assert not set(deleted)&set(ix) and np.array_equal(np.setdiff1d(np.arange(len(e['core'])),ix.astype(int)),deleted)
        path=z['path'];assert tuple(path[0])==(0,0) and tuple(path[-1])==(len(ix)-1,len(e['core'])-1)
        delta=np.diff(path,axis=0);assert np.all(delta>=0) and np.all(delta<=1) and np.all(delta.sum(1)>0)
        opt=optimum(z['cost']);val=float(z['cost'][tuple(path.T)].sum());np.testing.assert_allclose(val,opt,rtol=1e-12,atol=1e-10);np.testing.assert_allclose(val,m['dtw_accumulated'],rtol=0,atol=1e-12)
        expected=independent(qz,e,path,req)
        for k in expected:eq(z[k],expected[k])
        # Compare the original-time truth only here, outside prediction.
        predicted=np.array([np.median([j for ii,j in path if ii==i]) for i in range(len(ix))]);err=np.abs(predicted-ix);core=qz['core'];motion=core&(e['measured_change'][a].any(1)|e['measured_change'][b].any(1))
        for k,sel in [('core_timing',core),('change_timing',motion)]:
            x=err[sel];assert m[k]['n']==len(x)
            if len(x):
                np.testing.assert_allclose([m[k]['mae_frames'],m[k]['p95_frames'],m[k]['within2_frames']],[x.mean(),np.percentile(x,95),(x<=2).mean()],rtol=1e-12,atol=1e-12)
            if k=='change_timing':timingerrors[kind].extend(x.tolist())
        assert m['local_warning']==bool(z['persistent'].any()) and m['previous_warning']==bool(z['previous_angle_warning'].any())
        for key,field in [('expected_angle_cells','expected'),('unknown_angle_cells','unknown'),('mismatch_angle_cells','mismatch'),('persistent_angle_cells','persistent')]:assert m[key]==int(z[field].sum())
        removed=np.zeros(len(e['core']),bool);removed[deleted]=True;flag=z['persistent'].any((1,2))
        assert m['hit_deleted']==bool((flag&removed).any()) and m['deleted_frames_with_warning']==int((flag&removed).sum()) and m['preserved_frames_with_warning']==int((flag&~removed).sum())
        assert m['wrong_sign'] is None
        groups[kind].append(m);counters['cases']+=1;counters['dtw_cost_cells']+=z['cost'].size
        # One fixed reference domain for all transformations of each source.
        if cid not in table:table[cid]=z['expected']
        else:eq(table[cid],z['expected'])
    summary={}
    for kind,rr in groups.items():
        errs=np.array(timingerrors[kind]);summary[kind]=dict(n=len(rr),previous_warning=sum(r['previous_warning'] for r in rr),local_warning=sum(r['local_warning'] for r in rr),localized_deletion=sum(r['hit_deleted'] for r in rr),
          expected_angle_cells=sum(r['expected_angle_cells'] for r in rr),unknown_angle_cells=sum(r['unknown_angle_cells'] for r in rr),preserved_frames_flagged=sum(r['preserved_frames_with_warning'] for r in rr),
          change_timing_frames=len(errs),change_mae_frames=float(errs.mean()) if len(errs) else None,change_p95_frames=float(np.percentile(errs,95)) if len(errs) else None,change_within2=float((errs<=2).mean()) if len(errs) else None)
    test=fixtures();d.prev.prev.prev.pipe.check(f['pins']);d.prev.prev.prev.pipe.check(f['protected'])
    assert len(list((U/'cases').glob('*.npz')))==len(list((U/'cases').glob('*.json')))==297
    d.write(U/'VERIFY.json',dict(status='PASS',at=d.now(),counts=dict(counters),conditions=summary,fixtures=test,protected=len(f['protected']),
      verifier='Codex independent dynamic-programming optimal cost and scalar envelope reconstruction; same agent, not semantic expert.',
      limits=['Timing truth is transform provenance, not linguistic importance.', 'Same detector source gives correlated errors; unknowns stay unknown.', 'Candidate local convex support can hide monotonic omissions; expected fixture confirms limitation.', 'Independent verifier does not regenerate every frame-cost primitive; saved cost paths optimally checked; prior cost arithmetic verification reused.']))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    import json
    try:run()
    except BaseException:
        p=U/'VERIFY_FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
