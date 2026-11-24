"""Independent saved-result checks. No DTW rerun or parameter selection."""
import copy, itertools, traceback
from collections import Counter, defaultdict
import numpy as np
import study as d
U,P=d.U,d.P
E=1e-6

def same(a,b):
    assert np.array_equal(a,b,equal_nan=True), 'array mismatch'

def angles_acos(x):
    out=[]
    for ids in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)):
        for a,b,c in zip(ids,ids[1:],ids[2:]):
            v=x[...,a,:].astype(float)-x[...,b,:];w=x[...,c,:].astype(float)-x[...,b,:]
            nv=np.sqrt(np.sum(v*v,axis=-1));nw=np.sqrt(np.sum(w*w,axis=-1))
            with np.errstate(invalid='ignore',divide='ignore'):
                ang=np.degrees(np.arccos(np.clip(np.sum(v*w,axis=-1)/(nv*nw),-1,1)))
            out.append(np.where((nv>E)&(nw>E),ang,np.nan))
    return np.stack(out,axis=-1)

def span(z,inds,h):
    # Pairwise disjoint intervals independently of production max/min span rule.
    inds=list(inds);aa=np.zeros(15,bool);pp=False
    for i,j in itertools.combinations(inds,2):
        valid=z['angle_valid'][i,h]&z['angle_valid'][j,h]
        aa|=valid&((z['angle_low'][i,h]>z['angle_high'][j,h]+E)|(z['angle_low'][j,h]>z['angle_high'][i,h]+E))
        if z['palm_valid'][i,h] and z['palm_valid'][j,h]:
            dist=float(np.sqrt(np.sum((z['centre'][i,h]-z['centre'][j,h])**2)))
            pp|=dist>z['radius'][i,h]+z['radius'][j,h]+E
    return aa,pp

def checksaved(z,r,path,expected,required,saved,meta):
    same(saved['path'],path);same(saved['expected'],expected)
    assert np.all(np.diff(path,axis=0)>=0) and np.all(np.diff(path,axis=0)<=1)
    assert tuple(path[0])==(0,0) and tuple(path[-1])==(len(z['core'])-1,len(r['core'])-1)
    aw=np.zeros_like(z['angle_valid']);pw=np.zeros_like(z['observed'])
    for direction in (0,1):
        groups=defaultdict(set)
        for a,b in path:groups[int(a if direction==0 else b)].add(int(b if direction==0 else a))
        for fixed,values in groups.items():
            for h,side in enumerate(('L','R')):
                if side not in required:continue
                if direction==0:
                    inds=[j for j in sorted(values) if expected[j,h].any()];targets=[fixed];src=r
                else:
                    if not expected[fixed,h].any():continue
                    inds=[j for j in sorted(values) if z['core'][j]];targets=inds;src=z
                aa,pp=span(src,inds,h)
                aw[targets,h]|=aa;pw[targets,h]|=pp
    same(saved['angle_warning'],aw);same(saved['position_warning'],pw)
    gaps=np.zeros_like(z['angle_valid'])
    for h,side in enumerate(('L','R')):
        if side in required:
            gaps[:,h]=z['unconfirmed'][:,h]&z['core'][:,None]&~z['source_rest'][:,h,None]
    same(saved['insufficient_timeline'],gaps);same(saved['missing_timeline'],gaps&z['missing'])
    assert meta['wrong_action'] is None
    assert meta['warning_angle_frames']==int(aw.sum()) and meta['warning_position_frames']==int(pw.sum())
    assert meta['gap_angle_frames']==int(gaps.sum()) and meta['missing_angle_frames']==int((gaps&z['missing']).sum())

def run():
    f=d.read(U/'FROZEN.json');d.prev.prev.pipe.check(f['pins']);d.prev.prev.pipe.check(f['protected'])
    assert not list(U.glob('*RUNNING.json'))
    evidence={};data={};counts=Counter();original_counts=Counter();counts['protected_pins']=len(f['protected'])
    for row in f['rows']:
        cid=row['clip_id'];q=d.prev.sample(row);z=d.load(U/'evidence'/(cid+'.npz'))
        native=d.prev.prev.Obs.load(row['native']);origin,scale,_=d.prev.prev.body_frame(native)
        src=d.load(d.prev.R/'schedules'/(cid+'__source.npz'));same(z['source_rest'],src['rest'])
        bangle=angles_acos(q['pair_b']);av=(q['angle_state']==4)&np.isfinite(bangle)
        same(z['angle_valid'],av)
        np.testing.assert_allclose(z['angle_low'],np.where(av,np.minimum(q['angle_deg'],bangle),np.nan),rtol=0,atol=1e-6,equal_nan=True)
        np.testing.assert_allclose(z['angle_high'],np.where(av,np.maximum(q['angle_deg'],bangle),np.nan),rtol=0,atol=1e-6,equal_nan=True)
        xy=q['xy_px'][:,33:].reshape(-1,2,21,2).astype(float);vv=(q['observed'][:,33:]&q['usable'][:,33:]).reshape(-1,2,21)
        same(z['observed'],vv.any(2));pa=xy[:,:,[0,5,9,13,17]].mean(2);pb=q['pair_b'].astype(float)[:,:,[0,5,9,13,17]].mean(2)
        pv=vv[:,:,[0,5,9,13,17]].all(2)&q['pair_valid'][:,:,[0,5,9,13,17]].all(2)&q['pair_available']&np.isfinite(pa).all(2)&np.isfinite(pb).all(2)
        same(z['palm_valid'],pv);cen=((pa+pb)/2-origin[:,None,:])/scale;rad=np.sqrt(np.sum((pa-pb)**2,axis=2))/(2*scale)+1/scale
        cen[~pv]=np.nan;rad[~pv]=np.nan;same(z['centre'],cen);same(z['radius'],rad)
        same(z['core'],q['core']);same(z['missing'],q['angle_state']==0);same(z['unconfirmed'],q['angle_state']!=4)
        same(z['frame_id'],q['frame_id']);same(z['timestamp_us'],q['timestamp_us'])
        pos=np.zeros_like(z['observed']);ang=np.zeros_like(z['angle_valid'])
        for i in range(1,len(z['core'])):
            if z['frame_id'][i]!=z['frame_id'][i-1]+1 or z['timestamp_us'][i]<=z['timestamp_us'][i-1]:continue
            for h in (0,1):ang[i,h],pos[i,h]=span(z,[i-1,i],h)
        same(z['position_change'],pos);same(z['angle_change'],ang);same(z['measured_change'],pos|ang.any(2))
        same(z['lowered_witness'],src['rest']);same(z['rest_change_conflict'],src['rest']&(pos|ang.any(2)))
        same(z['observed_without_change_evidence'],z['observed']&~(pos|ang.any(2)))
        original_counts.update(clip=1,location_seen=int(z['observed'].sum()),position_change=int(pos.sum()),angle_change=int(ang.sum()),lowered=int(src['rest'].sum()),seen_and_lowered=int((z['observed']&src['rest']).sum()),rest_change_conflict=int(z['rest_change_conflict'].sum()))
        counts.update(clips=1,observed_no_change=int(z['observed_without_change_evidence'].sum()),frame_angle_cells=z['angle_valid'].size)
        evidence[cid]=z;data[cid]=q
    assert dict(original_counts)==d.read(U/'ORIGINAL.json')['counts']
    sums=defaultdict(lambda:dict(n=0,angle=0,pos=0,has=0));controlsum=defaultdict(lambda:dict(n=0,has=0,cells=0));missingcounts=Counter()
    for row in f['rows']:
        cid=row['clip_id'];z=evidence[cid];q=data[cid];req=f['roles'].get(row['word_code'],{}).get('required_sides',['L','R'])
        refs=[r for r in f['rows'] if r['word_code']==row['word_code'] and r['signer_id']!=row['signer_id']]
        for c in d.prev.CONDITIONS:
            av=np.zeros_like(z['angle_valid'],int);pv=np.zeros_like(z['observed'],int)
            for rr in refs:
                rid=rr['clip_id'];name=c+'__'+cid+'__'+rid;old=d.load(P/'pairs'/(name+'.npz'));saved=d.load(U/'pairs'/(name+'.npz'));meta=d.read(U/'pairs'/(name+'.json'))
                checksaved(z,evidence[rid],old['path'],old['expected'],req,saved,meta);av+=saved['angle_warning'];pv+=saved['position_warning'];counts['original_pairs']+=1
            vs=d.load(U/'queries'/(c+'__'+cid+'.npz'));m=d.read(U/'queries'/(c+'__'+cid+'.json'))
            same(vs['angle_votes'],av);same(vs['position_votes'],pv)
            assert m['angle_consensus_frames']==int((av>=5).sum()) and m['position_consensus_frames']==int((pv>=5).sum())
            ss=sums[c];ss['n']+=1;ss['angle']+=m['angle_consensus_frames'];ss['pos']+=m['position_consensus_frames'];ss['has']+=int((av>=5).any())
        for typ,side in [('whole','L'),('whole','R'),('short',req[-1])]:
            h=('L','R').index(side);unique_gap=None
            for rr in refs:
                rid=rr['clip_id'];name=typ+'_'+side+'__'+cid+'__'+rid;old=d.load(P/'stress'/(name+'.npz'));saved=d.load(U/'missing'/(name+'.npz'));meta=d.read(U/'missing'/(name+'.json'))
                inds=old['removed_query_rows'];zz=copy.deepcopy(z)
                for k in ('observed','palm_valid','source_rest','angle_valid'):zz[k][inds,h]=False
                for k in ('centre','radius','angle_low','angle_high'):zz[k][inds,h]=np.nan
                for k in ('missing','unconfirmed'):zz[k][inds,h]=True
                checksaved(zz,evidence[rid],old['path'],old['expected'],req,saved,meta)
                wanted=int(zz['core'][inds].sum())*15 if side in req else 0
                assert meta['deleted_cells_marked']==wanted and int(saved['missing_timeline'][inds,h].sum())==wanted
                assert meta['range_preserved']
                if unique_gap is None:unique_gap=saved['missing_timeline']
                else:same(unique_gap,saved['missing_timeline'])
                counts['missing_pairs']+=1
            missingcounts[typ+'_conditions']+=1;missingcounts[typ+'_required_conditions']+=int(side in req)
            missingcounts[typ+'_deleted_cells_unique']+=wanted
        for kind in ('speed0.75','speed1.5','pause','repeat2','repeat3'):
            # Transformation schema is independently checked below; no new extraction/path computation.
            if kind.startswith('repeat'):
                k=int(kind[-1]);zz={a:np.concatenate([v]*k) for a,v in z.items()};zz['frame_id']=np.arange(len(zz['core']));zz['timestamp_us']=np.rint(np.arange(len(zz['core']))*1e6/30).astype(np.int64)
            else:
                _,zz,ix=d.transform(q,z,kind)
                a=np.floor(ix).astype(int);b=np.ceil(ix).astype(int)
                for field in ('observed','palm_valid','angle_valid','source_rest','core'):same(zz[field],z[field][a]&z[field][b])
                same(zz['missing'],z['missing'][a]|z['missing'][b]);same(zz['unconfirmed'],~zz['angle_valid'])
                for field in ('centre','radius','angle_low','angle_high'):
                    w=(ix-a).reshape((-1,)+(1,)*(z[field].ndim-1));ex=(1-w)*z[field][a]+w*z[field][b]
                    ex[~(np.isfinite(z[field][a])&np.isfinite(z[field][b]))]=np.nan;same(zz[field],ex)
                if kind=='pause':assert len(ix)==len(z['core'])+15
                else:assert len(ix)==round((len(z['core'])-1)*float(kind[5:]))+1
            av=np.zeros_like(zz['angle_valid'],int)
            for rr in refs:
                rid=rr['clip_id'];rz=evidence[rid];name=kind+'__'+cid+'__'+rid;saved=d.load(U/'controls'/(name+'.npz'));meta=d.read(U/'controls'/(name+'.json'));old=d.load(P/'pairs'/('baseline__'+cid+'__'+rid+'.npz'))
                if kind.startswith('repeat'):
                    rrz={a:np.concatenate([v]*k) for a,v in rz.items()};expected=np.concatenate([old['expected']]*k);path=np.concatenate([old['path']+[j*len(z['core']),j*len(rz['core'])] for j in range(k)])
                else:rrz=rz;expected=old['expected'];path=saved['path']
                checksaved(zz,rrz,path,expected,req,saved,meta);av+=saved['angle_warning'];counts['control_pairs']+=1
            rowm=next(x for x in d.read(U/'CONTROLS.json')['rows'] if x['query']==cid and x['kind']==kind)
            assert rowm['consensus_angle_frames']==int((av>=5).sum()) and rowm['has_consensus']==bool((av>=5).any())
            ss=controlsum[kind];ss['n']+=1;ss['has']+=int((av>=5).any());ss['cells']+=int((av>=5).sum())
        print('CHECKED',cid,flush=True)
    assert counts['original_pairs']==648 and counts['missing_pairs']==648 and counts['control_pairs']==1080
    for sub,n in [('pairs',648),('missing',648),('controls',1080)]:
        assert len(list((U/sub).glob('*.json')))==n and len(list((U/sub).glob('*.npz')))==n
    tests=[]
    z=copy.deepcopy(next(iter(evidence.values())))
    z={k:v[:4].copy() for k,v in z.items()}
    z['frame_id']=np.arange(4);z['timestamp_us']=np.arange(4)*33333;z['core'][:]=True
    z['observed'][:]=True;z['palm_valid'][:]=True;z['angle_valid'][:]=True;z['centre'][:]=0;z['radius'][:]=.01;z['angle_low'][:]=40;z['angle_high'][:]=42;z['source_rest'][:]=False
    ss=d.C.states(z);assert not ss['measured_change'].any() and ss['location_seen'].all() and not ss['lowered_witness'].any();tests.append('visible_static_not_motion_or_rest')
    z['source_rest'][:]=True;assert d.C.states(z)['lowered_witness'].all();z['source_rest'][:]=False;tests.append('rest_requires_positive_witness')
    z['centre'][2,0,0]=1;z['angle_low'][2,0,0]=80;z['angle_high'][2,0,0]=82
    assert d.C.states(z)['position_change'][2,0] and d.C.states(z)['angle_change'][2,0,0];tests.append('separate_position_and_angle_changes')
    z['angle_valid'][1,0,0]=False;assert not d.C.states(z)['angle_change'][2,0,0] and d.C.states(z)['position_change'][2,0];tests.append('unknown_angle_preserves_position')
    z['frame_id'][2:]+=1;assert not d.C.states(z)['measured_change'][2].any();tests.append('no_bridge_across_frame_gap')
    z['frame_id']=np.arange(4);z['palm_valid'][1,0]=False;assert not d.C.states(z)['position_change'][2,0];tests.append('no_bridge_across_unobserved_palm')
    base=d.C.states(z);zz=copy.deepcopy(z);zz['centre']+=100;zz['timestamp_us']*=2
    for k in ('position_change','angle_change','lowered_witness'):same(base[k],d.C.states(zz)[k])
    tests.append('translation_and_timestamp_scale_invariance')
    d.prev.prev.pipe.check(f['pins']);d.prev.prev.pipe.check(f['protected'])
    result=dict(status='PASS',at=d.now(),verifier='Codex separate pairwise interval/cosine arithmetic; same agent, not independent semantic expert',counts=dict(counts),evidence_counts=dict(original_counts),original_summary=dict(sums),control_summary=dict(controlsum),missing=dict(missingcounts),tests=tests,
        limits=['Two detector estimates are not calibrated confidence; diagnostic specificity fails normal controls.', 'Uniform resampling is synthetic and cannot prove real camera speed behavior.', 'Repeated paths use known construction; no new repeat-count correctness test.', 'Saved resampled DTW paths checked for validity, not independently reoptimized; all original paths exact reused.', 'No clinical/semantic correctness or classifier accuracy measured.'])
    d.write(U/'VERIFY.json',result);print('PASS',dict(counts),dict(controlsum),flush=True)

if __name__=='__main__':
    try:run()
    except BaseException:
        p=U/'VERIFY_FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
