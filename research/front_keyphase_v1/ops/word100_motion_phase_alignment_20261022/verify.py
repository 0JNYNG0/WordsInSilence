"""Independent saved-result verification for the frozen motion alignment study.

No extraction, model inference, learning or production scoring. Invoke only after
RESULT.json exists and RUNNING.json is absent. First failure is preserved.
"""
import traceback
from collections import Counter
import numpy as np
import study as d
import verify_motion as vm

U=d.U
SLICES={'angle':slice(0,15),'palm':slice(15,17),'elbow':slice(17,19),'wrist':slice(19,21)}


def eq(a,b):
    np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-11,equal_nan=True)


def channels(e):
    n=len(e['core']);lo=np.full((n,2,21),np.nan);hi=lo.copy();valid=np.zeros(lo.shape,bool)
    for h in range(2):
        lo[:,h,:15]=e['angle_low'][:,h];hi[:,h,:15]=e['angle_high'][:,h];valid[:,h,:15]=e['angle_valid'][:,h]
        for axis in range(2):
            lo[:,h,15+axis]=e['centre'][:,h,axis]-e['radius'][:,h]
            hi[:,h,15+axis]=e['centre'][:,h,axis]+e['radius'][:,h]
            valid[:,h,15+axis]=e['palm_valid'][:,h]
            for k in range(2):
                at=17+2*k+axis;lo[:,h,at]=hi[:,h,at]=e['arm_xy'][:,h,k,axis]
                valid[:,h,at]=e['arm_valid'][:,h,k]
    valid&=np.isfinite(lo)&np.isfinite(hi)
    lo[~valid]=np.nan;hi[~valid]=np.nan
    return dict(lo=lo,hi=hi,valid=valid)


def domain(e,req):
    out=np.zeros((len(e['core']),2,21),bool)
    for h,side in enumerate(('L','R')):
        if side in req:out[:,h]=np.repeat((e['core']&~e['source_rest'][:,h])[:,None],21,axis=1)
    return out


def project(c,path,n):
    """Independent reconstruction, adapted from already verified LOPO verifier."""
    lo=np.full((n,2,21),np.nan);hi=lo.copy();complete=np.zeros(lo.shape,bool);partial=complete.copy()
    for j in range(n):
        ids=np.unique(path[path[:,1]==j,0]);assert len(ids)
        valid=c['valid'][ids];complete[j]=np.sum(valid,axis=0)==len(ids);partial[j]=np.any(valid,axis=0)&~complete[j]
        low=np.full((2,21),np.inf);high=np.full((2,21),-np.inf)
        # Include neighbors only from an observed directly matched sample.
        for i in ids:
            for k in (i-1,i,i+1):
                if k<0 or k>=len(c['valid']):continue
                allowed=c['valid'][i]&c['valid'][k]
                low[allowed]=np.minimum(low[allowed],c['lo'][k][allowed])
                high[allowed]=np.maximum(high[allowed],c['hi'][k][allowed])
        seen=valid.any(0);lo[j][seen]=low[seen];hi[j][seen]=high[seen]
    return dict(lo=lo,hi=hi,complete=complete,partial=partial)


def persistent(mask):
    out=np.zeros_like(mask)
    for j in range(len(mask)-2):
        same=mask[j]&mask[j+1]&mask[j+2]
        out[j:j+3]|=same
    return out


def reconstruct(qe,re,path,t,req):
    q=channels(qe);anchor=channels(re);p=project(q,path,len(re['core']));wanted=domain(re,req)
    comp=wanted&t['support']&p['complete'];diff=np.maximum(np.maximum(t['low']-p['hi'],p['lo']-t['high']),0.)
    diff[~comp]=np.nan;bad=comp&(diff>1e-6)
    ac=wanted&anchor['valid']&p['complete'];ag=np.maximum(np.maximum(anchor['lo']-p['hi'],p['lo']-anchor['hi']),0.)
    ag[~ac]=np.nan;ab=ac&(ag>1e-6)
    return dict(expected=wanted,comparable=comp,unknown=wanted&~comp,
        reference_insufficient=wanted&~t['support'],query_incomplete=wanted&~p['complete'],query_partial=wanted&p['partial'],
        gap=diff,mismatch=bad,persistent=persistent(bad),within_range=comp&~bad,
        base_comparable=ac,base_gap=ag,base_mismatch=ab,base_persistent=persistent(ab),
        query_low=p['lo'],query_high=p['hi'],query_complete=p['complete'])


def summaries(z):
    result={}
    fields=('expected','comparable','unknown','reference_insufficient','query_incomplete','query_partial','mismatch','persistent','within_range','base_comparable','base_mismatch','base_persistent')
    for name,sl in SLICES.items():
        out={k:int(np.count_nonzero(z[k][:,:,sl])) for k in fields}
        both=z['comparable'][:,:,sl]&z['base_comparable'][:,:,sl]
        out.update(warning=bool(z['persistent'][:,:,sl].any()),any_difference=bool(z['mismatch'][:,:,sl].any()),base_warning=bool(z['base_persistent'][:,:,sl].any()),
            common=int(both.sum()),common_old_difference=int((both&z['base_mismatch'][:,:,sl]).sum()),common_new_difference=int((both&z['mismatch'][:,:,sl]).sum()),
            resolved_observed=int((both&z['base_mismatch'][:,:,sl]&~z['mismatch'][:,:,sl]).sum()),
            old_difference_now_unknown=int((z['base_mismatch'][:,:,sl]&~z['comparable'][:,:,sl]).sum()))
        result[name]=out
    return result


def compare(before,after):
    out={}
    for name,sl in SLICES.items():
        common=before['comparable'][:,:,sl]&after['comparable'][:,:,sl]
        out[name]=dict(common=int(common.sum()),resolved=int((common&before['mismatch'][:,:,sl]&~after['mismatch'][:,:,sl]).sum()),
            added=int((common&~before['mismatch'][:,:,sl]&after['mismatch'][:,:,sl]).sum()),
            old_difference_now_unknown=int((before['mismatch'][:,:,sl]&after['unknown'][:,:,sl]).sum()),
            old_comparable_now_unknown=int((before['comparable'][:,:,sl]&after['unknown'][:,:,sl]).sum()))
    return out


def native(e,req):
    expected=np.zeros((len(e['core']),2),bool)
    for h,side in enumerate(('L','R')):
        if side in req:expected[:,h]=e['core']&~e['source_rest'][:,h]
    return dict(query_expected=expected,query_angle_unknown=expected[:,:,None]&~e['angle_valid'],
        query_palm_unknown=expected&~e['palm_valid'],query_arm_unknown=expected[:,:,None]&~e['arm_valid'])


def proxies(qe,re,path,req):
    core=re['core'][path[:,1]];active=domain(re,req)[path[:,1],:,0]
    return dict(core_path_steps=int(core.sum()),query_outside_core_steps=int((core&~qe['core'][path[:,0]]).sum()),
        query_rest_to_active_handsteps=int((active&qe['source_rest'][path[:,0]]).sum()))


def known_timing(source,transformed,path,ix):
    estimates=[]
    for i in range(len(ix)):
        mapped=sorted(int(j) for ii,j in path if ii==i);assert mapped
        estimates.append(float(np.median(mapped)))
    error=np.abs(np.asarray(estimates)-ix);low=np.floor(ix).astype(int);high=np.ceil(ix).astype(int)
    changing=transformed['core']&(source['measured_change'][low].any(1)|source['measured_change'][high].any(1))
    res={}
    for name,mask in [('core',transformed['core']),('moving',changing)]:
        selected=error[mask]
        res[name]=dict(n=int(mask.sum()),mae=float(np.mean(selected)) if len(selected) else None,
            p95=float(np.percentile(selected,95)) if len(selected) else None,within2=float(np.mean(selected<=2)) if len(selected) else None)
    return res,error,changing


def damage_independent(q,e,side,mode,groups):
    qq={k:v.copy() for k,v in q.items()};z={k:v.copy() for k,v in e.items()};h=0 if side=='L' else 1
    if mode=='shortgap':
        active=np.flatnonzero(e['core']);centre=active[len(active)//2];removed=np.array([centre-1,centre,centre+1])
    else:removed=np.arange(len(e['core']))
    offset=0;names=[]
    for gi,(name,(width,_)) in enumerate(groups.items()):
        remove=(name=='arm_'+side) if mode=='wholearm' else (name in {'shape_'+side,'orient_'+side,'handpos_'+side,'facerel_'+side,'motion_'+side,'bend_'+side,'bimanual'})
        if remove:
            ix=set(map(int,removed))
            if name=='motion_'+side:ix.update(int(i+1) for i in removed if i+1<len(e['core']))
            ix=sorted(ix);qq['present'][ix,gi]=False;qq['features'][ix,offset:offset+width]=np.nan;names.append(name)
        offset+=width
    if mode=='wholearm':
        z['arm_valid'][removed,h]=False;z['arm_xy'][removed,h]=np.nan
    else:
        for key in ('angle_valid','palm_valid','source_rest','observed'):z[key][removed,h]=False
        for key in ('angle_low','angle_high','centre','radius'):z[key][removed,h]=np.nan
        for key in ('missing','unconfirmed'):z[key][removed,h]=True
    return qq,z,removed,names


def legacy_cost(q,r,req,groups):
    """Flat one-sided1.5 legacy cost, independently summed by feature group."""
    shape=(len(q['features']),len(r['features']));numer=np.zeros(shape);denom=np.zeros(shape);offset=0
    for gi,(name,(width,weight)) in enumerate(groups.items()):
        weight=float(weight) if len(req)==2 or name.endswith('_'+req[0]) else 0.
        aq=q['features'][:,offset:offset+width];ar=r['features'][:,offset:offset+width];offset+=width
        if not weight:continue
        presentq=q['present'][:,gi];presentr=r['present'][:,gi]
        both=np.outer(presentq,presentr);one=presentq[:,None]!=presentr[None,:]
        complete=np.outer(np.isfinite(aq).all(1),np.isfinite(ar).all(1))
        dist=np.sqrt(np.mean((aq[:,None,:]-ar[None,:,:])**2,axis=-1));observed=both&complete&np.isfinite(dist)
        numerator=np.zeros(shape);numerator[observed]=dist[observed];numerator[one]=1.5
        counted=observed|one;numer+=numerator*weight;denom+=counted*weight
    return np.divide(numer,denom,out=np.zeros(shape),where=denom>0)


def run():
    assert (U/'RESULT.json').exists() and not (U/'RUNNING.json').exists()
    assert not (U/'VERIFY.json').exists(),'Completed verification must not be rerun'
    f=d.read(U/'FROZEN.json');result=d.read(U/'RESULT.json');d.check(f['pins']);d.check(f['protected'])
    fixed=d.read(U/'FIXTURE_RESULT.json')
    for filename,pinned in fixed['sha256'].items():assert d.sha(U/filename)==pinned,filename
    assert fixed['status']=='PASS' and all(fixed['checks'].values()) and len(fixed['checks'])==10
    assert [len(result[k]) for k in ('known','cross','missing')]==[297,2376,135]
    meta={x['clip_id']:x for x in f['rows']};samples={cid:d.L.prev.data(x) for cid,x in meta.items()}
    for q,e in samples.values():
        own=channels(e);existing=d.L.C.channels(e)
        for k in own:eq(own[k],existing[k])
    templates={};folds={x['query']:x for x in f['folds']}
    for fold in f['folds']:
        cid=fold['query'];people=[meta[x]['signer_id'] for x in fold['references']]
        assert len(people)==len(set(people))==8 and meta[cid]['signer_id'] not in people
        for rid in fold['references']:
            templates[cid,rid]=d.load(d.P/'templates'/(cid+'__'+rid+'.npz'))
    counts=Counter();missing_rest_expansion=0;seen=set()
    for folder in ('known','cross','missing'):
        for row in result[folder]:
            cid,rid,kind=row['query'],row['reference'],row['kind'];q,e=samples[cid];r,re=samples[rid]
            req=f['roles'].get(meta[cid]['word_code'],{}).get('required_sides',['L','R'])
            z=d.load(U/folder/(row['name']+'.npz'));key=(folder,row['name']);assert key not in seen;seen.add(key)
            if folder=='missing':
                qq,qe,removed,names=damage_independent(q,e,row['side'],kind,d.A.F.GROUPS)
                oq,oe,oi=d.damage(q,e,row['side'],kind)
                for key in qq:eq(qq[key],oq[key])
                for key in qe:eq(qe[key],oe[key])
                eq(removed,oi);eq(removed,row['removed'])
                expected_groups=['arm_'+row['side']] if kind=='wholearm' else [x+'_'+row['side'] for x in ('shape','orient','handpos','facerel','motion','bend')]+['bimanual']
                assert set(names)==set(expected_groups)
                bc=legacy_cost(qq,r,req,d.A.F.GROUPS);vm.verify_path(bc,z['base_path']);counts['independent_missing_legacy_dp']+=1
                template=templates[cid,rid]
                before_native=native(e,req);now_native=native(qe,req)
                assert not np.any(before_native['query_expected']&~now_native['query_expected'])
                missing_rest_expansion+=int((now_native['query_expected']&~before_native['query_expected']).sum())
            else:
                qq,qe,ix=d.L.prev.transform(q,e,kind);eq(ix,z['source_index'])
                if folder=='known':
                    base=d.load(d.K.U/'cases'/(row['name']+'.npz'));ch=channels(re)
                    template=dict(low=ch['lo'],high=ch['hi'],support=ch['valid'])
                    assert cid==rid
                else:
                    assert rid in folds[cid]['references'] and meta[cid]['signer_id']!=meta[rid]['signer_id']
                    base=d.load(d.P/'cases'/(row['name']+'.npz'));template=templates[cid,rid]
                eq(z['base_path'],base['path']);bc=base['cost'];counts['reused_base_path_cost']+=1
            vm.verify_cost(qe,re,req,z);eq(z['cost'],bc+z['addition']);vm.verify_path(z['cost'],z['path'])
            wanted=domain(re,req);eq(z['expected'],wanted)
            if folder!='known':eq(z['expected'],template['expected'])
            old=reconstruct(qe,re,z['base_path'],template,req);new=reconstruct(qe,re,z['path'],template,req)
            for k in new:eq(new[k],z[k])
            for k in ('comparable','mismatch','persistent'):eq(old[k],z['old_'+k])
            if folder=='cross':
                # Existing frozen array diagnostics must also agree; no changed template.
                for k in old:eq(old[k],base[k])
            assert not np.any(z['unknown']&z['within_range']) and np.isnan(z['gap'][z['unknown']]).all()
            assert np.array_equal(z['expected'],z['unknown']|z['comparable'])
            assert row['old']==summaries(old) and row['new']==summaries(new)
            assert row['comparison']==compare(old,new)
            assert row['old_proxy']==proxies(qe,re,z['base_path'],req) and row['new_proxy']==proxies(qe,re,z['path'],req)
            wantnative=native(qe,req)
            for k in wantnative:eq(wantnative[k],z[k])
            assert row['native_unknown_angles']==int(z['query_angle_unknown'].sum())
            assert row['native_unknown_palms']==int(z['query_palm_unknown'].sum())
            assert row['native_unknown_arms']==int(z['query_arm_unknown'].sum())
            eq(row['position_support_fraction'],z['position_count'][tuple(z['path'].T)].mean()/(3*len(req)))
            eq(row['direction_support_fraction'],z['direction_count'][tuple(z['path'].T)].mean()/(6*len(req)))
            assert row['path_changed']==(not np.array_equal(z['path'],z['base_path']))
            assert row['old_path_length']==len(z['base_path']) and row['new_path_length']==len(z['path']) and row['whole_action_correct'] is None
            if folder=='known':
                for condition,path in [('old',z['base_path']),('new',z['path'])]:
                    expected,error,moving=known_timing(re,qe,path,ix);eq(error,z[condition+'_error']);eq(moving,z['moving'])
                    for mask,values in expected.items():
                        for field,value in values.items():
                            if value is None:assert row[condition+'_timing'][mask][field] is None
                            else:eq(value,row[condition+'_timing'][mask][field])
                counts['known_timing_checks']+=1
            if folder=='missing':
                h=0 if row['side']=='L' else 1;sl=slice(17,21) if kind=='wholearm' else slice(0,17)
                if kind!='shortgap':assert not z['within_range'][:,h,sl].any()
                else:
                    for j in range(len(re['core'])):
                        matched=z['path'][z['path'][:,1]==j,0]
                        if np.isin(matched,removed).any():assert z['unknown'][j,h,sl][z['expected'][j,h,sl]].all()
                counts['missing_'+kind]+=1
            counts[folder]+=1
            if sum(counts[k] for k in ('known','cross','missing'))%270==0:print('VERIFIED',dict(counts),flush=True)
    assert len(seen)==2808
    d.check(f['pins']);d.check(f['protected'])
    d.write(U/'VERIFY.json',dict(status='PASS',at=d.now(),counts=dict(counts),total=2808,templates_pinned=216,
        fixtures_reused=fixed,protected=len(f['protected']),person_exclusion=True,expected_fixed=True,query_native_gaps_retained=True,
        missing_native_expected_expanded_cells=missing_rest_expansion,
        verifier='Codex phase_design_review subagent: independent per-point motion formula, DP optimum, interval reconstruction, grouping, legacy missing-input cost, known timing and masks. Prior frozen base-cost verification reused.',
        limits=['Cross-person phase correctness has no exact timing ground truth; proxy metrics cannot prove semantic stage matching.',
            'Direction support absent contributes zero added cost, not observed agreement.',
            'Missing controls deliberately remove positive resting witnesses too; native expected may expand, never shrink. New unknown cells are not wrong-sign labels.',
            'Core interval deletion is not a linguistic essentiality annotation. Whole-action correctness remains unset.']))
    print('ALL2808 VERIFIED',flush=True)


if __name__=='__main__':
    try:run()
    except BaseException:
        p=U/'VERIFY_FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
