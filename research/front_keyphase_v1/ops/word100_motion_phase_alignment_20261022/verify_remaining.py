"""Documented verifier precision repair; verify only the remaining 135 controls.

The first verifier and its first-error file are retained byte-for-byte. No study,
candidate, saved cost/path/score or tolerance is changed. `proof` discriminates
the failed branch with saved arithmetic identities and reproduces its signature.
`run` requires the proof and verifies only missing-input controls.
"""
import sys,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import study as d
vm=d.module('verify_motion',d.U/'verify_motion.py')
v=d.module('motion_original_verifier',d.U/'verify.py')
assert Path(vm.__file__).resolve()==(d.U/'verify_motion.py').resolve()
assert Path(v.__file__).resolve()==(d.U/'verify.py').resolve()

U=d.U


def legacy_cost(q,r,req,groups,detail=False):
    shape=(len(q['features']),len(r['features']));numer=np.zeros(shape);denom=np.zeros(shape);offset=0;details=[]
    for gi,(name,(width,weight)) in enumerate(groups.items()):
        weight=float(weight) if len(req)==2 or name.endswith('_'+req[0]) else 0.
        aq=q['features'][:,offset:offset+width];ar=r['features'][:,offset:offset+width];offset+=width
        if not weight:continue
        presentq=q['present'][:,gi];presentr=r['present'][:,gi]
        both=np.outer(presentq,presentr);one=presentq[:,None]!=presentr[None,:]
        complete=np.outer(np.isfinite(aq).all(1),np.isfinite(ar).all(1))
        squared=(aq[:,None,:]-ar[None,:,:])**2
        sums=np.nansum(squared,axis=-1);counts=np.sum(~np.isnan(squared),axis=-1)
        # Explicit runtime primitive dtype: float32 sum / int64 array -> float64.
        mean=np.divide(sums,np.maximum(counts,1));mean=np.where(counts>0,mean,np.nan)
        dist=np.sqrt(mean);observed=both&complete&np.isfinite(dist)
        numerator=np.zeros(shape);numerator[observed]=dist[observed];numerator[one]=1.5
        counted=observed|one;numer+=numerator*weight;denom+=counted*weight
        if detail:
            olddist=np.sqrt(np.mean(squared,axis=-1));ok=np.isfinite(dist)&np.isfinite(olddist)
            details.append(dict(group=name,feature_dtype=str(aq.dtype),squared_dtype=str(squared.dtype),
                old_mean_dtype=str(np.mean(squared,axis=-1).dtype),sum_dtype=str(sums.dtype),count_dtype=str(counts.dtype),
                corrected_mean_dtype=str(mean.dtype),max_group_distance_difference=float(np.max(np.abs(dist[ok]-olddist[ok]))) if ok.any() else None))
    cost=np.divide(numer,denom,out=np.zeros(shape),where=denom>0)
    return (cost,details) if detail else cost


def proof():
    assert not (U/'RUNNING.json').exists() and not (U/'VERIFY.json').exists()
    assert not (U/'VERIFY_RECOVERY_PROOF.json').exists()
    error=(U/'VERIFY_FIRST_ERROR.txt').read_text('utf8');code=(U/'verify.py').read_text('utf8')
    assert "vm.verify_cost(qe,re,req,z);eq(z['cost'],bc+z['addition']);vm.verify_path" in error
    assert "for folder in ('known','cross','missing'):" in code
    result=d.read(U/'RESULT.json');f=d.read(U/'FROZEN.json');lookup={x['clip_id']:x for x in f['rows']}
    row=result['missing'][0];q,e=d.L.prev.data(lookup[row['query']]);r,re=d.L.prev.data(lookup[row['reference']])
    req=f['roles'].get(lookup[row['query']]['word_code'],{}).get('required_sides',['L','R'])
    qq,qe,_,_=v.damage_independent(q,e,row['side'],row['kind'],d.A.F.GROUPS)
    z=d.load(U/'missing'/(row['name']+'.npz'));old=v.legacy_cost(qq,r,req,d.A.F.GROUPS);fixed,detail=legacy_cost(qq,r,req,d.A.F.GROUPS,True)
    oldtotal=old+z['addition'];diff=np.abs(z['cost']-oldtotal)
    bad=~np.isclose(z['cost'],oldtotal,rtol=2e-11,atol=2e-11,equal_nan=True)
    assert z['cost'].size==9682 and int(bad.sum())==4405
    np.testing.assert_allclose(diff[bad].max(),9.35070155e-9,rtol=1e-8,atol=1e-16)
    v.eq(z['cost'],fixed+z['addition'])
    # Single-case primitive audit to identify the precision cause, not rescoring a model.
    weights=d.A.C.weights_for(req,(False,False),d.A.F.GROUPS)
    runtime,_=d.A.legacy(qq,r,group_weights=weights,hand_participation=False,one_sided_mode='flat')
    v.eq(runtime,fixed);v.eq(z['cost'],runtime+z['addition'])
    # Narrow saved-array identity only: no new features, DP, diagnostics or scores
    # for the 2,673 cases already passed by the original ordered verifier.
    checked=0;outputs={}
    for folder in ('known','cross'):
        for row2 in result[folder]:
            file=U/folder/(row2['name']+'.npz');saved=d.load(file)
            basefile=(d.K.U/'cases' if folder=='known' else d.P/'cases')/(row2['name']+'.npz')
            base=d.load(basefile)
            assert np.array_equal(saved['cost'],base['cost']+saved['addition'])
            outputs[str(file)]=d.sha(file);checked+=1
    assert checked==2673
    for row2 in result['missing']:
        file=U/'missing'/(row2['name']+'.npz');outputs[str(file)]=d.sha(file)
    preserved={str(U/name):d.sha(U/name) for name in ('verify.py','verify_motion.py','VERIFY_FIRST_ERROR.txt','candidate.py','study.py','FROZEN.json','RESULT.json','FIXTURE_RESULT.json')}
    d.write(U/'VERIFY_RECOVERY_PROOF.json',dict(status='PASS_VERIFIER_PRECISION_CAUSE_IDENTIFIED',at=d.now(),
        first_failed_missing=row['name'],shape=list(z['cost'].shape),cells=z['cost'].size,violations=int(bad.sum()),max_old_error=float(diff.max()),
        max_corrected_error=float(np.abs(z['cost']-(fixed+z['addition'])).max()),runtime_base_match_max=float(np.abs(runtime-fixed).max()),
        precision_detail=detail,preserved=preserved,outputs=outputs,saved_equation_checks=checked,
        previous_execution=dict(session=94105,exit_code=1,chunk='cb6ad0',last_progress={'known':297,'cross':2133},
            established_completed={'known':297,'cross':2376},
            reasoning='Original deterministic folder order known,cross,missing. Failure line221 is the saved combined-cost equality. All2673 known/cross equality statements are bitwise true on immutable saved arrays. First missing case exactly reproduces the error signature 4405/9682 and9.35070155e-09. Thus previous2673 cases passed every earlier and later assertion before entering first missing iteration.'),
        recovery='Run remaining135 missingcontrols only. Same2e-11 tolerance; replace float32 mean/sqrt verifier primitive with float32 sum / int64 count -> float64 sqrt matching recorded runtime. Experiment and all saved outputs unchanged.'))
    print('PRECISION PROOF PASS;2673 REUSED;135 REMAIN',flush=True)


def run():
    proof=d.read(U/'VERIFY_RECOVERY_PROOF.json');assert proof['status']=='PASS_VERIFIER_PRECISION_CAUSE_IDENTIFIED'
    assert not (U/'RUNNING.json').exists() and not (U/'VERIFY.json').exists()
    d.check(proof['preserved']);d.check(proof['outputs'])
    f=d.read(U/'FROZEN.json');d.check(f['pins']);d.check(f['protected']);result=d.read(U/'RESULT.json')
    fixed=d.read(U/'FIXTURE_RESULT.json')
    for name,pin in fixed['sha256'].items():assert d.sha(U/name)==pin
    assert fixed['status']=='PASS' and len(fixed['checks'])==10 and all(fixed['checks'].values())
    lookup={x['clip_id']:x for x in f['rows']};samples={cid:d.L.prev.data(x) for cid,x in lookup.items()}
    counts=Counter();rest_expansion=0
    for row in result['missing']:
        cid,rid,mode=row['query'],row['reference'],row['kind'];q,e=samples[cid];r,re=samples[rid]
        req=f['roles'].get(lookup[cid]['word_code'],{}).get('required_sides',['L','R'])
        template=d.load(d.P/'templates'/(cid+'__'+rid+'.npz'));z=d.load(U/'missing'/(row['name']+'.npz'))
        qq,qe,removed,names=v.damage_independent(q,e,row['side'],mode,d.A.F.GROUPS)
        oq,oe,oi=d.damage(q,e,row['side'],mode)
        for key in qq:v.eq(qq[key],oq[key])
        for key in qe:v.eq(qe[key],oe[key])
        v.eq(removed,oi);v.eq(removed,row['removed'])
        wantgroups=['arm_'+row['side']] if mode=='wholearm' else [x+'_'+row['side'] for x in ('shape','orient','handpos','facerel','motion','bend')]+['bimanual']
        assert set(names)==set(wantgroups)
        bc=legacy_cost(qq,r,req,d.A.F.GROUPS);vm.verify_path(bc,z['base_path'])
        vm.verify_cost(qe,re,req,z);v.eq(z['cost'],bc+z['addition']);vm.verify_path(z['cost'],z['path'])
        old=v.reconstruct(qe,re,z['base_path'],template,req);new=v.reconstruct(qe,re,z['path'],template,req)
        for key in new:v.eq(new[key],z[key])
        for key in ('comparable','mismatch','persistent'):v.eq(old[key],z['old_'+key])
        v.eq(z['expected'],v.domain(re,req));v.eq(z['expected'],template['expected'])
        assert not np.any(z['unknown']&z['within_range']) and np.isnan(z['gap'][z['unknown']]).all()
        assert np.array_equal(z['expected'],z['unknown']|z['comparable'])
        assert row['old']==v.summaries(old) and row['new']==v.summaries(new) and row['comparison']==v.compare(old,new)
        assert row['old_proxy']==v.proxies(qe,re,z['base_path'],req) and row['new_proxy']==v.proxies(qe,re,z['path'],req)
        before_native=v.native(e,req);after_native=v.native(qe,req)
        assert not np.any(before_native['query_expected']&~after_native['query_expected'])
        rest_expansion+=int((after_native['query_expected']&~before_native['query_expected']).sum())
        for key in after_native:v.eq(after_native[key],z[key])
        assert row['native_unknown_angles']==int(z['query_angle_unknown'].sum())
        assert row['native_unknown_palms']==int(z['query_palm_unknown'].sum())
        assert row['native_unknown_arms']==int(z['query_arm_unknown'].sum())
        v.eq(row['position_support_fraction'],z['position_count'][tuple(z['path'].T)].mean()/(3*len(req)))
        v.eq(row['direction_support_fraction'],z['direction_count'][tuple(z['path'].T)].mean()/(6*len(req)))
        assert row['path_changed']==(not np.array_equal(z['path'],z['base_path']))
        assert row['old_path_length']==len(z['base_path']) and row['new_path_length']==len(z['path']) and row['whole_action_correct'] is None
        h=0 if row['side']=='L' else 1;sl=slice(17,21) if mode=='wholearm' else slice(0,17)
        if mode!='shortgap':assert not z['within_range'][:,h,sl].any()
        else:
            for j in range(len(re['core'])):
                matched=z['path'][z['path'][:,1]==j,0]
                if np.isin(matched,removed).any():assert z['unknown'][j,h,sl][z['expected'][j,h,sl]].all()
        counts[mode]+=1
        if sum(counts.values())%27==0:print('REMAINING VERIFIED',dict(counts),flush=True)
    assert counts=={'wholehand':45,'shortgap':45,'wholearm':45}
    d.check(f['pins']);d.check(f['protected']);d.check(proof['preserved']);d.check(proof['outputs'])
    d.write(U/'VERIFY.json',dict(status='PASS',at=d.now(),total=2808,
        original_passed={'known':297,'cross':2376},recovery_missing=dict(counts),templates_pinned=216,known_timing_checks=297,
        fixtures_reused=fixed,protected=len(f['protected']),person_exclusion=True,expected_fixed=True,query_native_gaps_retained=True,
        missing_native_expected_expanded_cells=rest_expansion,
        verifier='Codex phase_design_review subagent independent formulas and DP. Original2673 case verification reused with immutable saved arithmetic/ordering failure proof; remaining135 checked by separate precision-corrected verifier.',
        recovery_proof=str(U/'VERIFY_RECOVERY_PROOF.json'),original_failure_preserved=str(U/'VERIFY_FIRST_ERROR.txt'),
        unchanged_tolerance={'rtol':2e-11,'atol':2e-11},experimental_output_change=False,
        limitations=['Crossperson phases have no true correspondence labels; metrics are proxies.',
            'Undefined motion contributes zero added cost and zero evidence, not success.',
            'Missing controls remove positive hand-rest witnesses, so native expected may expand; new unknown is not a wrong-sign label.',
            'Core interval deletion does not certify linguistic essentiality. Whole-action correctness remains unset.']))
    print('ALL2808 VERIFIED;2673 REUSED+135 RECOVERY',flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        p=U/'VERIFY_RECOVERY_FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
