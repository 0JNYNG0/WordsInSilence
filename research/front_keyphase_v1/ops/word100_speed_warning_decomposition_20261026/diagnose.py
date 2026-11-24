"""Separate normal-speed resampling from alignment perturbation. No new DTW or model."""
from pathlib import Path
import sys,json,hashlib,datetime,importlib.util,traceback
from collections import Counter,defaultdict
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_point_sequence_split_20261022';B=O/'word100_coherent_reference_sequence_20261022';I=O/'word100_cross_person_motion_support_20261022/inputs'
NORMAL={'fast075','slow15','pace','pause'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=module('speed_reference_descriptor',B/'candidate.py');S=module('speed_component_split',P/'split_support.py')
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def inverse(ix,targets):
    """Inverse known time warp. A pause has an interval, not an arbitrary instant."""
    ix=np.asarray(ix,float);targets=np.asarray(targets,float)
    assert len(ix)>1 and np.isfinite(ix).all() and (np.diff(ix)>=0).all()
    assert targets.min()>=ix[0] and targets.max()<=ix[-1]
    low=[];high=[]
    for t in targets:
        l=int(np.searchsorted(ix,t,'left'));r=int(np.searchsorted(ix,t,'right'))
        if r>l:a,b=float(l),float(r-1)
        else:
            assert 0<l<len(ix)
            a=b=(l-1)+(t-ix[l-1])/(ix[l]-ix[l-1])
        low.append(a);high.append(b)
    lo=np.asarray(low);hi=np.asarray(high);return lo,hi,(lo+hi)/2
def mapped_oracle(c,rest,ix,baseline):
    _,_,mid=inverse(ix,baseline['index'])
    first,_,_=inverse(ix,baseline['first']);_,last,_=inverse(ix,baseline['last'])
    out=C.resample(c,mid)
    out.update(first=np.floor(first).astype(int),last=np.ceil(last).astype(int),index=mid,source_valid=c['valid'],source_rest=rest)
    return out
def descriptor_gap(q,a,cal,expected):
    ok=expected&q['valid']&a['valid']&cal['support'];gap=S.gaps(q,a)
    bad=ok[...,None]&(gap-cal['component_bound']>1e-6)
    return ok,bad
def selftest():
    lo,hi,mid=inverse([0,1,1,1,3],[0,.5,1,2,3])
    assert np.allclose(lo,[0,.5,1,3.5,4]) and np.allclose(hi,[0,.5,3,3.5,4]) and mid[2]==2
    assert np.allclose(np.interp(mid,np.arange(5),[0,1,1,1,3]),[0,.5,1,2,3])
    lo,hi,mid=inverse([0,2,4],[1,2,3]);assert np.allclose(mid,[.5,1,1.5])
    rejected=False
    try:inverse([0,1],[2])
    except AssertionError:rejected=True
    assert rejected
    # A known linear signal survives round-trip; a short peak lost by sampling does not.
    t=np.arange(5.);ix=np.array([0.,2.,4.]);_,_,m=inverse(ix,t)
    linear=np.interp(m,np.arange(3),np.interp(ix,t,2*t));assert np.allclose(linear,2*t)
    peak=np.array([0.,1.,0.,0.,0.]);roundtrip=np.interp(m,np.arange(3),np.interp(ix,t,peak));assert not np.allclose(roundtrip,peak)
    return dict(pause_interval_not_single_point=True,known_phase_inverse_exact=True,speed_inverse_exact=True,outside_domain_rejected=True,linear_signal_preserved=True,lost_short_peak_not_recoverable_by_timing=True)
def freeze():
    assert read(P/'VERIFY.json')['status']=='PASS'
    f=read(P/'FROZEN.json');r=read(P/'RESULT.json');paths=[P/'FROZEN.json',P/'RESULT.json',P/'VERIFY.json',B/'candidate.py',P/'split_support.py',Path(__file__)]
    paths+=list(I.glob('*.npz'))+list((P/'calibration').glob('*.npz'))
    selected={folder:[x for x in r[folder] if x['kind'] in NORMAL] for folder in ('known','cross')}
    for folder in selected:
        for row in selected[folder]:paths.append(P/folder/(row['name']+'.npz'))
        paths+=list((P/folder).glob('identity__*.npz'))
    check(f['protected'])
    write(U/'SELFTEST.json',dict(at=now(),status='PASS',checks=selftest()))
    write(U/'FROZEN.json',dict(at=now(),rows=f['rows'],roles=f['roles'],cases=selected,pins={str(p):sha(p) for p in paths},protected=f['protected'],
        authorization='Existing autonomous goal: investigate and reduce DTW feedback errors using public TRAIN only while separate wrist-input preparation runs; no model/production change.',
        protocol='Reuse972 saved normal speed cases (known108,cross864), identity paths, bounds and expected domains. A=saved actual DTW result. B=known time-inverse of baseline path, on time-resampled source; no DTW recompute. C=original source on original path (identity receipt). Compare all3 common-supported channels, retain fixed expected and all coverage losses. Separate new warnings resolved by known alignment, new warnings remaining after phase correction, and newly introduced oracle warnings. Known-time oracle is diagnostic only, not deployable.',
        controls='Do not change thresholds, peer exclusions, required/rest domains, normal coordinates, crossing, source or predicted paths. No difference flags fed back to classification or warning policy. Missing support never success. Forward time map inverse represents duplicate pause as interval; original and transformed source use same signals.',
        limitations='Three already-developed public words27clips, not all100 or independent people. Baseline cross-person path is not semantic ground truth; counterfactual measures invariance to speed relative to it. Residual may reflect interpolation/descriptor validity or peer bounds, not an actual user mistake. No actual speed derivative normalization is assumed.',
        training=0,user_data=0,production_changed=False))
    print('FROZEN', {k:len(v) for k,v in selected.items()},flush=True)
def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',dict(at=now()))
    rows={r['clip_id']:r for r in f['rows']};es={cid:load(I/(cid+'.npz')) for cid in rows};cs={cid:C.channels(e) for cid,e in es.items()}
    domains={cid:C.expected(e,f['roles'].get(rows[cid]['word_code'],{}).get('required_sides',['L','R'])) for cid,e in es.items()}
    anchors={}
    for cid,e in es.items():
        ii=np.arange(len(e['core']));anchors[cid]=C.descriptor(C.mapped(cs[cid],np.column_stack([ii,ii]),len(ii),e['source_rest']),domains[cid])
    (U/'cases').mkdir();results=[];pins={};aggregate=defaultdict(Counter)
    for folder,metas in f['cases'].items():
        for meta in metas:
            cid=meta['query'];rid=meta['reference'];z=load(P/folder/(meta['name']+'.npz'));identname='identity__'+cid+('__'+rid if folder=='cross' else '')
            old=load(P/folder/(identname+'.npz'));cal=load(P/'calibration'/(cid+'__'+rid+'.npz'));n=len(domains[rid]);ix=z['source_index']
            base=C.mapped(cs[cid],old['path'],n,es[cid]['source_rest']);baseqd=C.descriptor(base,domains[rid])
            ok0,bad0=descriptor_gap(baseqd,anchors[rid],cal,domains[rid]);assert np.array_equal(ok0,old['sequence_comparable']) and np.array_equal(bad0,old['component_bad'])
            cc=C.resample(cs[cid],ix);rest=es[cid]['source_rest'][np.floor(ix).astype(int)]&es[cid]['source_rest'][np.ceil(ix).astype(int)]
            actual=C.mapped(cc,z['path'],n,rest);aq=C.descriptor(actual,domains[rid]);oka,bada=descriptor_gap(aq,anchors[rid],cal,domains[rid])
            assert np.array_equal(oka,z['sequence_comparable']) and np.array_equal(bada,z['component_bad'])
            om=mapped_oracle(cc,rest,ix,base);oq=C.descriptor(om,domains[rid]);oko,bado=descriptor_gap(oq,anchors[rid],cal,domains[rid])
            common=ok0&oka&oko;newa=common[...,None]&bada&~bad0;newo=common[...,None]&bado&~bad0
            actual_phase=np.interp(actual['index'],np.arange(len(ix)),ix);oracle_phase=np.interp(om['index'],np.arange(len(ix)),ix)
            assert np.allclose(oracle_phase,base['index'],atol=1e-10,rtol=0)
            error=np.abs(actual_phase-base['index']);roundtrip=np.maximum(np.abs(oq['lo']-baseqd['lo']),np.abs(oq['hi']-baseqd['hi']))
            arrays=dict(expected=domains[rid],base_ok=ok0,actual_ok=oka,oracle_ok=oko,common=common,base_bad=bad0,actual_bad=bada,oracle_bad=bado,
                actual_new=newa,oracle_new=newo,removed_by_phase=newa&~newo,new_after_phase=newo&~newa,remaining=newa&newo,
                original_phase=base['index'],actual_phase=actual_phase,oracle_phase=oracle_phase,phase_error=error,roundtrip_difference=roundtrip,
                point_comparable=z['point_comparable'],point_bad=z['point_bad'],native_expected=z['native_expected'],native_unknown=z['native_unknown'])
            name=folder+'__'+meta['name'];p=U/'cases'/(name+'.npz');np.savez_compressed(p,**arrays);pins[str(p)]=sha(p)
            groups={}
            for g,sl in C.GROUPS.items():
                co=common[:,:,sl];gb={k:[int(arrays[k][:,:,sl,j].sum()) for j in range(5)] for k in ('actual_new','oracle_new','removed_by_phase','new_after_phase','remaining')}
                gb.update(expected=int(domains[rid][:,:,sl].sum()),common=int(co.sum()),actual_comparable=int(oka[:,:,sl].sum()),oracle_comparable=int(oko[:,:,sl].sum()),lost_oracle=int((oka[:,:,sl]&~oko[:,:,sl]).sum()),gained_oracle=int((~oka[:,:,sl]&oko[:,:,sl]).sum()))
                finite=roundtrip[:,:,sl][co];gb['roundtrip_exact_cells']=int((finite<=1e-10).all(-1).sum());gb['roundtrip_changed_cells']=int((finite>1e-10).any(-1).sum());groups[g]=gb
                agg=aggregate[folder+'__'+g]
                for k,val in gb.items():
                    if isinstance(val,list):
                        for j,num in enumerate(val):agg[k+'_'+str(j)]+=num
                    else:agg[k]+=val
            results.append(dict(name=name,folder=folder,kind=meta['kind'],query=cid,reference=rid,groups=groups,phase_error_mean=float(error.mean()),phase_error_max=float(error.max())))
        print('DONE',folder,len(metas),flush=True)
    check(f['pins']);check(f['protected']);write(U/'RESULT.json',dict(at=now(),status='DIAGNOSIS_COMPLETE_PENDING_RECOUNT',rows=results,aggregate={k:dict(v) for k,v in aggregate.items()},output_pins=pins,frozen_sha256=sha(U/'FROZEN.json'),new_dtw=0,new_training=0,new_detector=0,whole_action_verdict=None))
    (U/'RUNNING.json').unlink();print('COMPLETE',len(results),flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FAILURE_'+now().replace(':','_')+'.txt');p.write_text(traceback.format_exc(),'utf8');raise
