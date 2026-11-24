"""Research local time-sensitivity flag, not replacement alignment or correction.

Same offset for all 3 moments and all supported channels of one hand/window.
Keep original values/flags. No new warning and no lost evidence may be used to
make an offset acceptable. This does not certify an action as correct.
"""
import sys,json,hashlib,importlib.util,traceback
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;U=ROOT/'offset_candidate';O=ROOT.parent
P=O/'word100_point_sequence_split_20261022';B=O/'word100_coherent_reference_sequence_20261022'
I=O/'word100_cross_person_motion_support_20261022'/'inputs'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as h:json.dump(x,h,ensure_ascii=False,indent=2,allow_nan=False)
def mod(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def shifted(C,c,path,n,rest,delta):
    old=C.mapped(c,path,n,rest);raw=old['index']+delta
    inside=(raw>=0)&(raw<=len(c['valid'])-1);ix=np.clip(raw,0,len(c['valid'])-1)
    out=C.resample(c,ix);out['valid'] &= inside[:,None,None]
    # Use the full footprint, including old DTW ranges, to avoid hiding a gap.
    first=np.minimum(old['first'],np.floor(ix).astype(int));last=np.maximum(old['last'],np.ceil(ix).astype(int))
    return out|dict(first=first,last=last,index=ix,source_valid=c['valid'],source_rest=rest)

def select(oldbad,oldvalid,options_bad,options_valid):
    remaining=oldbad.copy();chosen=np.zeros(oldvalid.shape[:2],np.int8)
    resolved=np.zeros(oldvalid.shape[:2],int)
    for k,delta in enumerate((-1,1)):
        # A simultaneous hand-window hypothesis: never mix best offsets by angle.
        evidence_preserved=(~oldvalid|options_valid[k]).all(-1)
        no_new=(~(options_bad[k]&~oldbad&oldvalid[...,None])).all(axis=(-1,-2))
        removed=(oldbad&~options_bad[k]&oldvalid[...,None]).sum(axis=(-1,-2))
        take=evidence_preserved&no_new&(removed>resolved)
        remaining[take]=options_bad[k][take]&oldvalid[take,:,None]
        chosen[take]=delta;resolved[take]=removed[take]
    return remaining,chosen

def selftest():
    v=np.ones((3,2,21),bool);b=np.zeros((3,2,21,5),bool);b[1,0,0,0]=True
    obs=np.stack((b.copy(),b.copy()));ovs=np.stack((v.copy(),v.copy()))
    obs[0,1,0,0,0]=False
    out,ch=select(b,v,obs,ovs);assert not out[1,0].any() and ch[1,0]==-1
    obs[0,1,0,1,0]=True
    out,ch=select(b,v,obs,ovs);assert np.array_equal(out,b) and not ch.any()
    obs[0,1,0,1,0]=False;ovs[0,1,0,1]=False
    out,ch=select(b,v,obs,ovs);assert np.array_equal(out,b) and not ch.any()
    b[:]=False;obs[:]=False;out,ch=select(b,v,obs,ovs);assert not out.any() and not ch.any()
    return dict(shared_hand_offset=True,new_warning_veto=True,missing_evidence_veto=True,unchanged_normal=True)

def run():
    assert read(ROOT/'VERIFIED.json')['status']=='PASS' and read(ROOT/'CAUSAL_VERIFIED.json')['status']=='PASS'
    tests=selftest();f=read(ROOT/'FROZEN.json');assert not U.exists();U.mkdir();(U/'cases').mkdir()
    write(U/'SELFTEST.json',tests)
    pins=dict(f['pins']);pins[str(Path(__file__).resolve())]=sha(Path(__file__))
    write(U/'FROZEN.json',dict(scope='Same developed27TRAIN; 2808 saved comparisons. No new detector/TCN/DTW path or training.',
        protocol='Fixed offsets +-0.5 query sample. All three moments and every supported angle/coordinate for one hand move together. Whole source-span and rest evidence required; all original sequence-supported channels must survive. No new component warning allowed. Choose more removed warnings, ties prefer0 thennegative. Keep original point and raw sequence flags; label resolved flags time-sensitive, never normal. Different local windows may choose different offsets; NOT a coherent new global DTW path.',
        evaluation='Known normal135, cross normal1080, short deletion and long deletion, missing135. Report original and remaining flags and signal loss, all cases kept. Compare speed vs sameoriginal on common support. No use of known source correspondence by candidate beyond generating public synthetic input. No postresult threshold modification.',
        pins=pins,protected=f['protected']))
    for p,h in pins.items():assert sha(p)==h,p
    write(U/'RUNNING.json',{'stage':'public replay'})
    C=mod('offset_coherent',B/'candidate.py');S=mod('offset_split',P/'split_support.py')
    es={r['clip_id']:load(I/(r['clip_id']+'.npz')) for r in f['rows']}
    cs={k:C.channels(e) for k,e in es.items()};old=read(P/'RESULT.json');results=[]
    for folder in ('known','cross','missing'):
        for m in old[folder]:
            z=load(P/folder/(m['name']+'.npz'));e=es[m['query']];reference=es[m['reference']]
            if folder=='missing':
                qe={k:v.copy() for k,v in e.items()};h=('L','R').index(m['side']);ids=np.array(m['removed'],int)
                if m['kind']=='wholearm':qe['arm_valid'][ids,h]=False;qe['arm_xy'][ids,h]=np.nan
                else:
                    for k in ('angle_low','angle_high','centre','radius'):qe[k][ids,h]=np.nan
                    for k in ('angle_valid','palm_valid','source_rest','observed'):qe[k][ids,h]=False
                    for k in ('missing','unconfirmed'):qe[k][ids,h]=True
                qc=C.channels(qe);rest=qe['source_rest']
            else:
                ix=z['source_index'];qc=C.resample(cs[m['query']],ix)
                rest=e['source_rest'][np.floor(ix).astype(int)]&e['source_rest'][np.ceil(ix).astype(int)]
            n=len(reference['core']);identity=np.column_stack((np.arange(n),np.arange(n)))
            anchor=C.descriptor(C.mapped(cs[m['reference']],identity,n,reference['source_rest']),z['expected'])
            cal=load(P/'calibration'/(m['query']+'__'+m['reference']+'.npz'))
            ob=[];ov=[]
            for delta in (-.5,.5):
                desc=C.descriptor(shifted(C,qc,z['path'],n,rest,delta),z['expected'])
                valid=z['expected']&desc['valid']&anchor['valid']&cal['support']
                bad=valid[...,None]&(S.gaps(desc,anchor)-cal['component_bound']>1e-6)
                ob.append(bad);ov.append(valid)
            ob=np.stack(ob);ov=np.stack(ov)
            remaining,chosen=select(z['component_bad'],z['sequence_comparable'],ob,ov)
            sensitive=z['component_bad']&~remaining
            assert not (remaining&~z['component_bad']).any()
            name=folder+'__'+m['name']
            np.savez_compressed(U/'cases'/(name+'.npz'),remaining=remaining,timing_sensitive=sensitive,
                chosen_half_frames=chosen,option_bad=ob,option_valid=ov,
                expected=z['expected'],point_comparable=z['point_comparable'],point_bad=z['point_bad'],
                sequence_comparable=z['sequence_comparable'],raw_sequence_bad=z['component_bad'])
            results.append({k:m[k] for k in ('name','query','reference','kind','side','removed') if k in m}|dict(folder=folder,
                original_flags=int(z['component_bad'].sum()),remaining_flags=int(remaining.sum()),
                sensitive_flags=int(sensitive.sum()),shifted_hand_windows=int(np.count_nonzero(chosen)),whole_action_correct=None))
        print('OFFSET',folder,len(old[folder]),flush=True)
    for p,h in pins.items():assert sha(p)==h,p
    for p,h in f['protected'].items():assert sha(p)==h,p
    write(U/'RESULT.json',dict(rows=results,whole_action_correct=None,model_updates=0,user_access=0))
    (U/'RUNNING.json').unlink()

if __name__=='__main__':
    try:run()
    except BaseException:
        if U.exists() and not (U/'FIRST_ERROR.txt').exists():(U/'FIRST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
