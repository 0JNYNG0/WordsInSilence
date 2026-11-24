import sys,json,hashlib,gzip,traceback,importlib.util,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import evidence_bridge as E
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
P=O/'word100_point_sequence_split_20261022'
I=O/'word100_cross_person_motion_support_20261022'/'inputs'
B=O/'word100_coherent_reference_sequence_20261022'
NORMAL={'identity','fast075','slow15','pace','pause'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def module(name,p):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def freeze():
    assert not (U/'FROZEN.json').exists()
    f=read(P/'FROZEN.json');assert read(P/'VERIFY.json')['status']=='PASS';check(f['protected'])
    paths=[P/'FROZEN.json',P/'RESULT.json',P/'VERIFY.json',P/'split_support.py',B/'candidate.py',
           U/'evidence_bridge.py',U/'run_bridge.py',U/'verify_bridge.py']
    paths+=list(I.glob('*.npz'))+list((P/'calibration').glob('*.npz'))
    for folder in ('known','cross','missing'):paths+=list((P/folder).glob('*.npz'))
    (U/'before').mkdir()
    for p in (R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json'):shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',dict(at=now(),authorization='User approved ongoing autonomous public-data recognition and feedback improvement; preserve original models, no user or mentor data.',
        scope='Research-only reusable evidence payload and known-time causal audit. No operating API/model edits. Same developed 27TRAIN clips,3words,9record IDs. 2808 saved comparisons reused, not new independent evaluations.',
        rows=f['rows'],protected=f['protected'],pins={str(p):sha(p) for p in paths},
        bridge='Keep expected and moment/sequence masks and differences exactly. Separate42 hand/channel streams, fixed reference unique-time duration denominator, explicit unknown intervals. No global pass/fail, no guessed zero score, no sequence warning converted to correction. Current image-backed point/temporal estimate/region/identity contract; actual saved support reused without promoting estimates.',
        causal='Known-time normal controls only135=27identity+108speed variants. 2x2 diagnostic: saved DTW vs oracle correspondence from source_index, synthetic resampled values vs original source at composed corresponding time. Same frozen expected/calibration>=5people, full-span evidence, 5components. Common support across4 conditions. Oracle inverse uses mean index among repeated source_index values. No threshold or saved score/model selection changes. Oracle information evaluation-only, never deployable.',
        acceptance='All2808 payloads; masks and old point flags exactly retained; no missing/region/inferred hand promoted; fixed-reference denominator unchanged by query absence; roundtrip timings and independent sums. Diagnostic may reveal no improvement or regression; does not constitute semantic correctness or new detection.',
        forbidden=['user/mentor access','199/200 heldout reuse','new detector/training/classification','DTW reoptimization','model/web edits','download','Git','external']))
    write(U/'SELFTEST.json',E.selftest());print('FROZEN',flush=True)

def mapped_at(C,c,index,rest):
    z=C.resample(c,index)
    return z|dict(first=np.floor(index).astype(int),last=np.ceil(index).astype(int),index=index,
                  source_valid=c['valid'],source_rest=rest)

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    assert not (U/'RUNNING.json').exists();write(U/'RUNNING.json',dict(at=now()))
    (U/'payloads').mkdir();(U/'causal').mkdir()
    old=read(P/'RESULT.json');es={r['clip_id']:load(I/(r['clip_id']+'.npz')) for r in f['rows']}
    C=module('bridge_coherent',B/'candidate.py');S=module('bridge_split',P/'split_support.py')
    rows=[];totals={};causal=[]
    for folder in ('known','cross','missing'):
        for m in old[folder]:
            z=load(P/folder/(m['name']+'.npz'));e=es[m['reference']]
            payload=E.build(m,z,e['timestamp_us']);name=folder+'__'+m['name']+'.json.gz'
            data=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
            with (U/'payloads'/name).open('xb') as h:h.write(gzip.compress(data,mtime=0))
            group=folder+('/normal' if m['kind'] in NORMAL else '/altered')
            a=totals.setdefault(group,dict(cases=0,expected=0,point=0,sequence=0,point_only=0,difference_without_sequence=0))
            a['cases']+=1;a['expected']+=int(z['expected'].sum());a['point']+=int(z['point_comparable'].sum())
            a['sequence']+=int(z['sequence_comparable'].sum());a['point_only']+=int((z['point_comparable']&~z['sequence_comparable']).sum())
            a['difference_without_sequence']+=int((z['point_bad']&~z['sequence_comparable']).sum())
            rows.append(dict(folder=folder,name=m['name'],kind=m['kind'],query=m['query'],reference=m['reference'],
                payload=name,payload_sha256=sha(U/'payloads'/name),counts=payload['evidence_counts']))
            if folder!='known' or m['kind'] not in NORMAL:continue
            # Diagnostic only: source_index is unavailable for a real unknown user.
            c=C.channels(e);ix=z['source_index'];qc=C.resample(c,ix)
            rest=e['source_rest'][np.floor(ix).astype(int)]&e['source_rest'][np.ceil(ix).astype(int)]
            actual=C.mapped(qc,z['path'],len(e['core']),rest)
            unique,inv=np.unique(ix,return_inverse=True)
            mean=np.bincount(inv,weights=np.arange(len(ix)))/np.bincount(inv)
            oracle_index=np.interp(np.arange(len(e['core'])),unique,mean)
            oracle=mapped_at(C,qc,oracle_index,rest)
            original_indices=np.arange(len(e['core']))
            anchor=C.descriptor(C.mapped(c,np.column_stack((original_indices,original_indices)),len(e['core']),e['source_rest']),z['expected'])
            ds=[]
            for alignment in (actual,oracle):
                ds.append(C.descriptor(alignment,z['expected']))
                source_time=np.interp(alignment['index'],np.arange(len(ix)),ix)
                q_original=C.resample(c,source_time)
                corrected={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in alignment.items()}
                corrected.update(lo=q_original['lo'],hi=q_original['hi'],valid=q_original['valid']&alignment['valid'])
                ds.append(C.descriptor(corrected,z['expected']))
            cal=load(P/'calibration'/(m['query']+'__'+m['reference']+'.npz'))
            valids=np.stack([z['expected']&d['valid']&anchor['valid']&cal['support'] for d in ds])
            gaps=np.stack([S.gaps(d,anchor) for d in ds])
            bad=valids[...,None]&(gaps-cal['component_bound'][None]>S.EPS)
            assert np.array_equal(valids[0],z['sequence_comparable'])
            assert np.array_equal(bad[0],z['component_bad'])
            common=valids.all(0);counts=[int((b&common[...,None]).sum()) for b in bad]
            saved=dict(common=common,valids=valids,bad=bad,gaps=gaps,oracle_index=oracle_index,
                actual_index=actual['index'],expected=z['expected'])
            np.savez_compressed(U/'causal'/(m['name']+'.npz'),**saved)
            causal.append(dict(name=m['name'],kind=m['kind'],counts=counts,common_cells=int(common.sum()),
                original_cells=int(valids[0].sum()),excluded_from_common=int((valids[0]&~common).sum()),
                order=['original','source_values_only','correspondence_only','both_oracles'],
                sampling_resolved=int((bad[0]&~bad[1]&common[...,None]).sum()),
                correspondence_resolved=int((bad[0]&~bad[2]&common[...,None]).sum()),
                sampling_added=int((~bad[0]&bad[1]&common[...,None]).sum()),
                correspondence_added=int((~bad[0]&bad[2]&common[...,None]).sum())))
        print('DONE',folder,len(old[folder]),flush=True)
    check(f['pins']);check(f['protected']);assert len(rows)==2808 and len(causal)==135
    write(U/'RESULT.json',dict(at=now(),status='BUILT_PENDING_SEPARATE_VERIFICATION',rows=rows,totals=totals,causal=causal,
        larger_goal_complete=False,training=0,new_inference=0,user_mentor_access=0,operating_edits=0))
    (U/'RUNNING.json').unlink();print('COMPLETE',flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if not (U/'FIRST_ERROR.txt').exists():(U/'FIRST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
