"""Separate saved-result arithmetic; no model or detector inference."""
import sys,json,gzip,hashlib
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent
P=O/'word100_point_sequence_split_20261022'
I=O/'word100_cross_person_motion_support_20261022'/'inputs'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rebuild(intervals,n,t):
    out=np.zeros(n,bool)
    for a in intervals:
        start=a['first_index'];stop=a['end_index_exclusive'];assert 0<=start<stop<=n
        assert not out[start:stop].any()
        left=t[start]-(t[start]-t[start-1])/2 if start else t[0]-(t[1]-t[0])/2
        right=(t[stop]+t[stop-1])/2 if stop<n else t[-1]+(t[-1]-t[-2])/2
        assert abs(left-a['reference_start_s'])<1e-9 and abs(right-a['reference_end_s'])<1e-9
        out[start:stop]=True
    return out

f=read(U/'FROZEN.json');r=read(U/'RESULT.json')
inputs={x['clip_id']:load(I/(x['clip_id']+'.npz')) for x in f['rows']}
assert len(r['rows'])==2808
metrics=dict(cases=0,channels=0,point_cells=0,sequence_cells=0,point_only=0,
             preserved_point_difference_without_sequence=0,unknown_cells=0,
             completely_unobserved_channels=0,complete_channels=0,partial_channels=0,not_required_channels=0)
normal={};wholehand_removed=0;crossing={}
for row in r['rows']:
    pp=U/'payloads'/row['payload'];assert sha(pp)==row['payload_sha256']
    with gzip.open(pp,'rt',encoding='utf-8') as z:out=json.load(z)
    z=load(P/row['folder']/(row['name']+'.npz'))
    e=z['expected'];p=z['point_comparable'];s=z['sequence_comparable'];bad=z['point_bad']
    t=inputs[row['reference']]['timestamp_us'].astype(float)/1e6
    widths=np.zeros(len(t));widths[0]=t[1]-t[0];widths[-1]=t[-1]-t[-2];widths[1:-1]=(t[2:]-t[:-2])/2
    assert out['whole_action_correct'] is None and out['automatic_correction'] is None
    assert out['sequence_flags_diagnostic_only'] is True
    assert out['evidence_counts']=={k:int(z[k].sum()) for k in out['evidence_counts']}
    assert len(out['channels'])==42
    for a in out['channels']:
        h=('L','R').index(a['side']);c=a['channel'];den=sum(widths[j] for j in range(len(t)) if e[j,h,c])
        num=sum(widths[j] for j in range(len(t)) if p[j,h,c]);seq=sum(widths[j] for j in range(len(t)) if s[j,h,c])
        assert abs(den-a['required_seconds'])<1e-8 and abs(num-a['point_supported_seconds'])<1e-8
        assert abs(seq-a['sequence_supported_seconds'])<1e-8
        if not den:assert a['point_coverage'] is None and a['status']=='NOT_REQUIRED';metrics['not_required_channels']+=1
        else:
            assert abs(num/den-a['point_coverage'])<1e-8
            if num==0:assert a['status']=='INSUFFICIENT';metrics['completely_unobserved_channels']+=1
            elif (e[:,h,c]&~p[:,h,c]).any():assert a['status']=='PARTIAL';metrics['partial_channels']+=1
            else:assert a['status']=='OBSERVED';metrics['complete_channels']+=1
        for key,target in [('point_unknown_intervals',e[:,h,c]&~p[:,h,c]),('sequence_unknown_intervals',e[:,h,c]&~s[:,h,c]),('point_difference_intervals',bad[:,h,c])]:
            assert np.array_equal(rebuild(a[key],len(t),t),target)
        for k,v in enumerate(a['mean_sequence_component_gap']):
            if seq==0:assert v is None
            else:
                vals=[z['component_distance'][j,h,c,k]*widths[j] for j in range(len(t)) if s[j,h,c]]
                assert abs(sum(vals)/seq-v)<1e-8
        assert a['automatic_correction'] is None
        metrics['channels']+=1
    metrics['cases']+=1;metrics['point_cells']+=int(p.sum());metrics['sequence_cells']+=int(s.sum())
    metrics['point_only']+=int((p&~s).sum());metrics['preserved_point_difference_without_sequence']+=int((bad&~s).sum())
    metrics['unknown_cells']+=int((e&~p).sum())
    if row['folder']=='cross' and row['kind'] in ('identity','fast075','slow15','pace','pause'):
        normal.setdefault('point',0);normal['point']+=int(p.sum())
        normal.setdefault('point_only',0);normal['point_only']+=int((p&~s).sum())
        normal.setdefault('point_difference_without_sequence',0);normal['point_difference_without_sequence']+=int((bad&~s).sum())
        for h in range(2):
            allangles=p[:,h,:15].all(1)
            # Counterfactual only: do not claim operating API uses this exact gate.
            wholehand_removed+=int((p[:,h,:15]&~allangles[:,None]).sum())
    if row['folder']=='missing':
        base=P/'cross'/('identity__'+row['query']+'__'+row['reference']+'.npz')
        assert np.array_equal(e,load(base)['expected'])
    if row['folder']=='known' and row['kind']=='identity' and row['query']=='NIA_SL_WORD0185_REAL03_F':
        ids=np.flatnonzero(np.isin(inputs[row['query']]['frame_id'],list(range(101,112))+[113]))
        crossing={g:dict(expected=int(e[ids,:,sl].sum()),point=int(p[ids,:,sl].sum()),bad=int(bad[ids,:,sl].sum()))
                  for g,sl in [('angle',slice(0,15)),('palm',slice(15,17)),('elbow',slice(17,19)),('wrist',slice(19,21))]}

causal={};ns=0
for q in r['causal']:
    z=load(U/'causal'/(q['name']+'.npz'));old=load(P/'known'/(q['name']+'.npz'))
    cal=load(P/'calibration'/(q['name'].split('__',1)[1]+'__'+q['name'].split('__',1)[1]+'.npz'))
    reconstructed=z['valids'][...,None]&(z['gaps']>cal['component_bound'][None]+1e-6)
    assert np.array_equal(z['bad'],reconstructed)
    assert np.array_equal(z['bad'][0],old['component_bad'])
    assert np.array_equal(z['common'],z['valids'].all(0))
    counts=[int((b&z['common'][...,None]).sum()) for b in z['bad']];assert counts==q['counts']
    a=causal.setdefault(q['kind'],dict(cases=0,counts=[0]*4,common=0,excluded=0,sampling_resolved=0,correspondence_resolved=0,sampling_added=0,correspondence_added=0))
    a['cases']+=1;a['counts']=[x+y for x,y in zip(a['counts'],counts)];a['common']+=q['common_cells'];a['excluded']+=q['excluded_from_common']
    for key,ix,added in [('sampling_resolved',1,False),('correspondence_resolved',2,False),('sampling_added',1,True),('correspondence_added',2,True)]:
        mask=(~z['bad'][0]&z['bad'][ix]) if added else (z['bad'][0]&~z['bad'][ix])
        n=int((mask&z['common'][...,None]).sum());assert n==q[key];a[key]+=n
    ns+=1
assert ns==135
for p,h in f['pins'].items():assert sha(p)==h,p
for p,h in f['protected'].items():assert sha(p)==h,p
out=dict(status='PASS',verification='Same Codex separate implementation and saved arithmetic; not an independent human ground-truth review.',
    metrics=metrics,normal_cross=normal,whole_hand_gate_counterfactual_removed_cells=wholehand_removed,
    crossing=crossing,causal=causal,protected_hashes=len(f['protected']),source_pins=len(f['pins']),
    limitations=['Oracle correspondence and original sampling are analysis-only.','Saved validity reused, not new real RGB recovery or semantic correctness proof.','No new detector/TCN inference/training.'])
with (U/'VERIFIED.json').open('x',encoding='utf-8') as h:json.dump(out,h,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
