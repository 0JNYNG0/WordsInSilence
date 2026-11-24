"""Parent verification of separately authored descriptors and materialization."""
from pathlib import Path
from collections import Counter
import json, hashlib
import numpy as np
import prepare_scenarios as scenarios

U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
rows=read(U/'MANIFEST.json'); spec=read(U/'SCENARIOS.json')
counts=Counter(); names=set(); sources={}
for case in spec['cases']:
    assert case['case_id'] not in names; names.add(case['case_id'])
    idx=int(case['model_input']['bundle_input_ref'].split('_')[1]);row=rows[idx]
    assert row['source_group']==case['source_group']==case['evaluator_only']['clip_id']
    if idx not in sources:
        with np.load(row['arrays'],allow_pickle=False) as z:sources[idx]={k:z[k] for k in z.files}
    raw=sources[idx]; valid=raw['baseline_roi_usable'].copy()
    xy=raw['baseline_roi_xy_px'].copy();obs=raw['baseline_roi_observed'].copy()
    conf=raw['baseline_roi_conf'].copy(); ca=np.isfinite(conf)&valid
    score=raw['baseline_roi_hand_score'].copy(); sa=np.isfinite(score)
    tr=case['model_input']['coordinate_transform']
    if tr['kind']=='single_hand_missing':
        side=tr['side_index'];start,stop=tr['frame_range_half_open'];points=slice(33+21*side,54+21*side)
        eligible=np.flatnonzero(valid[:,points].all(1)&np.isfinite(xy[:,points]).all((1,2)))
        runs=np.split(eligible,np.where(np.diff(eligible)!=1)[0]+1)
        chosen=sorted([r for r in runs if len(r)],key=lambda r:(-len(r),int(r[0])))[0]
        assert len(chosen)>=11
        center=(int(chosen[0])+int(chosen[-1]))//2
        hole=stop-start;assert hole in [1,3,7]
        assert [start,stop]==[center-hole//2,center+hole//2+1]
        assert start>=chosen[0]+2 and stop<=chosen[-1]-1
        valid[start:stop,points]=False;ca[start:stop,points]=False
        sa[start:stop,side]=False
        counts['dropout']+=1
    else:assert tr['kind']=='none';counts['intact']+=1
    obs &=valid;xy[~valid]=0;conf[~ca]=0
    for side in [0,1]:sa[:,side]&=valid[:,33+side*21:54+side*21].any(1)
    score[~sa]=0
    expected=dict(baseline_roi_xy_px=xy,baseline_roi_usable=valid,baseline_roi_observed=obs,
        baseline_roi_conf=conf,baseline_roi_conf_available=ca,
        baseline_roi_hand_score=score,baseline_roi_hand_score_available=sa)
    for key in ['frame_id','timestamp_us','roi_box','width','height']:expected[key]=raw[key]
    actual=scenarios.materialize_model_input(row,case)
    assert set(actual)==set(expected)
    for key in expected:
        assert np.isfinite(actual[key]).all(),(case['case_id'],key)
        assert np.array_equal(actual[key],expected[key]),(case['case_id'],key)
    counts['cases']+=1
assert counts=={'intact':61,'dropout':333,'cases':394}
assert len(spec['hand_decisions'])==122
assert sum(not h['eligible'] for h in spec['hand_decisions'])==11
result={'status':'PASS','reviewer':'parent separate expected-array implementation, not scenario self-check',
        'counts':dict(counts),'source_clips':len(sources),'unsupported_hands':11,
        'selection_recomputed':True,'all_values_finite':True,'exact_mask_and_confidence_availability':True,
        'all_nontarget_coordinates_times_preserved':True,'source_groups_preserved':True,
        'new_inference':0,'new_training':0,
        'pins':{f:hashlib.sha256((U/f).read_bytes()).hexdigest() for f in ['SCENARIOS.json','prepare_scenarios.py','load_inputs.py']},
        'limit':'Proxy coordinate reconstruction; not independent anatomical truth or physical occlusion accuracy'}
with (U/'SCENARIO_INDEPENDENT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
