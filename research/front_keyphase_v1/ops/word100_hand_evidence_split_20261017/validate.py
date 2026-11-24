import json,hashlib,copy
from collections import Counter
from pathlib import Path
import numpy as np
from separate import separate_evidence,SeparatedHandDiagnostics,LOCATION
U=Path(__file__).resolve().parent;B=U.parent/'word100_hand_region_uncertainty_20261017'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    with (U/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
s=read(U/'SCOPE.json')
for p,h in {**s['dependencies'],**s['source_pins'],**s['protected_pins']}.items():assert sha(p)==h,p

# Metamorphic contract: unrelated shape uncertainty cannot alter location.
checks=[];perturbations=0
for state in LOCATION:
    a={'side':0,'state':state,'evidence':{'example':1.25}}
    values=[]
    for region_state in ('inside_missing','overlap_likely','boundary_outside_likely','boundary_uncertain','region_uncertain'):
        for source in ('unavailable','estimated_envelope'):
            for u in (0.,10.,100000.):
                b={'side':0,'state':region_state,'baseline_state':state,'region_status':source,'evidence':{'uncertainty_pixels':u,'region_age_seconds':.4}}
                before=copy.deepcopy((a,b));out=separate_evidence(a,b)
                assert (a,b)==before
                values.append(out['position']);perturbations+=1
                assert out['coordinate_repair'] is False and out['eligible_for_automatic_exclusion'] is False
                assert out['overlap']['occlusion_cause_confirmed'] is False
                assert out['shape']['estimated_region_is_observed_shape'] is False
                assert out['shape']['state']==('usable_observed_points' if state=='observed' else 'partial_observation' if state=='partial_observation' else 'unobserved')
                if source=='unavailable':assert out['overlap']['area_evidence']!='area_supported_suspicion'
    assert all(x==values[0] for x in values)
checks.extend(['position invariant to shape uncertainty210 perturbations','input objects unchanged','no aggregate state/automatic exclusion','unobserved shape never promoted','region fallback not called area-supported'])
a={'side':0,'state':'inside_missing','evidence':{'nested':{'x':[1,2]}}}
b={'side':0,'state':'inside_missing','baseline_state':'inside_missing','region_status':'unavailable','evidence':{}}
out=separate_evidence(a,b);out['position']['evidence']['nested']['x'][0]=99
assert a['evidence']['nested']['x'][0]==1;checks.append('consumer mutation cannot change source evidence')
for wrong in ({**b,'side':1},{**b,'baseline_state':'unknown'}):
    try:separate_evidence(a,wrong)
    except ValueError:pass
    else:raise AssertionError('mismatched records accepted')
checks.append('wrong side or mismatched observation rejected')

data=read(B/'PREDICTIONS.json');frames_by_id={r['clip_id']:r for r in data['records']}
records=[];counts={'position':Counter(),'shape':Counter(),'overlap':Counter(),'proximity_hint':Counter()}
for r in data['records']:
    frames=[]
    for f in r['frames']:
        hands=[separate_evidence(a,b) for a,b in zip(f['baseline'],f['hands'])]
        for h in hands:
            counts['position'][h['position']['state']]+=1
            counts['shape'][h['shape']['state']]+=1
            counts['overlap'][h['overlap']['area_evidence']]+=1
            counts['proximity_hint'][str(h['overlap']['proximity_hint'])]+=1
        frames.append({'frame':f['frame'],'timestamp_us':f['timestamp_us'],'core':f['core'],'hands':hands})
    records.append({'clip_id':r['clip_id'],'group':r['group'],'label':r['label'],'frames':frames})
lookup={r['clip_id']:r for r in records};stream_checked=0
for cid in s['stream_validation_clips']:
    r=next(x for x in s['records'] if x['clip_id']==cid)
    raw=np.load(r['paths']['raw']);obs=np.load(r['paths']['candidate']);ass=np.load(r['paths']['assignment'])['sides']
    adapter=SeparatedHandDiagnostics()
    for i,t in enumerate(obs['timestamp_us']):
        p,m,c=obs['xy_px'][i].copy(),obs['usable'][i].copy(),raw['pose_conf'][i].copy()
        before=[x.copy() for x in (p,m,c)];sides=ass[i,:int(raw['raw_count'][i])]
        out=adapter.step(p,m,c,int(t),int(raw['width'][i]),int(raw['height'][i]),
                         assigned=(bool(0 in sides),bool(1 in sides)),unassigned_raw=bool(any(x<0 for x in sides)))
        assert out==lookup[cid]['frames'][i]['hands'],(cid,i)
        assert all(np.array_equal(x,y,equal_nan=True) for x,y in zip((p,m,c),before))
        stream_checked+=2
checks.append('live causal adapter matches frozen public replay through complete clips')
labels=read(B/'VISUAL_LABELS.json')['labels']
comparisons=[]
for x in labels:
    f=frames_by_id[x['clip_id']]['frames'][x['frame']];side=x['side']
    comparisons.append({**x,'baseline':f['baseline'][side], 'combined_candidate':f['hands'][side],
                         'separated':lookup[x['clip_id']]['frames'][x['frame']]['hands'][side]})
out={'scope_sha256':sha(U/'SCOPE.json'),'source_predictions_sha256':sha(B/'PREDICTIONS.json'),
     'records':records,'counts':counts,'visual_comparisons':comparisons,
     'new_training':0,'new_detection':0,'classifier_inference':0,'user_recording_reads':0}
save('OUTPUT.json',out)
save('CONTRACT_TESTS.json',{'status':'PASS','checks':checks,'metamorphic_cases':perturbations,'public_stream_hand_states':stream_checked,'stream_clips':s['stream_validation_clips']})
for p,h in {**s['dependencies'],**s['source_pins'],**s['protected_pins']}.items():assert sha(p)==h,p
print(json.dumps({'status':'PASS','metamorphic_cases':perturbations,'stream_hand_states':stream_checked,'counts':counts},ensure_ascii=False,indent=2))
