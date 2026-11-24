"""Separate saved-pair arithmetic and input checks. Does not rebuild the bank."""
from pathlib import Path
from collections import Counter
from statistics import median
import json,hashlib,math,sys,copy
import numpy as np
U=Path(__file__).resolve().parent;B=U.parents[1]
sys.path.insert(0,str(B/'src'))
from wis_lab.service import word100_analysis as A,word100_multiref as M
from wis_lab.core.observation import Observation
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(name)
def same(a,b):
    if a is None or b is None:return a is b
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)
def oracle(pairs,key):
    hands={}
    for side in ['L','R']:
        v=[]
        for p in pairs:
            r=p['result']
            if r['status']!='OK':continue
            d=next(s for m in r['metrics'] if m['key']==key for s in m['sides'] if s['side']==side)
            if d['value'] is not None and math.isfinite(d['value']) and d['support_fraction']>=.5 and d.get('query_distinct_frames',0)>=8 and d.get('reference_distinct_frames',0)>=8:v.append(d['value'])
        hands[side]=(median(v) if len(v)>=5 else None,len(v))
    vals=[h[0] for h in hands.values()]
    return (sum(vals)/2 if all(v is not None for v in vals) else None),hands
bank=read(U/'BASELINE.json');inputs=read(U/'INPUTS.json');pairs=read(U/'TRAIN_PAIRS.json');dev=read(U/'DEVELOPMENT.json')
check('144 unique inputs:108 training and36 development',len(inputs)==len({r['clip_id'] for r in inputs})==144 and Counter(r['role'] for r in inputs)=={'train_candidate':108,'development_only':36})
check('input roles agree with manifest, no held-out/user clips',all((r['role']=='train_candidate' and r['signer_id'] in range(1,10)) or (r['role']=='development_only' and r['signer_id'] in [11,12,13]) for r in inputs))
check('144 source observations and cached features match hashes',all(sha(r['observation'])==r['observation_sha256'] and sha(r['sample_path'])==r['sample_sha256'] for r in inputs))
check('manifest and implementation hashes fixed',sha(B/'ops/word100_training_20261013/MANIFEST.json')==bank['manifest_sha256'] and all(sha(p)==h for p,h in bank['code_sha256'].items()))
for r in inputs:
    loaded=M.sample(r['sample_path'],r['sample_sha256'])
    fresh=A._sample(Observation.load(r['observation']))
    assert all(np.array_equal(loaded[k],fresh[k],equal_nan=True) for k in ('features','present'))
check('all144 cached numeric feature arrays/masks exactly equal source transforms; no pickle load',True)
check('864 unique ordered training comparisons, no self inclusion',len(pairs)==len({(p['query_clip'],p['reference_clip']) for p in pairs})==864 and all(p['query_signer']!=p['reference_signer'] and p['query_signer'] in range(1,10) and p['reference_signer'] in range(1,10) for p in pairs))
old=read(B/'ops/word100_robustness_20261013/dtw96/PAIRS.json');index={(p['query_clip'],p['reference_clip']):p['result'] for p in pairs}
check('96 archived pairs reused exactly;768 new comparisons',len(old)==96 and all(index[p['query_clip'],p['reference_clip']]==p['result'] for p in old))
coverage=[]
for word,item in bank['words'].items():
    assert len(item['references'])==len(item['leave_one_out'])==9
    assert all(r['role']=='train_candidate' and r['signer_id'] in range(1,10) for r in item['references'])
    for key in ['handshape','location','movement']:
        values=[];sidevalues={'L':[],'R':[]}
        for loo in item['leave_one_out']:
            pp=[p for p in pairs if p['query_clip']==loo['clip_id']]
            assert len(pp)==8 and all(p['word_id']==word for p in pp)
            value,hands=oracle(pp,key);m=next(m for m in loo['metrics'] if m['key']==key)
            assert same(value,m['value'])
            if value is not None:values.append(value)
            for s,(v,n) in hands.items():
                d=next(d for d in m['sides'] if d['side']==s)
                assert same(v,d['value']) and n==d['valid_references']
                if v is not None:sidevalues[s].append(v)
        b=item['baseline'][key]
        assert b['n']==len(values) and same(b['median'],median(values) if values else None)
        for side,sv in sidevalues.items():
            assert b['sides'][side]['n']==len(sv) and same(b['sides'][side]['median'],median(sv) if sv else None)
        coverage.append(dict(word=word,axis=key,n=b['n'],median=b['median']))
check('all108 performer scores and36 baseline scales independently recomputed',True)
available=0
for row in dev:
    assert row['signer_id'] in [11,12,13] and len(row['pairs'])==9
    assert all(p['reference_signer'] in range(1,10) for p in row['pairs'])
    for m in row['metrics']:
        v,hs=oracle(row['pairs'],m['key']);b=bank['words'][row['word_id']]['baseline'][m['key']]
        expected=v/b['median'] if v is not None and b['n']>=5 and b['median']>1e-6 else None
        assert same(v,m['value']) and same(expected,m['ratio'])
        available+=m['ratio'] is not None
        for d in m['sides']:
            s=d['side'];sv=b['sides'][s];hv=hs[s][0]
            expected=hv/sv['median'] if hv is not None and sv['n']>=5 and sv['median']>1e-6 else None
            assert same(expected,d['ratio'])
check('36 development results recomputed independently without calibrating',len(dev)==36)
# Numeric-path regression against the actual pre-edit implementation, no re-running old experiment.
import types
legacy=types.ModuleType('wis_lab.service._pre_multiref');legacy.__package__='wis_lab.service'
exec(compile((U/'before/word100_analysis.py').read_text('utf-8'),str(U/'before/word100_analysis.py'),'exec'),legacy.__dict__)
r1=Observation.load(inputs[0]['observation']);r2=Observation.load(inputs[1]['observation'])
check('refactor leaves original DTW numeric result exactly equal',legacy.compare(r1,r2)==A.compare(r1,r2))
check('identical observations have zero measured differences',all(m['value']==0 for m in A.compare(r1,r1)['metrics']))
sample=A._sample(r1);samepairs=[dict(result=A.compare_samples(sample,sample)) for _ in range(9)]
for key in M.AXES:
    assert next(m for m in M.aggregate(samepairs) if m['key']==key)['value']==0
missing=copy.deepcopy(samepairs)
for p in missing:
    for m in p['result']['metrics']:m['sides'][0]['value']=None
check('missing hand never turns into zero or one-hand combined score',all(m['value'] is None for m in M.aggregate(missing)))
check('fewer than5 valid references is withheld',all(m['value'] is None for m in M.aggregate(samepairs[:4])))
zero={key:dict(n=9,median=0,sides={s:dict(n=9,median=0) for s in ['L','R']}) for key in M.AXES}
check('near-zero baseline withheld, not divided',all(m['ratio'] is None for m in M.relative(M.aggregate(samepairs),zero)))
# Alter only wrist-position features after normalization: progressively farther -> larger measured location.
vals=[]
for offset in [0,.1,.2]:
    q=copy.deepcopy(sample)
    for side in ['L','R']:q['features'][:,A.OFFSETS['handpos_'+side]]+=offset
    vals.append(A.compare_samples(q,sample)['metrics'][1]['value'])
check('controlled larger positional discrepancy produces larger deviation',vals[0]<vals[1]<vals[2])
check('original recordings and frozen model unchanged',all(sha(p)==h for p,h in read(U/'ORIGINAL_GUARD.json').items()))
out=dict(status='PASS',checks=checks,baseline_coverage=coverage,development_available_axes=available,
    development_axes=108,meaning='Measurement and integration verification, not sign correctness accuracy',
    implementation='Codex',verification='Codex separate saved-data arithmetic; not independent human/expert review',
    execution_clarification='FROZEN_PLAN864 pairs means total;96 equivalent archived pairs reused and768 computed, no criterion change',
    baseline_sha256=sha(U/'BASELINE.json'))
(U/'INDEPENDENT_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(dict(status='PASS',checks=len(checks),development_available_axes=available,baseline_values=[c['n'] for c in coverage])))
