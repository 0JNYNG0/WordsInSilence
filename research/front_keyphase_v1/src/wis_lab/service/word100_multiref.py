"""Frozen training-only DTW reference bank; descriptive deviation, never correctness.

No model weights are trained or changed. User observations only flow through the
frozen transform and are never included in the reference bank or its scale.
"""
from functools import lru_cache
from pathlib import Path
import json
import threading
import numpy as np
from ..cli import ROOT
from ..core.observation import Observation, sha256_file

UNIT=Path(ROOT)/'ops/word100_multiref_20261014'
AXES=('handshape','location','movement')
MIN_SUPPORT=.5
MIN_FRAMES=8
MIN_REFS=5
EPS=1e-6
SCHEMA='wis_training_relative_dtw_v1'
_LOCK=threading.Lock()

def aggregate(pairs):
    """Median across distinct references per hand, then equal mean of both hands.

    An absent hand is never treated as zero or silently removed from the mean.
    """
    result=[]
    for key in AXES:
        sides=[]
        for side in ('L','R'):
            values=[];supports=[]
            for pair in pairs:
                r=pair['result']
                if r['status']!='OK':continue
                m=next(m for m in r['metrics'] if m['key']==key)
                d=next(d for d in m['sides'] if d['side']==side)
                if (d['value'] is not None and np.isfinite(d['value']) and
                    d['support_fraction']>=MIN_SUPPORT and
                    d.get('query_distinct_frames',0)>=MIN_FRAMES and
                    d.get('reference_distinct_frames',0)>=MIN_FRAMES):
                    values.append(d['value']);supports.append(d['support_fraction'])
            sides.append(dict(side=side,valid_references=len(values),
                value=float(np.median(values)) if len(values)>=MIN_REFS else None,
                median_support=float(np.median(supports)) if supports else None))
        values=[d['value'] for d in sides]
        ready=all(v is not None for v in values)
        result.append(dict(key=key,label={'handshape':'손가락 굽힘','location':'손 위치','movement':'이동 궤적'}[key],
            unit='degrees' if key=='handshape' else 'shoulder_width_percent',sides=sides,
            value=float(np.mean(values)) if ready else None,status='OK' if ready else 'INSUFFICIENT'))
    return result

def distribution(values):
    v=[float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(n=len(v),median=float(np.median(v)),q25=float(np.quantile(v,.25)),
                q75=float(np.quantile(v,.75)),min=min(v),max=max(v)) if v else dict(n=0,median=None)

def relative(metrics,baseline):
    out=[]
    for metric in metrics:
        m=dict(metric);b=baseline[m['key']];scale=b.get('median')
        valid=m['value'] is not None and b['n']>=MIN_REFS and scale is not None and scale>EPS
        m.update(baseline=b,ratio=float(m['value']/scale) if valid else None,
                 status='OK' if valid else 'INSUFFICIENT')
        m['sides']=[dict(d) for d in m['sides']]
        for d in m['sides']:
            sb=b['sides'][d['side']];sv=sb.get('median')
            d['baseline_median']=sv
            d['ratio']=float(d['value']/sv) if d['value'] is not None and sb['n']>=MIN_REFS and sv is not None and sv>EPS else None
        out.append(m)
    return out

@lru_cache(maxsize=1)
def bank():
    cfg=json.loads((Path(ROOT)/'config/word100_multiref.json').read_text('utf-8'))
    path=Path(cfg['baseline_path'])
    if sha256_file(path)!=cfg['baseline_sha256']:raise ValueError('DTW_BASELINE_HASH_MISMATCH')
    data=json.loads(path.read_text('utf-8'))
    from . import word100
    if data['model_sha256']!=word100.PIN:raise ValueError('DTW_BASELINE_WRONG_MODEL')
    # A source edit requires a fresh validation/pin; never silently reuse stale scales.
    for p,h in data['code_sha256'].items():
        if sha256_file(p)!=h:raise ValueError('DTW_BASELINE_CODE_MISMATCH')
    return data,cfg['baseline_sha256']

@lru_cache(maxsize=128)
def sample(path,expected):
    if sha256_file(path)!=expected:raise ValueError('DTW_SAMPLE_HASH_MISMATCH')
    # compare_samples uses only numeric features/masks; do not deserialize the
    # unrelated Python-object metadata stored by uniform_time_sample.
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in ('features','present')}

def compare_query(obs,word_id):
    from . import word100_analysis as A
    data,pin=bank()
    if word_id not in data['words']:
        return dict(schema=SCHEMA,status='UNAVAILABLE',reason='WORD_NOT_IN_PRACTICE_BANK',metrics=[])
    item=data['words'][word_id]
    try:q=A._sample(obs)
    except (ValueError,A.F.FeatureError) as exc:
        return dict(schema=SCHEMA,status='INSUFFICIENT',reason=str(exc),metrics=[],baseline_sha256=pin)
    pairs=[dict(reference_clip=r['clip_id'],reference_signer=r['signer_id'],
                result=A.compare_samples(q,sample(r['sample_path'],r['sample_sha256']))) for r in item['references']]
    metrics=relative(aggregate(pairs),item['baseline'])
    return dict(schema=SCHEMA,status='OK' if all(m['ratio'] is not None for m in metrics) else 'PARTIAL',
        word_id=word_id,label=item['label'],reference_count=len(pairs),baseline_performers=len(item['leave_one_out']),
        baseline_sha256=pin,model_sha256=data['model_sha256'],metrics=metrics,
        comparisons=pairs,accepted=None,is_a_verdict=False,
        notes=['1.0배는 이 단어의 학습자 간 대표적 차이 수준입니다. 정답률이나 합격선이 아닙니다.',
               '본인은 학습자9명과 비교합니다. 기준은 각 학습자가 자신을 제외한8명과 비교한 값입니다.',
               '각 손에서 비교 가능한 참고5개 이상을 사용합니다. 양손 근거가 부족하면 전체 수치를 보류합니다.',
               '손가락 굽힘은2D 마디각 차이이며 손바닥 방향·접촉까지 판단하지 않습니다.',
               '촬영 조건과 개인차도 차이에 포함됩니다. 본인 촬영은 기준·학습·튜닝에 사용하지 않습니다.'])

def saved_comparison(base,result,obs_hash):
    """Append a version-bound analysis sidecar; never rewrite original result/NPZ."""
    try:_,pin=bank()
    except (OSError,ValueError,AssertionError) as exc:
        return dict(schema=SCHEMA,status='UNAVAILABLE',reason=str(exc),metrics=[])
    path=base.with_name(base.name+'.training_dtw_'+pin[:12]+'.json')
    with _LOCK:
        if path.exists():
            value=json.loads(path.read_text('utf-8'))
            if (value.get('query_observation_sha256')!=obs_hash or value.get('baseline_sha256')!=pin or
                value.get('word_id')!=result['requested_word_id'] or value.get('model_sha256')!=result['practice100']['checkpoint_sha256']):
                raise ValueError('DTW_SAVED_BINDING_MISMATCH')
            return value
        value=compare_query(Observation.load(str(base.with_suffix('.npz'))),result['requested_word_id'])
        value.update(query_observation_sha256=obs_hash,baseline_sha256=pin,word_id=result['requested_word_id'],
                     model_sha256=result['practice100']['checkpoint_sha256'],session_id=result['session_id'],trial_id=result['trial_id'])
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        tmp.replace(path)
        return value
