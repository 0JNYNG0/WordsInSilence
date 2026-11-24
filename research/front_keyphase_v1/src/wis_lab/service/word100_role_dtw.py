"""Reference-declared hand roles; never infer optional hands from the user's absence.

The legacy feature engine and bank stay immutable. Two-hand comparisons delegate
exactly to the legacy path. A single-hand comparison uses that side's alignment,
support gates and a newly frozen TRAIN-only descriptive scale, never a pass score.
"""
from functools import lru_cache
from pathlib import Path
import copy
import json
import threading
import numpy as np
from ..cli import ROOT
from ..core.observation import Observation, sha256_file
from . import word100_analysis as A, word100_multiref as M

SCHEMA='wis_reference_hand_role_dtw_v1'
UNIT=Path(ROOT)/'ops/word100_role_dtw_20261017'
_LOCK=threading.Lock()

def required_tuple(sides):
    sides=tuple(sides)
    if sides not in (('L',),('R',),('L','R')):
        raise ValueError('INVALID_REQUIRED_HANDS')
    return sides

def compare_samples(q,ref,required):
    required=required_tuple(required)
    if required==('L','R'):
        return A.compare_samples(q,ref)
    out=dict(schema=SCHEMA,status='INSUFFICIENT',is_a_verdict=False,accepted=None,metrics=[],
             required_sides=list(required),sampling_budget=A.BUDGET,band_ratio=A.BAND)
    if min(len(q['features']),len(ref['features']))<M.MIN_FRAMES:
        return dict(out,reason='TOO_FEW_OBSERVED_FRAMES')
    weights={g:(weight if g.endswith('_'+required[0]) else 0.0) for g,(_,weight) in A.F.GROUPS.items()}
    cost,evidence=A.frame_cost_matrix(q,ref,group_weights=weights,hand_participation=False,one_sided_mode='flat')
    path=A._optimal_path(cost,A.BAND)
    if not path:return dict(out,reason='NO_DTW_PATH')
    qi,ri=np.asarray(path).T
    support=float(evidence[qi,ri].mean())
    out.update(path_pairs=len(path),alignment_evidence_fraction=support,
               query_sampled_frames=len(q['features']),reference_sampled_frames=len(ref['features']))
    if support<A.MIN_PATH_EVIDENCE_FRACTION:return dict(out,reason='INSUFFICIENT_REQUIRED_HAND_OBSERVATION')
    out.update(status='OK',dtw_distance=float(cost[qi,ri].mean()))
    for key,label,unit,group in [('handshape','손가락 굽힘','degrees','bend'),('location','손 위치','shoulder_width_percent','handpos'),('movement','이동 궤적','shoulder_width_percent','handpos')]:
        sides=[]
        for side in ('L','R'):
            if side not in required:
                sides.append(dict(side=side,required=False,status='NOT_REQUIRED',value=None,compared_pairs=0,path_pairs=len(path),support_fraction=0.0))
                continue
            a,av=A._group(q,f'{group}_{side}');b,bv=A._group(ref,f'{group}_{side}')
            valid=av[qi]&bv[ri]; n=int(valid.sum())
            detail=dict(side=side,required=True,value=None,compared_pairs=n,path_pairs=len(path),support_fraction=n/len(path))
            if n:
                aa=a[qi[valid]].copy();bb=b[ri[valid]].copy()
                if key=='handshape':
                    differences=np.abs(aa[:,:15]-bb[:,:15])*180
                    detail['value']=float(differences.mean())
                    detail['fingers']=[dict(name=name,value=float(differences[:,i*3:i*3+3].mean())) for i,name in enumerate(('엄지','검지','중지','약지','소지'))]
                else:
                    if key=='movement':
                        aa-=a[np.flatnonzero(av)[0]];bb-=b[np.flatnonzero(bv)[0]]
                    detail['value']=float(np.linalg.norm(aa-bb,axis=1).mean()*100)
                detail.update(query_distinct_frames=int(len(set(qi[valid]))),reference_distinct_frames=int(len(set(ri[valid]))))
            sides.append(detail)
        values=[d['value'] for d in sides if d['required'] and d['value'] is not None]
        out['metrics'].append(dict(key=key,label=label,unit=unit,sides=sides,value=float(np.mean(values)) if values else None))
    return out

def aggregate(pairs,required):
    required=required_tuple(required)
    metrics=M.aggregate(pairs)
    for m in metrics:
        m['required_sides']=list(required)
        for d in m['sides']:
            d['required']=d['side'] in required
            if not d['required']: d.update(value=None,status='NOT_REQUIRED',valid_references=0,median_support=None)
        values=[d['value'] for d in m['sides'] if d['required']]
        ready=all(v is not None for v in values)
        m.update(value=float(np.mean(values)) if ready else None,status='OK' if ready else 'INSUFFICIENT')
    return metrics

def relative(metrics,baseline):
    out=M.relative(metrics,baseline)
    for m in out:
        for s in m['sides']:
            if s.get('required') is False:
                s.update(value=None,ratio=None,status='NOT_REQUIRED')
    return out

@lru_cache(maxsize=1)
def bank():
    cfg=json.loads((Path(ROOT)/'config/word100_role_dtw.json').read_text('utf-8'))
    path=Path(cfg['bank_path'])
    if sha256_file(path)!=cfg['bank_sha256']:raise ValueError('ROLE_DTW_BANK_HASH_MISMATCH')
    data=json.loads(path.read_text('utf-8'))
    old,pin=M.bank()
    if pin!=data['legacy_bank_sha256']:raise ValueError('ROLE_DTW_LEGACY_BANK_MISMATCH')
    for p,h in data['code_sha256'].items():
        if sha256_file(p)!=h:raise ValueError('ROLE_DTW_CODE_HASH_MISMATCH')
    if sha256_file(data['roles_path'])!=data['roles_sha256']:raise ValueError('ROLE_DTW_ROLES_MISMATCH')
    return data,cfg['bank_sha256']

def compare_query(obs,word_id):
    data,pin=bank()
    if word_id not in data['words']:
        return dict(schema=SCHEMA,status='UNAVAILABLE',reason='NO_REVIEWED_REFERENCE_ROLE',metrics=[])
    item=data['words'][word_id];required=item['required_sides']
    try:q=A._sample(obs)
    except (ValueError,A.F.FeatureError) as exc:
        return dict(schema=SCHEMA,status='INSUFFICIENT',reason=str(exc),metrics=[],word_id=word_id)
    pairs=[dict(reference_clip=x['clip_id'],reference_signer=x['signer_id'],result=compare_samples(q,M.sample(x['sample_path'],x['sample_sha256']),required)) for x in item['references']]
    metrics=relative(aggregate(pairs,required),item['baseline'])
    names={'L':'왼손','R':'오른손'};label='·'.join(names[s] for s in required)
    return dict(schema=SCHEMA,status='OK' if all(m['ratio'] is not None for m in metrics) else 'PARTIAL',
        word_id=word_id,label=item['label'],required_sides=required,role_label=label+' 비교',role_basis=item['role_basis'],
        reference_count=len(pairs),baseline_performers=9,baseline_sha256=pin,legacy_bank_sha256=data['legacy_bank_sha256'],
        metrics=metrics,comparisons=pairs,accepted=None,is_a_verdict=False,
        notes=[f'공개 참고 동작에서 필요한 {label}을 비교합니다. 사용자에게 안 보인다는 이유로 손을 제외하지 않습니다.',
               '움직이지 않더라도 받침이나 모양을 만드는 손은 필요한 손에 포함합니다.',
               '필요한 손의 관측이 부족하면 수치를 보류합니다. 비교하지 않는 손을 0점으로 계산하지 않습니다.',
               '학습 촬영자끼리의 차이는 참고 범위이며 올바른 수어의 합격선이 아닙니다.',
               '본인 촬영은 역할·기준·학습·튜닝에 사용하지 않습니다. 2D 관절 검출의 오차도 차이에 포함됩니다.'])

def saved_comparison(base,result,obs_hash):
    """New versioned sidecar; original result and legacy DTW remain untouched."""
    try:_,pin=bank()
    except (OSError,ValueError,AssertionError) as exc:
        return dict(schema=SCHEMA,status='UNAVAILABLE',reason=str(exc),metrics=[])
    path=base.with_name(base.name+'.role_dtw_'+pin[:12]+'.json')
    with _LOCK:
        if path.exists():
            value=json.loads(path.read_text('utf-8'))
            for k,v in [('query_observation_sha256',obs_hash),('baseline_sha256',pin),('word_id',result['requested_word_id']),('session_id',result['session_id']),('trial_id',result['trial_id'])]:
                if value.get(k)!=v:raise ValueError('ROLE_DTW_SAVED_BINDING_MISMATCH')
            return value
        value=compare_query(Observation.load(str(base.with_suffix('.npz'))),result['requested_word_id'])
        value.update(query_observation_sha256=obs_hash,baseline_sha256=pin,word_id=result['requested_word_id'],session_id=result['session_id'],trial_id=result['trial_id'])
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),'utf-8');tmp.replace(path)
        return value
