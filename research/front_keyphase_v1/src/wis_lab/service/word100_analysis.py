"""Read-only explanation of frozen word100 results; never training or a correctness verdict.

DTW aligns the requested demonstration, not a guessed top1. Measurements use natural2D
units, never the old18-word calibration, pass thresholds, or role heuristics.
"""
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import re
import numpy as np
from ..cli import ROOT
from ..core.observation import Observation, sha256_file
from ..core import features as F
from ..core.dtw import frame_cost_matrix, _optimal_path, MIN_PATH_EVIDENCE_FRACTION
from . import word100, diagnostic_capture

SCHEMA='wis_word100_dtw_descriptive_v1'
RUNS=Path(ROOT)/'runs'
BUDGET=96
BAND=0.2
OFFSETS={}
cursor=0
for name,(width,_) in F.GROUPS.items():
    OFFSETS[name]=slice(cursor,cursor+width);cursor+=width

@lru_cache(maxsize=1)
def references():
    d=json.loads((Path(ROOT)/'config/word100_analysis_references.json').read_text('utf-8'))
    out={x['word_id']:x for x in d['items']}
    assert set(out)==set(word100.config()['word_codes'])
    assert all(x['source_role']=='train_candidate' and 1<=x['signer_id']<=9 for x in out.values())
    return out

@lru_cache(maxsize=100)
def video_path(word_id):
    x=references().get(word_id)
    if not x:raise ValueError('UNKNOWN_WORD')
    if sha256_file(x['video_path'])!=x['source_sha256']:raise ValueError('REFERENCE_HASH_MISMATCH')
    return x['video_path']

def _paths(session_id,trial_id):
    if not re.fullmatch(r'ses_[0-9a-f]{12}',session_id) or not re.fullmatch(r'tri_[0-9a-f]{12}',trial_id):
        raise ValueError('INVALID_RESULT_ID')
    base=RUNS/session_id/'trials'/trial_id
    # No caller-supplied filesystem paths, and no escape through a junction/symlink.
    if not base.resolve().is_relative_to(RUNS.resolve()):raise ValueError('INVALID_RESULT_PATH')
    return base

def saved_result(session_id,trial_id):
    base=_paths(session_id,trial_id)
    r=json.loads(base.with_suffix('.result.json').read_text('utf-8'))
    if r.get('session_id')!=session_id or r.get('trial_id')!=trial_id:raise ValueError('RESULT_ID_MISMATCH')
    out=r.get('practice100') or {}
    if out.get('checkpoint_sha256')!=word100.PIN:raise ValueError('WRONG_MODEL')
    return r,base

def expanded_result(r):
    r=copy.deepcopy(r)
    out=r['practice100']
    if out.get('model_status')=='READY':
        out['ranking']=word100.rank_logits(out['logits'])
        # Full rank is explanatory; it never influences recognition.
        full=word100.rank_logits(out['logits'],100)
        r['requested_rank_all100']=next((x['rank'] for x in full if x['word_id']==r['requested_word_id']),None)
        r['requested_rank_in_top5']=next((x['rank'] for x in out['ranking'] if x['word_id']==r['requested_word_id']),None)
    return r

def history(limit=30):
    entries=[]
    # Local result history contains no coordinates, images, participant names or arbitrary paths.
    sessions=sorted(RUNS.glob('ses_*'),key=lambda p:p.stat().st_mtime,reverse=True)[:100]
    labels=dict(zip(word100.config()['word_codes'],word100.config()['labels']))
    for session in sessions:
        try:
            meta=json.loads((session/'session.json').read_text('utf-8'))
            if str(meta.get('participant_id','')).startswith('SYNTHETIC'):continue
        except (OSError,ValueError):continue
        reviewed=set()
        try:
            for line in (session/'trial_reviews.jsonl').read_text('utf-8').splitlines():
                try:reviewed.add(json.loads(line)['trial_id'])
                except (ValueError,KeyError):pass
        except OSError:pass
        for p in (session/'trials').glob('*.result.json'):
            try:
                r,_=saved_result(session.name,p.name.removesuffix('.result.json'))
                expanded=expanded_result(r);rank=expanded['practice100'].get('ranking') or []
                stamp=p.stat().st_mtime
                entries.append(dict(session_id=session.name,trial_id=r['trial_id'],requested_word_id=r['requested_word_id'],
                    label=labels.get(r['requested_word_id'],r['requested_word_id']),
                    top1=rank[0]['label'] if rank else None,requested_rank=expanded.get('requested_rank_all100'),
                    reviewed=r['trial_id'] in reviewed,recorded_at=datetime.fromtimestamp(stamp,timezone.utc).isoformat(),
                    _time=stamp))
            except (OSError,ValueError,KeyError):continue
    entries.sort(key=lambda r:r['_time'],reverse=True)
    for r in entries:r.pop('_time')
    return dict(items=entries[:limit],retrospective=True)

def _sample(obs):
    return F.uniform_time_sample(F.sequence_features(obs),BUDGET)

def _group(seq,name):
    arr=seq['features'][:,OFFSETS[name]]
    valid=seq['present'][:,F.GROUP_ORDER.index(name)] & np.isfinite(arr).all(axis=1)
    return arr,valid

def compare(query,reference):
    """Synthetic tests can exercise measurements without a model, files, or camera."""
    try:q=_sample(query);ref=_sample(reference)
    except (F.FeatureError,ValueError) as exc:
        return dict(schema=SCHEMA,status='INSUFFICIENT',is_a_verdict=False,accepted=None,metrics=[],reason=str(exc))
    return compare_samples(q,ref)


def compare_samples(q,ref):
    """Same measurements with precomputed features, for the frozen training bank."""
    out=dict(schema=SCHEMA,status='INSUFFICIENT',is_a_verdict=False,accepted=None,
        sampling_budget=BUDGET,band_ratio=BAND,metrics=[],
        time_policy='first-to-last observed hand on BOTH sides; preserve internal gaps; nearest real-time samples',
        comparison_meaning='Difference from one displayed2D reference, not distance from a proven pass threshold')
    if min(len(q['features']),len(ref['features']))<8:return dict(out,reason='TOO_FEW_OBSERVED_FRAMES')
    cost,evidence=frame_cost_matrix(q,ref,hand_participation=False,one_sided_mode='flat')
    path=_optimal_path(cost,BAND)
    if not path:return dict(out,reason='NO_DTW_PATH')
    qi,ri=np.asarray(path).T
    support=float(evidence[qi,ri].mean())
    out.update(path_pairs=len(path),alignment_evidence_fraction=support,
        query_sampled_frames=len(q['features']),reference_sampled_frames=len(ref['features']))
    if support<MIN_PATH_EVIDENCE_FRACTION:return dict(out,reason='INSUFFICIENT_PAIRED_OBSERVATION')
    out.update(status='OK',dtw_distance=float(cost[qi,ri].mean()))
    specs=[('handshape','손가락 굽힘','degrees','bend'),('location','손 위치','shoulder_width_percent','handpos'),
           ('movement','이동 궤적','shoulder_width_percent','handpos')]
    for key,label,unit,group in specs:
        sides=[]
        for side in ('L','R'):
            a,av=_group(q,f'{group}_{side}');b,bv=_group(ref,f'{group}_{side}')
            valid=av[qi]&bv[ri];n=int(valid.sum());detail=dict(side=side,compared_pairs=n,
                path_pairs=len(path),support_fraction=n/len(path),value=None)
            if n:
                aa=a[qi[valid]].copy();bb=b[ri[valid]].copy()
                if key=='handshape':
                    differences=np.abs(aa[:,:15]-bb[:,:15])*180
                    detail['value']=float(differences.mean())
                    detail['fingers']=[dict(name=name,value=float(differences[:,i*3:i*3+3].mean()))
                                       for i,name in enumerate(('엄지','검지','중지','약지','소지'))]
                else:
                    if key=='movement':
                        aa-=a[np.flatnonzero(av)[0]];bb-=b[np.flatnonzero(bv)[0]]
                    detail['value']=float(np.linalg.norm(aa-bb,axis=1).mean()*100)
                detail.update(query_distinct_frames=int(len(set(qi[valid]))),reference_distinct_frames=int(len(set(ri[valid]))))
            sides.append(detail)
        values=[s['value'] for s in sides if s['value'] is not None]
        out['metrics'].append(dict(key=key,label=label,unit=unit,value=float(np.mean(values)) if values else None,
            sides=sides,support_fraction=sum(s['compared_pairs'] for s in sides)/(2*len(path)),
            interpretation='lower means closer to THIS reference, not a pass/fail score',
            aggregation='equal mean over measured hands; unobserved hands never replaced with0'))
    out['notes']=['손가락 각도는 화면에 투영된2D 각도이며 실제3D 관절각이 아닙니다.',
        '손 위치는 어깨 중심 기준, 이동 궤적은 각 손의 첫 관측 위치 기준입니다. 어깨너비를100으로 환산합니다.',
        '수행 속도 차이는DTW로 정렬합니다. 준비·마무리도 손이 보이는 구간에는 포함됩니다.',
        '관측되지 않은 손은0점으로 채우지 않습니다. 비교 비율은DTW 대응쌍 기준이며 실제 시간 비율이 아닙니다.',
        '하나의 참고 영상과의 차이입니다. 다른 올바른 수어 변형이나 가림도 차이로 나타날 수 있습니다.']
    return out

@lru_cache(maxsize=100)
def _reference_obs(word_id,expected_hash):
    x=references()[word_id]
    if sha256_file(x['observation'])!=expected_hash:raise ValueError('REFERENCE_OBSERVATION_HASH_MISMATCH')
    return Observation.load(x['observation'])

@lru_cache(maxsize=64)
def _details(session_id,trial_id,obs_hash,ref_hash,word_id):
    base=_paths(session_id,trial_id)
    obs=Observation.load(str(base.with_suffix('.npz')))
    ref=_reference_obs(word_id,ref_hash)
    measured=compare(obs,ref)
    measured.update(reference_word_id=word_id,reference_label=references()[word_id]['label'],
        reference_clip_id=references()[word_id]['clip_id'],reference_observation_sha256=ref_hash,
        query_observation_sha256=obs_hash)
    return dict(dtw=measured,observation_quality=obs.quality_summary())

def analysis(session_id,trial_id):
    r,base=saved_result(session_id,trial_id);r=expanded_result(r)
    x=references()[r['requested_word_id']]
    bound=r.get('reference_binding') or {}
    if bound.get('demonstration_clip_id')!=x['clip_id'] or bound.get('source_sha256')!=x['source_sha256']:
        raise ValueError('REFERENCE_BINDING_MISMATCH')
    obs_hash=sha256_file(base.with_suffix('.npz'))
    if obs_hash!=r['observation_sha256']:raise ValueError('OBSERVATION_HASH_MISMATCH')
    # Recheck before serving a cached explanation; original files remain read-only.
    if sha256_file(x['observation'])!=x['observation_sha256']:raise ValueError('REFERENCE_OBSERVATION_HASH_MISMATCH')
    r['analysis']=copy.deepcopy(_details(session_id,trial_id,obs_hash,x['observation_sha256'],x['word_id']))
    from . import word100_multiref
    r['analysis']['training_comparison']=word100_multiref.saved_comparison(base,r,obs_hash)
    r['analysis']['ranking_source']='original saved100 logits; no inference rerun'
    r['analysis']['is_a_verdict']=False
    return r

def capture(session_id,trial_id):
    r,base=saved_result(session_id,trial_id)
    path=base.with_suffix('.capture.zip');manifest=diagnostic_capture.read_manifest(path)
    if (manifest['session_id']!=session_id or manifest['trial_id']!=trial_id or
            manifest['observation_sha256']!=r['observation_sha256']):raise ValueError('CAPTURE_ID_MISMATCH')
    return path,manifest
