"""Explicit100-word practice. Recognition only; no legacy18-word DTW or pass judgement."""
from functools import lru_cache
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from ..cli import ROOT
from ..core.observation import sha256_file
from . import neural

KEY = 'word100_tcn_context_v1'
CANDIDATE_KEY = 'word100_tcn_roi_c_v1'
T3F_KEY = 'word100_tcn_t3f_v1'
PIN = 'e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'
_LOCK = threading.Lock()

@lru_cache(maxsize=1)
def config():
    d = json.loads((Path(ROOT)/'config/word100_practice.json').read_text('utf-8'))
    assert d['model_key'] == KEY and d['checkpoint_sha256'] == PIN
    assert len(d['word_codes']) == len(d['labels']) == len(set(d['labels'])) == 100
    assert len(set(d['word_codes'])) == 100
    assert len({x['word_id'] for x in d['items']}) == len(d['items'])
    for x in d['items']:
        assert x['source_role'] == 'train_candidate' and 1 <= x['signer_id'] <= 9
        assert d['labels'][d['word_codes'].index(x['word_id'])] == x['label']
    return d

@lru_cache(maxsize=1)
def _load():
    d = config()
    assert sha256_file(d['checkpoint']) == PIN, 'CHECKPOINT_HASH_MISMATCH'
    src = str(Path(neural.PILOT)/'src')
    if src not in sys.path: sys.path.insert(0, src)
    import torch
    from pilot.model_masked_v1 import MaskedSignTCN
    ck = torch.load(d['checkpoint'], map_location='cpu', weights_only=True)
    assert ck['classes'] == d['labels'] and ck['word_codes'] == d['word_codes']
    assert ck['in_channels'] == 280
    model = MaskedSignTCN(n_classes=100, in_channels=280, **ck['config'])
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    return model

def is_key(key):
    return key in (KEY, CANDIDATE_KEY, T3F_KEY)


def identity(key=KEY):
    if key == T3F_KEY:
        from . import word100_t3f
        return word100_t3f.identity()
    if key == CANDIDATE_KEY:
        from . import word100_candidate
        return word100_candidate.identity()
    if key != KEY:
        raise ValueError('UNKNOWN_PRACTICE_MODEL')
    d = config()
    try:
        with _LOCK: _load()
        status = 'READY'
    except Exception as exc:
        status = 'MODEL_UNAVAILABLE'
    return dict(model_key=KEY, model_id=KEY, display_name='기존 모델 M0', checkpoint_sha256=PIN, model_status=status,
                trained_class_count=100, class_labels=d['labels'], class_word_codes=d['word_codes'],
                feature_schema='pilot.data.build_channels', opted_in=True, no_fallback_to_default=True)

def catalog():
    d = config()
    return dict(model=identity(), models=[identity(), identity(CANDIDATE_KEY), identity(T3F_KEY)], reference_set_id=d['reference_set_id'],
                items=[{k:v for k,v in x.items() if k in
                        ('word_id','label','display_name','clip_id','duration_ms')} for x in d['items']],
                class_count=100, practice_count=len(d['items']))

def reference(word_id):
    item = next((x for x in config()['items'] if x['word_id'] == word_id), None)
    if item is None: raise ValueError('UNKNOWN_PRACTICE_WORD')
    return item

@lru_cache(maxsize=12)
def video_path(word_id):
    item = reference(word_id)
    assert sha256_file(item['video_path']) == item['source_sha256'], 'REFERENCE_HASH_MISMATCH'
    return item['video_path']

def binding(word_id, clip_id):
    x = reference(word_id)
    if clip_id != x['clip_id']: raise ValueError('REFERENCE_WORD_MISMATCH')
    video_path(word_id)
    return dict(status='BOUND', reference_set_id=config()['reference_set_id'],
                demonstration_clip_id=x['clip_id'], demonstration_word_id=word_id,
                source_sha256=x['source_sha256'], source_role='train_candidate',
                bound_at='trial_creation', expert_validation=False)

def rank_logits(logits, limit=5):
    """Display ranking only; use the original100 logits without rerunning or narrowing."""
    d=config(); values=np.asarray(logits,dtype=np.float64)
    if values.shape!=(100,) or not np.isfinite(values).all():
        raise ValueError('INVALID_SAVED_LOGITS')
    probs=np.exp(values-values.max());probs/=probs.sum()
    order=np.argsort(-values,kind='stable')[:limit]
    return [dict(rank=j+1,word_id=d['word_codes'][i],label=d['labels'][i],
                 model_score=float(probs[i])) for j,i in enumerate(order)]


def recognise(obs):
    """Deliberately accepts neither requested word nor demonstration: always rank all100."""
    d = config()
    out = dict(schema='wis_word100_recognition_v1', model_key=KEY, checkpoint_sha256=PIN,
               candidate_count=100, ranking=[], accepted=None, execution_assessment='UNCONFIGURED',
               is_a_verdict=False)
    start = time.perf_counter()
    try:
        import torch
        with _LOCK:
            model = _load()
            clip, why, stats = neural._channels(obs)
            out['support'] = stats
            if clip is None:
                return dict(out, model_status='INSUFFICIENT_OBSERVATION', reason=why)
            x = torch.from_numpy(clip['x'][None].astype(np.float32))
            pad = torch.ones(x.shape[:2], dtype=torch.float32)
            with torch.inference_mode(): logits = model(x, pad)[0].numpy()
        assert logits.shape == (100,) and np.isfinite(logits).all()
        out.update(model_status='READY', ranking=rank_logits(logits),
                   logits=logits.tolist(), score_meaning='uncalibrated model score, not execution correctness')
    except Exception as exc:
        out.update(model_status='MODEL_ERROR', reason=type(exc).__name__)
    out['elapsed_ms'] = round((time.perf_counter()-start)*1000, 2)
    return out

def result(obs, trial, candidate_obs=None, candidate_error=None):
    key = trial['research_model']['model_key']
    baseline = None
    if key == T3F_KEY:
        from . import word100_t3f
        if candidate_obs is not None and not (np.array_equal(obs.frame_id, candidate_obs.frame_id)
                and np.array_equal(obs.timestamp_us, candidate_obs.timestamp_us)):
            candidate_obs = None
            candidate_error = 'CANDIDATE_FRAME_IDENTITY_MISMATCH'
        r, baseline = (word100_t3f.recognise_pair(candidate_obs) if candidate_obs is not None
                       else word100_t3f.unavailable(candidate_error or 'MISSING_C_OBSERVATION'))
    elif key == CANDIDATE_KEY:
        from . import word100_candidate
        if candidate_obs is not None and not (np.array_equal(obs.frame_id,candidate_obs.frame_id)
                and np.array_equal(obs.timestamp_us,candidate_obs.timestamp_us)):
            candidate_obs = None
            candidate_error = 'CANDIDATE_FRAME_IDENTITY_MISMATCH'
        r = word100_candidate.recognise(candidate_obs) if candidate_obs is not None else dict(
            model_key=CANDIDATE_KEY, checkpoint_sha256=word100_candidate.PIN,
            model_status='CANDIDATE_OBSERVATION_UNAVAILABLE', reason=candidate_error or 'MISSING_C_OBSERVATION',
            ranking=[], accepted=None, is_a_verdict=False)
        baseline = recognise(obs)
    elif key == KEY:
        r = recognise(obs)
    else:
        raise ValueError('UNKNOWN_PRACTICE_MODEL')
    requested = trial['requested_word_id']
    for reading in (r, baseline):
        if reading is not None:
            all_ranks = rank_logits(reading['logits'], 100) if reading.get('model_status') == 'READY' else []
            reading['requested_rank_all100'] = next((x['rank'] for x in all_ranks if x['word_id'] == requested), None)
    rank = next((x['rank'] for x in r['ranking'] if x['word_id'] == requested), None)
    top = r['ranking'][0]['word_id'] if r['ranking'] else None
    output = dict(schema='wis_word100_practice_result_v1', practice100=r,
                research_model=trial['research_model'], reference_binding=trial['reference_binding'],
                reference_set_id=config()['reference_set_id'], requested_word_id=requested,
                requested_rank_in_top3=rank if rank and rank<=3 else None,
                requested_rank_in_top5=rank, word_label_match=(top == requested) if top else None,
                top1_word_id=top, observation_status='OK' if r['model_status']=='READY' else r['model_status'],
                accepted=None, execution_assessment='UNCONFIGURED', is_a_verdict=False,
                feedback_status='HANDSHAPE_AND_DTW_NOT_VALIDATED_FOR_THIS_VOCABULARY')
    output['requested_rank_all100'] = r.get('requested_rank_all100')
    output['data_use'] = 'evaluation_only_never_training_tuning_or_model_selection'
    if baseline is not None:
        output['comparison'] = dict(baseline=baseline, same_capture=True,
            same_frames=bool(candidate_obs is not None and np.array_equal(obs.frame_id,candidate_obs.frame_id)
                             and np.array_equal(obs.timestamp_us,candidate_obs.timestamp_us)),
            scope='complete frozen pipelines on the same captured RGB frames; not weights-only comparison',
            dtw_policy='unchanged original observations and frozen training reference bank; not C weights')
        if key == T3F_KEY:
            output['comparison'].update(
                same_input=bool(r.get('input_sha256') and r.get('input_sha256') == baseline.get('input_sha256')
                                and r.get('input_shape') == baseline.get('input_shape')),
                input_sha256=r.get('input_sha256'), input_shape=r.get('input_shape'),
                scope='identical frozen C channels and padding; T3-F versus preserved C',
                dtw_policy='unchanged requested-word reference comparison, shared by both classifiers')
    return output
