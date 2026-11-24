"""C-aware result transport around the UNCHANGED, hash-pinned DTW implementation.

Classifier identity and the older frozen DTW bank identity are deliberately separate.
No model inference, training, threshold fit or calibration is performed by this adapter.
"""
import copy
import json
from datetime import datetime, timezone
from . import word100, word100_analysis as frozen, word100_multiref, word100_role_dtw
from ..core.observation import sha256_file

RUNS = frozen.RUNS
video_path = frozen.video_path
expanded_result = frozen.expanded_result


def saved_result(session_id, trial_id):
    base = frozen._paths(session_id, trial_id)
    result = json.loads(base.with_suffix('.result.json').read_text('utf-8'))
    if result.get('session_id') != session_id or result.get('trial_id') != trial_id:
        raise ValueError('RESULT_ID_MISMATCH')
    out = result.get('practice100') or {}
    if out.get('model_key') == word100.KEY:
        return frozen.saved_result(session_id, trial_id)
    from .word100_candidate import KEY, PIN
    expected_baseline = (word100.KEY, word100.PIN)
    if out.get('model_key') == word100.T3F_KEY:
        from . import word100_t3f
        expected_baseline = (KEY, PIN)
        KEY, PIN = word100_t3f.KEY, word100_t3f.PIN
    identity = result.get('research_model') or {}
    if (out.get('model_key'), out.get('checkpoint_sha256')) != (KEY, PIN):
        raise ValueError('WRONG_MODEL')
    if (identity.get('model_key'), identity.get('checkpoint_sha256')) != (KEY, PIN):
        raise ValueError('CANDIDATE_IDENTITY_MISMATCH')
    binding = result.get('candidate_observation') or {}
    if out.get('model_status') == 'READY' and not binding:
        raise ValueError('MISSING_CANDIDATE_OBSERVATION_BINDING')
    if binding and sha256_file(base.with_name(base.name+'.candidate_c.npz')) != binding.get('observation_sha256'):
        raise ValueError('CANDIDATE_OBSERVATION_HASH_MISMATCH')
    baseline = (result.get('comparison') or {}).get('baseline') or {}
    if (baseline.get('model_key'),baseline.get('checkpoint_sha256')) != expected_baseline:
        raise ValueError('BASELINE_IDENTITY_MISMATCH')
    if out.get('model_key') == word100.T3F_KEY:
        paired = result.get('comparison') or {}
        if out.get('model_status') == baseline.get('model_status') == 'READY':
            hashes = [x.get('input_sha256') for x in (out, baseline, paired)]
            shapes = [x.get('input_shape') for x in (out, baseline, paired)]
            if (not paired.get('same_frames') or not paired.get('same_input')
                    or not hashes[0] or len(set(hashes)) != 1
                    or not shapes[0] or not shapes[0] == shapes[1] == shapes[2]):
                raise ValueError('T3F_PAIRED_INPUT_MISMATCH')
    return result, base


def history(limit=30):
    entries = []
    labels = dict(zip(word100.config()['word_codes'], word100.config()['labels']))
    for session in sorted(RUNS.glob('ses_*'), key=lambda p:p.stat().st_mtime, reverse=True)[:100]:
        try:
            meta = json.loads((session/'session.json').read_text('utf-8'))
            if str(meta.get('participant_id','')).startswith('SYNTHETIC'):
                continue
        except (OSError, ValueError):
            continue
        reviewed = set()
        try:
            for line in (session/'trial_reviews.jsonl').read_text('utf-8').splitlines():
                try: reviewed.add(json.loads(line)['trial_id'])
                except (ValueError, KeyError): pass
        except OSError: pass
        for path in (session/'trials').glob('*.result.json'):
            try:
                result,_ = saved_result(session.name, path.name.removesuffix('.result.json'))
                result = expanded_result(result)
                out = result['practice100']; ranking = out.get('ranking') or []
                stamp = path.stat().st_mtime
                entries.append(dict(session_id=session.name,trial_id=result['trial_id'],
                    requested_word_id=result['requested_word_id'],label=labels.get(result['requested_word_id']),
                    top1=ranking[0]['label'] if ranking else None,requested_rank=result.get('requested_rank_all100'),
                    reviewed=result['trial_id'] in reviewed,model_key=out['model_key'],
                    checkpoint_sha256=out['checkpoint_sha256'],
                    recorded_at=datetime.fromtimestamp(stamp,timezone.utc).isoformat(),_time=stamp))
            except (OSError, ValueError, KeyError): continue
    entries.sort(key=lambda r:r['_time'],reverse=True)
    for row in entries: row.pop('_time')
    return dict(items=entries[:limit],retrospective=True)


def analysis(session_id, trial_id):
    result, base = saved_result(session_id, trial_id)
    if result['practice100']['model_key'] == word100.KEY:
        result = frozen.analysis(session_id, trial_id)
        return _with_hand_roles(result, base, result['observation_sha256'])
    result = expanded_result(result)
    ref = frozen.references()[result['requested_word_id']]
    bound = result.get('reference_binding') or {}
    if bound.get('demonstration_clip_id') != ref['clip_id'] or bound.get('source_sha256') != ref['source_sha256']:
        raise ValueError('REFERENCE_BINDING_MISMATCH')
    obs_hash = sha256_file(base.with_suffix('.npz'))
    if obs_hash != result['observation_sha256']:
        raise ValueError('OBSERVATION_HASH_MISMATCH')
    if sha256_file(ref['observation']) != ref['observation_sha256']:
        raise ValueError('REFERENCE_OBSERVATION_HASH_MISMATCH')
    details = copy.deepcopy(frozen._details(session_id,trial_id,obs_hash,ref['observation_sha256'],ref['word_id']))
    # Explicit DTW-only binding: old primary observations + old TRAIN bank.
    # The classifier response itself is never rewritten or resaved here.
    dtw_binding = copy.deepcopy(result)
    dtw_binding['practice100'] = dict(model_key=word100.KEY,checkpoint_sha256=word100.PIN)
    details['training_comparison'] = copy.deepcopy(word100_multiref.saved_comparison(base,dtw_binding,obs_hash))
    details.update(classifier_checkpoint_sha256=result['practice100']['checkpoint_sha256'],
        dtw_reference_model_sha256=word100.PIN,
        dtw_pipeline='unchanged original observation and frozen TRAIN bank; not classifier weights',
        ranking_source='original saved100 logits; no inference rerun',is_a_verdict=False)
    result['analysis'] = details
    return _with_hand_roles(result, base, obs_hash)


def _with_hand_roles(result, base, obs_hash):
    """Preserve legacy explanations; expose the new reference-role comparison."""
    details = result['analysis']
    details['legacy_training_comparison'] = details['training_comparison']
    details['legacy_dtw'] = details['dtw']
    role = word100_role_dtw.saved_comparison(base, result, obs_hash)
    details['training_comparison'] = role
    details['dtw_pipeline'] = 'reference-required hands from public TRAIN; original observation; frozen legacy retained'
    if role.get('required_sides') and len(role['required_sides']) == 1:
        reference_id = details['dtw'].get('reference_clip_id')
        pair = next((p for p in role.get('comparisons', []) if p['reference_clip'] == reference_id), None)
        if pair:
            current = copy.deepcopy(pair['result'])
            current.update(reference_word_id=result['requested_word_id'], reference_clip_id=reference_id,
                           notes=role['notes'], query_observation_sha256=obs_hash)
            details['dtw'] = current
    # Add query-side time evidence without changing the pinned DTW scores/bank.
    # Only runs when a user requests this trial; no bulk history replay or fitting.
    from . import word100_observation_coverage as coverage
    from ..core.observation import Observation
    try:
        if sha256_file(base.with_suffix('.npz')) != obs_hash:
            raise ValueError('OBSERVATION_HASH_MISMATCH')
        details['observation_coverage'] = coverage.summarize(
            Observation.load(str(base.with_suffix('.npz'))), role.get('required_sides'))
        details['observation_coverage']['query_observation_sha256'] = obs_hash
    except (OSError, ValueError, AssertionError) as exc:
        details['observation_coverage'] = dict(schema=coverage.SCHEMA, status='UNAVAILABLE',
            metrics=[], reason=str(exc), is_a_verdict=False, accepted=None)
    return result


def capture(session_id, trial_id):
    result,base = saved_result(session_id,trial_id)
    path = base.with_suffix('.capture.zip')
    manifest = frozen.diagnostic_capture.read_manifest(path)
    if (manifest['session_id'] != session_id or manifest['trial_id'] != trial_id or
            manifest['observation_sha256'] != result['observation_sha256']):
        raise ValueError('CAPTURE_ID_MISMATCH')
    return path,manifest
