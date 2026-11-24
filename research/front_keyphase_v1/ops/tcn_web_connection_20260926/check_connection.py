"""Stored-coordinate replay check of the TCN web connection (no camera, no MediaPipe, no writes to
the session). Tolerances are fixed in FROZEN_CHECK_PLAN.json before this ran."""
import hashlib, importlib.util, json, pathlib, sys
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.core.observation import Observation                       # noqa: E402
from wis_lab.service import app as svc, neural                        # noqa: E402

KEY = 'tcn_control_83212868'
SID = 'ses_2bb3148c29e8'
RUN = L / 'runs' / SID
plan = json.loads((H / 'FROZEN_CHECK_PLAN.json').read_text(encoding='utf-8'))
assert plan['status'] == 'FROZEN_BEFORE_COMPARISON'
TOL_LOGIT, TOL_LEAD = 1e-5, 1e-4

sp = importlib.util.spec_from_file_location('rec_rc', L / 'ops/temporal_recovery_candidate_20260926/run_compare.py')
rc = importlib.util.module_from_spec(sp); sp.loader.exec_module(rc)
import torch                                                           # noqa: E402
ck = torch.load(L / plan['model']['path'], map_location='cpu', weights_only=True)
post_model, post_classes = rc.u.rebuild_from_payload(ck)
post_model.eval()

paired = {t['trial_id']: t for t in json.loads((L / 'ops/start_provisional_candidate_20260926/web_prep'
          / f'SESSION_{SID}_PAIRED.json').read_text(encoding='utf-8'))['trials']}
session_files_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*') if p.is_file()}


def eq(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')


def logits(model, ch):
    xx, pad, _ = rc.collate([dict(ch, label=0)])
    with torch.inference_mode():
        return model(torch.from_numpy(xx), torch.from_numpy(pad))[0].numpy().astype(np.float64)


rows = []
cache = neural._load(KEY)
assert cache['status'] == 'READY', cache
assert list(cache['classes']) == post_classes, 'class order differs'
for tid in json.loads((RUN / 'session.json').read_text(encoding='utf-8'))['trial_ids']:
    obs = Observation.load(str(RUN / 'trials' / f'{tid}.npz'))
    # service path, exactly as the live trial calls it (no requested word passed)
    research = svc._research_comparison(obs, {'research_model': svc._research_choice(KEY),
                                              'requested_word_id': None})
    full = research['scopes']['full']
    svc_ch, why, _ = neural._channels(obs)
    post_ch = rc.channels(obs.xy_px, obs.usable, obs.timestamp_us)
    lg_s, lg_p = logits(cache['model'], svc_ch), logits(post_model, post_ch)
    o = np.argsort(-lg_p)
    stored = paired[tid]['operating']
    # the requested word never reaches the ranking: a different requested word gives the same answer
    other = svc._research_comparison(obs, {'research_model': svc._research_choice(KEY),
                                           'requested_word_id': 'NA'})
    rows.append({
        'trial_id': tid,
        'tcn_input_identical': eq(svc_ch, post_ch),
        'service_top1_93': full['top1'], 'post_hoc_top1_93': post_classes[int(o[0])],
        'stored_paired_top1': stored['pred'],
        'top1_identical': full['top1'] == post_classes[int(o[0])] == stored['pred'],
        'logit_max_abs_diff_93': float(np.max(np.abs(lg_s - lg_p))),
        'service_lead': round(full['top3'][0]['score'] - full['top3'][1]['score'], 5),
        'post_hoc_lead': round(float(lg_p[o[0]] - lg_p[o[1]]), 5),
        'stored_paired_lead': stored['lead_over_second'],
        'full_candidate_count': full['candidate_count'],
        'project18_top1': research['scopes']['project18']['top1'],
        'identity': research['identity_fixed_at_trial_creation'],
        'model_status': research['model_status'], 'status_label': research['status_label'],
        'ranking_independent_of_requested_word': other['scopes']['full']['top3'] == full['top3'],
    })
    r = rows[-1]
    r['lead_within_tol'] = abs(r['service_lead'] - r['stored_paired_lead']) <= TOL_LEAD
    r['logits_within_tol'] = r['logit_max_abs_diff_93'] <= TOL_LOGIT
    print(tid, r['service_top1_93'], r['post_hoc_top1_93'], r['stored_paired_top1'], r['logit_max_abs_diff_93'],
          r['service_lead'], r['stored_paired_lead'], r['tcn_input_identical'], flush=True)

# opt-in off: no research reading; the default recogniser is still the base model
off = svc._research_choice(None)
default_id = neural.model_identity()
# failure modes, in memory only (the registry file is not edited): wrong hash and missing file
neural.MODELS['_check_bad_hash'] = dict(neural.MODELS[KEY], model_id='_check_bad_hash', sha256='0' * 64)
neural.MODELS['_check_missing'] = dict(neural.MODELS[KEY], model_id='_check_missing', path=str(H / 'no_such.pt'))
obs0 = Observation.load(str(RUN / 'trials' / f"{rows[0]['trial_id']}.npz"))
bad = neural.recognise(obs0, model_key='_check_bad_hash')
miss = neural.recognise(obs0, model_key='_check_missing')
unknown = neural.recognise(obs0, model_key='_no_such_key')
for k in ('_check_bad_hash', '_check_missing'):
    del neural.MODELS[k]

session_files_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*') if p.is_file()}
checks = {
    'all_inputs_identical': all(r['tcn_input_identical'] for r in rows),
    'all_top1_identical': all(r['top1_identical'] for r in rows),
    'all_logits_within_1e-5': all(r['logits_within_tol'] for r in rows),
    'all_leads_within_1e-4_of_stored': all(r['lead_within_tol'] for r in rows),
    'full_scope_is_93': all(r['full_candidate_count'] == 93 for r in rows),
    'identity_is_pinned_model': all(r['identity']['checkpoint_sha256'] == plan['model']['sha256'] for r in rows),
    'ranking_ignores_requested_word': all(r['ranking_independent_of_requested_word'] for r in rows),
    'opt_in_off_means_no_research_reading': off['opted_in'] is False,
    'default_recogniser_unchanged': default_id['checkpoint_sha256'].startswith('56a17a4d') and neural.DEFAULT_MODEL == 'base',
    'bad_hash_reported_not_substituted': bad['model_status'] == 'CHECKPOINT_HASH_MISMATCH' and bad['ranking'] is None,
    'missing_file_reported_not_substituted': miss['model_status'] == 'CHECKPOINT_MISSING' and miss['ranking'] is None,
    'unknown_key_reported_not_substituted': unknown['model_status'] == 'UNKNOWN_MODEL_KEY',
    'session_files_unchanged': session_files_before == session_files_after,
}
out = {'schema': 'wis_tcn_web_connection_check_v1', 'plan_sha256': hashlib.sha256((H / 'FROZEN_CHECK_PLAN.json').read_bytes()).hexdigest(),
       'kind': 'STORED_COORDINATE_REPLAY - not a live webcam check', 'camera_opened': False,
       'checks': checks, 'verdict': 'PASS' if all(checks.values()) else 'FAIL', 'trials': rows}
(H / 'CONNECTION_CHECK.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(checks, indent=1), '\nVERDICT', out['verdict'])
