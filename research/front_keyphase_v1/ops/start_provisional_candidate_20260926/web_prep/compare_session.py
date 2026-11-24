"""Paired comparison for ONE opted-in web session: operating (primary) vs candidate (shadow) on the
SAME frames. Fixed before any user capture. Nothing here tunes, selects or trains anything.

Run after the user's test:  py -3.13 ops/start_provisional_candidate_20260926/web_prep/compare_session.py <session_id>
Self-test (no user data):   py -3.13 .../compare_session.py --selftest

Per take it reports: slots that differ (frame, side, kind), whether the take has RGB samples at
those frames (so a person can judge them; a differing frame WITHOUT a sample stays UNDECIDABLE),
and the base model's reading of BOTH inputs, as two SEPARATE things:
  matches_selected_word - the prediction equals the word chosen on the page (requested_word_id).
                          This is NOT correctness: the person may have signed something else.
  correct_vs_confirmed  - computed ONLY when the participant confirmed what they actually signed
                          (existing trial review, review_log.latest_by_trial). The confirmation is a
                          self-report, not a verification that the signing itself was correct.
The model reading is reported, never used to choose a rule or coefficient.
"""
import json, pathlib, sys
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[2]
REC = L / 'ops' / 'temporal_recovery_candidate_20260926'
sys.path[:0] = [str(L), str(L / 'src'), str(REC)]


def _rc():
    import importlib.util
    sp = importlib.util.spec_from_file_location('recovery_run_compare', REC / 'run_compare.py')
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def diff(pxy, pus, sxy, sus, content, same):
    out = []
    for r in range(pxy.shape[0]):
        for s in (0, 1):
            a, b = content(pxy, pus, r, s), content(sxy, sus, r, s)
            if (a is None and b is None) or same(a, b):
                continue
            if a is not None and b is None:
                kind = 'MOVED_OUT' if same(a, content(sxy, sus, r, 1 - s)) else 'CANDIDATE_LACKS'
            elif a is None:
                kind = 'MOVED_IN' if same(b, content(pxy, pus, r, 1 - s)) else 'CANDIDATE_ADDS'
            else:
                kind = 'OTHER_DETECTION'
            out.append({'row': r, 'side': 'LR'[s], 'kind': kind})
    return out


def top_prediction(rc, model, classes, ch):
    """label-free reading: the model's top class and its lead over the runner-up"""
    import torch
    xx, pad, _ = rc.collate([dict(ch, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(xx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
    order = np.argsort(-logit)
    return {'pred': classes[int(order[0])], 'lead_over_second': round(float(logit[order[0]] - logit[order[1]]), 5)}


def readings(pred, selected_label, confirmation):
    """keeps 'matches the selected word' and 'correct vs what was actually signed' apart"""
    out = {'pred': pred['pred'], 'lead_over_second': pred['lead_over_second'],
           'matches_selected_word': (None if selected_label is None else pred['pred'] == selected_label)}
    if confirmation is None:
        out['correct_vs_confirmed'], out['correct_status'] = None, 'NOT_CONFIRMED'
    elif confirmation['label'] is None:
        out['correct_vs_confirmed'], out['correct_status'] = None, confirmation['status']
    else:
        out['correct_vs_confirmed'] = pred['pred'] == confirmation['label']
        out['correct_status'] = 'CONFIRMED_BY_PARTICIPANT_SELF_REPORT'
    return out


def compare_pair(prim, shad, selected_label, confirmation, rc, model, classes, rgb_rows=()):
    d = diff(prim.xy_px, prim.usable, shad.xy_px, shad.usable, rc.content, rc.same)
    rgb = set(rgb_rows)
    for x in d:
        x['rgb_sample'] = x['row'] in rgb
        x['judgement'] = 'PENDING_VISUAL' if x['rgb_sample'] else 'UNDECIDABLE_NO_STORED_FRAME'
    res = {'frames': int(prim.n_frames), 'differing_slots': len(d), 'differences': d,
           'differences_with_rgb_sample': sum(x['rgb_sample'] for x in d),
           'differences_undecidable_no_frame': sum(not x['rgb_sample'] for x in d)}
    for tag, o in (('operating', prim), ('candidate', shad)):
        ch = rc.channels(o.xy_px, o.usable, o.timestamp_us)
        res[tag] = None if ch is None else readings(top_prediction(rc, model, classes, ch),
                                                    selected_label, confirmation)
    return res


def confirmation_for(review, word_label, classes):
    """the participant's latest confirmation of what they actually signed, or None"""
    if review is None:
        return None
    base = {'source': review.get('source'), 'is_expert_verification': review.get('is_expert_verification'),
            'revision': review.get('revision'), 'followed_guidance': review.get('followed_guidance'),
            'followed_demonstration_form': review.get('followed_demonstration_form'),
            'actual_word_id': review.get('actual_word_id')}
    actual = review.get('actual_word_id')
    if actual in (None, 'UNKNOWN'):
        return {**base, 'label': None, 'status': 'CONFIRMED_UNKNOWN'}
    lab = word_label.get(actual)
    if lab not in classes:
        return {**base, 'label': None, 'status': 'CONFIRMED_WORD_NOT_IN_MODEL'}
    return {**base, 'label': lab, 'status': 'CONFIRMED'}


def load_model(rc):
    import hashlib, torch
    ck = L / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert hashlib.sha256(ck.read_bytes()).hexdigest().startswith('83212868')
    model, classes = rc.u.rebuild_from_payload(torch.load(ck, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    return model, classes


EXCLUDED_SESSIONS = {'ses_d41f150a3764': 'accidental real-camera test run 2026-09-26; excluded from all use'}


def run_session(sid):
    assert sid not in EXCLUDED_SESSIONS, EXCLUDED_SESSIONS.get(sid)
    from wis_lab.core.observation import Observation
    from wis_lab.service import diagnostic_capture as dc
    from wis_lab.service import review_log
    rc = _rc(); model, classes = load_model(rc)
    run = L / 'runs' / sid
    sess = json.loads((run / 'session.json').read_text(encoding='utf-8'))
    assert sess.get('shadow_assignment') is True, 'session was not opted in'
    reviews, review_err = review_log.latest_by_trial(str(run))
    assert review_err is None, review_err          # an unreadable store is not an empty one
    word_label = {c['word_id']: c['display_name'] for c in
                  json.loads((L / 'config' / 'source_manifest.json').read_text(encoding='utf-8'))['clips']}
    rows = []
    for rj in sorted((run / 'trials').glob('*.result.json')):
        res = json.loads(rj.read_text(encoding='utf-8'))
        sh = res.get('shadow_assignment') or {}
        tid = rj.name[:-len('.result.json')]
        if sh.get('status') != 'SAVED':
            rows.append({'trial_id': tid, 'shadow': sh.get('status')}); continue
        prim = Observation.load(str(run / 'trials' / f'{tid}.npz'))
        shad = Observation.load(str(L / sh['path']))
        assert np.array_equal(prim.frame_id, shad.frame_id)
        word = (res.get('task') or {}).get('requested_word_id')
        selected = word_label.get(word) if word_label.get(word) in classes else None
        conf = confirmation_for(reviews.get(tid), word_label, classes)
        rgb = []
        cz = run / 'trials' / f'{tid}.capture.zip'
        if cz.exists():
            rgb = [f['observation_row'] for f in dc.read_manifest(cz)['frames']]
        rows.append({'trial_id': tid, 'selected_word_id': word, 'selected_label': selected,
                     'confirmation': conf,
                     **compare_pair(prim, shad, selected, conf, rc, model, classes, rgb)})
        print(tid, word, rows[-1]['differing_slots'], rows[-1].get('operating'), rows[-1].get('candidate'), flush=True)
    out = H / f'SESSION_{sid}_PAIRED.json'
    out.write_text(json.dumps({'session_id': sid, 'trials': rows,
                               'reading_rule': 'matches_selected_word is agreement with the word chosen on the page, not '
                                               'correctness. correct_vs_confirmed exists only for trials the participant '
                                               'confirmed; that confirmation is a self-report, not a verification of the signing.',
                               'use': 'evaluation only; no training, no per-person calibration, no rule/coefficient choice'},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', out)


def selftest():
    """Two worker-driven takes from stored public clips (no user data): the paired comparison must
    find exactly the slots where the frozen candidate differs from the operating rule."""
    import check_web_path as cw
    from wis_lab.service import camera as cam_mod
    rc = _rc(); model, classes = load_model(rc)
    takes = [cw.frames_of(cw.src('NIKL_MAEKJU_01')), cw.frames_of(cw.src('NIKL_GAM_01'))]
    w = cam_mod.CameraWorker(cw.CONTRACT, shadow_assignment=True)
    with cw.ShadowLandmarker(cw.CONTRACT) as lm:
        got = cw.drive(w, lm, takes, 0.2)
    out = []
    # take 1: nothing confirmed -> no correctness. take 2: a synthetic confirmation of a DIFFERENT
    # word than the one selected -> the two readings must diverge. Synthetic, no user data.
    cases = [('맥주', None), ('감', {'label': '맥주', 'status': 'CONFIRMED'})]
    for (obs, meta), (sel, conf) in zip(got, cases):
        r = compare_pair(obs, meta['_shadow_obs'], sel, conf, rc, model, classes,
                         rgb_rows=range(0, obs.n_frames, 3))
        out.append({k: r[k] for k in ('frames', 'differing_slots', 'differences_with_rgb_sample',
                                      'differences_undecidable_no_frame', 'operating', 'candidate')})
        assert r['differing_slots'] > 0 and r['operating'] is not None and r['candidate'] is not None
        assert all(x['judgement'] == ('PENDING_VISUAL' if x['rgb_sample'] else 'UNDECIDABLE_NO_STORED_FRAME')
                   for x in r['differences'])
    assert out[0]['operating']['correct_vs_confirmed'] is None
    assert out[0]['operating']['correct_status'] == 'NOT_CONFIRMED'
    assert out[0]['operating']['matches_selected_word'] is True
    assert out[1]['operating']['matches_selected_word'] is True
    assert out[1]['operating']['correct_vs_confirmed'] is False
    assert confirmation_for({'actual_word_id': 'UNKNOWN'}, {}, classes)['status'] == 'CONFIRMED_UNKNOWN'
    assert confirmation_for(None, {}, classes) is None
    try:
        run_session('ses_d41f150a3764')
        raise SystemExit('excluded session was accepted')
    except AssertionError:
        pass
    print('SELFTEST', out)
    (H / 'COMPARE_SELFTEST.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    sys.path.insert(0, str(H))
    selftest() if sys.argv[1:] == ['--selftest'] else run_session(sys.argv[1])
