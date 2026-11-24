"""Independent saved-value arithmetic only. Never import runtime/model modules.

Run once only after root announces PUBLIC_RUN_FINISHED. Reads JSON/NPZ and
hashes pinned files; never decodes video, deserializes checkpoints, or forwards.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import traceback
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
CHECKS = []


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def value_sha(v):
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def npz(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def check(name, ok):
    CHECKS.append({'check': name, 'pass': bool(ok)})
    if not ok:
        raise AssertionError(name)


def equal(name, a, b):
    check(name, np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def close(name, a, b, atol=2e-7):
    check(name, np.allclose(a, b, rtol=0, atol=atol, equal_nan=False))


def additive(name, before, after, exceptions=()):
    for k, v in before.items():
        if k not in exceptions:
            check(name + ':' + k, k in after and after[k] == v)


def finite_nonnegative(name, value):
    check(name, isinstance(value, (float, int)) and not isinstance(value, bool)
          and math.isfinite(value) and value >= 0)


def axis(name, data, ids, ts):
    equal(name + ':frames', data['frame_id'], ids)
    equal(name + ':PTS', data['timestamp_us'], ts)


def reconstruct_masks(a):
    raw, m = a['x'], a['mask']
    v, d = raw[:, 2:4] - raw[:, 4:6], raw[:, 36:38] - raw[:, 4:6]
    den = (v * v).sum(1)
    good = m[:, [2, 3, 4, 5, 36, 37]].all(1) & np.isfinite(den) & (den > .0025)
    along = (d * v).sum(1) / np.maximum(den, 1e-6)
    side = (d[:, 0] * v[:, 1] - d[:, 1] * v[:, 0]) / np.maximum(den, 1e-6)
    z = np.column_stack([raw[:, :61], along, side, a['retained_probability'], a['relation_probability']])
    mask = np.column_stack([m[:, :61] & np.isfinite(raw[:, :61]), good, good,
                           np.isfinite(a['retained_probability']), np.isfinite(a['relation_probability'])])
    mask &= a['image_available'][:, None]
    z = np.where(mask, z, 0).astype(np.float32)
    mask &= np.isfinite(z)
    z = np.where(mask, z, 0)
    keep = read(O/'word100_learned_motion_invariance_20261102/TRAINING_CONTRACT.json')['motion_value_indices']
    selected = np.zeros(73, bool); selected[keep] = True
    rgb = a['image_available'] & np.isfinite(a['rgb']).all(1)
    available = rgb | mask[:, selected].any(1)
    # Independently enumerate maximal observed fragments; single rows are not motion anchors.
    fragments, cursor = [], 0
    while cursor < len(available):
        if not available[cursor]:
            cursor += 1; continue
        end = cursor
        while end + 1 < len(available) and available[end + 1]:
            end += 1
        fragments.append((cursor, end)); cursor = end + 1
    anchor = np.zeros(len(available), bool)
    for lo, hi in fragments:
        if hi > lo:
            anchor[lo:hi + 1] = True
    gs = read(O/'word100_learned_motion_order_20261102/NORMALIZATION.json')
    rs = read(O/'word100_learned_rgb_motion_20261102/RGB_NORMALIZATION.json')
    geo = np.concatenate([np.where(mask, (z - np.asarray(gs['mean'])) / np.asarray(gs['std']), 0), mask], 1).astype(np.float32)
    video = np.concatenate([np.where(rgb[:, None], (a['rgb'] - np.asarray(rs['mean'])) / np.asarray(rs['std']), 0), rgb[:, None]], 1).astype(np.float32)
    x = np.concatenate([geo, video], 1)
    check('independent915 finite/shape', x.shape == (len(z), 915) and np.isfinite(x).all())
    return mask, selected, rgb, available, anchor, fragments, x


def rgb_support(a, windows, ids, ts):
    n = len(ids)
    starts = list(range(0, n - 15, 2))
    if starts[-1] != n - 16:
        starts.append(n - 16)
    equal('RGB actual window starts', windows['starts'], starts)
    centers = (ts[np.asarray(starts)] + ts[np.asarray(starts) + 15]) / 2
    equal('RGB original-PTS window centers', windows['center_timestamp_us'], centers)
    axis('RGB window/native axis', windows, ids, ts)
    for key, native in [('features', 'rgb'), ('prelinear_windows', 'prelinear')]:
        v = windows[key]
        check(key + ':shape/finite', v.shape == (len(starts), 768) and np.isfinite(v).all())
        restored = np.stack([np.interp(ts, centers, v[:, j]) for j in range(768)], 1).astype(np.float32)
        equal(key + ':exact native interpolation', restored, a[native])
    supports = []
    for i, t in enumerate(ts):
        if not a['image_available'][i] or not np.isfinite(a['rgb'][i]).all():
            supports.append([]); continue
        if t <= centers[0]:
            used = [0]
        elif t >= centers[-1]:
            used = [len(centers) - 1]
        else:
            right = int(np.searchsorted(centers, t, side='left'))
            used = [right] if centers[right] == t else [right - 1, right]
        supports.append(sorted({j for k in used for j in range(starts[k], starts[k] + 16)}))
    return supports


def verify():
    source = read(U/'PUBLIC_INPUT_FROZEN.json')
    run = read(U/'PUBLIC_RUN_RESULT.json'); finished = read(U/'PUBLIC_RUN_FINISHED.json')
    started = read(U/'PUBLIC_RUN_STARTED.json'); frozen = read(U/'EXECUTION_FROZEN.json')
    jid = run['job_id']; d = U/'public_runtime_runs'/jid/'analysis'
    check('job identity', finished['job_id'] == started['job_id'] == jid and len(jid) == 32)
    check('one job directory', [p.name for p in (U/'public_runtime_runs').iterdir() if p.is_dir()] == [jid])
    check('public source exact SHA', sha(source['source']) == source['source_sha256'] == run['source_sha256'] == started['source_sha256'])
    check('registered10 integration exposure', source['registered_id'] == 10 and source['word_id'] == run['word_id'] == 'WORD1133'
          and source['role'] == 'new_exposed_development_integration_not_accuracy_test' and source['future_fresh_holdout_claim'] is False)
    for k in ['physical_occurrence_gold', 'whole_clause_truth', 'educational_error']:
        check('no exact/educational gold:' + k, source[k] is None)
    check('finished run hash', finished['status'] == 'complete' and finished['result_sha256'] == sha(U/'PUBLIC_RUN_RESULT.json'))
    for p, h in frozen['pins'].items():
        check('execution pin:' + p, sha(p) == h)
    # Check the review was for actual code; no executable imports are needed.
    review = read(U/'CODE_REVIEW.json')
    check('review approved', review['approved_for_one_public_integration'] is True)
    for p, h in review['code_pins'].items():
        check('review code pin:' + p, sha(p) == h)
    a = npz(d/'analysis.npz'); pred = npz(d/'LOCAL_EVENT_PREDICTIONS.npz')
    windows = npz(d/'RGB_NATIVE_SUPPORT.npz'); order_arrays = npz(d/'order_arrays.npz')
    ids, ts = a['frame_id'], a['timestamp_us']; n = len(ids)
    equal('source original frames', ids, source['frame_id']); equal('source original PTS', ts, source['timestamp_us'])
    check('native rows/strict axes', n == source['native_frames'] and n >= 16 and ids.dtype.kind in 'iu'
          and ts.dtype.kind in 'iu' and (np.diff(ids) > 0).all() and (np.diff(ts) > 0).all())
    axis('local predictions', pred, ids, ts); axis('order arrays', order_arrays, ids, ts)
    for name in ['observations.npz', 'raw.npz', 'assignment.npz', 'face.npz']:
        p = d/name
        if p.exists():
            v = npz(p)
            if 'frame_id' in v and 'timestamp_us' in v:
                axis('producer ' + name, v, ids, ts)
    supports = rgb_support(a, windows, ids, ts)
    mask, selected, rgb, av, anchor, fragments, x915 = reconstruct_masks(a)
    for name, expected in [('rgb_available', rgb), ('evidence_available', av), ('motion_anchor', anchor)]:
        equal('prediction ' + name, pred[name], expected)
        check('prediction boolean ' + name, pred[name].dtype.kind == 'b')
    logits, probabilities = pred['logits'], pred['probabilities']
    check('native x5 finite logits/probabilities', logits.shape == probabilities.shape == (n, 5)
          and np.isfinite(logits).all() and np.isfinite(probabilities).all())
    sigmoid = 1 / (1 + np.exp(-logits.astype(np.float64)))
    close('saved sigmoid', probabilities, sigmoid)
    check('probability range', ((probabilities >= 0) & (probabilities <= 1)).all())
    base, ordered, local, live = [read(d/name) for name in ['SOURCE_FEEDBACK.json', 'ORDER_FEEDBACK.json', 'LOCAL_EVENT_FEEDBACK.json', 'LIVE_FEEDBACK.json']]
    additive('SOURCE->ORDER', base, ordered); additive('ORDER->LOCAL', ordered, local); additive('LOCAL->LIVE', local, live)
    legacy = read(d/'FEEDBACK.json')
    additive('legacy->SOURCE except measured latency', legacy, base, ['latency'])
    additive('legacy latency preserved except total', legacy['latency'], base['latency'], ['total_analysis_seconds'])
    check('fresh result pointer/hash', Path(run['result']).resolve() == (d/'LIVE_FEEDBACK.json').resolve()
          and sha(run['result']) == run['result_sha256'])
    path = np.asarray(base['path'])
    check('path axes', base['path_axes'] == ['query_native_index', 'learned_canonical_index'])
    check('path monotone native complete', path.ndim == 2 and path.shape[1] == 2 and path.dtype.kind in 'iu'
          and np.array_equal(np.unique(path[:, 0]), np.arange(n)) and (np.diff(path, axis=0) >= 0).all())
    check('base source/rows', base['source_sha256'] == source['source_sha256'] and base['word_id'] == source['word_id'] and base['frames'] == n)
    close('overlay native seconds', base['overlay']['times'], ts/1e6, 1e-9)
    rc = read(d/'RUNTIME_INPUT_CONTRACT.json'); lc = read(d/'LOCAL_INPUT_CONTRACT.json'); pre = read(d/'LOCAL_EVENT_INPUT_CONTRACT.json')
    axis_hash = hashlib.sha256(ids.tobytes() + ts.tobytes()).hexdigest()
    for name, c in [('runtime', rc), ('local', lc), ('local preflight', pre['job'])]:
        check(name + ':identity', c['job_id'] == jid and c['word_id'] == source['word_id'] and c['source_sha256'] == source['source_sha256'])
        check(name + ':input binding', Path(c['input']).resolve() == (d/'analysis.npz').resolve() and c['input_sha256'] == sha(d/'analysis.npz'))
        check(name + ':window binding', Path(c['rgb_support_path']).resolve() == (d/'RGB_NATIVE_SUPPORT.npz').resolve() and c['rgb_support_sha256'] == sha(d/'RGB_NATIVE_SUPPORT.npz'))
        check(name + ':axis hash', c['native_axis_sha256'] == axis_hash)
        check(name + ':inference-only', all(c[k] is False for k in ['training', 'tuning', 'candidate_selection']))
        check(name + ':base hash', sha(c['base']) == c['base_sha256'])
    check('ordered base value hash', lc['base_value_sha256'] == pre['base_value_sha256'] == value_sha(ordered))
    for p, h in pre['dependency_pins'].items():
        check('local dependency:' + p, sha(p) == h)
    check('local implementation pin', pre['implementation_sha256'] == sha(U/'local_runtime.py'))
    for key, expected in [('native_frame_ids', ids), ('timestamp_us', ts), ('image_available', a['image_available']),
                          ('rgb_available', rgb), ('evidence_available', av), ('motion_anchor', anchor)]:
        equal('preflight ' + key, pre[key], expected)
    equal('geometry feature selection', pre['geometry_valid'], np.r_[selected, selected].astype(np.float32))
    check('preflight before forward', pre['status'] == 'FROZEN_BEFORE_LOCAL_FORWARD' and pre['input_dim'] == 915
          and pre['source_video_read'] is False and pre['new_DTW'] == pre['new_training'] == 0)
    feedback = local['local_event_feedback']; envelope = feedback['identity']; proposals = pre['proposals']
    check('envelope source/axis', envelope['source_sha256'] == source['source_sha256'] and envelope['native_frame_ids'] == ids.tolist()
          and envelope['timestamp_us'] == ts.tolist() and envelope['clip_id'] == jid)
    check('proposals/hash binding', value_sha(proposals) == pre['proposals_value_sha256'] == envelope['proposals_value_sha256'])
    check('prediction hash binding', envelope['AH_prediction_sha256'] == sha(d/'LOCAL_EVENT_PREDICTIONS.npz'))
    train = read(O/'word100_local_event_learning_20261103/TRAIN_RESULT.json')
    check('frozen AH model identity', envelope['AH_model_sha256'] == train['model_sha256'] == sha(train['model']))
    check('local fixed thresholds', feedback['thresholds'] == {'high': .8, 'low': .2, 'joint_support': .5})
    check('scope not score searched', feedback['primary_scopes_fixed_before_readout'] is True
          and feedback['scope_search'] is False and feedback['full_clip_argmax_used'] is False)
    direct = read(O/'word100_source_component_connector_20261102/DIRECT_TARGETS.json')['rows']
    fixed = [r for r in direct if r['role'] == 'fit' and r['person'] == 1 and r['mask'] and r['y'] == 1
             and r['word_id'] == source['word_id'] and r['target'] == 'right_hand_traverses_left_arm_toward_wrist' and r['source_clause_ids']]
    canonical = [min(f for r in fixed for f in r['native_frames']), max(f for r in fixed for f in r['native_frames'])]
    constraints = read(O/'word100_guide_feature_supervision_20261102/COMPARISON_CONTRACT.json')['constraints']
    source_constraint = next(c for c in constraints if c['clause_id'] == 'well_brush')
    check('one supported selected query', len(proposals) == len(feedback['observations']) == 1 and proposals[0]['target_index'] == 0)
    for q, obs in zip(proposals, feedback['observations']):
        j = q['target_index']; lo, hi = q['indices']; ii = np.arange(lo, hi+1)
        hit = path[(path[:, 1] >= canonical[0]) & (path[:, 1] <= canonical[1]), 0]
        check('source nominal/path scope', q['canonical_extent'] == canonical and q['indices'] == [int(hit.min()), int(hit.max())])
        equal('proposal original frames', q['native_frame_ids'], ids[ii]); equal('proposal original PTS', q['timestamp_us'], ts[ii])
        check('source constraint exact', q['source_constraint'] == source_constraint == obs['source_constraint'])
        check('source relation remains partial', q['source_binding_relation'] == 'RELATED_ONLY' and source_constraint['condition_relation'] == 'RELATED_ONLY')
        check('saved proposal unchanged', q == obs['proposal'])
        old = next(m for m in base['motion_observations'] if m['target'] == j)
        ca, cb = old['canonical_frames']; oldhit = path[(path[:, 1] >= ca) & (path[:, 1] <= cb), 0]
        equal('old DTW motion scope preserved', old['frame_ids'], ids[int(oldhit.min()):int(oldhit.max())+1])
        eligible = ii[anchor[ii]]; high = eligible[probabilities[eligible, j] >= .8]
        best = int(eligible[np.argmax(probabilities[eligible, j])]) if len(eligible) else None
        maximum = float(probabilities[best, j]) if best is not None else None
        complete = bool(len(ii) >= 2 and av[ii].all() and anchor[ii].all())
        expected_status = ('LOCAL_MOVEMENT_EVIDENCE_OBSERVED' if len(high) else 'INSUFFICIENT_OBSERVATION' if not complete
                           else 'LOCAL_MOVEMENT_NOT_CONFIRMED' if maximum is not None and maximum <= .2 else 'LOCAL_MOVEMENT_UNCERTAIN')
        check('scope max/status', obs['probability'] == maximum and obs['status'] == expected_status and obs['scope_evidence_complete'] == complete)
        equal('high anchors', obs['high_anchor_native_frames'], ids[high]); equal('high anchor PTS', obs['high_anchor_timestamp_us'], ts[high])
        check('best original axis', obs['maximum_anchor_native_frame'] == (int(ids[best]) if best is not None else None)
              and obs['maximum_anchor_timestamp_us'] == (int(ts[best]) if best is not None else None))
        equal('missing evidence frames', obs['unavailable_native_frames'], ids[ii[~av[ii]]]); equal('nonanchor frames', obs['nonanchor_native_frames'], ids[ii[~anchor[ii]]])
        check('scope denominators', obs['native_scope_denominator'] == len(ii) and obs['anchor_count'] == int(anchor[ii].sum()))
        check('point denominator', len(obs['native_point_estimates']) == len(ii))
        for i, point in zip(ii, obs['native_point_estimates']):
            check('point native/score/masks:' + str(i), point == dict(frame_id=int(ids[i]), timestamp_us=int(ts[i]),
                probability=float(probabilities[i, j]) if anchor[i] else None, raw_model_probability=float(probabilities[i, j]),
                motion_anchor=bool(anchor[i]), evidence_available=bool(av[i]), rgb_available=bool(rgb[i])))
        coverage = float(mask[best, selected].mean()) if best is not None else None
        mode = ('RGB_AND_JOINTS' if coverage is not None and coverage >= .5 else 'IMAGE_TIME_ESTIMATE') if best is not None and rgb[best] else ('JOINT_TIME_ESTIMATE' if best is not None else 'NO_AVAILABLE_ANCHOR')
        check('point evidence provenance', obs['maximum_anchor_evidence_mode'] == mode and obs['maximum_anchor_geometry_coverage'] == coverage)
        tcn = sorted({k for start, end in fragments for i in ii if start <= i <= end and anchor[i]
                      for k in range(max(start, i-7), min(end, i+7)+1)})
        rgb_rows = sorted({r for i in tcn for r in supports[i]}); outside = sorted(set(rgb_rows)-set(ii))
        s = obs['structural_context_support']
        check('saved structural context matches preflight', s == pre['structural_support'][str(j)])
        for key, expected in [('tcn_input_indices', tcn), ('rgb_input_indices', rgb_rows), ('tcn_native_frames', ids[tcn]),
                              ('rgb_native_frames', ids[rgb_rows]), ('rgb_timestamp_us', ts[rgb_rows]),
                              ('rgb_outside_nominal_native_frames', ids[outside])]:
            equal('structural support:' + key, s[key], expected)
        st = next(s for s in base['stages'] if s['guide_id'] == q['guide_id'])
        linked = st['neural_stage_agreement'] >= .5 and st['image_time_coverage'] >= .8 and st['unique_query_frames'] >= 2
        check('existing guide consistency', obs['guide_time_binding']['status'] == ('SUPPORTED' if linked else 'UNCONFIRMED'))
        binding = ('RELATED_OBSERVATION_WITH_STAGE_CONSISTENCY' if linked else 'RELATED_OBSERVATION_STAGE_UNCONFIRMED') if len(high) else ('SOURCE_ASSOCIATION_ONLY_INSUFFICIENT_OBSERVATION' if expected_status == 'INSUFFICIENT_OBSERVATION' else 'SOURCE_ASSOCIATION_ONLY_MOVEMENT_UNCONFIRMED')
        check('source binding status', obs['source_binding'] == binding)
        for key in ['whole_clause_truth', 'whole_sign_correctness', 'complete_action', 'educational_error', 'required_repetitions',
                    'source_variant_applicability_verified', 'source_local_applicability_verified']:
            check('no gold/whole completion:' + key, obs[key] is None)
        check('no correction/calibrated truth', obs['correction_enabled'] is False and obs['probability_is_calibrated_linguistic_truth'] is False)
    # Partial-order evidence is checked against saved masks/times, not relabelled as correctness.
    pairs = read(d/'ORDER_PAIRS.json'); items = ordered['partial_order_feedback']['items']
    check('order pair count', len(pairs) == len(items))
    for k, pair in enumerate(pairs):
        check('pair saved status:' + str(k), pair['status'] == items[k]['observation_status'])
        for prefix in ['first_evidence', 'second_evidence', 'raw_physical_first', 'raw_physical_second']:
            rr = np.asarray(pair[prefix+'_frames'], dtype=int)
            check('pair index bounds:' + prefix + str(k), ((rr >= 0) & (rr < n)).all())
            close('pair native seconds:' + prefix + str(k), pair[prefix+'_seconds'], ts[rr]/1e6, 1e-9)
        equal('pair missing:' + str(k), pair['unobserved_frames'], np.flatnonzero(~a['image_available']))
        check('order no completion:' + str(k), items[k]['complete_action_verified'] is False and items[k]['missing_action_verified'] is False and items[k]['education_error'] is None)
    order_header = ordered['partial_order_feedback']
    check('order whole truth remains unknown', order_header['whole_sign_correctness'] is None and order_header['completion_unknown'] is True)
    lstart = read(d/'LOCAL_EVENT_STARTED.json'); lfinish = read(d/'LOCAL_EVENT_FINISHED.json')
    check('local execution markers', lstart['job_id'] == lfinish['job_id'] == jid and lstart['planned_forwards'] == lfinish['model_forwards'] == 1)
    check('local result/prediction hashes', lfinish['result_sha256'] == sha(d/'LOCAL_EVENT_FEEDBACK.json') and lfinish['prediction_sha256'] == sha(d/'LOCAL_EVENT_PREDICTIONS.npz'))
    check('single local forward and no added DTW/training', feedback['new_model_forwards'] == run['local_event_forwards'] == 1
          and feedback['new_DTW'] == feedback['new_training'] == lfinish['new_DTW'] == lfinish['new_training'] == 0)
    calls = run['actual_dtw_calls']
    check('one actual learned prototype DTW', len(calls) == 1 and calls[0]['query_frames'] == n and calls[0]['learned_prototype_frames'] == int(path[:, 1].max())+1)
    check('zero runtime references', run['individual_reference_video_comparisons'] == base['runtime_individual_reference_video_comparisons'] == 0)
    check('new integration not accuracy', run['new_accuracy_test'] is False and run['training'] == 0 and run['whole_goal_complete'] is False)
    warm = read(d.parent/'WARMUP.json')
    timings = {'cold_job_seconds': run['cold_job_seconds'], 'warmup_seconds': warm['total_warmup_seconds'],
        'server_analysis_seconds': run['server_analysis_seconds'], 'source_analysis_seconds': base['latency']['total_analysis_seconds'],
        'order_head_seconds': ordered['partial_order_runtime']['head_seconds'], 'local_head_seconds': feedback['head_forward_seconds'],
        'local_runtime_seconds': feedback['local_runtime_seconds']}
    for key, v in timings.items():
        finite_nonnegative(key, v)
    check('analysis latency exactly saved', run['server_analysis_seconds'] == live['local_combined_feedback_latency_seconds'])
    check('timing nesting', timings['cold_job_seconds'] >= timings['warmup_seconds']+timings['server_analysis_seconds']
          and timings['server_analysis_seconds'] >= timings['source_analysis_seconds']
          and timings['local_runtime_seconds'] >= timings['local_head_seconds'])
    check('head not e2e claim', feedback['timing_is_end_to_end_latency'] is False and run['latency_excludes'] == 'browser upload/render and user recording')
    check('no runtime failure artifacts', not list(U.glob('PUBLIC_RUN_FAILURE_*.txt')) and not (d/'LOCAL_EVENT_FIRST_FAILURE.txt').exists())
    return {'job_id': jid, 'native_frames': n, 'source_sha256': source['source_sha256'],
        'model_sha256': train['model_sha256'], 'RGB_windows': len(windows['starts']), 'timing': timings,
        'observations': [{'target': v['target_index'], 'status': v['status'], 'probability': v['probability'],
                          'native_frames': v['proposal']['native_frame_ids'], 'source_binding': v['source_binding']} for v in feedback['observations']],
        'pins_checked': len(frozen['pins']), 'input915_value_sha256': hashlib.sha256(x915.tobytes()).hexdigest(),
        'limits_ko': ['새 입력의 exact 동작/교육 정답이 없어 높은 점수를 정확도나 전체 수행 완료로 집계하지 않는다.',
                     '원본 영상 재디코드 없이 root가 동결한 실제 PTS와 저장 native 배열을 대조했다.',
                     'RGB 보간·sigmoid·범위 집계는 저장값 산술 검산이다. 모델/encoder 추론의 수학적 재현은 하지 않았다.',
                     'DTW/forward 횟수는 동결 코드·단회 marker·실행 계측과 저장결과의 일치 확인이다.',
                     '같은 Codex의 별도 저장 검산이며 독립 수어 전문가 정답 검증이 아니다.']}


def main():
    output = U/'SAVED_RUNTIME_VERIFIED.json'
    if output.exists():
        raise FileExistsError('Preserve first verification; do not overwrite/re-run.')
    if not (U/'PUBLIC_RUN_FINISHED.json').exists():
        raise RuntimeError('Root completion notification and finished result are required; no model execution is performed.')
    report = {'status': 'RUNNING', 'new_model_imports': 0, 'checkpoint_deserializations': 0,
              'new_model_forwards': 0, 'new_training': 0, 'new_video_decodes': 0,
              'verifier_sha256': sha(__file__)}
    try:
        tree = ast.parse(Path(__file__).read_text('utf-8'))
        imports = sorted({n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                         {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
        check('pure verifier import allowlist', set(imports) <= {'pathlib', 'ast', 'hashlib', 'json', 'math', 'traceback', 'numpy'})
        report.update(verify()); report['status'] = 'PASS_SAVED_ARITHMETIC_NOT_ACCURACY'
    except BaseException:
        report['status'] = 'FAIL_SAVED_VERIFICATION'; report['exception'] = traceback.format_exc()
        raise
    finally:
        report['checks'] = CHECKS; report['check_count'] = len(CHECKS)
        report['failed_count'] = sum(not c['pass'] for c in CHECKS)
        with output.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({k: report[k] for k in ['status', 'check_count', 'failed_count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
