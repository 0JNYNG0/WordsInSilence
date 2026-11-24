"""Synthetic native-input and additive-contract checks; no model/real-video run."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import numpy as np

U = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_ak_synthetic_runtime', U / 'local_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def main():
    checks = []
    def check(name, value):
        assert bool(value), name
        checks.append(name)
    def rejects(name, call):
        try:
            call()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)

    before_path = sys.path.copy()
    generic = {k: sys.modules.get(k) for k in ['common', 'data', 'train_local', 'adapter', 'engine', 'local_events']}
    torch_before = 'torch' in sys.modules
    rt = runtime.LocalEventRuntime()
    check('metadata construction no model or forward', rt.model is None and rt.forwards == rt.model_loads == 0)
    check('no torch import for metadata checks', ('torch' in sys.modules) == torch_before)
    check('no generic module/path collision', sys.path == before_path and all(sys.modules.get(k) is v for k, v in generic.items()))
    n = 120; rng = np.random.default_rng(7)
    ids = np.arange(n, dtype=np.int64) * 2 + 10  # deliberate PTS-drop frame gaps
    ts = np.arange(n, dtype=np.int64) * 33333
    starts = np.asarray(list(range(0, n - 15, 2)), dtype=np.int64)
    if starts[-1] != n - 16:
        starts = np.r_[starts, n - 16]
    centers = (ts[starts] + ts[starts + 15]) / 2
    windows = dict(features=rng.normal(size=(len(starts), 768)).astype(np.float32),
                   prelinear_windows=rng.normal(size=(len(starts), 768)).astype(np.float32),
                   starts=starts, center_timestamp_us=centers, frame_id=ids.copy(), timestamp_us=ts.copy())
    a = dict(x=rng.normal(size=(n, 69)).astype(np.float32), mask=np.ones((n, 69), bool),
             retained_probability=np.full((n, 7), .5, dtype=np.float32),
             relation_probability=np.full((n, 3), .5, dtype=np.float32), frame_id=ids, timestamp_us=ts,
             image_available=np.ones(n, bool))
    for source, dest in [('features', 'rgb'), ('prelinear_windows', 'prelinear')]:
        a[dest] = np.stack([np.interp(ts, centers, windows[source][:, j]) for j in range(768)], 1).astype(np.float32)
    original = {k: v.copy() for k, v in a.items()}
    p = rt.prepare(a)
    check('finite native915', p['x'].shape == (n, 915) and p['x'].dtype == np.float32 and np.isfinite(p['x']).all())
    check('nonconsecutive native IDs preserved', np.array_equal(p['ids'], ids) and np.array_equal(p['ts'], ts))
    check('all synthetic observations retained', p['evidence_available'].all() and p['motion_anchor'].all())
    check('input arrays unchanged', all(np.array_equal(a[k], original[k], equal_nan=True) for k in a))
    validated = runtime.verify_rgb_windows(a, windows, p)
    check('actual window interpolation joins', validated['features_and_prelinear_reconstructed'] is True and validated['new_encoder_forwards'] == 0)
    wrong_windows = copy.deepcopy(windows); wrong_windows['center_timestamp_us'][1] += 1
    rejects('changed RGB window center rejected', lambda: runtime.verify_rgb_windows(a, wrong_windows, p))
    wrong_values = copy.deepcopy(windows); wrong_values['features'][0, 0] += .2
    rejects('changed RGB window value rejected', lambda: runtime.verify_rgb_windows(a, wrong_values, p))
    wrong_axis = copy.deepcopy(windows); wrong_axis['frame_id'][4] += 1
    rejects('RGB source axis mismatch rejected', lambda: runtime.verify_rgb_windows(a, wrong_axis, p))
    missing = {k: v for k, v in a.items() if k != 'relation_probability'}
    rejects('no silently generated relation evidence', lambda: rt.prepare(missing))
    repeated = copy.deepcopy(a); repeated['timestamp_us'][8] = repeated['timestamp_us'][7]
    rejects('duplicate PTS rejected', lambda: rt.prepare(repeated))
    fractional = copy.deepcopy(a); fractional['frame_id'] = fractional['frame_id'].astype(float)
    rejects('fractional axis dtype rejected', lambda: rt.prepare(fractional))
    bad_probability = copy.deepcopy(a); bad_probability['retained_probability'][1, 2] = 1.2
    rejects('invalid auxiliary probability rejected', lambda: rt.prepare(bad_probability))
    no_image_field = {k: v for k, v in a.items() if k != 'image_available'}
    rejects('explicit image availability required', lambda: rt.prepare(no_image_field))
    gapped = copy.deepcopy(a); gapped['image_available'][57:61] = False
    gp = rt.prepare(gapped)
    check('blackout masks all modalities despite finite values', not gp['mask'][57:61].any() and not gp['available'][57:61].any() and not gp['motion_anchor'][57:61].any())
    check('gap keeps all native slots', len(gp['ids']) == n and gp['observed_fragments'] == [[0, 56], [61, 119]])
    s = runtime.composed_support(gp, [55, 56])
    check('TCN does not cross observation gap', max(s['tcn_input_indices']) == 56)
    check('RGB outside nominal support explicit', len(s['rgb_outside_nominal_native_frames']) > 0 and s['rgb_window_rows'] == 16)
    blank = copy.deepcopy(a); blank['image_available'][:] = False
    bp = rt.prepare(blank)
    check('complete absence not normal evidence', not bp['evidence_available'].any() and not bp['motion_anchor'].any() and (bp['x'] == 0).all())
    isolated = copy.deepcopy(blank); isolated['image_available'][40] = True
    ip = rt.prepare(isolated)
    check('one observed row is not movement anchor', ip['evidence_available'][40] and not ip['motion_anchor'].any())
    geometry_only = copy.deepcopy(a); geometry_only['rgb'][:] = np.nan
    jp = rt.prepare(geometry_only)
    check('available geometry survives missing RGB', jp['evidence_available'].all() and not jp['available'].any())
    image_only = copy.deepcopy(a); image_only['mask'][:] = False
    image_only['retained_probability'][:] = np.nan; image_only['relation_probability'][:] = np.nan
    vp = rt.prepare(image_only)
    check('available RGB survives missing pose and auxiliaries', vp['evidence_available'].all() and not vp['mask'].any())

    def base_for(word):
        catalog = [m for m in rt.motion_guides if m['word_id'] == word]
        stages = []; motions = []
        for m in catalog:
            lo, hi = m['canonical_frames']
            stages.append(dict(guide_id=m['guide_id'], frame_ids=ids[lo:hi + 1].tolist(),
                description='합성 설명', neural_stage_agreement=.9, image_time_coverage=1., unique_query_frames=hi - lo + 1))
            motions.append(dict(target=m['target'], guide_id=m['guide_id'], canonical_frames=m['canonical_frames'],
                frame_ids=ids[lo:hi + 1].tolist(), status='LEGACY_VALUE', probability=.123))
        return dict(word_id=word, source_sha256='a' * 64, path=np.column_stack([np.arange(n), np.arange(n)]).tolist(),
            path_axes=['query_native_index', 'learned_canonical_index'], stages=stages, motion_observations=motions,
            source_components=[dict(status='PRESERVE_THIS')], partial_order_feedback=dict(unseen='preserve me'),
            additional_unknown_old_field=[1, 'keep', None])

    base = base_for('WORD1133')
    proposals = rt.make_proposals(base, p)
    check('target0 existing canonical source extent', proposals[0]['canonical_extent'] == [49, 64] and proposals[0]['indices'] == [49, 64])
    check('source partial relation remains partial', proposals[0]['source_constraint']['condition_relation'] == 'RELATED_ONLY' and proposals[0]['source_variant_applicability_verified'] is None)
    for word in ['WORD2523', 'WORD1567', 'WORD0739']:
        q = rt.make_proposals(base_for(word), p)
        check(word + ' other motions never source-clause truth', all(x['source_constraint'] is None and x['source_binding_relation'] == 'REFERENCE_OBSERVATION_ONLY' for x in q))
    wrong_scope = copy.deepcopy(base); wrong_scope['motion_observations'][0]['frame_ids'] = ids[48:65].tolist()
    rejects('old proposal cannot be widened', lambda: rt.make_proposals(wrong_scope, p))
    wrong_semantics = copy.deepcopy(base); wrong_semantics['path_axes'] = ['reference_frame', 'query_frame']
    rejects('swapped DTW axes rejected', lambda: rt.make_proposals(wrong_semantics, p))
    wrong_target = copy.deepcopy(base); wrong_target['motion_observations'][0]['target'] = 4
    rejects('word guide route mismatch rejected', lambda: rt.make_proposals(wrong_target, p))
    duplicate = copy.deepcopy(base); duplicate['local_event_feedback'] = {}
    rejects('existing local result cannot be replaced', lambda: rt.make_proposals(duplicate, p))

    def synthetic_attach(prepared, score, high_at=None, stage_ok=True):
        b = copy.deepcopy(base)
        if not stage_ok:
            b['stages'][0]['neural_stage_agreement'] = .1
        pp = copy.deepcopy(prepared)
        pp['input_identity'] = dict(clip_id='synthetic-job-only', word_id=b['word_id'], source_sha256='a' * 64, input_sha256='b' * 64)
        q = rt.make_proposals(b, pp)
        prob = np.full((n, 5), score, np.float32); prob[~pp['motion_anchor']] = .5
        if high_at is not None:
            prob[high_at, 0] = .95
        pred = dict(probabilities=prob, frame_id=ids.copy(), timestamp_us=ts.copy(),
            motion_anchor=pp['motion_anchor'], evidence_available=pp['evidence_available'], rgb_available=pp['available'])
        envelope = dict(**pp['input_identity'], AH_prediction_sha256='c' * 64,
            base_value_sha256=runtime.value_sha(b), proposals_value_sha256=runtime.value_sha(q),
            native_frame_ids=ids.tolist(), timestamp_us=ts.tolist())
        out = rt._adapter.attach(b, envelope, pp, pred, q, {0: runtime.composed_support(pp, q[0]['indices'])})
        check('old fields preserved #' + str(len(checks)), {k: v for k, v in out.items() if k != 'local_event_feedback'} == b)
        return out['local_event_feedback']['observations'][0]

    ob = synthetic_attach(p, .05, 53)
    check('observed local high evidence', ob['status'] == 'LOCAL_MOVEMENT_EVIDENCE_OBSERVED')
    check('source high has inherited stage consistency only', ob['source_binding'] == 'RELATED_OBSERVATION_WITH_STAGE_CONSISTENCY' and ob['guide_time_binding']['source_applicability_verified'] is None)
    ob = synthetic_attach(gp, .05, 53)
    check('high anchor preserved despite another gap', ob['status'] == 'LOCAL_MOVEMENT_EVIDENCE_OBSERVED' and not ob['scope_evidence_complete'])
    check('nonanchor displayed probability null', all(x['probability'] is None and x['raw_model_probability'] == .5 for x in ob['native_point_estimates'] if not x['motion_anchor']))
    ob = synthetic_attach(gp, .05)
    check('low plus gap stays insufficient', ob['status'] == 'INSUFFICIENT_OBSERVATION' and ob['source_binding'] == 'SOURCE_ASSOCIATION_ONLY_INSUFFICIENT_OBSERVATION')
    ob = synthetic_attach(bp, .05)
    check('all missing no probability confirmation', ob['probability'] is None and ob['anchor_count'] == 0 and ob['status'] == 'INSUFFICIENT_OBSERVATION')
    ob = synthetic_attach(p, .05)
    check('low observed means unconfirmed not error', ob['status'] == 'LOCAL_MOVEMENT_NOT_CONFIRMED' and ob['educational_error'] is None and not ob['correction_enabled'])
    check('source association not observed when no high', ob['source_binding'] == 'SOURCE_ASSOCIATION_ONLY_MOVEMENT_UNCONFIRMED')
    ob = synthetic_attach(p, .05, 53, False)
    check('high cannot override weak guide guard', ob['source_binding'] == 'RELATED_OBSERVATION_STAGE_UNCONFIRMED')
    check('semantic completion stays null', all(ob[k] is None for k in ['whole_clause_truth', 'whole_sign_correctness', 'complete_action', 'educational_error', 'required_repetitions']))
    rejects('private job prohibited in this public test unit', lambda: rt.validate_job(base, U / 'unused.npz', dict(source_role='private'), U))
    rejects('training permission never inferred', lambda: rt.validate_job(base, U / 'unused.npz', dict(source_role='public_runtime_inference', training=True, tuning=False, candidate_selection=False), U))
    tree = ast.parse((U / 'local_runtime.py').read_text(encoding='utf-8'))
    generic_imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module in ['data', 'train_local', 'common', 'engine', 'adapter', 'replay']]
    check('no historical generic module imports', not generic_imports)
    check('all checks avoided model load/inference', rt.model is None and rt.model_loads == rt.forwards == 0)
    check('generic worker modules remain untouched', sys.path == before_path and all(sys.modules.get(k) is v for k, v in generic.items()))
    report = dict(status='PASS_SYNTHETIC_LOCAL_RUNTIME_CONTRACT', checks_count=len(checks), checks=checks,
        model_loads=0, model_forwards=0, encoder_calls=0, source_videos_read=0, real_cache_inputs_read=0,
        training=0, new_DTW=0, actual_runtime_pass_claimed=False,
        pins={str(U / name): hashlib.sha256((U / name).read_bytes()).hexdigest() for name in ['local_runtime.py', 'check_local_runtime.py']})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
