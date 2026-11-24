"""Post-freeze hand-finger dependence diagnostic; never trains or selects a model.

--self-test: synthetic geometry/metric checks only (no artifact writes).
--check-ready: inspect completion/pins without model or source-clip access.
--run: once-only diagnostic in a NEW sibling handshape_probe/ directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

U = Path(__file__).resolve().parent
W = U.parent.parent
RUN = W / 'runs/stgcn_fullhands49_v1'
METHOD = W / 'runs/stgcn_valid_window_v1'
METHOD_UNIT = U.parent / 'stgcn_valid_window_pair_20261006'
MANIFEST = U.parent / 'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json'
OUT = U / 'handshape_probe'
PLAN_SHA = 'cbf4e9618c585579ac02e893c5927f9c4e7e1b6c202658801a984a96a037a35e'
MANIFEST_SHA = '84f273120b322bde05e945d4e5ee2d6c7ff2314c8d4b8cc889b85a0c000fd91f'
METHOD_SHA = '3ca59574a903b2e5ec4651b3d9b981767c97ce624904dba1cadfb878e9156834'
TCN_SHA = '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
IDS = ('ALDA', 'GAMSAHAMNIDA', 'GAM', 'GONGBU', 'INSA', 'JADA', 'JOESONGHADA',
       'JOTA', 'MAEKJU', 'MEOKDA', 'MOREUDA', 'MUL', 'NA', 'NEO', 'PPANG',
       'SARANG', 'SILTA', 'UNDONG')
IDS = tuple('NIKL_' + x + '_01' for x in IDS)
MODELS = ('candidate49', 'valid27', 'tcn832')
TCN_PINS = {
    W / 'tools_prep_union93_candidate.py': '614ae82d2bf4a992d395a0e28bbe73f634d3ead7ac8e484b664e3f420cb85529',
    W / 'ops/partial_encoder_pair_20260922/generated_v1.py': '443dd9cc0e4c1a59d36a9738b59899067faf59d56274407732407288b6cb09a6',
    W / 'ops/partial_encoder_pair_20260922/local_io.py': '4435393cd7d37c1e28601867aa4e06385b966c5a5895eecf8aa7a9862799b45a',
    W / 'ops/partial_encoder_pair_20260922/evaluate_pair.py': 'e4220a744122156d479e6136371d85f31e752da83efe288df9f477ab7befbc1e',
    W.parent / 'sign_model_pilot_v1/src/pilot/data.py': '5daa52beee310572c4298c74702fdabf08f0931ce38cff9f20201b341f1aa81d',
    W.parent / 'sign_model_pilot_v1/src/pilot/web_input.py': '727f6d77c9384b5b9c9d7b5aef35ce6239d4392f01e0b107c9a2fe46b569d07f',
    W.parent / 'sign_model_pilot_v1/src/pilot/model_masked_v1.py': 'fb8a963a76395fa6c6ac2281f84c968e672b4940a47c4ab932856f9c0d5f1828',
    W / 'src/wis_lab/core/observation.py': '88adc956aa2b9e2b34817ecbafe3fc7fe9ce70cd464d15804edd9cd5b89defff',
}


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(f'{value.dtype.str}|{value.shape}|'.encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def write_json(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_map(pins):
    for path, expected in pins.items():
        need(Path(path).is_file() and sha(path) == expected, f'Changed/missing pin: {path}')


def readiness():
    """No imports of runtime/model APIs and no source-clip reads before this gate."""
    required = [RUN / name for name in ('TRAINING_COMPLETION.json', 'FROZEN_MODEL.json',
                'EVALUATION_EXECUTION.json', 'EVALUATION_SCORES.npz', 'EVALUATION_ROWS.json',
                'EVALUATION_INPUT_DIAGNOSTICS.json', 'AUDIT_FINAL_METRICS.json')]
    required += [U / 'TRAINING_TRACE_AUDIT.json', U / 'EXECUTION_CONTEXT.json']
    missing = [str(p) for p in required if not p.is_file()]
    need(not missing, 'Not ready; no model or source access: ' + ', '.join(missing))
    need(sha(U / 'HANDSHAPE_EVALUATION_PLAN.md') == PLAN_SHA, 'Frozen probe plan changed')
    need(sha(MANIFEST) == MANIFEST_SHA, 'Source manifest changed')
    ctx, trace, audit = read(U / 'EXECUTION_CONTEXT.json'), read(U / 'TRAINING_TRACE_AUDIT.json'), read(RUN / 'AUDIT_FINAL_METRICS.json')
    need(trace['status'] == 'PASS' and trace['finalized'] is True, 'Training is not finalized PASS')
    need(audit['verification_passed'] is True, 'Word-score independent audit is incomplete')
    verify_map(ctx['library_pins'])
    verify_map(audit['sources'])
    verify_map(TCN_PINS)
    need(audit['model_sources']['valid27'] == METHOD_SHA, 'Wrong 27-point comparator')
    need(audit['model_sources']['tcn832'] == TCN_SHA, 'Wrong TCN comparator')
    need(sha(METHOD / 'final_model.npz') == METHOD_SHA, '27-point model file changed')
    ex = read(RUN / 'EVALUATION_EXECUTION.json')
    need(ex['model_state_unchanged'] and ex['no_training_or_selection'] and not ex['user_data_used'], 'Evaluation contract incomplete')
    need(ex['scores_sha256'] == sha(RUN / 'EVALUATION_SCORES.npz'), 'Evaluated scores changed')
    need(ex['final_model_sha256'] == audit['model_sources']['candidate49'], '49-point evaluated/frozen identity differs')
    method_ctx = read(METHOD_UNIT / 'EXECUTION_CONTEXT.json')
    verify_map(method_ctx['library_pins'])
    return ctx, method_ctx, audit


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics(original, changed, y):
    a, b = np.asarray(original, dtype=np.float64), np.asarray(changed, dtype=np.float64)
    need(a.shape == b.shape == (93,) and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid score vectors')
    # Stable class-index tie break matches prior sorted(..., key=-score) evaluation.
    oa, ob = np.argsort(-a, kind='stable'), np.argsort(-b, kind='stable')
    ra, rb = int(np.flatnonzero(oa == y)[0]) + 1, int(np.flatnonzero(ob == y)[0]) + 1
    ca, cb = a - a.mean(), b - b.mean()
    std = float(np.sqrt(np.mean(ca * ca)))
    ma = float(a[y] - np.max(np.delete(a, y)))
    mb = float(b[y] - np.max(np.delete(b, y)))
    return {'original_top1': int(oa[0]), 'transformed_top1': int(ob[0]),
            'original_word_rank': ra, 'transformed_word_rank': rb, 'rank_change': rb - ra,
            'top1_changed': bool(oa[0] != ob[0]), 'original_word_was_top1': bool(oa[0] == y),
            'original_word_retained': bool(ob[0] == y) if oa[0] == y else None,
            'original_logit_std': std,
            'relative_score_change_rms': float(np.sqrt(np.mean((cb - ca) ** 2)) / std) if std > 1e-12 else None,
            'original_word_margin_change_normalized': (mb - ma) / std if std > 1e-12 else None}


def self_test():
    import handshape_probe_geometry as geo
    result = geo.self_test()
    a = np.arange(93, dtype=np.float64)
    same = metrics(a, a.copy(), 92)
    need(same['relative_score_change_rms'] == 0 and same['rank_change'] == 0, 'Identity metric failed')
    offset = metrics(a, a + 41, 92)
    need(offset['relative_score_change_rms'] == 0, 'Class-common score shift changed relative metric')
    b = a.copy(); b[92] = -1
    changed = metrics(a, b, 92)
    scaled = metrics(a * 3.7, b * 3.7, 92)
    need(changed['top1_changed'] and changed['rank_change'] == 92, 'Rank metric failed')
    need(np.isclose(changed['relative_score_change_rms'], scaled['relative_score_change_rms']), 'Scale-normalized metric failed')
    need(metrics(np.zeros(93), np.zeros(93), 1)['relative_score_change_rms'] is None, 'Degenerate variance not marked undefined')
    return {'status': 'PASS', 'geometry': {k: v for k, v in result.items() if k != 'cases'}, 'metric_checks': 5, 'real_source_reads': 0,
            'model_inferences': 0, 'artifact_writes': 0}


def summary(records):
    out = {}
    for model in MODELS:
        out[model] = {}
        for kind in ('internal', 'rigid'):
            selected = [r for r in records if r['kind'] == kind]
            active_rows = [r for r in selected if r['models'][model]['applicable']]
            # Repeated rigid controls remain paired conditions, but are not extra
            # independent response evidence or extra weight in a video's mean.
            unique = {(r['clip_id'], r['models'][model]['input_key']): r for r in active_rows}
            applicable = [r['models'][model]['metrics'] for r in unique.values()]
            clip_means = []
            per_clip = {}
            for clip_id in IDS:
                clip_rows = [r for r in unique.values() if r['clip_id'] == clip_id]
                values = [r['models'][model]['metrics']['relative_score_change_rms'] for r in clip_rows]
                values = [v for v in values if v is not None]
                if values:
                    clip_means.append(float(np.mean(values)))
                per_clip[clip_id] = {'unique_applicable_inputs': len(clip_rows),
                    'top1_changed': sum(r['models'][model]['metrics']['top1_changed'] for r in clip_rows),
                    'mean_relative_score_change': float(np.mean(values)) if values else None}
            retained = [r for r in applicable if r['original_word_was_top1']]
            out[model][kind] = {'scheduled_conditions': len(selected), 'applicable_conditions': len(active_rows),
                'not_applicable_conditions': len(selected) - len(active_rows), 'unique_applicable_inputs': len(unique),
                'unique_input_top1_changed': sum(m['top1_changed'] for m in applicable),
                'unique_original_top1_inputs': len(retained),
                'unique_original_word_retained': sum(m['original_word_retained'] for m in retained),
                'clip_equal_weight_mean_relative_score_change': float(np.mean(clip_means)) if clip_means else None,
                'clips_in_relative_score_mean': len(clip_means), 'per_clip': per_clip}
    return out


def run():
    ctx, ctx27, word_audit = readiness()
    need(not OUT.exists(), 'handshape_probe/ already exists; preserve partial/completed outputs for review')
    OUT.mkdir()  # exclusive directory creation: no resume, overwrite or result-driven rerun
    source_pins = {str(p): h for p, h in TCN_PINS.items()}
    source_pins.update({str(U / name): sha(U / name) for name in ('handshape_probe.py', 'handshape_probe_geometry.py', 'HANDSHAPE_EVALUATION_PLAN.md')})
    source_pins.update({str(p): sha(p) for p in (MANIFEST, U / 'EXECUTION_CONTEXT.json',
        U / 'TRAINING_TRACE_AUDIT.json', RUN / 'AUDIT_FINAL_METRICS.json', RUN / 'EVALUATION_SCORES.npz',
        METHOD / 'EVALUATION_SCORES.npz', METHOD / 'FROZEN_MODEL.json')})
    write_json('STARTED.json', {'started_utc': now(), 'plan_sha256': PLAN_SHA, 'code_and_evidence_pins': source_pins,
        'no_training_or_selection': True, 'no_original_input_inference': True, 'scheduled_variants': 720})
    try:
        _execute(ctx, ctx27, word_audit, source_pins)
    except BaseException as exc:
        write_json('FAILURE.json', {'status': 'FAIL_CLOSED', 'at_utc': now(), 'type': type(exc).__name__,
            'message': str(exc), 'instruction': 'Preserve this directory; no automatic rerun/resume or threshold change.'})
        raise


def _execute(ctx, ctx27, word_audit, source_pins):
    import torch
    import handshape_probe_geometry as geo
    import runtime_binding as rb
    need(Path(rb.__file__).resolve() == U / 'runtime_binding.py', 'Wrong runtime binding imported')
    rb.configure_cpu()
    rb.trace_pass(execute=False)
    frozen49 = rb.gw.validate_frozen(RUN, ctx)
    frozen27 = rb.gw.validate_frozen(METHOD, ctx27)
    need(frozen49['final_model_sha256'] == word_audit['model_sources']['candidate49'], '49-point freeze differs')
    need(frozen27['final_model_sha256'] == METHOD_SHA, '27-point freeze differs')
    plan, cfg, source_meta, factory49, _, tf = rb.setup()
    need(sha(rb.P['baseline_checkpoint']) == TCN_SHA, 'TCN checkpoint changed')
    sys.path.insert(0, str(W))
    import tools_prep_union93_candidate as tcn_source
    from pilot.data import build_channels
    from pilot.web_input import to_native_like, active_span
    tcn_eval = load_module('handshape_pinned_prefix_api', W / 'ops/partial_encoder_pair_20260922/generated_v1.py')
    # Import origins are checked as well as source-file hashes.
    for obj, expected in ((tcn_source, W / 'tools_prep_union93_candidate.py'),
                          (sys.modules['pilot.data'], W.parent / 'sign_model_pilot_v1/src/pilot/data.py'),
                          (sys.modules['pilot.web_input'], W.parent / 'sign_model_pilot_v1/src/pilot/web_input.py')):
        need(Path(obj.__file__).resolve() == expected.resolve(), 'Imported TCN API from wrong directory')
    manifest, rows_doc = read(MANIFEST), read(RUN / 'EVALUATION_ROWS.json')
    rows = [r for r in manifest['rows'] if r['group'] == 'public' and r['condition'] == 'intact']
    need(tuple(r['id'] for r in rows) == IDS and len(rows) == 18, 'Public reference list changed')
    classes = rows_doc['classes']
    need(classes == ctx['classes'] == ctx27['classes'] == manifest['classes'], 'Class order differs')
    all_rows = rows_doc['rows']
    indices = [next(i for i, r in enumerate(all_rows) if r == row) for row in rows]
    with np.load(RUN / 'EVALUATION_SCORES.npz', allow_pickle=False) as archive:
        originals = {'candidate49': archive['candidate_scores'][indices].copy(),
                     'valid27': archive['old_stgcn_scores'][indices].copy(),
                     'tcn832': archive['baseline_scores'][indices].copy()}
        need(np.array_equal(archive['y'][indices], [r['y'] for r in rows]), 'Original label ordering mismatch')
    with np.load(METHOD / 'EVALUATION_SCORES.npz', allow_pickle=False) as archive:
        need(np.array_equal(originals['valid27'], archive['candidate_scores'][indices]), 'Stored 27-point scores differ')
        need(np.array_equal(originals['tcn832'], archive['baseline_scores'][indices]), 'Stored TCN scores differ')
    diags49 = read(RUN / 'EVALUATION_INPUT_DIAGNOSTICS.json')
    diags27 = read(METHOD / 'EVALUATION_INPUT_DIAGNOSTICS.json')
    prepare49 = rb.make_prepare(tf)
    norm = tf.CenterAndScaleNormalize(reference_points_preset=rb.ra.NORMALIZE_PRESET, frame_level=False)

    def prepare27(xy, observed):
        return rb.gp.prepare_windows(xy, observed, rb.ra.LAYOUTS['mp75'][1], [3, 4], 60, 10,
                                     lambda t: norm({'frames': t})['frames'])

    def tcn_input(xy, usable, stamps, span):
        nat, present = to_native_like(xy, usable)
        need(active_span(present) == span, 'TCN time span changed despite fixed mask')
        lo, hi = span
        return build_channels(nat[lo:hi + 1], present[lo:hi + 1], stamps[lo:hi + 1])['x']

    prepared = {m: {} for m in MODELS}
    score_cache = {m: {} for m in MODELS}
    condition_rows, identity_rows, raw_pins = [], [], {}

    def key_and_store(model, x, valid, original_score=None):
        x = np.ascontiguousarray(x)
        key = array_sha(x) + (':' + array_sha(valid) if valid is not None else '')
        if original_score is not None:
            if key in score_cache[model]:
                need(np.array_equal(score_cache[model][key], original_score), 'Same original input has different cached scores')
            score_cache[model][key] = original_score.copy()
            # A variant encountered earlier can equal a later original clip.
            # Original scores take precedence; never re-infer that input.
            prepared[model].pop(key, None)
        elif key not in score_cache[model] and key not in prepared[model]:
            prepared[model][key] = (x.copy(), valid.copy() if valid is not None else None)
        return key

    for clip_index, (row, evaluation_index) in enumerate(zip(rows, indices)):
        path = Path(row['path'])
        need(row['layout'] == 'mp75' and row['mask_hand'] is None, 'Unexpected public input layout/hand mask')
        need('/runs/ses_' not in str(path).replace('\\', '/').lower(), 'User session data prohibited')
        need(sha(path) == row['sha256'], 'Public source changed')
        raw_pins[str(path)] = row['sha256']
        with np.load(path, allow_pickle=False) as archive:
            full = {k: archive[k].copy() for k in ('xy_px', 'observed', 'usable', 'timestamp_us')}
        need(full['xy_px'].dtype == np.float32, 'Unexpected original coordinate dtype; do not silently cast raw geometry')
        for mask_name in ('observed', 'usable'):
            need(full[mask_name].dtype.kind in 'bui' and np.isin(full[mask_name], (0, 1)).all(), 'Invalid stored visibility mask')
        full_xy = full['xy_px'].astype(np.float32)
        observed = full['observed'].astype(bool)
        usable = full['usable'].astype(bool)
        times = full['timestamp_us'].astype(np.int64)
        need(full_xy.shape[1:] == (75, 2) and observed.shape == usable.shape == full_xy.shape[:2], 'Raw input shape mismatch')
        need(np.isfinite(full_xy[observed | usable]).all(), 'Nonfinite observed/usable coordinate')
        need(np.isfinite(full['timestamp_us']).all() and (np.diff(times) > 0).all(), 'Timestamp contract violated')
        lo, end = row['start'], row['end_exclusive']
        xy, mask = full_xy[lo:end].copy(), observed[lo:end].copy()
        raw_before = {k: array_sha(v) for k, v in full.items()}
        x49b, d49 = prepare49(xy, mask, 'mp75'); x49 = x49b[0]
        x27b, d27 = prepare27(xy, mask); x27 = x27b[0]
        canonical49, canonical_diag = rb.gp.load_row(row, source_meta['layout_specs'], prepare49)
        need(torch.equal(x49, canonical49), '49-point direct/canonical identity mismatch')
        need(rb.gp.tensor_sha256(x49) == diags49[evaluation_index]['input_tensor_sha256'], '49-point saved input differs')
        need(rb.gp.tensor_sha256(x27) == diags27[evaluation_index]['input_tensor_sha256'], '27-point saved input differs')
        need(torch.equal(x49[..., list(rb.fullhands49.OLD_TO_NEW)], x27), 'Common27 original coordinates differ')
        valid = rb.window_bridge.window_valid_from_indices(d49['indices']).numpy()
        need(d49['indices'] == d27['indices'], 'ST-GCN original time indices differ')
        sample_indices = np.asarray([i for i in d49['indices'] if i >= 0], dtype=np.int64)
        nat, present = to_native_like(full_xy, usable)
        span = active_span(present)
        need(span is not None, 'TCN has no usable time span')
        xt = tcn_input(full_xy, usable, times, span)
        old_nat, old_present, old_times = tcn_source._native_from_nikl(str(path), row['sha256'])
        need(np.array_equal(xt, build_channels(old_nat, old_present, old_times)['x']), 'TCN original source-route identity failed')
        base_x = {'candidate49': x49.numpy(), 'valid27': x27.numpy(), 'tcn832': xt}
        base_keys = {m: key_and_store(m, base_x[m], valid if m != 'tcn832' else None,
                                    originals[m][clip_index]) for m in MODELS}
        identity_rows.append({'clip_id': row['id'], 'source_sha256': row['sha256'],
            'evaluation_index': evaluation_index, 'slice': [lo, end], 'tcn_span': [span[0], span[1] + 1],
            'original_input_keys': base_keys, 'original_scores_reused': True,
            'stgcn_saved_tensor_identity': True, 'tcn_original_preprocess_identity': True})
        for hand in ('left', 'right'):
            offset = 33 if hand == 'left' else 54
            for finger, info in geo.FINGERS.items():
                for degrees in (-15.0, 15.0):
                    pair_id = f"{row['id']}:{hand}:{finger}:{degrees:+g}"
                    pair_eligibility = None
                    for kind in ('internal', 'rigid'):
                        changed_xy, eligible, geometry = geo.transform(xy, mask, hand, finger, degrees, kind)
                        if pair_eligibility is None:
                            pair_eligibility = eligible.copy()
                        need(np.array_equal(pair_eligibility, eligible), 'Paired eligibility differs')
                        a49b, ad49 = prepare49(changed_xy, mask, 'mp75'); a49 = a49b[0]
                        a27b, ad27 = prepare27(changed_xy, mask); a27 = a27b[0]
                        need(ad49['indices'] == d49['indices'] == ad27['indices'], 'Transformation changed ST-GCN time')
                        need(ad49['shoulder_distance'] == d49['shoulder_distance'], 'Transformation changed body scale')
                        need(torch.equal(a49[..., list(rb.fullhands49.OLD_TO_NEW)], a27), 'Common27 transformed inputs differ')
                        # A tip can coincide with its pivot in a folded projection. Do not
                        # call an added-point-only response a fair common-information result.
                        tip = offset + info['chain'][-1]
                        tip_changed = np.any(changed_xy[:, tip] != xy[:, tip], axis=1) & eligible
                        st_common_effect = bool(tip_changed[sample_indices].any())
                        changed_full = full_xy.copy(); changed_full[lo:end] = changed_xy
                        at = tcn_input(changed_full, usable, times, span)
                        need(np.array_equal(at[:, 186:], xt[:, 186:]), 'Transformation changed TCN validity/dt channels')
                        tcn_edit_frames = np.arange(lo, end)[tip_changed]
                        tcn_common_effect = bool(((tcn_edit_frames >= span[0]) & (tcn_edit_frames <= span[1])).any())
                        xs = {'candidate49': a49.numpy(), 'valid27': a27.numpy(), 'tcn832': at}
                        model_rows = {}
                        for model in MODELS:
                            common_effect = tcn_common_effect if model == 'tcn832' else st_common_effect
                            actual_changed = not np.array_equal(xs[model], base_x[model])
                            applicable = bool(eligible.any() and common_effect and actual_changed)
                            k = key_and_store(model, xs[model], valid if model != 'tcn832' else None) if applicable else None
                            model_rows[model] = {'applicable': applicable,
                                'status': 'APPLICABLE' if applicable else 'NOT_APPLICABLE',
                                'reason': None if applicable else ('no_eligible_frame' if not eligible.any() else
                                          'no_retained_tip_effect_in_time_span' if not common_effect else 'no_prepared_input_change'),
                                'prepared_input_changed': actual_changed, 'common_tip_changed_in_time_span': common_effect,
                                'input_key': k, 'input_shape': list(xs[model].shape)}
                        condition_rows.append({'condition_id': pair_id + ':' + kind, 'pair_id': pair_id,
                            'clip_id': row['id'], 'clip_index': clip_index, 'label': row['label'], 'y': int(row['y']),
                            'hand': hand, 'finger': finger, 'degrees': degrees, 'kind': kind,
                            'source_sha256': row['sha256'], 'slice': [lo, end], 'geometry': geometry,
                            'stgcn_sampled_eligible_frames': int(eligible[sample_indices].sum()),
                            'common_applicable_all_models': all(m['applicable'] for m in model_rows.values()),
                            'models': model_rows})
        need({k: array_sha(v) for k, v in full.items()} == raw_before, 'Loaded source arrays were mutated')
        need(sha(path) == row['sha256'], 'Raw source file changed during preparation')
        print(json.dumps({'stage': 'geometry_prepared_no_model_inference', 'completed_reference_clips': clip_index + 1,
                          'of': 18}), flush=True)
    need(len(condition_rows) == 720 and len(identity_rows) == 18, 'Frozen condition counts differ')
    write_json('INPUT_AUDIT.json', {'status': 'PASS', 'identity_rows': identity_rows,
        'scheduled_conditions': 720, 'unique_new_inputs': {m: len(prepared[m]) for m in MODELS},
        'original_inferences': 0, 'source_sha256': raw_pins,
        'all_original_inputs_match_previous_routes': True, 'common27_exact': True,
        'masks_times_protected': True, 'condition_geometry_checks': 'PASS'})
    write_json('INPUT_CONDITIONS.json', condition_rows)
    verify_map(source_pins); verify_map(raw_pins)

    # Only now, after every identity/geometry condition has passed, build/load models.
    model49 = rb.gw.load_final_state(factory49(), RUN, ctx)
    dpc, _, _ = rb.ra.load_reference(rb.P['reference_sources'], plan['pins']['source_pins'])
    art = Path(rb.P['artifact_dir'])
    tensors, _ = rb.stgcn_core.read_encoder_tensors(art / 'raw_dpc.zip', art / 'TENSOR_METADATA.json',
                                                   plan['pins']['pretrained_zip'], encoder_prefix='model.')
    enc = {k[len('conv_encoder.'):]: v for k, v in tensors.items() if k.startswith('conv_encoder.')}
    gru = {k[len('agg.'):]: v for k, v in tensors.items() if k.startswith('agg.')}
    model27, _ = rb.ra.make_reference_model(dpc, enc, gru, num_classes=93, seed=cfg['seed'])
    model27 = rb.window_bridge.bind_window_forward(model27, rb.valid_window_forward.forward_valid_windows)
    model27 = rb.gw.load_final_state(model27, METHOD, ctx27)
    payload = torch.load(rb.P['baseline_checkpoint'], map_location='cpu', weights_only=True)
    model_tcn, tcn_classes = tcn_source.rebuild_from_payload(payload)
    need(tcn_classes == classes, 'TCN class order differs')
    need(Path(sys.modules['pilot.model_masked_v1'].__file__).resolve() ==
         (W.parent / 'sign_model_pilot_v1/src/pilot/model_masked_v1.py').resolve(), 'Wrong TCN model API imported')
    models = {'candidate49': model49, 'valid27': model27, 'tcn832': model_tcn}
    before = {m: rb.gw.state_digest(v.state_dict()) for m, v in models.items()}
    write_json('INFERENCE_STARTED.json', {'at_utc': now(), 'model_state_sha256': before,
        'model_files': word_audit['model_sources'], 'unique_new_input_counts': {m: len(prepared[m]) for m in MODELS},
        'original_input_inferences': 0, 'selection_or_training': False})
    inference_counts = {}
    for name, model in models.items():
        model.eval()
        keys = list(prepared[name])
        need(not (set(keys) & set(score_cache[name])), 'Queued input already has a reused score')
        inference_counts[name] = len(keys)
        if keys:
            if name == 'tcn832':
                clips = [prepared[name][k][0] for k in keys]
                prefix = tcn_eval.extract_prefix(model, clips)
                prediction = tcn_eval.predict_prefix(model, prefix, batch_size=32)
                # Preserve the original evaluator's pooled-float32 + affine-float64 convention.
                pooled = prediction['pooled'].astype(np.float64)
                scores = pooled @ payload['state_dict']['head.weight'].numpy().astype(np.float64).T
                scores += payload['state_dict']['head.bias'].numpy().astype(np.float64)
            else:
                x = torch.from_numpy(np.stack([prepared[name][k][0] for k in keys]))
                valid = torch.from_numpy(np.stack([prepared[name][k][1] for k in keys]))
                scores = rb.gw.score_batches(model, x, 16, window_valid=valid)
            need(scores.shape == (len(keys), 93) and np.isfinite(scores).all(), 'Invalid transformed scores')
            score_cache[name].update({k: score.copy() for k, score in zip(keys, scores)})
        need(rb.gw.state_digest(model.state_dict()) == before[name], 'Frozen model state changed')
    arrays = {}
    for model in MODELS:
        keys = list(score_cache[model])
        arrays[model + '_input_keys'] = np.asarray(keys)
        arrays[model + '_scores'] = np.stack([score_cache[model][k] for k in keys])
        arrays[model + '_original_scores'] = originals[model]
    for row in condition_rows:
        for model in MODELS:
            d = row['models'][model]
            if d['applicable']:
                d['metrics'] = metrics(originals[model][row['clip_index']], score_cache[model][d['input_key']], row['y'])
            else:
                d['metrics'] = None
    by_pair = defaultdict(dict)
    for row in condition_rows:
        by_pair[row['pair_id']][row['kind']] = row
    pairs = []
    for pair_id, arms in by_pair.items():
        entry = {'pair_id': pair_id, 'clip_id': arms['internal']['clip_id'], 'models': {}}
        for model in MODELS:
            a, b = arms['internal']['models'][model], arms['rigid']['models'][model]
            active = a['applicable'] and b['applicable']
            va = a['metrics']['relative_score_change_rms'] if active else None
            vb = b['metrics']['relative_score_change_rms'] if active else None
            entry['models'][model] = {'both_applicable': bool(active),
                'internal_minus_rigid_relative_score_change': va - vb if va is not None and vb is not None else None,
                'interpretation': 'descriptive response contrast; not a quality or handshape-correctness score'}
        pairs.append(entry)
    verify_map(source_pins); verify_map(raw_pins)
    verify_map(ctx['library_pins']); verify_map(ctx27['library_pins'])
    with (OUT / 'SCORES.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_json('CONDITIONS.json', condition_rows)
    write_json('PAIRED_RESPONSES.json', pairs)
    write_json('RESULT.json', {'status': 'COMPLETED_PENDING_INDEPENDENT_REVIEW', 'completed_utc': now(),
        'plan_sha256': PLAN_SHA, 'source_pins': source_pins, 'model_state_unchanged': True,
        'original_conditions': 18, 'scheduled_internal_conditions': 360, 'scheduled_rigid_conditions': 360,
        'unique_new_input_inferences': inference_counts, 'original_input_inferences': 0,
        'new_training': 0, 'candidate_selection': False, 'user_data_used': False,
        'summary': summary(condition_rows), 'classes': classes, 'code_sha256': sha(__file__),
        'output_sha256': {n: sha(OUT / n) for n in ('INPUT_AUDIT.json', 'INPUT_CONDITIONS.json', 'SCORES.npz', 'CONDITIONS.json', 'PAIRED_RESPONSES.json')},
        'limits': ['Public training references, not unseen-person validation.',
                   'No per-recording expert handshape labels: conditional dependence, not shape accuracy.',
                   'Two-dimensional rotations need not be anatomically/linguistically valid.',
                   'Rigid-hand rotation changes orientation and is not a word-invariance requirement.',
                   'Same angles produce different input displacement magnitudes; greater response is not better.',
                   'Eligibility transitions can introduce artificial temporal variation.',
                   'Observed ST-GCN and usable TCN masks/time policies stay different by the original contract.',
                   'No model adoption or quality PASS/FAIL is derived from this diagnostic.']})
    print(json.dumps({'status': 'COMPLETED_PENDING_INDEPENDENT_REVIEW', 'output': str(OUT),
                      'new_inferences': inference_counts, 'original_inferences': 0}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--check-ready', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
    elif args.check_ready:
        try:
            readiness()
        except (ValueError, FileNotFoundError) as exc:
            print(json.dumps({'status': 'NOT_READY', 'reason': str(exc), 'model_inferences': 0,
                              'source_clip_reads': 0, 'writes': 0}, ensure_ascii=False))
            raise SystemExit(2)
        print(json.dumps({'status': 'READY', 'model_inferences': 0, 'source_clip_reads': 0, 'writes': 0}))
    else:
        run()


if __name__ == '__main__':
    main()
