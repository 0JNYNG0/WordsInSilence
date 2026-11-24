"""Add observed body-progression support to immutable L selections.

Draft until root review. No former model, rank head, encoder or semantic head is
loaded. The new pointwise head runs once per existing scenario/seed, with its
output shared by both preselected L conditions. Old extents and semantics are
copied exactly; this is a different readout, not a like-for-like accuracy delta.
"""
from pathlib import Path
import ast
import collections
import copy
import datetime
import hashlib
import importlib.util
import itertools
import json
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
M = Path(__file__).resolve().parent
H = M.parent
G = H.parent
W = G / 'candidate_description_supervision'
L = H / 'full_bag_rate_rank_learning'
C5 = H / 'observed_action_anchor_design'
E = M / 'evaluation'
PREVIOUS = L / 'evaluation/COMPARISON.json'
SEMANTICS = L / 'changed_span_shadow/RESULTS.json'
CONTRACT = E / 'CONTRACT_DRAFT.json'
FIT_REVIEW = M / 'SAVED_FIT_VERIFICATION.json'
MODES = ['full_bag_rate_rank', 'old_local_full_plus_fullbag_half']
FAMILIES = [2, 3, 4, 5, 7]
AX = {2: 0, 3: 1, 4: 2, 5: 4, 7: 3}
SEEDS = [42, 43, 44]


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def key(row):
    return row['seed'], row['clip_id'], row['kind']


def output_path(path):
    path = Path(path).resolve()
    assert path.is_relative_to(E.resolve()), path
    return path


def write(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_npz(path, **arrays):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def stop():
    for directory in [E, M, H, G]:
        for name in ['STOP', 'STOP.txt', 'STOP_REQUESTED', 'STOP_REQUESTED.json']:
            assert not (directory / name).exists(), directory / name


def check(pins):
    for path, digest in pins.items():
        assert sha(path) == digest, path


def pin(pins, path, expected=None):
    path = str(Path(path).resolve())
    digest = sha(path)
    assert expected is None or digest == expected, path
    assert path not in pins or pins[path] == digest, path
    pins[path] = digest


def protected_match(protected):
    assert isinstance(protected, dict) and len(protected) == 477
    assert protected == read(G / 'PROTECTIONS.json')
    assert protected == read(M / 'INPUTS_COMPLETE.json')['protected_pins']
    check(protected)


def fit_review(pins):
    review = read(FIT_REVIEW)
    assert review['status'] == 'SAVED_FIT_METRICS_SEPARATELY_VERIFIED_NOT_GENERALIZATION_OR_ADOPTION'
    assert review['protected_hashes_verified'] == 477
    assert review['fit_sources'] == 24 and review['fit_sequences'] == 144
    assert review['new_inference'] is False and review['training_rerun'] is False
    assert review['probe_arrays_opened'] == 0
    pin(pins, FIT_REVIEW)
    pin(pins, M / 'INPUTS_COMPLETE.json', review['input_sha256'])
    pin(pins, M / 'TRAINING_COMPLETE.json', review['training_complete_sha256'])
    pin(pins, M / 'PRETRAIN_VERIFICATION.json', review['pretrain_sha256'])
    pin(pins, M / 'verify_training_saved.py', review['verifier_sha256'])
    trained = {r['seed']: r for r in read(M / 'TRAINING_COMPLETE.json')['rows']}
    assert len(review['runs']) == 3 and {r['seed'] for r in review['runs']} == set(SEEDS)
    for run in review['runs']:
        assert run['model_sha256'] == trained[run['seed']]['model_sha256']
        assert run['predictions_sha256'] == trained[run['seed']]['predictions_sha256']
        assert run['normalization_exactly_unchanged']
        provenance = run['by_provenance']
        assert provenance['original_direct_motion']['count'] == 822
        assert provenance['original_direct_motion']['new_correct'] == 822
        assert provenance['own_terminal_body_hold']['count'] == 522
        assert provenance['other_family_terminal_body_hold']['count'] == 126
        assert sum(v['regressions'] for v in provenance.values()) == 0
        assert len(run['by_family_class']) == 10 and len(run['by_source']) == 24
    # Keep source/family breakdown and the remaining old failures, not only gains.
    return review


def load_pure_contract():
    """Use the actual body_context AST without importing old model modules."""
    source = W / 'source_boundary_core.py'
    tree = ast.parse(source.read_text('utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'body_context')
    framing = ast.parse((W / 'framing_boundary_core.py').read_text('utf-8'))
    halo = next(ast.literal_eval(n.value) for n in framing.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'HALO_US' for t in n.targets))
    assert halo == 133333
    core = ast.parse((W / 'core.py').read_text('utf-8'))
    values = {t.id: ast.literal_eval(n.value) for n in core.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ['F', 'AX']}
    assert values == {'F': FAMILIES, 'AX': AX}
    namespace = {'np': np, 'HALO_US': halo}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    spec = importlib.util.spec_from_file_location('_body_progress_order_contract', C5 / 'progress_order.py')
    order = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order)
    return namespace['body_context'], order


def observed(q):
    return np.stack([q[:, AX[f] * 2, 8] > .5 for f in FAMILIES], axis=1)


def context_input(q, ts, body_context):
    assert q.ndim == 3 and q.shape[1:] == (10, 10) and np.isfinite(q).all()
    assert ts.shape == (len(q),) and np.issubdtype(ts.dtype, np.integer)
    assert len(ts) >= 2 and (np.diff(ts) > 0).all()
    x = body_context(q, ts)
    current = np.concatenate([q[..., :4], q[..., 8:9]], -1).reshape(-1, 50)
    assert x.shape == (len(ts), 150) and x.dtype == np.float32 and np.isfinite(x).all()
    assert np.array_equal(x[:, 50:100], current.astype(np.float32))
    mask = observed(q)
    current_mask = x[:, 50:100].reshape(-1, 10, 5)[:, :, 4]
    assert np.array_equal(mask, np.stack([current_mask[:, AX[f] * 2] > .5 for f in FAMILIES], 1))
    return x, mask


def training_parity(pins, body_context):
    """No model inference. Check saved FIT x, masks, PTS and feature ordering.

    All 144 new FIT sequences are exact copies of the matching old sequence x.
    Their current body block reconstructs the full before/current/after context.
    For the original 16 sources (96 sequences), additionally reconstruct directly
    from immutable full/upper z and actual_crop q caches used by the original
    prepare_source_boundary. Added-source normalized q is not re-extracted.
    """
    inputs_path = M / 'INPUTS_COMPLETE.json'
    boundary_path = W / 'source_boundary_candidate_v3/INPUTS_COMPLETE.json'
    ready_path = G.parent / 'word100_shared_motion_learning_20261109/READY.json'
    pin(pins, inputs_path)
    pin(pins, boundary_path)
    pin(pins, ready_path)
    meta = read(inputs_path)
    boundary = read(boundary_path)
    sequences = meta['sequences']
    assert len(sequences) == 144 and meta['family_order'] == FAMILIES
    assert {r['person'] for r in sequences} == {2, 3, 4, 5, 6, 7}
    assert all(r['split'] == 'train' for r in sequences)
    original_sources = {r['clip_id'] for r in boundary['rows'] if r['split'] == 'train'}
    raw_sources = {r['clip_id']: r for r in read(ready_path)['rows']
                   if r['role'] == 'fit' and r['clip_id'] in original_sources}
    assert set(raw_sources) == original_sources and len(original_sources) == 16
    caches = {}
    for row in boundary['rows']:
        if row['split'] != 'train':
            continue
        k = row['clip_id'], row['view']
        # Augmented candidate rows omit raw/raw_sha256. The original preparer
        # gets time from READY by clip_id, not from every candidate row.
        raw = raw_sources[row['clip_id']]
        if 'raw' in row:
            assert row['raw'] == raw['path'] and row['raw_sha256'] == raw['sha256']
        descriptor = dict(cache=row['cache'], cache_sha256=row['cache_sha256'],
                          raw=raw['path'], raw_sha256=raw['sha256'])
        assert k not in caches or caches[k] == descriptor
        caches[k] = descriptor
    rows = []
    for sequence in sequences:
        stop()
        pin(pins, sequence['path'], sequence['sha256'])
        pin(pins, sequence['original_path'], sequence['original_sha256'])
        with np.load(sequence['path'], allow_pickle=False) as archive:
            x, ts, mask = [archive[k].copy() for k in ['x', 'timestamp_us', 'observed']]
            supervised = archive['mask'].copy()
        assert mask.dtype == np.bool_
        with np.load(sequence['original_path'], allow_pickle=False) as original:
            assert np.array_equal(x, original['x'])
        current = x[:, 50:100].reshape(-1, 10, 5)
        q = np.zeros((len(x), 10, 10), dtype=x.dtype)
        q[..., :4], q[..., 8] = current[..., :4], current[..., 4]
        rebuilt, rebuilt_mask = context_input(q, ts, body_context)
        assert np.array_equal(x, rebuilt), ('FIT context mismatch', sequence)
        assert np.array_equal(mask, rebuilt_mask) and not ((supervised > 0) & ~mask).any()
        raw_checked = False
        cache = caches.get((sequence['clip_id'], sequence['view']))
        if cache is not None:
            pin(pins, cache['cache'], cache['cache_sha256'])
            pin(pins, cache['raw'], cache['raw_sha256'])
            with np.load(cache['cache'], allow_pickle=False) as cached:
                original_q = (cached['q'].reshape(-1, 10, 10) if sequence['view'] == 'actual_crop'
                              else cached['z'][:, 292:392].reshape(-1, 10, 10)).copy()
            with np.load(cache['raw'], allow_pickle=False) as raw:
                original_ts = raw['timestamp_us'].copy()
                assert np.array_equal(raw['frame_id'], np.arange(len(original_ts)))
            if sequence['reverse']:
                original_q = original_q[::-1].copy()
                original_ts = original_ts[-1] - original_ts[::-1]
            raw_x, raw_mask = context_input(original_q, original_ts, body_context)
            assert np.array_equal(ts, original_ts) and np.array_equal(x, raw_x)
            assert np.array_equal(mask, raw_mask)
            raw_checked = True
        rows.append(dict(clip_id=sequence['clip_id'], view=sequence['view'], reverse=sequence['reverse'],
                         frames=len(ts), copied_training_x_exact=True, context_order_exact=True,
                         family_mask_exact=True, direct_original_cache_checked=raw_checked))
    assert len(rows) == 144 and sum(r['direct_original_cache_checked'] for r in rows) == 96
    return dict(status='FIT_CONTEXT_MASK_AND_ORDER_EXACT', sequences=144,
                original_cache_sequences=96, added_source_saved_context_sequences=48,
                added_source_limit='Saved normalized current body block reused; no pose/encoder replay or new normalization.',
                no_probe_arrays=True, no_model_forward=True, rows=rows)


def validate_old(previous, semantics, scenarios):
    expected = {(seed, s['clip_id'], s['kind']) for s in scenarios for seed in SEEDS}
    assert len(scenarios) == 82 and len(expected) == 246
    maps = []
    for rows in [previous['rows'], previous['predictions'], semantics['rows']]:
        mapping = {key(r): r for r in rows}
        assert len(rows) == len(mapping) == 246 and set(mapping) == expected
        maps.append(mapping)
    rm, pm, sm = maps
    for scenario in scenarios:
        assert '_REAL11_' in scenario['clip_id'] or '_REAL12_' in scenario['clip_id'] or '_REAL13_' in scenario['clip_id']
        for seed in SEEDS:
            k = seed, scenario['clip_id'], scenario['kind']
            assert pm[k]['source'] == scenario['source'] and pm[k]['source_sha256'] == scenario['source_sha256']
            for mode in MODES:
                cards = rm[k]['conditions'][mode]['cards']
                semantic_cards = sm[k]['conditions'][mode]
                assert [(c['card'], c['chosen']['span'] if c['chosen'] else None) for c in cards] == [
                    (c['card'], c['span']) for c in semantic_cards]
                assert all(c['family'] in FAMILIES for c in cards)
                assert len({c['card'] for c in cards}) == len(cards)
                assert all(c['completion'] is None for c in semantic_cards)
                assert all(c['semantic_completion'] is None for c in cards)
    return rm, pm, sm


def freeze():
    """Run only after root has reviewed this draft; no model forward."""
    stop()
    assert not (E / 'FROZEN.json').exists() and not (E / 'INPUT_PARITY.json').exists()
    contract = read(CONTRACT)
    assert contract['conditions'] == MODES and contract['seeds'] == SEEDS
    assert contract['head_forward_calls'] == 246
    pins = {}
    protected = read(G / 'PROTECTIONS.json')
    protected_match(protected)
    for path in [Path(__file__), CONTRACT, M / 'model.py', M / 'prepare.py', M / 'train.py',
                 G / 'PROTECTIONS.json',
                 PREVIOUS, SEMANTICS, L / 'evaluation/FROZEN.json', L / 'DECISION.json',
                 W / 'source_boundary_core.py', W / 'framing_boundary_core.py', W / 'core.py',
                 W / 'prepare_source_boundary.py', W / 'prepare_rank_learning.py',
                 C5 / 'progress_order.py', H / 'G2_REFERENCE_LINEAGE_REVIEW.json']:
        pin(pins, path)
    fit = fit_review(pins)
    body_context, _ = load_pure_contract()
    parity = training_parity(pins, body_context)
    previous, semantics = read(PREVIOUS), read(SEMANTICS)
    assert read(L / 'DECISION.json')['G2_pass'] is False
    scenarios = read(L / 'evaluation/FROZEN.json')['scenarios']
    validate_old(previous, semantics, scenarios)
    for scenario in scenarios:
        pin(pins, scenario['source'], scenario['source_sha256'])
    # Saved previous outputs are hash-pinned, not rerun or used as head inputs.
    for prediction in previous['predictions']:
        pin(pins, prediction['path'], prediction['sha256'])
    check(pins)
    protected_match(protected)
    write(E / 'INPUT_PARITY.json', dict(at=stamp(), **parity))
    pin(pins, E / 'INPUT_PARITY.json')
    write(E / 'FROZEN.json', dict(at=stamp(), status='FIXED_BODY_PROGRESS_EVALUATION_BEFORE_FORWARD',
        conditions=MODES, seeds=SEEDS, scenarios=scenarios, family_order=FAMILIES,
        contract=contract, pins=pins, protected_pins=protected,
        FIT_review=dict(path=str(FIT_REVIEW), sha256=sha(FIT_REVIEW), status=fit['status']),
        parity_sha256=sha(E / 'INPUT_PARITY.json'),
        no_model_forward=True, no_old_model_inference=True, no_private=True, no_final8=True, no_web=True))


def progression_for_condition(cards, ts, logits, mask, order):
    """Prediction-only API: card identity, family and immutable span, no labels."""
    outputs = []
    for card in cards:
        assert set(card) == {'card', 'family', 'span'}
        family_index = FAMILIES.index(card['family'])
        support = order.support(ts, card['span'], logits[:, family_index], mask[:, family_index])
        outputs.append(dict(card=card['card'], family=card['family'], **support))
    pairs = []
    for first, second in itertools.combinations(outputs, 2):
        if sorted([first['family'], second['family']]) != [2, 3]:
            continue
        extent = order.extent_relation(first['candidate_extent_us'], second['candidate_extent_us'])
        progress = order.extent_relation(first['support_extent_us'], second['support_extent_us'])
        pairs.append(dict(cards=[first['card'], second['card']],
            original_candidate_extent_order=extent, observed_body_progression_order=progress,
            unresolved_reasons=[c['reason'] for c in [first, second] if c['reason'] is not None],
            expected_order_used=False, candidate_spans_unchanged=True, completion=None))
    return outputs, pairs


def summarize(rows):
    """Evaluation expectations enter here only, after all selections/readouts."""
    summary, gates = [], []
    for seed, mode, kind in itertools.product(SEEDS, MODES, sorted({r['kind'] for r in rows})):
        subset = [r for r in rows if r['seed'] == seed and r['kind'] == kind]
        new_pairs = [p for r in subset for p in r['observed_progression_order'][mode]]
        old_pairs = [p for r in subset for p in r['pts_order'][mode]]
        supports = [c for r in subset for c in r['observed_progression'][mode]]
        summary.append(dict(seed=seed, condition=mode, kind=kind, scenarios=len(subset),
            original_candidate_PTS_order=dict(collections.Counter(p['state'] for p in old_pairs)),
            observed_body_progression_order=dict(collections.Counter(p['observed_body_progression_order']['state'] for p in new_pairs)),
            support_states=dict(collections.Counter(c['state'] for c in supports)),
            metrics_are_not_equivalent=True))
        if kind in ['full_normal', 'speed075', 'speed125']:
            assert len(new_pairs) == len(old_pairs) == 3
            failures = [dict(clip_id=r['clip_id'], pair=p) for r in subset
                        for p in r['observed_progression_order'][mode]
                        if p['observed_body_progression_order']['state'] != 'forward']
            gates.append(dict(seed=seed, condition=mode, gate=kind + '_observed_order',
                              required=3, passed=3-len(failures), pass_all=not failures, failures=failures))
    for seed, mode in itertools.product(SEEDS, MODES):
        cases = []
        for row in rows:
            if row['seed'] != seed or row['kind'] in ['full_normal', 'speed075', 'speed125']:
                continue
            for old, new in zip(row['pts_order'][mode], row['observed_progression_order'][mode]):
                if old['state'] == 'reverse':
                    cases.append(dict(clip_id=row['clip_id'], kind=row['kind'], cards=old['cards'],
                        state=new['observed_body_progression_order']['state']))
        assert cases
        failures = [c for c in cases if c['state'] != 'reverse']
        gates.append(dict(seed=seed, condition=mode, gate='edited_reverse_retained',
                          required=len(cases), passed=len(cases)-len(failures), pass_all=not failures, failures=failures))
    return summary, gates


def run():
    stop()
    assert not (E / 'EXECUTION_FROZEN.json').exists()
    assert not (E / 'COMPARISON.json').exists() and not (E / 'predictions').exists()
    cfg = read(E / 'FROZEN.json')
    check(cfg['pins'])
    protected_match(cfg['protected_pins'])
    assert cfg['conditions'] == MODES and cfg['seeds'] == SEEDS
    training_path = M / 'TRAINING_COMPLETE.json'
    training = read(training_path)
    assert training['status'] == 'BODY_PROGRESS_TRAINED_NOT_ADOPTED'
    assert len(training['rows']) == 3 and {r['seed'] for r in training['rows']} == set(SEEDS)
    execution_pins = {str(training_path): sha(training_path)}
    for row in training['rows']:
        pin(execution_pins, row['model'], row['model_sha256'])
        assert row['condition'] == 'body_progress_vs_reviewed_hold' and row['steps'] == 1200
    check(execution_pins)
    write(E / 'EXECUTION_FROZEN.json', dict(at=stamp(), pins=execution_pins,
        models_fixed_before_forward=True, expected_forward_calls=246, conditions_share_point_predictions=True))
    import torch
    spec = importlib.util.spec_from_file_location('_new_body_progress_head', M / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.FAMILIES == FAMILIES
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models = {}
    for row in training['rows']:
        state = torch.load(row['model'], map_location='cpu', weights_only=True)
        model = module.BodyProgressHead(state['mean'], state['std'])
        model.load_state_dict(state)
        model.eval()
        models[row['seed']] = model
    body_context, order = load_pure_contract()
    previous, semantics = read(PREVIOUS), read(SEMANTICS)
    old_rows, old_predictions, semantic_rows = validate_old(previous, semantics, cfg['scenarios'])
    rows, predictions, inputs, timing = [], [], [], []
    forward_calls = 0
    for scenario in cfg['scenarios']:
        stop()
        cid, kind = scenario['clip_id'], scenario['kind']
        assert sha(scenario['source']) == scenario['source_sha256']
        with np.load(scenario['source'], allow_pickle=False) as source:
            q = source['z'][:, 292:392].reshape(-1, 10, 10).copy()
            ts = source['ts'].copy()
        x, mask = context_input(q, ts, body_context)
        input_path = E / 'inputs' / f'{cid}_{kind}.npz'
        write_npz(input_path, x=x, timestamp_us=ts, body_observed=mask)
        inputs.append(dict(clip_id=cid, kind=kind, source=scenario['source'], source_sha256=scenario['source_sha256'],
                           path=str(input_path), sha256=sha(input_path), frames=len(ts), family_order=FAMILIES))
        tensor = torch.from_numpy(x)
        for seed in SEEDS:
            stop()
            k = seed, cid, kind
            tick = time.perf_counter_ns()
            with torch.inference_mode():
                logits = models[seed](tensor).numpy().copy()
            elapsed = (time.perf_counter_ns() - tick) / 1e9
            forward_calls += 1
            assert logits.shape == (len(ts), 5) and np.isfinite(logits).all()
            timing.append(dict(seed=seed, clip_id=cid, kind=kind, seconds=elapsed, forward_calls=1,
                scope='new CPU body-progress head forward plus output materialization only',
                excludes='loading, body-context construction, old heads, encoder, semantics, camera, browser and display'))
            path = E / 'predictions' / str(seed) / f'{cid}_{kind}.npz'
            write_npz(path, progression_logits=logits, timestamp_us=ts, body_observed=mask,
                      family_order=np.asarray(FAMILIES, dtype=np.int32))
            baseline = old_rows[k]
            row = copy.deepcopy(baseline)
            row['saved_semantic_conditions'] = copy.deepcopy(semantic_rows[k]['conditions'])
            row['observed_progression'], row['observed_progression_order'] = {}, {}
            for mode in MODES:
                cards = baseline['conditions'][mode]['cards']
                payload = [dict(card=c['card'], family=c['family'], span=c['chosen']['span'] if c['chosen'] else None) for c in cards]
                support, pairs = progression_for_condition(payload, ts, logits, mask, order)
                row['observed_progression'][mode] = support
                row['observed_progression_order'][mode] = pairs
                assert len(pairs) == len(baseline['pts_order'][mode])
                for new, old in zip(pairs, baseline['pts_order'][mode]):
                    assert new['cards'] == old['cards']
                    assert new['original_candidate_extent_order']['state'] == old['state']
                    assert new['original_candidate_extent_order']['overlap_duration_us'] == old['overlap_duration_us']
                for card, supported in zip(cards, support):
                    if card['chosen'] is not None:
                        a, b = card['chosen']['span']
                        original_observed = np.flatnonzero(mask[a:b+1, FAMILIES.index(card['family'])]) + a
                        assert original_observed.tolist() == card['chosen']['body_observed']
                    assert supported['completion'] is None
                if kind == 'all_hand_missing':
                    shape = [e for c in row['saved_semantic_conditions'][mode] for e in c['evidence'] if e['task'] == 'shape']
                    assert shape and all(e['state'] == 'observation_unknown' for e in shape)
            assert all(row[name] == value for name, value in baseline.items())
            assert row['saved_semantic_conditions'] == semantic_rows[k]['conditions']
            rows.append(row)
            predictions.append(dict(seed=seed, clip_id=cid, kind=kind, path=str(path), sha256=sha(path),
                source=scenario['source'], source_sha256=scenario['source_sha256'],
                input_path=str(input_path), input_sha256=sha(input_path),
                previous_prediction=old_predictions[k]['path'], previous_prediction_sha256=old_predictions[k]['sha256'],
                shared_by_conditions=MODES, forward_calls=1))
        print('body progression readout', len(rows), '/ 246', flush=True)
    assert forward_calls == len(rows) == len(predictions) == len(timing) == 246 and len(inputs) == 82
    summary, gates = summarize(rows)
    check(cfg['pins'])
    check(execution_pins)
    protected_match(cfg['protected_pins'])
    stop()
    write(E / 'COMPARISON.json', dict(at=stamp(), status='BODY_PROGRESS_READOUT_REQUIRES_INDEPENDENT_SAVED_REVIEW',
        rows=rows, predictions=predictions, inputs=inputs, summary=summary, observed_order_gates=gates, timing=timing,
        forward_calls=forward_calls, original_rows_exact=True, original_extents_and_cards_exact=True,
        saved_semantic_conditions_exact=True, original_extent_summary=previous['summary'],
        original_semantic_summary=semantics['summary'], metrics_are_not_equivalent=True,
        original_L_decision=read(L / 'DECISION.json'), original_L_extent_G2_pass=False,
        saved_FIT_review=read(FIT_REVIEW), protected_hashes_verified=477,
        missing_hands='Body evidence retained where body mask is valid. Shape observation_unknown and no educational/whole-sign completion remain unchanged.',
        exposure='Repeatedly exposed public DEV11/12/13. FIT input parity only uses FIT02-07. Probe08/09 and DEV are not fresh final evidence.',
        timing_scope='New small CPU head only; not full pipeline or user-visible latency.',
        training_sha256=sha(training_path), frozen_sha256=sha(E / 'FROZEN.json'),
        original_comparison_sha256=sha(PREVIOUS), original_semantics_sha256=sha(SEMANTICS),
        no_old_model_inference=True, no_encoder_inference=True, no_rank_reselection=True,
        no_semantic_inference=True, no_private=True, no_final8=True, no_web=True,
        semantic_completion=None, educational_completion=None, G2_pass=None, G3_pass=None))


if __name__ == '__main__':
    try:
        assert len(sys.argv) == 2 and sys.argv[1] in ['freeze-reviewed', 'run-reviewed']
        freeze() if sys.argv[1] == 'freeze-reviewed' else run()
    except BaseException:
        write(E / f'FIRST_FAILURE_{time.time_ns()}.json', dict(at=stamp(), traceback=traceback.format_exc()))
        raise
