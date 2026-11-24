"""G2 cause isolation on J's changed spans, not G3 acceptance or web adoption.

Freeze first; the parent reviews this code before run-reviewed. All old semantic
results are copied. Existing identical requests are reused across old conditions.
Only previously unscored intervals use the original frozen semantic observer.
"""
from pathlib import Path
import collections
import copy
import hashlib
import importlib.util
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
J = Path(__file__).resolve().parent
H = J.parent
G = H.parent
O = G.parent
W = G / 'candidate_description_supervision'
L = W / 'local_rank_learning'
E = J / 'changed_span_shadow'
OLD = L / 'evaluation/shadow_components/RESULTS.json'
OLD_COMP = L / 'evaluation/COMPARISON.json'
NEW_COMP = J / 'evaluation/COMPARISON.json'
REQ = O / 'word100_guide_requirement_learning_20261111'
OBSERVER = O / 'word100_broad_factorized_learning_20261110'


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with p.open('x', encoding='utf-8') as f:
        f.write(text)


def rowkey(r):
    return r['seed'], r['clip_id'], r['kind']


def requestkey(cid, kind, card, span):
    return cid, kind, card, None if span is None else tuple(span)


def keytext(key):
    cid, kind, card, span = key
    return '|'.join([cid, kind, card, 'none' if span is None else f'{span[0]}:{span[1]}'])


def selected_span(c):
    return c['chosen']['span'] if c['chosen'] is not None else None


def contract():
    """Saved arithmetic only. No model import, inference, fitting or selection."""
    old = read(OLD)
    base = read(OLD_COMP)
    new = read(NEW_COMP)
    om = {rowkey(r): r for r in old['rows']}
    bm = {rowkey(r): r for r in base['rows']}
    nm = {rowkey(r): r for r in new['rows']}
    pm = {rowkey(r): r for r in new['predictions']}
    bpm = {rowkey(r): r for r in base['predictions']}
    assert set(om) == set(bm) == set(nm) == set(pm) == set(bpm)
    assert len(om) == 246
    cache = {}
    origins = {}
    for r in old['rows']:
        assert r['clip_id'].startswith('NIA_SL_WORD')
        assert r['clip_id'].split('_')[3] in ('REAL11', 'REAL12', 'REAL13')
        for mode, cards in r['conditions'].items():
            baseline_cards = {c['card']: c for c in bm[rowkey(r)]['conditions'][mode]['cards']}
            for c in cards:
                assert c['span'] == selected_span(baseline_cards[c['card']])
                key = requestkey(r['clip_id'], r['kind'], c['card'], c['span'])
                if key in cache:
                    assert cache[key] == c, ('prior semantic duplicate differs', key)
                else:
                    cache[key] = c
                    origins[key] = dict(seed=r['seed'], condition=mode)
    plan = []
    sources = {}
    requests = {}
    changed = 0
    total = 0
    for r in new['rows']:
        k = rowkey(r)
        assert all(r['conditions'][mode] == bm[k]['conditions'][mode] for mode in bm[k]['conditions'])
        assert pm[k]['source'] == bpm[k]['source']
        assert pm[k]['source_sha256'] == bpm[k]['source_sha256']
        oldcards = {c['card']: c for c in om[k]['conditions']['local_full']}
        rankcards = r['conditions']['splice_rank']['cards']
        assert set(oldcards) == {c['card'] for c in rankcards}
        for c in rankcards:
            total += 1
            before = oldcards[c['card']]
            span = selected_span(c)
            key = requestkey(r['clip_id'], r['kind'], c['card'], span)
            if before['span'] == span:
                action = 'copy_same_local_full'
            else:
                changed += 1
                assert c['family'] in (2, 3)
                action = 'reuse_saved_identical_request' if key in cache else 'new_request'
                if key not in requests:
                    requests[key] = dict(request_id=keytext(key), clip_id=r['clip_id'], kind=r['kind'],
                        card=c['card'], family=c['family'], span=span, action=action,
                        saved_origin=origins.get(key))
            plan.append(dict(seed=r['seed'], clip_id=r['clip_id'], kind=r['kind'], card=c['card'],
                old_span=before['span'], new_span=span, action=action, request_id=keytext(key)))
        sid = r['clip_id'], r['kind']
        source = dict(clip_id=sid[0], kind=sid[1], source=pm[k]['source'], sha256=pm[k]['source_sha256'])
        if sid in sources:
            assert sources[sid] == source
        sources[sid] = source
    assert changed == 77 and len(requests) == 55
    counts = collections.Counter(q['action'] for q in requests.values())
    assert counts['reuse_saved_identical_request'] == 18 and counts['new_request'] == 37
    return dict(old=old, new=new, oldmap=om, saved=cache, plan=plan, requests=requests,
        sources=sources, total=total, changed=changed)


def check_pins(pins):
    for path, expected in pins.items():
        assert sha(path) == expected, ('changed pin', path)


def stop_guard():
    for root in (G, H, J):
        for name in ('STOP', 'STOP.txt', 'STOP_REQUESTED', 'STOP_REQUESTED.json'):
            assert not (root / name).exists(), ('stop marker', str(root / name))


def freeze():
    stop_guard()
    assert not (E / 'FROZEN.json').exists()
    d = contract()
    pins = dict(read(G / 'PROTECTIONS.json'))
    files = [Path(__file__), OLD, OLD_COMP, NEW_COMP,
        L / 'evaluation/shadow_components/FROZEN.json', J / 'TRAINING_COMPLETE.json',
        J / 'evaluation/FROZEN.json', J / 'evaluate.py', W / 'shadow_components.py',
        REQ / 'requirement_core.py', REQ / 'GUIDE_CAPABILITY.json',
        OBSERVER / 'core.py', OBSERVER / 'NORMALIZATION.json', OBSERVER / 'runs/coupled_42/model.pt',
        O / 'word100_shared_motion_learning_20261109/core.py',
        O / 'word100_motion_components_20261109/motion.py',
        O / 'word100_rtmw_joint_learning_20261108/feedback_model.py',
        O / 'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json',
        O / 'word100_action_feedback_demo_20261108/CARDS.json',
        O / 'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json']
    pins.update({str(p): sha(p) for p in files})
    pins.update({s['source']: s['sha256'] for s in d['sources'].values()})
    check_pins(pins)
    oldcfg = read(L / 'evaluation/shadow_components/FROZEN.json')
    assert oldcfg['observer_checkpoint'] == 'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt'
    write_new(E / 'FROZEN.json', dict(status='FROZEN_AWAITING_PARENT_CODE_REVIEW', at=time.time(),
        purpose='G2 cause isolation: changed J extents with frozen existing semantic heads. No G3 pass declaration.',
        conditions=list(d['old']['rows'][0]['conditions']) + ['splice_rank'],
        rows=246, source_scenarios=82, total_cards=d['total'], changed_card_occurrences=77,
        changed_unique_requests=55, identical_saved_unique_requests=18, new_unique_requests=37,
        new_condition='splice_rank', reference='local_full', observer_seed=42, selector_seeds=[42, 43, 44],
        plan=d['plan'], requests=list(d['requests'].values()), sources=list(d['sources'].values()),
        observer_checkpoint=oldcfg['observer_checkpoint'],
        semantics='Exact old shadow geometry/body wrist, same attrs and direction heads, same guide capability, >=.8 availability and .65/.35 decisions. PD/hand candidates not used.',
        shape_policy='Same late-half scope as old shadow. When pointwise cache absent, only valid frames needed by new intervals go through the Linear(64,7) attribute layer once per source; write reusable pointwise cache. No temporal reencoding.',
        direction_policy='Only previously unscored source/span/axis/side requests use old movement head; h is saved whole-source fh. Old results reused unchanged. A hold/uncertain result is not opposite motion.',
        no_final8=True, no_private=True, no_training=True, no_TCN_or_DTW=True, no_pose_or_encoder_inference=True,
        no_web_change=True, no_threshold_change=True, semantic_completion=None,
        limitations=['Previously exposed development; not independent generalization.',
            'Edited controls are not naturally filmed learner errors.',
            'Current live web has a different consumer/model lineage; no runtime replacement here.'], pins=pins))
    print('Frozen77 changed cards /55 requests:18 saved reuse,37 new requests. No inference.', flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def execute():
    stop_guard()
    cfg = read(E / 'FROZEN.json')
    check_pins(cfg['pins'])
    assert not (E / 'RUN_STARTED.json').exists() and not (E / 'RESULTS.json').exists()
    d = contract()
    assert d['plan'] == cfg['plan'] and list(d['requests'].values()) == cfg['requests']
    write_new(E / 'RUN_STARTED.json', dict(at=time.time(), frozen_sha256=sha(E / 'FROZEN.json'),
        command='run-reviewed', note='Parent-controlled execution after code review; no completed result reruns.'))
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    rc = load_module('_splice_changed_semantics_req', REQ / 'requirement_core.py')
    net = rc.frozen_observer()
    assert isinstance(net.shared.base.attributes, torch.nn.Linear)
    assert net.shared.base.attributes.in_features == 64 and net.shared.base.attributes.out_features == 7
    caps = {(r['word'], r['card'], r['feature'], r['expected']): r['qualified']
        for r in read(REQ / 'GUIDE_CAPABILITY.json')['rows']}
    new_requests = [q for q in cfg['requests'] if q['action'] == 'new_request']
    by_source = collections.defaultdict(list)
    for q in new_requests:
        by_source[q['clip_id'], q['kind']].append(q)
    computed = {}
    execution = []
    attrs_records = []
    movement_records = []
    tick = time.perf_counter()
    for sid, requests in by_source.items():
        stop_guard()
        source = d['sources'][sid]
        assert sha(source['source']) == source['sha256']
        with np.load(source['source'], allow_pickle=False) as z:
            h, x, ts = z['fh'].copy(), z['x'].copy(), z['ts'].copy()
        assert h.shape == (len(x), 64) and x.shape[1] == 280 and len(ts) == len(x)
        assert np.isfinite(h).all() and np.isfinite(x).all() and np.all(np.diff(ts) > 0)
        word = sid[0].split('_')[2]
        av = rc.prior.shared.available(x)
        needed = set()
        scopes = {}
        for q in requests:
            if q['span'] is None:
                scopes[q['request_id']] = None
                continue
            a, b = q['span']
            assert 0 <= a <= b < len(x)
            guide = next(v for v in rc.CARDS[word] if v['id'] == q['card'])
            aa = a + (b-a)//2 if guide.get('shape_scope') == 'late' else a
            scopes[q['request_id']] = (aa, b)
            for j in guide.get('features', {}):
                j = int(j)
                needed.update(int(v) for v in np.arange(aa, b+1)[av[aa:b+1, j]])
        frame_indices = np.asarray(sorted(needed), dtype=np.int64)
        values = np.empty((0, 7), np.float32)
        if len(frame_indices):
            with torch.inference_mode():
                values = net.shared.base.attributes(torch.tensor(h[frame_indices])).sigmoid().numpy()
        frame_lookup = {int(f): i for i, f in enumerate(frame_indices)}
        p = E / 'pointwise_cache' / f'{sid[0]}_{sid[1]}.npz'
        p.parent.mkdir(parents=True, exist_ok=True)
        assert not p.exists()
        np.savez_compressed(p, frame_indices=frame_indices, attribute_probabilities=values)
        attrs_records.append(dict(clip_id=sid[0], kind=sid[1], source=source['source'],
            source_sha256=source['sha256'], path=str(p), sha256=sha(p), selected_valid_frames=len(frame_indices),
            total_source_frames=len(x), reason='Old shadow retained interval aggregates only. New span needs these pointwise probabilities; no source encoding or old interval recomputation.'))
        movement_cache = {}
        for q in requests:
            if q['span'] is None:
                card = dict(card=q['card'], span=None, status='association_unknown', evidence=[], completion=None)
                computed[q['request_id']] = card
                continue
            a, b = q['span']
            aa, _ = scopes[q['request_id']]
            guide = next(v for v in rc.CARDS[word] if v['id'] == q['card'])
            spec = next(v for v in rc.FAMILIES[word] if v['id'] == q['card'])
            items = []
            for j, want in guide.get('features', {}).items():
                j = int(j)
                valid = av[aa:b+1, j]
                ok = bool(valid.mean() >= .8)
                ids = np.arange(aa, b+1)[valid]
                pp = np.asarray([values[frame_lookup[int(f)], j] for f in ids], dtype=np.float32)
                pmean = float(pp.mean()) if len(pp) else None
                pmean = pmean if want or pmean is None else 1-pmean
                qualified = caps.get((word, q['card'], j, want), False)
                state = rc.decide(pmean, ok, qualified) if pmean is not None else 'observation_unknown'
                items.append(dict(task='shape', feature=j, expected=want, span=[aa, b],
                    probability=pmean, state=state, qualified=qualified, observed_count=int(valid.sum())))
            axis = spec['axis']
            side = spec['hands'][0]
            if axis is not None:
                if spec['sign'] is None:
                    items.append(dict(task='direction', state='sequence_required',
                        reason='Alternation cannot be judged from first/last displacement alone'))
                else:
                    mk = a, b, axis, side
                    if mk not in movement_cache:
                        g, ok, info = rc.prior.geometry(x, ts, a, b, axis, side, 'L' if side == 'R' else 'R')
                        with torch.inference_mode():
                            probs = net.movement(torch.tensor(h[None]), torch.tensor(g[None]),
                                torch.tensor([[0, a, b]])).softmax(-1)[0].numpy()
                        movement_cache[mk] = dict(probabilities=probs.tolist(), eligible=bool(ok))
                        movement_records.append(dict(clip_id=sid[0], kind=sid[1], span=[a, b], axis=axis,
                            moving=side, eligible=bool(ok), probabilities=probs.tolist(),
                            geometry=g.tolist(), info=info, source_sha256=source['sha256']))
                    vv = movement_cache[mk]
                    want = int(spec['sign']+1)
                    items.append(dict(task='direction', axis=axis, desired=want, span=[a, b],
                        probabilities=vv['probabilities'],
                        state=rc.direction_verdict(vv['probabilities'], want, vv['eligible'])))
            states = [v['state'] for v in items]
            status = ('difference_evidence' if 'difference' in states or 'opposite_direction' in states else
                'supported_features_only' if states and all(v == 'supported' for v in states) else 'insufficient_evidence')
            computed[q['request_id']] = dict(card=q['card'], span=[a, b], status=status,
                evidence=items, completion=None)
        print('New semantic source', sid, 'selected attribute frames', len(frame_indices),
            'direction requests', len(movement_cache), flush=True)
    assert len(computed) == 37
    result_rows = []
    for r in d['new']['rows']:
        k = rowkey(r)
        before = d['oldmap'][k]
        conditions = copy.deepcopy(before['conditions'])
        oldcards = {c['card']: c for c in before['conditions']['local_full']}
        cards = []
        for c in r['conditions']['splice_rank']['cards']:
            span = selected_span(c)
            key = requestkey(r['clip_id'], r['kind'], c['card'], span)
            if span == oldcards[c['card']]['span']:
                after = copy.deepcopy(oldcards[c['card']])
                action = 'copy_same_local_full'
            elif key in d['saved']:
                after = copy.deepcopy(d['saved'][key])
                action = 'reuse_saved_identical_request'
            else:
                after = copy.deepcopy(computed[keytext(key)])
                action = 'new_request'
            assert after['span'] == span and after['card'] == c['card'] and after['completion'] is None
            cards.append(after)
            execution.append(dict(seed=r['seed'], clip_id=r['clip_id'], kind=r['kind'], card=c['card'],
                request_id=keytext(key), action=action, old_span=oldcards[c['card']]['span'], new_span=span))
        conditions['splice_rank'] = cards
        assert all(conditions[mode] == before['conditions'][mode] for mode in before['conditions'])
        result_rows.append(dict(seed=r['seed'], clip_id=r['clip_id'], kind=r['kind'], conditions=conditions))
    assert execution == cfg['plan']
    summary = copy.deepcopy(d['old']['summary'])
    for seed in [42, 43, 44]:
        for kind in sorted({r['kind'] for r in result_rows}):
            cards = [c for r in result_rows if r['seed'] == seed and r['kind'] == kind
                for c in r['conditions']['splice_rank']]
            items = [v for c in cards for v in c['evidence']]
            summary.append(dict(seed=seed, condition='splice_rank', kind=kind, cards=len(cards),
                statuses=dict(collections.Counter(c['status'] for c in cards)),
                shape=dict(collections.Counter(v['state'] for v in items if v['task'] == 'shape')),
                direction=dict(collections.Counter(v['state'] for v in items if v['task'] == 'direction'))))
    assert all(v['state'] == 'observation_unknown' for r in result_rows if r['kind'] == 'all_hand_missing'
        for c in r['conditions']['splice_rank'] for v in c['evidence'] if v['task'] == 'shape')
    stop_guard()
    check_pins(cfg['pins'])
    write_new(E / 'RESULTS.json', dict(status='G2_CHANGED_SPAN_CAUSE_ISOLATION_NOT_G3_PASS', at=time.time(),
        rows=result_rows, summary=summary, execution=execution, pointwise_cache=attrs_records,
        new_requests=list(computed), new_movement_requests=movement_records,
        changed_card_occurrences=77, changed_unique_requests=55, saved_identical_requests=18, new_unique_requests=37,
        observer_seed=42, selector_seeds=[42, 43, 44], original_all_conditions_preserved=True,
        old_predictions_rerun=False, common_encoder_rerun=False, TCN_DTW_rerun=False,
        total_seconds=time.perf_counter()-tick, timing_scope='Cached h/x to changed-span frozen semantics only; not camera/RTMW/end-to-end latency.',
        frozen_sha256=sha(E / 'FROZEN.json'), web_changed=False, semantic_completion=None, G3_pass=None))
    print('Saved changed-span isolation. No G3 approval and no web adoption.', flush=True)


if __name__ == '__main__':
    try:
        assert len(sys.argv) == 2 and sys.argv[1] in ('freeze', 'run-reviewed')
        freeze() if sys.argv[1] == 'freeze' else execute()
    except BaseException:
        E.mkdir(parents=True, exist_ok=True)
        failure = E / ('FIRST_FAILURE_' + str(time.time_ns()) + '.txt')
        failure.write_text(traceback.format_exc(), encoding='utf-8')
        raise
