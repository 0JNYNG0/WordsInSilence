"""One frozen, offline identity-veto comparison. No training, capture, or source edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ['original', 'original_repair', 'masked', 'masked_repair']
NEW_ARMS = ['gated_repair', 'masked_gated']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def ah(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(f'{a.shape}|{a.dtype}|'.encode() + a.tobytes()).hexdigest()


def write(path, data):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def pins_ok(pins):
    got = {p: digest(p) for p in pins}
    assert got == pins, [p for p in pins if got[p] != pins[p]]
    return len(got)


def invariants(xy, pr, ts, out, ungated):
    est = out['estimated']
    assert est.dtype == np.bool_ and est.shape == pr.shape
    assert not np.any(est & pr.astype(bool))
    assert np.all(~est | ungated['estimated'])
    assert same(out['xy'][~est], xy[~est])
    assert same(out['present'][~est], pr[~est])
    assert same(out['xy'][est], ungated['xy'][est])
    assert same(out['present'][est], ungated['present'][est])
    assert same(out['ts'], ts)
    assert bool(out['identity']) == (not bool(est.any()))
    assert sum(bool(e['accepted']) for e in out['decisions']) * 21 == int(est.sum())


def evaluate(cfg, output):
    started = time.time()
    pins_ok(cfg['pins'])
    old = json.loads(Path(cfg['prior_result']).read_text(encoding='utf-8'))
    review = json.loads(Path(cfg['visual_review']).read_text(encoding='utf-8'))
    for p in [cfg['helper_dir'], cfg['old_implementation'], cfg['loader_dir'], cfg['pilot_src'], cfg['lab_root']]:
        sys.path.insert(0, p)
    import run_audit as R
    import handgap as HG
    import framerate_impl as F
    from pilot.data import build_channels
    from pilot.web_input import to_native_like
    import gate_candidate as G
    import verify_gate as V
    assert Path(HG.__file__).resolve() == Path(cfg['old_implementation'], 'handgap.py').resolve()
    assert Path(G.__file__).resolve() == HERE / 'gate_candidate.py'
    assert Path(V.__file__).resolve() == HERE / 'verify_gate.py'
    api = {'F': F, 'build_channels': build_channels, 'to_native_like': to_native_like,
           'native_roots': cfg['native_roots'], 'mp_roots': cfg['mp_roots']}
    # No media libraries or model loading are needed to decide stage 1.
    assert not any(k in sys.modules for k in ['cv2', 'mediapipe'])
    rows = {'native': R.load_native_rows(cfg['native_metadata']),
            'mediapipe': R.load_mp_rows(cfg['mp_manifest'])}
    cache = {}
    classes = None
    for group, rr in rows.items():
        with np.load(cfg[f'scores_{group}'], allow_pickle=False) as z:
            assert z['ids'].tolist() == [r['id'] for r in rr]
            assert z['arms'].tolist() == ARMS
            cl = z['classes'].tolist()
            assert len(cl) == 93
            if classes is None:
                classes = cl
            assert cl == classes
            assert z['true_indices'].tolist() == [classes.index(r['label']) for r in rr]
            cache[group] = z['logits'].astype(np.float64)
            assert cache[group].shape == (len(rr), 4, 93)
        assert [r['id'] for r in old['groups'][group]['rows']] == [r['id'] for r in rr]
    source_pins = {}
    for group, rr in rows.items():
        for row in rr:
            for name, expected in row['files'].values():
                assert digest(name) == expected, name
                source_pins[name] = expected

    def load(group, row):
        if group == 'native':
            n = F.load_native(row['meta'], {R.NATIVE_ROLE}, set(R.NATIVE_SIGNERS), cfg['native_roots'])
            return n['xy'], n['present'], n['ts'], None
        xy, pr, ts, prepared, _diff = R.load_mp(api, row, 1e-5)
        return xy, pr, ts, prepared

    raw = {}
    decisions = {}
    reference_checks = 0

    def candidate(xy, pr, ts):
        nonlocal reference_checks
        before = (ah(xy), ah(pr), ah(ts))
        out = G.restore(xy, pr, ts)
        base = HG.restore(xy, pr, ts)
        invariants(xy, pr, ts, out, base)
        assert before == (ah(xy), ah(pr), ah(ts))
        ref = V.reference_decisions(xy, pr, ts)
        V.compare_decisions(ref, out['decisions'])
        V.verify_candidate_output(xy, pr, ts, out)
        reference_checks += 1
        return out, base

    # All MP rows, not only rows with positive outcomes. Visual labels never enter candidate.
    for row in rows['mediapipe']:
        inp = load('mediapipe', row)
        raw[('mediapipe', row['id'])] = inp
        out, _base = candidate(*inp[:3])
        for d in out['decisions']:
            decisions[(row['id'], int(d['frame']), d['side'])] = d
    events = []
    totals = {k: {'accepted': 0, 'rejected': 0} for k in ['clear_same', 'wrong_endpoint', 'uncertain']}
    reason_counts = Counter()
    for e in review['events']:
        # Review stores original frame IDs; use the prior audit's local-row mapping.
        prov = cfg['_provenance'][e['event'] - 1]
        assert prov['sample_id'] == e['sample_id'] and prov['side'] == e['side']
        d = decisions[(e['sample_id'], prov['local_row'], e['side'])]
        totals[e['classification']]['accepted' if d['accepted'] else 'rejected'] += 1
        reason_counts.update(d['reasons'])
        events.append({'event': e['event'], 'sample_id': e['sample_id'],
                       'visual_class': e['classification'], **d})
    assert len(events) == len(decisions) == 88
    checks = {'all_seven_wrong_rejected': totals['wrong_endpoint']['rejected'] == 7,
              'majority_clear_same_retained': totals['clear_same']['accepted'] >= 39,
              'original_observations_preserved': True,
              'reference_agreement': True}
    stage1 = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'totals': totals, 'reason_counts': dict(reason_counts), 'events': events,
              'independent_reference_rows': reference_checks, 'model_calls': 0}
    write(output / 'STAGE1.json', stage1)
    print(json.dumps({'stage1': stage1['status'], 'totals': totals}, ensure_ascii=False), flush=True)
    if stage1['status'] != 'PASS':
        pins_ok(cfg['pins']); pins_ok(source_pins)
        result = {'status': 'REJECTED_STAGE1_NO_MODEL_EXECUTION', 'stage1': stage1,
                  'seconds': time.time() - started, 'forwards': 0, 'training': 0,
                  'source_pins_unchanged': len(source_pins), 'protected_pins_unchanged': len(cfg['pins'])}
        write(output / 'RESULT.json', result)
        return result

    # Prepare every changed model input before loading the model. Rebuild old four-arm
    # features only to prove exact cache matches, never re-score identical inputs.
    records = {'native': [], 'mediapipe': []}
    pending = []
    coord = {'native': [], 'mediapipe': []}
    for group, rr in rows.items():
        for idx, row in enumerate(rr):
            inp = raw.get((group, row['id'])) or load(group, row)
            xy, pr, ts, prepared = inp
            natural, natural_base = candidate(xy, pr, ts)
            m = HG.mask_one_hand(xy, pr, ts, row['id'])
            repaired, repair_base = candidate(m['xy'], m['present'], m['ts'])
            base_raw = {'original': (xy, pr, ts),
                        'original_repair': (natural_base['xy'], natural_base['present'], ts),
                        'masked': (m['xy'], m['present'], ts),
                        'masked_repair': (repair_base['xy'], repair_base['present'], ts)}
            bases = {a: build_channels(*v) for a, v in base_raw.items()}
            if prepared is not None:
                bases['original']['x'] = prepared
            prior = old['groups'][group]['rows'][idx]
            for arm in ARMS:
                assert ah(bases[arm]['x']) == prior['input']['input_sha256'][arm], (group, row['id'], arm)
            body = float(bases['original']['body_scale'])
            assert np.isfinite(body) and body > 0
            for e in m['events']:
                j, side = e['frame'], e['side']; hs = HG.HANDS[side]
                accepted = bool(repaired['estimated'][j, hs].all())
                ce = {'id': row['id'], 'frame': j, 'side': side, 'accepted': accepted}
                if accepted:
                    lin = np.linalg.norm(repaired['xy'][j, hs].astype(np.float64) - xy[j, hs], axis=-1) / body
                    prev = np.linalg.norm(xy[j-1, hs].astype(np.float64) - xy[j, hs], axis=-1) / body
                    ce.update(linear_mean=float(lin.mean()), previous_mean=float(prev.mean()))
                coord[group].append(ce)
            rec = {'id': row['id'], 'label': row['label'], 'signer': row['signer_id'],
                   'frames': len(ts), 'arms': {}, 'input_hashes': {}, 'cache': {},
                   'decisions': {'gated_repair': natural['decisions'], 'masked_gated': repaired['decisions']}}
            known = {ah(bases[a]['x']): (a, cache[group][idx, k]) for k, a in enumerate(ARMS)}
            new_seen = {}
            for arm, rep in [('gated_repair', natural), ('masked_gated', repaired)]:
                x = build_channels(rep['xy'], rep['present'], rep['ts'])['x']
                assert x.dtype == np.float32 and x.shape == (len(ts), 280) and np.isfinite(x).all()
                hx = ah(x); rec['input_hashes'][arm] = hx
                if hx in known:
                    old_arm, logits = known[hx]
                    rec['cache'][arm] = 'saved:' + old_arm
                    rec['arms'][arm] = {'logits': logits.tolist()}
                elif hx in new_seen:
                    rec['cache'][arm] = 'new_same_row:' + new_seen[hx]
                else:
                    rec['cache'][arm] = 'new'
                    new_seen[hx] = arm
                    pending.append((group, idx, arm, x))
            for k, arm in enumerate(ARMS):
                rec['arms'][arm] = {'logits': cache[group][idx, k].tolist()}
            records[group].append(rec)
    write(output / 'PREFLIGHT.json', {'status': 'PASS', 'rows': records,
          'expected_new_forwards': len(pending), 'independent_reference_checks': reference_checks,
          'coordinate_events': coord, 'all_prior_input_hashes_reproduced': True})
    print(json.dumps({'preflight': 'PASS', 'new_forwards': len(pending)}), flush=True)

    forward_count = 0
    if pending:
        import torch
        import tools_prep_union93_candidate as U
        from pilot.data import collate
        torch.set_num_threads(2)
        payload = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
        model, mc = U.rebuild_from_payload(payload)
        model.eval()
        assert list(mc) == classes and not any(m.training for m in model.modules())
        state_before = {k: ah(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}
        for group, idx, arm, x in pending:
            bx, pad, _ = collate([{'x': x, 'frames': len(x), 'label': 0}])
            with torch.inference_mode():
                logits = model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
            assert logits.shape == (93,) and np.isfinite(logits).all()
            records[group][idx]['arms'][arm] = {'logits': logits.tolist()}
            forward_count += 1
        assert state_before == {k: ah(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}
    assert forward_count == len(pending)
    summary = {}
    stage2_checks = {}
    for group, rr in records.items():
        for rec in rr:
            for a in NEW_ARMS:
                if rec['cache'][a].startswith('new_same_row:'):
                    base = rec['cache'][a].split(':', 1)[1]
                    rec['arms'][a] = {'logits': list(rec['arms'][base]['logits'])}
            for a, v in rec['arms'].items():
                v.update(R.score_row(np.array(v['logits']), classes, rec['label']))
        natural_newwrong = [r['id'] for r in rr if r['arms']['original']['correct'] and not r['arms']['gated_repair']['correct']]
        masked_newwrong = [r['id'] for r in rr if r['arms']['masked']['correct'] and not r['arms']['masked_gated']['correct']]
        retained = [e for e in coord[group] if e['accepted']]
        cm = {'injected_events': len(coord[group]), 'accepted_events': len(retained),
              'linear_mean_on_retained': float(np.mean([e['linear_mean'] for e in retained])) if retained else None,
              'previous_mean_on_same_retained': float(np.mean([e['previous_mean'] for e in retained])) if retained else None}
        summary[group] = {'n': len(rr), 'correct': {a: sum(r['arms'][a]['correct'] for r in rr) for a in ARMS + NEW_ARMS},
                          'natural_newwrong_vs_original': natural_newwrong,
                          'masked_newwrong_vs_masked': masked_newwrong,
                          'coordinate': cm,
                          'per_signer': {str(s): {a: sum(r['arms'][a]['correct'] for r in rr if r['signer'] == s) for a in ARMS + NEW_ARMS} for s in sorted({r['signer'] for r in rr})}}
        stage2_checks[group + '_natural_no_newwrong'] = not natural_newwrong
        stage2_checks[group + '_masked_no_newwrong'] = not masked_newwrong
        stage2_checks[group + '_coordinate_better_than_copy'] = bool(retained and cm['linear_mean_on_retained'] < cm['previous_mean_on_same_retained'])
    required = 'NIA_SL_SEN0472_REAL11_F_a01_044'
    recovered = next(r for r in records['native'] if r['id'] == required)
    stage2_checks['prior_synthetic_recovery_preserved'] = bool(recovered['arms']['masked_gated']['correct'])
    pins_ok(cfg['pins']); pins_ok(source_pins)
    assert not any(k in sys.modules for k in ['cv2', 'mediapipe'])
    result = {'status': 'LIMITED_DEVELOPMENT_PASS_NOT_DEPLOYED' if all(stage2_checks.values()) else 'REJECTED_STAGE2',
              'stage1': stage1, 'stage2_checks': stage2_checks, 'summary': summary,
              'rows': records, 'new_forwards': forward_count, 'classes': classes,
              'training': 0, 'camera': 0, 'operational_change': 0,
              'source_pins_unchanged': len(source_pins), 'protected_pins_unchanged': len(cfg['pins']),
              'seconds': time.time() - started}
    write(output / 'RESULT.json', result)
    for group, rr in records.items():
        with open(output / ('SCORES_' + group + '.npz'), 'xb') as f:
            np.savez_compressed(f, ids=np.array([r['id'] for r in rr]), classes=np.array(classes),
                                arms=np.array(ARMS + NEW_ARMS),
                                logits=np.array([[r['arms'][a]['logits'] for a in ARMS + NEW_ARMS] for r in rr]))
    print(json.dumps({'result': result['status'], 'summary': summary}, ensure_ascii=False), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); cfg_path = Path(args.config).resolve(); cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    output = Path(args.out).resolve()
    assert output.parent == HERE and not output.exists(), 'Output must be a NEW immediate directory under this unit'
    output.mkdir()
    cfg['_provenance'] = json.loads(Path(cfg['provenance']).read_text(encoding='utf-8'))['events']
    try:
        return evaluate(cfg, output)
    except BaseException as exc:
        write(output / 'FAILURE.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
