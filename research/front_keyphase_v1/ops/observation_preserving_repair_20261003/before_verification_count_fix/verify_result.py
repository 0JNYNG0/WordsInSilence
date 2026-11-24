"""Independent result audit: Fraction reference + official TCN; candidate code not imported."""
from pathlib import Path
import json
import sys
import time

import numpy as np

from verify_reference import reference, filehash, load_json

HERE = Path(__file__).resolve().parent
CASES = ['clean', 'regular_half', 'isolated_drop', 'burst_drop', 'clock_noise']
ARMS = ['original', 'unrestored', 'global_linear', 'local_insert']


def score(v, true):
    order = np.argsort(-v, kind='stable')
    rival = next(j for j in order if j != true)
    return dict(pred_index=int(order[0]), rank=int(np.flatnonzero(order == true)[0]) + 1,
                correct=bool(order[0] == true), margin=float(v[true] - v[rival]))


def run():
    started = time.perf_counter()
    cfg = load_json(HERE / 'CONFIG_LOCAL_ONLY.json')
    prep = load_json(HERE / 'INDEPENDENT_PREFLIGHT.json')
    result = load_json(HERE / 'result/RESULT.json')
    sources = load_json(HERE / 'result/SOURCE_HASHES.json')
    for p, h in cfg['pins'].items():
        assert filehash(p) == h, p
    for k in ['lab_root', 'pilot_src', 'loader_dir', 'helper_dir', 'irregular_dir']:
        sys.path.insert(0, cfg[k])
    import run_audit as R
    import irregular_audit as I
    import corruptions as C
    import framerate_impl as F
    from pilot.data import build_channels, collate
    from pilot.web_input import to_native_like
    import tools_prep_union93_candidate as U
    import torch
    api = dict(F=F, build_channels=build_channels, to_native_like=to_native_like,
               native_roots=cfg['native_roots'], mp_roots=cfg['mp_roots'])
    rows = dict(native=R.load_native_rows(cfg['native_metadata']),
                mediapipe=R.load_mp_rows(cfg['mp_manifest']))
    all_scores, groups = {}, {}
    new_calls, cached = 0, 0
    for g, rs in rows.items():
        n = len(rs)
        with np.load(HERE / f'result/SCORES_{g}.npz', allow_pickle=False) as z:
            assert z['cases'].tolist() == CASES and z['arms'].tolist() == ARMS
            assert z['ids'].tolist() == [r['id'] for r in rs]
            classes = z['classes'].tolist()
            true = z['true_indices'].copy()
            logits = z['logits'].copy()
        assert logits.shape == (n, 5, 4, 93) and np.isfinite(logits).all()
        assert true.tolist() == [classes.index(r['label']) for r in rs]
        assert filehash(HERE / f'result/SCORES_{g}.npz') == result['hashes']['scores'][g]
        all_scores[g] = (logits, classes)
        with np.load(cfg['prior_scores'][g], allow_pickle=False) as z:
            prior, pc, pa = z['logits'].copy(), z['cases'].tolist(), z['arms'].tolist()
            assert z['ids'].tolist() == [r['id'] for r in rs] and z['classes'].tolist() == classes
        with np.load(cfg['regular_scores'][g], allow_pickle=False) as z:
            reg, ra = z['logits'].copy(), z['arms'].tolist()
            assert z['ids'].tolist() == [r['id'] for r in rs] and z['classes'].tolist() == classes
        groups[g] = {}
        for ci, case in enumerate(CASES):
            report = result['groups'][g]['cases'][case]
            table = report['rows']
            assert [r['id'] for r in table] == [r['id'] for r in rs]
            expected = {}
            if case == 'clean':
                expected = {a: reg[:, ra.index('original')] for a in ARMS[:3]}
            elif case == 'regular_half':
                expected = {a: reg[:, ra.index(k)] for a, k in zip(ARMS[:3], ['original', 'sparse', 'linear'])}
            else:
                expected = {a: prior[:, pc.index(case), pa.index(k)] for a, k in zip(ARMS[:3], ['original', 'corrupted', 'linear'])}
            for ai, a in enumerate(ARMS[:3]):
                assert np.array_equal(logits[:, ci, ai], expected[a]), (g, case, a)
            by_signer, by_class = {}, {}
            pred = logits[:, ci].argmax(axis=2)
            correct = pred == true[:, None]
            totals = {a: int(correct[:, ai].sum()) for ai, a in enumerate(ARMS)}
            assert totals == report['per_arm_correct']
            for i, (data_row, actual) in enumerate(zip(rs, table)):
                p = prep['groups'][g]['rows'][i]
                assert p['id'] == actual['id'] and actual['true'] == data_row['label']
                assert actual['signer_id'] == data_row['signer_id']
                ref = p['cases'][case]
                cell = actual['cell']
                assert ref['input_sha256'] == cell['input_sha256'], (g, i, case, 'features')
                assert ref['candidate_raw_sha256'] == cell['raw_sha256']['local_insert']
                assert ref['source_index_sha256'] == cell['source_index_sha256']
                assert ref['cache_match'] == cell['cache']
                assert ref['inserted'] == cell['insertion']['inserted_frames']
                assert ref['estimated_joints'] == cell['insertion']['inserted_joints']
                assert ref['source_frame_count'] == cell['insertion']['source_frames_covered_once']
                if ref['cache_match']:
                    cached += 1
                    assert np.array_equal(logits[i, ci, 3], expected[ref['cache_match']][i])
                else:
                    new_calls += 1
                for ai, a in enumerate(ARMS):
                    sc = score(logits[i, ci, ai], int(true[i]))
                    rr = actual['arms'][a]
                    assert classes[sc['pred_index']] == rr['pred']
                    assert sc['rank'] == rr['rank'] and sc['correct'] == rr['correct']
                    assert abs(sc['margin'] - rr['margin']) < 1e-10
                for dest, key in [(by_signer, str(data_row['signer_id'])), (by_class, data_row['label'])]:
                    item = dest.setdefault(key, dict(n=0, **{a: 0 for a in ARMS}))
                    item['n'] += 1
                    for ai, a in enumerate(ARMS):
                        item[a] += int(correct[i, ai])
            assert by_signer == report['per_signer'] and by_class == report['per_class']
            comparisons = {}
            for ai, a in enumerate(ARMS[:3]):
                gain = np.flatnonzero(~correct[:, ai] & correct[:, 3]).tolist()
                loss = np.flatnonzero(correct[:, ai] & ~correct[:, 3]).tolist()
                rp = report['comparisons'][f'{a}->local_insert']
                assert [r['id'] for r in rp['gains']] == [rs[i]['id'] for i in gain]
                assert [r['id'] for r in rp['losses']] == [rs[i]['id'] for i in loss]
                assert rp['net'] == len(gain) - len(loss)
                comparisons[a] = dict(gains=len(gain), losses=len(loss), gain_ids=[rs[i]['id'] for i in gain], loss_ids=[rs[i]['id'] for i in loss])
            groups[g][case] = dict(totals=totals, per_signer=by_signer, comparisons=comparisons)
        hashes = R.hash_sources(rs, cfg['native_roots' if g == 'native' else 'mp_roots'])
        assert hashes == prep['groups'][g]['sources'] == sources['sources']['before'][g] == sources['sources']['after'][g]
    assert new_calls == result['execution_counts']['cells_new_forward'] == 1764
    assert cached == sum(result['execution_counts']['cells_reused'].values()) == 2296

    # Independent inference: reconstruct the candidate input using ONLY this reference.
    torch.set_num_threads(2)
    model, model_classes = U.rebuild_from_payload(torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True))
    model.eval()
    before = {k: R.sha256_array(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}
    probes = []
    for g, indices in cfg['repro_indices'].items():
        saved, classes = all_scores[g]
        assert [str(x) for x in model_classes] == classes
        for i in indices:
            row = rows[g][i]
            base = I.prepare_base(R, api, g, row, cfg['tolerance'])
            for ci, case in enumerate(CASES):
                if case == 'clean':
                    xy, pr, ts = base['xy'], base['pr'], base['ts']
                elif case == 'regular_half':
                    xy, pr, ts = base['xy'][::2], base['pr'][::2], base['ts'][::2]
                else:
                    cor = C.corrupt(base['xy'], base['pr'], base['ts'], row['id'], case)
                    xy, pr, ts = cor['xy'], cor['present'], cor['ts']
                ref = reference(xy, pr, ts)
                x = build_channels(ref['xy'], ref['present'], ref['ts'])['x']
                assert R.sha256_array(x) == prep['groups'][g]['rows'][i]['cases'][case]['input_sha256']['local_insert']
                bx, mask, _ = collate([dict(x=x, frames=len(x), label=0)])
                with torch.inference_mode():
                    got = model(torch.from_numpy(bx), torch.from_numpy(mask))[0].numpy().astype(np.float64)
                delta = float(np.abs(got - saved[i, ci, 3]).max())
                assert delta <= cfg['tolerance'] and got.argmax() == saved[i, ci, 3].argmax(), (g, i, case, delta)
                probes.append(dict(group=g, id=row['id'], case=case, max_abs_diff=delta))
    assert before == {k: R.sha256_array(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}
    for p, h in cfg['pins'].items():
        assert filehash(p) == h
    assert 'cv2' not in sys.modules and 'mediapipe' not in sys.modules and 'local_insert' not in sys.modules
    gates = []
    for g in groups:
        for c in ['isolated_drop', 'burst_drop', 'clock_noise']:
            v = groups[g][c]
            gates.append(v['comparisons']['unrestored']['losses'] == 0)
            gates.extend(s['local_insert'] >= s['unrestored'] for s in v['per_signer'].values())
        gates.append(groups[g]['regular_half']['comparisons']['unrestored']['losses'] == 0)
    ng = groups['native']['isolated_drop']['totals']
    gates.append(ng['local_insert'] - ng['unrestored'] >= 7)
    assert all(gates) == result['admission']['all_conditions_hold']
    report = dict(status='PASS', reviewer='Codex independent reference + official model; no candidate import',
                  input_cells=4060, new_candidate_cells=new_calls, reused_cells=cached,
                  groups=groups, probes=probes, independent_forwards=len(probes),
                  max_logit_diff=max(r['max_abs_diff'] for r in probes), admission_all_hold=all(gates),
                  model_unchanged=True, protected_paths=len(cfg['pins']),
                  source_files=sum(len(g['sources']) for g in prep['groups'].values()),
                  seconds=time.perf_counter() - started)
    with (HERE / 'INDEPENDENT_VERIFICATION.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=report['status'], probes=len(probes), max_diff=report['max_logit_diff'],
                          admission_all_hold=all(gates), seconds=report['seconds']), ensure_ascii=False))


if __name__ == '__main__':
    run()
