"""Codex independent verification: official model forward plus pooling hook.

Does not import the candidate forward implementation. No camera or training.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

H = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg = read(H / 'CONFIG_LOCAL_ONLY.json')
sys.path.insert(0, cfg['irregular_dir'])
import verify_irregular as V
import torch

CASES = ['isolated_drop', 'burst_drop', 'clock_noise']
ARMS = ['grid_baseline', 'pool_only', 'blocks_only', 'blocks_and_pool']


def hook_forward(model, x, support, arm):
    """Use official forward for all blocks; replace only its head's pooled input."""
    batch = torch.from_numpy(x[None])
    keep = torch.from_numpy(support.astype(np.float32)[None])
    block_mask = keep if arm in ['blocks_only', 'blocks_and_pool'] else torch.ones_like(keep)
    pool = keep if arm in ['pool_only', 'blocks_and_pool'] else torch.ones_like(keep)
    last = {}

    def capture(module, args, output):
        last['h'] = output

    def repool(module, args):
        value = torch.einsum('bct,bt->bc', last['h'], pool) / pool.sum(dim=1, keepdim=True)
        return (value,)

    capture_handle = model.blocks[-1].register_forward_hook(capture)
    head_handle = model.head.register_forward_pre_hook(repool)
    try:
        with torch.inference_mode():
            result = model(batch, block_mask)[0].numpy().copy()
    finally:
        capture_handle.remove()
        head_handle.remove()
    return result


def compare(scores, a, b, truth, ids):
    pa, pb = np.argmax(scores[a], axis=-1), np.argmax(scores[b], axis=-1)
    ca, cb = pa == truth, pb == truth
    return {
        'gains': [ids[i] for i in np.flatnonzero(~ca & cb)],
        'losses': [ids[i] for i in np.flatnonzero(ca & ~cb)],
        'gains_n': int((~ca & cb).sum()), 'losses_n': int((ca & ~cb).sum()),
        'top1_changed': int((pa != pb).sum()),
    }


def main():
    for p, digest in cfg['pins'].items():
        assert sha(p) == digest, p
    prior = read(cfg['prior_result'])
    report = read(H / 'result/RESULT.json')
    torch.set_num_threads(2)
    payload = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    model, classes = V.u.rebuild_from_payload(payload)
    model.eval()
    before = {k: V.array_sha(t.numpy()) for k, t in model.state_dict().items()}
    manifests = {'native': read(cfg['native_metadata'])['annotation_dev'],
                 'mediapipe': read(cfg['mp_manifest'])['rows']}
    source_hashes = {}
    out = {'status': 'PASS', 'reviewer': 'Codex',
           'independent_path': 'official model.forward with final-block capture/head pooling hook; candidate forward not imported',
           'groups': {}, 'counts': {'cells': 0, 'forwards': 0}, 'max_logit_difference': 0.0,
           'production_applied': False}
    for group, rows in manifests.items():
        with np.load(cfg['prior_scores'][group], allow_pickle=False) as z:
            old_logits = z['logits'].copy()
            old_arms = z['arms'].tolist()
            assert z['cases'].tolist() == CASES and z['classes'].tolist() == list(classes)
            ids = z['ids'].tolist()
            truth = z['true_indices'].copy()
        with np.load(H / f'result/SCORES_{group}.npz', allow_pickle=False) as z:
            logits = z['logits'].copy()
            assert z['cases'].tolist() == CASES and z['arms'].tolist() == ARMS
            assert z['classes'].tolist() == list(classes) and z['ids'].tolist() == ids
            assert np.array_equal(z['true_indices'], truth)
        assert logits.shape == (len(rows), 3, 4, 93) and np.isfinite(logits).all()
        assert np.array_equal(logits[:, :, 0], old_logits[:, :, old_arms.index('linear')])
        case_out = {}
        for ci, case in enumerate(CASES):
            scores = {a: logits[:, ci, ai] for ai, a in enumerate(ARMS)}
            scores.update(prior_original=old_logits[:, ci, old_arms.index('original')],
                          prior_corrupted=old_logits[:, ci, old_arms.index('corrupted')])
            pred = {a: np.argmax(v, axis=-1) for a, v in scores.items()}
            totals = {a: int((p == truth).sum()) for a, p in pred.items()}
            per_signer, per_class = {}, {}
            for key, table, values in [('signer_id', per_signer, sorted({r['signer_id'] for r in rows})),
                                       ('label', per_class, sorted({r['label'] for r in rows}))]:
                for value in values:
                    idx = np.array([i for i, r in enumerate(rows) if r[key] == value])
                    table[str(value)] = {'n': len(idx), **{a: int((p[idx] == truth[idx]).sum()) for a, p in pred.items()}}
            pairs = [(a, b) for a in ['grid_baseline', 'prior_corrupted', 'prior_original'] for b in ARMS[1:]]
            comps = {f'{a}->{b}': compare(scores, a, b, truth, ids) for a, b in pairs}
            saved = report['groups'][group]['cases'][case]
            for a in ARMS:
                assert saved['per_arm_correct'][a] == totals[a]
            for sid, d in per_signer.items():
                for a in ARMS:
                    assert saved['per_signer'][sid][a] == d[a]
            for cls, d in per_class.items():
                for a in ARMS:
                    assert saved['per_class'][cls][a] == d[a]
            for key, d in comps.items():
                got = saved['comparisons'][key]
                assert got['gains_n'] == d['gains_n'] and got['losses_n'] == d['losses_n']
                assert got['top1_changed'] == d['top1_changed']
                assert [r['id'] for r in got['gains']] == d['gains']
                assert [r['id'] for r in got['losses']] == d['losses']
            case_out[case] = {'totals': totals, 'per_signer': per_signer, 'per_class': per_class,
                              'comparisons': comps, 'excluded_frames': 0, 'rows_with_exclusions': 0}
        for i, row in enumerate(rows):
            sid = row['id'] if group == 'native' else row['sample_id']
            assert sid == ids[i] and classes[int(truth[i])] == row['label']
            xy, pr, ts = V.load_raw(row, group, source_hashes)
            for ci, case in enumerate(CASES):
                a, b, t, _ = V.reference_corrupt(xy, pr, ts, sid, case)
                ref = V.independent_restore(a, b, t, 'linear')
                actual = V.timegrid.restore(a, b, t, 'linear')
                assert np.array_equal(ref['present'], actual['present'])
                assert np.array_equal(ref['exact_source_index'], actual['exact_source_index'])
                # Independent rule: enumerate generated grid instants and mark only wholly unsupported.
                keep = np.ones(len(ref['ts']), dtype=np.float32)
                for j, source_index in enumerate(ref['exact_source_index']):
                    if source_index == -1 and not np.count_nonzero(ref['present'][j]):
                        keep[j] = 0
                assert keep[0] == 1 and keep[-1] == 1
                x = V.build_channels(actual['xy'], actual['present'], actual['ts'])['x']
                assert V.array_sha(x) == prior['groups'][group]['cases'][case]['rows'][i]['case']['input_sha256']['linear']
                assert np.all(x[keep == 0, :-1] == 0)
                saved_row = report['groups'][group]['cases'][case]['rows'][i]
                assert saved_row['id'] == sid
                saved_cell = saved_row['cell']
                assert saved_cell['input_sha256'] == V.array_sha(x)
                assert saved_cell['support_sha256'] == V.array_sha(keep)
                assert saved_cell['excluded_indices'] == np.flatnonzero(keep == 0).tolist()
                assert saved_cell['excluded_n'] == int((keep == 0).sum())
                assert saved_cell['frames'] == len(keep)
                d = case_out[case]
                d['excluded_frames'] += int((keep == 0).sum())
                d['rows_with_exclusions'] += int((keep == 0).any())
                out['counts']['cells'] += 1
                if keep.all():
                    for ai in range(1, 4):
                        assert np.array_equal(logits[i, ci, ai], logits[i, ci, 0])
                if i in cfg['repro_indices'][group]:
                    for ai, arm in enumerate(ARMS[1:], 1):
                        prediction = hook_forward(model, x, keep, arm)
                        delta = float(np.max(np.abs(prediction - logits[i, ci, ai])))
                        assert delta <= 1e-5, (group, sid, case, arm, delta)
                        assert np.argmax(prediction) == np.argmax(logits[i, ci, ai])
                        out['counts']['forwards'] += 1
                        out['max_logit_difference'] = max(out['max_logit_difference'], delta)
        for case, d in case_out.items():
            saved_support = report['groups'][group]['cases'][case]['support']
            assert saved_support['excluded_frames_total'] == d['excluded_frames']
            assert saved_support['cells_with_exclusions'] == d['rows_with_exclusions']
        out['groups'][group] = case_out
    assert before == {k: V.array_sha(t.numpy()) for k, t in model.state_dict().items()}
    for p, digest in source_hashes.items():
        assert sha(p) == digest, p
    for p, digest in cfg['pins'].items():
        assert sha(p) == digest, p
    out['source_files_hashed'] = len(source_hashes)
    out['protected_paths_hashed'] = len(cfg['pins'])
    out['primary_burst_nets'] = {g: out['groups'][g]['burst_drop']['totals']['blocks_and_pool'] -
                               out['groups'][g]['burst_drop']['totals']['grid_baseline'] for g in out['groups']}
    out['primary_positive_both_groups'] = all(n > 0 for n in out['primary_burst_nets'].values())
    assert report['outcome']['diagnostic_contribution'] == out['primary_positive_both_groups']
    changed_cells = sum(d['rows_with_exclusions'] for g in out['groups'].values() for d in g.values())
    assert report['execution_counts']['forwards']['new_diagnostic'] == 3 * changed_cells
    assert report['execution_counts']['cells_reused'] == out['counts']['cells'] - changed_cells
    with (H / 'INDEPENDENT_VERIFICATION.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status': out['status'], 'counts': out['counts'],
                      'max_logit_difference': out['max_logit_difference'],
                      'primary_burst_nets': out['primary_burst_nets']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
