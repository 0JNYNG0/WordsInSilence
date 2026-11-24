"""Manual no-image forward equations and independent saved-score accounting."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from torch.nn import functional as F

U = Path(__file__).resolve().parent
B = U.parents[1]
R = B/'runs/rgb_input_contribution_v1'
OLD = B/'runs/rgb_tcn_development_pair_v1'
sys.path.insert(0, str(B.parent/'sign_model_pilot_v1/src'))
from pilot.model_masked_v1 import MaskedSignTCN


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    plan = read(U/'FROZEN_DIAGNOSTIC.json')
    assert all(sha(p) == h for p, h in plan['pins'].items())
    complete = read(R/'COMPLETION.json')
    assert sha(R/'SCORES.npz') == complete['scores_sha256']
    assert sha(R/'ROWS.json') == complete['rows_sha256']
    assert sha(U/'FROZEN_DIAGNOSTIC.json') == complete['plan_sha256']
    data, meta = load(R/'SCORES.npz'), read(R/'ROWS.json')
    rows, classes, y = meta['rows'], meta['classes'], data['y']
    keys = ['baseline832', 'A_pose_adapted', 'B_rgb_off', 'B_flags_only', 'B_rgb_on']
    assert len(rows) == len({r['id'] for r in rows}) == 36
    assert y.tolist() == [classes.index(r['label']) for r in rows]
    for k in keys:
        assert data[k].shape == (36, 93) and np.isfinite(data[k]).all()
    original_rows = read(OLD/'EVALUATION_ROWS.json')
    original_scores = load(OLD/'EVALUATION_SCORES.npz')
    assert original_rows['classes'] == classes
    indices = [next(i for i, r in enumerate(original_rows['rows']) if r['id'] == row['id']) for row in rows]
    for current, saved in [('baseline832', 'baseline'), ('A_pose_adapted', 'A'), ('B_rgb_on', 'B')]:
        assert np.array_equal(data[current], original_scores[saved][indices])

    torch.set_num_threads(2)
    parent = torch.load(B/'runs/partial_encoder_pair_v1/checkpoint_control_final.pt', map_location='cpu', weights_only=True)
    st = torch.load(OLD/'checkpoint_B.pt', map_location='cpu', weights_only=True)['state_dict']
    a = torch.load(OLD/'checkpoint_A.pt', map_location='cpu', weights_only=True)['state_dict']
    assert parent['classes'] == classes
    for k, value in parent['state_dict'].items():
        if not k.startswith('head.'):
            assert torch.equal(value, st['base.'+k]) and torch.equal(value, a['base.'+k])
    base = MaskedSignTCN(width=parent['width'], dilations=tuple(parent['dilations']),
                         kernel_size=parent['kernel_size'], dropout=parent['dropout'], n_classes=93)
    base.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith('base.')}, strict=True)
    base.eval().requires_grad_(False)
    errors = {'B_rgb_off': 0., 'B_flags_only': 0.}
    # Independent direct matrices; do not import the adapter, bridge or diagnose module.
    with torch.inference_mode():
        for i, row in enumerate(rows):
            raw = load(row['feature_path'])
            assert sha(row['feature_path']) == row['feature_sha256']
            for name, expected in row['preserved_array_sha256'].items():
                arr = np.ascontiguousarray(raw[name])
                got = hashlib.sha256(str(arr.shape).encode()+arr.dtype.str.encode()+arr.tobytes()).hexdigest()
                assert got == expected
            x = torch.from_numpy(raw['pose_x'])[None]
            n = x.shape[1]
            assert row['frames'] == n and row['observed_images'] == int(raw['observed_rgb'].sum())
            assert not raw['region_estimated'].any()
            mask = torch.ones(1, 1, n)
            h0 = base.proj(x.transpose(1, 2))
            flags = torch.from_numpy(raw['observed_rgb'].astype(np.float32))
            flag_input = F.linear(flags, st['rgb_projection.weight'][:, 1152:1154]).T[None]
            for name, h in [('B_rgb_off', h0.clone()), ('B_flags_only', h0+flag_input)]:
                for block in base.blocks:
                    h = block(h, mask)
                pooled = h.sum(2)/n
                pred = F.linear(pooled, st['base.head.weight'], st['base.head.bias']).numpy()[0]
                error = float(np.max(np.abs(pred-data[name][i])))
                errors[name] = max(errors[name], error)
                assert error <= plan['verification']['atol'], (row['id'], name, error)

    pred = {k: data[k].argmax(1) for k in keys}
    margin, rank = {}, {}
    for k in keys:
        other = data[k].astype(np.float64).copy()
        truth = other[np.arange(36), y].copy()
        other[np.arange(36), y] = -np.inf
        margin[k] = truth-other.max(1)
        rank[k] = (data[k] > truth[:, None]).sum(1)+1
    contrasts = [('baseline832', 'B_rgb_off'), ('A_pose_adapted', 'B_rgb_off'),
                 ('B_rgb_off', 'B_flags_only'), ('B_flags_only', 'B_rgb_on'),
                 ('B_rgb_off', 'B_rgb_on')]
    def group_stats(ids):
        ix = np.array(ids, dtype=int)
        out = {'n': len(ix), 'conditions': {}, 'contrasts': {}}
        for k in keys:
            correct = int((pred[k][ix] == y[ix]).sum())
            out['conditions'][k] = {'correct': correct, 'accuracy_percent': 100*correct/len(ix),
               'mean_true_label_margin': float(margin[k][ix].mean()),
               'mean_true_label_rank': float(rank[k][ix].mean())}
        for left, right in contrasts:
            c0, c1 = pred[left][ix] == y[ix], pred[right][ix] == y[ix]
            delta = data[right][ix].astype(np.float64)-data[left][ix].astype(np.float64)
            dm = margin[right][ix]-margin[left][ix]
            out['contrasts'][right+'_vs_'+left] = {
              'changed_top1': int((pred[left][ix] != pred[right][ix]).sum()),
              'improved_ids': [rows[j]['id'] for j in ix[~c0 & c1]],
              'regressed_ids': [rows[j]['id'] for j in ix[c0 & ~c1]],
              'delta_pp': 100*(int(c1.sum())-int(c0.sum()))/len(ix),
              'mean_true_label_margin_delta': float(dm.mean()),
              'margin_up_over_atol': int((dm > 1e-5).sum()),
              'margin_down_over_atol': int((dm < -1e-5).sum()),
              'max_score_delta': float(abs(delta).max()),
              'centered_score_rms': float(np.sqrt(np.mean((delta-delta.mean(1, keepdims=True))**2)))}
        return out
    groups = {g: group_stats([i for i,r in enumerate(rows) if r['group'] == g])
              for g in ('ksl_primary34', 'ksl_previous2')}
    groups.update({label: group_stats([i for i,r in enumerate(rows) if r['group'] == 'ksl_primary34' and r['label'] == label])
                   for label in ('공부', '먹다')})
    details = []
    for i,row in enumerate(rows):
        details.append({'id': row['id'], 'label': row['label'], 'group': row['group'],
          'conditions': {k: {'top1': classes[int(pred[k][i])], 'correct': bool(pred[k][i] == y[i]),
                             'true_label_margin': float(margin[k][i]), 'true_label_rank': int(rank[k][i])} for k in keys}})
    write(U/'PER_CLIP_RESULT.json', details)
    result = {'status': 'VERIFIED_DIAGNOSTIC_COMPLETE', 'groups': groups,
       'verification': {'manual_max_abs_error': errors, 'all_source_hashes_unchanged': True,
                        'same_non_head_TCN_across_parent_A_B': True, 'saved_on_scores_exact_reuse': True,
                        'all36_included': True},
       'training': False, 'production_adoption': False,
       'interpretation': plan['interpretation'], 'scores_sha256': sha(R/'SCORES.npz')}
    write(U/'VERIFIED_RESULT.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
