"""Codex independent verifier. No candidate helper imports, no training/camera/network."""
import hashlib
import json
import pathlib
import sys
import numpy as np

H = pathlib.Path(__file__).resolve().parent
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
cfg = read(H / 'CONFIG_LOCAL_ONLY.json')
r = read(H / 'result/RESULT.json')
hashes = read(H / 'result/SOURCE_HASHES.json')
z = np.load(H / 'result/SCORES.npz', allow_pickle=False)
meta = read(cfg['metadata'])['annotation_dev']
arms = z['arms'].tolist()
classes = z['classes'].tolist()
scores = z['logits']
truth = np.array([classes.index(m['label']) for m in meta])
assert scores.shape == (699, 4, 93) and np.isfinite(scores).all()
assert z['ids'].tolist() == [m['id'] for m in meta] == [x['id'] for x in r['rows']]
assert np.array_equal(truth, z['true_indices'])
pred = scores.argmax(axis=2)
correct = pred == truth[:, None]
totals = dict(zip(arms, correct.sum(axis=0).tolist()))
assert totals == r['per_arm_correct']
old = np.load(cfg['saved_logits'], allow_pickle=False)
assert np.array_equal(scores[:, 0], old['original_logits'])
assert np.array_equal(scores[:, 1], old['decimated_logits'])
for i, row in enumerate(r['rows']):
    for a, arm in enumerate(arms):
        order = np.argsort(-scores[i, a], kind='stable')
        other = np.delete(scores[i, a], truth[i]).max()
        item = row['arms'][arm]
        assert item['pred'] == classes[pred[i, a]]
        assert item['correct'] == bool(correct[i, a])
        assert item['rank'] == int(np.where(order == truth[i])[0][0]) + 1
        assert abs(item['margin'] - float(scores[i, a, truth[i]] - other)) < 1e-12
for name, item in dict(r['comparisons'], **r['context_comparisons']).items():
    a, b = [arms.index(k) for k in name.split('->')]
    gains = ~correct[:, a] & correct[:, b]
    losses = correct[:, a] & ~correct[:, b]
    assert item['gains_n'] == int(gains.sum())
    assert item['losses_n'] == int(losses.sum())
    assert item['top1_changed'] == int((pred[:, a] != pred[:, b]).sum())
    assert {x['id'] for x in item['gains']} == {meta[i]['id'] for i in np.where(gains)[0]}
    assert {x['id'] for x in item['losses']} == {meta[i]['id'] for i in np.where(losses)[0]}
for signer in (11, 12, 13):
    ix = [i for i, m in enumerate(meta) if m['signer_id'] == signer]
    assert dict(n=len(ix), **dict(zip(arms, correct[ix].sum(axis=0).tolist()))) == r['per_signer'][str(signer)]
for p, digest in cfg['pins'].items():
    assert sha(p) == digest
assert hashes['pins']['before'] == hashes['pins']['after'] == cfg['pins']
assert hashes['sources']['before'] == hashes['sources']['after']
for m in meta:
    assert sha(m['path']) == m['sha256'] == hashes['sources']['before'][m['id']]['sha256']
assert {m['signer_id'] for m in meta} == {11, 12, 13}
assert {m['role'] for m in meta} == {'validation'}
assert sha(H / 'temporal_factor_audit.py') == r['hashes']['code']
assert sha(H / 'CONFIG_LOCAL_ONLY.json') == r['hashes']['config']

sys.path[:0] = [cfg['lab_root'], cfg['pilot_src']]
from pilot.data import build_channels
import tools_prep_union93_candidate as u
import torch
torch.set_num_threads(2)
model, cls = u.rebuild_from_payload(torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True))
assert cls == classes
model.eval()
probe_results = []
mask_differences = 0
scale_relative_changes = {k: [] for k in ('body', 'left_hand', 'right_hand')}
for i, m in enumerate(meta):
    with np.load(m['path'], allow_pickle=False) as raw:
        xy, pr, ts = raw['xy_native'], raw['coordinate_present'], raw['timestamp_nominal_us']
        a = build_channels(xy, pr, ts)
        b = build_channels(xy[::2], pr[::2], ts[::2])
    original_scales = [a['body_scale'], a['hand_scales']['left'], a['hand_scales']['right']]
    sparse_scales = [b['body_scale'], b['hand_scales']['left'], b['hand_scales']['right']]
    for key, x, y in zip(scale_relative_changes, original_scales, sparse_scales):
        val = r['rows'][i]['scales']['original'][key]
        assert (val is None and not np.isfinite(x)) or val == x
        val2 = r['rows'][i]['scales']['decimated'][key]
        assert (val2 is None and not np.isfinite(y)) or val2 == y
        if np.isfinite(x) and np.isfinite(y):
            scale_relative_changes[key].append(abs(y / x - 1))
    keep = np.arange(0, len(ts), 2)
    d = np.take(a['x'], keep, axis=0)
    c = d.copy()
    c[:, 279] = 0
    c[1:, 279] = (ts[keep[1:]] - ts[keep[:-1]]).astype(np.float32) / 1e6
    mask_diff = int(np.count_nonzero(b['x'][:, 186:279] != c[:, 186:279]))
    assert mask_diff == r['rows'][i]['b_vs_c_input']['all_valid_186_279']['entries_differ']
    mask_differences += mask_diff
    assert np.array_equal(c[:, 279], b['x'][:, 279])
    if i in cfg['repro_indices']:
        for arm_index, x in ((2, c), (3, d)):
            with torch.inference_mode():
                actual = model(torch.from_numpy(x[None]), torch.ones((1, len(x)), dtype=torch.float32))[0].numpy()
            delta = float(np.max(np.abs(actual - scores[i, arm_index])))
            assert delta <= 1e-5
            probe_results.append({'index': i, 'arm': arms[arm_index], 'max_abs_logit_delta': delta})
scale_summary = {k: {'median_absolute_relative_change': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'max': float(max(v))}
                 for k, v in scale_relative_changes.items()}
subset = correct[:, 0] & ~correct[:, 1]
out = {'status': 'PASS', 'reviewer': 'Codex', 'all_rows_all_arms_metrics_verified': 699,
       'input_normalization_dt_masks_rebuilt_rows': 699, 'mask_entries_differ': mask_differences,
       'new_forwards': len(probe_results), 'probes': probe_results, 'per_arm_correct': totals,
       'scale_change': scale_summary, 'source_and_pin_hashes_checked': [699, len(cfg['pins'])],
       'model_sha256': sha(cfg['checkpoint']), 'training': 0, 'camera': 0, 'user_data': 0,
       'original_new_errors_after_decimation': int(subset.sum()),
       'same_errors_recovered_by_fixed_scale': int((subset & correct[:, 2]).sum()),
       'same_errors_recovered_by_fixed_scale_old_dt': int((subset & correct[:, 3]).sum())}
with (H / 'INDEPENDENT_VERIFICATION.json').open('x', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps(out, ensure_ascii=False))
