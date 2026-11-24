"""Append local hand channels to frozen global49 inputs, without model inference."""
import json

import numpy as np
import torch

import runtime_binding as rb

SRC = rb.Path(rb.P['source_unit']) / 'prepared'
OUTPUTS = ('training.npz', 'ROW_DIAGNOSTICS.json', 'TRAINING_META.json',
           'evaluation.npz', 'EVALUATION_INPUT_DIAGNOSTICS.json',
           'EVALUATION_ROWS.json', 'INPUT_PREPARATION.json')


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def load_rows(rows, specs, tf, old, old_diags):
    if len(rows) != len(old_diags) or old['x'].shape != (len(rows), 6, 2, 10, 49):
        raise ValueError('source row/diagnostics/cache shapes disagree')
    xs, indices, diags = [], [], []
    for i, row in enumerate(rows):
        x, diag = rb.load_local_row(row, specs, tf)
        previous = old_diags[i]
        p, q = diag['prepare_diag'], previous['prepare_diag']
        if previous['id'] != row['id']:
            raise ValueError('source diagnostic order differs')
        if p['T'] != row['end_exclusive'] - row['start'] or p['T'] != q['T']:
            raise ValueError('source span changed')
        if p['indices'] != q['indices']:
            raise ValueError('sampled frame indices changed')
        if not exact(x[:, :2].numpy(), old['x'][i]):
            raise ValueError('global channels changed at row %s' % row['id'])
        if diag['global_input_tensor_sha256'] != previous['input_tensor_sha256']:
            raise ValueError('global input hash changed')
        for key in ('source_sha256', 'layout', 'slice', 'mask_hand', 'hand_slice'):
            if diag[key] != previous[key]:
                raise ValueError('%s changed in row %s' % (key, row['id']))
        xs.append(x)
        indices.append(p['indices'])
        diags.append({'id': row['id'], **diag, 'global2_exact': True})
    valid = rb.window_bridge.window_valid_from_indices(indices)
    if not exact(valid.numpy(), old['window_valid']) or not bool(valid.any(dim=1).all()):
        raise ValueError('valid-window mask changed or empty row')
    for d, v in zip(diags, valid):
        d['valid_windows'] = int(v.sum())
    return torch.stack(xs), valid, diags


def main():
    if any((rb.PREP / n).exists() for n in OUTPUTS):
        raise SystemExit('prepared artifacts already exist; no overwrite or automatic rebuild')
    rb.configure_cpu()
    plan, _, meta, _, _, tf = rb.setup()
    specs = rb.layout_specs49(meta['layout_specs'])
    old = rb.load_npz(SRC / 'training.npz')
    expected = {'x', 'y', 'weight', 'fit', 'internal_val', 'primary', 'window_valid'}
    if set(old) != expected or rb.sha(SRC / 'training.npz') != meta['cache_sha256']:
        raise ValueError('source training cache keys/hash mismatch')
    if len(meta['rows']) != 2346:
        raise ValueError('training row count changed')
    x, valid, diags = load_rows(meta['rows'], specs, tf, old, rb.read(SRC / 'ROW_DIAGNOSTICS.json'))

    rows_doc = rb.read(SRC / 'EVALUATION_ROWS.json')
    eval_old = rb.load_npz(SRC / 'evaluation.npz')
    eval_diags = rb.read(SRC / 'EVALUATION_INPUT_DIAGNOSTICS.json')
    if rows_doc['classes'] != meta['classes'] or len(rows_doc['rows']) != 855:
        raise ValueError('evaluation rows/classes changed')
    if set(eval_old) != {'x', 'y', 'window_valid'}:
        raise ValueError('unexpected evaluation cache keys')
    if eval_old['y'].tolist() != [r['y'] for r in rows_doc['rows']]:
        raise ValueError('evaluation row labels disagree')
    ex, ev, ediags = load_rows(rows_doc['rows'], specs, tf, eval_old, eval_diags)

    copied = {k: v for k, v in old.items() if k != 'x'}
    rb.write_npz(rb.PREP / 'training.npz', x=x.numpy(), **copied)
    check = rb.load_npz(rb.PREP / 'training.npz')
    if not all(exact(check[k], v) for k, v in copied.items()):
        raise ValueError('saved non-input arrays changed')
    rb.write_json(rb.PREP / 'ROW_DIAGNOSTICS.json', diags)
    new_meta = dict(meta)
    new_meta.update(cache_sha256=rb.sha(rb.PREP / 'training.npz'),
                    shape=list(x.shape), input_policy=plan['input_policy'])
    new_meta['localshape'] = {
        'source_cache_sha256': meta['cache_sha256'],
        'source_meta_sha256': rb.sha(SRC / 'TRAINING_META.json'),
        'row_diagnostics_sha256': rb.sha(rb.PREP / 'ROW_DIAGNOSTICS.json'),
        'inference_shape': list(rb.INFERENCE_SHAPE), 'global2_training_exact': len(diags)}
    rb.write_json(rb.PREP / 'TRAINING_META.json', new_meta)
    rb.write_npz(rb.PREP / 'evaluation.npz', x=ex.numpy(), y=eval_old['y'], window_valid=ev.numpy())
    rb.write_json(rb.PREP / 'EVALUATION_INPUT_DIAGNOSTICS.json', ediags)
    rb.write_bytes(rb.PREP / 'EVALUATION_ROWS.json', (SRC / 'EVALUATION_ROWS.json').read_bytes())
    record = {
        'status': 'PASS', 'created_utc': rb.now(), 'plan_sha256': rb.sha(rb.U / 'FROZEN_EXPERIMENT.json'),
        'script_sha256': rb.sha(__file__), 'training_rows': len(diags), 'evaluation_rows': len(ediags),
        'global2_training_exact': len(diags), 'global2_evaluation_exact': len(ediags),
        'training_valid_windows': int(valid.sum()), 'evaluation_valid_windows': int(ev.sum()),
        'non_input_arrays_exact': True, 'scores_read': False, 'model_forward_calls': 0,
        'outputs': {n: rb.sha(rb.PREP / n) for n in OUTPUTS[:-1]},
        'source_files': {str(SRC / n): rb.sha(SRC / n) for n in OUTPUTS[:-1]}}
    rb.write_json(rb.PREP / 'INPUT_PREPARATION.json', record)
    print(json.dumps({k: record[k] for k in ('status', 'training_rows', 'evaluation_rows',
                                           'global2_training_exact', 'global2_evaluation_exact')}))


if __name__ == '__main__':
    main()
