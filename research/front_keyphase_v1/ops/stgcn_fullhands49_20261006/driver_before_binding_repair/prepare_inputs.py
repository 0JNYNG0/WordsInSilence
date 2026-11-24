# Build graph49 inputs from the accepted 27-joint rows. CPU only: no model forwards, no training, no scores.
import json

import numpy as np
import torch

import runtime_binding as rb

SRC = rb.Path(rb.P['source_unit']) / 'prepared'
REF = rb.Path(rb.P['reference_run'])
SOURCE_ARRAYS = ['fit', 'internal_val', 'primary', 'weight', 'x27', 'y']


def load_rows(rows, specs, prepare):
    xs, indices, diags = [], [], []
    for row in rows:
        rb.check_row_path(row)
        x, diag = rb.gp.load_row(row, specs, prepare)
        prep = diag['prepare_diag']
        length = int(row['end_exclusive']) - int(row['start'])
        if tuple(x.shape) != rb.INFERENCE_SHAPE or x.dtype != torch.float32:
            raise SystemExit('unexpected x49 shape/dtype for row %s' % row['id'])
        if int(prep['T']) != length or int(np.asarray(prep['indices']).max()) >= length:
            raise SystemExit('prepared length/indices disagree with source span for row %s' % row['id'])
        xs.append(x)
        indices.append(prep['indices'])
        diags.append({'id': row['id'], 'layout': row.get('layout'), 'T': length,
                      'x49_sha256': rb.gp.tensor_sha256(x)})
    valid = rb.window_bridge.window_valid_from_indices(indices)
    if tuple(valid.shape) != (len(rows), 6) or valid.dtype != torch.bool or not bool(valid.any(dim=1).all()):
        raise SystemExit('window_valid shape/dtype mismatch or row without a valid window')
    for d, v in zip(diags, valid):
        d['valid_windows'] = int(v.sum())
    return torch.stack(xs), valid, diags


def main():
    for name in ('training.npz', 'evaluation.npz'):
        if (rb.PREP / name).exists():
            raise SystemExit('prepared/%s already exists; refusing to rebuild' % name)
    torch.set_num_threads(2)
    _plan, _cfg, src_meta, _factory, _augment, tf = rb.setup()
    prepare, specs = rb.make_prepare(tf), rb.layout_specs49(src_meta['layout_specs'])
    old_to_new = list(rb.fullhands49.OLD_TO_NEW)

    src_cache = SRC / 'training.npz'
    if rb.sha(src_cache) != src_meta['cache_sha256']:
        raise SystemExit('source cache hash differs from source TRAINING_META')
    src = rb.load_npz(src_cache)
    rows = src_meta['rows']
    if sorted(src) != SOURCE_ARRAYS or src['x27'].shape != (len(rows), 6, 2, 10, 27):
        raise SystemExit('unexpected source cache layout')
    x49, valid, diags = load_rows(rows, specs, prepare)
    x27 = torch.from_numpy(src['x27'])
    train_equal = [bool(torch.equal(x49[i][..., old_to_new], x27[i])) for i in range(len(rows))]
    if not all(train_equal):
        raise SystemExit('common27 mismatch in %d training rows' % train_equal.count(False))
    for d in diags:
        d['common27_equal'] = True

    ref_rows_path, old_diag_path = REF / 'EVALUATION_ROWS.json', REF / 'EVALUATION_INPUT_DIAGNOSTICS.json'
    ref_doc, old_diag = rb.read(ref_rows_path), rb.read(old_diag_path)
    old_rows = old_diag['rows'] if isinstance(old_diag, dict) else old_diag
    if ref_doc['classes'] != src_meta['classes'] or len(ref_doc['rows']) != 855 or len(old_rows) != 855:
        raise SystemExit('reference evaluation rows/classes do not match the frozen plan')
    ex49, evalid, ediags = load_rows(ref_doc['rows'], specs, prepare)
    for i, (d, old) in enumerate(zip(ediags, old_rows)):
        if old.get('id', d['id']) != d['id']:
            raise SystemExit('evaluation diagnostics row order differs at %d' % i)
        d['input27_sha256'] = rb.gp.tensor_sha256(ex49[i][..., old_to_new].contiguous())
        d['common27_equal'] = d['input27_sha256'] == old['input_tensor_sha256']
    eval_equal = sum(d['common27_equal'] for d in ediags)
    if eval_equal != len(ediags):
        raise SystemExit('common27 hash mismatch in %d evaluation rows' % (len(ediags) - eval_equal))
    ey = np.asarray([int(r['y']) for r in ref_doc['rows']], dtype=np.int64)

    copied = {k: src[k] for k in SOURCE_ARRAYS if k != 'x27'}
    rb.write_npz(rb.PREP / 'training.npz', x49=x49.numpy(), window_valid=valid.numpy(), **copied)
    check = rb.load_npz(rb.PREP / 'training.npz')
    if any(check[k].dtype != v.dtype or check[k].tobytes() != v.tobytes() for k, v in copied.items()):
        raise SystemExit('copied source arrays are not byte-exact')
    rb.write_json(rb.PREP / 'ROW_DIAGNOSTICS.json', {'rows': diags, 'common27_exact_rows': len(diags)})
    meta = dict(src_meta)
    meta['cache_sha256'] = rb.sha(rb.PREP / 'training.npz')
    meta['fullhands49'] = {
        'source_cache_sha256': src_meta['cache_sha256'],
        'source_meta_sha256': rb.sha(SRC / 'TRAINING_META.json'),
        'layout_specs49': specs, 'old_to_new': old_to_new, 'inference_shape': list(rb.INFERENCE_SHAPE),
        'row_diagnostics_sha256': rb.sha(rb.PREP / 'ROW_DIAGNOSTICS.json'),
    }
    rb.write_json(rb.PREP / 'TRAINING_META.json', meta)

    rb.write_npz(rb.PREP / 'evaluation.npz', x49=ex49.numpy(), window_valid=evalid.numpy(), y=ey)
    rb.write_json(rb.PREP / 'EVALUATION_INPUT_DIAGNOSTICS.json', {'rows': ediags, 'common27_hash_matches': eval_equal})
    rb.write_bytes(rb.PREP / 'EVALUATION_ROWS.json', ref_rows_path.read_bytes())

    outputs = ('training.npz', 'ROW_DIAGNOSTICS.json', 'TRAINING_META.json', 'evaluation.npz',
               'EVALUATION_INPUT_DIAGNOSTICS.json', 'EVALUATION_ROWS.json')
    record = {
        'created_utc': rb.now(),
        'plan_sha256': rb.sha(rb.U / 'FROZEN_EXPERIMENT.json'),
        'script_sha256': rb.sha(__file__),
        'counts': {'training_rows': len(rows), 'evaluation_rows': len(ediags),
                   'training_valid_windows': int(valid.sum()), 'evaluation_valid_windows': int(evalid.sum())},
        'common27': {'training_exact_rows': len(diags), 'evaluation_hash_matches': eval_equal},
        'tensor_sha256': {'train_x49': rb.gp.tensor_sha256(x49), 'train_window_valid': rb.gp.tensor_sha256(valid),
                          'eval_x49': rb.gp.tensor_sha256(ex49), 'eval_window_valid': rb.gp.tensor_sha256(evalid)},
        'outputs': {n: rb.sha(rb.PREP / n) for n in outputs},
        'source_files': {str(p): rb.sha(p) for p in (SRC / 'TRAINING_META.json', src_cache, ref_rows_path, old_diag_path)},
    }
    rb.write_json(rb.PREP / 'INPUT_PREPARATION.json', record)
    print(json.dumps({k: record[k] for k in ('counts', 'common27', 'outputs')}, indent=2))


if __name__ == '__main__':
    main()
