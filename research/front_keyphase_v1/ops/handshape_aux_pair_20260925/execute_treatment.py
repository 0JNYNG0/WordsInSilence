"""One fixed treatment run. The control is reused, not retrained."""
import hashlib, importlib.util, json, pathlib, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
R = L / 'runs/handshape_aux_pair_v1'
PRIOR = L / 'ops/partial_encoder_pair_20260922'
sys.path[:0] = [str(H), str(PRIOR), str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


impl = module('aux_impl', H / 'generated_v1.py')
import tools_prep_union93_candidate as u  # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def write(path, v):
    def plain(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {k: plain(y) for k, y in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(y) for y in x]
        return x
    with pathlib.Path(path).open('x', encoding='utf-8') as f:
        json.dump(plain(v), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def unpack(v, off):
    return [v[a:b].copy() for a, b in zip(off[:-1], off[1:])]


def main():
    p = read(H / 'comparison_policy.json')
    ver = read(H / 'ADAPTER_VERIFICATION.json')
    assert p['status'] == 'FROZEN_BEFORE_TRAINING' and ver['status'] == 'PASS'
    assert sha(H / 'generated_v1.py') == ver['module_sha256']
    assert sha(H / 'shape_labels.npz') == ver['labels_sha256']
    assert sha(L / p['parent']['path']) == p['parent']['sha256']
    assert sha(L / p['control']['path']) == p['control']['sha256']
    assert not R.exists(), 'never overwrite a completed run'

    ck = torch.load(L / p['parent']['path'], map_location='cpu', weights_only=True)
    parent, classes = u.rebuild_from_payload(ck)
    parent.eval()
    torch.set_num_threads(int(p['execution_settings']['cpu_threads']))

    with np.load(PRIOR / 'train_prefixes.npz', allow_pickle=False) as z:
        assert z['classes'].tolist() == classes
        base = unpack(z['base_prefix'], z['base_offsets'])
        ctx = unpack(z['pair_context_prefix'], z['pair_context_offsets'])
        by, cy = z['base_y'].copy(), z['pair_context_y'].copy()
        teacher = z['teacher_base_logits'].copy()
        bi, ci = z['base_indices'].copy(), z['context_indices'].copy()
    with np.load(H / 'shape_labels.npz', allow_pickle=False) as z:
        bs, cs = z['base_shape_y'].copy(), z['pair_context_shape_y'].copy()
        codes = z['shape_codes'].tolist()
    assert len(base) == 2122 and len(ctx) == 224 and bi.shape == ci.shape == (660, 16)
    assert len(codes) == p['labels']['shape_classes']

    R.mkdir(exist_ok=False)
    pins = {str(H / n): sha(H / n) for n in
            ['comparison_policy.json', 'generated_v1.py', 'shape_labels.npz',
             'ADAPTER_VERIFICATION.json', 'FROZEN_LABEL_SET.json', 'execute_treatment.py']}
    write(R / 'preflight.json', {
        'status': 'READY_FOR_ONE_TREATMENT_RUN', 'pins': pins,
        'parent': p['parent'], 'control_reused_not_retrained': p['control'],
        'train_rows': {'base': len(base), 'context': len(ctx)},
        'labelled_rows': {'base': int((bs >= 0).sum()), 'context': int((cs >= 0).sum())},
        'user_rows': 0, 'reserved_rows': 0, 'development_rows_in_loss': 0})

    s = p['execution_settings']
    started = time.time()

    def progress(done, total):
        print(json.dumps({'steps': done, 'of': total, 'seconds': round(time.time() - started, 1)}), flush=True)

    out = impl.train_arm(parent, base, by, bs, ctx, cy, cs, teacher, bi, ci,
                         n_shape_classes=len(codes), aux_weight=s['lambda'],
                         learning_rate=s['learning_rate'], weight_decay=s['weight_decay'],
                         anchor_coefficient=s['anchor_coefficient'],
                         output_coefficient=s['output_coefficient'],
                         with_aux=True, progress=progress)
    elapsed = time.time() - started

    st = out['state_dict']
    assert out['updates'] == 660
    init = parent.state_dict()
    changed = sorted(k for k in init if not torch.equal(init[k], st['base.' + k]))
    expect = sorted([f'blocks.3.{n}' for n in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                                               'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']]
                    + ['head.weight', 'head.bias'])
    assert changed == expect, changed
    assert int(torch.count_nonzero(st['shape_head.weight'])) > 0
    assert out['frozen_parameters_unchanged'] and out['buffers_unchanged']
    assert all(torch.isfinite(v).all() for v in st.values() if v.is_floating_point())

    payload = {'kind': 'handshape_auxiliary_candidate_v1', 'classes': classes,
               'shape_codes': codes, 'state_dict': st,
               'parent': p['parent'], 'updates': 660, 'aux_weight': s['lambda'],
               'policy_sha256': sha(H / 'comparison_policy.json'),
               'implementation_sha256': sha(H / 'generated_v1.py'),
               'labels_sha256': sha(H / 'shape_labels.npz'),
               'registered_as_default': False}
    path = R / 'checkpoint_treatment_final.pt'
    with path.open('xb') as f:
        torch.save(payload, f)
    with (R / 'training_arrays.npz').open('xb') as f:
        np.savez_compressed(f, base_indices=bi, context_indices=ci,
                            losses=out['losses'], aux_rows_per_step=out['aux_rows_per_step'])
    for k, v in pins.items():
        assert sha(k) == v, k

    fin = out['losses'][-1]
    write(R / 'training.json', {
        'status': 'TRAINING_COMPLETE_CHECKPOINT_FROZEN_BEFORE_DEVELOPMENT_EVALUATION',
        'at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'candidate': {'path': str(path), 'sha256': sha(path),
                      'changed_parent_tensors': changed,
                      'trainable_names': out['trainable_names'],
                      'trainable_count': sum(int(st[n].numel()) for n in out['trainable_names'])},
        'control_reused_not_retrained': p['control'],
        'parent': p['parent'],
        'aux_weight': s['lambda'],
        'aux_rows_per_step': {'min': int(out['aux_rows_per_step'].min()),
                              'max': int(out['aux_rows_per_step'].max()),
                              'mean': float(out['aux_rows_per_step'].mean()),
                              'of_batch': 32},
        'final_losses': dict(zip(out['loss_columns'], [float(x) for x in fin])),
        'first_losses': dict(zip(out['loss_columns'], [float(x) for x in out['losses'][0]])),
        'anchor_scope': out['anchor_scope'], 'aux_reduction': out['aux_reduction'],
        'aux_init': out['aux_init'],
        'elapsed_training_seconds': round(elapsed, 2),
        'real_training_arms': 1, 'optimizer_updates': 660, 'attempts': 1,
        'seed_or_coefficient_search': 0,
        'policy_sha256': sha(H / 'comparison_policy.json'),
        'arrays_sha256': sha(R / 'training_arrays.npz'),
        'user_arrays_read': 0, 'reserved_arrays_read': 0, 'development_arrays_read': 0,
        'default_promoted': False})
    print('status FROZEN | changed tensors', len(changed), '| seconds', round(elapsed, 2))
    print('losses first ->', dict(zip(out['loss_columns'], [round(float(x), 4) for x in out['losses'][0]])))
    print('losses final ->', dict(zip(out['loss_columns'], [round(float(x), 4) for x in fin])))
    print('aux rows/step mean %.1f of 32' % out['aux_rows_per_step'].mean())
    print('checkpoint', sha(path))


if __name__ == '__main__':
    main()
