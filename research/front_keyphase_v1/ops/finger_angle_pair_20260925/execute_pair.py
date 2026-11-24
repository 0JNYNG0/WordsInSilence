"""One fixed matched-pair training. Outputs are frozen before any development evaluation."""
import hashlib, json, pathlib, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
R = L / 'runs/finger_angle_pair_v1'
sys.path[:0] = [str(H), str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u                # noqa: E402
import angle_block as AB                                # noqa: E402
from generated_v1 import run_pair                       # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def write(path, value):
    def plain(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v
    with pathlib.Path(path).open('x', encoding='utf-8') as f:
        json.dump(plain(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def unpack(v, off):
    return [v[a:b].copy() for a, b in zip(off[:-1], off[1:])]


def main():
    p = read(H / 'comparison_policy.json')
    ready = read(H / 'INPUT_READY.json')
    assert p['status'] == 'FROZEN_BEFORE_TRAINING'
    assert ready['status'] == 'PASS'
    assert sha(H / 'geometry_blocks.npz') == ready['blocks_sha256']
    assert sha(H / 'angle_block.py') == ready['module_sha256']['angle_block.py']
    assert sha(H / 'generated_v1.py') == ready['module_sha256']['generated_v1.py']
    assert sha(p['matched_pair']['parent']['path']) == p['matched_pair']['parent']['sha256']
    assert sha(ready['input_cache']['path']) == ready['input_cache']['sha256']
    assert not R.exists(), 'never overwrite or rerun a completed pair'

    with np.load(ready['input_cache']['path'], allow_pickle=False) as z:
        classes = z['classes'].tolist()
        base_x = unpack(z['base_x'], z['base_offsets'])
        ctx_x = unpack(z['pair_context_x'], z['pair_context_offsets'])
        base_y, ctx_y = z['base_y'].copy(), z['pair_context_y'].copy()
        teacher = z['teacher_base_logits'].copy()
        bidx, cidx = z['base_indices'].copy(), z['context_indices'].copy()
    with np.load(H / 'geometry_blocks.npz', allow_pickle=False) as z:
        assert z['classes'].tolist() == classes
        extra = {f: {'base': unpack(z[f'base_{f}'], z[f'base_{f}_offsets']),
                     'context': unpack(z[f'context_{f}'], z[f'context_{f}_offsets'])}
                 for f in ('angle', 'chord')}
        scales = {f: z[f'{f}_scale'].copy() for f in ('angle', 'chord')}
    assert len(base_x) == 2122 and len(ctx_x) == 224 and len(classes) == 93
    assert bidx.shape == cidx.shape == (660, 16)
    for f in ('angle', 'chord'):
        assert [len(a) for a in extra[f]['base']] == [len(a) for a in base_x]
        assert [len(a) for a in extra[f]['context']] == [len(a) for a in ctx_x]

    ck = torch.load(p['matched_pair']['parent']['path'], map_location='cpu', weights_only=True)
    parent, cl = u.rebuild_from_payload(ck)
    parent.eval()
    torch.set_num_threads(int(p['settings']['cpu_threads']))
    assert cl == classes

    R.mkdir(exist_ok=False)
    pins = {str(H / n): sha(H / n) for n in
            ['comparison_policy.json', 'INPUT_READY.json', 'geometry_blocks.npz',
             'angle_block.py', 'generated_v1.py', 'verify_inputs.py', 'execute_pair.py']}
    write(R / 'preflight.json', {
        'status': 'READY_FOR_ONE_MATCHED_PAIR', 'pins': pins,
        'parent': p['matched_pair']['parent'],
        'control_feature': 'chord', 'treatment_feature': 'angle',
        'train_clips': [len(base_x), len(ctx_x)], 'classes': len(classes),
        'user_training_records': 0, 'reserved_signer_records': 0, 'ksl_records': 0,
        'evaluation_started': False})

    started = time.time()

    def progress(arm, done, total):
        print(json.dumps({'arm': arm, 'steps': done, 'of': total,
                          'seconds': round(time.time() - started, 1)}), flush=True)

    s = p['settings']
    out = run_pair(parent, base_x, base_y, ctx_x, ctx_y, teacher, bidx, cidx,
                   control_extra=extra['chord'], treatment_extra=extra['angle'],
                   control_scale=scales['chord'], treatment_scale=scales['angle'],
                   n_values=AB.EXTRA_VALUES, width=AB.EXTRA_WIDTH,
                   learning_rate=s['learning_rate'], weight_decay=s['weight_decay'],
                   anchor_coefficient=s['anchor_coefficient'],
                   output_coefficient=s['output_coefficient'], progress=progress)
    elapsed = time.time() - started

    initial = parent.state_dict()
    models, arrays = {}, {'base_indices': bidx, 'context_indices': cidx,
                          'angle_scale': scales['angle'], 'chord_scale': scales['chord']}
    for arm in ('control', 'treatment'):
        v = out[arm]
        st = v['state_dict']
        assert v['updates'] == 660
        assert set(v['trainable_names']) == {'extra_weight', 'base.head.weight', 'base.head.bias'}
        changed = [k for k in initial if not torch.equal(initial[k], st['base.' + k])]
        assert set(changed) == {'head.weight', 'head.bias'}, changed
        assert v['extra_weight_nonzero'] and v['frozen_parameters_unchanged'] and v['buffers_unchanged']
        assert all(torch.isfinite(t).all() for t in st.values() if t.is_floating_point())
        count = sum(int(st[n].numel()) for n in v['trainable_names'])
        payload = {'kind': 'research_finger_angle_pair_v1', 'classes': classes, 'state_dict': st,
                   'feature': 'angle' if arm == 'treatment' else 'chord',
                   'extra_value_channels': AB.EXTRA_VALUES, 'extra_width': AB.EXTRA_WIDTH,
                   'parent': p['matched_pair']['parent'], 'updates': 660,
                   'policy_sha256': sha(H / 'comparison_policy.json'),
                   'implementation_sha256': sha(H / 'generated_v1.py'),
                   'feature_module_sha256': sha(H / 'angle_block.py'),
                   'registered_as_default': False}
        path = R / f'checkpoint_{arm}_final.pt'
        with path.open('xb') as f:
            torch.save(payload, f)
        models[arm] = {'path': str(path), 'sha256': sha(path), 'feature': payload['feature'],
                       'changed_parent_tensors': changed,
                       'trainable_names': v['trainable_names'], 'trainable_count': count,
                       'added_projection_nonzero': True,
                       'frozen_encoder_and_buffers_unchanged': True}
        arrays[arm + '_losses'] = v['losses']
    assert models['control']['trainable_count'] == models['treatment']['trainable_count']

    with (R / 'training_arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    for k, d in pins.items():
        assert sha(k) == d, k
    assert sha(p['matched_pair']['parent']['path']) == p['matched_pair']['parent']['sha256']

    write(R / 'training.json', {
        'status': 'TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION',
        'at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'models': models, 'settings': out['settings'],
        'trainable_count_matched': models['control']['trainable_count'],
        'final_losses': {arm: {k: float(v) for k, v in
                               zip(['ce', 'anchor', 'output', 'total'], out[arm]['losses'][-1])}
                         for arm in ('control', 'treatment')},
        'elapsed_training_seconds': round(elapsed, 2),
        'real_training_arms': 2, 'optimizer_updates': 1320, 'attempts': 1,
        'retry_for_accuracy': 0, 'seed_or_coefficient_search': 0,
        'policy_sha256': sha(H / 'comparison_policy.json'),
        'arrays_sha256': sha(R / 'training_arrays.npz'),
        'user_arrays_read': 0, 'development_arrays_read': 0, 'reserved_arrays_read': 0,
        'default_promoted': False})
    print('status BOTH_FROZEN | trainable each', models['control']['trainable_count'],
          '| seconds', round(elapsed, 2))
    for arm in ('control', 'treatment'):
        print(f"  {arm:9s} {models[arm]['feature']:6s} {models[arm]['sha256'][:16]}")


if __name__ == '__main__':
    main()
