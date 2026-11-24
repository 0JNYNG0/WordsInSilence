"""The two fixed runs: control reproduction (lambda 0) then candidate (lambda 0.1). Nothing else.
Writes runs/supcon_pair_v1/ (never overwrites). Then compares the reproduction with af6c0119."""
import hashlib, json, pathlib, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
R = L / 'runs/supcon_pair_v1'
sys.path[:0] = [str(H)]
import supcon_impl as impl                     # noqa: E402
from preflight_checks import load_training, sha, read   # noqa: E402


def write(path, v):
    with pathlib.Path(path).open('x', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=1, allow_nan=False, default=float)
        f.write('\n')


def main():
    pre = read(H / 'PREFLIGHT.json')
    assert pre['status'] == 'PASS'
    assert pre['policy_sha256'] == sha(H / 'comparison_policy.json') and pre['impl_sha256'] == sha(H / 'supcon_impl.py')
    assert not R.exists(), 'never overwrite a completed run'
    p, meta, d, ck, parent, classes = load_training()
    s, c = p['execution_settings'], p['contrastive_term']
    torch.set_num_threads(int(s['cpu_threads']))
    R.mkdir(parents=True)
    records = {}
    expect = sorted([f'base.blocks.3.{n}' for n in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                                                    'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']]
                    + ['base.head.weight', 'base.head.bias'])
    for arm, lam in (('control_reproduction', 0.0), ('candidate', float(c['lambda']))):
        t0 = time.time()
        out = impl.train_arm(parent, d['base'], d['by'], d['bs'], d['br'], d['ctx'], d['cy'], d['cs'], d['cr'],
                             d['teacher'], d['bi'], d['ci'], lam=lam, tau=float(c['tau']),
                             learning_rate=s['learning_rate'], weight_decay=s['weight_decay'],
                             anchor_coefficient=s['anchor_coefficient'], output_coefficient=s['output_coefficient'],
                             progress=lambda a, b: print(json.dumps({'arm': arm, 'steps': a, 'of': b}), flush=True))
        el = time.time() - t0
        st = out['state_dict']
        init = parent.state_dict()
        changed = sorted('base.' + k for k in init if not torch.equal(init[k], st['base.' + k]))
        assert out['updates'] == 660 and out['trainable_names'] == expect and changed == expect, changed
        assert out['frozen_parameters_unchanged'] and out['buffers_unchanged']
        # plain MaskedSignTCN state dict so the web/eval loaders can read it like af6c0119
        plain = {k[len('base.'):]: v for k, v in st.items()}
        payload = {**{k: v for k, v in ck.items() if k not in ('state_dict',)},
                   'state_dict': plain, 'kind': f'supcon_pair_{arm}', 'step': 660,
                   'parent_checkpoint_sha256': p['parent']['sha256'],
                   'contrastive': {'lambda': lam, 'tau': float(c['tau']), 'term_in_graph': out['term_in_graph']},
                   'policy_sha256': sha(H / 'comparison_policy.json'), 'impl_sha256': sha(H / 'supcon_impl.py'),
                   'registered_as_default': False,
                   'is_not': 'a promoted model; nothing in the web registry points at this file'}
        path = R / f'checkpoint_{arm}.pt'
        with path.open('xb') as f:
            torch.save(payload, f)
        records[arm] = {'path': str(path), 'sha256': sha(path), 'lambda': lam, 'seconds': round(el, 2),
                        'changed_tensors': changed, 'frozen_unchanged': True, 'buffers_unchanged': True,
                        'first_losses': dict(zip(out['loss_columns'], out['losses'][0].tolist())),
                        'final_losses': dict(zip(out['loss_columns'], out['losses'][-1].tolist())),
                        'anchors_per_step_mean': float(out['anchors_per_step'].mean())}
        np.save(R / f'losses_{arm}.npy', out['losses'])
        print(arm, 'done', round(el, 1), 's', records[arm]['sha256'][:16], flush=True)

    # ---- reproduction vs af6c0119 (bitwise), and the loss trajectory against its record
    prior = torch.load(L / p['prior_control']['path'], map_location='cpu', weights_only=True)['state_dict']
    repro = torch.load(records['control_reproduction']['path'], map_location='cpu', weights_only=True)['state_dict']
    names = [n[len('base.'):] for n in expect]
    tensor_cmp = {n: {'bitwise_equal': bool(torch.equal(prior[n], repro[n])),
                      'max_abs_diff': float((prior[n] - repro[n]).abs().max())} for n in names}
    all_else_equal = all(torch.equal(prior[k], repro[k]) for k in prior if k not in names)
    with np.load(L / 'runs/partial_encoder_pair_v1/training_arrays.npz', allow_pickle=False) as z:
        rec_losses = z['treatment_losses']
    mine = np.load(R / 'losses_control_reproduction.npy')[:, [0, 1, 2, 4]]
    diff = np.abs(mine - rec_losses)
    first_diff_step = int(np.argmax(diff.max(1) > 0)) if (diff.max(1) > 0).any() else None
    repro_cmp = {'trainable_tensors': tensor_cmp,
                 'all_trainable_bitwise_equal': all(v['bitwise_equal'] for v in tensor_cmp.values()),
                 'non_trainable_tensors_equal': all_else_equal,
                 'loss_trajectory_max_abs_diff': float(diff.max()),
                 'first_step_with_any_loss_difference': first_diff_step,
                 'loss_trajectory_bitwise_equal': bool(diff.max() == 0)}
    write(R / 'training.json', {
        'status': 'TRAINING_COMPLETE_CHECKPOINTS_FROZEN_BEFORE_EVALUATION',
        'at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'runs': records, 'training_runs': 2, 'optimizer_updates_each': 660,
        'reproduction_vs_af6c0119': repro_cmp,
        'policy_sha256': sha(H / 'comparison_policy.json'), 'impl_sha256': sha(H / 'supcon_impl.py'),
        'user_arrays_read': 0, 'reserved_arrays_read': 0, 'development_arrays_read': 0,
        'coefficient_or_seed_search': 0, 'default_promoted': False})
    print(json.dumps(repro_cmp, indent=1)[:1500])


if __name__ == '__main__':
    main()
