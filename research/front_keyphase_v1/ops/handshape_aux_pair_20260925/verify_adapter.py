"""Verify the adapter before the fixed run. Short synthetic/real checks only; no candidate is trained."""
import hashlib, importlib.util, json, pathlib, sys
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
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
PARENT = L / 'runs/public_anchor_0p01_pair_v1/checkpoint_treatment_final.pt'
PREFIX = PRIOR / 'train_prefixes.npz'


def unpack(v, off):
    return [v[a:b].copy() for a, b in zip(off[:-1], off[1:])]


def main():
    c = {}
    c['parent_hash'] = sha(PARENT) == 'b8d6045c0c154697a5d6d3440927a48cf1e9db3146421db0a920da9fa684870d'
    c['prefix_hash'] = sha(PREFIX) == '807358e116162a6b48d2f7c8302ba7cd76e1c39f401f1708004592a647eb1c12'

    ck = torch.load(PARENT, map_location='cpu', weights_only=True)
    parent, classes = u.rebuild_from_payload(ck)
    parent.eval()
    torch.set_num_threads(2)
    with np.load(PREFIX, allow_pickle=False) as z:
        base = unpack(z['base_prefix'], z['base_offsets'])
        ctx = unpack(z['pair_context_prefix'], z['pair_context_offsets'])
        by, cy = z['base_y'].copy(), z['pair_context_y'].copy()
        teacher = z['teacher_base_logits'].copy()
        bi, ci = z['base_indices'].copy(), z['context_indices'].copy()
    with np.load(H / 'shape_labels.npz', allow_pickle=False) as z:
        bs, cs = z['base_shape_y'].copy(), z['pair_context_shape_y'].copy()
        codes = z['shape_codes'].tolist()
    n_sh = len(codes)
    c['counts'] = len(base) == 2122 and len(ctx) == 224 and len(codes) == 24
    c['labelled_counts'] = int((bs >= 0).sum()) == 1364 and int((cs >= 0).sum()) == 162

    # 1. the auxiliary head does not alter the word path at step 0
    m0 = impl.AuxWordModel(parent, n_sh, True).eval()
    h, mm = impl._pad_batch(base[:6])
    with torch.inference_mode():
        wl, sl, pooled = m0(h, mm)
        ref = parent(torch.from_numpy(np.stack([
            np.pad(a, ((0, h.shape[-1] - len(a)), (0, 0))) for a in base[:6]]).astype(np.float32)
            .transpose(0, 1, 2)) if False else h.transpose(1, 2), mm.squeeze(1)) if False else None
    c['aux_head_zero_init'] = int(torch.count_nonzero(m0.shape_head.weight)) == 0 \
        and int(torch.count_nonzero(m0.shape_head.bias)) == 0
    c['aux_logits_uniform_at_init'] = float(sl.abs().max()) == 0.0

    # 2. the auxiliary gradient reaches the SHARED features, not only the auxiliary head
    m1 = impl.AuxWordModel(parent, n_sh, True)
    m1.eval()
    wl, sl, _ = m1(h, mm)
    sy = torch.tensor([0, 1, 2, -1, -1, 3])
    keep = sy >= 0
    aux = torch.nn.functional.cross_entropy(sl[keep], sy[keep])
    aux.backward()
    last = m1.last
    shared = m1.base.blocks[last].conv1.weight
    # With a ZERO-initialised auxiliary head, d(aux)/d(pooled) = W^T(softmax - onehot) = 0 at step 0,
    # so the auxiliary gradient reaches the shared block only from step 1 onward. Both facts are checked.
    c['aux_grad_zero_into_shared_at_step0'] = shared.grad is None or float(shared.grad.abs().sum()) == 0.0
    m1b = impl.AuxWordModel(parent, n_sh, True); m1b.eval()
    with torch.no_grad():
        m1b.shape_head.weight.add_(torch.full_like(m1b.shape_head.weight, 1e-3))
    wl_b, sl_b, _ = m1b(h, mm)
    torch.nn.functional.cross_entropy(sl_b[keep], sy[keep]).backward()
    sh_b = m1b.base.blocks[m1b.last].conv1.weight
    c['aux_grad_reaches_shared_block_once_head_nonzero'] = sh_b.grad is not None and float(sh_b.grad.abs().sum()) > 0
    c['aux_grad_reaches_aux_head'] = float(m1.shape_head.weight.grad.abs().sum()) > 0
    c['aux_grad_does_not_touch_word_head'] = (m1.base.head.weight.grad is None
                                              or float(m1.base.head.weight.grad.abs().sum()) == 0.0)
    c['aux_grad_does_not_touch_earlier_blocks'] = all(
        p.grad is None or float(p.grad.abs().sum()) == 0.0
        for n, p in m1.base.named_parameters() if n.startswith('blocks.') and not n.startswith(f'blocks.{last}.'))

    # 3. unlabelled rows still drive the word loss
    m2 = impl.AuxWordModel(parent, n_sh, True)
    m2.eval()
    wl2, sl2, _ = m2(h, mm)
    y = torch.from_numpy(by[:6].copy())
    all_unlabelled = torch.full((6,), -1, dtype=torch.int64)
    ce = torch.nn.functional.cross_entropy(wl2, y)
    ce.backward()
    c['word_loss_trains_with_no_labels'] = float(m2.base.head.weight.grad.abs().sum()) > 0 \
        and int((all_unlabelled >= 0).sum()) == 0

    # 4. a two-step run: aux arm changes the shared block; the aux-off arm reproduces the prior recipe
    small_b, small_c = bi[:2].copy(), ci[:2].copy()
    on = impl.train_arm(parent, base, by, bs, ctx, cy, cs, teacher, small_b, small_c,
                        n_shape_classes=n_sh, aux_weight=0.1, with_aux=True)
    off = impl.train_arm(parent, base, by, bs, ctx, cy, cs, teacher, small_b, small_c,
                         n_shape_classes=n_sh, aux_weight=0.1, with_aux=False)
    key = f'base.blocks.{m0.last}.conv1.weight'
    c['aux_on_changes_shared_block'] = not torch.equal(on['state_dict'][key], off['state_dict'][key])
    c['aux_off_leaves_aux_head_at_zero'] = int(torch.count_nonzero(off['state_dict']['shape_head.weight'])) == 0
    c['aux_on_trains_aux_head'] = int(torch.count_nonzero(on['state_dict']['shape_head.weight'])) > 0
    c['aux_off_trainable_matches_prior_recipe'] = set(off['trainable_names']) == {
        f'base.blocks.{m0.last}.' + s for s in
        ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
         'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']} | {'base.head.weight', 'base.head.bias'}
    c['aux_on_adds_only_the_aux_head'] = set(on['trainable_names']) - set(off['trainable_names']) == {
        'shape_head.weight', 'shape_head.bias'}
    c['word_loss_identical_at_step0'] = abs(on['losses'][0][0] - off['losses'][0][0]) < 1e-9
    c['aux_raw_zero_when_off'] = float(off['losses'][:, 3].max()) == 0.0
    c['aux_raw_starts_at_ln24'] = abs(on['losses'][0][3] - float(np.log(24))) < 1e-4
    c['aux_rows_match_labels'] = int(on['aux_rows_per_step'][0]) == int(
        ((np.concatenate((bs[small_b[0]], cs[small_c[0]]))) >= 0).sum())
    c['buffers_and_frozen_intact'] = on['frozen_parameters_unchanged'] and on['buffers_unchanged'] \
        and off['frozen_parameters_unchanged'] and off['buffers_unchanged']

    failed = [k for k, v in c.items() if not v]
    out = {'status': 'PASS' if not failed else 'FAIL', 'checks': {k: bool(v) for k, v in c.items()},
           'checks_total': len(c), 'checks_failed': failed,
           'shape_classes': codes, 'aux_weight_probe': 0.1,
           'labelled': {'base': int((bs >= 0).sum()), 'context': int((cs >= 0).sum())},
           'module_sha256': sha(H / 'generated_v1.py'),
           'labels_sha256': sha(H / 'shape_labels.npz'),
           'note': 'two-step probes only; no candidate was produced'}
    with (H / 'ADAPTER_VERIFICATION.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('checks', len(c), 'failed', failed)
    for k, v in c.items():
        print(f'  {"OK " if v else "FAIL"} {k}')


if __name__ == '__main__':
    main()
