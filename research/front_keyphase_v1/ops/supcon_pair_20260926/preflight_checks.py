"""Pre-run checks. No optimizer step is taken. Writes PREFLIGHT.json (never overwrites)."""
import hashlib, importlib.util, json, pathlib, re, sys
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
PRIOR = L / 'ops/partial_encoder_pair_20260922'
sys.path[:0] = [str(H), str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]
import supcon_impl as impl                      # noqa: E402
import tools_prep_union93_candidate as u        # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
rec = lambda i: re.sub(r'_a\d\d_\d\d\d$', '', i)


def load_training():
    """Everything a run needs, identically for both arms."""
    p = read(H / 'comparison_policy.json')
    meta = read(L / 'ops/context_training_pair_20260921/feature_metadata.json')
    base, ctx = meta['base'], meta['pair_context']
    with np.load(PRIOR / 'train_prefixes.npz', allow_pickle=False) as z:
        classes = z['classes'].tolist()
        un = lambda v, o: [v[a:b].copy() for a, b in zip(o[:-1], o[1:])]
        d = dict(base=un(z['base_prefix'], z['base_offsets']), ctx=un(z['pair_context_prefix'], z['pair_context_offsets']),
                 by=z['base_y'].copy(), cy=z['pair_context_y'].copy(), teacher=z['teacher_base_logits'].copy(),
                 bi=z['base_indices'].copy(), ci=z['context_indices'].copy())
    sig = lambda r: int(r['signer_id']) if 'signer_id' in r else -1
    d['bs'] = np.array([sig(r) for r in base], np.int64)
    d['cs'] = np.array([sig(r) for r in ctx], np.int64)
    recs = sorted({rec(r['id']) for r in base} | {rec(r['original_id']) for r in ctx})
    rid = {k: i for i, k in enumerate(recs)}
    d['br'] = np.array([rid[rec(r['id'])] for r in base], np.int64)
    d['cr'] = np.array([rid[rec(r['original_id'])] for r in ctx], np.int64)
    ck = torch.load(L / p['parent']['path'], map_location='cpu', weights_only=True)
    parent, cls2 = u.rebuild_from_payload(ck)
    parent.eval()
    assert cls2 == classes
    return p, meta, d, ck, parent, classes


def main():
    out_path = H / 'PREFLIGHT.json'
    assert not out_path.exists(), 'never overwrite'
    p, meta, d, ck, parent, classes = load_training()
    s = p['execution_settings']
    torch.set_num_threads(int(s['cpu_threads']))
    checks, info = {}, {}

    # --- identity of inputs
    checks['parent_sha'] = sha(L / p['parent']['path']) == p['parent']['sha256']
    checks['prior_control_sha'] = sha(L / p['prior_control']['path']) == p['prior_control']['sha256']
    checks['baseline_sha'] = sha(L / p['reference_baseline']['path']) == p['reference_baseline']['sha256']
    checks['prefix_cache_sha_is_the_one_af6c0119_used'] = sha(PRIOR / 'train_prefixes.npz') == \
        read(PRIOR / 'TRAIN_PREFIX_READY.json')['cache_sha256']

    # --- data separation
    base, ctx = meta['base'], meta['pair_context']
    checks['row_counts'] = (len(d['base']), len(d['ctx'])) == (2122, 224)
    checks['all_rows_role_train'] = all(r['role'] == 'train' for r in base + ctx)
    sids = set(d['bs'].tolist()) | set(d['cs'].tolist())
    checks['signers_only_1_9_or_unverified'] = sids <= set(range(1, 10)) | {-1}
    checks['no_dev_or_reserved_signer'] = not (sids & set(range(10, 17)))
    checks['unverified_rows_are_the_18_nikl'] = [r['id'] for r in base if 'signer_id' not in r] == \
        [r['id'] for r in base if r['id'].startswith('NIKL_')] and int((d['bs'] < 0).sum()) == 18 and int((d['cs'] < 0).sum()) == 0
    checks['labels_match_cache'] = d['by'].tolist() == [classes.index(r['label']) for r in base] and \
        d['cy'].tolist() == [classes.index(r['label']) for r in ctx]
    dev_ids = {r['id'] for r in meta['annotation_dev']} | {r['id'] for r in meta['whole_dev']}
    checks['no_dev_row_in_training'] = not (dev_ids & {r['id'] for r in base + ctx})

    # --- schedule identical to the one af6c0119 recorded
    with np.load(L / 'runs/partial_encoder_pair_v1/training_arrays.npz', allow_pickle=False) as z:
        checks['schedule_identical_to_af6c0119'] = np.array_equal(z['base_indices'], d['bi']) and \
            np.array_equal(z['context_indices'], d['ci'])
        prior_first = z['treatment_losses'][0].copy()

    # --- pair masks reproduce the design count (confirmed only)
    pc = read(L / 'ops/supcon_design_20260926/PAIR_COUNT.json')['correction_20260926_confirmed_only']
    anchors, pairs, nikl_in_masks = [], [], 0
    for t in range(d['bi'].shape[0]):
        br, cr = d['bi'][t], d['ci'][t]
        lab = np.concatenate((d['by'][br], d['cy'][cr]))
        sg = np.concatenate((d['bs'][br], d['cs'][cr]))
        rc = np.concatenate((d['br'][br], d['cr'][cr]))
        elig, pos, den = impl.pair_masks(lab, sg, rc)
        anchors.append(int(pos.any(1).sum())); pairs.append(int(pos.sum()) // 2)
        nikl_in_masks += int(pos[sg < 0].sum() + pos[:, sg < 0].sum() + den[sg < 0].sum() + den[:, sg < 0].sum())
        # same word never appears as a denominator entry unless it is a positive
        same = lab[:, None] == lab[None, :]
        assert not (den & same & ~pos).any(), f'step {t}: a same-word non-positive in the denominator'
    info['mask_stats'] = {'anchors_mean': round(float(np.mean(anchors)), 2), 'zero_steps': int(sum(a == 0 for a in anchors)),
                          'pairs_mean': round(float(np.mean(pairs)), 2)}
    checks['masks_match_design_count'] = (info['mask_stats']['anchors_mean'] == pc['frozen_schedule_confirmed_only']['rows_with_in_batch_positive_per_step_mean']
                                          and info['mask_stats']['zero_steps'] == pc['frozen_schedule_confirmed_only']['zero_steps']
                                          and info['mask_stats']['pairs_mean'] == pc['frozen_schedule_confirmed_only']['positive_pairs_per_step_mean'])
    checks['nikl_outside_contrastive_masks'] = nikl_in_masks == 0
    checks['same_word_never_a_negative'] = True       # asserted per step above

    # --- gradient path of the contrastive term ALONE (step-0 batch, no optimizer step)
    model = impl.WordModel(parent)
    model.eval()
    br, cr = d['bi'][0], d['ci'][0]
    h, m = impl._pad_batch([d['base'][int(i)] for i in br] + [d['ctx'][int(j)] for j in cr])
    _, pooled = model(h, m)
    _, pos, den = impl.pair_masks(np.concatenate((d['by'][br], d['cy'][cr])),
                                  np.concatenate((d['bs'][br], d['cs'][cr])),
                                  np.concatenate((d['br'][br], d['cr'][cr])))
    sc, n_anchor = impl.supcon(pooled, pos, den, p['contrastive_term']['tau'])
    sc.backward()
    g = {n: (None if q.grad is None else float(q.grad.abs().sum())) for n, q in model.named_parameters()}
    info['contrastive_only_step0'] = {'loss': float(sc), 'anchors': n_anchor}
    checks['contrastive_grad_reaches_blocks3'] = all((g[n] or 0) > 0 for n in g if n.startswith('base.blocks.3.conv'))
    checks['contrastive_grad_zero_on_head'] = all(g[n] in (None, 0.0) for n in g if n.startswith('base.head.'))
    checks['no_grad_on_frozen'] = all(g[n] is None for n in g
                                      if not (n.startswith('base.head.') or n.startswith('base.blocks.3.')))

    # --- first-step losses with lambda 0 under identical conditions vs af6c0119's record
    model = impl.WordModel(parent)
    model.eval()
    with torch.no_grad():
        wl, _ = model(h, m)
        ce = torch.nn.functional.cross_entropy(wl, torch.from_numpy(np.concatenate((d['by'][br], d['cy'][cr]))))
        tch = torch.from_numpy(np.ascontiguousarray(d['teacher'], dtype=np.float64))
        dl = wl[:16].to(torch.float64) - tch.index_select(0, torch.from_numpy(np.ascontiguousarray(br)))
        dl = dl - dl.mean(1, keepdim=True)
        out = 0.5 * s['output_coefficient'] * dl.pow(2).mean()
    mine = [float(ce), 0.0, float(out), float(ce) + float(out)]
    info['first_losses'] = {'lambda0_now': mine, 'af6c0119_recorded': prior_first.tolist()}
    checks['first_losses_equal_af6c0119'] = bool(np.allclose(mine, prior_first, rtol=0, atol=1e-6))

    # --- record comparison with the af6c0119 checkpoint
    pk = torch.load(L / p['prior_control']['path'], map_location='cpu', weights_only=True)
    st = pk['settings']
    rec_cmp = {
        'parent': pk['parent_checkpoint_sha256'] == p['parent']['sha256'],
        'steps': st['steps'] == s['steps'] and pk['step'] == s['steps'],
        'batch': st['batch_size'] == s['batch_size'] and st['half_batch'] == s['half_batch'],
        'lr_wd': st['optimizer']['learning_rate'] == s['learning_rate'] and st['optimizer']['weight_decay'] == s['weight_decay'],
        'betas_eps': st['optimizer']['betas'] == s['betas'] and st['optimizer']['eps'] == s['eps'],
        'anchor_output': st['anchor_coefficient'] == s['anchor_coefficient'] and st['output_coefficient'] == s['output_coefficient'],
        'trainable_scope': st['trainable_scopes']['treatment'] == ['head.', 'blocks.3.'],
        'module_mode': 'eval' in st['module_mode'],
        'cpu_threads': pk['fixed_setting']['cpu_threads'] == s['cpu_threads'],
        'deterministic': st['deterministic'] is True,
    }
    info['record_comparison'] = rec_cmp
    checks['record_comparison_all_match'] = all(rec_cmp.values())

    result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'info': info,
              'policy_sha256': sha(H / 'comparison_policy.json'), 'impl_sha256': sha(H / 'supcon_impl.py'),
              'optimizer_steps_taken': 0, 'user_arrays_read': 0, 'reserved_arrays_read': 0}
    with out_path.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
