"""Public evaluation after the candidate is frozen. No result-driven change is made.

The candidate's encoder is byte-identical to the reused control, so the frozen pooled
features already computed for the control are the candidate's pooled features too. That
equality is VERIFIED here against two independent caches before any score is used; only
the final affine (head) differs, and it is recomputed for both arms.
"""
import numpy as np
import torch
from local_io import H, L, R, read, sha, write, policy, parent_model
from wis_lab.service.neural import PROJECT_18

GROUPS = ['annotation_dev', 'whole_dev', 'public', 'mp']
TARGET_NAMES = set(PROJECT_18.values())


def summarize(rows, arm, scope):
    """Identical counting to ops/partial_encoder_pair_20260922/evaluate_pair.py."""
    rs = [r for r in rows if scope in r['models'][arm]]
    labels = sorted({r['true'] for r in rs})
    detail, recalls, f1s = {}, [], []
    for label in labels:
        support = sum(r['true'] == label for r in rs)
        tp = sum(r['true'] == label and r['models'][arm][scope]['pred'] == label for r in rs)
        predicted = sum(r['models'][arm][scope]['pred'] == label for r in rs)
        recall = tp / support
        precision = tp / predicted if predicted else 0.
        f1 = 2 * tp / (support + predicted)
        detail[label] = dict(n=support, correct=tp, predicted=predicted,
                             recall=recall, precision=precision, f1=f1)
        recalls.append(recall)
        f1s.append(f1)
    people = {str(s): dict(n=sum(r.get('signer_id') == s for r in rs),
                           correct=sum(r.get('signer_id') == s and r['models'][arm][scope]['correct'] for r in rs))
              for s in sorted({r['signer_id'] for r in rs if 'signer_id' in r})}
    return dict(n=len(rs), correct=sum(r['models'][arm][scope]['correct'] for r in rs),
                ground_truth_classes=len(labels),
                macro_recall=float(np.mean(recalls)) if recalls else None,
                macro_f1=float(np.mean(f1s)) if f1s else None,
                per_class=detail, per_signer=people)


def main():
    p = policy()
    train = read(R / 'training.json')
    assert train['status'] == 'TRAINING_COMPLETE_CHECKPOINT_FROZEN_BEFORE_DEVELOPMENT_EVALUATION'
    assert sha(H / 'local_policy.json') == train['policy_sha256']
    assert sha(train['candidate']['path']) == train['candidate']['sha256']

    ck, parent, classes = parent_model(p)
    heads, payloads = {}, {}
    for arm, path in [('control', p['control']['path']), ('candidate', train['candidate']['path'])]:
        payloads[arm] = torch.load(path, map_location='cpu', weights_only=True)
        sd = payloads[arm]['state_dict']
        heads[arm] = (sd['head.weight'].numpy().astype(float), sd['head.bias'].numpy().astype(float))

    checks = {}
    # the only difference between the two arms must be the head
    checks['candidate_encoder_identical_to_control'] = all(
        torch.equal(payloads['candidate']['state_dict'][k], payloads['control']['state_dict'][k])
        for k in payloads['candidate']['state_dict'] if not k.startswith('head.'))
    checks['candidate_head_differs_from_control'] = not np.array_equal(heads['candidate'][0], heads['control'][0])
    checks['candidate_encoder_identical_to_parent'] = all(
        torch.equal(payloads['candidate']['state_dict'][k], ck['state_dict'][k])
        for k in payloads['candidate']['state_dict'] if not k.startswith('head.'))

    meta = read(p['paths']['metadata'])
    old = read(p['control']['evaluation'])
    assert old['status'] == 'PUBLIC_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT'
    oldmap = {(r['group'], r['id'], r['condition']): r for r in old['rows']}

    pooled = {}
    with np.load(p['paths']['control_eval_arrays'], allow_pickle=False) as ev, \
            np.load(p['paths']['pooled_cache'], allow_pickle=False) as fc:
        assert ev['classes'].tolist() == classes == fc['classes'].tolist()
        maxdiff = 0.
        for g in GROUPS:
            a = ev[g + '__control__pooled'].copy()
            b = fc[g + '_x']
            d = float(np.max(np.abs(a.astype(float) - b.astype(float))))
            maxdiff = max(maxdiff, d)
            assert np.allclose(a, b, atol=1e-4, rtol=1e-5), (g, d)
            pooled[g] = a
        checks['pooled_two_independent_caches_agree'] = maxdiff < 1e-4
        control_scores_cached = {g: ev[g + '__control__scores'].copy() for g in GROUPS}

    target_idx = [i for i, c in enumerate(classes) if c in TARGET_NAMES]
    checks['project18_indices_18'] = len(target_idx) == 18

    rows, scores = [], {}
    for g in GROUPS:
        for arm in heads:
            w, b = heads[arm]
            scores[g, arm] = pooled[g].astype(float) @ w.T + b
        # recomputed control must reproduce the frozen control evaluation exactly
        d = float(np.max(np.abs(scores[g, 'control'] - control_scores_cached[g])))
        assert d < 1e-9, (g, d)
        for i, spec in enumerate(meta[g]):
            label = spec.get('label', spec.get('true', spec.get('reference_label')))
            yi = classes.index(label)
            cond = spec.get('condition', 'intact')
            row = dict(group=g, id=spec['id'], condition=cond, true=label, models={})
            if 'signer_id' in spec:
                row['signer_id'] = spec['signer_id']
            oldrow = oldmap[g, spec['id'], cond]
            assert oldrow['true'] == label
            for arm in heads:
                s = scores[g, arm][i]
                row['models'][arm] = {}
                for scope, inds in [('full93', list(range(93))), ('project18', target_idx)]:
                    if yi not in inds:
                        continue
                    order = sorted(inds, key=lambda j: -float(s[j]))
                    rival = next(j for j in order if j != yi)
                    row['models'][arm][scope] = dict(pred=classes[order[0]], correct=bool(order[0] == yi),
                                                     rank=order.index(yi) + 1,
                                                     margin=float(s[yi] - s[rival]))
            # frozen control rows must match what was recorded before this unit existed
            for scope in row['models']['control']:
                assert row['models']['control'][scope]['pred'] == oldrow['models']['control'][scope]['pred']
                assert row['models']['control'][scope]['correct'] == oldrow['models']['control'][scope]['correct']
            rows.append(row)
    checks['control_rows_reproduce_frozen_evaluation'] = True

    summary = {}
    for g in GROUPS:
        rs = [r for r in rows if r['group'] == g]
        summary[g] = {arm: {cond: {scope: summarize([r for r in rs if r['condition'] == cond], arm, scope)
                                   for scope in ['full93', 'project18']}
                            for cond in sorted({r['condition'] for r in rs})}
                      for arm in ['control', 'candidate']}

    def s(g, arm='candidate', cond='intact', scope='full93'):
        return summary[g][arm][cond][scope]

    # ---- predeclared retention gates (COMPARISON_PLAN.evaluation.candidate_retention)
    gates = {
        'annotation_total_nondecreasing':
            s('annotation_dev')['correct'] >= s('annotation_dev', 'control')['correct'],
        'annotation_macro_recall_nondecreasing':
            s('annotation_dev')['macro_recall'] >= s('annotation_dev', 'control')['macro_recall'] - 1e-12,
        'annotation_each_signer_nondecreasing':
            all(s('annotation_dev')['per_signer'][str(k)]['correct']
                >= s('annotation_dev', 'control')['per_signer'][str(k)]['correct'] for k in (11, 12, 13)),
        'annotation_target_nondecreasing':
            s('annotation_dev', scope='project18')['correct']
            >= s('annotation_dev', 'control', scope='project18')['correct'],
        'whole_no_loss':
            all(s('whole_dev', scope=sc)['correct'] >= s('whole_dev', 'control', scope=sc)['correct']
                for sc in ['full93', 'project18']),
    }
    for cond in sorted({r['condition'] for r in rows if r['group'] == 'public'}):
        gates['public_no_loss_' + cond] = all(
            s('public', cond=cond, scope=sc)['correct'] >= s('public', 'control', cond=cond, scope=sc)['correct']
            for sc in ['full93', 'project18'])
    for cond in sorted({r['condition'] for r in rows if r['group'] == 'mp'}):
        gates['mp_no_loss_' + cond] = all(
            s('mp', cond=cond, scope=sc)['correct'] >= s('mp', 'control', cond=cond, scope=sc)['correct']
            for sc in ['full93', 'project18'])

    # ---- per-row changes vs the reused control
    changes = {}
    for g in GROUPS:
        rs = [r for r in rows if r['group'] == g]
        changes[g] = {
            'gains': [{'id': r['id'], 'condition': r['condition'], 'true': r['true'],
                       'control_pred': r['models']['control']['full93']['pred']}
                      for r in rs if r['models']['candidate']['full93']['correct']
                      and not r['models']['control']['full93']['correct']],
            'regressions': [{'id': r['id'], 'condition': r['condition'], 'true': r['true'],
                             'candidate_pred': r['models']['candidate']['full93']['pred']}
                            for r in rs if not r['models']['candidate']['full93']['correct']
                            and r['models']['control']['full93']['correct']],
        }

    # ---- per-target-word table on annotation_dev (the only group with signer split)
    adev = [r for r in rows if r['group'] == 'annotation_dev']
    per_target = {}
    for name in sorted(TARGET_NAMES):
        rs = [r for r in adev if r['true'] == name]
        if not rs:
            per_target[name] = {'n': 0, 'control': None, 'candidate': None,
                                'note': 'no public development case for this target'}
            continue
        per_target[name] = {
            'n': len(rs),
            'control': sum(r['models']['control']['full93']['correct'] for r in rs),
            'candidate': sum(r['models']['candidate']['full93']['correct'] for r in rs),
            'control_project18': sum(r['models']['control']['project18']['correct'] for r in rs),
            'candidate_project18': sum(r['models']['candidate']['project18']['correct'] for r in rs),
            'mean_margin_control': float(np.mean([r['models']['control']['full93']['margin'] for r in rs])),
            'mean_margin_candidate': float(np.mean([r['models']['candidate']['full93']['margin'] for r in rs])),
        }

    # ---- training sanity on the 26 new windows (explicitly not generalization)
    ready = read(H / 'TREATMENT_PREFIX_READY.json')
    from generated_v1 import predict_prefix
    from local_io import unpack
    with np.load(H / 'treatment_prefixes.npz', allow_pickle=False) as z:
        ctx = unpack(z['context_prefix'], z['context_offsets'])
        cy = z['context_y'].copy()
    new_prefix, new_y = ctx[224:], cy[224:]
    w, b = heads['candidate']
    new_scores = predict_prefix(parent, new_prefix)['pooled'].astype(float) @ w.T + b
    new_correct = int((new_scores.argmax(1) == new_y).sum())
    wc, bc = heads['control']
    new_scores_c = predict_prefix(parent, new_prefix)['pooled'].astype(float) @ wc.T + bc
    new_correct_control = int((new_scores_c.argmax(1) == new_y).sum())

    failed = [k for k, v in checks.items() if not v]
    assert not failed, failed
    gates_pass = all(gates.values())
    gain = s('annotation_dev')['correct'] - s('annotation_dev', 'control')['correct']
    if not gates_pass:
        decision = 'REJECTED_BY_PREDECLARED_RETENTION_GATES'
    elif gain > 0:
        decision = 'RETENTION_PASS_WITH_DEVELOPMENT_GAIN_PENDING_AUDIT'
    else:
        decision = 'RETENTION_PASS_NONREGRESSION_NO_DEMONSTRATED_IMPROVEMENT'

    arrays = {'classes': np.array(classes)}
    for g in GROUPS:
        for arm in heads:
            arrays[g + '__' + arm + '__scores'] = scores[g, arm]
        arrays[g + '__pooled'] = pooled[g]
    with (R / 'evaluation_arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)

    result = {
        'status': 'PUBLIC_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT',
        'decision': decision,
        'annotation_dev_gain_vs_control': gain,
        'checks': checks, 'gates': gates,
        'summary': summary, 'changes': changes, 'per_target_annotation_dev': per_target,
        'new_window_training_sanity': {
            'candidate_correct': new_correct, 'control_correct': new_correct_control, 'of': 26,
            'note': 'these 26 windows are training inputs for the candidate; not generalization'},
        'rows': rows,
        'candidate': train['candidate'], 'control': p['control'], 'parent': p['parent'],
        'policy_sha256': sha(H / 'local_policy.json'),
        'training_record_sha256': sha(R / 'training.json'),
        'arrays_sha256': sha(R / 'evaluation_arrays.npz'),
        'prefix_cache_sha256': ready['cache_sha256'],
        'encoder_runs_this_evaluation': 0,
        'pooled_features_reused_from_frozen_caches': True,
        'control_retrained': False,
        'user_arrays_or_scores_read': 0, 'reserved_arrays_read': 0, 'ksl_arrays_read': 0,
        'default_promoted': False,
        'limits': [
            'head.* only was trained; the encoder, last block and all normalization buffers are the parent\'s',
            'the 13 added recordings are signer 02, already a training signer; no new person was added',
            'no public development case exists for 너, 싫다, 자다, so their added data cannot be scored here',
            'annotation699 and whole74 share source recordings; not independent cases',
            'public MP7 uses training signers 01/02 and two windows per clip',
            'source-label supervision only; no execution-correctness or performance-quality claim',
            'no new-person or web-domain generalization claim',
        ],
    }
    write(R / 'evaluation.json', result)
    print('decision:', decision)
    print('gates:', {k: bool(v) for k, v in gates.items()})
    print('annotation_dev full93  control', s('annotation_dev', 'control')['correct'],
          '-> candidate', s('annotation_dev')['correct'], 'of', s('annotation_dev')['n'])
    print('annotation_dev macro_recall control %.6f -> candidate %.6f'
          % (s('annotation_dev', 'control')['macro_recall'], s('annotation_dev')['macro_recall']))
    print('annotation_dev project18 control', s('annotation_dev', 'control', scope='project18')['correct'],
          '-> candidate', s('annotation_dev', scope='project18')['correct'],
          'of', s('annotation_dev', scope='project18')['n'])
    print('whole_dev control', s('whole_dev', 'control')['correct'], '-> candidate', s('whole_dev')['correct'])
    print('per signer:', {k: (s('annotation_dev', 'control')['per_signer'][k]['correct'],
                              s('annotation_dev')['per_signer'][k]['correct']) for k in ('11', '12', '13')})
    print('gains', [(c['true'], c['id']) for c in changes['annotation_dev']['gains']])
    print('regressions', [(c['true'], c['id'], c['candidate_pred']) for c in changes['annotation_dev']['regressions']])
    print('new-window sanity candidate %d / 26 (control %d / 26)' % (new_correct, new_correct_control))


if __name__ == '__main__':
    main()
