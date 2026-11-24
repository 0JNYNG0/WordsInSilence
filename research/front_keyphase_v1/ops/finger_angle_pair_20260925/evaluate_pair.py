"""Public evaluation after both checkpoints are frozen. No result-driven change is made."""
import hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
R = L / 'runs/finger_angle_pair_v1'
OLD = L / 'ops/partial_encoder_pair_20260922'
sys.path[:0] = [str(H), str(OLD), str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# local_io puts its OWN directory at the front of sys.path, which would shadow this unit's
# generated_v1 with the partial-encoder one. Load this unit's modules by explicit path.
AB = module('angle_pair_features', H / 'angle_block.py')
_impl = module('angle_pair_impl', H / 'generated_v1.py')
ExtraInputTCN = _impl.ExtraInputTCN
old = module('frozen_partial_io', OLD / 'local_io.py')
from wis_lab.service.neural import PROJECT_18             # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
write = old.write
GROUPS = ['annotation_dev', 'whole_dev', 'public', 'mp']
TARGETS = set(PROJECT_18.values())


def summarize(rows, arm, scope):
    rs = [r for r in rows if scope in r['models'][arm]]
    labels = sorted({r['true'] for r in rs})
    detail, recalls = {}, []
    for label in labels:
        n = sum(r['true'] == label for r in rs)
        tp = sum(r['true'] == label and r['models'][arm][scope]['pred'] == label for r in rs)
        detail[label] = dict(n=n, correct=tp, recall=tp / n)
        recalls.append(tp / n)
    people = {str(s): dict(n=sum(r.get('signer_id') == s for r in rs),
                           correct=sum(r.get('signer_id') == s and r['models'][arm][scope]['correct']
                                       for r in rs))
              for s in sorted({r['signer_id'] for r in rs if 'signer_id' in r})}
    return dict(n=len(rs), correct=sum(r['models'][arm][scope]['correct'] for r in rs),
                ground_truth_classes=len(labels),
                macro_recall=float(np.mean(recalls)) if recalls else None,
                per_class=detail, per_signer=people)


def main():
    started = time.time()
    pol = read(H / 'comparison_policy.json')
    train = read(R / 'training.json')
    assert train['status'] == 'TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION'
    assert sha(H / 'comparison_policy.json') == train['policy_sha256']

    oldp = old.policy()                       # reuse the already-pinned evaluation input routing
    ck, parent, classes = old.parent_model(oldp)
    torch.set_num_threads(2)

    models, payloads = {}, {}
    for arm, spec in train['models'].items():
        assert sha(spec['path']) == spec['sha256']
        pay = torch.load(spec['path'], map_location='cpu', weights_only=True)
        assert pay['classes'] == classes
        m = ExtraInputTCN(parent, pay['state_dict']['extra_scale'],
                          pay['extra_value_channels'], pay['extra_width'])
        m.load_state_dict(pay['state_dict'])
        m.eval()
        models[arm], payloads[arm] = m, pay

    checks = {}
    checks['two_arms_same_trainable_count'] = (
        train['models']['control']['trainable_count'] == train['models']['treatment']['trainable_count'])
    checks['control_is_chord'] = payloads['control']['feature'] == 'chord'
    checks['treatment_is_angle'] = payloads['treatment']['feature'] == 'angle'
    checks['encoder_identical_between_arms'] = all(
        torch.equal(payloads['control']['state_dict'][k], payloads['treatment']['state_dict'][k])
        for k in payloads['control']['state_dict']
        if not k.startswith('base.head.') and k != 'extra_weight' and k != 'extra_scale')
    checks['heads_differ'] = not torch.equal(payloads['control']['state_dict']['base.head.weight'],
                                             payloads['treatment']['state_dict']['base.head.weight'])

    groups, meta = old.evaluation_inputs(oldp)
    checks['group_sizes'] = [len(groups[g]) for g in GROUPS] == [699, 74, 32, 14]
    target_idx = [i for i, c in enumerate(classes) if c in TARGETS]
    checks['project18_indices'] = len(target_idx) == 18

    rows, scores, arrays = [], {}, {'classes': np.array(classes)}
    for g in GROUPS:
        xs = groups[g]
        extras = [AB.blocks(a) for a in xs]
        for arm, m in models.items():
            feat = 1 if payloads[arm]['feature'] == 'angle' else 0
            out = []
            with torch.inference_mode():
                for i in range(0, len(xs), 32):
                    chunk, ex = xs[i:i + 32], [extras[j][feat] for j in range(i, min(i + 32, len(xs)))]
                    t = max(a.shape[0] for a in chunk)
                    X = np.zeros((len(chunk), t, 280), dtype=np.float32)
                    E = np.zeros((len(chunk), t, AB.EXTRA_WIDTH), dtype=np.float32)
                    M = np.zeros((len(chunk), t), dtype=np.float32)
                    for k, (a, e) in enumerate(zip(chunk, ex)):
                        X[k, :len(a)] = a
                        E[k, :len(e)] = e
                        M[k, :len(a)] = 1.0
                    out.append(m(torch.from_numpy(X), torch.from_numpy(E),
                                 torch.from_numpy(M)).numpy())
            scores[g, arm] = np.concatenate(out, axis=0).astype(np.float64)
            arrays[f'{g}__{arm}__scores'] = scores[g, arm]
        for i, spec in enumerate(meta[g]):
            label = spec.get('label', spec.get('true', spec.get('reference_label')))
            yi = classes.index(label)
            row = dict(group=g, id=spec['id'], condition=spec.get('condition', 'intact'),
                       true=label, models={})
            if 'signer_id' in spec:
                row['signer_id'] = spec['signer_id']
            for arm in models:
                s = scores[g, arm][i]
                row['models'][arm] = {}
                for scope, inds in [('full93', list(range(93))), ('project18', target_idx)]:
                    if yi not in inds:
                        continue
                    order = sorted(inds, key=lambda j: -float(s[j]))
                    rival = next(j for j in order if j != yi)
                    row['models'][arm][scope] = dict(pred=classes[order[0]],
                                                     correct=bool(order[0] == yi),
                                                     rank=order.index(yi) + 1,
                                                     margin=float(s[yi] - s[rival]))
            rows.append(row)
        print(json.dumps({'evaluated_group': g, 'clips': len(xs)}), flush=True)

    summary = {}
    for g in GROUPS:
        rs = [r for r in rows if r['group'] == g]
        summary[g] = {arm: {cond: {sc: summarize([r for r in rs if r['condition'] == cond], arm, sc)
                                   for sc in ['full93', 'project18']}
                            for cond in sorted({r['condition'] for r in rs})}
                      for arm in ['control', 'treatment']}

    def s(g, arm='treatment', cond='intact', scope='full93'):
        return summary[g][arm][cond][scope]

    ref = pol['reference_baselines_reported_alongside']['frozen_control_83212868']
    gates = {
        'annotation_dev_increase':
            s('annotation_dev')['correct'] > s('annotation_dev', 'control')['correct'],
        'whole_dev_maintained_74':
            s('whole_dev')['correct'] == 74 and s('whole_dev', 'control')['correct'] == 74,
        'annotation_each_signer_nondecreasing':
            all(s('annotation_dev')['per_signer'][str(k)]['correct']
                >= s('annotation_dev', 'control')['per_signer'][str(k)]['correct'] for k in (11, 12, 13)),
    }
    for cond in sorted({r['condition'] for r in rows if r['group'] == 'public'}):
        gates['public_no_loss_' + cond] = all(
            s('public', cond=cond, scope=sc)['correct'] >= s('public', 'control', cond=cond, scope=sc)['correct']
            for sc in ['full93', 'project18'])
    for cond in sorted({r['condition'] for r in rows if r['group'] == 'mp'}):
        gates['mp_no_loss_' + cond] = all(
            s('mp', cond=cond, scope=sc)['correct'] >= s('mp', 'control', cond=cond, scope=sc)['correct']
            for sc in ['full93', 'project18'])

    changes = {}
    for g in GROUPS:
        rs = [r for r in rows if r['group'] == g]
        changes[g] = {
            'gains': [{'id': r['id'], 'condition': r['condition'], 'true': r['true'],
                       'control_pred': r['models']['control']['full93']['pred']}
                      for r in rs if r['models']['treatment']['full93']['correct']
                      and not r['models']['control']['full93']['correct']],
            'regressions': [{'id': r['id'], 'condition': r['condition'], 'true': r['true'],
                             'treatment_pred': r['models']['treatment']['full93']['pred']}
                            for r in rs if not r['models']['treatment']['full93']['correct']
                            and r['models']['control']['full93']['correct']]}

    adev = [r for r in rows if r['group'] == 'annotation_dev']
    per_target = {}
    for name in sorted(TARGETS):
        rs = [r for r in adev if r['true'] == name]
        per_target[name] = ({'n': 0} if not rs else {
            'n': len(rs),
            'control': sum(r['models']['control']['full93']['correct'] for r in rs),
            'treatment': sum(r['models']['treatment']['full93']['correct'] for r in rs),
            'mean_margin_control': float(np.mean([r['models']['control']['full93']['margin'] for r in rs])),
            'mean_margin_treatment': float(np.mean([r['models']['treatment']['full93']['margin'] for r in rs]))})

    per_word_delta = {}
    for g in GROUPS:
        rs = [r for r in rows if r['group'] == g]
        for lab in sorted({r['true'] for r in rs}):
            sub = [r for r in rs if r['true'] == lab]
            c0 = sum(r['models']['control']['full93']['correct'] for r in sub)
            c1 = sum(r['models']['treatment']['full93']['correct'] for r in sub)
            if c0 != c1:
                per_word_delta[f'{g}|{lab}'] = {'n': len(sub), 'control': c0, 'treatment': c1,
                                                'delta': c1 - c0}

    failed = [k for k, v in checks.items() if not v]
    assert not failed, failed
    gates_pass = all(gates.values())
    delta = s('annotation_dev')['correct'] - s('annotation_dev', 'control')['correct']
    decision = ('ANGLE_ADOPTION_ELIGIBLE_PENDING_AUDIT' if gates_pass else
                'REJECTED_BY_PREDECLARED_GATES')
    with (R / 'evaluation_arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)

    result = {
        'status': 'PUBLIC_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT',
        'decision': decision,
        'annotation_dev_delta_treatment_minus_control': delta,
        'checks': checks, 'gates': gates, 'summary': summary, 'changes': changes,
        'per_target_annotation_dev': per_target, 'per_word_changes': per_word_delta,
        'reference_baseline_frozen_control_83212868': ref,
        'rows': rows, 'models': train['models'], 'parent': pol['matched_pair']['parent'],
        'policy_sha256': sha(H / 'comparison_policy.json'),
        'training_record_sha256': sha(R / 'training.json'),
        'arrays_sha256': sha(R / 'evaluation_arrays.npz'),
        'seconds': round(time.time() - started, 2),
        'user_arrays_or_scores_read': 0, 'reserved_arrays_read': 0, 'ksl_arrays_read': 0,
        'default_promoted': False, 'training_runs_this_step': 0,
        'limits': pol['explicit_limits'] + [
            'whole_dev 74 is a ceiling and was a retention condition only, never an improvement target',
            'the frozen control 83212868 is a different training and is reported as context, never as this pair control',
            'annotation699 and whole74 share source recordings; not independent cases',
            'public MP7 uses training signers 01/02']}
    write(R / 'evaluation.json', result)

    print('\ndecision:', decision)
    print('gates:', {k: bool(v) for k, v in gates.items()})
    print('annotation_dev full93   chord(control)', s('annotation_dev', 'control')['correct'],
          '-> angle(treatment)', s('annotation_dev')['correct'], 'of 699')
    print('annotation_dev macro    %.6f -> %.6f' % (s('annotation_dev', 'control')['macro_recall'],
                                                    s('annotation_dev')['macro_recall']))
    print('annotation_dev p18      ', s('annotation_dev', 'control', scope='project18')['correct'],
          '->', s('annotation_dev', scope='project18')['correct'], 'of 33')
    print('whole_dev               ', s('whole_dev', 'control')['correct'], '->', s('whole_dev')['correct'], 'of 74')
    print('per signer              ', {k: (s('annotation_dev', 'control')['per_signer'][k]['correct'],
                                           s('annotation_dev')['per_signer'][k]['correct'])
                                       for k in ('11', '12', '13')})
    print('reference frozen 83212868:', ref)
    for g in GROUPS:
        print(f'  {g:15s} gains {len(changes[g]["gains"])} regressions {len(changes[g]["regressions"])}')


if __name__ == '__main__':
    main()
