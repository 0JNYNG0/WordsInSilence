"""Public evaluation of the treatment. Control and baseline results are reused, not recomputed."""
import hashlib, importlib.util, json, pathlib, sys, time
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
gen = module('prior_gen', PRIOR / 'generated_v1.py')
old = module('prior_io', PRIOR / 'local_io.py')
import tools_prep_union93_candidate as u              # noqa: E402
from wis_lab.service.neural import PROJECT_18         # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
write = old.write
GROUPS = ['annotation_dev', 'whole_dev', 'public', 'mp']
TARGETS = set(PROJECT_18.values())


def summarize(rows, arm, scope):
    rs = [r for r in rows if scope in r['models'][arm]]
    labels = sorted({r['true'] for r in rs})
    per, recalls = {}, []
    for lb in labels:
        n = sum(r['true'] == lb for r in rs)
        tp = sum(r['true'] == lb and r['models'][arm][scope]['correct'] for r in rs)
        per[lb] = dict(n=n, correct=tp, recall=tp / n)
        recalls.append(tp / n)
    people = {str(s): dict(n=sum(r.get('signer_id') == s for r in rs),
                           correct=sum(r.get('signer_id') == s and r['models'][arm][scope]['correct'] for r in rs))
              for s in sorted({r['signer_id'] for r in rs if 'signer_id' in r})}
    return dict(n=len(rs), correct=sum(r['models'][arm][scope]['correct'] for r in rs),
                ground_truth_classes=len(labels),
                macro_recall=float(np.mean(recalls)) if recalls else None,
                per_class=per, per_signer=people)


def main():
    started = time.time()
    p = read(H / 'comparison_policy.json')
    tr = read(R / 'training.json')
    assert tr['status'].startswith('TRAINING_COMPLETE')
    assert sha(tr['candidate']['path']) == tr['candidate']['sha256']

    oldp = old.policy()
    ck, parent, classes = old.parent_model(oldp)
    torch.set_num_threads(2)
    pay = torch.load(tr['candidate']['path'], map_location='cpu', weights_only=True)
    cand = impl.AuxWordModel(parent, len(pay['shape_codes']), True)
    cand.load_state_dict(pay['state_dict'])
    cand.eval()

    checks = {}
    ctrl_sd = torch.load(L / p['control']['path'], map_location='cpu', weights_only=True)['state_dict']
    checks['candidate_differs_from_control_in_shared_block'] = not torch.equal(
        pay['state_dict']['base.blocks.3.conv1.weight'], ctrl_sd['blocks.3.conv1.weight'])
    checks['earlier_blocks_identical_to_parent'] = all(
        torch.equal(pay['state_dict']['base.' + k], ck['state_dict'][k])
        for k in ck['state_dict'] if k.startswith('blocks.') and not k.startswith('blocks.3.'))
    checks['aux_head_trained'] = int(torch.count_nonzero(pay['state_dict']['shape_head.weight'])) > 0

    groups, meta = old.evaluation_inputs(oldp)
    checks['group_sizes'] = [len(groups[g]) for g in GROUPS] == [699, 74, 32, 14]
    target_idx = [i for i, c in enumerate(classes) if c in TARGETS]

    prior = read(L / 'runs/partial_encoder_pair_v1/evaluation.json')
    priormap = {(r['group'], r['id'], r['condition']): r for r in prior['rows']}

    rows, arrays = [], {'classes': np.array(classes)}
    for g in GROUPS:
        prefixes = gen.extract_prefix(parent, groups[g])
        outs = []
        with torch.inference_mode():
            for i in range(0, len(prefixes), 32):
                h, m = impl._pad_batch(prefixes[i:i + 32])
                wl, _, _ = cand(h, m)
                outs.append(wl.numpy())
        score = np.concatenate(outs, axis=0).astype(np.float64)
        arrays[g + '__candidate__scores'] = score
        for i, spec in enumerate(meta[g]):
            lb = spec.get('label', spec.get('true', spec.get('reference_label')))
            yi = classes.index(lb)
            cond = spec.get('condition', 'intact')
            row = dict(group=g, id=spec['id'], condition=cond, true=lb, models={})
            if 'signer_id' in spec:
                row['signer_id'] = spec['signer_id']
            pr = priormap[g, spec['id'], cond]
            assert pr['true'] == lb
            row['models']['baseline_83212868'] = pr['models']['control']      # head-only project baseline
            row['models']['control_af6c0119'] = pr['models']['treatment']     # blocks.3+head, this pair's control
            s = score[i]
            row['models']['candidate'] = {}
            for scope, inds in [('full93', list(range(93))), ('project18', target_idx)]:
                if yi not in inds:
                    continue
                order = sorted(inds, key=lambda j: -float(s[j]))
                rival = next(j for j in order if j != yi)
                row['models']['candidate'][scope] = dict(pred=classes[order[0]], correct=bool(order[0] == yi),
                                                         rank=order.index(yi) + 1,
                                                         margin=float(s[yi] - s[rival]))
            rows.append(row)
        print(json.dumps({'evaluated_group': g, 'clips': len(groups[g])}), flush=True)

    ARMS = ['baseline_83212868', 'control_af6c0119', 'candidate']
    summary = {}
    for g in GROUPS:
        rs = [r for r in rows if r['group'] == g]
        summary[g] = {a: {c: {sc: summarize([r for r in rs if r['condition'] == c], a, sc)
                              for sc in ['full93', 'project18']}
                          for c in sorted({r['condition'] for r in rs})} for a in ARMS}

    def S(g, a='candidate', c='intact', sc='full93'):
        return summary[g][a][c][sc]

    gates = {
        'annotation_dev_increase_vs_control':
            S('annotation_dev')['correct'] > S('annotation_dev', 'control_af6c0119')['correct'],
        'annotation_per_signer_nondecreasing_vs_control':
            all(S('annotation_dev')['per_signer'][k]['correct']
                >= S('annotation_dev', 'control_af6c0119')['per_signer'][k]['correct'] for k in ('11', '12', '13')),
        'whole_dev_no_loss':
            S('whole_dev')['correct'] >= S('whole_dev', 'control_af6c0119')['correct'],
    }
    for c in sorted({r['condition'] for r in rows if r['group'] == 'public'}):
        gates['public_no_loss_' + c] = all(
            S('public', c=c, sc=sc)['correct'] >= S('public', 'control_af6c0119', c=c, sc=sc)['correct']
            for sc in ['full93', 'project18'])
    for c in sorted({r['condition'] for r in rows if r['group'] == 'mp'}):
        gates['mp_no_loss_' + c] = all(
            S('mp', c=c, sc=sc)['correct'] >= S('mp', 'control_af6c0119', c=c, sc=sc)['correct']
            for sc in ['full93', 'project18'])

    changes = {}
    for against in ['control_af6c0119', 'baseline_83212868']:
        changes[against] = {}
        for g in GROUPS:
            rs = [r for r in rows if r['group'] == g]
            changes[against][g] = {
                'gains': [{'id': r['id'], 'true': r['true'], 'condition': r['condition'],
                           'other_pred': r['models'][against]['full93']['pred']}
                          for r in rs if r['models']['candidate']['full93']['correct']
                          and not r['models'][against]['full93']['correct']],
                'regressions': [{'id': r['id'], 'true': r['true'], 'condition': r['condition'],
                                 'candidate_pred': r['models']['candidate']['full93']['pred']}
                                for r in rs if not r['models']['candidate']['full93']['correct']
                                and r['models'][against]['full93']['correct']]}

    adev = [r for r in rows if r['group'] == 'annotation_dev']
    per_word = {}
    for lb in sorted({r['true'] for r in adev}):
        sub = [r for r in adev if r['true'] == lb]
        v = {a: sum(r['models'][a]['full93']['correct'] for r in sub) for a in ARMS}
        if len(set(v.values())) > 1:
            per_word[lb] = dict(n=len(sub), **v)

    # auxiliary head accuracy on TRAINING rows only - sanity read, never generalisation
    with np.load(PRIOR / 'train_prefixes.npz', allow_pickle=False) as z:
        bpre = [z['base_prefix'][a:b].copy() for a, b in zip(z['base_offsets'][:-1], z['base_offsets'][1:])]
    with np.load(H / 'shape_labels.npz', allow_pickle=False) as z:
        bs = z['base_shape_y'].copy()
    keep = np.flatnonzero(bs >= 0)
    hit = 0
    with torch.inference_mode():
        for i in range(0, len(keep), 64):
            idx = keep[i:i + 64]
            h, m = impl._pad_batch([bpre[int(j)] for j in idx])
            _, sl, _ = cand(h, m)
            hit += int((sl.numpy().argmax(1) == bs[idx]).sum())
    aux_train_acc = hit / len(keep)

    failed = [k for k, v in checks.items() if not v]
    assert not failed, failed
    gates_pass = all(gates.values())
    delta_ctrl = S('annotation_dev')['correct'] - S('annotation_dev', 'control_af6c0119')['correct']
    delta_base = S('annotation_dev')['correct'] - S('annotation_dev', 'baseline_83212868')['correct']
    decision = ('METHOD_EFFECT_PASS_PENDING_AUDIT' if gates_pass else 'REJECTED_BY_PREDECLARED_GATES')
    with (R / 'evaluation_arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)

    result = {
        'status': 'PUBLIC_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT',
        'decision': decision,
        'method_effect_vs_control': delta_ctrl,
        'project_improvement_vs_baseline': delta_base,
        'project_improvement_claimed': bool(delta_base > 0 and gates_pass),
        'checks': checks, 'gates': gates, 'summary': summary, 'changes': changes,
        'per_word_differences_annotation_dev': per_word,
        'auxiliary_head_training_accuracy': {'value': aux_train_acc, 'rows': int(len(keep)),
                                             'meaning': 'training rows only; a sanity read, never generalisation'},
        'rows': rows, 'arms': ARMS,
        'candidate': tr['candidate'], 'control': p['control'], 'baseline': p['reference_baseline'],
        'policy_sha256': sha(H / 'comparison_policy.json'),
        'training_record_sha256': sha(R / 'training.json'),
        'arrays_sha256': sha(R / 'evaluation_arrays.npz'),
        'control_retrained': False, 'baseline_retrained': False,
        'user_arrays_or_scores_read': 0, 'reserved_arrays_read': 0, 'default_promoted': False,
        'limits': p['explicit_limits'],
        'seconds': round(time.time() - started, 2)}
    write(R / 'evaluation.json', result)

    print('\ndecision:', decision)
    print('gates:', {k: bool(v) for k, v in gates.items()})
    for a in ARMS:
        print(f"  {a:20s} annotation_dev {S('annotation_dev', a)['correct']}/699  "
              f"macro {S('annotation_dev', a)['macro_recall']:.6f}  "
              f"p18 {S('annotation_dev', a, sc='project18')['correct']}/33  "
              f"whole {S('whole_dev', a)['correct']}/74  "
              f"signer {[S('annotation_dev', a)['per_signer'][k]['correct'] for k in ('11','12','13')]}")
    for against in ['control_af6c0119', 'baseline_83212868']:
        g = sum(len(v['gains']) for v in changes[against].values())
        r = sum(len(v['regressions']) for v in changes[against].values())
        print(f'  vs {against}: gains {g}, regressions {r}')
    print('aux head training accuracy %.4f on %d rows' % (aux_train_acc, len(keep)))


if __name__ == '__main__':
    main()
