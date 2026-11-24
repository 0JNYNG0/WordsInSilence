"""Public development evaluation: control reproduction and candidate are evaluated; the baseline
83212868 and the prior af6c0119 come from the stored evaluation (not recomputed).
Method control = the lambda-0 reproduction (fixed by the policy rule; see training.json)."""
import hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
R = L / 'runs/supcon_pair_v1'
PRIOR = L / 'ops/partial_encoder_pair_20260922'
sys.path[:0] = [str(H), str(PRIOR), str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


import supcon_impl as impl                              # noqa: E402
gen = module('prior_gen', PRIOR / 'generated_v1.py')
old = module('prior_io', PRIOR / 'local_io.py')
from wis_lab.service.neural import PROJECT_18           # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
GROUPS = ['annotation_dev', 'whole_dev', 'public', 'mp']
TARGETS = set(PROJECT_18.values())
ARMS = ['baseline_83212868', 'prior_af6c0119_reference_only', 'control_reproduction', 'candidate']


def summarize(rows, arm, scope):
    rs = [r for r in rows if scope in r['models'][arm]]
    labels = sorted({r['true'] for r in rs})
    per, recalls = {}, []
    for lb in labels:
        n = sum(r['true'] == lb for r in rs)
        tp = sum(r['true'] == lb and r['models'][arm][scope]['correct'] for r in rs)
        per[lb] = dict(n=n, correct=tp)
        recalls.append(tp / n)
    people = {str(s): dict(n=sum(r.get('signer_id') == s for r in rs),
                           correct=sum(r.get('signer_id') == s and r['models'][arm][scope]['correct'] for r in rs))
              for s in sorted({r['signer_id'] for r in rs if 'signer_id' in r})}
    return dict(n=len(rs), correct=sum(r['models'][arm][scope]['correct'] for r in rs),
                macro_recall=float(np.mean(recalls)) if recalls else None, per_class=per, per_signer=people)


def main():
    started = time.time()
    p = read(H / 'comparison_policy.json')
    tr = read(R / 'training.json')
    assert tr['status'].startswith('TRAINING_COMPLETE')
    assert not tr['reproduction_vs_af6c0119']['all_trainable_bitwise_equal'], 'rule: equal -> reuse af6c0119 instead'
    for a in ('control_reproduction', 'candidate'):
        assert sha(tr['runs'][a]['path']) == tr['runs'][a]['sha256']
    assert not (R / 'evaluation.json').exists()

    oldp = old.policy()
    ck, parent, classes = old.parent_model(oldp)
    assert sha(L / p['parent']['path']) == p['parent']['sha256']
    torch.set_num_threads(4)
    models = {}
    for a in ('control_reproduction', 'candidate'):
        pay = torch.load(tr['runs'][a]['path'], map_location='cpu', weights_only=True)
        wm = impl.WordModel(parent)
        wm.load_state_dict({'base.' + k: v for k, v in pay['state_dict'].items()})
        wm.eval()
        models[a] = wm
    checks = {'earlier_blocks_identical_to_parent': all(
        torch.equal(models[a].base.state_dict()[k], ck['state_dict'][k])
        for a in models for k in ck['state_dict'] if not (k.startswith('blocks.3.') or k.startswith('head.')))}

    groups, meta = old.evaluation_inputs(oldp)
    checks['group_sizes'] = [len(groups[g]) for g in GROUPS] == [699, 74, 32, 14]
    target_idx = [i for i, c in enumerate(classes) if c in TARGETS]
    prior = read(L / 'runs/partial_encoder_pair_v1/evaluation.json')
    priormap = {(r['group'], r['id'], r['condition']): r for r in prior['rows']}

    rows, pooled_dev, arrays = [], {}, {'classes': np.array(classes)}
    for g in GROUPS:
        prefixes = gen.extract_prefix(parent, groups[g])
        scores, pools = {}, {}
        with torch.inference_mode():
            for a, wm in models.items():
                outs, pl = [], []
                for i in range(0, len(prefixes), 32):
                    h, m = impl._pad_batch(prefixes[i:i + 32])
                    wl, po = wm(h, m)
                    outs.append(wl.numpy()); pl.append(po.numpy())
                scores[a] = np.concatenate(outs).astype(np.float64)
                pools[a] = np.concatenate(pl)
                arrays[f'{g}__{a}__scores'] = scores[a]
        if g == 'annotation_dev':
            pooled_dev = pools
        for i, spec in enumerate(meta[g]):
            lb = spec.get('label', spec.get('true', spec.get('reference_label')))
            yi = classes.index(lb)
            cond = spec.get('condition', 'intact')
            row = dict(group=g, id=spec['id'], condition=cond, true=lb, models={})
            if 'signer_id' in spec:
                row['signer_id'] = spec['signer_id']
            pr = priormap[g, spec['id'], cond]
            assert pr['true'] == lb
            row['models']['baseline_83212868'] = pr['models']['control']
            row['models']['prior_af6c0119_reference_only'] = pr['models']['treatment']
            for a in models:
                s = scores[a][i]
                row['models'][a] = {}
                for scope, inds in [('full93', list(range(93))), ('project18', target_idx)]:
                    if yi not in inds:
                        continue
                    order = sorted(inds, key=lambda j: -float(s[j]))
                    rival = next(j for j in order if j != yi)
                    row['models'][a][scope] = dict(pred=classes[order[0]], correct=bool(order[0] == yi),
                                                   rank=order.index(yi) + 1, margin=float(s[yi] - s[rival]))
            rows.append(row)
        print(json.dumps({'evaluated_group': g, 'clips': len(groups[g])}), flush=True)

    summary = {g: {a: {c: {sc: summarize([r for r in rows if r['group'] == g and r['condition'] == c], a, sc)
                           for sc in ['full93', 'project18']}
                       for c in sorted({r['condition'] for r in rows if r['group'] == g})} for a in ARMS}
               for g in GROUPS}

    def S(g, a='candidate', c='intact', sc='full93'):
        return summary[g][a][c][sc]

    def gates_vs(ref):
        gt = {'annotation_dev_increase': S('annotation_dev')['correct'] > S('annotation_dev', ref)['correct'],
              'annotation_macro_recall_not_lower': S('annotation_dev')['macro_recall'] >= S('annotation_dev', ref)['macro_recall'] - 1e-12,
              'each_dev_signer_not_lower': all(S('annotation_dev')['per_signer'][k]['correct'] >= S('annotation_dev', ref)['per_signer'][k]['correct']
                                               for k in ('11', '12', '13')),
              'project18_annotation_not_lower': S('annotation_dev', sc='project18')['correct'] >= S('annotation_dev', ref, sc='project18')['correct'],
              'whole_dev_no_loss': S('whole_dev')['correct'] >= S('whole_dev', ref)['correct']}
        for g in ('public', 'mp'):
            for c in sorted({r['condition'] for r in rows if r['group'] == g}):
                gt[f'{g}_no_loss_{c}'] = all(S(g, c=c, sc=sc)['correct'] >= S(g, ref, c=c, sc=sc)['correct'] for sc in ['full93', 'project18'])
        return gt

    gates = {'method_effect_vs_control_reproduction': gates_vs('control_reproduction'),
             'project_improvement_vs_baseline_83212868': gates_vs('baseline_83212868')}

    changes = {}
    for against in ['control_reproduction', 'baseline_83212868', 'prior_af6c0119_reference_only']:
        changes[against] = {}
        for g in GROUPS:
            rs = [r for r in rows if r['group'] == g]
            changes[against][g] = {
                'gains': [{'id': r['id'], 'true': r['true'], 'condition': r['condition'], 'signer': r.get('signer_id'),
                           'other_pred': r['models'][against]['full93']['pred']}
                          for r in rs if r['models']['candidate']['full93']['correct'] and not r['models'][against]['full93']['correct']],
                'regressions': [{'id': r['id'], 'true': r['true'], 'condition': r['condition'], 'signer': r.get('signer_id'),
                                 'candidate_pred': r['models']['candidate']['full93']['pred']}
                                for r in rs if not r['models']['candidate']['full93']['correct'] and r['models'][against]['full93']['correct']]}
    adev = [r for r in rows if r['group'] == 'annotation_dev']
    per_word = {}
    for lb in sorted({r['true'] for r in adev}):
        sub = [r for r in adev if r['true'] == lb]
        v = {a: sum(r['models'][a]['full93']['correct'] for r in sub) for a in ARMS}
        if len(set(v.values())) > 1:
            per_word[lb] = dict(n=len(sub), **v)
    repro_vs_prior = sum(r['models']['control_reproduction']['full93']['pred'] != r['models']['prior_af6c0119_reference_only']['full93']['pred']
                         for r in rows)

    # diagnostic, not a gate: cross-signer same-word vs different-word cosine on annotation_dev
    lab = np.array([r['true'] for r in adev])
    sg = np.array([r['signer_id'] for r in adev])
    diag = {}
    for a, P in pooled_dev.items():
        z = P / np.linalg.norm(P, axis=1, keepdims=True)
        sim = z @ z.T
        same = (lab[:, None] == lab[None, :]) & (sg[:, None] != sg[None, :])
        diff = lab[:, None] != lab[None, :]
        diag[a] = {'same_word_diff_signer': float(sim[same].mean()), 'different_word': float(sim[diff].mean()),
                   'gap': float(sim[same].mean() - sim[diff].mean())}

    failed = [k for k, v in checks.items() if not v]
    assert not failed, failed
    me = all(gates['method_effect_vs_control_reproduction'].values())
    pi = all(gates['project_improvement_vs_baseline_83212868'].values())
    with (R / 'evaluation_arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    result = {'status': 'PUBLIC_EVALUATION_COMPLETE',
              'decision': {'method_effect': 'PASS' if me else 'FAIL', 'project_improvement': 'PASS' if pi else 'FAIL',
                           'adopt': bool(me and pi)},
              'method_control': 'control_reproduction (lambda 0; not bitwise equal to af6c0119, see training.json)',
              'checks': checks, 'gates': gates, 'summary': summary, 'changes': changes,
              'per_word_differences_annotation_dev': per_word,
              'control_reproduction_vs_af6c0119_prediction_differences': int(repro_vs_prior),
              'diagnostic_cosine_annotation_dev': diag, 'rows': rows, 'arms': ARMS,
              'policy_sha256': sha(H / 'comparison_policy.json'), 'training_record_sha256': sha(R / 'training.json'),
              'arrays_sha256': sha(R / 'evaluation_arrays.npz'),
              'user_arrays_or_scores_read': 0, 'reserved_arrays_read': 0, 'default_promoted': False,
              'seconds': round(time.time() - started, 2)}
    with (R / 'evaluation.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1, allow_nan=False)
    for a in ARMS:
        print(f"{a:32s} dev {S('annotation_dev', a)['correct']}/699 macro {S('annotation_dev', a)['macro_recall']:.6f} "
              f"p18 {S('annotation_dev', a, sc='project18')['correct']}/33 whole {S('whole_dev', a)['correct']}/74 "
              f"signers {[S('annotation_dev', a)['per_signer'][k]['correct'] for k in ('11', '12', '13')]} "
              f"public {[S('public', a, c=c)['correct'] for c in ('intact', 'reviewed_masked', 'additional_train')]} "
              f"mp {[S('mp', a, c=c)['correct'] for c in ('whole_clip', 'provided_annotation')]}")
    print('decision', result['decision'])
    print('failed gates', {k: [g for g, v in gg.items() if not v] for k, gg in gates.items()})
    for against in ['control_reproduction', 'baseline_83212868']:
        print('vs', against, {g: (len(v['gains']), len(v['regressions'])) for g, v in changes[against].items()})
    print('repro vs af6c0119 prediction differences', repro_vs_prior)
    print('diagnostic', json.dumps(diag))


if __name__ == '__main__':
    main()
