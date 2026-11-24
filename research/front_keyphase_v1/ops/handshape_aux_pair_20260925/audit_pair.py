"""Independent recomputation. No training, no encoder run, no model forward."""
import hashlib, json, pathlib, sys
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
R = L / 'runs/handshape_aux_pair_v1'
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.service.neural import PROJECT_18  # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
GROUPS = ['annotation_dev', 'whole_dev', 'public', 'mp']


def main():
    p = read(H / 'comparison_policy.json')
    tr = read(R / 'training.json')
    ev = read(R / 'evaluation.json')
    ver = read(H / 'ADAPTER_VERIFICATION.json')
    a = {}

    a['policy_frozen_before_training'] = p['status'] == 'FROZEN_BEFORE_TRAINING'
    a['adapter_verification_pass'] = ver['status'] == 'PASS' and ver['checks_failed'] == []
    a['training_hash'] = sha(R / 'training.json') == ev['training_record_sha256']
    a['arrays_hash'] = sha(R / 'evaluation_arrays.npz') == ev['arrays_sha256']
    a['candidate_hash'] = sha(tr['candidate']['path']) == tr['candidate']['sha256']
    a['parent_unchanged'] = sha(L / p['parent']['path']) == p['parent']['sha256']
    a['control_unchanged'] = sha(L / p['control']['path']) == p['control']['sha256']
    a['control_not_retrained'] = ev['control_retrained'] is False and p['control']['retrain'] is False
    a['one_attempt_no_search'] = tr['attempts'] == 1 and tr['seed_or_coefficient_search'] == 0
    a['default_not_promoted'] = tr['default_promoted'] is False and ev['default_promoted'] is False
    a['no_user_or_reserved'] = (tr['user_arrays_read'] == 0 and tr['reserved_arrays_read'] == 0
                                and tr['development_arrays_read'] == 0
                                and ev['user_arrays_or_scores_read'] == 0 and ev['reserved_arrays_read'] == 0)

    cand = torch.load(tr['candidate']['path'], map_location='cpu', weights_only=True)
    sd = cand['state_dict']
    parent = torch.load(L / p['parent']['path'], map_location='cpu', weights_only=True)['state_dict']
    ctrl = torch.load(L / p['control']['path'], map_location='cpu', weights_only=True)['state_dict']
    changed = sorted(k for k in parent if not torch.equal(parent[k], sd['base.' + k]))
    expect = sorted([f'blocks.3.{n}' for n in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                                               'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']]
                    + ['head.weight', 'head.bias'])
    a['changed_exactly_blocks3_and_head'] = changed == expect
    a['earlier_blocks_and_proj_untouched'] = all(
        torch.equal(parent[k], sd['base.' + k]) for k in parent
        if k.startswith('proj.') or (k.startswith('blocks.') and not k.startswith('blocks.3.')))
    a['candidate_shared_block_differs_from_control'] = not torch.equal(
        sd['base.blocks.3.conv1.weight'], ctrl['blocks.3.conv1.weight'])
    a['aux_head_learned'] = int(torch.count_nonzero(sd['shape_head.weight'])) > 0
    a['aux_weight_is_the_declared_one'] = abs(cand['aux_weight'] - p['execution_settings']['lambda']) < 1e-12

    # labels
    with np.load(H / 'shape_labels.npz', allow_pickle=False) as z:
        bs, cs = z['base_shape_y'], z['pair_context_shape_y']
        codes = z['shape_codes'].tolist()
    fs = read(H / 'FROZEN_LABEL_SET.json')
    a['label_classes_63'] = len(fs['label_map']) == 63
    a['shape_classes_24'] = len(codes) == 24 == len({v for v in fs['label_map'].values()})
    a['excluded_30_deduped'] = len(set(fs['excluded'])) == 30
    a['targets_labelled_16'] = len(fs['targets_labelled']) == 16
    a['joesonghada_held_and_unlabelled'] = '죄송하다' in fs['held'] and '죄송하다' not in fs['label_map']
    a['insa_excluded_multi_code'] = '인사' in fs['multi_code'] and '인사' not in fs['label_map']
    a['base_labelled_1364'] = int((bs >= 0).sum()) == 1364
    a['context_labelled_162'] = int((cs >= 0).sum()) == 162

    # independent count recomputation from stored scores + stored prior rows
    classes = np.array(read(L / 'runs/partial_encoder_pair_v1/evaluation.json')['rows'] and
                       None) if False else None
    with np.load(R / 'evaluation_arrays.npz', allow_pickle=False) as z:
        cls = z['classes'].tolist()
        counts = {}
        for g in GROUPS:
            s = z[f'{g}__candidate__scores']
            rs = [r for r in ev['rows'] if r['group'] == g]
            assert len(rs) == s.shape[0]
            tgt = {i for i, c in enumerate(cls) if c in set(PROJECT_18.values())}
            t93 = t18 = 0
            sig = {}
            for i, r in enumerate(rs):
                yi = cls.index(r['true'])
                ok = int(np.argmax(s[i]) == yi)
                assert cls[int(np.argmax(s[i]))] == r['models']['candidate']['full93']['pred']
                t93 += ok
                if 'signer_id' in r:
                    sig.setdefault(str(r['signer_id']), 0)
                    sig[str(r['signer_id'])] += ok
                if yi in tgt:
                    order = sorted(tgt, key=lambda j: -float(s[i][j]))
                    t18 += int(order[0] == yi)
            counts[g] = dict(full93=t93, project18=t18, per_signer=sig)
    a['recount_annotation_690'] = counts['annotation_dev']['full93'] == 690
    a['recount_whole_74'] = counts['whole_dev']['full93'] == 74
    a['recount_public_32'] = counts['public']['full93'] == 32
    a['recount_mp_14'] = counts['mp']['full93'] == 14
    a['recount_project18_32'] = counts['annotation_dev']['project18'] == 32
    a['recount_per_signer'] = counts['annotation_dev']['per_signer'] == {'11': 229, '12': 233, '13': 228}

    S = ev['summary']
    tot = lambda g, arm, sc='full93': sum(S[g][arm][c][sc]['correct'] for c in S[g][arm])
    a['baseline_691_reused'] = tot('annotation_dev', 'baseline_83212868') == 691
    a['control_690_reused'] = tot('annotation_dev', 'control_af6c0119') == 690
    a['candidate_equals_control_total'] = tot('annotation_dev', 'candidate') == 690
    a['gate_increase_correctly_failed'] = ev['gates']['annotation_dev_increase_vs_control'] is False
    a['retention_gates_passed'] = all(v for k, v in ev['gates'].items() if k != 'annotation_dev_increase_vs_control')
    a['decision_is_rejection'] = ev['decision'] == 'REJECTED_BY_PREDECLARED_GATES'
    a['no_project_improvement_claimed'] = ev['project_improvement_claimed'] is False
    a['project_delta_is_minus_one'] = ev['project_improvement_vs_baseline'] == -1
    g = sum(len(v['gains']) for v in ev['changes']['control_af6c0119'].values())
    r = sum(len(v['regressions']) for v in ev['changes']['control_af6c0119'].values())
    a['one_gain_one_regression_vs_control'] = g == 1 and r == 1
    g2 = sum(len(v['gains']) for v in ev['changes']['baseline_83212868'].values())
    r2 = sum(len(v['regressions']) for v in ev['changes']['baseline_83212868'].values())
    a['one_gain_two_regressions_vs_baseline'] = g2 == 1 and r2 == 2

    failed = [k for k, v in a.items() if v is False]
    out = {'status': 'PASS' if not failed else 'FAIL', 'assertions_total': len(a),
           'assertions_failed': failed, 'assertions': {k: bool(v) for k, v in a.items()},
           'recomputed_counts': counts, 'training_runs': 0, 'model_forward_passes': 0,
           'verdict': 'The recorded rejection reproduces. The candidate matches its control on '
                      'annotation_dev (690) with one gain and one regression, keeps every retention '
                      'condition, and sits one below the existing project baseline (691).'}
    with (H / 'INDEPENDENT_AUDIT.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('assertions', len(a), 'failed', failed)
    print('status', out['status'])


if __name__ == '__main__':
    main()
