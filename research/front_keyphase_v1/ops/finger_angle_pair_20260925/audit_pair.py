"""Independent recomputation of the recorded result. No training, no model forward."""
import hashlib, importlib.util, json, pathlib, sys
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
R = L / 'runs/finger_angle_pair_v1'
sys.path[:0] = [str(H), str(L), str(L / 'src')]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


AB = module('angle_pair_features', H / 'angle_block.py')
from wis_lab.service.neural import PROJECT_18            # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
GROUPS = ['annotation_dev', 'whole_dev', 'public', 'mp']


def main():
    pol = read(H / 'comparison_policy.json')
    ready = read(H / 'INPUT_READY.json')
    tr = read(R / 'training.json')
    ev = read(R / 'evaluation.json')
    a = {}

    # ---- artifacts ---------------------------------------------------------
    a['policy_hash_in_training'] = sha(H / 'comparison_policy.json') == tr['policy_sha256']
    a['training_hash_in_evaluation'] = sha(R / 'training.json') == ev['training_record_sha256']
    a['evaluation_arrays_hash'] = sha(R / 'evaluation_arrays.npz') == ev['arrays_sha256']
    a['blocks_hash'] = sha(H / 'geometry_blocks.npz') == ready['blocks_sha256']
    a['parent_unchanged'] = sha(pol['matched_pair']['parent']['path']) == pol['matched_pair']['parent']['sha256']
    a['input_cache_unchanged'] = sha(ready['input_cache']['path']) == ready['input_cache']['sha256']
    a['policy_was_frozen_before_training'] = pol['status'] == 'FROZEN_BEFORE_TRAINING'

    # ---- the two arms differ only where they must --------------------------
    pay = {}
    for arm, spec in tr['models'].items():
        a[f'{arm}_hash'] = sha(spec['path']) == spec['sha256']
        pay[arm] = torch.load(spec['path'], map_location='cpu', weights_only=True)
    parent = torch.load(pol['matched_pair']['parent']['path'], map_location='cpu',
                        weights_only=True)['state_dict']
    c, t = pay['control']['state_dict'], pay['treatment']['state_dict']
    a['features_are_chord_and_angle'] = (pay['control']['feature'] == 'chord'
                                         and pay['treatment']['feature'] == 'angle')
    a['encoder_identical_between_arms'] = all(
        torch.equal(c[k], t[k]) for k in c
        if not k.startswith('base.head.') and k not in ('extra_weight', 'extra_scale'))
    a['encoder_identical_to_parent'] = all(
        torch.equal(c[k], parent[k[len('base.'):]]) for k in c
        if k.startswith('base.') and not k.startswith('base.head.'))
    a['only_head_changed_from_parent'] = all(
        sorted(k[len('base.'):] for k in sd if k.startswith('base.')
               and not torch.equal(sd[k], parent[k[len('base.'):]])) == ['head.bias', 'head.weight']
        for sd in (c, t))
    a['extra_weight_learned_nonzero'] = all(int(torch.count_nonzero(sd['extra_weight'])) > 0
                                            for sd in (c, t))
    a['extra_weight_shapes_match'] = c['extra_weight'].shape == t['extra_weight'].shape
    a['trainable_counts_matched'] = (tr['models']['control']['trainable_count']
                                     == tr['models']['treatment']['trainable_count'] == 10909)
    a['both_660_updates'] = tr['optimizer_updates'] == 1320 and tr['real_training_arms'] == 2
    a['no_seed_or_coefficient_search'] = tr['seed_or_coefficient_search'] == 0 and tr['attempts'] == 1
    a['default_not_promoted'] = tr['default_promoted'] is False and ev['default_promoted'] is False
    a['no_user_or_reserved_read'] = (tr['user_arrays_read'] == 0 and tr['reserved_arrays_read'] == 0
                                     and ev['user_arrays_or_scores_read'] == 0
                                     and ev['reserved_arrays_read'] == 0 and ev['ksl_arrays_read'] == 0)

    # ---- the two blocks are width matched and share validity ---------------
    with np.load(H / 'geometry_blocks.npz', allow_pickle=False) as z:
        classes = z['classes'].tolist()
        ga, ca = z['base_angle'], z['base_chord']
        a['blocks_same_shape'] = ga.shape == ca.shape and ga.shape[1] == AB.EXTRA_WIDTH == 76
        a['blocks_share_validity'] = np.array_equal(ga[:, AB.EXTRA_VALUES:], ca[:, AB.EXTRA_VALUES:])
        m = ga[:, AB.EXTRA_VALUES:] > 0.5
        a['invalid_zero_in_both'] = (int(np.count_nonzero(ga[:, :AB.EXTRA_VALUES][~m])) == 0
                                     and int(np.count_nonzero(ca[:, :AB.EXTRA_VALUES][~m])) == 0)
        a['blocks_are_different_features'] = not np.allclose(ga[:, :AB.EXTRA_VALUES],
                                                             ca[:, :AB.EXTRA_VALUES])
    a['classes_93'] = len(classes) == 93

    # ---- independent count recomputation from stored scores ----------------
    target = {i for i, x in enumerate(classes) if x in set(PROJECT_18.values())}
    rows = ev['rows']
    counts = {}
    with np.load(R / 'evaluation_arrays.npz', allow_pickle=False) as z:
        assert z['classes'].tolist() == classes
        for g in GROUPS:
            rs = [r for r in rows if r['group'] == g]
            for arm in ('control', 'treatment'):
                s = z[f'{g}__{arm}__scores']
                assert s.shape[0] == len(rs)
                tot93 = tot18 = 0
                per_signer, per_class = {}, {}
                for i, r in enumerate(rs):
                    yi = classes.index(r['true'])
                    ok = int(np.argmax(s[i]) == yi)
                    # the recorded row must agree with a fresh argmax over the stored scores
                    assert classes[int(np.argmax(s[i]))] == r['models'][arm]['full93']['pred']
                    tot93 += ok
                    pc = per_class.setdefault(r['true'], [0, 0]); pc[0] += 1; pc[1] += ok
                    if 'signer_id' in r:
                        ps = per_signer.setdefault(str(r['signer_id']), [0, 0]); ps[0] += 1; ps[1] += ok
                    if yi in target:
                        ti = sorted(target, key=lambda j: -float(s[i][j]))
                        tot18 += int(ti[0] == yi)
                counts[g, arm] = dict(full93=tot93, project18=tot18,
                                      macro=float(np.mean([v[1] / v[0] for v in per_class.values()])),
                                      per_signer={k: v[1] for k, v in per_signer.items()})
    a['rows_agree_with_stored_scores'] = True

    S = ev['summary']
    rec = lambda g, arm, sc: sum(S[g][arm][cd][sc]['correct'] for cd in S[g][arm])
    a['annotation_control_690'] = counts['annotation_dev', 'control']['full93'] == rec('annotation_dev', 'control', 'full93') == 690
    a['annotation_treatment_685'] = counts['annotation_dev', 'treatment']['full93'] == rec('annotation_dev', 'treatment', 'full93') == 685
    a['annotation_project18_33_both'] = (counts['annotation_dev', 'control']['project18']
                                         == counts['annotation_dev', 'treatment']['project18'] == 33)
    a['whole_control_74'] = counts['whole_dev', 'control']['full93'] == 74
    a['whole_treatment_72'] = counts['whole_dev', 'treatment']['full93'] == 72
    a['public_32_both'] = counts['public', 'control']['full93'] == counts['public', 'treatment']['full93'] == 32
    a['mp_14_both'] = counts['mp', 'control']['full93'] == counts['mp', 'treatment']['full93'] == 14
    a['macro_recall_decreased'] = counts['annotation_dev', 'treatment']['macro'] < counts['annotation_dev', 'control']['macro']
    a['per_signer_matches'] = (counts['annotation_dev', 'control']['per_signer'] == {'11': 228, '12': 234, '13': 228}
                               and counts['annotation_dev', 'treatment']['per_signer'] == {'11': 229, '12': 231, '13': 225})

    # ---- gate arithmetic ---------------------------------------------------
    a['gate_increase_correctly_failed'] = ev['gates']['annotation_dev_increase'] is False and 685 < 690
    a['gate_whole_retention_correctly_failed'] = ev['gates']['whole_dev_maintained_74'] is False and 72 != 74
    a['gate_signer_correctly_failed'] = ev['gates']['annotation_each_signer_nondecreasing'] is False
    a['gate_public_mp_correctly_passed'] = all(v for k, v in ev['gates'].items() if k.startswith(('public_', 'mp_')))
    a['decision_is_rejection'] = ev['decision'] == 'REJECTED_BY_PREDECLARED_GATES'
    a['whole_dev_was_retention_only'] = 'MAINTENANCE ONLY' in pol['predeclared_decision']['whole_dev_74']
    gains = sum(len(v['gains']) for v in ev['changes'].values())
    regs = sum(len(v['regressions']) for v in ev['changes'].values())
    a['two_gains_nine_regressions'] = gains == 2 and regs == 9

    # ---- reference baseline kept separate from the pair control ------------
    ref = ev['reference_baseline_frozen_control_83212868']
    a['reference_is_a_different_model'] = ref['sha256'] != tr['models']['control']['sha256']
    a['reference_numbers_preserved'] = ref['annotation_dev_699'] == 691 and ref['whole_dev_74'] == 74
    a['pair_control_is_not_the_reference'] = counts['annotation_dev', 'control']['full93'] != ref['annotation_dev_699']

    failed = [k for k, v in a.items() if v is False]
    out = {'status': 'PASS' if not failed else 'FAIL',
           'assertions_total': len(a), 'assertions_failed': failed,
           'assertions': {k: bool(v) for k, v in a.items()},
           'recomputed_counts': {f'{g}|{arm}': v for (g, arm), v in counts.items()},
           'training_runs': 0, 'model_forward_passes': 0,
           'user_arrays_read': 0, 'reserved_arrays_read': 0,
           'verdict': ('The recorded rejection reproduces independently. Against its own width-matched '
                       'chord control the angle arm lost 5 annotation_dev cases and 2 whole_dev cases, '
                       'gained 2, and left public and MP unchanged.')}
    with (H / 'INDEPENDENT_AUDIT.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('assertions', len(a), 'failed', failed)
    print('annotation_dev  chord', counts['annotation_dev', 'control']['full93'],
          '-> angle', counts['annotation_dev', 'treatment']['full93'])
    print('whole_dev       chord', counts['whole_dev', 'control']['full93'],
          '-> angle', counts['whole_dev', 'treatment']['full93'])
    print('status', out['status'])


if __name__ == '__main__':
    main()
