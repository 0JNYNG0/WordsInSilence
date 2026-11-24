"""Read-only attribution of fixed checkpoint decisions; no optimizer or model changes."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

OPS = Path(__file__).resolve().parent
WEB = OPS.parent.parent
sys.path[:0] = [str(WEB), str(WEB / 'src'), str(WEB.parent / 'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u
from pilot.data import build_channels, collate
from wis_lab.core.observation import Observation
from wis_lab.service import neural

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    plan = read(OPS / 'plan_LOCAL_ONLY.json')
    target = OPS / 'result_LOCAL_ONLY.json'
    assert not target.exists() and not (OPS / 'features_LOCAL_ONLY.npz').exists()
    for path, digest in plan['source_protected'].items():
        assert sha(path) == digest, path
    torch.set_num_threads(2)
    models = {}
    for role, spec in plan['models'].items():
        assert sha(spec['path']) == spec['sha256']
        models[role] = torch.load(spec['path'], weights_only=True, map_location='cpu')
    parent_payload = models['parent']
    classes = parent_payload['classes']
    parent_state = parent_payload['state_dict']
    for role, payload in models.items():
        assert payload['classes'] == classes
        assert set(payload['state_dict']) == set(parent_state)
        assert all(torch.equal(v, payload['state_dict'][name])
                   for name, v in parent_state.items() if not name.startswith('head.'))
    model, model_classes = u.rebuild_from_payload(parent_payload)
    assert model_classes == classes
    model.eval()
    model.requires_grad_(False)
    heads = {role: (p['state_dict']['head.weight'].numpy().astype(np.float64),
                    p['state_dict']['head.bias'].numpy().astype(np.float64))
             for role, p in models.items()}
    words = {wid: spec['label'] for wid, spec in plan['words'].items()}
    target_indices = [classes.index(label) for label in words.values()]
    scopes = {'project18': target_indices, 'full93': list(range(len(classes)))}
    saved = {path: read(path) for cohort in plan['saved_outputs'].values() for path in cohort.values()}
    captured = []
    def capture(module, args):
        captured.append(args[0].detach().cpu().numpy().copy()[0])
    handle = model.head.register_forward_pre_hook(capture)
    records, features, actual_parent_logits = [], [], []
    max_affine_error, max_saved_score_error, saved_score_checks, prediction_checks = 0., 0., 0, 0
    try:
        with torch.inference_mode():
            for cohort, rows in plan['cohorts'].items():
                for item in rows:
                    if cohort == 'public_training18':
                        assert sha(item['npz']) == item['sha256']
                        native, present, timestamps = u._native_from_nikl(item['npz'], item['sha256'])
                        clip = build_channels(native, present, timestamps)
                        ident, truth = item['id'], item['label']
                        input_sha = item['sha256']
                    else:
                        assert sha(item['observation']) == item['observation_sha256']
                        obs = Observation.load(item['observation'])
                        clip, reason, support = neural._channels(obs)
                        assert clip is not None, reason
                        ident, truth = item['trial_id'], item['actual_label']
                        input_sha = item['observation_sha256']
                    x, pad, _ = collate([{'x': clip['x'], 'frames': len(clip['x']), 'label': 0}])
                    n = len(captured)
                    original_logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy().copy()
                    assert len(captured) == n + 1
                    z = captured[-1].astype(np.float64)
                    logits = {role: weight @ z + bias for role, (weight, bias) in heads.items()}
                    max_affine_error = max(max_affine_error, float(np.max(np.abs(logits['parent'] - original_logit))))
                    assert max_affine_error <= plan['score_agreement_tolerance']
                    yi = classes.index(truth)
                    record = {'cohort': cohort, 'id': ident, 'truth': truth, 'input_sha256': input_sha,
                              'feature_index': len(features), 'frames': int(len(clip['x'])),
                              'feature_norm': float(np.linalg.norm(z)), 'models': {}, 'changes_vs_parent': {}}
                    for role, scores in logits.items():
                        record['models'][role] = {}
                        for scope, indices in scopes.items():
                            order = sorted(indices, key=lambda i: -scores[i])
                            competitor = next(i for i in order if i != yi)
                            pred = classes[order[0]]
                            margin = float(scores[yi] - scores[competitor])
                            record['models'][role][scope] = {'prediction': pred, 'correct': pred == truth,
                                'true_score': float(scores[yi]), 'competitor': classes[competitor],
                                'competitor_score': float(scores[competitor]), 'true_margin': margin,
                                'truth_rank': order.index(yi) + 1}
                            if role != 'parent':
                                parent_scores = logits['parent']
                                parent_same_margin = float(parent_scores[yi] - parent_scores[competitor])
                                w, b = heads[role]
                                pw, pb = heads['parent']
                                weight_change = float(((w[yi]-w[competitor])-(pw[yi]-pw[competitor])) @ z)
                                bias_change = float((b[yi]-b[competitor])-(pb[yi]-pb[competitor]))
                                residual = margin - parent_same_margin - weight_change - bias_change
                                assert abs(residual) < 1e-10
                                record['changes_vs_parent'].setdefault(role, {})[scope] = {
                                    'competitor': classes[competitor], 'parent_margin_same_competitor': parent_same_margin,
                                    'child_margin': margin, 'weight_contribution': weight_change,
                                    'bias_contribution': bias_change, 'residual': residual}
                        expected_path = plan['saved_outputs'].get(cohort, {}).get(role)
                        if expected_path:
                            old = saved[expected_path]
                            old_role = role.removeprefix('quota_')
                            if 'predictions' in old:
                                output = old['predictions'][ident][old_role]
                                expected = {'project18': words[output['project18']], 'full93': output['full93']}
                                score_rows = []
                            else:
                                output = next(r for r in old['rows'] if r['trial_id'] == ident)['arms'][old_role]
                                expected = {'project18': words[output['scopes']['project18']['top1']],
                                            'full93': output['scopes']['full']['top1']}
                                score_rows = output['scopes']['project18']['order'] + output['scopes']['full']['top3']
                            for scope, pred in expected.items():
                                assert record['models'][role][scope]['prediction'] == pred, (cohort, ident, role, scope)
                                prediction_checks += 1
                            for row in score_rows:
                                difference = abs(float(scores[classes.index(row['label'])]) - row['score'])
                                max_saved_score_error = max(max_saved_score_error, difference)
                                assert difference <= plan['score_agreement_tolerance']
                                saved_score_checks += 1
                    records.append(record)
                    features.append(captured[-1])
                    actual_parent_logits.append(original_logit)
    finally:
        handle.remove()
    assert len(records) == 54
    assert all(p.grad is None for p in model.parameters())
    public_features = {r['truth']: features[r['feature_index']].astype(np.float64)
                       for r in records if r['cohort'] == 'public_training18'}
    for row in records:
        if row['cohort'].startswith('self_'):
            z = features[row['feature_index']].astype(np.float64)
            similarity = {label: float(z @ v / (np.linalg.norm(z)*np.linalg.norm(v)))
                          for label, v in public_features.items()}
            ordered = sorted(similarity, key=lambda label: -similarity[label])
            row['public_reference_feature_geometry'] = {
                'own_reference_cosine': similarity[row['truth']],
                'own_reference_cosine_rank_among18': ordered.index(row['truth']) + 1,
                'closest_label': ordered[0], 'closest_cosine': similarity[ordered[0]],
                'norm_ratio_to_own_reference': float(np.linalg.norm(z)/np.linalg.norm(public_features[row['truth']])),
                'meaning': 'Descriptive geometry only; not a classifier or correctness verdict'}
    for path, digest in plan['source_protected'].items():
        assert sha(path) == digest
    for spec in plan['models'].values():
        assert sha(spec['path']) == spec['sha256']
    np.savez_compressed(OPS / 'features_LOCAL_ONLY.npz', features=np.stack(features),
                        parent_logits=np.stack(actual_parent_logits))
    result = {'status': 'ATTRIBUTION_RECONSTRUCTED_PENDING_INDEPENDENT_RECOUNT',
              'plan_sha256': sha(OPS / 'plan_LOCAL_ONLY.json'), 'cohort_sizes': {k: len(v) for k,v in plan['cohorts'].items()},
              'encoder_forwards': 54, 'child_encoder_forwards': 0, 'optimizer_steps': 0,
              'all_non_head_tensors_equal': True, 'all_parameters_no_grad': True,
              'max_parent_affine_error': max_affine_error, 'max_saved_score_error': max_saved_score_error,
              'saved_prediction_checks': prediction_checks, 'saved_score_checks': saved_score_checks,
              'features_sha256': sha(OPS / 'features_LOCAL_ONLY.npz'), 'records': records,
              'models_changed': False, 'training_user_data_used': False,
              'limits': 'One development participant, public training examples, linear decision attribution only; no causal claim about detector quality or population generalization'}
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'encoder_forwards', 'max_parent_affine_error',
                                          'max_saved_score_error', 'saved_prediction_checks', 'saved_score_checks')}))

if __name__ == '__main__':
    main()
