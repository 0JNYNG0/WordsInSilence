"""Independent scalar score/ranking audit; no encoder, video or input reprocessing."""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import torch

H = Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]
from wis_lab.service.neural import PROJECT_18

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plan, result = read(H / 'plan.json'), read(H / 'result.json')
    assert result['plan_sha256'] == sha(H / 'plan.json')
    assert result['arrays_sha256'] == sha(H / 'features_and_scores.npz')
    for p, digest in plan['source_pins'].items():
        assert sha(p) == digest, p
    confirmation = read(plan['semantic_confirmation'])
    expected = {'KSL77_00_48': '공부', 'KSL77_00_16': '먹다'}
    assert {r['id']: r['label'] for r in confirmation['pairs']} == expected
    assert confirmation['training_registration'] is False
    rows = {(r['id'], r['model']): r for r in result['rows']}
    assert len(rows) == 4 and len(result['rows']) == 4
    errors, rankings = [], 0
    with np.load(H / 'features_and_scores.npz', allow_pickle=False) as z:
        features, classes, ids = z['features'], z['classes'].tolist(), z['ids'].tolist()
        assert features.shape == (2, 64) and len(classes) == 93
        assert set(ids) == set(expected)
        target = [j for j, c in enumerate(classes) if c in set(PROJECT_18.values())]
        assert len(target) == 18
        for name, spec in plan['models'].items():
            assert sha(spec['path']) == spec['sha256']
            ck = torch.load(spec['path'], map_location='cpu', weights_only=True)
            assert ck['classes'] == classes
            w, b = ck['state_dict']['head.weight'].numpy(), ck['state_dict']['head.bias'].numpy()
            for i, ident in enumerate(ids):
                scalar = [math.fsum(float(features[i, k]) * float(w[j, k]) for k in range(64)) + float(b[j]) for j in range(93)]
                errors.extend(abs(float(z[name][i, j]) - scalar[j]) for j in range(93))
                row, y = rows[ident, name], classes.index(expected[ident])
                assert row['label'] == expected[ident]
                for scope, indices in [('full93', list(range(93))), ('project18', target)]:
                    order = sorted(indices, key=lambda j: (-scalar[j], j))
                    r = row['scopes'][scope]
                    assert r['prediction'] == classes[order[0]]
                    assert r['correct'] == (order[0] == y)
                    assert r['true_rank'] == order.index(y) + 1
                    margin = scalar[y] - max(scalar[j] for j in indices if j != y)
                    assert abs(r['true_minus_best_other'] - margin) < 1e-10
                    assert [r['label'] for r in r['top3']] == [classes[j] for j in order[:3]]
                    for item, j in zip(r['top3'], order[:3]):
                        assert abs(item['logit'] - scalar[j]) < 1e-10
                    rankings += 1
    assert len(errors) == 372 and rankings == 8 and max(errors) < 1e-10
    assert all(result[k] == 0 for k in ('training_runs', 'new_extraction', 'other34_media_read', 'user_recordings_read', 'reserved_arrays_read'))
    output = dict(status='PASS', result_sha256=sha(H / 'result.json'),
                  semantic_pairs_bound=2, score_values_recomputed=len(errors), rankings_recomputed=rankings,
                  max_score_error=max(errors), encoder_forwards_this_audit=0,
                  training_runs=0, signer_disjointness_verified=False,
                  decision='DIAGNOSTIC_ONLY_NO_SELECTION_OR_PROMOTION')
    with (H / 'INDEPENDENT_AUDIT.json').open('x', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
