"""Operational evaluation of two confirmed public clips with existing model APIs."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

H = Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]
from tools_prep_union93_candidate import rebuild_from_payload
from pilot.data import collate
from wis_lab.service.neural import PROJECT_18

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, data):
    with (H / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    started = time.perf_counter()
    assert not (H / 'result.json').exists()
    plan = read(H / 'plan.json')
    for p, digest in plan['source_pins'].items():
        assert sha(p) == digest, p
    checkpoints = {}
    for name, spec in plan['models'].items():
        assert sha(spec['path']) == spec['sha256']
        checkpoints[name] = torch.load(spec['path'], map_location='cpu', weights_only=True)
    parent = checkpoints['pre_context_parent']
    classes = parent['classes']
    assert len(classes) == len(set(classes)) == 93
    nonhead = [k for k in parent['state_dict'] if not k.startswith('head.')]
    assert len(nonhead) == 58
    for ck in checkpoints.values():
        assert ck['classes'] == classes
        assert all(torch.equal(parent['state_dict'][k], ck['state_dict'][k]) for k in nonhead)
    model, _ = rebuild_from_payload(parent)
    model.eval().requires_grad_(False)
    torch.set_num_threads(2)
    captured = []
    handle = model.head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().numpy().copy()[0]))
    direct = []
    try:
        for spec in plan['rows']:
            assert spec['id'] in ('KSL77_00_48', 'KSL77_00_16')
            assert sha(spec['input']) == spec['input_sha256']
            with np.load(spec['input'], allow_pickle=False) as z:
                x = z['x'].copy()
            assert x.shape == (spec['frames'], 280) and np.isfinite(x).all()
            bx, mask, _ = collate([dict(x=x, frames=len(x), label=0)])
            with torch.inference_mode():
                direct.append(model(torch.from_numpy(bx), torch.from_numpy(mask)).numpy()[0])
    finally:
        handle.remove()
    features = np.array(captured)
    assert features.shape == (2, 64)
    direct = np.array(direct)
    target = [i for i, c in enumerate(classes) if c in set(PROJECT_18.values())]
    assert len(target) == 18
    score_arrays = {}
    report_rows = []
    for name, ck in checkpoints.items():
        w = ck['state_dict']['head.weight'].numpy().astype(np.float64)
        b = ck['state_dict']['head.bias'].numpy().astype(np.float64)
        scores = features.astype(np.float64) @ w.T + b
        score_arrays[name] = scores
        for i, spec in enumerate(plan['rows']):
            y = classes.index(spec['label'])
            row = dict(id=spec['id'], label=spec['label'], model=name, scopes={})
            for scope, indices in [('full93', list(range(93))), ('project18', target)]:
                order = sorted(indices, key=lambda j: (-float(scores[i, j]), j))
                rival = next(j for j in order if j != y)
                row['scopes'][scope] = dict(
                    prediction=classes[order[0]], correct=bool(order[0] == y),
                    true_rank=order.index(y) + 1,
                    true_minus_best_other=float(scores[i, y] - scores[i, rival]),
                    top3=[dict(label=classes[j], logit=float(scores[i, j])) for j in order[:3]])
            report_rows.append(row)
    delta = float(np.max(np.abs(direct - score_arrays['pre_context_parent'])))
    assert delta < 5e-5
    assert np.array_equal(np.argmax(direct, axis=1), np.argmax(score_arrays['pre_context_parent'], axis=1))
    assert all(torch.equal(model.state_dict()[k], parent['state_dict'][k]) for k in parent['state_dict'])
    for p, digest in plan['source_pins'].items():
        assert sha(p) == digest, p
    for spec in plan['models'].values():
        assert sha(spec['path']) == spec['sha256']
    with (H / 'features_and_scores.npz').open('xb') as f:
        np.savez_compressed(f, features=features, direct_parent_scores=direct,
                            classes=np.array(classes), ids=np.array([r['id'] for r in plan['rows']]),
                            **score_arrays)
    result = dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT', rows=report_rows,
                  plan_sha256=sha(H / 'plan.json'), script_sha256=sha(__file__),
                  arrays_sha256=sha(H / 'features_and_scores.npz'),
                  shared_encoder_tensors=58, new_encoder_forwards=2,
                  distinct_clip_model_outputs=4, score_values=372, scoped_rankings=8,
                  direct_parent_float32_max_difference=delta,
                  duration_seconds=time.perf_counter() - started,
                  training_runs=0, new_extraction=0, other34_media_read=0,
                  user_recordings_read=0, reserved_arrays_read=0, models_modified=False,
                  selection_or_promotion=False, new_person_generalization_claim=False,
                  semantic_confirmation=plan['semantic_confirmation'], limits=plan['interpretation_limits'])
    save('result.json', result)
    print(json.dumps(dict(status=result['status'], rows=report_rows, forwards=2), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
