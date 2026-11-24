"""One fixed sparse-probe evaluation, only after root's concrete code review."""
import collections
import time
import traceback

from probe_common import *


def main():
    stop()
    assert not (U / 'EVALUATION_STARTED.json').exists(), 'Do not rerun a completed or partial inference.'
    assert not (U / 'RESULTS.json').exists()
    cfg = read(U / 'EVALUATION_CONTRACT.json')
    check(cfg['pins'])
    check(cfg['protected_pins'])
    assert len(cfg['protected_pins']) == 477 and cfg['protected_pins'] == read(G / 'PROTECTIONS.json')
    approval = read(U / 'ROOT_REVIEW.json')
    assert approval['status'] == 'APPROVED_FOR_SINGLE_EVALUATION'
    assert approval['contract_sha256'] == sha(U / 'EVALUATION_CONTRACT.json')
    required = {str(U / name): sha(U / name) for name in ['evaluate_probe.py', 'prepare_probe.py', 'probe_common.py', 'RGB_REVIEW.json', 'INPUTS_COMPLETE.json', 'LABEL_AUDIT.json']}
    assert all(approval['pins'].get(p) == digest for p, digest in required.items())
    check(approval['pins'])
    assert cfg['seeds'] == SEEDS and cfg['threshold_logit'] == 0. and cfg['new_forwards_per_seed'] == 48
    meta = read(U / 'INPUTS_COMPLETE.json')
    fit_meta = read(BP / 'INPUTS_COMPLETE.json')
    training = read(BP / 'TRAINING_COMPLETE.json')
    assert {r['seed'] for r in training['rows']} == set(SEEDS) and len(training['rows']) == 3
    assert training['training_frozen_sha256'] == sha(BP / 'TRAINING_FROZEN.json')
    assert meta['family_order'] == F and len(meta['sequences']) == 48
    rows = meta['sequences']
    assert all(r['split'] == 'probe' and r['person'] in [8, 9] for r in rows)
    for key in ['clip_id', 'person', 'source_sha256']:
        assert not ({r[key] for r in rows} & {r[key] for r in fit_meta['sequences']})
    data = []
    for r in rows:
        assert sha(r['path']) == r['sha256']
        with np.load(r['path'], allow_pickle=False) as z:
            d = {k: z[k].copy() for k in ['x', 'y', 'mask', 'observed', 'weights', 'label_provenance', 'source_frame_id']}
        assert d['x'].shape == (r['length'], 150)
        assert all(d[k].shape == (r['length'], 5) for k in ['y', 'mask', 'observed', 'weights', 'label_provenance'])
        assert np.isfinite(d['x']).all() and np.isfinite(d['weights']).all()
        assert not ((d['mask'] > 0) & ~d['observed']).any()
        data.append(d)
    flat = {k: np.concatenate([d[k] for d in data]) for k in ['y', 'mask', 'weights', 'label_provenance']}
    assert np.isclose(flat['weights'].sum(), 1.)
    models = []
    for seed in SEEDS:
        run = next(r for r in training['rows'] if r['seed'] == seed)
        parent = next(r for r in fit_meta['parent_models'] if r['seed'] == seed)
        assert parent['condition'] == 'boundary_extended'
        assert run['initial_model_sha256'] == parent['model_sha256']
        assert run['initial_model'] == parent['model'] and run['normalization_unchanged'] is True
        assert sha(run['model']) == run['model_sha256']
        assert sha(parent['predictions']) == parent['predictions_sha256']
        models.append((seed, run, parent))
    # Import torch and construct only the new head after all admission/pin checks.
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model_module = load_module('_bp_fixed_probe_head', BP / 'model.py')
    assert model_module.FAMILIES == F
    write(U / 'EVALUATION_STARTED.json', dict(at=stamp(), contract_sha256=sha(U / 'EVALUATION_CONTRACT.json'),
          root_review_sha256=sha(U / 'ROOT_REVIEW.json'), models=[dict(seed=s, model=r['model'], model_sha256=r['model_sha256'],
          parent_predictions=p['predictions'], parent_predictions_sha256=p['predictions_sha256']) for s, r, p in models]))
    outputs, failures = [], []
    for seed, run, parent in models:
        stop()
        state = torch.load(run['model'], map_location='cpu', weights_only=True)
        assert set(state) == {'mean', 'std', 'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias'}
        assert all(torch.isfinite(v).all() for v in state.values()) and (state['std'] > 0).all()
        net = model_module.BodyProgressHead(state['mean'], state['std'])
        net.load_state_dict(state)
        net.eval()
        old_all, new_all, predictions, sequence_rows = [], [], {}, []
        forward_calls = 0
        with np.load(parent['predictions'], allow_pickle=False) as stored:
            assert {f'logits_{i}' for i in range(192)} <= set(stored.files)
            for i, (r, d) in enumerate(zip(rows, data)):
                stop()
                old = stored[f"logits_{r['original_sequence_index']}"][:, :, 2].copy()
                assert old.shape == (r['length'], 5) and np.isfinite(old).all()
                with torch.inference_mode():
                    new = net(torch.from_numpy(d['x'])).numpy()
                forward_calls += 1
                assert new.shape == old.shape and np.isfinite(new).all()
                old_all.append(old)
                new_all.append(new)
                predictions[f'logits_{i}'] = new
                sequence_rows.append(dict(input_index=i, original_sequence_index=r['original_sequence_index'],
                     clip_id=r['clip_id'], person=r['person'], view=r['view'], reverse=r['reverse'],
                     by_provenance={NAMES[c]: summarize(old, new, d['y'], d['label_provenance'] == c) for c in NAMES}))
                oc, nc = (old >= 0) == (d['y'] > 0), (new >= 0) == (d['y'] > 0)
                for t, j in np.argwhere((d['mask'] > 0) & ~nc):
                    failures.append(dict(seed=seed, clip_id=r['clip_id'], view=r['view'], reverse=r['reverse'],
                        source_frame_id=int(d['source_frame_id'][t]), query_frame_id=int(t), family=F[j],
                        label_code=int(d['label_provenance'][t, j]), kind='regression' if oc[t, j] else 'persisting_failure',
                        old_logit=float(old[t, j]), new_logit=float(new[t, j])))
        assert forward_calls == 48
        dst = U / 'runs' / str(seed)
        dst.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(dst / 'PROBE_PREDICTIONS.npz', **predictions)
        write(dst / 'PREDICTION_INDEX.json', dict(input_sha256=sha(U / 'INPUTS_COMPLETE.json'), rows=sequence_rows))
        old, new, y = np.concatenate(old_all), np.concatenate(new_all), flat['y']
        by_group = {}
        for grouping in ['clip_id', 'person', 'view', 'reverse']:
            by_group[grouping] = []
            for group in sorted({r[grouping] for r in rows}):
                ix = [i for i, r in enumerate(rows) if r[grouping] == group]
                a, b = np.concatenate([old_all[i] for i in ix]), np.concatenate([new_all[i] for i in ix])
                yy = np.concatenate([data[i]['y'] for i in ix])
                provenance = np.concatenate([data[i]['label_provenance'] for i in ix])
                by_group[grouping].append(dict(value=group, by_provenance={NAMES[c]: summarize(a, b, yy, provenance == c) for c in NAMES}))
        out = dict(seed=seed, model_sha256=run['model_sha256'], predictions=str(dst / 'PROBE_PREDICTIONS.npz'),
                   predictions_sha256=sha(dst / 'PROBE_PREDICTIONS.npz'), prediction_index=str(dst / 'PREDICTION_INDEX.json'),
                   prediction_index_sha256=sha(dst / 'PREDICTION_INDEX.json'), new_forward_calls=forward_calls, parent_forward_calls=0,
                   by_provenance={NAMES[c]: summarize(old, new, y, flat['label_provenance'] == c) for c in NAMES},
                   by_family_class=[dict(family=f, label=v, **summarize(old[:, j], new[:, j], y[:, j], (flat['mask'][:, j] > 0) & (y[:, j] == v))) for j, f in enumerate(F) for v in [0, 1]],
                   equal_scope_weighted=weighted_summary(old, new, y, flat['weights']), by_group=by_group)
        write(dst / 'RESULT.json', out)
        outputs.append(out)
        print(json.dumps(dict(seed=seed, by_provenance=out['by_provenance'], equal_scope_weighted=out['equal_scope_weighted'])), flush=True)
    check(cfg['pins'])
    check(cfg['protected_pins'])
    stop()
    write(U / 'RESULTS.json', dict(at=stamp(), status='FIXED_EXISTING_PROBE_EVALUATED_NOT_ADOPTED', runs=outputs,
          failures=failures, counts=meta['counts'], new_forward_calls=144, parent_forward_calls=0, encoder_forward_calls=0,
          root_review_sha256=sha(U / 'ROOT_REVIEW.json'), contract_sha256=sha(U / 'EVALUATION_CONTRACT.json'),
          protected_hashes_verified=477, no_training=True, no_model_selection=True, no_changed_candidate_interval=True,
          semantic_completion=None, whole_sign_correctness=None, web_changed=False,
          limitations=read(U / 'RGB_REVIEW.json')['limitations']))


if __name__ == '__main__':
    try:
        assert len(sys.argv) == 2 and sys.argv[1] == 'run-reviewed'
        main()
    except BaseException:
        (U / f'FIRST_EVALUATION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(), 'utf-8')
        raise
