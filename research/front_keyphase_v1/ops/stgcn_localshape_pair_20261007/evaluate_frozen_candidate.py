# Score the frozen global49+localXY candidate once on prepared 855-row inputs and pack stored comparator scores.
import json
import os

import numpy as np
import torch

import runtime_binding as rb

rb.configure_cpu()
METHOD = rb.Path(rb.P['method_run'])
REFERENCE = rb.Path(rb.P['reference_run'])
OUTPUTS = ('EVALUATION_STARTED.json', 'EVALUATION_SCORES.npz', 'EVALUATION_EXECUTION.json',
           'EVALUATION_ROWS.json', 'EVALUATION_INPUT_DIAGNOSTICS.json')


def main():
    existing = [n for n in OUTPUTS if (rb.RUN / n).exists()]
    if existing:
        raise SystemExit('evaluation artifacts already exist (%s); not repeating' % ', '.join(existing))
    plan, _cfg, _src_meta, factory4, _augment, _tf = rb.setup()
    ctx = rb.context(plan, rb.read(rb.PREP / 'TRAINING_META.json'))
    completion_path = rb.RUN / 'TRAINING_COMPLETION.json'
    if not completion_path.exists():
        raise SystemExit('TRAINING_COMPLETION.json is required')
    completion = rb.read(completion_path)
    if completion['context'] != ctx or completion['context_sha256'] != rb.digest(ctx):
        raise SystemExit('pinned context files changed since training completed')
    trace = rb.trace_pass(execute=False)
    frozen = rb.gw.validate_frozen(rb.RUN, ctx)
    if frozen['final_model_sha256'] != completion['final_model_sha256']:
        raise SystemExit('frozen model differs from TRAINING_COMPLETION')

    rows_doc = rb.read(rb.PREP / 'EVALUATION_ROWS.json')
    ids, n, c = [r['id'] for r in rows_doc['rows']], len(rows_doc['rows']), len(ctx['classes'])
    prep = rb.load_npz(rb.PREP / 'evaluation.npz')
    x, valid = torch.from_numpy(prep['x']), torch.from_numpy(prep['window_valid'])
    if (n != 855 or c != 93 or x.dtype != torch.float32 or valid.dtype != torch.bool
            or tuple(x.shape) != (n,) + rb.INFERENCE_SHAPE or tuple(valid.shape) != (n, 6)
            or rows_doc['classes'] != ctx['classes']
            or not bool(torch.isfinite(x).all()) or not bool(valid.any(dim=1).all())):
        raise SystemExit('prepared evaluation input does not match rows/classes')
    if prep['y'].tolist() != [int(r['y']) for r in rows_doc['rows']]:
        raise SystemExit('prepared evaluation labels differ from rows')
    stored = {}
    for name, run in (('method', METHOD), ('reference', REFERENCE)):
        doc, arrays = rb.read(run / 'EVALUATION_ROWS.json'), rb.load_npz(run / 'EVALUATION_SCORES.npz')
        if doc != rows_doc:
            raise SystemExit('%s run rows/classes differ from prepared evaluation rows' % name)
        if arrays['y'].tolist() != prep['y'].tolist():
            raise SystemExit('%s run labels differ' % name)
        if any(arrays[k].shape != (n, c) or not np.isfinite(arrays[k]).all()
               for k in ('candidate_scores', 'baseline_scores')):
            raise SystemExit('%s run score shape mismatch' % name)
        stored[name] = arrays
    method, reference = stored['method'], stored['reference']
    if not np.array_equal(method['baseline_scores'], reference['baseline_scores']):
        raise SystemExit('Stored TCN baseline scores disagree across comparator runs')

    model = rb.gw.load_final_state(factory4(), rb.RUN, ctx)
    model.eval()
    before = rb.gw.state_digest(model.state_dict())
    started = rb.now()
    frozen_sha = rb.sha(rb.RUN / 'FROZEN_MODEL.json')
    input_sha = rb.gp.tensor_sha256(x)
    valid_sha = rb.gp.tensor_sha256(valid)
    rb.write_json(rb.RUN / 'EVALUATION_STARTED.json',
                  {'started_utc': started, 'pid': os.getpid(), 'context_sha256': rb.digest(ctx),
                   'frozen_model_sha256': frozen_sha,
                   'final_model_sha256': frozen['final_model_sha256'],
                   'prepared_evaluation_sha256': rb.sha(rb.PREP / 'evaluation.npz'),
                   'input_sha256': input_sha, 'window_valid_sha256': valid_sha})
    with torch.no_grad():
        candidate = np.ascontiguousarray(rb.gw.score_batches(model, x, 16, window_valid=valid))
    if rb.gw.state_digest(model.state_dict()) != before:
        raise SystemExit('model state changed during scoring; stop for review')
    if candidate.shape != (n, c) or not np.isfinite(candidate).all():
        raise SystemExit('candidate scores have unexpected shape or non-finite values; stop for review')

    rb.write_npz(rb.RUN / 'EVALUATION_SCORES.npz', candidate_scores=candidate,
                 baseline_scores=method['baseline_scores'], old_stgcn_scores=method['candidate_scores'],
                 old_reference_scores=reference['candidate_scores'], y=method['y'])
    for name in ('EVALUATION_ROWS.json', 'EVALUATION_INPUT_DIAGNOSTICS.json'):
        rb.write_bytes(rb.RUN / name, (rb.PREP / name).read_bytes())
    comparators = (METHOD / 'EVALUATION_SCORES.npz', METHOD / 'EVALUATION_ROWS.json',
                   REFERENCE / 'EVALUATION_SCORES.npz', REFERENCE / 'EVALUATION_ROWS.json')
    record = {
        'started_utc': started, 'finished_utc': rb.now(), 'pid': os.getpid(), 'rows': n, 'batch_size': 16,
        'context_sha256': rb.digest(ctx), 'training_completion_sha256': rb.sha(completion_path), 'trace': trace,
        'input_sha256': input_sha, 'window_valid_sha256': valid_sha,
        'prepared_evaluation_sha256': rb.sha(rb.PREP / 'evaluation.npz'),
        'final_model_sha256': frozen['final_model_sha256'], 'model_state_digest': before,
        'frozen_model_sha256': frozen_sha, 'model_state_unchanged': True,
        'no_training_or_selection': True, 'user_data_used': False, 'comparator_inferences': 0,
        'candidate_scores_sha256': rb.gp.tensor_sha256(torch.from_numpy(candidate)),
        'scores_sha256': rb.sha(rb.RUN / 'EVALUATION_SCORES.npz'),
        'comparator_sha256': {str(p): rb.sha(p) for p in comparators}, 'comparator_hashed_utc': rb.now(),
        'baseline_scores_equal_across_runs': bool(np.array_equal(method['baseline_scores'], reference['baseline_scores'])),
    }
    rb.write_json(rb.RUN / 'EVALUATION_EXECUTION.json', record)
    print(json.dumps({k: record[k] for k in ('rows', 'final_model_sha256', 'scores_sha256', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
