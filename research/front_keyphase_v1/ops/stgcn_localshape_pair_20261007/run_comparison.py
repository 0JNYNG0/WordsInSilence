# Train the frozen global49+localXY candidate once through generic_workflow, then hand off to evaluation.
import argparse
import json
import os
import subprocess
import sys

import runtime_binding as rb

rb.configure_cpu()
MARKER = 'EVALUATED_PENDING_INDEPENDENT_FINAL_AUDIT.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    plan, cfg, _src_meta, factory4, augment, _tf = rb.setup()
    meta = rb.read(rb.PREP / 'TRAINING_META.json')
    ctx = rb.context(plan, meta)
    ctx_sha = rb.digest(ctx)
    if plan['refit_num_samples'] != 2352:
        raise SystemExit('unexpected refit_num_samples in frozen plan')
    if rb.RUN.exists() != args.resume:
        raise SystemExit('run_dir must be new, or already exist when --resume is given')
    stored = rb.U / 'EXECUTION_CONTEXT.json'
    if stored.exists():
        if rb.read(stored) != ctx:
            raise SystemExit('EXECUTION_CONTEXT differs from the current pinned context')
    elif args.resume:
        raise SystemExit('Cannot resume without the frozen EXECUTION_CONTEXT')
    else:
        rb.write_json(stored, ctx)
    initial = rb.gw.state_digest(factory4().state_dict())
    if initial != ctx['new_init_digest']:
        raise SystemExit('factory4 initial digest differs from INPUT_EXPANSION_CHECK')
    rb.write_json(rb.U / ('LIVE_EXECUTION_%s_%d.json' % (rb.stamp(), os.getpid())),
                  {'pid': os.getpid(), 'started_utc': rb.now(), 'run_dir': str(rb.RUN),
                   'resume': args.resume, 'context_sha256': ctx_sha})

    completion = rb.RUN / 'TRAINING_COMPLETION.json'
    if completion.exists():
        done = rb.read(completion)
        frozen = rb.gw.validate_frozen(rb.RUN, ctx)
        if done['context_sha256'] != ctx_sha or done['final_model_sha256'] != frozen['final_model_sha256']:
            raise SystemExit('TRAINING_COMPLETION does not match the current context/frozen model')
    else:
        train, val, all_data = rb.datasets(meta)
        refit_cfg = dict(cfg, num_samples=plan['refit_num_samples'])
        result = rb.gw.run_experiment(factory4, train, val, all_data, cfg, refit_cfg, str(rb.RUN), ctx,
                                      augment=augment, resume=args.resume)
        frozen = rb.gw.validate_frozen(rb.RUN, ctx)
        record = {'completed_utc': rb.now(), 'context_sha256': ctx_sha, 'context': ctx,
                  'initial_state_digest': initial,
                  'final_model_sha256': frozen['final_model_sha256'], 'result': result}
        json.dumps(record, allow_nan=False)  # Fail before creating completion if serialization is invalid.
        rb.write_json(completion, record)
    trace = rb.trace_pass()

    scores = rb.RUN / 'EVALUATION_SCORES.npz'
    if scores.exists():
        execution = rb.RUN / 'EVALUATION_EXECUTION.json'
        if (not execution.exists()
                or rb.read(execution)['training_completion_sha256'] != rb.sha(completion)
                or rb.read(execution)['scores_sha256'] != rb.sha(scores)
                or rb.read(execution)['context_sha256'] != ctx_sha
                or rb.read(execution)['final_model_sha256'] != frozen['final_model_sha256']):
            raise SystemExit('existing scores lack a matching completion record; stop for review')
        print('evaluation already recorded; not repeating')
    elif (rb.RUN / 'EVALUATION_STARTED.json').exists():
        raise SystemExit('evaluation was started but not completed; stop for review, do not rerun')
    else:
        proc = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(rb.U / 'evaluate_frozen_candidate.py')], cwd=str(rb.U))
        if proc.returncode != 0:
            raise SystemExit('evaluate_frozen_candidate.py failed (rc=%d); stop for review' % proc.returncode)

    marker = rb.RUN / MARKER
    record = {'scores_sha256': rb.sha(scores), 'training_completion_sha256': rb.sha(completion),
              'context_sha256': ctx_sha}
    if marker.exists():
        old = rb.read(marker)
        if any(old.get(k) != v for k, v in record.items()):
            raise SystemExit('existing audit marker disagrees with saved artifacts')
    else:
        rb.write_json(marker, dict(record, trace=trace, marked_utc=rb.now(),
                                   next='root independent final audit of saved metrics; no additional handshape scoring in this unit'))
    print('EVALUATED_PENDING_INDEPENDENT_FINAL_AUDIT')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
