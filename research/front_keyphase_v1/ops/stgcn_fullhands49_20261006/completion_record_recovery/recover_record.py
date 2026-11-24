"""Recover only the missing completion JSON after validated, finished training.

No model creation, training, forward pass, or pinned-file modification. The
original workflow returns a Path in result.output_dir; convert that one field
to its identical string representation for JSON publication.
"""
from pathlib import Path
import json
import sys

U = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(U))
import runtime_binding as rb


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    completion = rb.RUN / 'TRAINING_COMPLETION.json'
    need(not completion.exists(), 'Completion already exists; do not overwrite')
    need(not (rb.RUN / 'EVALUATION_STARTED.json').exists(), 'Evaluation already started')
    need(not (rb.RUN / 'EVALUATION_SCORES.npz').exists(), 'Scores already exist')
    ctx = rb.read(U / 'EXECUTION_CONTEXT.json')
    ctx_sha = rb.digest(ctx)
    for name, expected in ctx['library_pins'].items():
        need(rb.sha(name) == expected, f'Changed pinned file: {name}')
    trace = rb.read(U / 'TRAINING_TRACE_AUDIT.json')
    need(trace['status'] == 'PASS' and trace['finalized'] is True,
         'Require finalized independent training audit')
    need(trace['context_sha256'] == ctx_sha, 'Audit context differs')
    frozen = rb.gw.validate_frozen(rb.RUN, ctx)
    binding = {'context_sha256': ctx_sha, 'run_sha256': frozen['run_sha256']}
    need(trace['run_sha256'] == binding['run_sha256'], 'Audit run differs')
    result = rb.gw._completed(rb.RUN, ctx, binding)
    need(isinstance(result['output_dir'], Path) and result['output_dir'] == rb.RUN,
         'Unexpected result.output_dir type or value')
    converted = dict(result, output_dir=str(result['output_dir']))
    # Reject any second non-JSON field rather than applying a broad default=str.
    json.dumps(converted, allow_nan=False)
    initial = rb.read(rb.RUN / 'INIT_SELECT.json')
    need(initial == rb.read(rb.RUN / 'INIT_REFIT.json'), 'Initial records differ')
    need(initial['state_digest'] == ctx['new_init_digest'], 'Initial digest differs')
    names = ('SELECT_RESULT.json', 'REFIT_RESULT.json', 'FROZEN_MODEL.json',
             'final_model.npz', 'refit_last.pt', 'INIT_SELECT.json', 'INIT_REFIT.json')
    before = {name: rb.sha(rb.RUN / name) for name in names}
    frozen_time = (rb.RUN / 'FROZEN_MODEL.json').stat().st_mtime
    record = {'completed_utc': rb.now(), 'context_sha256': ctx_sha, 'context': ctx,
              'initial_state_digest': initial['state_digest'],
              'final_model_sha256': frozen['final_model_sha256'], 'result': converted,
              'completion_record_recovery': {
                  'reason': 'Original completion writer rejected WindowsPath after final model was saved',
                  'converted_field': 'result.output_dir', 'training_repeated': False,
                  'original_frozen_file_mtime_unix': frozen_time,
                  'evidence_dir': str(Path(__file__).resolve().parent),
                  'script_sha256': rb.sha(__file__),
                  'finalized_trace_sha256': rb.sha(U / 'TRAINING_TRACE_AUDIT.json')}}
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode('utf-8')
    with completion.open('xb') as stream:
        stream.write(payload)
    after = {name: rb.sha(rb.RUN / name) for name in names}
    need(before == after, 'Training artifact changed during completion publication')
    for name, expected in ctx['library_pins'].items():
        need(rb.sha(name) == expected, f'Pinned file changed during publication: {name}')
    report = {'status': 'PASS', 'checked_utc': rb.now(), 'context_sha256': ctx_sha,
              'run_sha256': binding['run_sha256'], 'model_sha256': frozen['final_model_sha256'],
              'converted_field': 'result.output_dir: WindowsPath to same path string',
              'training_artifacts_before': before, 'training_artifacts_after': after,
              'completion_sha256': rb.sha(completion),
              'pinned_files_unchanged': True, 'training_or_model_inference_executed': False}
    with (Path(__file__).parent / 'RECOVERY_CHECK.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'PASS', 'model_sha256': frozen['final_model_sha256'],
                      'training_repeated': False, 'completion_sha256': rb.sha(completion)}))


if __name__ == '__main__':
    main()
