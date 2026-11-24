"""Mocked driver lifecycle checks: no real training, scoring, or model load."""
import contextlib
import itertools
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import run_comparison as driver


def main():
    rb = driver.rb
    unit_root = Path(__file__).resolve().parent
    fixtures = unit_root / 'connection_smoke_fixtures_v2'
    fixtures.mkdir(exist_ok=False)
    checks = []
    sequence = itertools.count()
    for case in ('complete', 'started_only'):
        unit, run = fixtures / case / 'unit', fixtures / case / 'run'
        prep = unit / 'prepared'
        prep.mkdir(parents=True)
        (prep / 'TRAINING_META.json').write_text('{}', encoding='utf-8')
        ctx = {'new_init_digest': 'a' * 64}
        counts = {'train': 0, 'eval': 0}
        model = types.SimpleNamespace(state_dict=lambda: {})

        def train(*args, **kwargs):
            assert isinstance(args[6], str), 'workflow output directory must serialize as str'
            assert args[7] == ctx
            counts['train'] += 1
            run.mkdir()
            if case == 'started_only':
                (run / 'EVALUATION_STARTED.json').write_text('{}')
            return {'output_dir': args[6], 'selected_epoch': 1, 'nested': ['ordinary JSON', 1]}

        def evaluate(*args, **kwargs):
            counts['eval'] += 1
            scores = run / 'EVALUATION_SCORES.npz'
            scores.write_bytes(b'synthetic fixture only')
            rb.write_json(run / 'EVALUATION_EXECUTION.json', {
                'training_completion_sha256': rb.sha(run / 'TRAINING_COMPLETION.json'),
                'scores_sha256': rb.sha(scores), 'context_sha256': rb.digest(ctx),
                'final_model_sha256': 'b' * 64})
            return types.SimpleNamespace(returncode=0)

        bindings = [
            (rb, 'U', unit), (rb, 'RUN', run), (rb, 'PREP', prep),
            # Separate real CLI runs have different PIDs/timestamps. Model that identity here.
            (rb, 'stamp', lambda: 'fixture_%03d' % next(sequence)),
            (rb, 'setup', lambda: ({'refit_num_samples': 2352}, {}, None, lambda: model, None, None)),
            (rb, 'context', lambda *a: ctx), (rb, 'datasets', lambda *a: ({}, {}, {})),
            (rb.gw, 'state_digest', lambda *a: 'a' * 64),
            (rb.gw, 'validate_frozen', lambda *a: {'final_model_sha256': 'b' * 64}),
            (rb.gw, 'run_experiment', train),
            (rb, 'trace_pass', lambda *a, **k: {'status': 'PASS', 'finalized': True}),
            (driver.subprocess, 'run', evaluate)]
        with contextlib.ExitStack() as stack:
            for obj, name, value in bindings:
                stack.enter_context(patch.object(obj, name, value))

            def invoke(resume=False):
                with patch.object(sys, 'argv', ['run_comparison.py'] + (['--resume'] if resume else [])):
                    driver.main()

            if case == 'complete':
                invoke()
                invoke(True)
                assert counts == {'train': 1, 'eval': 1}
                completion = json.loads((run / 'TRAINING_COMPLETION.json').read_text())
                assert isinstance(completion['result']['output_dir'], str)
                (run / 'EVALUATION_SCORES.npz').write_bytes(b'intentionally altered fixture')
                try:
                    invoke(True)
                except SystemExit as exc:
                    assert 'existing scores lack' in str(exc)
                else:
                    raise AssertionError('tampered scores accepted')
                assert counts == {'train': 1, 'eval': 1}
                checks.extend(['completion serializes paths', 'resume skips completed training/evaluation',
                               'altered existing score rejected'])
            else:
                try:
                    invoke()
                except SystemExit as exc:
                    assert 'started but not completed' in str(exc)
                else:
                    raise AssertionError('started evaluation automatically repeated')
                assert counts == {'train': 1, 'eval': 0}
                checks.append('started-only evaluation refuses rescore')
    rb.write_json(unit_root / 'CONNECTION_SMOKE_CHECK.json', {
        'status': 'PASS', 'checks': checks,
        'scope': 'mocked orchestration only; no real models/training/evaluation',
        'driver_sha256': rb.sha(unit_root / 'run_comparison.py'),
        'test_sha256': rb.sha(__file__), 'fixtures': str(fixtures), 'model_training_steps': 0})
    print('PASS:', ', '.join(checks))


if __name__ == '__main__':
    main()
