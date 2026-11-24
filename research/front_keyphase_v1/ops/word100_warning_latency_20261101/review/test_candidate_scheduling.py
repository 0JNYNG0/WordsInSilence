"""Independent queue mocks only. No video, detector, GPU, head or DP execution."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import threading
import time
import traceback
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parents[1]
B = U.parent / 'word100_multipeople_live_feedback_20261101'
EXPECTED = 'c5891dd0dd19a02724b7084ee7b3015337568d90ccb2d2ee17d3581cccbe4098'
results = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(ok, name):
    results.append(dict(check=name, pass_=bool(ok)))
    if not ok:
        raise AssertionError(name)


def no_workers():
    return not [t for t in threading.enumerate() if t.name == 'candidate-signrep-single-consumer']


def run():
    check(sha(U / 'candidate_adapter.py') == EXPECTED, 'final reviewed candidate pin')
    spec = importlib.util.spec_from_file_location('_independent_candidate_mock', U / 'candidate_adapter.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    base_nodes = {n.name: n for n in ast.parse((B / 'adapter.py').read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef)}
    new_nodes = {n.name: n for n in ast.parse((U / 'candidate_adapter.py').read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef)}
    same = [k for k in base_nodes if k in new_nodes and ast.dump(base_nodes[k], include_attributes=False) == ast.dump(new_nodes[k], include_attributes=False)]
    check(len(same) == 18, '18 original functions unchanged including geometry/head/model/normalization')

    class SharedBuffer:
        def __init__(self, pause=0.):
            self.value = np.zeros((1, 768), np.float32)
            self.calls = []
            self.threads = []
            self.pause = pause
        def __call__(self, inputs):
            if self.pause:
                time.sleep(self.pause)
            v = float(inputs[0][0, 0, 0, 0, 0])
            self.calls.append(v)
            self.threads.append(threading.get_ident())
            self.value.fill(v)
            return [self.value]

    def items(count):
        reused = np.zeros((1, 3, 16, 224, 224), np.float32)
        for i in range(count):
            reused.fill(i + 1)
            yield i * 2, reused
            reused.fill(-999.)

    check(no_workers(), 'no starting worker residue')
    fake = SharedBuffer(.002)
    profile = {}
    output, starts = candidate._window_pipeline(items(8), fake, total_windows=8, profile=profile)
    check(starts.tolist() == list(range(0, 16, 2)), 'full exact start order')
    check(fake.calls == list(range(1, 9)), 'producer-owned buffer copy prevents mutation leakage')
    check(np.array_equal(output, np.repeat(np.arange(1, 9, dtype=np.float32)[:, None], 768, axis=1)),
          'every output survives subsequent shared compiled-buffer overwrite')
    fake.value.fill(-111.)
    check(float(output[-1, -1]) == 8., 'returned outputs own memory')
    check(len(set(fake.threads)) == 1 and fake.threads[0] != threading.get_ident(), 'one sequential worker owns all calls')
    check(profile['maximum_queue_depth'] <= 4 and profile['windows'] == 8, 'queue bound and full output count')
    check(no_workers(), 'successful return joins worker')

    class ProgressFailure(RuntimeError):
        pass
    events = []
    def broken_progress(event):
        events.append(event)
        raise ProgressFailure('independent progress callback failure')
    fake = SharedBuffer(.004)
    failed = False
    try:
        candidate._window_pipeline(items(22), fake, broken_progress, total_windows=22)
    except ProgressFailure as exc:
        failed = str(exc) == 'independent progress callback failure'
    check(failed, 'progress exception propagated without partial success')
    check(len(events) == 1 and events[0]['windows'] == 20, 'failure occurs at the registered prepared-window notification')
    check(no_workers(), 'progress failure joins current synchronous consumer')
    ended = len(fake.calls)
    time.sleep(.02)
    check(len(fake.calls) == ended and ended <= 20, 'no continued device-like calls after failed return')
    check(sha(U / 'candidate_adapter.py') == EXPECTED, 'candidate source unchanged after mocks')
    return dict(status='PASS_INDEPENDENT_SCHEDULING_MOCKS', checks=len(results), failure_count=0,
        tests=results, unchanged_functions=same,
        source_pins={str(U / 'candidate_adapter.py'): EXPECTED, str(B / 'adapter.py'): sha(B / 'adapter.py')},
        execution=dict(public_video_decode=0, model_load=0, real_GPU=0, head_forward=0, DP=0,
            fake_compiled_pipeline_runs=2),
        limitations=['Mechanics only: speedup and exact real encoder outputs remain pending root execution.',
                    'In-flight synchronous work is joined, not forcibly interrupted. A device call that itself hangs is outside this scheduling-only cancellation guarantee.'])


if __name__ == '__main__':
    dest = U / 'review/CANDIDATE_SCHEDULING_VERIFICATION.json'
    if dest.exists():
        raise FileExistsError('First independent result must remain')
    try:
        result = run()
    except BaseException:
        result = dict(status='FAIL_PRESERVED', tests=results, traceback=traceback.format_exc())
    result['verifier_sha256'] = sha(__file__)
    with dest.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result.get(k) for k in ['status', 'checks', 'failure_count']}))
    raise SystemExit(0 if result['status'].startswith('PASS') else 1)
