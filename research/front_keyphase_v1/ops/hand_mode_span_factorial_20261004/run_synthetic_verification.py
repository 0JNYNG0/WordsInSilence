"""Codex execution receipt for Claude's synthetic-only tests; no model or device."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import sys
import unittest

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == '__main__':
    receipt_path = HERE / 'SYNTHETIC_CHECK.json'
    if receipt_path.exists():
        raise SystemExit('Existing receipt preserved; no repeat')
    sys.path.insert(0, str(HERE))
    sources = {n: sha(HERE / n) for n in (
        'span_factorial.py', 'test_span_factorial.py', 'API_REFERENCE.txt',
        'CONFIG_EXAMPLE.json', 'FROZEN_PLAN.md')}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        suite = unittest.defaultTestLoader.loadTestsFromName('test_span_factorial')
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    same = all(sha(HERE / n) == h for n, h in sources.items())
    forbidden = [n for n in sys.modules if n.split('.')[0] in ('mediapipe', 'av', 'torch') or 'camera' in n.lower()]
    ok = result.wasSuccessful() and result.testsRun > 0 and not result.skipped and same and not forbidden
    receipt = {
        'status': 'PASS' if ok else 'FAIL', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
        'runner_sha256': sources['span_factorial.py'],
        'test_sha256': sources['test_span_factorial.py'], 'source_hashes': sources,
        'sources_unchanged': same, 'forbidden_modules': forbidden,
        'model_forwards': 0, 'device_calls': 0, 'log': stream.getvalue()}
    with receipt_path.open('x', encoding='utf-8') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(stream.getvalue())
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('log', 'source_hashes')}, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)
