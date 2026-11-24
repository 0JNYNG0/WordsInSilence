"""Focused independent integrity checks; synthetic bytes only, no adapter import."""
import argparse, copy, hashlib, json, sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / 'generated_v2'))
import frozen_replay as driver

original = json.loads((here / 'fixtures_v2/normal/out/manifest_LOCAL_ONLY.json').read_text())
case = here / 'integrity_fixtures'
case.mkdir(exist_ok=False)
toy = case / 'toy.txt'
toy.write_text('synthetic bytes', encoding='utf-8')
checks = []
def reject(name, fn, expected):
    try:
        fn()
    except (RuntimeError, ValueError) as exc:
        checks.append({'name': name, 'passed': expected in str(exc), 'diagnostic': str(exc)})
    else:
        checks.append({'name': name, 'passed': False})

m = copy.deepcopy(original)
m['hashes'] = {str(toy): '0' * 64}
reject('changed pinned bytes rejected', lambda: driver.reverify(m), 'changed since prepare')
m = copy.deepcopy(original)
m['absent_paths'].append(str(toy))
reject('previously missing artifact appearing rejected', lambda: driver.reverify(m), 'absent file appeared')
m = copy.deepcopy(original)
m['protected_sha256'] = {str(toy): {'present': True, 'expected': '0' * 64}}
reject('protected bytes changed rejected', lambda: driver.reverify(m), 'protected file changed')
m = copy.deepcopy(original)
m['roots']['out_dir'] = str(case / 'redirected')
manifest = case / 'redirected_manifest.json'
manifest.write_text(json.dumps(m), encoding='utf-8')
reject('manifest cannot redirect output', lambda: driver.cmd_execute(argparse.Namespace(manifest=str(manifest))), "manifest's own directory")
raw = {'checkpoint_sha256': '0' * 64, 'accepted': None, 'execution_assessment': 'UNCONFIGURED'}
reject('inference wrong checkpoint rejected', lambda: driver.normalize('parent', raw, '1' * 64), 'checkpoint sha mismatch')
raw.update(checkpoint_sha256='1' * 64, accepted=True)
reject('classification cannot become execution acceptance', lambda: driver.normalize('parent', raw, '1' * 64), 'accepted must be None')
report = {'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL', 'passed': sum(c['passed'] for c in checks), 'total': len(checks), 'checks': checks, 'real_model_loads': 0, 'human_records_used': 0, 'camera_calls': 0}
(here / 'verification_integrity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('status','passed','total')}))
sys.exit(0 if report['status'] == 'PASS' else 1)
