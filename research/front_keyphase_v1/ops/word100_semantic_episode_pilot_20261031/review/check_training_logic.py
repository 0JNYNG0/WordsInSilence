"""Synthetic control review only: no project data, model imports, or model forward."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

U = Path(__file__).resolve().parents[1]
source = (U / 'episode.py').read_text(encoding='utf-8-sig')
tree = ast.parse(source)
names = {'support', 'encode', 'geometry', 'old_predict', 'automatic', 'runs', 'score'}
subset = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
env = {'np': np, 'COLS': [28, 29, 36, 37, 30, 31, 32, 33], 'SUPPORT': .75,
       'WINDOW': 13, 'CLASSES': ['up', 'down', 'other']}

def stub_infer(model, items):
    assert model is None
    return np.tile(np.array([[.9, .05, .05]]), (len(items), 1))

env['infer'] = stub_infer
exec(compile(subset, str(U / 'episode.py'), 'exec'), env)

def synthetic(n=13):
    return {'x': np.zeros((n, 69)), 'mask': np.ones((n, 69), bool),
            'frame_id': np.arange(n), 'timestamp_us': np.arange(n) * 33333}

checks = []
def check(name, condition, detail=None):
    checks.append({'name': name, 'pass': bool(condition), 'detail': detail})

a = synthetic()
a['mask'][6, 28:30] = False
p = np.zeros((13, 10)); p[:, 5] = .9
result = env['automatic'](a, None, p)
check('unobserved_center_unknown_all_three_methods',
      all(int(result[k][6]) == -1 for k in ('pred', 'geometry', 'old')),
      {k: int(result[k][6]) for k in ('pred', 'geometry', 'old')})

a = synthetic(); a['mask'][:] = False
result = env['automatic'](a, None, p)
check('all_missing_unknown_all_three_methods', all((result[k] == -1).all() for k in ('pred', 'geometry', 'old')))

a = synthetic(); a['x'][6, 29] = float('nan')
result = env['automatic'](a, None, p)
check('nonfinite_center_unknown_all_three_methods', all(int(result[k][6]) == -1 for k in ('pred', 'geometry', 'old')))

a = synthetic(3); a['frame_id'] = np.array([0, 1, 7]); a['timestamp_us'] = np.array([0, 33333, 233331])
a['x'][:, 28:38] = np.arange(3)[:, None]
z, _, _ = env['encode'](a)
check('delta_does_not_bridge_original_frame_gap', (z[2, 8:16] == 0).all() and (z[2, 24:32] == 0).all())
r = env['runs'](np.array([0, 0, 0]), a['frame_id'], a['timestamp_us'])
check('run_splits_original_frame_gap', len(r) == 2 and r[0]['end_frame'] == 1 and r[1]['start_frame'] == 7, r)
r = env['runs'](np.array([0, 0, 0]), np.arange(3), np.array([0, 33333, 300000]))
check('run_splits_large_pts_gap', len(r) == 2, r)
r = env['runs'](np.array([0, -1, 0]), np.arange(3), np.arange(3) * 33333)
check('unknown_run_not_filled', [v['state'] for v in r] == ['up', 'unknown', 'up'])
s = env['score']([0, 1, 2], [-1, -1, -1])
check('unknown_predictions_remain_denominator', s['n'] == 3 and s['unknown'] == 3 and s['correct'] == 0 and s['macro_f1'] == 0, s)

etree = ast.parse((U / 'experiment.py').read_text(encoding='utf-8-sig'))
train = next(n for n in etree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
text = ast.get_source_segment((U / 'experiment.py').read_text(encoding='utf-8-sig'), train)
check('fit_role_and_eligibility_filter_present', "e['role']=='fit' and e['eligible']" in text)
check('fixed_160_epoch_range_present', 'range(1,161)' in text)
calls = {n.func.id for n in ast.walk(train) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
check('no_development_metric_or_evaluation_in_training', not calls.intersection({'evaluate', 'automatic', 'score', 'old_predict', 'geometry'}), sorted(calls))
check('epoch_exposure_ids_logged', "event_ids=[e['event_id'] for e in fit]" in text)
check('class_weight_derived_from_fit_y', 'torch.bincount(y,minlength=3)' in text)

out = {'status': 'PASS' if all(c['pass'] for c in checks) else 'FAIL',
       'checks': checks, 'passed': sum(c['pass'] for c in checks), 'total': len(checks),
       'execution_scope': 'AST-extracted pure numpy functions, synthetic arrays, constant prediction stub; no torch/model forward, no project inputs or score outputs, no training',
       'source_sha256': {n: hashlib.sha256((U / n).read_bytes()).hexdigest() for n in ('episode.py', 'experiment.py')}}
path = U / 'review' / 'TRAINING_LOGIC_CHECKS.json'
with path.open('x', encoding='utf8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': out['status'], 'passed': out['passed'], 'total': out['total']}, ensure_ascii=False))
