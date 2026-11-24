"""Independent synthetic lifecycle checks. Never reads actual research data."""
from pathlib import Path
import copy
import json
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
import generic_workflow as gw

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent
TAG = sys.argv[1] if len(sys.argv) > 1 else str(time.time_ns())
OUT = ROOT / ('synthetic_workflow_' + TAG)
OUT.mkdir(exist_ok=False)

def factory():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        return nn.Sequential(nn.Flatten(), nn.Linear(3240, 6), nn.BatchNorm1d(6), nn.Dropout(.2), nn.Linear(6, 3))

g = torch.Generator().manual_seed(1702)
all_data = dict(x=torch.randn(6, 6, 2, 10, 27, generator=g),
                y=torch.arange(6) % 3, weight=torch.tensor([1, 0, 2, 1, 0, 1], dtype=torch.float64),
                ids=[f'row{i}' for i in range(6)], origins=[f'original{i}' for i in range(6)])
train = {k: (v[:4].clone() if isinstance(v, torch.Tensor) else v[:4]) for k, v in all_data.items()}
val = {k: (v[4:].clone() if isinstance(v, torch.Tensor) else v[4:]) for k, v in all_data.items()}
val['primary'] = torch.tensor([True, False])
cfg = dict(schema='generic_cpu_v1', num_classes=3, batch_size=2, max_epochs=3,
           min_epochs=2, patience=2, base_lr=.001, schedule_epochs=3, seed=1703, num_samples=8)
refit_cfg = dict(cfg, num_samples=12)
context = dict(purpose='synthetic independent lifecycle verification', classes=['a','b','c'])
before = {k:v.clone() for k,v in all_data.items() if isinstance(v, torch.Tensor)}
result = {'tag': TAG, 'real_data_used': 0, 'checks': []}
try:
    r = gw.run_experiment(factory, train, val, all_data, cfg, refit_cfg, str(OUT/'run'), context)
    result['checks'].append('select_and_fresh_refit_completed')
    sj = json.loads((OUT/'run'/'SELECT_RESULT.json').read_text())
    rj = json.loads((OUT/'run'/'REFIT_RESULT.json').read_text())
    assert len(rj['history']) == sj['best_epoch']
    assert all(h['sample_counts'][1] == 0 for h in sj['history'])
    assert all(h['sample_counts'][1] == h['sample_counts'][4] == 0 for h in rj['history'])
    assert all(h['val']['primary_count'] == 1 and h['val']['all_count'] == 2 for h in sj['history'])
    result['checks'].append('zero_sampling_and_primary_validation')
    loaded = gw.load_final_state(factory(), str(OUT/'run'), context)
    scores = gw.score_batches(loaded, val['x'], 2)
    loaded2 = gw.load_final_state(factory(), str(OUT/'run'), context)
    assert np.array_equal(scores, gw.score_batches(loaded2, val['x'], 2))
    result['checks'].append('safe_final_state_exact_logits')
    r2 = gw.run_experiment(factory, train, val, all_data, cfg, refit_cfg, str(OUT/'run'), context, resume=True)
    assert r2['selected_epoch'] == sj['best_epoch']
    result['checks'].append('completed_resume_no_retraining')
    for k,v in before.items():
        assert torch.equal(v, all_data[k]), k
    result['checks'].append('source_tensors_unchanged')
    final = OUT/'run'/'final_model.npz'
    original = final.read_bytes()
    final.write_bytes(original + b'corruption')
    try:
        gw.validate_frozen(str(OUT/'run'), context)
    except ValueError:
        result['checks'].append('final_corruption_rejected')
    else:
        raise AssertionError('corruption not rejected')
    final.write_bytes(original)
    result['status'] = 'PASS'
except Exception:
    result['status'] = 'FAIL'
    result['traceback'] = traceback.format_exc()
(ROOT / ('WORKFLOW_CHECK_' + TAG + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
if result['status'] != 'PASS':
    sys.exit(1)
