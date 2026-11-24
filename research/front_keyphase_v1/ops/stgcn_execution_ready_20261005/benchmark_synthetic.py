"""Codex execution-only timing probe, no real clips or persisted model weights."""
from pathlib import Path
import json
import statistics
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'stgcn_reference_alignment_20261005'
sys.path.insert(0, str(BASE))
from reference_adapter import load_reference

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(20261005)
dpc, _, _ = load_reference(BASE / 'reference', 'e9b38fc92391a7e913b1c1c266b65701885bea112c4dc775a026de7b1443e304')
model = dpc.DPC_RNN_Finetuner(num_class=93).train()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.randn(16, 6, 2, 10, 27)
y = torch.arange(16, dtype=torch.long)
train_seconds = []
for i in range(8):
    start = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    elapsed = time.perf_counter() - start
    if i >= 2:
        train_seconds.append(elapsed)
model.eval()
eval_seconds = []
with torch.no_grad():
    for _ in range(4):
        start = time.perf_counter()
        assert torch.isfinite(model(x)).all()
        eval_seconds.append(time.perf_counter() - start)
train_med = statistics.median(train_seconds)
eval_med = statistics.median(eval_seconds)
result = {
    'purpose': 'resource estimate only; random initialized model and synthetic input',
    'torch_version': torch.__version__, 'device': 'cpu', 'threads': 2,
    'batch_size': 16, 'input_shape': list(x.shape), 'optimizer': 'Adam',
    'warmup_updates': 2, 'timed_updates': 6, 'train_seconds': train_seconds,
    'eval_seconds': eval_seconds, 'median_update_seconds': train_med,
    'median_eval_batch_seconds': eval_med,
    'rough_fit_epoch_seconds_131updates_17valbatches': 131 * train_med + 17 * eval_med,
    'rough_refit_epoch_seconds_147updates': 147 * train_med,
    'actual_training_data_used': 0, 'accuracy_evaluations': 0,
    'checkpoint_written': False,
    'limitation': 'includes gradient finite checks; excludes preprocessing, checkpoint IO; CPU load may vary',
}
(HERE / 'SYNTHETIC_TIMING.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
