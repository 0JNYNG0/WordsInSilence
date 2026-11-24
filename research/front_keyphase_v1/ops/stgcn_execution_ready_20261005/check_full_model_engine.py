"""Independent synthetic integration check; no actual clips or saved weights."""
from pathlib import Path
import hashlib
import json
import sys
import torch

U = Path(__file__).resolve().parent
sys.path.insert(0, str(U.parent / 'stgcn_reference_alignment_20261005'))
sys.path.insert(0, str(U.parent / 'stgcn_transfer_compare_20261005'))
import reference_adapter as ra
from stgcn_core import read_encoder_tensors
import generic_pose_io as gp
from training_engine import run_training

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
plan = json.loads((U / 'FROZEN_EXPERIMENT.json').read_text())
artifact = U.parent / 'openhands_artifact_20261005'
assert hashlib.sha256((artifact / 'TENSOR_METADATA.json').read_bytes()).hexdigest() == plan['pins']['pretrained_metadata']
tensors, _ = read_encoder_tensors(artifact / 'raw_dpc.zip', artifact / 'TENSOR_METADATA.json', plan['pins']['pretrained_zip'], encoder_prefix='model.')
enc = {k[len('conv_encoder.'):]:v for k,v in tensors.items() if k.startswith('conv_encoder.')}
gru = {k[len('agg.'):]:v for k,v in tensors.items() if k.startswith('agg.')}
dpc, tf, _ = ra.load_reference(U.parent / 'stgcn_reference_alignment_20261005/reference', plan['pins']['source_pins'])
model, audit = ra.make_reference_model(dpc, enc, gru, num_classes=93, seed=20261005)
before = {k:v.clone() for k,v in model.state_dict().items()}
shear, rotate = tf.ShearTransform(.1), tf.RotatationTransform(.1)
augment = lambda x: gp.augment_windows(x, lambda t: rotate(shear({'frames':t}))['frames'])
gen = torch.Generator().manual_seed(20261005)
data = dict(x=torch.randn(16,6,2,10,27,generator=gen), y=torch.arange(16),
            weight=torch.ones(16,dtype=torch.float64), ids=[str(i) for i in range(16)],
            origins=[str(i) for i in range(16)])
cfg = dict(plan['training_config'], num_samples=16)
r = run_training(model, data, None, cfg, augment, mode='refit', fixed_epochs=1)
h = r['history'][0]
assert h['updates'] == 1 and h['grad_missing'] == 0 and h['grad_finite_checks'] == 108
after = model.state_dict()
bn = [k for k in before if k.endswith('num_batches_tracked')]
assert bn and all(int(after[k]) == int(before[k])+1 for k in bn)
graphs = [k for k in before if k.endswith('.A') or k == 'A']
assert graphs and all(torch.equal(before[k],after[k]) for k in graphs)
assert all(p.requires_grad for p in model.parameters())
assert all(torch.isfinite(t).all() for t in after.values())
assert any(not torch.equal(before[k],after[k]) for k in before if k.startswith('agg.') )
result = dict(status='PASS', real_data_used=0, optimizer_updates=1, weights_saved=False,
              trainable_gradient_tensors=h['grad_finite_checks'], missing_gradients=h['grad_missing'],
              bn_counters_advanced=len(bn), graph_buffers_unchanged=len(graphs), gru_updated=True,
              source_model_audit=audit)
(U/'FULL_MODEL_ENGINE_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='source_model_audit'},indent=2))
