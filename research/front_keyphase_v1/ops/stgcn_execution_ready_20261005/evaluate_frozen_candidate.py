"""Codex evaluation/verification entry point using the verified Claude APIs.

Runs only after the selection and fresh refit have completed and the final
checkpoint is frozen. No learning, selection, changes to the model or camera.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import time
import numpy as np
import torch

U = Path(__file__).resolve().parent
W = U.parent.parent
RUN = W / 'runs/stgcn_full_path_v1'
sys.path.insert(0, str(U.parent / 'stgcn_reference_alignment_20261005'))
sys.path.insert(0, str(U.parent / 'stgcn_transfer_compare_20261005'))
import reference_adapter as ra
from stgcn_core import read_encoder_tensors
import generic_pose_io as gp
import generic_workflow as gw

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
assert (U / 'TRAINING_COMPLETION.json').is_file(), 'Training not complete'
assert (RUN / 'REFIT_RESULT.json').is_file(), 'Refit not complete'
trace = json.loads((U / 'TRAINING_TRACE_AUDIT.json').read_text())
assert trace['status'] == 'PASS' and trace['finalized'], 'Run check_training_trace.py after refit before scoring'
assert not (RUN / 'EVALUATION_SCORES.npz').exists(), 'Final scores already exist; do not repeat'
context = json.loads((U / 'EXECUTION_CONTEXT.json').read_text(encoding='utf-8'))
for path, expected in context['library_pins'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
frozen = gw.validate_frozen(str(RUN), context)
frozen_sha = hashlib.sha256((RUN / 'FROZEN_MODEL.json').read_bytes()).hexdigest()
plan = json.loads((U / 'FROZEN_EXPERIMENT.json').read_text())
manifest_path = U.parent / 'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == plan['pins']['input_manifest']
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
assert manifest['classes'] == context['classes']
groups = {'annotation_dev':699, 'whole_dev':74, 'public':32, 'mp':14, 'ksl36':36}
rows = [r for r in manifest['rows'] if r['group'] in groups]
assert len(rows) == 855
assert {g:sum(r['group'] == g for r in rows) for g in groups} == groups
assert all(r['role'] != 'train' and '/runs/ses_' not in r['path'].replace('\\','/') for r in rows)
baseline_path = W / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == plan['pins']['baseline832']
baseline = np.asarray([r['baseline_scores'] for r in rows], dtype=np.float64)
assert baseline.shape == (855,93) and np.isfinite(baseline).all()
labels = np.asarray([r['y'] for r in rows], dtype=np.int64)
assert all(manifest['classes'][int(y)] == r['label'] for r,y in zip(rows,labels))

artifact = U.parent / 'openhands_artifact_20261005'
assert hashlib.sha256((artifact / 'TENSOR_METADATA.json').read_bytes()).hexdigest() == plan['pins']['pretrained_metadata']
tensors, _ = read_encoder_tensors(artifact / 'raw_dpc.zip', artifact / 'TENSOR_METADATA.json', plan['pins']['pretrained_zip'], encoder_prefix='model.')
encoder = {k[len('conv_encoder.'):]:v for k,v in tensors.items() if k.startswith('conv_encoder.')}
gru = {k[len('agg.'):]:v for k,v in tensors.items() if k.startswith('agg.')}
dpc, tf, _ = ra.load_reference(U.parent / 'stgcn_reference_alignment_20261005/reference', plan['pins']['source_pins'])
model, _ = ra.make_reference_model(dpc, encoder, gru, num_classes=93, seed=plan['training_config']['seed'])
model = gw.load_final_state(model, str(RUN), context)
state_before = gw.state_digest(model.state_dict())
specs = json.loads((U / 'prepared/TRAINING_META.json').read_text(encoding='utf-8'))['layout_specs']
normalize = tf.CenterAndScaleNormalize(reference_points_preset=ra.NORMALIZE_PRESET, frame_level=False)
prepare = lambda xy,mask,layout: gp.prepare_windows(xy,mask,ra.LAYOUTS[layout][1],[3,4],60,10,lambda t:normalize({'frames':t})['frames'])
started = time.perf_counter()
loaded = [gp.load_row(row, specs, prepare) for row in rows]
x = torch.stack([sample for sample,_ in loaded])
diagnostics = [dict(id=r['id'], group=r['group'], condition=r['condition'], **diag) for r,(_,diag) in zip(rows,loaded)]
assert tuple(x.shape) == (855,6,2,10,27) and bool(torch.isfinite(x).all())
scores = gw.score_batches(model, x, 16)
assert scores.shape == (855,93) and np.isfinite(scores).all()
assert state_before == gw.state_digest(model.state_dict()), 'Inference changed model state'
assert frozen_sha == hashlib.sha256((RUN / 'FROZEN_MODEL.json').read_bytes()).hexdigest()
np.savez(RUN / 'EVALUATION_SCORES.npz', candidate_scores=scores, baseline_scores=baseline, y=labels)
(RUN / 'EVALUATION_ROWS.json').write_text(json.dumps({'classes':manifest['classes'], 'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
(RUN / 'EVALUATION_INPUT_DIAGNOSTICS.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
record = dict(status='SCORED_NOT_YET_INDEPENDENTLY_AUDITED', at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              final_model_sha256=frozen['final_model_sha256'], frozen_model_sha256=frozen_sha,
              input_sha256=gp.tensor_sha256(x), classes=93, rows=855,
              score_sha256=hashlib.sha256((RUN/'EVALUATION_SCORES.npz').read_bytes()).hexdigest(),
              baseline_model_sha256=plan['pins']['baseline832'],
              model_state_unchanged=True, seconds=time.perf_counter()-started,
              no_training_or_selection=True, user_data_used=False)
(RUN / 'EVALUATION_EXECUTION.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
