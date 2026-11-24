"""Independent official-model integration check on synthetic tensors only.

One disposable optimizer update; no source clips, saved run inference, model
checkpoint writes, camera/device enumeration, or candidate-config changes.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import traceback

import numpy as np
import torch

U = Path(__file__).resolve().parent
W = U.parent.parent
OLD = U.parent / 'stgcn_execution_ready_20261005'
REFERENCE = U.parent / 'stgcn_reference_alignment_20261005'
CORE = U.parent / 'stgcn_transfer_compare_20261005'
sys.path.insert(0, str(REFERENCE))
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(U))

import reference_adapter as ra
from stgcn_core import read_encoder_tensors
import training_engine as engine
import generic_workflow as workflow
import generic_pose_io as pose_io
import window_bridge as bridge
import valid_window_forward as valid_forward


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset(data, indices):
    return {k: ([v[i] for i in indices] if isinstance(v, list) else v[indices].clone())
            for k, v in data.items()}


def check():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    modules = (engine, workflow, pose_io, bridge, valid_forward)
    for module in modules:
        need(Path(module.__file__).resolve().parent == U, f'Wrong imported unit: {module.__name__}')
    need(Path(ra.__file__).resolve() == REFERENCE / 'reference_adapter.py', 'Wrong reference adapter')
    pinned_files = [U / (m.__name__ + '.py') for m in modules]
    pinned_files += [U / 'CONFIG.json', U / 'FROZEN_EXPERIMENT.json', U / 'WINDOW_METADATA.npz',
                     U / 'METADATA_AUDIT.json', REFERENCE / 'reference_adapter.py', CORE / 'stgcn_core.py']
    before_hashes = {str(p): sha(p) for p in pinned_files}
    plan = json.loads((U / 'FROZEN_EXPERIMENT.json').read_text(encoding='utf-8'))
    applied = json.loads((U / 'CLAUDE_APPLIED.json').read_text(encoding='utf-8'))
    for name in ('training_engine.py', 'generic_workflow.py', 'window_bridge.py'):
        need(sha(U / name) == applied['files'][name], f'Applied-library pin mismatch: {name}')
    meta_audit = json.loads((U / 'METADATA_AUDIT.json').read_text(encoding='utf-8'))
    need(meta_audit['status'] == 'PASS', 'Metadata audit not passed')
    need(sha(U / 'WINDOW_METADATA.npz') == meta_audit['metadata_sha256'], 'Metadata cache changed')
    artifact = U.parent / 'openhands_artifact_20261005'
    need(sha(artifact / 'TENSOR_METADATA.json') == plan['pins']['pretrained_metadata'], 'Pretrained metadata pin')
    tensors, _ = read_encoder_tensors(artifact / 'raw_dpc.zip', artifact / 'TENSOR_METADATA.json',
                                      plan['pins']['pretrained_zip'], encoder_prefix='model.')
    enc = {k[len('conv_encoder.'):]: v for k, v in tensors.items() if k.startswith('conv_encoder.')}
    gru = {k[len('agg.'):]: v for k, v in tensors.items() if k.startswith('agg.')}
    need(len(enc) == 170 and len(gru) == 4, 'Pretrained-state counts')
    dpc, transforms, _ = ra.load_reference(REFERENCE / 'reference', plan['pins']['source_pins'])
    model, _ = ra.make_reference_model(dpc, enc, gru, num_classes=93, seed=plan['training_config']['seed'])
    original_keys = list(model.state_dict())
    initial = workflow.state_digest(model.state_dict())
    old_init = json.loads((W / 'runs/stgcn_full_path_v1/INIT_SELECT.json').read_text())
    need(initial == old_init['state_digest'] == '087479ad99b7a568158982fb55b9e0b14090e1d491424d0a929a022003e59a5d',
         'Fresh reference/head initialization differs from prior comparison')
    model = bridge.bind_window_forward(model, valid_forward.forward_valid_windows)
    need(list(model.state_dict()) == original_keys and workflow.state_digest(model.state_dict()) == initial,
         'Binding changed model state or names')
    need(len(list(model.parameters())) == 108, 'Unexpected trainable-parameter count')

    generator = torch.Generator(device='cpu').manual_seed(20261006)
    x = torch.randn(4, 6, 2, 10, 27, generator=generator)
    mask = torch.tensor([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1],
                         [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    x[~mask] = 12345.0  # finite poison makes accidental encoder inclusion detectable
    data = {'x': x, 'y': torch.arange(4), 'weight': torch.ones(4, dtype=torch.float64),
            'ids': [f'synthetic:{i}' for i in range(4)], 'origins': [f'synthetic-origin:{i}' for i in range(4)],
            'window_valid': mask}
    engine.validate_dataset(data, 93)
    original_data_digest = engine._dataset_signature(data)
    cfg = dict(plan['training_config'], batch_size=4, num_samples=4)
    engine.validate_config(cfg)
    sampler = torch.Generator(device='cpu').manual_seed(engine.epoch_seed(cfg['seed'], 1))
    draws = torch.multinomial(data['weight'] / data['weight'].sum(), 4, replacement=True, generator=sampler)
    expected_mask = mask[draws]
    need(torch.unique(expected_mask.sum(dim=1)).numel() > 1, 'Synthetic sampled batch needs mixed lengths')
    need(bool((~expected_mask).any()), 'Synthetic sampled batch needs invalid windows')
    hooks, seen = [], {'model': [], 'encoder': [], 'final_bn': []}

    def model_hook(_module, args, kwargs):
        xb, wb = args[0], kwargs['window_valid']
        need(torch.equal(wb, expected_mask), 'Engine passed mismatched sampled metadata')
        xb.requires_grad_(True)
        xb.retain_grad()
        seen['model'].append((xb, wb.detach().clone()))

    def encoder_hook(_module, args):
        seen['encoder'].append(args[0].detach().clone())

    def bn_hook(_module, args):
        seen['final_bn'].append(tuple(args[0].shape))

    hooks.append(model.register_forward_pre_hook(model_hook, with_kwargs=True))
    hooks.append(model.conv_encoder.register_forward_pre_hook(encoder_hook))
    hooks.append(model.final_bn.register_forward_pre_hook(bn_hook))
    before_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    shear, rotate = transforms.ShearTransform(.1), transforms.RotatationTransform(.1)
    augment = lambda sample: pose_io.augment_windows(sample, lambda t: rotate(shear({'frames': t}))['frames'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['base_lr'], weight_decay=0.0)
    try:
        record = engine._train_epoch(model, optimizer, data, cfg, 1, augment)
    finally:
        for hook in hooks:
            hook.remove()
    need(record['updates'] == 1 and record['grad_missing'] == 0 and record['grad_finite_checks'] == 108,
         'Actual-model optimizer/gradient record')
    need(len(seen['model']) == len(seen['encoder']) == len(seen['final_bn']) == 1,
         'Expected one model/encoder/final BN call for the minibatch')
    xb, wb = seen['model'][0]
    need(torch.equal(seen['encoder'][0], xb.detach()[wb]), 'Encoder did not receive exactly gathered valid windows')
    need(seen['encoder'][0].shape[0] == int(expected_mask.sum()), 'Encoder batch is not sum(valid windows)')
    need(seen['final_bn'][0] == (4, 256), 'Classifier BN must keep video batch size four')
    need(xb.grad is not None and bool(torch.isfinite(xb.grad).all()) and bool((xb.grad[~wb] == 0).all()),
         'Invalid windows received gradient')
    need(bool((xb.grad[wb].abs().flatten(1).sum(dim=1) > 0).any()), 'No valid-input gradient')
    need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
         'Actual parameter gradients missing/nonfinite')
    need(all(float(s['step']) == 1 for s in optimizer.state.values()) and len(optimizer.state) == 108,
         'Optimizer did not update all parameter tensors once')
    after_state = model.state_dict()
    changed = [k for k, _ in model.named_parameters() if not torch.equal(before_state[k], after_state[k])]
    need(changed and any(k.startswith('agg.') for k in changed)
         and any(k.startswith('conv_encoder.') for k in changed) and any(k.startswith('final_fc.') for k in changed),
         'Encoder, GRU or classifier did not change')
    bn_keys = [k for k in original_keys if k.endswith('num_batches_tracked')]
    need(all(int(after_state[k]) == int(before_state[k]) + 1 for k in bn_keys), 'BN must update once')
    graph_keys = [k for k in original_keys if k == 'A' or k.endswith('.A')]
    need(graph_keys and all(torch.equal(before_state[k], after_state[k]) for k in graph_keys), 'Graph buffers changed')
    need(engine._dataset_signature(data) == original_data_digest, 'Training mutated synthetic dataset')

    singleton = subset(data, [1])
    singleton['primary'] = torch.ones(1, dtype=torch.bool)
    engine.validate_dataset(singleton, 93, validation=True)
    state_before_eval = workflow.state_digest(model.state_dict())
    modes_before = [m.training for m in model.modules()]
    metrics = engine._evaluate(model, singleton, cfg)
    need([m.training for m in model.modules()] == modes_before, 'Internal evaluation did not restore modes')
    logits = workflow.score_batches(model, singleton['x'], 1, window_valid=singleton['window_valid'])
    need(logits.shape == (1, 93) and np.isfinite(logits).all(), 'Singleton score shape/value')
    ce = torch.nn.functional.cross_entropy(torch.from_numpy(logits).double(), singleton['y']).item()
    need(abs(ce - metrics['primary_ce']) <= 1e-12, 'Internal evaluation and score path disagree')
    need(metrics['primary_correct'] == int(logits.argmax(axis=1)[0] == int(singleton['y'][0])), 'Singleton row alignment')
    need(workflow.state_digest(model.state_dict()) == state_before_eval, 'Evaluation changed model state')
    need([m.training for m in model.modules()] == modes_before, 'Scoring did not restore modes')

    fit, val = subset(data, [0, 1, 2]), subset(data, [3])
    workflow._check_splits(fit, val, data)
    altered_val = subset(data, [3])
    altered_val['window_valid'][0, 0] = False
    need(engine._dataset_signature(altered_val) != engine._dataset_signature(val), 'Mask absent from engine hash')
    need(workflow._dataset_digest(altered_val) != workflow._dataset_digest(val), 'Mask absent from workflow hash')
    try:
        workflow._check_splits(fit, altered_val, data)
    except ValueError:
        mismatch_rejected = True
    else:
        raise AssertionError('Split metadata mismatch was not rejected')
    need(all(sha(Path(p)) == digest for p, digest in before_hashes.items()), 'Input/library/config file changed')
    return {'status': 'PASS', 'initial_state_digest': initial, 'initial_matches_old_run': True,
            'state_names_unchanged': True, 'pretrained_encoder_states': len(enc), 'pretrained_gru_states': len(gru),
            'synthetic_rows': 4, 'sampled_rows': draws.tolist(), 'sampled_window_counts': expected_mask.sum(1).tolist(),
            'optimizer_updates': 1, 'all_108_gradient_tensors_finite': True, 'changed_parameter_tensors': len(changed),
            'encoder_calls': 1, 'encoder_valid_windows': int(expected_mask.sum()), 'final_bn_video_batch': 4,
            'bn_counters_advanced': len(bn_keys), 'invalid_window_gradient_zero': True, 'graph_buffers_unchanged': True,
            'singleton_internal_and_score_paths_agree': True, 'evaluation_state_unchanged': True,
            'metadata_signature_and_split_mismatch_checked': mismatch_rejected,
            'existing_metadata_audit_reused': True, 'real_data_training_or_scoring': 0,
            'source_clips_loaded': 0, 'checkpoints_saved': 0, 'candidate_config_changed': False,
            'libraries_and_inputs_sha256': before_hashes,
            'limits': 'One disposable official-model update using synthetic B4 only; not candidate learning or a performance estimate.'}


def main():
    destination = U / 'INTEGRATION_CHECK.json'
    need(not destination.exists(), 'Integration result exists; do not overwrite a completed or failed check')
    try:
        result, code = check(), 0
    except Exception as exc:
        result = {'status': 'FAIL', 'error_type': type(exc).__name__, 'reason': str(exc), 'traceback': traceback.format_exc()}
        code = 1
    result['checked_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['checker_sha256'] = sha(Path(__file__))
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'libraries_and_inputs_sha256'}, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
