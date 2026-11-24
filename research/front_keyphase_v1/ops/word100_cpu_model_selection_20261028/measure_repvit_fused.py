"""Optional deployment timing after fixed fit-only RepViT fusion parity.

This never changes the original checkpoint or the native latency selection rule.
No correctness labels are read or evaluated.  No final-test images are opened.
"""
from pathlib import Path
import argparse
import copy
import gc
import importlib
import json
import os
import sys
import time
import traceback

import measure_cpu as base

U = Path(__file__).resolve().parent
P = U / 'REPVIT_FUSED_LATENCY_PROTOCOL.json'
RESULT = U / 'REPVIT_FUSED_RESULTS.json'
PARTIAL = U / 'REPVIT_FUSED_LATENCY_PARTIAL.json'
LOCK = U / 'REPVIT_FUSED_LATENCY_RUNNING.json'
PARITY_ARCHIVE = U / 'REPVIT_FUSED_PARITY_LOGITS.npz'
PARITY_TEMP = PARITY_ARCHIVE.with_name(PARITY_ARCHIVE.name + '.tmp')


def freeze():
    assert not any(p.exists() for p in [P, RESULT, PARTIAL, LOCK, PARITY_ARCHIVE, PARITY_TEMP]), 'Preserve frozen/final/partial fused timing artifacts'
    native = base.read(base.PROTOCOL)
    assert native['measure_cpu_sha256'] == base.sha(U / 'measure_cpu.py')
    module = U / 'deps' / 'timm' / 'models' / 'repvit.py'
    if not module.exists():
        module = U.parent / 'word100_feature_learning_pipeline_20261027' / 'deps' / 'timm' / 'models' / 'repvit.py'
    assert module.is_file()
    protocol = {
        'version': 1, 'frozen_at': base.now(), 'scope': 'Optional RepViT deployment optimization only; no new architecture or accuracy evaluation',
        'native_protocol_sha256': base.sha(base.PROTOCOL), 'native_code_sha256': base.sha(U / 'measure_cpu.py'),
        'fused_code_sha256': base.sha(Path(__file__)), 'repvit_module': str(module), 'repvit_module_sha256': base.sha(module),
        'name': 'repvit', 'checkpoint_seed': 28101, 'checkpoint_epoch': 20,
        'fusion': 'deepcopy(native_complete_model).eval(); fused.fuse(); function mutates copy and returns None',
        'parity': {'inputs': 'All300 fixed fit crops from CPU_LATENCY_PROTOCOL, batch16, no labels',
                   'full_logits': 45, 'atol': 0.0001, 'rtol': 0.0001,
                   'allclose_call': 'torch.allclose(fused_logits, native_logits, atol=1e-4, rtol=1e-4)',
                   'argmax_identical_required': 300, 'check_all_logits_finite': True,
                   'archive': {'file': PARITY_ARCHIVE.name,
                               'arrays': {'native_logits': 'float32[300,45]', 'fused_logits': 'float32[300,45]',
                                          'ids': 'unicode[300] in frozen selected order',
                                          'checkpoint_sha256': 'scalar unicode', 'protocol_sha256': 'scalar unicode'},
                               'sha256_in_result': True, 'labels_saved': False,
                               'purpose': 'Independent parity arithmetic, not accuracy re-evaluation'},
                   'failure': 'Write PARITY_FAILED_NO_TIMING result; no tolerance relaxation and no timing'},
        'timing': 'Only after parity pass, reuse exact measure_cpu.benchmark_model warmups/single50/clip120and300x5/fullnative preprocessing; CPU4FP32evalinference_mode',
        'requires_native_latency_complete_first': True, 'no_training_overlap': True,
        'original_checkpoint_immutable': True, 'native_model_state_hash_preservation': True,
        'native_selection_rule_unchanged': True,
        'interpretation': 'Additional deployment result; do not substitute fused p95 into the frozen native model-selection tiebreak. Component timing is not full feedback latency.',
        'private_data_access': False, 'final_test_access': False, 'accuracy_scoring': False, 'training': False,
    }
    base.write(P, protocol)
    print(json.dumps({'status': 'FROZEN_CODE_ONLY', 'protocol_sha256': base.sha(P), 'model_inference': 0}))


def validate(p):
    assert base.sha(Path(__file__)) == p['fused_code_sha256']
    assert base.sha(U / 'measure_cpu.py') == p['native_code_sha256']
    assert base.sha(base.PROTOCOL) == p['native_protocol_sha256']
    assert base.sha(p['repvit_module']) == p['repvit_module_sha256']
    assert not base.LOCK.exists(), 'Native timing active/unreviewed; no overlapping timing'
    native_result = base.read(base.RESULT)
    assert native_result['status'] == 'COMPLETE' and native_result['protocol_sha256'] == base.sha(base.PROTOCOL)
    assert {r['name'] for r in native_result['models']} == set(base.MODELS)
    native_protocol = base.read(base.PROTOCOL)
    training, training_sha = base.validate_ready(native_protocol)
    return native_protocol, native_result, training, training_sha


def run(confirm_idle):
    assert confirm_idle, 'Confirm no concurrent training only after independently checking active work'
    assert not RESULT.exists(), 'Completed fused result exists; do not rerun'
    assert not PARTIAL.exists() and not LOCK.exists(), 'Preserve interrupted result and lock; reviewed recovery required'
    assert not PARITY_ARCHIVE.exists() and not PARITY_TEMP.exists(), 'Parity archive exists; preserve, no re-inference'
    p = base.read(P)
    native_protocol, native_result, training, training_sha = validate(p)
    import compare_cpu as c
    import torch
    import numpy as np
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1
    module = importlib.import_module('timm.models.repvit')
    assert base.sha(module.__file__) == p['repvit_module_sha256']
    cp = U / 'models' / 'repvit_28101.pt'
    cp_sha = base.sha(cp)
    meta = base.read(cp.with_suffix('.json'))
    assert cp_sha == meta['checkpoint_sha256']
    native_row = next(r for r in native_result['models'] if r['name'] == 'repvit')
    assert cp_sha == native_row['checkpoint_sha256']
    protected = c.pins()
    with LOCK.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'at': base.now(), 'protocol_sha256': base.sha(P)}, f)
    out = {'status': 'RUNNING', 'started_at': base.now(), 'protocol_sha256': base.sha(P),
           'native_result_sha256': base.sha(base.RESULT), 'checkpoint_sha256': cp_sha,
           'name': 'repvit_fused', 'source_name': 'repvit', 'seed': 28101, 'epoch': 20,
           'environment': {'torch': torch.__version__, 'timm': c.timm.__version__, 'python': sys.version,
                           'cpu_registry_name': base.cpu_registry_name(), 'threads': 4, 'interop_threads': 1,
                           'device': 'cpu', 'dtype': 'float32', 'mkldnn_enabled': torch.backends.mkldnn.enabled,
                           'concurrent_training_operator_confirmed_absent': True},
           'training': False, 'accuracy_scoring': False, 'final_test_access': False,
           'user_mentor_access': False, 'native_selection_rule_unchanged': True}
    base.write(PARTIAL, out)
    try:
        m, _ = c.load_original('repvit')
        c.replace_and_configure('repvit', m, 28101)
        checkpoint = torch.load(cp, map_location='cpu', weights_only=True)
        assert checkpoint['protocol_sha256'] == training_sha
        assert checkpoint['name'] == 'repvit' and checkpoint['seed'] == 28101 and checkpoint['epoch'] == 20
        m.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        m.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)
        c.check_frozen(m, training['specs']['repvit'])
        before = c.hashes(m)
        fused = copy.deepcopy(m).eval()
        assert callable(getattr(fused, 'fuse', None))
        returned = fused.fuse()
        assert returned is None, 'Unexpected timm fusion API; do not guess a new path'
        fused.eval().requires_grad_(False)
        assert all(not v.training for v in fused.modules())
        assert all(v.device.type == 'cpu' and v.dtype == torch.float32 for v in fused.parameters())
        count = 0; argmax_equal = 0; allclose = True; max_abs = 0.; max_tolerance_fraction = 0.
        native_parts = []; fused_parts = []; parity_ids = []
        with torch.inference_mode():
            for group in base.chunks(native_protocol['selected']):
                assert not (U / 'RUNNING.json').exists() and not base.LOCK.exists()
                x = torch.stack([c.image_tensor(r['image'], c.SIZES['repvit']) for r in group])
                a = m(x); b = fused(x)
                assert a.shape == b.shape == (len(group), 45)
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                delta = (a - b).abs()
                max_abs = max(max_abs, float(delta.max()))
                tolerance = p['parity']['atol'] + p['parity']['rtol'] * a.abs()
                max_tolerance_fraction = max(max_tolerance_fraction, float((delta / tolerance).max()))
                allclose = allclose and bool(torch.allclose(b, a, atol=p['parity']['atol'], rtol=p['parity']['rtol']))
                argmax_equal += int((a.argmax(1) == b.argmax(1)).sum())
                count += len(group)
                native_parts.append(a.cpu().numpy().copy())
                fused_parts.append(b.cpu().numpy().copy())
                parity_ids.extend(r['id'] for r in group)
        assert count == 300 and c.hashes(m) == before and base.sha(cp) == cp_sha
        native_logits = np.concatenate(native_parts, axis=0)
        fused_logits = np.concatenate(fused_parts, axis=0)
        assert native_logits.shape == fused_logits.shape == (300, 45)
        assert native_logits.dtype == fused_logits.dtype == np.float32
        assert parity_ids == [r['id'] for r in native_protocol['selected']]
        assert len(set(parity_ids)) == 300
        with PARITY_TEMP.open('xb') as archive:
            np.savez_compressed(archive, native_logits=native_logits, fused_logits=fused_logits,
                                ids=np.asarray(parity_ids, dtype=str),
                                checkpoint_sha256=np.asarray(cp_sha), protocol_sha256=np.asarray(base.sha(P)))
        assert not PARITY_ARCHIVE.exists()
        os.replace(PARITY_TEMP, PARITY_ARCHIVE)
        archive_sha = base.sha(PARITY_ARCHIVE)
        parity_pass = allclose and argmax_equal == 300
        out['parity'] = {'fit_crops': count, 'scalar_logits': count * 45, 'atol': p['parity']['atol'],
                         'rtol': p['parity']['rtol'], 'max_abs_logit_difference': max_abs,
                         'max_difference_over_allclose_tolerance': max_tolerance_fraction,
                         'allclose': allclose, 'argmax_equal': argmax_equal, 'passed': parity_pass,
                         'archive': str(PARITY_ARCHIVE), 'archive_sha256': archive_sha,
                         'archive_arrays': ['native_logits', 'fused_logits', 'ids', 'checkpoint_sha256', 'protocol_sha256'],
                         'original_model_state_unchanged': True, 'checkpoint_unchanged': True}
        base.write(PARTIAL, out)
        del native_parts, fused_parts, native_logits, fused_logits, parity_ids
        if parity_pass:
            del m, x, a, b, delta, tolerance; gc.collect()
            timing = base.benchmark_model(fused, 'repvit', native_protocol, c, torch)
            out.update(timing)
            out.update(status='COMPLETE', timing_executed=True, structural_fusion=True,
                       parameters=sum(x.numel() for x in fused.parameters()))
        else:
            out.update(status='PARITY_FAILED_NO_TIMING', timing_executed=False,
                       explanation='Fixed parity condition failed; tolerance unchanged; no speed claim.')
        assert not (U / 'RUNNING.json').exists() and not base.LOCK.exists()
        assert base.sha(cp) == cp_sha
        assert base.sha(PARITY_ARCHIVE) == archive_sha
        out.update(finished_at=base.now(), checkpoint_unchanged=True,
                   protected_unchanged=c.pins() == protected, interpretation=p['interpretation'])
        base.write(RESULT, out); base.write(PARTIAL, out); LOCK.unlink()
        print(json.dumps({'status': out['status'], 'parity': out['parity'], 'timing_executed': out['timing_executed']}), flush=True)
    except BaseException as exc:
        base.write(U / 'logs' / f'REPVIT_FUSED_LATENCY_ERROR_{time.time_ns()}.json',
                   {'at': base.now(), 'type': type(exc).__name__, 'message': str(exc),
                    'traceback': traceback.format_exc(), 'lock_preserved': True, 'partial_preserved': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'validate', 'run'])
    parser.add_argument('--confirm-no-concurrent-training', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze': freeze()
    elif args.command == 'validate':
        validate(base.read(P)); print(json.dumps({'status': 'READY', 'timing_executed': False}))
    else: run(args.confirm_no_concurrent_training)
