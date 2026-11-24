"""Frozen fit-only full-model CPU latency; never score or open final-test images.

Run only after compare_cpu.py has completed all nine training runs.  This is
component timing, not record-to-feedback latency.  Prefix caches are not used.
"""
from pathlib import Path
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback

U = Path(__file__).resolve().parent
PROTOCOL = U / 'CPU_LATENCY_PROTOCOL.json'
RESULT = U / 'LATENCY_RESULTS.json'
PARTIAL = U / 'CPU_LATENCY_PARTIAL.json'
LOCK = U / 'CPU_LATENCY_RUNNING.json'
MANIFEST = U / 'TRAINING_MANIFEST.json'
MODELS = ['v3', 'repvit', 'v4hybrid']
FIT_SIGNERS = {'signer01', 'signer02', 'signer04', 'signer05'}
SEED = 28101
ORDER_SALT = 'WordsInSilence-fit-only-CPU-latency-20261028-v1|'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def cpu_registry_name():
    if os.name != 'nt':
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    except OSError:
        return None


def write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def select_fit(rows):
    fit = [r for r in rows if r['role'] == 'fit']
    assert len(fit) == 2489, 'Unexpected fixed training input count'
    assert {r['signer'] for r in fit} == FIT_SIGNERS
    assert len({r['id'] for r in fit}) == len(fit)
    ordered = sorted(fit, key=lambda r: (hashlib.sha256((ORDER_SALT + r['id']).encode()).hexdigest(), r['id']))
    return ordered[:300]


def chunks(items, batch=16):
    assert batch == 16
    return [items[i:i + batch] for i in range(0, len(items), batch)]


def summarize(seconds):
    # NumPy-compatible linear interpolation, fixed before any timing result.
    assert seconds and all(math.isfinite(x) and x >= 0 for x in seconds)
    vals = sorted(float(x) for x in seconds)
    def quantile(q):
        pos = (len(vals) - 1) * q
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo) if lo != hi else vals[lo]
    return {'repeats': len(vals), 'raw_seconds': list(seconds),
            'p50_seconds': quantile(.5), 'p95_seconds': quantile(.95),
            'p50_ms': 1000 * quantile(.5), 'p95_ms': 1000 * quantile(.95),
            'min_seconds': vals[0], 'max_seconds': vals[-1],
            'p95_note': 'Linear sample quantile; five clip repetitions do not estimate a population tail reliably.'}


def freeze():
    assert not any(p.exists() for p in [PROTOCOL, RESULT, PARTIAL, LOCK]), 'Preserve existing frozen timing artifacts'
    rows = select_fit(read(MANIFEST)['rows'])
    selected = [{k: r[k] for k in ['id', 'image', 'image_sha256', 'signer', 'role']} for r in rows]
    for r in selected:
        assert r['role'] == 'fit' and r['signer'] in FIT_SIGNERS
    p = {
        'version': 1, 'frozen_at': now(), 'scope': 'Public fit-only CPU full-image-model component latency, no accuracy scoring',
        'device': 'cpu', 'dtype': 'float32', 'intraop_threads': 4, 'interop_threads': 1,
        'mode': 'eval + torch.inference_mode; no autocast, compilation or structural fusion',
        'repvit_fusion': 'Native timing is the frozen selection rule. timm reparameterize_model and RepVit.fuse are available; native timing alone does not describe the fastest deployment. A later separately reported fused measurement needs fixed-tolerance full-logit and argmax parity on these300 fit crops; not implemented or timed here.',
        'model_order': MODELS, 'checkpoint_seed': SEED, 'checkpoint_epoch': 20,
        'checkpoint_contract': 'models/{name}_28101.pt; strict full state; training result JSON SHA and FROZEN_PROTOCOL hash must match',
        'require_all_training_runs_completed': 9, 'require_training_lock_absent': True,
        'require_operator_no_concurrent_training_confirmation': True,
        'input_sizes': {'v3': 224, 'repvit': 224, 'v4hybrid': 256},
        'preprocessing': 'compare_cpu.image_tensor: PNG decode to RGB; entire supplied crop preserved by bicubic letterbox, gray128; ImageNet mean/std; torch.stack batch',
        'selection': {'population': 'TRAINING_MANIFEST rows with role fit only', 'fit_count': 2489,
                      'order': 'ascending SHA256(salt + id), tie id', 'salt': ORDER_SALT,
                      'selected_count': 300, 'clip_counts': [120, 300], 'subset': '120 are first120 of300; single crop first1',
                      'labels_used': False, 'validation_images_accessed': 0, 'final_test_images_accessed': 0},
        'selected': selected, 'batch_size': 16, 'batch_warmup': 3, 'single_warmup': 10,
        'single_repeats': 50, 'clip_repeats': 5,
        'clock': 'time.perf_counter, CPU eager synchronously completed outputs',
        'quantile': 'linear interpolation at (n-1)*q, q=.5/.95; raw samples retained',
        'single_timing': 'Full m(x), preprocessed one-image tensor, no load/decode/stack included',
        'model_only_clip_timing': 'Full m(x) for all batches, wall clock around whole batch loop; predecoded tensors retained, no prefix features',
        'preprocessing_inclusive_clip_timing': 'Wall clock around all PNG reads + RGB decode + letterbox + numeric normalization + batch stacking + full m(x)',
        'cache': 'Up to300 RGB tensors in batches <=16, released before next model; OS disk cache is warm; cold storage performance not measured',
        'excluded_from_both': ['checkpoint load', 'model construction', 'MediaPipe detection', 'video decoding',
                               'hand crop extraction', 'TCN', 'DTW', 'network/server/UI', 'user recording time'],
        'interpretation': '120 hand crops can illustrate 60 frames with two hands each. Inputs here are independent fit stills, not an actual60-frame clip, and these times are not end-to-end feedback latency.',
        'latency_expectation': 'About3s user soft reference; >5s undesirable for full feedback. Neither is an admission threshold on this component alone.',
        'repeat_guard': 'Existing final result refuses inference. Existing partial/lock requires explicit reviewed recovery, not silent rerun.',
        'training_manifest_sha256': sha(MANIFEST), 'compare_cpu_sha256': sha(U / 'compare_cpu.py'),
        'measure_cpu_sha256': sha(Path(__file__)), 'protected_manifest_sha256': sha(U / 'PROTECTED.json'),
        'user_mentor_access': False, 'accuracy_scoring': False, 'training': False,
    }
    write(PROTOCOL, p)
    print(json.dumps({'status': 'FROZEN_CODE_ONLY', 'selected_fit': len(selected), 'protocol_sha256': sha(PROTOCOL)}))


def validate_ready(p):
    assert sha(Path(__file__)) == p['measure_cpu_sha256'], 'Latency implementation changed after freeze'
    assert sha(U / 'compare_cpu.py') == p['compare_cpu_sha256']
    assert sha(MANIFEST) == p['training_manifest_sha256']
    assert sha(U / 'PROTECTED.json') == p['protected_manifest_sha256']
    assert not (U / 'RUNNING.json').exists(), 'Training active or interrupted; do not overlap timing'
    training = read(U / 'FROZEN_PROTOCOL.json')
    training_sha = sha(U / 'FROZEN_PROTOCOL.json')
    assert training['code_sha256'] == p['compare_cpu_sha256']
    assert training['manifest_sha256'] == p['training_manifest_sha256']
    assert training['models'] == MODELS and training['torch_threads'] == 4 and training['device'] == 'CPU'
    assert training['seeds'] == [28101, 28102, 28103] and training['epochs'] == 20
    complete = read(U / 'TRAINING_COMPLETE.json')
    assert complete['runs'] == 9 and complete['test_predictions'] == 0 and complete['protocol_sha256'] == training_sha
    assert read(U / 'STATUS.json')['state'] == 'TRAINING_COMPLETE'
    # All nine saved results must exist before timing; only seed28101 checkpoints
    # are opened.  No development logits or final-test data are opened.
    for name in MODELS:
        for seed in training['seeds']:
            meta = read(U / 'models' / f'{name}_{seed}.json')
            assert meta['name'] == name and meta['seed'] == seed and meta['epoch'] == 20
            assert meta['protocol_sha256'] == training_sha and meta['test_predictions'] == 0
            assert (U / 'models' / f'{name}_{seed}.pt').is_file()
    expected = select_fit(read(MANIFEST)['rows'])
    assert p['selected'] == [{k: r[k] for k in ['id', 'image', 'image_sha256', 'signer', 'role']} for r in expected]
    for r in p['selected']:
        assert r['role'] == 'fit' and r['signer'] in FIT_SIGNERS
        assert Path(r['image']).resolve().is_relative_to((U.parent / 'word100_feature_learning_pipeline_20261027' / 'data' / 'deephand_images').resolve())
        assert sha(r['image']) == r['image_sha256']
    return training, training_sha


def benchmark_model(m, name, p, c, torch):
    """Shared native/fused timing, full m(x); caller handles readiness and lock."""
    groups = chunks(p['selected'])
    batches = [torch.stack([c.image_tensor(r['image'], c.SIZES[name]) for r in group]) for group in groups]
    assert sum(len(x) for x in batches) == 300 and max(map(len, batches)) <= 16
    assert all(torch.isfinite(x).all() and x.dtype == torch.float32 for x in batches)
    x1 = batches[0][:1]
    with torch.inference_mode():
        for _ in range(p['batch_warmup']):
            y = m(batches[0])
        for _ in range(p['single_warmup']):
            y = m(x1)
        assert y.shape == (1, 45) and torch.isfinite(y).all()
        single = []
        for _ in range(p['single_repeats']):
            start = time.perf_counter(); y = m(x1); single.append(time.perf_counter() - start)
        assert torch.isfinite(y).all()
        clips = []
        for n in p['selection']['clip_counts']:
            chosen = []; left = n
            for x in batches:
                if left <= 0: break
                chosen.append(x[:min(len(x), left)]); left -= len(chosen[-1])
            assert sum(len(x) for x in chosen) == n
            model_seconds = []
            inclusive_seconds = []
            for repeat in range(p['clip_repeats']):
                assert not (U / 'RUNNING.json').exists(), 'Training began during timing'
                start = time.perf_counter()
                for x in chosen: y = m(x)
                model_seconds.append(time.perf_counter() - start)
                assert torch.isfinite(y).all()
                start = time.perf_counter()
                for group in chunks(p['selected'][:n]):
                    x = torch.stack([c.image_tensor(r['image'], c.SIZES[name]) for r in group])
                    y = m(x)
                inclusive_seconds.append(time.perf_counter() - start)
                assert torch.isfinite(y).all()
            clips.append({'hand_crops': n, 'batches': [len(x) for x in chosen],
                          'full_model_only': summarize(model_seconds),
                          'rgb_decode_preprocess_full_model': summarize(inclusive_seconds)})
    return {'single_full_model': summarize(single), 'clips': clips,
            'cached_rgb_tensor_bytes': sum(x.numel() * x.element_size() for x in batches)}


def run(confirm_idle):
    assert confirm_idle, 'Use --confirm-no-concurrent-training only after independently checking active work'
    assert not RESULT.exists(), 'Completed timing exists: no remeasurement'
    assert not PARTIAL.exists() and not LOCK.exists(), 'Preserve partial/lock; reviewed recovery required'
    p = read(PROTOCOL)
    training, training_sha = validate_ready(p)
    # Import the exact frozen training module only after training completion.
    import compare_cpu as c
    import torch
    torch.set_num_threads(p['intraop_threads'])
    torch.set_num_interop_threads(p['interop_threads'])
    torch.use_deterministic_algorithms(True)
    assert torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1
    protected = c.pins()
    with LOCK.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'at': now(), 'protocol_sha256': sha(PROTOCOL), 'command': 'measure_cpu.py run'}, f)
    output = {'status': 'RUNNING', 'at_started': now(), 'protocol_sha256': sha(PROTOCOL),
              'training_protocol_sha256': training_sha, 'training_complete_sha256': sha(U / 'TRAINING_COMPLETE.json'),
              'environment': {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
                              'cpu_registry_name': cpu_registry_name(), 'mkldnn_enabled': torch.backends.mkldnn.enabled,
                              'logical_cpu_count': os.cpu_count(), 'torch': torch.__version__, 'timm': c.timm.__version__,
                              'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                              'cuda_used': False, 'dtype': 'float32', 'concurrent_training_operator_confirmed_absent': True,
                              'unknown_background_OS_load': True},
              'models': [], 'accuracy_scoring': False, 'final_test_images_accessed': 0,
              'validation_images_accessed': 0, 'user_mentor_access': False,
              'component_only': True, 'cold_disk_timing': False, 'prefix_cache_used': False}
    write(PARTIAL, output)
    try:
        for name in p['model_order']:
            assert not (U / 'RUNNING.json').exists(), 'Training began during latency run'
            cp = U / 'models' / f'{name}_{SEED}.pt'
            meta = read(cp.with_suffix('.json'))
            assert sha(cp) == meta['checkpoint_sha256']
            m, _ = c.load_original(name)
            c.replace_and_configure(name, m, SEED)
            checkpoint = torch.load(cp, map_location='cpu', weights_only=True)
            assert checkpoint['name'] == name and checkpoint['seed'] == SEED
            assert checkpoint['epoch'] == 20 and checkpoint['protocol_sha256'] == training_sha
            m.load_state_dict(checkpoint['model'], strict=True)
            m.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)
            c.check_frozen(m, training['specs'][name])
            del checkpoint
            assert all(not x.training for x in m.modules())
            assert all(v.device.type == 'cpu' and v.dtype == torch.float32 for v in m.parameters())
            assert c.SIZES[name] == p['input_sizes'][name]
            timing = benchmark_model(m, name, p, c, torch)
            record = {'name': name, 'seed': SEED, 'epoch': 20, 'checkpoint': str(cp), 'checkpoint_sha256': sha(cp),
                      'input_size': c.SIZES[name], 'parameters': sum(x.numel() for x in m.parameters()),
                      'full_forward': 'm(x), unchanged full model including official head path and trained45 classifiers',
                      'structural_fusion': False, **timing,
                      'measured_at': now(), 'predictions_saved': False, 'accuracy_computed': False}
            output['models'].append(record); write(PARTIAL, output)
            print(json.dumps({'name': name, 'single_p95_ms': record['single_full_model']['p95_ms'],
                              'clip_inclusive_p95_seconds': {x['hand_crops']: x['rgb_decode_preprocess_full_model']['p95_seconds'] for x in record['clips']}}), flush=True)
            del m; gc.collect()
        assert not (U / 'RUNNING.json').exists()
        output.update(status='COMPLETE', at_finished=now(), protected_unchanged=c.pins() == protected,
                      interpretation=p['interpretation'], exclusions=p['excluded_from_both'])
        assert not RESULT.exists()
        write(RESULT, output)
        # Keep the completed partial journal as evidence; final output is the
        # completion marker and the only allowed read/reuse entry point.
        write(PARTIAL, output)
        LOCK.unlink()
    except BaseException as exc:
        err = {'at': now(), 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
               'completed_models': [r['name'] for r in output['models']], 'lock_preserved': True,
               'partial_preserved': True, 'protocol_sha256': sha(PROTOCOL)}
        (U / 'logs').mkdir(exist_ok=True)
        write(U / 'logs' / f'CPU_LATENCY_ERROR_{time.time_ns()}.json', err)
        raise


def self_test():
    # Pure arithmetic/manifest tests only: no image decoding or model inference.
    rows = read(MANIFEST)['rows']; selected = select_fit(rows)
    assert selected == select_fit(list(reversed(rows)))
    assert len(selected) == 300 and len({r['id'] for r in selected}) == 300
    assert all(r['role'] == 'fit' and r['signer'] in FIT_SIGNERS for r in selected)
    assert [len(x) for x in chunks(selected[:120])] == [16] * 7 + [8]
    assert [len(x) for x in chunks(selected)] == [16] * 18 + [12]
    assert summarize([1, 2, 3, 4, 5])['p95_seconds'] == 4.8
    assert summarize([2] * 50)['p50_seconds'] == 2
    assert summarize([.1])['p95_seconds'] == .1
    print(json.dumps({'status': 'PASS', 'checks': 8, 'image_decoding': 0, 'model_inference': 0, 'training': 0,
                      'selected_fit': len(selected), 'selected_signers': sorted({r['signer'] for r in selected})}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'self-test', 'validate', 'run'])
    parser.add_argument('--confirm-no-concurrent-training', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze': freeze()
    elif args.command == 'self-test': self_test()
    elif args.command == 'validate':
        validate_ready(read(PROTOCOL))
        print(json.dumps({'status': 'READY', 'model_inference': 0, 'timing_executed': False}))
    else: run(args.confirm_no_concurrent_training)
