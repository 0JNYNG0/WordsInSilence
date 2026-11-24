"""Prepare/freeze/run the approved9-run comparison. No extraction or implicit training.

The new body-boundary rule lives outside this module. `freeze` requires an independently
approved readiness record, exact prepared input hashes and the new preparation context.
`run` is resumable at committed epoch boundaries; completed scores are never re-inferred.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
LAB = U.parent.parent
ROOT = LAB.parent.parent
OLD = LAB / 'ops/word100_training_20261013'
BASE = LAB / 'ops/word100_roi_training_20261017'
PILOT = ROOT / 'research/sign_model_pilot_v1/src'
sys.path.insert(0, str(PILOT))

import numpy as np
import torch
from pilot.model_masked_v1 import MaskedSignTCN
from training_core import (ARMS, SEEDS, VIEWS, MODEL, TRAIN, schedule,
                           validate_channels, padded_arrays, prediction,
                           detailed_metrics, paired, aggregate, progression_gate)

CONTEXT = U / 'training_CONTEXT.json'
PROGRESS = U / 'training_PROGRESS.json'
M0 = LAB / 'runs/word100_tcn_v1/whole_context/best.pt'
M0_SHA = 'e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'
MANIFEST_SHA = '3140045381a3316153f0f998979a63fda8609340409ce05f09e673715ebfa76c'
MODEL_SHA = 'fb8a963a76395fa6c6ac2281f84c968e672b4940a47c4ab932856f9c0d5f1828'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(p, value, new=False):
    p = Path(p)
    if new:
        assert not p.exists(), 'Immutable output already exists: ' + str(p)
    temp = p.with_name(p.name + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, p)


def atomic_torch(p, value):
    temp = p.with_name(p.name + '.writing')
    torch.save(value, temp)
    os.replace(temp, p)


def event(stage, **kw):
    entry = dict(at_utc=utc(), pid=os.getpid(), stage=stage, **kw)
    with (U / 'training_EVENTS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
    atomic_json(PROGRESS, entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def pins_ok(pins):
    for p, expected in pins.items():
        assert sha(p) == expected, 'Pinned input changed: ' + p


def runtime():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                torch=str(torch.__version__), numpy=np.__version__, platform=platform.platform())


def configure_torch():
    torch.set_num_threads(TRAIN['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def freeze(args):
    """Consume readiness; do not fabricate readiness or know the boundary formula."""
    if CONTEXT.exists():
        c = read(CONTEXT)
        assert Path(c['readiness']).resolve() == Path(args.readiness).resolve()
        assert Path(c['preparation_context']).resolve() == Path(args.prep_context).resolve()
        assert Path(c['records_directory']).resolve() == Path(args.records_dir).resolve()
        pins_ok(c['pins'])
        print('Existing frozen context verified; no overwrite.', flush=True)
        return
    auth = read(U / 'AUTHORIZATION.json')
    assert auth['approved'] is True
    ready_path = Path(args.readiness).resolve()
    prep_path = Path(args.prep_context).resolve()
    records_dir = Path(args.records_dir).resolve()
    ready = read(ready_path)
    assert ready['status'] == 'PASS' and ready['training_allowed'] is True
    assert ready['blockers'] == []
    assert ready['preparation_context_sha256'] == sha(prep_path)
    assert ready['manifest_sha256'] == MANIFEST_SHA
    assert ready['independent_input_validation_passed'] is True
    assert ready['motion_preservation_review_passed'] is True
    assert ready['fresh_sensor_detector_state_verified'] is True
    assert ready['roi_hidden_coordinate_invariance_passed'] is True
    assert sha(OLD / 'MANIFEST.json') == MANIFEST_SHA and sha(M0) == M0_SHA
    assert sha(PILOT / 'pilot/model_masked_v1.py') == MODEL_SHA
    baseline_check = read(BASE / 'BASELINE_SAVED_INDEPENDENT.json')
    assert baseline_check['status'] == 'PASS_SAVED_BASELINE_INDEPENDENT_RECOUNT'
    assert baseline_check['prediction_sha256'] == sha(BASE / 'baseline_PREDICTIONS.json')
    assert baseline_check['top1_correct'] == 290 and baseline_check['top5_correct'] == 299
    manifest = read(OLD / 'MANIFEST.json')
    assert len(manifest) == 1199 and len({r['clip_id'] for r in manifest}) == 1199
    assert Counter(r['role'] for r in manifest) == {'train_candidate': 899, 'development_only': 300}
    assert set(ready['record_sha256']) == {r['clip_id'] for r in manifest}
    dev_ids = {r['clip_id'] for r in manifest if r['role'] == 'development_only'}
    for view in VIEWS:
        assert set(ready['missingness_groups'][view]) == dev_ids
        assert set(ready['missingness_groups'][view].values()) <= {
            'one_hand_never_usable', 'internal_reentry', 'other'}
    pin_paths = [ready_path, prep_path, U / 'AUTHORIZATION.json', OLD / 'MANIFEST.json',
                 OLD / 'FROZEN_EXPERIMENT.json', M0, BASE / 'baseline_PREDICTIONS.json',
                 BASE / 'BASELINE_SAVED_INDEPENDENT.json', Path(__file__),
                 U / 'training_core.py', PILOT / 'pilot/model_masked_v1.py',
                 PILOT / 'pilot/data.py', PILOT / 'pilot/web_input.py']
    pins = {str(p.resolve()): sha(p) for p in pin_paths}
    # Inputs are trusted only after both the ready record and each record's output pins match.
    rows = []
    failures = {v: [] for v in ('masked', 'sensor', 'm0_sensor')}
    for original in manifest:
        cid = original['clip_id']
        rp = records_dir / (cid + '.json')
        assert sha(rp) == ready['record_sha256'][cid]
        rec = read(rp)
        for k in ('clip_id', 'role', 'signer_id', 'label_index', 'word_code'):
            assert rec[k] == original[k], 'Manifest identity mismatch: ' + cid + ':' + k
        assert rec['context_sha256'] == sha(prep_path)
        train = rec['role'] == 'train_candidate'
        assert rec['signer_id'] in (range(1, 10) if train else (11, 12, 13))
        pins[str(rp)] = sha(rp)
        row = {k: rec[k] for k in ('clip_id', 'role', 'signer_id', 'label_index', 'word_code')}
        row['views'] = {}
        modes = ('native', 'masked', 'sensor') if train else ('masked', 'sensor', 'm0_sensor')
        for mode in modes:
            view = rec['views'][mode]
            cp = view['channel_file']
            if cp is None:
                assert not train, 'TRAIN view unavailable: ' + cid + ':' + mode
                reason = view['feature_info'].get('reason') or view['feature_info'].get('status')
                assert reason, 'Missing input needs an explicit frozen cause'
                row['views'][mode] = dict(channel_file=None, reason=reason)
                failures[mode].append(cid)
                continue
            cp = str(Path(cp).resolve())
            assert rec['outputs'][cp] == sha(cp), 'Channel not pinned in record outputs: ' + cp
            pins[cp] = sha(cp)
            with np.load(cp, allow_pickle=False) as z:
                for key in (('x_whole', 'x_context') if train else ('x_whole',)):
                    validate_channels(z[key])
            row['views'][mode] = dict(channel_file=cp, sha256=pins[cp], reason=None)
        rows.append(row)
    # Preserve the original manifest order for the original seed+epoch permutation convention.
    f = read(OLD / 'FROZEN_EXPERIMENT.json')
    assert f['model'] == dict(type='MaskedSignTCN', **MODEL)
    c = dict(schema='body_boundary_ABC_training_v1', frozen_at_utc=utc(),
             authorization=str((U / 'AUTHORIZATION.json').resolve()),
             readiness=str(ready_path), preparation_context=str(prep_path),
             records_directory=str(records_dir), pins=pins, runtime=runtime(),
             ordered_rows=rows, labels=f['labels'], word_codes=f['classes'],
             model=MODEL, training=TRAIN, seeds=list(SEEDS), arms=list(ARMS),
             fixed_input_failures=failures, missingness_groups=ready['missingness_groups'],
             train_count=899, development_count=300,
             untested_final_signers=True, production_replacement=False,
             resume_policy='Atomic epoch checkpoint: discard only incomplete epoch and repeat from committed model/optimizer/RNG. Never retune or restart a completed run.',
             boundary_definition='Opaque pinned preparation context; no boundary constants in training code.')
    atomic_json(CONTEXT, c, new=True)
    event('CONTEXT_FROZEN_NO_TRAINING', rows=1199, pins=len(pins), failures={k: len(v) for k,v in failures.items()})


def load_data(c):
    train, dev = [], []
    for row in c['ordered_rows']:
        r = copy.deepcopy(row)
        r['arrays'] = {}
        for mode, info in r['views'].items():
            if info['channel_file'] is None:
                r['arrays'][mode] = None
            else:
                with np.load(info['channel_file'], allow_pickle=False) as z:
                    keys = ('x_whole', 'x_context') if r['role'] == 'train_candidate' else ('x_whole',)
                    r['arrays'][mode] = {k: z[k].copy() for k in keys}
        (train if r['role'] == 'train_candidate' else dev).append(r)
    assert len(train) == 899 and len(dev) == 300
    return train, dev


def torch_batch(rows, epoch=None, arm=None, view=None):
    if view is not None:
        xs = [r['arrays'][view]['x_whole'] for r in rows]
    else:
        choices = [schedule(r['clip_id'], epoch, arm) for r in rows]
        xs = [r['arrays'][s][t] for r, (s,t) in zip(rows, choices)]
    x, p = padded_arrays(xs)
    y = np.array([r['label_index'] for r in rows], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(p), torch.from_numpy(y)


def state_hash(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def predict(m, dev, view):
    before = state_hash(m.state_dict())
    m.eval()
    good = [r for r in dev if r['arrays'][view] is not None]
    out = [prediction(r, reason=r['views'][view]['reason']) for r in dev if r['arrays'][view] is None]
    with torch.no_grad():
        for at in range(0, len(good), TRAIN['batch_size']):
            rows = good[at:at+TRAIN['batch_size']]
            x, p, _ = torch_batch(rows, view=view)
            logits = m(x, p)
            assert tuple(logits.shape) == (len(rows),100) and torch.isfinite(logits).all()
            out.extend(prediction(r, a) for r,a in zip(rows, logits.numpy()))
    assert state_hash(m.state_dict()) == before, 'Inference changed frozen parameters or BN states'
    out.sort(key=lambda r: r['clip_id'])
    return out


def normalization_of_saved(rows, dev):
    expected = {r['clip_id']: r for r in dev}
    assert set(expected) == {r['clip_id'] for r in rows} and len(rows) == 300
    out = []
    for r in rows:
        e = expected[r['clip_id']]
        assert all(r[k] == e[k] for k in ('label_index', 'signer_id'))
        out.append(prediction(r, r['logits'], r.get('reason')))
    return sorted(out, key=lambda r:r['clip_id'])


def m0_scores(c, dev):
    native = normalization_of_saved(read(BASE / 'baseline_PREDICTIONS.json'), dev)
    assert sum(r['correct'] for r in native) == 290
    path = U / 'training_m0_sensor.json'
    if path.exists():
        saved = read(path)
        assert saved['training_context_sha256'] == sha(CONTEXT) and saved['checkpoint_sha256'] == M0_SHA
        sensor = normalization_of_saved(saved['predictions'], dev)
        assert saved['metrics'] == detailed_metrics(sensor, c['missingness_groups']['sensor'])
        return dict(native=native, sensor=sensor)
    guard = U / 'training_m0_sensor_STARTED.json'
    assert not guard.exists(), 'M0 sensor inference was interrupted. Preserve evidence and review recovery; never silently repeat.'
    atomic_json(guard, dict(at_utc=utc(), pid=os.getpid(), context_sha256=sha(CONTEXT)), new=True)
    ck = torch.load(M0, map_location='cpu', weights_only=True)
    assert ck['classes'] == c['labels'] and ck['word_codes'] == c['word_codes']
    m = MaskedSignTCN(**MODEL); m.load_state_dict(ck['state_dict'])
    start = time.perf_counter()
    sensor = predict(m, dev, 'm0_sensor')
    assert sha(M0) == M0_SHA
    atomic_json(path, dict(status='SAVED_ONCE', training_context_sha256=sha(CONTEXT),
                          checkpoint_sha256=M0_SHA, seconds=time.perf_counter()-start,
                          predictions=sensor, metrics=detailed_metrics(sensor, c['missingness_groups']['sensor'])), new=True)
    event('M0_SENSOR_SCORED_ONCE', top1=sum(r['correct'] for r in sensor))
    return dict(native=native, sensor=sensor)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def selected_checkpoint(best, c, arm, seed):
    return dict(state_dict=best['model'], classes=c['labels'], word_codes=c['word_codes'],
                config={k:v for k,v in MODEL.items() if k not in ('in_channels','n_classes')},
                in_channels=280, feature_schema=dict(builder='pilot.data.build_channels'),
                model_type=MaskedSignTCN.model_type, model_version=MaskedSignTCN.model_version,
                epoch=best['epoch'], step=best['step'], development_ce=best['ce'], arm=arm, seed=seed,
                initialization='fresh random; AIHub899 only', training_context_sha256=sha(CONTEXT))


def run_arm(c, train, dev, arm, seed):
    prefix = f'training_{arm}_{seed}'
    last_path = U / (prefix + '_last.pt')
    selected_path = U / (prefix + '_best.pt')
    result_path = U / (prefix + '_RESULT.json')
    if result_path.exists():
        result = read(result_path)
        assert result['context_sha256'] == sha(CONTEXT)
        assert sha(selected_path) == result['checkpoint_sha256']
        assert result['status'] == 'COMPLETE_DEVELOPMENT_ONLY'
        for view in VIEWS:
            rows = normalization_of_saved(result['predictions'][view], dev)
            assert result['metrics'][view] == detailed_metrics(rows, c['missingness_groups'][view])
        assert abs(result['selected_ce']-(result['metrics']['masked']['mean_ce']+
                                          result['metrics']['sensor']['mean_ce'])/2) < 1e-12
        event('COMPLETED_RUN_REUSED', arm=arm, seed=seed)
        return result
    set_seed(seed)
    m = MaskedSignTCN(**MODEL)
    assert m.parameter_count() == 37636
    init_hash = state_hash(m.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=TRAIN['lr'], weight_decay=TRAIN['weight_decay'])
    if last_path.exists():
        last = torch.load(last_path, map_location='cpu', weights_only=True)
        assert last['context_sha256'] == sha(CONTEXT) and last['arm'] == arm and last['seed'] == seed
        assert last['initial_state_sha256'] == init_hash
        m.load_state_dict(last['model']); opt.load_state_dict(last['optimizer'])
        torch.set_rng_state(last['torch_rng'])
        event('EPOCH_CHECKPOINT_RESUMED', arm=arm, seed=seed, last_committed_epoch=last['epoch'],
              note='Uncommitted partial epoch, if any, is discarded; no completed epoch or saved selected score repeated.')
    else:
        last = dict(context_sha256=sha(CONTEXT), arm=arm, seed=seed, epoch=0, step=0,
                    initial_state_sha256=init_hash, model=copy.deepcopy(m.state_dict()),
                    optimizer=copy.deepcopy(opt.state_dict()), torch_rng=torch.get_rng_state(),
                    best=None, history=[], elapsed_seconds=0.0)
        atomic_torch(last_path, last)
    elapsed_prior = last['elapsed_seconds']
    started = time.perf_counter()
    for epoch in range(last['epoch']+1, TRAIN['epochs_max']+1):
        if last['best'] and last['epoch'] - last['best']['epoch'] >= TRAIN['early_stopping_patience']:
            break
        assert elapsed_prior + time.perf_counter() - started < TRAIN['wall_safety_seconds_per_arm'], 'Safety budget exceeded; incomplete checkpoint preserved'
        m.train()
        order = np.random.default_rng(seed+epoch).permutation(len(train))
        assert len(set(order.tolist())) == 899
        loss_sum = 0.0; exposure = Counter()
        for at in range(0, len(order), TRAIN['batch_size']):
            rows = [train[i] for i in order[at:at+TRAIN['batch_size']]]
            exposure.update('/'.join(schedule(r['clip_id'],epoch,arm)) for r in rows)
            x,p,y = torch_batch(rows, epoch=epoch, arm=arm)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(m(x,p), y)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), TRAIN['gradient_clip_norm'])
            assert torch.isfinite(norm)
            opt.step(); last['step'] += 1
            loss_sum += float(loss.item()) * len(rows)
        assert sum(exposure.values()) == 899 and last['step'] == epoch * math.ceil(899/32)
        scores = {v: predict(m,dev,v) for v in VIEWS}
        mm = {v: detailed_metrics(scores[v],c['missingness_groups'][v]) for v in VIEWS}
        ce = (mm['masked']['mean_ce']+mm['sensor']['mean_ce'])/2
        assert math.isfinite(ce)
        if last['best'] is None or ce < last['best']['ce'] - TRAIN['strict_ce_improvement']:
            last['best'] = dict(epoch=epoch, step=last['step'], ce=ce,
                                model=copy.deepcopy(m.state_dict()), predictions=scores, metrics=mm)
        elapsed = elapsed_prior + time.perf_counter() - started
        h = dict(epoch=epoch, steps=last['step'], train_ce=loss_sum/899,
                 selected_rule_ce=ce, masked_ce=mm['masked']['mean_ce'], sensor_ce=mm['sensor']['mean_ce'],
                 masked_top1=mm['masked']['top1_correct'], sensor_top1=mm['sensor']['top1_correct'],
                 best_epoch=last['best']['epoch'], exposure=dict(exposure), elapsed_seconds=elapsed)
        last['history'].append(h)
        last.update(epoch=epoch, model=copy.deepcopy(m.state_dict()),
                    optimizer=copy.deepcopy(opt.state_dict()), torch_rng=torch.get_rng_state(), elapsed_seconds=elapsed)
        atomic_torch(last_path,last)
        atomic_json(U / (prefix+'_history.json'),last['history'])
        event('EPOCH_COMMITTED', arm=arm, seed=seed, **h)
        if elapsed > TRAIN['wall_safety_seconds_per_arm']:
            raise RuntimeError('Safety budget exceeded after committed epoch; run incomplete, no final score publication')
    assert last['epoch'] == TRAIN['epochs_max'] or last['epoch']-last['best']['epoch'] >= TRAIN['early_stopping_patience']
    assert last['elapsed_seconds'] <= TRAIN['wall_safety_seconds_per_arm'], 'Safety-stopped checkpoint remains incomplete'
    best = last['best']
    # The selected scores were already computed during epoch selection. Do not score again.
    ck = selected_checkpoint(best,c,arm,seed)
    if selected_path.exists():
        saved_ck = torch.load(selected_path,map_location='cpu',weights_only=True)
        assert saved_ck['training_context_sha256'] == sha(CONTEXT) and saved_ck['epoch'] == best['epoch']
        assert state_hash(saved_ck['state_dict']) == state_hash(best['model'])
    else:
        atomic_torch(selected_path,ck)
    result = dict(status='COMPLETE_DEVELOPMENT_ONLY', context_sha256=sha(CONTEXT), arm=arm,seed=seed,
                  initial_state_sha256=init_hash, checkpoint_sha256=sha(selected_path),
                  selected_epoch=best['epoch'], epochs_completed=last['epoch'], total_steps=last['step'],
                  selected_ce=best['ce'], seconds=last['elapsed_seconds'], parameter_count=37636,
                  metrics=best['metrics'], predictions=best['predictions'], selected_scores_reinferred=False,
                  final_blind_test=False, user_data_used=False, production_model_replaced=False)
    atomic_json(result_path,result,new=True)
    event('RUN_COMPLETE',arm=arm,seed=seed,selected_epoch=best['epoch'],epochs_completed=last['epoch'])
    return result


def compare(c, results, m0):
    path = U / 'training_COMPARISON.json'
    if path.exists():
        saved=read(path); assert saved['context_sha256'] == sha(CONTEXT)
        pins_ok(saved['result_pins']); return saved
    baseline={v:detailed_metrics(rows,c['missingness_groups']['masked' if v=='native' else 'sensor']) for v,rows in m0.items()}
    summaries={arm:{v:aggregate([results[(arm,s)]['metrics'][v] for s in SEEDS]) for v in VIEWS} for arm in ARMS}
    pairs={}
    for seed in SEEDS:
        assert len({results[(a,seed)]['initial_state_sha256'] for a in ARMS}) == 1
        pairs[str(seed)]={}
        for v in VIEWS:
            a,b,cc=(results[(arm,seed)]['predictions'][v] for arm in ARMS)
            pairs[str(seed)][v]=dict(A_to_B=paired(a,b),B_to_C=paired(b,cc),
                                     M0_pipeline_to_C=paired(m0['native' if v=='masked' else 'sensor'],cc))
    result_files=[U / f'training_{a}_{s}_RESULT.json' for a in ARMS for s in SEEDS]
    result_files += [U / 'training_m0_sensor.json']
    result=dict(status='COMPLETE_PENDING_INDEPENDENT_RECOUNT',context_sha256=sha(CONTEXT),
                arms=summaries,M0=baseline,paired=pairs,
                progression=progression_gate(summaries,baseline),
                result_pins={str(p):sha(p) for p in result_files},
                predefined_future_deployment_seed=SEEDS[0],
                final_blind_test=False, production_replaced=False,
                caveat='Same repeatedly used300 development clips select checkpoints and report results. Actual-crop view is not a real-webcam/new-user test. Small gains are not generalization proof.')
    atomic_json(path,result,new=True)
    return result


def process_live(pid):
    # Windows read-only PID liveness, used only to guard duplicate drivers.
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype=wintypes.BOOL
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        kernel.CloseHandle.restype=wintypes.BOOL
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            # Only ERROR_INVALID_PARAMETER establishes an absent process; access denied is live/unknown.
            return ctypes.get_last_error()!=87
        code=wintypes.DWORD()
        try:
            assert kernel.GetExitCodeProcess(handle,ctypes.byref(code))
            return code.value==259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid,0); return True
    except ProcessLookupError:
        return False


def run(args):
    assert CONTEXT.exists(), 'A separately approved ready context must be frozen before running'
    c=read(CONTEXT)
    assert c['model']==MODEL and c['training']==TRAIN and c['seeds']==list(SEEDS)
    assert c['runtime']==runtime(), 'Runtime changed; do not silently resume in a different environment'
    pins_ok(c['pins'])
    lock=U / 'training_RUNNING.json'
    if lock.exists():
        prev=read(lock)
        assert not process_live(prev['pid']), 'A driver is already alive, or PID status is uncertain'
        event('DEAD_PROCESS_LOCK_RECOVERED',previous=prev)
        lock.unlink()
    with lock.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),at_utc=utc(),context_sha256=sha(CONTEXT)),f)
    try:
        configure_torch()
        train,dev=load_data(c)
        m0=m0_scores(c,dev)
        results={}
        for seed in SEEDS:
            for arm in ARMS:
                results[(arm,seed)]=run_arm(c,train,dev,arm,seed)
        result=compare(c,results,m0)
        pins_ok(c['pins'])
        event('COMPARISON_COMPLETE_PENDING_INDEPENDENT_CHECK',progression=result['progression'])
    except BaseException as exc:
        failure=dict(at_utc=utc(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                     context_sha256=sha(CONTEXT),training_incomplete=True)
        first=U / 'training_FIRST_ERROR.json'
        if not first.exists(): atomic_json(first,failure,new=True)
        event('FAILED_OR_INTERRUPTED',error_type=failure['type'],error=failure['message'])
        raise
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid(): lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze',help='Validate hashes/readiness and freeze inputs; no inference/training')
    f.add_argument('--readiness',required=True)
    f.add_argument('--prep-context',required=True)
    f.add_argument('--records-dir',required=True)
    sub.add_parser('run',help='Run/recover all9 runs after frozen independent readiness')
    args=p.parse_args()
    if args.command=='freeze': freeze(args)
    else: run(args)


if __name__=='__main__': main()
