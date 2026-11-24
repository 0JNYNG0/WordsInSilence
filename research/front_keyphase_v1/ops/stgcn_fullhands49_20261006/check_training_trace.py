"""Independent, read-only training-record audit; never imports a model or trains.

Only this unit's TRAINING_TRACE_AUDIT.json is written. In-progress history is a
snapshot, not a completion claim. Run again after the training process finishes.
"""
from pathlib import Path
import datetime
import hashlib
import io
import json
import math
import os

import numpy as np

U = Path(__file__).resolve().parent
W = U.parent.parent
R = W / 'runs/stgcn_fullhands49_v1'
SCHEDULE_U = U.parent / 'stgcn_execution_ready_20261005'
OLD_U = U.parent / 'stgcn_valid_window_pair_20261006'
OLD_R = W / 'runs/stgcn_valid_window_v1'
SCHEDULE_SHA = '4756eaba49b13def1531ddbef139630acb0a116cae5f1a3972ff5369e50ddaba'


class PendingSnapshot(Exception):
    """A producer has not finished publishing an atomic file/companion pair."""


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_json(path, expected=None):
    raw = path.read_bytes()
    if expected is not None:
        need(sha(raw) == expected, f'Pinned file mismatch: {path}')
    return json.loads(raw), raw


def signed_bytes(path, context_sha, run_sha=None):
    companion = path.with_name(path.name + '.meta.json')
    if not companion.is_file():
        raise PendingSnapshot(f'Companion not yet present: {companion.name}')
    meta, _ = read_json(companion)
    raw = path.read_bytes()
    need(meta['file'] == path.name, f'Companion filename mismatch: {path.name}')
    need(meta['context_sha256'] == context_sha, f'Companion context mismatch: {path.name}')
    if run_sha is not None:
        need(meta['run_sha256'] == run_sha, f'Companion run mismatch: {path.name}')
    if meta['sha256'] != sha(raw) or meta['size'] != len(raw):
        raise PendingSnapshot(f'File/companion mismatch; retry after publication: {path.name}')
    return raw


def bound_json(name, context_sha, run_sha=None):
    value = json.loads(signed_bytes(R / name, context_sha, run_sha))
    need(value['context_sha256'] == context_sha, f'Record context mismatch: {name}')
    if run_sha is not None:
        need(value['run_sha256'] == run_sha, f'Record run mismatch: {name}')
    return value


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def pinned_json(path, library_pins):
    """Require the record itself to be a frozen execution-context dependency."""
    matches = [expected for name, expected in library_pins.items()
               if Path(name).resolve() == path.resolve()]
    need(len(matches) == 1, f'Required context pin missing or duplicated: {path}')
    return read_json(path, matches[0])


def audit_history(stage, history, schedule, cfg, complete):
    expected = schedule['stages'][stage]
    need(isinstance(history, list) and history, f'{stage}: no completed epoch records')
    need(len(history) <= cfg['max_epochs'], f'{stage}: exceeded epoch budget')
    best_key, best_epoch, stop_epoch = None, None, None
    for index, h in enumerate(history):
        epoch = index + 1
        e = expected['epochs'][index]
        need(h['epoch'] == epoch == e['epoch'], f'{stage}: epoch order')
        need(h['epoch_seed'] == (cfg['seed'] + epoch) % (2 ** 32), f'{stage}: epoch seed')
        need(h['sample_sha256'] == e['sample_sha256'], f'{stage}/{epoch}: sample order hash')
        raw_counts = h['sample_counts']
        need(len(raw_counts) == expected['row_count'], f'{stage}/{epoch}: row count')
        need(all(type(v) is int and v >= 0 for v in raw_counts), f'{stage}/{epoch}: invalid counts')
        counts = np.asarray(raw_counts, dtype='<i8')
        need(sha(counts.tobytes()) == e['row_counts_sha256'], f'{stage}/{epoch}: row frequency hash')
        count = expected['draws_per_epoch']
        need(int(counts.sum()) == count == h['train_count'], f'{stage}/{epoch}: sample total')
        need(h['updates'] == count // cfg['batch_size'], f'{stage}/{epoch}: updates')
        need(h['grad_missing'] == 0 and h['grad_finite_checks'] == 108 * h['updates'],
             f'{stage}/{epoch}: all 108 gradient records')
        need(finite_number(h['train_ce']) and h['train_ce'] >= 0, f'{stage}/{epoch}: loss')
        need(type(h['train_correct']) is int and 0 <= h['train_correct'] <= count,
             f'{stage}/{epoch}: train correct count')
        need(h['train_accuracy'] == h['train_correct'] / count, f'{stage}/{epoch}: train accuracy')
        wanted_lr = .5 * cfg['base_lr'] * (1 + math.cos(math.pi * index / cfg['schedule_epochs']))
        need(h['lr'] == wanted_lr, f'{stage}/{epoch}: learning rate')
        if stage == 'select':
            val = h['val']
            need(val['primary_count'] == 233 and val['all_count'] == 257,
                 f'select/{epoch}: validation denominator')
            correct, ce = val['primary_correct'], val['primary_ce']
            need(type(correct) is int and 0 <= correct <= 233 and finite_number(ce) and ce >= 0,
                 f'select/{epoch}: validation values')
            need(val['primary_accuracy'] == correct / 233, f'select/{epoch}: validation accuracy')
            key = (-correct, ce)
            improved = best_key is None or key < best_key
            if improved:
                best_key, best_epoch = key, epoch
            need(h['is_best'] is improved, f'select/{epoch}: selection marker')
            need(h['stale'] == epoch - best_epoch, f'select/{epoch}: patience count')
            if epoch >= cfg['min_epochs'] and h['stale'] >= cfg['patience']:
                if stop_epoch is None:
                    stop_epoch = epoch
            need(stop_epoch is None or epoch == stop_epoch, 'Selection continued after its stopping rule')
        else:
            need(h['val'] is None and h['stale'] is None and h['is_best'] is False,
                 f'refit/{epoch}: unexpected validation or selection')
    if complete and stage == 'select':
        need(cfg['min_epochs'] <= len(history) <= cfg['max_epochs'], 'Selection termination budget')
        need(stop_epoch == len(history) or len(history) == cfg['max_epochs'], 'Premature selection stop')
    return {'stage': stage, 'completed_epochs': len(history), 'selected_epoch': best_epoch,
            'patience_stop_epoch': stop_epoch, 'sampling_gradients_lr_selection': 'PASS'}


def audit():
    plan, plan_raw = read_json(U / 'FROZEN_EXPERIMENT.json')
    schedule, _ = read_json(SCHEDULE_U / 'SCHEDULE_AUDIT.json', SCHEDULE_SHA)
    old_plan, _ = read_json(OLD_U / 'FROZEN_EXPERIMENT.json', plan['source_plan_sha256'])
    cfg = plan['training_config']
    need(cfg == old_plan['training_config'] == schedule['config'], 'Changed training configuration')
    need(plan['refit_num_samples'] == old_plan['refit_num_samples'] == 2352, 'Changed refit exposure')
    context_path = U / 'EXECUTION_CONTEXT.json'
    if not context_path.is_file():
        raise PendingSnapshot('EXECUTION_CONTEXT.json not yet published')
    context, _ = read_json(context_path)
    context_sha = sha(canonical(context).encode('utf-8'))
    need(isinstance(context.get('library_pins'), dict) and context['library_pins'], 'Missing library pins')
    for path, expected in context['library_pins'].items():
        need(sha(Path(path).read_bytes()) == expected, f'Library pin mismatch: {path}')
    pinned_json(U / 'FROZEN_EXPERIMENT.json', context['library_pins'])
    transfer, transfer_raw = pinned_json(U / 'GRAPH_TRANSFER_CHECK.json', context['library_pins'])
    need(transfer['status'] == 'PASS', 'Graph-transfer independent audit did not pass')
    initial_digest = transfer['actual_initial_state_digest']
    need(isinstance(initial_digest, str) and len(initial_digest) == 64
         and all(ch in '0123456789abcdef' for ch in initial_digest),
         'Invalid audited 49-point initial-state digest')
    prepared, prepared_raw = pinned_json(U / 'prepared/TRAINING_META.json', context['library_pins'])
    need(prepared['shape'] == [2346, 6, 2, 10, 49], 'Prepared training shape is not the 49-point contract')
    need(prepared['classes'] == context['classes'] and len(prepared['classes']) == cfg['num_classes'],
         'Prepared class order differs from execution context')
    need(prepared['cache_sha256'] == context['prepared_cache_sha256'], 'Prepared cache context mismatch')
    need(context['inference_shape'] == [6, 2, 10, 49], 'Execution context uses another graph size')
    if not (R / 'INIT_SELECT.json').is_file():
        raise PendingSnapshot('New selection has not initialized')
    initial = bound_json('INIT_SELECT.json', context_sha)
    run_sha = initial['run_sha256']
    need(initial['state_digest'] == initial_digest, 'Selection differs from the independently audited 49-point initial state')
    checks = [{'context_library_pins': 'PASS', 'old_schedule_sha256': SCHEDULE_SHA,
               'initial_state_matches_49_point_transfer_audit': True,
               'graph_transfer_check_sha256': sha(transfer_raw),
               'actual_initial_state_digest': initial_digest,
               'training_meta_sha256': sha(prepared_raw), 'prepared_training_shape': prepared['shape']}]
    if (R / 'INIT_REFIT.json').is_file():
        refit_init = bound_json('INIT_REFIT.json', context_sha, run_sha)
        need(refit_init['state_digest'] == initial['state_digest'], 'Refit is not recorded from fresh initial state')
        checks.append({'recorded_fresh_refit_initial_state': 'PASS'})
    records = {}
    for stage, name in [('select', 'SELECT_RESULT.json'), ('refit', 'REFIT_RESULT.json')]:
        if (R / name).is_file():
            records[stage] = bound_json(name, context_sha, run_sha)
    progress = None
    if (R / 'progress.json').is_file():
        progress, _ = read_json(R / 'progress.json')
        need(progress['context_sha256'] == context_sha and progress['run_sha256'] == run_sha,
             'Progress belongs to another execution')
        need(progress['phase'] in ('select', 'refit'), 'Unknown progress phase')
        need(progress['epoch'] == len(progress['history']), 'Progress epoch/history mismatch')
        need(progress['record'] == progress['history'][-1], 'Progress last record mismatch')
    for stage in ('select', 'refit'):
        if stage in records:
            history, complete = records[stage]['history'], True
        elif progress is not None and progress['phase'] == stage:
            history, complete = progress['history'], False
        else:
            continue
        check = audit_history(stage, history, schedule, cfg, complete)
        checks.append(check)
        if stage == 'select' and complete:
            selected = records['select']
            need(selected['best_epoch'] == check['selected_epoch'], 'Selected E* mismatch')
            stop = selected['stop_reason']
            if check['patience_stop_epoch'] is not None:
                need(stop == 'patience', 'Wrong patience stop reason')
            else:
                wanted_stop = 'budget_limited_improving' if history[-1]['is_best'] else 'max_epochs'
                need(stop == wanted_stop, 'Wrong budget stop reason')
            best_raw = signed_bytes(R / 'select_best.pt', context_sha, run_sha)
            need(sha(best_raw) == selected['best_checkpoint_sha256'], 'Selected checkpoint hash')
        elif stage == 'refit':
            need('select' in records and (R / 'INIT_REFIT.json').is_file(), 'Refit before completed selection/fresh init')
            need(len(history) <= records['select']['best_epoch'], 'Refit exceeded selected E*')
    finalized = 'refit' in records
    if finalized:
        refit = records['refit']
        need(refit['selected_epoch'] == records['select']['best_epoch'] == len(refit['history']), 'Refit length != E*')
        need(refit['stop_reason'] == 'fixed_epochs', 'Refit stopping rule changed')
        frozen_raw = signed_bytes(R / 'FROZEN_MODEL.json', context_sha, run_sha)
        frozen = json.loads(frozen_raw)
        need(sha(frozen_raw) == refit['frozen_sha256'], 'Frozen metadata hash')
        need(frozen['context'] == context and frozen['context_sha256'] == context_sha
             and frozen['run_sha256'] == run_sha, 'Frozen execution context')
        need(frozen['selected_epoch'] == refit['selected_epoch'], 'Frozen E*')
        need(frozen['inference_shape'] == context['inference_shape'], 'Frozen inference shape changed')
        need(frozen['final_model_file'] == 'final_model.npz', 'Unexpected model filename')
        model_raw = signed_bytes(R / 'final_model.npz', context_sha, run_sha)
        need(sha(model_raw) == frozen['final_model_sha256'] == refit['final_model_sha256'], 'Final model hash')
        old_frozen, _ = read_json(OLD_R / 'FROZEN_MODEL.json')
        need(frozen['state_names'] == old_frozen['state_names'], 'Model state names changed')
        need(frozen['classes'] == old_frozen['classes'] == context['classes'], 'Class order changed')
        names, digest = frozen['state_names'], hashlib.sha256()
        with np.load(io.BytesIO(model_raw), allow_pickle=False) as npz:
            need(set(npz.files) == {f'arr_{i}' for i in range(len(names))}, 'NPZ state count')
            state = {name: np.ascontiguousarray(npz[f'arr_{i}']) for i, name in enumerate(names)}
            for name in sorted(state):
                array = state[name]
                need(array.dtype.kind in 'fiu' and np.isfinite(array).all(), f'Invalid final state: {name}')
                digest.update(canonical([name, array.dtype.str, list(array.shape)]).encode('utf-8'))
                digest.update(array.tobytes())
        need(digest.hexdigest() == frozen['state_digest'], 'Final state digest')
        last_raw = signed_bytes(R / 'refit_last.pt', context_sha, run_sha)
        need(sha(last_raw) == refit['refit_last_sha256'], 'Refit last checkpoint hash')
        checks.append({'refit_epochs_equal_selected': True, 'final_hash_and_state_digest': 'PASS',
                       'unchanged_state_names_and_class_order': True})
    if not any('completed_epochs' in c for c in checks):
        raise PendingSnapshot('Initialized, but no completed epoch is available to audit yet')
    return {'status': 'PASS', 'finalized': finalized, 'checks': checks,
            'context_sha256': context_sha, 'run_sha256': run_sha,
            'plan_sha256': sha(plan_raw), 'read_only_training_artifacts': True,
            'model_or_training_execution': False,
            'scope': 'Saved records, independent arithmetic and file/array hashes; no new gradient execution. '
                     'The separately pinned graph-transfer audit supplies the 49-point initial digest; '
                     'this checker does not repeat transfer, model or input preparation tests.'}


def main():
    try:
        result, exit_code = audit(), 0
    except PendingSnapshot as exc:
        result = {'status': 'PENDING_SNAPSHOT', 'finalized': False, 'reason': str(exc)}
        exit_code = 2
    except Exception as exc:
        result = {'status': 'FAIL', 'finalized': False, 'error_type': type(exc).__name__, 'reason': str(exc)}
        exit_code = 1
    result['checked_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['run_dir'] = str(R)
    result['auditor_sha256'] = sha(Path(__file__).read_bytes())
    destination = U / 'TRAINING_TRACE_AUDIT.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
