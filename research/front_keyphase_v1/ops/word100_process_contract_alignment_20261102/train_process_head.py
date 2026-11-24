"""One frozen candidate. No training occurs on import or without --execute-frozen."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
import os
import random
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from process_head import ProcessEvidenceNet, QUERIES, batch
from process_data import build_fit_dataset, multiplicities, process_identity, read, sha, SIDE

U = Path(__file__).resolve().parent
OUT = Path('W:/WordsInSilence_derived/word100_process_contract_alignment_20261102/process_run/seed42')
TRAINING_PROFILE = {
    'seed': 42, 'epochs': 80, 'candidates': 1, 'parameters': 47980,
    'optimizer': 'AdamW', 'learning_rate': .0005, 'weight_decay': .01,
    'gradient_clip_norm': 2, 'group_batch_size': 12, 'torch_threads': 4,
    'optimizer_steps': 320, 'fit_groups': 45,
    'normalization': 'Original FIT216 unique actual input RGB only, identical algorithm to AB; exclude runtime copies and DEV',
    'group_identity': ['person', 'source_sha256', 'physical_query'],
    'process_loss': 'Within each group: inverse physical-sequence multiplicity and class-balanced whole loss; same for separately justified nominal-scope targets; average the available two objectives',
    'point_loss': 'Inverse multiplicity of source-frame and relation-dimension, group-normalized; unknown mask0',
    'netmotion_loss': 'Inverse multiplicity of exact source-frame interval, group-normalized; unlabelled/null mask0',
    'loss_weights': {'process': 1., 'point': .25, 'netmotion': .25},
    'nominal_process_forward': 'Existing ProcessEvidenceNet.forward on only selected input rows; process_logits. Frozen RGB may carry outside context, disclosed separately.',
    'whole_and_nominal_targets_interchangeable': False,
    'existing_motion_branch_target': 'Observed net query movement over its exact reviewed interval, never whole-process completion',
    'word_person_edit_seam_time_as_neural_features': False,
    'checkpoint_selection': 'Final epoch80 only; no DEV optimization, stopping, thresholds or selection',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def weighted_mean(items, zero):
    if not items:
        return zero
    losses = [F.binary_cross_entropy_with_logits(logit, logit.new_tensor(float(label))) * weight
              for logit, label, weight in items]
    return torch.stack(losses).sum() / sum(weight for _, _, weight in items)


def balanced_mean(items, zero):
    present = [weighted_mean([t for t in items if t[1] == value], zero)
               for value in (0, 1) if any(t[1] == value for t in items)]
    return torch.stack(present).mean() if present else zero


def training_step(model, records, sequences, weights):
    """Calculate losses only. Caller owns optimizer; no metadata becomes input."""
    x, lengths = batch([sequences[r['array_key']] for r in records])
    query = torch.tensor([QUERIES.index(r['physical_query']) for r in records], dtype=torch.long)
    pred = model(x, lengths, query)
    positions = defaultdict(list)
    for i, r in enumerate(records):
        positions[r['group_key']].append(i)

    # One forward per distinct local selection in this batch. Whole-case spans
    # reuse the exact same forward instead of adding a duplicate context call.
    local = {}
    full_lookup = {(r['array_key'], tuple(range(len(sequences[r['array_key']]))), r['physical_query']): i
                   for i, r in enumerate(records)}
    for r in records:
        for t in r['nominal_process_targets']:
            if t['observed_process_evidence'] is None:
                continue
            key = (r['array_key'], tuple(t['output_indices']), r['physical_query'])
            if key not in full_lookup:
                local[key] = sequences[r['array_key']][t['output_indices']]
    local_keys = list(local)
    local_logits = {}
    if local_keys:
        lx, ll = batch([local[k] for k in local_keys])
        lq = torch.tensor([QUERIES.index(k[2]) for k in local_keys], dtype=torch.long)
        lp = model(lx, ll, lq)['process_logits']
        local_logits = {k: lp[i] for i, k in enumerate(local_keys)}

    # The old net-motion branch keeps its own labelled-interval meaning.
    motions = {}
    for r in records:
        for t in r['fully_retained_motion_interval_targets']:
            if t['label'] is not None:
                key = (r['array_key'], tuple(t['output_indices']), r['physical_query'])
                motions[key] = sequences[r['array_key']][t['output_indices']]
    motion_keys = list(motions)
    motion_logits = {}
    if motion_keys:
        mx, ml = batch([motions[k] for k in motion_keys])
        mq = torch.tensor([QUERIES.index(k[2]) for k in motion_keys], dtype=torch.long)
        mp = model.interval_logits(mx, ml, mq)
        motion_logits = {k: mp[i] for i, k in enumerate(motion_keys)}

    group_losses, values = [], []
    for g, ix in positions.items():
        zero = pred['process_logits'][ix].sum() * 0
        whole_items, nominal_items, point_items, motion_items = [], [], [], []
        wc = weights[g]
        for i in ix:
            r = records[i]
            label = r['observed_process_evidence']
            if label is not None:
                identity = process_identity(r)
                whole_items.append((pred['process_logits'][i], label, 1. / wc['whole'][identity]))
            for t in r['nominal_process_targets']:
                label = t['observed_process_evidence']
                if label is None:
                    continue
                key = (r['array_key'], tuple(t['output_indices']), r['physical_query'])
                logit = pred['process_logits'][full_lookup[key]] if key in full_lookup else local_logits[key]
                nominal_items.append((logit, label, 1. / wc['nominal'][process_identity(r, t['output_indices'])]))
            for t in r['sparse_relation_targets']:
                for dim, field in enumerate(SIDE):
                    if t[field] is None:
                        continue
                    assert r['axis'][t['output_index']]['image_available']
                    point_items.append((pred['point_logits'][i, t['output_index'], dim], t[field],
                                        1. / wc['point'][(t['source_frame'], dim)]))
            for t in r['fully_retained_motion_interval_targets']:
                if t['label'] is None:
                    continue
                key = (r['array_key'], tuple(t['output_indices']), r['physical_query'])
                motion_items.append((motion_logits[key], t['label'], 1. / wc['netmotion'][tuple(t['source_frames'])]))
        components = []
        if whole_items:
            components.append(balanced_mean(whole_items, zero))
        if nominal_items:
            components.append(balanced_mean(nominal_items, zero))
        process = torch.stack(components).mean() if components else zero
        point = weighted_mean(point_items, zero)
        motion = weighted_mean(motion_items, zero)
        group_losses.append(process + .25 * point + .25 * motion)
        values.append(dict(process=float(process.detach()), point=float(point.detach()), netmotion=float(motion.detach())))
    loss = torch.stack(group_losses).mean()
    assert torch.isfinite(loss)
    return loss, values


def main(contract_path):
    contract_path = Path(contract_path).resolve()
    assert not OUT.exists(), 'No silent restart of completed/interrupted candidate'
    contract = read(contract_path)
    assert contract['status'] == 'READY_FOR_ONE_FIXED_TRAINING_RUN' and contract['training_allowed'] is True
    assert contract['seed'] == 42 and contract['epochs'] == 80 and contract['candidates'] == 1
    assert contract['training_profile'] == TRAINING_PROFILE
    assert Path(contract['output_directory']).resolve() == OUT.resolve()
    for path, expected in contract['pins'].items():
        assert sha(path) == expected, path
    for path in [Path(__file__), U / 'process_head.py', U / 'process_data.py']:
        assert contract['pins'][str(path.resolve())] == sha(path)
    span = contract['nominal_span_manifest']
    assert span['required'] is True and sha(span['path']) == span['sha256']
    data = build_fit_dataset(span['path'], require_spans=True)
    for path, expected in data['pins'].items():
        assert contract['pins'].get(path) == expected, f'Unfrozen data dependency: {path}'
    assert data['development_arrays_opened'] == 0 and data['prediction_fields_loaded'] == []
    weights = multiplicities(data['records'])
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = ProcessEvidenceNet()
    assert sum(p.numel() for p in model.parameters()) == 47980
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'STARTED.json', dict(at=now(), pid=os.getpid(), contract_sha256=sha(contract_path), training_profile=TRAINING_PROFILE))
    write(OUT / 'NORMALIZATION.json', data['stats'])
    write(OUT / 'DATA_CONSUMPTION.json', dict(counts=data['counts'], nominal_span_status=data['nominal_span_status'],
          sensor_fields=data['sensor_fields_loaded'], prediction_fields_loaded=[], development_arrays_opened=0,
          pins=data['pins'], source_person_query_groups=len(data['groups']),
          nominal_positive_targets_not_copied_from_whole_view=True,
          reviewed_absence_containment_must_be_explicit=True, metadata_used_as_features=False))
    groups = data['groups']
    keys = sorted(groups)
    history, steps = [], 0
    start = time.perf_counter()
    for epoch in range(80):
        random.shuffle(keys)
        model.train()
        losses, component_values = [], []
        for offset in range(0, len(keys), 12):
            batch_records = [r for key in keys[offset:offset + 12] for r in groups[key]]
            loss, components = training_step(model, batch_records, data['sequences'], weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
            component_values.extend(components)
        history.append(dict(epoch=epoch + 1, training_loss=float(np.mean(losses)),
                            **{k: float(np.mean([v[k] for v in component_values])) for k in ('process', 'point', 'netmotion')}))
        write(U / 'TRAIN_PROGRESS.json', dict(at=now(), status='RUNNING', pid=os.getpid(),
              epoch=epoch + 1, total_epochs=80, optimizer_steps=steps, elapsed_seconds=time.perf_counter() - start,
              last_loss=history[-1]['training_loss']))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print('alignment training', epoch + 1, history[-1]['training_loss'], flush=True)
    assert steps == 320
    for path, expected in contract['pins'].items():
        assert sha(path) == expected, path
    torch.save(model.state_dict(), OUT / 'model.pt')
    write(OUT / 'HISTORY.json', history)
    write(U / 'TRAIN_RESULT.json', dict(at=now(), status='FIXED_CANDIDATE_TRAINED_NOT_EVALUATED',
          checkpoint=str(OUT / 'model.pt'), checkpoint_sha256=sha(OUT / 'model.pt'),
          normalization=str(OUT / 'NORMALIZATION.json'), normalization_sha256=sha(OUT / 'NORMALIZATION.json'),
          contract_sha256=sha(contract_path), parameters=47980, epochs=80, seed=42, candidates=1,
          optimizer_steps=steps, seconds=time.perf_counter() - start, history=history,
          fit_counts=data['counts'], nominal_span_status=data['nominal_span_status'],
          old_weights_changed=False, development_used_for_optimization=False,
          unknown_as_negative=False, positive_whole_target_arbitrarily_copied_to_nominal=False,
          reviewed_absence_containment_explicit=True,
          new_accuracy_claim=False, whole_goal_complete=False))
    write(U / 'TRAIN_PROGRESS.json', dict(at=now(), status='COMPLETE_NOT_EVALUATED', active=None,
                                        epoch=80, total_epochs=80, optimizer_steps=steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', default=str(U / 'ALIGNMENT_FROZEN.json'))
    parser.add_argument('--execute-frozen', action='store_true')
    args = parser.parse_args()
    assert args.execute_frozen, 'Training requires explicit --execute-frozen and a validated root freeze'
    try:
        main(args.contract)
    except BaseException:
        p = U / 'TRAIN_FIRST_FAILURE.txt'
        if not p.exists():
            p.write_text(traceback.format_exc(), encoding='utf-8')
        raise
