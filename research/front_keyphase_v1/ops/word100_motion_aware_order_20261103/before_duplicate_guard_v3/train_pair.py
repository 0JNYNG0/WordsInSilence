"""One frozen-budget matched comparison; no old training/evaluation entrypoint."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn
import motion_order_model as mm

U = Path(__file__).resolve().parent
EPOCHS = 80
SOURCES_PER_BATCH = 6


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def verify_pins(pins):
    for path, expected in pins.items():
        assert mm.sha(path) == expected, 'Pinned file changed: ' + path


def update_progress(**updates):
    """Only a live status file is replaced; markers/checkpoints stay immutable."""
    path = U / 'TRAIN_PROGRESS.json'
    previous = mm.read(path) if path.exists() else {}
    previous.update(updates)
    previous.update(pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat())
    temporary = U / 'TRAIN_PROGRESS.json.tmp'
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(previous, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def assert_not_started():
    assert not (U / 'TRAIN_STARTED.json').exists() and not (U / 'runs').exists(), 'Do not repeat training'


def source_groups(rows):
    groups = {}
    for row in rows:
        assert int(row['person']) in range(1, 10), 'FIT only'
        assert row['source_group_id'] == row['clip_id']
        groups.setdefault(row['source_group_id'], []).append(row)
    assert len(rows) == 72 and len(groups) == 36
    for group in groups.values():
        assert len(group) == 2 and {r['kind'] for r in group} == {'original', 'actual_reversed_pixels'}
        group.sort(key=lambda r: 0 if r['kind'] == 'original' else 1)
    return groups


def class_weights():
    # Reuse the already frozen Y FIT252 weights; do not refit on DEV or
    # change weights when reducing the matched training inputs to FIT72.
    report = mm.read(mm.Y / 'TRAINING_RESULT.json')
    pos = np.asarray(report['positive_counts'], dtype=np.float64)
    neg = np.asarray(report['negative_counts'], dtype=np.float64)
    assert pos.shape == neg.shape == (8,)
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), .25, 4), dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Requires root freeze and independent code approval')
    args = parser.parse_args()
    if not args.run:
        parser.error('No implicit training; use --run only after the frozen review gate')
    assert_not_started()
    freeze = mm.read(U / 'FROZEN.json')
    review = mm.read(U / 'CODE_REVIEW.json')
    assert review['approved_for_training'] is True
    verify_pins(freeze['pins'])
    verify_pins(review['code_pins'])
    preflight = mm.read(U / 'REAL_PREFLIGHT.json')
    assert preflight['status'] == 'PASS'
    verify_pins(preflight['code_pins'])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    from input_data import load_rows
    rows = load_rows('fit')
    groups = source_groups(rows)
    keys = sorted(groups)
    assert EPOCHS * math.ceil(len(keys) / SOURCES_PER_BATCH) == 480
    known = sum(q['target'] >= 0 for r in rows for q in r['order_queries'])
    assert known == 180, 'Fixed original/reverse question denominator differs'
    old = mm.legacy()
    stats = mm.read(old.D / 'NORMALIZATION.json')
    flow_normal = mm.read(old.FLOW_UNIT / 'FLOW_INPUTS.json')
    weights = class_weights()
    torch.manual_seed(42)
    start_model = mm.MotionOrderNet(mm.CONDITIONS[0])
    start_state = {n: v.detach().cpu().clone() for n, v in start_model.state_dict().items()}
    frozen_state = start_model.frozen_state()
    write_new(U / 'TRAIN_STARTED.json', dict(at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), conditions=list(mm.CONDITIONS), rows=72, source_groups=36, known_pair_questions=known, epochs=80, steps_per_arm=480, seed=42, positive_weights=weights.tolist()))
    update_progress(status='TRAINING', condition=None, epoch=0, optimizer_steps=0,
                    total_optimizer_steps=0, expected_steps_per_arm=480, expected_total_steps=960,
                    training_marker_sha256=mm.sha(U / 'TRAIN_STARTED.json'))
    reports = []
    for arm_index, condition in enumerate(mm.CONDITIONS):
        torch.manual_seed(42)
        net = mm.MotionOrderNet(condition)
        net.load_state_dict(start_state, strict=True)
        net.assert_frozen(frozen_state)
        rng = np.random.default_rng(42)
        dest = U / 'runs' / condition
        dest.mkdir(parents=True, exist_ok=False)
        parameters = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.0005, weight_decay=.01)
        history = []
        shuffle = []
        steps = 0
        started = time.perf_counter()
        update_progress(status='TRAINING', condition=condition, epoch=0, optimizer_steps=0,
                        total_optimizer_steps=arm_index*480)
        for epoch in range(EPOCHS):
            net.train()
            order = rng.permutation(len(keys)).tolist()
            shuffle.append(order)
            losses = []
            for offset in range(0, len(order), SOURCES_PER_BATCH):
                selected = [keys[k] for k in order[offset:offset+SOURCES_PER_BATCH]]
                batch = [row for key in selected for row in groups[key]]
                x, rgb, lengths, _, _ = old.batch_j(batch, stats)
                flow = old.flow_batch(batch, flow_normal)
                auxiliary = mm.auxiliary_batch(batch)
                action, role, occurrence = net(x, rgb, lengths, flow, auxiliary)
                targets = old.occurrence_targets(batch, x.shape[1])
                cl = mm.occurrence_group_loss(occurrence, targets, weights, [r['source_group_id'] for r in batch])
                pl, _ = mm.pair_group_loss(action.softmax(-1), occurrence.sigmoid(), batch)
                loss = cl + .25 * pl
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 2, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                update_progress(status='TRAINING', condition=condition, epoch=epoch+1,
                                optimizer_steps=steps, total_optimizer_steps=arm_index*480+steps)
                losses.append([float(loss.detach()), float(cl.detach()), float(pl.detach())])
            net.assert_frozen(frozen_state)
            item = dict(epoch=epoch+1, steps=steps, mean_loss=np.mean(losses, axis=0).tolist(), elapsed_seconds=time.perf_counter()-started)
            history.append(item)
            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(json.dumps(dict(condition=condition, **item)), flush=True)
        assert steps == 480
        net.assert_frozen(frozen_state)
        changes = net.validate_trained(start_state, frozen_state)
        model = dest / 'model.pt'
        torch.save(net.state_dict(), model)
        report = dict(condition=condition, model=str(model), path=str(model), sha256=mm.sha(model), checkpoint_sha256=mm.sha(model), steps=steps, epochs=EPOCHS, seed=42, seconds=time.perf_counter()-started, trainable_parameters=1792, frozen_parameters_and_buffers_exact=True, changes=changes, history=history, source_shuffle=shuffle)
        write_new(dest / 'COMPLETE.json', report)
        update_progress(status='ARM_COMPLETE', condition=condition, epoch=80, optimizer_steps=480,
                        total_optimizer_steps=(arm_index+1)*480, last_complete_checkpoint_sha256=mm.sha(model))
        reports.append(report)
    assert reports[0]['source_shuffle'] == reports[1]['source_shuffle']
    verify_pins(freeze['pins'])
    verify_pins(review['code_pins'])
    write_new(U / 'TRAIN_RESULT.json', dict(status='TWO_MATCHED_FINAL_ARMS_COMPLETE', runs=reports, initial_y=dict(path=str(mm.INITIAL_Y), sha256=mm.initial_y_sha()), source_groups=36, input_variants=72, positive_weights=weights.tolist(), class_weight_source='Existing Y FIT252 fixed counts', known_pair_questions=180, same_initial_state=True, same_source_shuffle=True, new_AH_training=0, old_Y_training_reexecuted=False, raw_rank_is_not_confirmed_order=True, whole_sign_correctness=None, whole_goal_complete=False))
    update_progress(status='COMPLETE', condition=None, epoch=80, optimizer_steps=480,
                    total_optimizer_steps=960, result_sha256=mm.sha(U/'TRAIN_RESULT.json'))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure_text = traceback.format_exc()
        failure = U / 'TRAIN_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(failure_text, encoding='utf-8')
        if (U / 'TRAIN_STARTED.json').exists():
            try:
                update_progress(status='FAILED', first_failure_path=str(failure),
                                first_failure_sha256=mm.sha(failure))
            except BaseException:
                # Do not hide the original training failure if the disk or
                # live status writer is itself unavailable.
                pass
        raise
