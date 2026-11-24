"""Independent saved-array change audit; never imports or calls HandAssigner."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parent
PALM = [0, 5, 9, 13, 17]


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def intervals(mask, ts):
    ids = np.flatnonzero(mask)
    result = []
    if not len(ids):
        return result
    edges = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    cadence = int(np.median(np.diff(ts)))
    for run in edges:
        lo, hi = int(run[0]), int(run[-1])
        end = int(ts[hi + 1]) if hi + 1 < len(ts) else int(ts[hi]) + cadence
        result.append(dict(first_frame=lo, last_frame=hi, frames=len(run),
            start_seconds=float(ts[lo]/1e6), end_exclusive_seconds=end/1e6,
            duration_seconds=float((end-ts[lo])/1e6)))
    return result


def inspect_cost(trace, i, gate, ratio, name):
    cost = np.array([[np.inf if v is None else v for v in row]
        for row in trace[name]], dtype=float)
    clear = []
    for side in [0, 1]:
        value = cost[i, side]
        rivals = [cost[i, 1-side]] + [cost[j, side] for j in range(len(cost)) if j != i]
        if value <= gate and all(not np.isfinite(v) or value < ratio*v for v in rivals):
            clear.append(side)
    return clear


def run():
    result = read(U/'REPLAY_RESULT.json')
    capture = read(U/'RAW_CAPTURE_CONTEXT.json')
    prior = read(U.parent/'word100_upperbody_preflight_20261014/PLAN.json')
    item_by_id = {r['clip_id']: r for r in prior['items']}
    scope = read(U/'SCOPE.json')
    params = scope['parameters']
    pins = dict(capture['pins'])
    for row in result['rows']:
        pins.update(row['outputs'])
    for row in read(U/'RAW_CAPTURE_RESULT.json')['rows']:
        pins.update(row['output_sha256'])
    for path in [Path(__file__), U/'REPLAY_RESULT.json', U/'RAW_CAPTURE_RESULT.json']:
        pins[str(path.resolve())] = digest(path)
    for p, expected in pins.items():
        assert digest(p) == expected, p
    all_changes, gap_rows = [], []
    actual_counts = Counter()
    for row in result['rows']:
        cid, mode = row['clip_id'], row['mode']
        rd, cd = U/'raw_capture'/cid/mode, U/'replay'/cid/mode
        raw, old, new = load(rd/'raw.npz'), load(rd/'baseline.npz'), load(cd/'candidate.npz')
        sides = load(cd/'assignment.npz')['sides']
        trace = read(cd/'trace.json')
        declared = read(cd/'changes.json')
        a, b = item_by_id[cid]['core_seconds']
        actual = []
        dropped_side = np.zeros((len(raw['raw_count']), 2), bool)
        for f in range(len(raw['raw_count'])):
            for i in range(int(raw['raw_count'][f])):
                old_side, new_side = int(raw['baseline_side'][f, i]), int(sides[f, i])
                if old_side == new_side:
                    actual_counts['unchanged'] += 1
                    continue
                category = 'newly_assigned' if old_side < 0 else 'dropped' if new_side < 0 else 'side_changed'
                actual_counts[category] += 1
                t = float(raw['timestamp_us'][f]/1e6)
                points = raw['hand_xy_px'][f, i]
                w, h = int(raw['width'][f]), int(raw['height'][f])
                finite = np.isfinite(points).all(1)
                inside = finite & (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
                score = float(raw['raw_score'][f, i])
                old_usable = old['usable'][f, 33+21*old_side:54+21*old_side] if old_side >= 0 else np.zeros(21, bool)
                pose_clear = inspect_cost(trace[f], i, params['pose_gate'], params['ambiguity_ratio'], 'pose_distance')
                track_clear = inspect_cost(trace[f], i, params['track_gate'], params['ambiguity_ratio'], 'track_distance')
                label = int(raw['raw_label'][f, i])
                if not finite.all():
                    diagnosis = 'nonfinite_hand_points'
                elif not inside[PALM].all():
                    diagnosis = 'one_or_more_palm5_points_outside_strict_image'
                elif not np.isfinite(score) or score < .5:
                    diagnosis = 'score_below_0_5_or_nonfinite'
                elif track_clear and pose_clear and track_clear[0] != pose_clear[0] and score >= params['label_min_score'] and label == pose_clear[0]:
                    diagnosis = 'clear_temporal_vs_current_pose_label_conflict'
                elif pose_clear and score >= params['label_min_score'] and label != pose_clear[0]:
                    diagnosis = 'clear_pose_vs_high_score_raw_label_conflict'
                elif not pose_clear and not track_clear and score < params['label_min_score']:
                    diagnosis = 'no_clear_association_and_label_below_0_8'
                else:
                    diagnosis = 'other_association_ambiguity_or_one_to_one_competition'
                change = dict(clip_id=cid, label=row['label'], mode=mode, frame=f,
                    original_frame_id=int(raw['frame_id'][f]), raw_index=i, timestamp_seconds=t,
                    category=category, old_side=old_side, new_side=new_side,
                    raw_label=label, raw_label_name=str(raw['raw_label_name'][f, i]), raw_score=score,
                    annotation_seconds=[a,b], phase='before' if t < a else 'after' if t > b else 'core',
                    finite_hand_points=int(finite.sum()), strict_inside_points=int(inside.sum()),
                    strict_palm5_inside_points=int(inside[PALM].sum()),
                    original_usable_points=int(old_usable.sum()), original_any_usable=bool(old_usable.any()),
                    original_all21_usable=bool(old_usable.all()),
                    pose_wrists_xy=raw['pose_xy_px'][f, [15,16]], pose_wrists_usable=raw['pose_usable'][f, [15,16]],
                    clear_pose_sides=pose_clear, clear_track_sides=track_clear,
                    saved_pose_distance=trace[f]['pose_distance'][i], saved_track_distance=trace[f]['track_distance'][i],
                    raw_palm5_xy=points[PALM], candidate_reason=trace[f]['reasons'][i], diagnostic_category=diagnosis,
                    candidate_any_hand_observed=bool(new['observed'][f,33:].any()),
                    candidate_any_hand_usable=bool(new['usable'][f,33:].any()),
                    potential_inside_high_score_abstention=bool(category=='dropped' and inside.all() and score>=.8))
                if old_side >= 0:
                    other = 1-old_side
                    change['other_hand_observed_after'] = bool(new['observed'][f,33+21*other:54+21*other].any())
                    change['other_hand_usable_after'] = bool(new['usable'][f,33+21*other:54+21*other].any())
                    if category == 'dropped':
                        dropped_side[f, old_side] = True
                actual.append((f, i, old_side, new_side, category))
                all_changes.append(change)
        assert actual == [(c['frame'], c['raw_index'], c['old_side'], c['new_side'], c['category']) for c in declared]
        ts = raw['timestamp_us']
        entry = dict(clip_id=cid, label=row['label'], mode=mode, per_side=[])
        for side in [0,1]:
            sl = slice(33+21*side,54+21*side)
            new_missing = old['usable'][:,sl].any(1) & ~new['usable'][:,sl].any(1)
            drop_missing = new_missing & dropped_side[:,side]
            runs = intervals(drop_missing, ts)
            for interval in runs:
                f0, f1 = interval['first_frame'], interval['last_frame']+1
                other = 1-side
                osl = slice(33+21*other,54+21*other)
                interval['other_hand_observed_every_frame'] = bool(new['observed'][f0:f1,osl].any(1).all())
                interval['other_hand_usable_every_frame'] = bool(new['usable'][f0:f1,osl].any(1).all())
            entry['per_side'].append(dict(side=side,new_missing_due_to_dropped_detection_frames=int(drop_missing.sum()),
                dropped_new_missing_runs=runs,any_change_new_missing_runs=intervals(new_missing,ts)))
        new_both_absent = old['usable'][:,33:].any(1) & ~new['usable'][:,33:].any(1)
        entry['new_both_hands_unusable_runs'] = intervals(new_both_absent,ts)
        gap_rows.append(entry)
    for key in ['unchanged','dropped','side_changed','newly_assigned']:
        assert actual_counts[key] == result['counts'][key]
    dropped = [c for c in all_changes if c['category']=='dropped']
    potential = [c for c in dropped if c['potential_inside_high_score_abstention']]
    all_runs = [dict(clip_id=row['clip_id'], label=row['label'], mode=row['mode'], side=s['side'], **run)
        for row in gap_rows for s in row['per_side'] for run in s['dropped_new_missing_runs']]
    for p, expected in pins.items():
        assert digest(p) == expected, p
    summary = dict(actual_counts=actual_counts, changed_detections=len(all_changes),
        dropped_by_diagnostic_category=Counter(c['diagnostic_category'] for c in dropped),
        dropped_by_phase=Counter(c['phase'] for c in dropped),
        dropped_original_any_usable=sum(c['original_any_usable'] for c in dropped),
        dropped_original_all21_usable=sum(c['original_all21_usable'] for c in dropped),
        dropped_palm5_not_all_strict_inside=sum(c['strict_palm5_inside_points']<5 for c in dropped),
        dropped_score_below_0_5=sum(c['raw_score']<.5 for c in dropped),
        dropped_valid_palm_high_score=sum(c['strict_palm5_inside_points']==5 and c['raw_score']>=.8 for c in dropped),
        dropped_all21_inside_high_score=len(potential),
        dropped_other_hand_observed_after=sum(c['other_hand_observed_after'] for c in dropped),
        dropped_other_hand_usable_after=sum(c['other_hand_usable_after'] for c in dropped),
        max_consecutive_new_missing_due_to_drop=max(all_runs,key=lambda r:r['duration_seconds']) if all_runs else None)
    output = dict(status='COMPLETE_NUMERIC_AUDIT_REQUIRES_VISUAL_INTERPRETATION',
        method='Independent saved-array comparison; no HandAssigner import, execution or new inference',
        duration_definition='Consecutive missing rows cover frame PTS to following frame PTS; last frame uses median cadence',
        limitation='Changes/abstentions are not accuracy labels; original assignment may be wrong. Inside high-score drops need visual review. Counted core is annotation interval, not linguistic proof of essentiality.',
        protected_files_unchanged=len(pins), summary=summary, potential_regression_review=potential,
        side_changes_and_new_assignments=[c for c in all_changes if c['category']!='dropped'],
        dropped_detections=dropped, consecutive_missing=gap_rows,
        input_sha256={str(p):digest(p) for p in [U/'REPLAY_RESULT.json',U/'RAW_CAPTURE_CONTEXT.json',Path(__file__)]})
    with (U/'CHANGE_AUDIT.json').open('x',encoding='utf-8') as f:
        json.dump(clean(output),f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(clean(summary),ensure_ascii=False))


if __name__=='__main__':
    run()
