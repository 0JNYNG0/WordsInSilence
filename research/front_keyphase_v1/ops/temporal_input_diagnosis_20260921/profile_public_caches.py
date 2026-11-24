"""Read-only operational diagnosis. No classifier, extraction, cropping or training."""
from __future__ import annotations
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path.insert(0, str(LAB / 'src'))
from wis_lab.core.observation import Observation
from wis_lab.core.features import sequence_features, active_span, GROUP_ORDER, GROUPS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def rate_summary(rate, clock, ids, lo, hi, window):
    # Each rate at row i describes the observed interval (i-1, i).
    indices = np.arange(lo + 1, hi + 1, dtype=int)
    known = indices[np.isfinite(rate[indices])]
    dt = np.diff(clock, prepend=clock[0])
    vals = rate[known]
    parts = {}
    if window is not None:
        a, b = window
        conditions = {
            'inside': (clock[known - 1] >= a) & (clock[known] <= b),
            'before': clock[known] < a,
            'after': clock[known - 1] > b,
        }
        used = np.zeros(len(known), dtype=bool)
        for mask in conditions.values():
            used |= mask
        conditions['crosses_boundary'] = ~used
        for name, mask in conditions.items():
            ii = known[mask]
            vv = rate[ii]
            parts[name] = {
                'observed_steps': int(len(ii)),
                'observed_seconds': float(dt[ii].sum()),
                'rate_time_integral': float(np.dot(vv, dt[ii])),
                'p50': float(np.quantile(vv, .5)) if len(vv) else None,
                'p95': float(np.quantile(vv, .95)) if len(vv) else None,
                'max': float(vv.max()) if len(vv) else None,
            }
    total = float(np.dot(vals, dt[known])) if len(known) else None
    peak = None
    if len(known):
        i = int(known[np.argmax(vals)])
        region = None
        if window is not None:
            a, b = window
            region = ('inside' if clock[i-1] >= a and clock[i] <= b else
                      'before' if clock[i] < a else
                      'after' if clock[i-1] > b else 'crosses_boundary')
        peak = {'row': i, 'frame_id': int(ids[i]), 'interval_seconds':
                [float(clock[i-1]), float(clock[i])], 'rate': float(rate[i]),
                'annotation_relation': region}
    outside = (parts['before']['rate_time_integral'] + parts['after']['rate_time_integral']) if parts else None
    return {
        'possible_steps': len(indices), 'observed_steps': len(known),
        'unknown_steps': len(indices) - len(known),
        'observed_fraction': len(known) / len(indices) if len(indices) else None,
        'observed_seconds': float(dt[known].sum()),
        'quantiles': {str(q): float(np.quantile(vals, q)) if len(vals) else None
                      for q in (0, .25, .5, .75, .95, 1)},
        'rate_time_integral': total, 'peak': peak, 'annotation_parts': parts,
        'strict_outside_integral_fraction': outside / total if outside is not None and total else None,
    }


def main():
    plan = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
    annotations = json.loads((HERE / 'source_annotations.json').read_text(encoding='utf-8'))
    outputs = [HERE / 'profiles.json', HERE / 'public_profiles.npz']
    assert not any(p.exists() for p in outputs), 'Preserve existing outputs'
    assert len(plan['inputs']) == 26
    for path, digest in plan['source_pins'].items():
        assert sha(path) == digest, path
    source_extra = {}
    for name in ('phase_feedback.py',):
        path = LAB / 'src/wis_lab/core' / name
        source_extra[str(path)] = sha(path)
    rows, arrays = [], {}
    for item in plan['inputs']:
        row = {**item, 'status': 'ERROR'}
        try:
            cache = Path(item['cache'])
            assert sha(cache) == item['cache_sha256']
            obs = Observation.load(cache)
            seq = sequence_features(obs, trim=False)
            lo, hi = active_span(obs)
            clock = obs.timestamp_us.astype(np.float64) / 1e6
            window = item['focus_window_s']
            any_hand = obs.usable[:, 33:75].any(axis=1)
            annotation = annotations.get(item['id'])
            if annotation:
                doc = annotation['document']
                ann_rows = doc['data']
                focus_matches = [r for r in ann_rows if
                    [r['start'], r['end']] == window and
                    item['focus_label'] in [a['name'] for a in r['attributes']]]
                assert len(focus_matches) == 1
                inside = (clock >= window[0]) & (clock <= window[1])
                visible_rows = np.arange(lo, hi + 1)
                coverage = {
                    'annotation_count': len(ann_rows),
                    'all_labels': [[a['name'] for a in r['attributes']] for r in ann_rows],
                    'single_annotation': len(ann_rows) == 1,
                    'source_duration_s': doc['metaData']['duration'],
                    'focus_rows': np.flatnonzero(inside).tolist(),
                    'focus_frames': int(inside.sum()),
                    'focus_frames_in_visible_span': int(inside[visible_rows].sum()),
                    'visible_span_frames_outside_focus': int((~inside[visible_rows]).sum()),
                    'visible_span_outside_focus_fraction': float((~inside[visible_rows]).mean()),
                    'visible_span_seconds_before_focus': max(0., window[0] - clock[lo]),
                    'visible_span_seconds_after_focus': max(0., clock[hi] - window[1]),
                }
            else:
                coverage = None
            hands = {}
            prefix = item['id'] + '__'
            arrays[prefix + 'features'] = seq['features']
            arrays[prefix + 'present'] = seq['present']
            arrays[prefix + 'timestamp_us'] = obs.timestamp_us
            arrays[prefix + 'frame_id'] = obs.frame_id
            arrays[prefix + 'usable'] = obs.usable
            for side in ('L', 'R'):
                hand = seq['movement_observation']['hands'][side]
                pos = np.array([p if p is not None else [np.nan, np.nan]
                                for p in hand['wrist_xy']], dtype=float)
                hands[side] = {'wrist_observed_frames': int(np.isfinite(pos).all(axis=1).sum())}
                for key, source_key in (
                    ('wrist_speed', 'wrist_speed_sw_per_second'),
                    ('shape_rate', 'shape_rate_hand_local_per_second'),
                ):
                    rate = np.array([v if v is not None else np.nan for v in hand[source_key]], dtype=float)
                    arrays[prefix + side + '_' + key] = rate
                    hands[side][key] = {
                        'whole_clip': rate_summary(rate, clock, obs.frame_id, 0, obs.n_frames-1, window),
                        'visible_span': rate_summary(rate, clock, obs.frame_id, lo, hi, window),
                    }
            sidecar_path = cache.with_suffix('.sidecar.json')
            row.update({
                'status': 'PROFILED', 'frames': obs.n_frames,
                'sidecar_sha256': sha(sidecar_path) if sidecar_path.exists() else None,
                'clock_description': obs.sidecar.get('clock'),
                'timestamp_seconds': [float(clock[0]), float(clock[-1])],
                'frame_id_endpoints': [int(obs.frame_id[0]), int(obs.frame_id[-1])],
                'shoulder_width_px': seq['shoulder_width_px'],
                'visible_span': [lo, hi], 'visible_span_frames': hi-lo+1,
                'usable_hand_frames': int(any_hand.sum()),
                'visible_span_internal_missing_hand_frames': int((~any_hand[lo:hi+1]).sum()),
                'annotation_coverage': coverage, 'hands': hands,
            })
        except Exception as exc:
            row['error'] = type(exc).__name__ + ': ' + str(exc)
            row['traceback'] = traceback.format_exc()
        rows.append(row)
        print(item['id'], row['status'], flush=True)
    for path, digest in plan['source_pins'].items():
        assert sha(path) == digest, path
    for path, digest in source_extra.items():
        assert sha(path) == digest, path
    with outputs[1].open('xb') as f:
        np.savez_compressed(f, **arrays)
    write_new(outputs[0], {
        'schema': 'fixed_public_visibility_motion_profile_result_v1',
        'at': datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'plan_sha256': sha(HERE / 'plan.json'),
        'annotations_sha256': sha(HERE / 'source_annotations.json'),
        'diagnostic_script_sha256': sha(__file__), 'additional_source_pins': source_extra,
        'array_file_sha256': sha(outputs[1]),
        'feature_group_order': GROUP_ORDER, 'feature_group_widths': {g: GROUPS[g][0] for g in GROUP_ORDER},
        'scope': 'Full clip, trim=False; visible span summaries also reported without recomputing normalization',
        'rate_integral_definition': 'sum(observed_rate[i] * delta_seconds[i]); wrist is observed path length in shoulder-width units; shape is observed feature-vector path length',
        'boundary_policy': 'Both interval endpoints inside annotation; wholly before/after separate; crossing intervals separate. Missing steps omitted, never zero-filled or bridged.',
        'limits': ['Descriptive only, no semantic onset/end detector',
                   'Annotation-to-decoded-frame origin not independently established',
                   'Annotation completeness/exact action boundaries unverified',
                   'Shape change and wrist speed may include detector noise',
                   'NIKL18 lacks time boundaries; 7 single-annotation and 1 multi-annotation AIHub clips are distinct'],
        'counters': {'planned': 26, 'profiled': sum(r['status']=='PROFILED' for r in rows),
                     'errors': sum(r['status']!='PROFILED' for r in rows), 'classifier_inference': 0,
                     'training': 0, 'new_extraction': 0, 'participant_records': 0, 'reserved_records': 0},
        'rows': rows,
    })
    print(json.dumps({'profiles': str(outputs[0]), 'profiled': sum(r['status']=='PROFILED' for r in rows)}))
    return 0 if all(r['status']=='PROFILED' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
