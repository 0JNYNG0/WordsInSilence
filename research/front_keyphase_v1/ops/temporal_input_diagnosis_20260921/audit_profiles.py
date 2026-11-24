"""Independent scalar recount of saved public profiles; no feature re-extraction."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / 'profiles.json').read_text(encoding='utf-8'))
plan = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
annotations = json.loads((HERE / 'source_annotations.json').read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'public_profiles.npz') == report['array_file_sha256']
assert sha(HERE / 'plan.json') == report['plan_sha256']
assert sha(HERE / 'source_annotations.json') == report['annotations_sha256']
assert [r['id'] for r in report['rows']] == [r['id'] for r in plan['inputs']]
assert all(r['status'] == 'PROFILED' for r in report['rows'])
offsets, cursor = {}, 0
for g in report['feature_group_order']:
    offsets[g] = slice(cursor, cursor + report['feature_group_widths'][g])
    cursor += report['feature_group_widths'][g]
counts = {'clips': 0, 'rates': 0, 'missing_rate_slots': 0, 'rate_summaries': 0,
          'annotation_partitions': 0, 'source_visibility_recounts': 0}
max_error = 0.


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None, (a, b)
    else:
        assert math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-6), (a, b)


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    ix = (len(values) - 1) * q
    lo, hi = math.floor(ix), math.ceil(ix)
    return values[lo] + (values[hi] - values[lo]) * (ix - lo)


with np.load(HERE / 'public_profiles.npz', allow_pickle=False) as arrays:
    for row in report['rows']:
        key = row['id'] + '__'
        f, valid = arrays[key+'features'], arrays[key+'present']
        ts, ids = arrays[key+'timestamp_us'], arrays[key+'frame_id']
        clock = [int(t) / 1e6 for t in ts]
        assert len(ts) == row['frames']
        with np.load(row['cache'], allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw['timestamp_us'], ts)
            np.testing.assert_array_equal(raw['frame_id'], ids)
            usable = raw['usable']
            np.testing.assert_array_equal(usable, arrays[key+'usable'])
            seen = [i for i in range(len(ts)) if any(usable[i, 33:75])]
        assert [seen[0], seen[-1]] == row['visible_span']
        assert len(seen) == row['usable_hand_frames']
        assert row['visible_span_frames'] - len(seen) == row['visible_span_internal_missing_hand_frames']
        counts['source_visibility_recounts'] += 1
        window = row['focus_window_s']
        if window is not None:
            a = row['annotation_coverage']
            doc = annotations[row['id']]['document']
            assert a['annotation_count'] == len(doc['data'])
            ii = [i for i, t in enumerate(clock) if window[0] <= t <= window[1]]
            assert ii == a['focus_rows']
            assert len(ii) == a['focus_frames']
            outside = sum(not(window[0] <= clock[i] <= window[1]) for i in range(seen[0], seen[-1]+1))
            assert outside == a['visible_span_frames_outside_focus']
            close(outside / row['visible_span_frames'], a['visible_span_outside_focus_fraction'])
        for side in ('L', 'R'):
            for name, group in (('wrist_speed', 'handpos_'+side), ('shape_rate', 'shape_'+side)):
                x = f[:, offsets[group]]
                gi = report['feature_group_order'].index(group)
                vv = [bool(valid[i, gi]) and all(math.isfinite(float(v)) for v in x[i]) for i in range(len(ts))]
                rate = arrays[key+side+'_'+name]
                for i in range(len(ts)):
                    dt = (int(ts[i])-int(ts[i-1]))/1e6 if i else 0.
                    known = i > 0 and 0 < dt <= .2 and vv[i-1] and vv[i]
                    if not known:
                        assert np.isnan(rate[i])
                        counts['missing_rate_slots'] += 1
                        continue
                    delta = x[i] - x[i-1]  # Retain source float32 subtraction, scalar norm independently.
                    expected = math.sqrt(math.fsum(float(v)*float(v) for v in delta)) / dt
                    close(expected, float(rate[i]))
                    max_error = max(max_error, abs(expected-float(rate[i])))
                    counts['rates'] += 1
                for scope, (lo, hi) in {'whole_clip': (0,len(ts)-1), 'visible_span': tuple(row['visible_span'])}.items():
                    s = row['hands'][side][name][scope]
                    known = [i for i in range(lo+1, hi+1) if math.isfinite(float(rate[i]))]
                    vals = [float(rate[i]) for i in known]
                    assert s['possible_steps'] == hi-lo
                    assert s['observed_steps'] == len(known)
                    assert s['unknown_steps'] == hi-lo-len(known)
                    for q, value in s['quantiles'].items():
                        close(value, quantile(vals, float(q)))
                    terms = {i: float(rate[i])*(int(ts[i])-int(ts[i-1]))/1e6 for i in known}
                    total = math.fsum(terms.values()) if known else None
                    close(total, s['rate_time_integral'])
                    if known:
                        peak = max(known, key=lambda i: float(rate[i]))
                        assert peak == s['peak']['row']
                        assert int(ids[peak]) == s['peak']['frame_id']
                    else:
                        assert s['peak'] is None
                    if window is not None:
                        partitions = {p: [] for p in ('inside','before','after','crosses_boundary')}
                        for i in known:
                            p = ('inside' if window[0] <= clock[i-1] and clock[i] <= window[1] else
                                 'before' if clock[i] < window[0] else
                                 'after' if clock[i-1] > window[1] else 'crosses_boundary')
                            partitions[p].append(i)
                        for p, ii in partitions.items():
                            ps = s['annotation_parts'][p]
                            assert ps['observed_steps'] == len(ii)
                            close(math.fsum(terms[i] for i in ii), ps['rate_time_integral'])
                            close(quantile([float(rate[i]) for i in ii], .95), ps['p95'])
                        strict = math.fsum(terms[i] for p in ('before','after') for i in partitions[p])
                        close(strict / total if total else None, s['strict_outside_integral_fraction'])
                        assert s['peak']['row'] in partitions[s['peak']['annotation_relation']]
                        counts['annotation_partitions'] += 1
                    counts['rate_summaries'] += 1
        counts['clips'] += 1

single = [r for r in report['rows'] if r['annotation_coverage'] and r['annotation_coverage']['single_annotation']]
multi = [r['id'] for r in report['rows'] if r['annotation_coverage'] and not r['annotation_coverage']['single_annotation']]
aggregate = {
    'single_annotation_clips': len(single), 'multi_annotation_clips': multi,
    'single_annotation_visible_span_equals_full_clip': sum(r['visible_span_frames']==r['frames'] for r in single),
    'single_annotation_outside_frame_fraction_range': [min(r['annotation_coverage']['visible_span_outside_focus_fraction'] for r in single), max(r['annotation_coverage']['visible_span_outside_focus_fraction'] for r in single)],
    'single_annotation_wrist_peaks_wholly_outside_annotation': sum(r['hands'][s]['wrist_speed']['visible_span']['peak']['annotation_relation'] in ('before','after') for r in single for s in ('L','R')),
    'single_annotation_wrist_peak_hand_clip_pairs': len(single)*2,
    'nikl_with_no_observed_left_wrist_rate': [r['id'] for r in report['rows'] if r['group']=='nikl_public18' and r['hands']['L']['wrist_speed']['visible_span']['observed_steps']==0],
}
result = {'status':'PASS', 'scope':'Independent scalar recount of new descriptive results, not recognition validation',
          'counts':counts, 'max_rate_absolute_error':max_error, 'aggregate':aggregate,
          'profiles_sha256':sha(HERE/'profiles.json'), 'audit_script_sha256':sha(__file__)}
with (HERE/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:
    json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    f.write('\n')
print(json.dumps(result,ensure_ascii=False))
