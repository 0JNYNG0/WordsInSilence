"""Exposed-development coordinate diagnosis only; no label/model decisions.

All saved normal/speed axis2 intervals are included. Compare the previously
documented unit projection and legacy arm-fraction formula; do not select a
formula by these outcomes, rerun scores, or relabel observed coordinates.
"""
from pathlib import Path
import json
import hashlib
import datetime
import numpy as np

U = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dest = U / 'ENDPOINT_MEASUREMENT_AUDIT.json'
    assert not dest.exists()
    source = U / 'evaluation/RESULTS.json'
    result = read(source)
    records = [r for r in result['predictions'] if r['kind'] in ('full_normal', 'speed075', 'speed125')]
    unique = {}
    for r in records:
        key = (r['clip_id'], r['kind'], *r['span'])
        unique.setdefault(key, []).append(r)
    inputs = {}
    pins = {str(source): sha(source), str(Path(__file__)): sha(__file__)}
    rows = []
    for key, rr in unique.items():
        r = rr[0]
        path = r['source']
        if path not in inputs:
            assert sha(path) == r['source_sha256']
            pins[path] = r['source_sha256']
            with np.load(path, allow_pickle=False) as z:
                inputs[path] = z['x'].copy(), z['ts'].copy()
        x, ts = inputs[path]
        p = x[:, :102].reshape(-1, 51, 2).astype(float)
        valid = x[:, 186:237] > .5
        shoulders = valid[:, 1] & valid[:, 2]
        shoulder_scale = np.median(np.linalg.norm(p[:, 1] - p[:, 2], axis=1)[shoulders])
        assert shoulder_scale > 1e-5
        p = (p - ((p[:, 1] + p[:, 2]) / 2)[:, None]) / shoulder_scale
        p[:, :, 0] *= np.sign(p[:, 2, 0] - p[:, 1, 0])[:, None]
        palm = p[:, [30, 35, 39, 43, 47]].mean(1)
        rel = palm - p[:, 3]
        arm = p[:, 5] - p[:, 3]
        length = np.linalg.norm(arm, axis=1)
        unit = arm / np.maximum(length[:, None], 1e-8)
        projected = (rel * unit).sum(1)
        fraction = projected / np.maximum(length, 1e-8)
        good = valid[:, [30, 35, 39, 43, 47, 3, 5]].all(1) & shoulders & (length > .05)
        a, b = r['span']
        assert bool(good[a]) and bool(good[b]) and good[a:b+1].mean() >= .8
        ids = np.flatnonzero(good[a:b+1]) + a
        delta = float(projected[b] - projected[a])
        assert abs(delta - r['g'][40]) < 1e-6
        rel_term = float(((rel[b] - rel[a]) * (unit[b] + unit[a]) / 2).sum())
        rot_term = float(((rel[b] + rel[a]) / 2 * (unit[b] - unit[a])).sum())
        assert abs(delta - rel_term - rot_term) < 1e-10
        f_delta = float(fraction[b] - fraction[a])
        f_unit_term = delta * (1 / length[b] + 1 / length[a]) / 2
        f_length_term = (projected[b] + projected[a]) / 2 * (1 / length[b] - 1 / length[a])
        assert abs(f_delta - f_unit_term - f_length_term) < 1e-10
        steps = np.diff(projected[ids])
        angles = np.degrees(np.arccos(np.clip((unit[ids[1:]] * unit[ids[:-1]]).sum(1), -1, 1)))
        frames = []
        for f in range(max(0, a-3), min(len(x), b+4)):
            frames.append(dict(frame=f, seconds=float(ts[f]) / 1e6, in_selected=a <= f <= b,
                retained_palm_anchor_available=bool(good[f]), projected=float(projected[f]),
                legacy_arm_fraction=float(fraction[f]), anchor_length=float(length[f]),
                palm=palm[f].tolist(), anchor_elbow=p[f, 3].tolist(), anchor_wrist=p[f, 5].tolist(),
                body_moving_wrist=p[f, 6].tolist()))
        rows.append(dict(clip_id=r['clip_id'], kind=r['kind'], span=r['span'], seeds=[p['seed'] for p in rr],
            saved_verdicts=[dict(seed=p['seed'], state=p['state'], logits=p['logits'], probabilities=p['probabilities']) for p in rr],
            eligible=r['eligible'], coverage=float(good[a:b+1].mean()), retained_good_frames=int(good[a:b+1].sum()), frames=b-a+1,
            projected_start=float(projected[a]), projected_end=float(projected[b]), projected_delta=delta,
            projected_range=float(np.ptp(projected[ids])), projected_path=float(abs(steps).sum()),
            positive_steps=int((steps > 0).sum()), negative_steps=int((steps < 0).sum()),
            projected_minimum_frame=int(ids[np.argmin(projected[ids])]), projected_maximum_frame=int(ids[np.argmax(projected[ids])]),
            legacy_fraction_start=float(fraction[a]), legacy_fraction_end=float(fraction[b]), legacy_fraction_delta=f_delta,
            anchor_length_start=float(length[a]), anchor_length_end=float(length[b]), anchor_length_min=float(length[ids].min()),
            anchor_length_max=float(length[ids].max()), anchor_rotation_degrees=float(angles.sum()),
            unit_projection_product_difference=dict(relative_point_term=rel_term, anchor_rotation_term=rot_term),
            fraction_product_difference=dict(unit_projection_term=float(f_unit_term), reciprocal_length_term=float(f_length_term)),
            source=path, source_sha256=r['source_sha256'], samples=frames))
    assert sum(len(r['seeds']) for r in rows) == 27
    for path, digest in pins.items():
        assert sha(path) == digest
    out = dict(at=datetime.datetime.now().astimezone().isoformat(), scope='All saved27 normal/speed axis2 outputs; exposed development diagnosis, not independent evaluation or new direction gold.',
        formulas=dict(unit_projection='dot(palm-elbow, unit(other_wrist-elbow)), in shoulder widths',
            legacy_arm_fraction='dot(palm-elbow, other_wrist-elbow)/squared_anchor_length',
            decomposition='Exact endpoint product differences; components do not independently constitute true motion.'),
        rows=rows, unique_intervals=len(rows), saved_outputs=27, pins=pins,
        no_model_forward=True, no_rescoring=True, no_relabeling=True, no_training=True,
        limitations=['Coordinate paths cannot determine physical return/reset vs detector error without RGB review.',
            'Legacy arm fraction may compensate anchor foreshortening but also amplifies a short/noisy anchor denominator; not selected by this diagnosis.',
            'No coordinate formula proves palm contact, semantic direction or whole-action completion.'])
    payload = json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)
    with dest.open('x', encoding='utf-8') as f:
        f.write(payload + '\n')
    print('Coordinate-only endpoint audit complete:', len(rows), 'unique intervals /27 saved outputs', flush=True)


if __name__ == '__main__':
    main()
