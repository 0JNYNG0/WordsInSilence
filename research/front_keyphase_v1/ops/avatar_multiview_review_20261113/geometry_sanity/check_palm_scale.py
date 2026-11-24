"""Append scale-controlled diagnostics; original audit and source stay unchanged."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np

U = Path(__file__).resolve().parent


def main():
    target = U / 'PALM_SCALE_CHECK.json'
    assert not target.exists(), 'Do not replace finished scale check'
    protocol = json.loads((U / 'PROTOCOL.json').read_text('utf-8'))
    source = Path(protocol['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol['source_sha256']
    with np.load(source, allow_pickle=False) as z:
        xyz = z['official_xyz'][..., :3].copy()
        score = z['official_xyz'][..., 3].copy()
    edges = protocol['edges']
    length = np.stack([np.linalg.norm(xyz[:, :, b] - xyz[:, :, a], axis=-1) for a, b in edges], axis=-1)
    palm_scale = np.median(np.linalg.norm(xyz[:, :, [5, 9, 13, 17]] - xyz[:, :, 0, None], axis=-1), axis=-1)
    assert np.isfinite(palm_scale).all() and (palm_scale > 1e-12).all()
    scaled = length / palm_scale[:, :, None]
    raw_step = abs(np.diff(length, axis=0)) / np.median(length, axis=0)
    scaled_step = abs(np.diff(scaled, axis=0)) / np.median(scaled, axis=0)
    raw_flags, normalized_flags = raw_step > .5, scaled_step > .5
    summaries = []
    for side, label in enumerate(['L', 'R']):
        # Independent scalar Euclidean distances and Python median establish formula.
        scalar = []
        for t in range(127):
            values = sorted(math.dist(xyz[t, side, 0], xyz[t, side, k]) for k in [5, 9, 13, 17])
            scalar.append((values[1] + values[2]) / 2)
        assert np.max(abs(palm_scale[:, side] - scalar)) < 1e-14
        for k, (a, b) in enumerate(edges):
            v = scaled[:, side, k]
            summaries.append(dict(side=label, edge=[a, b], kind='palm_spoke' if k < 5 else 'finger_segment',
                                  length_over_current_palm=dict(minimum=float(v.min()), median=float(np.median(v)), maximum=float(v.max())),
                                  largest_step_over_temporal_median=float(scaled_step[:, side, k].max()),
                                  largest_step_from_frame=int(scaled_step[:, side, k].argmax()),
                                  flagged_pairs=[[int(i), int(i+1)] for i in np.flatnonzero(normalized_flags[:, side, k])]))
    raw_finger = raw_flags[:, :, 5:]
    normalized_finger = normalized_flags[:, :, 5:]
    worst = max([v for v in summaries if v['kind'] == 'finger_segment'], key=lambda v: v['largest_step_over_temporal_median'])
    out = dict(status='SCALE_CONTROLLED_SANITY_COMPLETE_NOT_ACCURACY_VALIDATION',
               normalization='Per-frame median 3D wrist0-to-MCP5/9/13/17 distance, independently for each hand; source XYZ untouched.',
               score_not_geometry=True, score_values=np.unique(score).tolist(),
               physical_unit_confirmed=False,
               raw_finger_step_flags=int(raw_finger.sum()), normalized_finger_step_flags=int(normalized_finger.sum()),
               same_flag_after_scale_control=int((raw_finger & normalized_finger).sum()),
               raw_flags_removed_by_scale_control=int((raw_finger & ~normalized_finger).sum()),
               new_relative_flags=int((~raw_finger & normalized_finger).sum()),
               finger_step_denominator=int(raw_finger.size),
               normalized_all_bone_flags=int(normalized_flags.sum()),
               normalized_unique_frame_pairs=int(normalized_flags.any(axis=(1,2)).sum()),
               palm_scale={label: dict(minimum=float(palm_scale[:,s].min()), median=float(np.median(palm_scale[:,s])),
                                      maximum=float(palm_scale[:,s].max()),
                                      largest_adjacent_change_over_median=float(abs(np.diff(palm_scale[:,s])).max()/np.median(palm_scale[:,s])))
                           for s,label in enumerate(['L','R'])},
               worst_normalized_finger=worst, bones=summaries,
               interpretation='Changes remain after removal of overall per-frame hand scale. This is consistent with local landmark instability; source/camera reprojection is required to establish the cause.',
               diagnostic_threshold='>50% of the same normalized segment temporal median; heuristic screen, not an educational error rate.',
               independent_palm_formula='Python math.dist and sorted median agree with NumPy within 1e-14 raw units.',
               source_sha256_before=protocol['source_sha256'], source_sha256_after=hashlib.sha256(source.read_bytes()).hexdigest())
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='bones'},ensure_ascii=False))


if __name__=='__main__':
    main()
