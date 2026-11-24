"""Read-only geometry sanity audit of one pre-existing public AIHub 3D cache."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import traceback
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parents[1]
OLD = O / 'word100_aihub_coordinate_quality_20261026'
CID = 'NIA_SL_WORD1567_REAL02_F'
SOURCE = OLD / 'arrays' / (CID + '.npz')
RECORD = OLD / 'records' / (CID + '.json')
EDGES = [(0, 4*f+1) for f in range(5)] + [(4*f+j, 4*f+j+1) for f in range(5) for j in (1, 2, 3)]
SIDES = ('L', 'R')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(name, data):
    with (U / name).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def summarize(x):
    x = np.asarray(x, dtype=float)
    v = x[np.isfinite(x)]
    return dict(n=int(v.size), minimum=float(v.min()), p05=float(np.percentile(v, 5)),
                median=float(np.median(v)), p95=float(np.percentile(v, 95)), maximum=float(v.max()),
                mean=float(v.mean()), standard_deviation=float(v.std())) if len(v) else dict(n=0)


def main():
    assert not (U / 'GEOMETRY_SANITY.json').exists(), 'Completed audit is immutable; no duplicate run.'
    record = read(RECORD)
    source_hash = sha(SOURCE)
    assert source_hash == record['arrays_sha']
    protocol = dict(clip_id=CID, public_record_id=2, source=str(SOURCE), source_sha256=source_hash,
                    record=str(RECORD), record_sha256=sha(RECORD), cache_order=['anatomical_left', 'anatomical_right'],
                    point_order='OpenPose hand 21: wrist0, thumb1-4, index5-8, middle9-12, ring13-16, little17-20',
                    edges=[list(v) for v in EDGES], near_zero_threshold_raw_units=1e-12,
                    jump_screen='absolute adjacent length change / temporal median bone length > 0.5',
                    jump_screen_interpretation='Diagnostic flag only, not validated anatomical or sign-language rejection threshold.',
                    unit='Not stored in this NPZ; raw numeric units only. Do not label these values metres or millimetres.',
                    no_interpolation=True, no_retargeting=True, no_inference=True,
                    limits='Same-clip cameras absent. Finite values/bone consistency do not establish image correspondence or 3D accuracy.',
                    references=['https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md',
                                str(OLD / 'study.py') + ':112-115'])
    write('PROTOCOL.json', protocol)
    with np.load(SOURCE, allow_pickle=False) as z:
        raw = z['official_xyz'].copy()
        frames = z['frame_id'].copy()
        pts = z['timestamp_us'].copy()
        arrays = {key: dict(shape=list(z[key].shape), dtype=str(z[key].dtype)) for key in z.files}
    assert raw.shape == (127, 2, 21, 4)
    assert np.array_equal(frames, np.arange(127))
    assert np.all(np.diff(pts) > 0)
    xyz = raw[..., :3]
    valid = np.isfinite(xyz).all(axis=-1) & (raw[..., 3] > 0)
    lengths = np.stack([np.linalg.norm(xyz[:, :, b] - xyz[:, :, a], axis=-1) for a, b in EDGES], axis=-1)
    bone_valid = np.stack([valid[:, :, a] & valid[:, :, b] for a, b in EDGES], axis=-1)
    lengths[~bone_valid] = np.nan
    medians = np.nanmedian(lengths, axis=0)
    relative_step = np.abs(np.diff(lengths, axis=0)) / np.maximum(medians[None], 1e-12)
    flags = relative_step > .5
    rows = []
    for side in range(2):
        for edge_index, (a, b) in enumerate(EDGES):
            values = lengths[:, side, edge_index]
            delta = relative_step[:, side, edge_index]
            stats = summarize(values)
            rows.append(dict(side=SIDES[side], edge=[a, b], kind='palm_spoke' if edge_index < 5 else 'finger_segment',
                             length=stats, zero_length_count=int((values <= 1e-12).sum()),
                             coefficient_of_variation=float(np.nanstd(values) / np.nanmean(values)),
                             p95_p05_ratio=float(np.nanpercentile(values, 95) / np.nanpercentile(values, 5)),
                             largest_step_over_median=float(np.nanmax(delta)),
                             largest_step_from_frame=int(np.nanargmax(delta)),
                             large_step_pairs=[[int(frames[i]), int(frames[i+1])] for i in np.flatnonzero(flags[:, side, edge_index])]))
    points_step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    palm_reference = np.nanmedian(np.linalg.norm(xyz[:, :, 9] - xyz[:, :, 0], axis=-1), axis=0)
    normalized_point_steps = points_step / palm_reference[None, :, None]
    left_wrist, right_wrist = xyz[:, 0, 0], xyz[:, 1, 0]
    same_distance = np.linalg.norm(np.diff(left_wrist, axis=0), axis=-1) + np.linalg.norm(np.diff(right_wrist, axis=0), axis=-1)
    swap_distance = np.linalg.norm(left_wrist[1:] - right_wrist[:-1], axis=-1) + np.linalg.norm(right_wrist[1:] - left_wrist[:-1], axis=-1)
    duplicate_pairs = [[int(frames[i-1]), int(frames[i])] for i in range(1, len(frames)) if np.array_equal(raw[i], raw[i-1])]
    summary = dict(clip_id=CID, frames=127, bone_frame_values=int(lengths.size), adjacent_bone_steps=int(relative_step.size),
                   finite_xyz_points=int(np.isfinite(xyz).all(axis=-1).sum()), total_xyz_points=int(valid.size),
                   positive_3d_status_points=int(valid.sum()), status_values=np.unique(raw[..., 3]).tolist(),
                   zero_length_bone_frames=int((lengths <= 1e-12).sum()),
                   flags_over_half_median=dict(all_bones=int(flags.sum()), palm_spokes=int(flags[:, :, :5].sum()),
                                              finger_segments=int(flags[:, :, 5:].sum()),
                                              unique_frame_pairs=int(flags.any(axis=(1, 2)).sum()),
                                              left=int(flags[:, 0].sum()), right=int(flags[:, 1].sum())),
                   duplicate_3d_frame_pairs=duplicate_pairs,
                   anatomical_labels='Left/right field order is verified in extraction code; physical anatomical correctness not visually re-established in this audit.',
                   wrist_nearest_identity_screen=dict(swapped_assignment_shorter_pairs=[[int(frames[i]), int(frames[i+1])] for i in np.flatnonzero(swap_distance < same_distance)],
                                                      limit='Nearest temporal distance is only a diagnostic; real hand crossing can also change proximity.'),
                   native_frames=dict(first=int(frames[0]), last=int(frames[-1]), contiguous=True),
                   timestamps=dict(first_us=int(pts[0]), last_us=int(pts[-1]), step_us_values=np.unique(np.diff(pts)).tolist(),
                                   provenance='Copied from historical MediaPipe/native observation stream; independent camera exposures not stored here.'),
                   units=dict(physical_unit_confirmed=False, xyz_coordinate_ranges=[summarize(xyz[..., i]) for i in range(3)],
                              palm_reference_raw_units=palm_reference.tolist(), convention='No axis/origin/scale metadata in cache; no absolute human size claim.'),
                   normalized_point_steps={SIDES[s]: summarize(normalized_point_steps[:, s]) for s in range(2)},
                   arrays=arrays, bones=rows,
                   limits=['No camera matrix payload in this cache; no reprojected image test.',
                           'Official 3D status1 is not a probability of correct joint placement.',
                           'Palm spokes are landmark distances, not rigid anatomical bones. Finger inter-joint segments are reported separately.',
                           'No source video viewing, correction, new extraction, animation, training or personal data access in this appendix.'])
    write('GEOMETRY_SANITY.json', summary)
    with (U / 'BONE_LENGTHS.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'timestamp_us', 'side', 'joint_a', 'joint_b', 'kind', 'length_raw_units', 'length_over_temporal_median'])
        for i, frame in enumerate(frames):
            for side in range(2):
                for k, (a, b) in enumerate(EDGES):
                    writer.writerow([int(frame), int(pts[i]), SIDES[side], a, b, 'palm' if k < 5 else 'finger',
                                     float(lengths[i, side, k]), float(lengths[i, side, k] / medians[side, k])])
    # Standalone scientific plot, no image editing or new video-derived assets.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True)
    time_s = pts / 1e6
    for side, ax in enumerate(axes):
        relative = lengths[:, side, 5:] / medians[side, 5:]
        worst = int(np.nanargmax(np.nanmax(relative_step[:, side, 5:], axis=0)))
        ax.plot(time_s, relative, color='#647b98', alpha=.3, linewidth=1)
        a, b = EDGES[5 + worst]
        ax.plot(time_s, relative[:, worst], color='#162d50', linewidth=2.3, label=f'가장 큰 순간 변화: 관절 {a}–{b}')
        ax.axhline(1, color='#8593a5', linestyle='--', linewidth=1)
        ax.set_title(('왼손' if side == 0 else '오른손') + ' 손가락 연결점 사이 거리', loc='left', fontsize=13, color='#172b47')
        ax.set_ylabel('평소 길이 대비 배율', fontsize=11)
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=10, loc='upper right')
    axes[-1].set_xlabel('캐시에 기록된 시간 (초)', fontsize=11)
    fig.suptitle('같은 손가락 뼈의 길이가 시간에 따라 크게 변하는지 확인', fontsize=17, color='#172b47')
    fig.savefig(U / 'BONE_LENGTH_SANITY.png', dpi=160)
    plt.close(fig)
    assert sha(SOURCE) == source_hash
    write('VERIFIED.json', dict(source_sha256_before=source_hash, source_sha256_after=sha(SOURCE),
                                source_unchanged=True, expected_bone_frames=127*2*20,
                                actual_bone_frames=int(lengths.size), expected_steps=126*2*20,
                                actual_steps=int(relative_step.size), script_sha256=sha(Path(__file__)),
                                status='COMPUTATION_COMPLETE_NOT_GEOMETRIC_ACCURACY_CERTIFICATION'))
    print(json.dumps({k: summary[k] for k in ['frames', 'zero_length_bone_frames', 'flags_over_half_median', 'duplicate_3d_frame_pairs', 'wrist_nearest_identity_screen']}, ensure_ascii=False))
    print('WORST', json.dumps(sorted(rows, key=lambda r: r['largest_step_over_median'], reverse=True)[:6], ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        path = U / 'FIRST_ERROR.txt'
        if not path.exists():
            path.write_text(traceback.format_exc(), encoding='utf-8')
        raise
