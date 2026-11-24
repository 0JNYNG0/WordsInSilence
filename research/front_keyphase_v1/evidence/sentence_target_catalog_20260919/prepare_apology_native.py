"""Isolated acquisition audit. Reuse existing readers/adapters; never run training."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import collections
import hashlib
import json
import math
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PILOT = ROOT / 'research/sign_model_pilot_v1'
CACHE = ROOT / 'research/front_keyphase_v1/runs/apology_native_prep_v1'
sys.path[:0] = [str(PILOT), str(PILOT / 'src')]
sys.dont_write_bytecode = True
import expand_build as existing
from pilot.data import build_channels


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    started = time.monotonic()
    rows = json.loads((OUT / 'candidate_rows.json').read_text(encoding='utf-8'))['rows']
    rows = [r for r in rows if r['target_candidate_not_approved'] == 'JOESONGHADA'
            and r['signer_id'] in [1,2,3,4,5,6,7,8,9,11,12,13]]
    assert len(rows) == 12 and len({r['signer_id'] for r in rows}) == 12
    assert all(r['sentence_id'] == '0356' and r['numeric_bounds_valid'] for r in rows)
    assert all(not r['training_approved'] and r['semantic_target_id'] is None for r in rows)
    pins = [Path(existing.__file__), Path(existing.PREP) / 'zip_probe.py',
            PILOT / 'src/pilot/data.py', Path(existing.PREP) / 'model_contract.json',
            PILOT / 'expand/manifest.json', OUT / 'candidate_rows.json']
    before = {str(p): sha(p) for p in pins}
    CACHE.mkdir(parents=True, exist_ok=False)
    prepared, failures, member_evidence = [], [], []
    for row in rows:
        signer = row['signer_id']
        expected_role = 'train_candidate' if signer <= 9 else 'development_only'
        assert row['split_role'] == expected_role
        archive = Path(existing.AIHUB) / f'[라벨]{signer:02}_real_sen_keypoint.zip'
        stat = archive.stat()
        try:
            members, stats = existing.selected_directory(archive, {row['clip_id']})
            frame_map = {}
            for member in members:
                match = existing.FRAME_RE.search(member['name'])
                if not match:
                    continue
                frame = int(match.group(1))
                assert frame not in frame_map, 'duplicate frame index'
                frame_map[frame] = member
            start = math.ceil(row['annotation_start_s'] * existing.FPS - 1e-9)
            stop = math.ceil(row['annotation_end_s'] * existing.FPS - 1e-9)
            frames = list(range(start, stop))
            missing = [f for f in frames if f not in frame_map]
            assert not missing, f'missing frames {missing}'
            assert len(frames) >= 6
            with archive.open('rb') as handle:
                arr = np.stack([existing.decode(handle, frame_map[f]) for f in frames])
            assert arr.shape == (len(frames), 67, 3) and np.isfinite(arr).all()
            present = np.isfinite(arr).all(-1) & ~np.all(arr[..., :2] == 0, axis=-1)
            indices = np.asarray(frames, dtype=np.int64)
            nominal_us = np.rint(indices / existing.FPS * 1e6).astype(np.int64)
            channels = build_channels(arr[..., :2], present, nominal_us)
            x = channels['x']
            assert x.shape == (len(frames), 280) and np.isfinite(x).all()
            common = np.concatenate([present[:, [0,5,2,6,3,7,4,12,9]], present[:,25:]], axis=1)
            body_mask = common & (present[:,5] & present[:,2])[:,None]
            assert np.array_equal(x[:,186:237], body_mask)
            for native, offset, xy_offset in [(25,237,102),(46,258,144)]:
                mask = present[:, native:native+21] & present[:,native,None]
                side = 'left' if native == 25 else 'right'
                if not np.isfinite(channels['hand_scales'][side]):
                    mask[:] = False
                assert np.array_equal(x[:,offset:offset+21], mask)
                assert np.all(x[:,xy_offset:xy_offset+42].reshape(-1,21,2)[~mask] == 0)
            assert np.all(x[:,:102].reshape(-1,51,2)[~body_mask] == 0)
            assert x[0,-1] == 0
            assert np.allclose(x[1:,-1], np.diff(nominal_us) / 1e6, rtol=0, atol=1e-8)
            dest = CACHE / (row['clip_id'] + '_candidate.npz')
            np.savez_compressed(dest, xy_native=arr[...,:2], confidence_native=arr[...,2],
                                coordinate_present=present, frame_index=indices,
                                timestamp_nominal_us=nominal_us, x=x)
            with np.load(dest, allow_pickle=False) as saved:
                assert np.array_equal(saved['xy_native'], arr[...,:2])
                assert np.array_equal(saved['confidence_native'], arr[...,2])
                assert np.array_equal(saved['coordinate_present'], present)
                assert np.array_equal(saved['frame_index'], indices)
                assert np.array_equal(saved['timestamp_nominal_us'], nominal_us)
                again = build_channels(saved['xy_native'], saved['coordinate_present'], saved['timestamp_nominal_us'])
                assert np.array_equal(again['x'], saved['x']) and np.array_equal(saved['x'], x)
            stat_after = archive.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns)
            prepared.append({**row, 'npz': str(dest), 'npz_sha256': sha(dest), 'npz_bytes': dest.stat().st_size,
                'frames': len(frames), 'first_frame': start, 'last_frame': stop-1,
                'source_archive': str(archive), 'source_archive_bytes': stat.st_size,
                'source_archive_mtime_ns': stat.st_mtime_ns, 'source_stat_unchanged': True,
                'crc_verified_frame_members': len(frames), 'directory_stats': stats,
                'native_confidence_min_max': [float(arr[...,2].min()),float(arr[...,2].max())],
                'left_all21_coordinate_present_frames': int(present[:,25:46].all(1).sum()),
                'right_all21_coordinate_present_frames': int(present[:,46:67].all(1).sum()),
                'both_hands_no_coordinates_frames': int((~present[:,25:].any(1)).sum()),
                'body_scale': channels['body_scale'], 'channels': 280,
                'source_schema': 'OpenPose BODY25 + left21 + right21; 2D plus native confidence',
                'timebase': 'source frame index / nominal30fps; original video PTS and frame-origin alignment UNVERIFIED',
                'numeric_contract_status': 'PASS', 'web_detector_domain_equivalence': 'UNVERIFIED'})
            member_evidence.append({'signer_id': signer, 'clip_id': row['clip_id'],
                                    'selected_frame_members': [frame_map[f] for f in frames]})
            print(json.dumps({'signer': signer, 'role': expected_role, 'frames': len(frames), 'status': 'PASS'}), flush=True)
        except Exception as exc:
            failures.append({'signer_id': signer, 'clip_id': row['clip_id'], 'error': repr(exc)})
            print(json.dumps(failures[-1]), flush=True)
    unchanged = before == {str(p): sha(p) for p in pins}
    result = {'status': 'PASS_NATIVE_CANDIDATE_INPUT_ONLY' if len(prepared) == 12 and not failures and unchanged else 'FAIL',
              'at': datetime.now(timezone(timedelta(hours=9))).isoformat(),
              'prepared_clips': len(prepared), 'prepared_frames': sum(r['frames'] for r in prepared),
              'role_counts': dict(collections.Counter(r['split_role'] for r in prepared)),
              'input_code_manifest_pins': before, 'pins_unchanged': unchanged, 'rows': prepared, 'failures': failures,
              'output_bytes': sum(r['npz_bytes'] for r in prepared), 'elapsed_seconds': round(time.monotonic()-started,3),
              'reserved_test_arrays_read': 0, 'unassigned_arrays_read': 0, 'training_approved': False,
              'training_runs': 0, 'classifier_inference_calls': 0, 'production_changes': 0,
              'interpretation': 'Candidate preparation only. Native coordinate-present is not MediaPipe usable or coordinate accuracy. Labels, nominal frame alignment and lexical equivalence need review before training.'}
    dump(OUT / 'apology_native_frame_provenance.json', member_evidence)
    dump(OUT / 'apology_native_prep.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','input_code_manifest_pins']}, ensure_ascii=False),flush=True)
    return 0 if result['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
