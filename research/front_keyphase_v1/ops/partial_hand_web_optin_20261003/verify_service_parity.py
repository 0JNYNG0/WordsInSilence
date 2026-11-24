"""Independent frozen-data to actual service parity, no camera or extraction."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

UNIT = Path(__file__).resolve().parent
LAB = UNIT.parents[1]
OLD = LAB / 'ops/partial_hand_reconstruction_20261003'
CFG = json.loads((OLD / 'CONFIG_LOCAL_ONLY.json').read_text(encoding='utf-8-sig'))
sys.path[:0] = [str(LAB / 'src'), CFG['pilot_src'], CFG['helper_dir'], str(OLD)]
import run_audit as R
from wis_lab.service import neural as N
ref_spec = importlib.util.spec_from_file_location('independent_handgap_reference', OLD / 'verify_reference.py')
ref = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref)
reference_restore, array_hash = ref.reference_restore, ref.array_hash
from pilot.web_input import to_native_like
from pilot.data import build_channels


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('before', 'after'))
    phase = ap.parse_args().phase
    pins = dict(CFG['pins'])
    allowed_changes = {}
    if phase == 'after':
        manifest = json.loads((UNIT / 'APPLIED_MANIFEST.json').read_text(encoding='utf-8'))
        for rel, record in manifest['files'].items():
            p = str(LAB / rel)
            if p in pins:
                assert pins[p] == record['before'], (rel, 'unexpected old pin')
                pins[p] = record['after']
                allowed_changes[p] = record
    for p, h in pins.items():
        assert sha(p) == h, ('protected hash', p)
    rows = R.load_mp_rows(CFG['mp_manifest'])
    old_result = json.loads((OLD / 'result/RESULT.json').read_text())['groups']['mediapipe']['rows']
    with np.load(OLD / 'result/SCORES_mediapipe.npz', allow_pickle=False) as f:
        old_scores, labels, ids = f['logits'].copy(), f['classes'].tolist(), f['ids'].tolist()
    if phase == 'after':
        import torch
        torch.set_num_threads(2)
    records = []
    for i, row in enumerate(rows):
        ip, ih = row['files']['input']; op, oh = row['files']['observation']
        assert sha(ip) == ih and sha(op) == oh
        with np.load(ip, allow_pickle=False) as f:
            x, fid, ts = f['x'].copy(), f['frame_id'].copy(), f['timestamp_us'].copy()
        with np.load(op, allow_pickle=False) as f:
            positions = {int(v): j for j, v in enumerate(f['frame_id'])}
            ix = [positions[int(v)] for v in fid]
            obs = SimpleNamespace(xy_px=f['xy_px'][ix].copy(), usable=f['usable'][ix].copy(),
                                  timestamp_us=f['timestamp_us'][ix].copy(), frame_id=f['frame_id'][ix].copy())
        hashes = {k: array_hash(v) for k, v in vars(obs).items()}
        built, why, stats = N._channels(obs)
        assert why is None, (row['id'], why)
        assert built['x'].shape == x.shape and np.array_equal(built['x'], x), (row['id'], 'service input differs')
        assert np.array_equal(ts, obs.timestamp_us)
        rec = {'id': row['id'], 'index': i, 'frames': len(ts), 'raw_input_hash': array_hash(x), 'raw_exact': True}
        if phase == 'after':
            assert ids[i] == row['id'] == old_result[i]['id']
            nat, pr = to_native_like(obs.xy_px, obs.usable.astype(bool))
            restored = reference_restore(nat, pr, ts)
            expected_x = build_channels(restored['xy'], restored['present'], restored['ts'])['x']
            holder = {}
            module = N._handgap_module()
            assert module['status'] == 'READY'
            service_clip, reason, _ = N._channels(obs, N._handgap_transform(obs, module['module'], holder))
            assert reason is None and np.array_equal(service_clip['x'], expected_x)
            expected_fills = sorted((e['frame'], ('left', 'right')[e['side']]) for e in restored['events'])
            actual_fills = sorted((e['source_row'], e['side']) for e in holder['fills'])
            assert actual_fills == expected_fills
            for e in holder['fills']:
                assert e['source_frame_id'] == int(fid[e['source_row']])
                assert e['timestamp_us'] == int(ts[e['source_row']])
            answer = N.recognise(obs, 'tcn_control_83212868_handgap')
            assert answer['model_status'] == 'READY'
            audit = answer['hand_gap_comparison']['logit_audit']
            assert audit['label_order'] == labels
            scores = np.array([audit['original_coordinates'], audit['short_gap_restored']], dtype=np.float64)
            delta = np.max(np.abs(scores - old_scores[i, :2]), axis=1)
            assert np.all(delta <= 1e-5), (row['id'], delta.tolist())
            assert np.array_equal(scores.argmax(axis=1), old_scores[i, :2].argmax(axis=1))
            assert answer['input_transform']['provenance'] == holder
            rec.update(repaired_input_hash=array_hash(expected_x), restored_exact=True,
                       filled_hand_frames=len(actual_fills), max_logit_diff=delta.tolist(),
                       raw_top1=labels[int(scores[0].argmax())], repaired_top1=labels[int(scores[1].argmax())])
        assert hashes == {k: array_hash(v) for k, v in vars(obs).items()}, ('mutated', row['id'])
        records.append(rec)
    for p, h in pins.items():
        assert sha(p) == h
    assert 'cv2' not in sys.modules and 'mediapipe' not in sys.modules
    out = {'status': 'PASS', 'phase': phase, 'rows': records, 'row_count': len(records),
           'model_calls': len(records) * 2 if phase == 'after' else 0,
           'allowed_code_changes': allowed_changes, 'camera': 0, 'user_data': 0,
           'max_score_diff': max(max(v['max_logit_diff']) for v in records) if phase == 'after' else None}
    target = 'BASELINE_INPUT_PARITY.json' if phase == 'before' else 'SERVICE_PARITY.json'
    with (UNIT / target).open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'allowed_code_changes')}))


if __name__ == '__main__':
    main()
