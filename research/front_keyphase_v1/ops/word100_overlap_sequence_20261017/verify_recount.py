"""Separate read-only recount: no audit import, classifier, detector, or user input."""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

U = Path(__file__).resolve().parent
OPS = U.parent


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


a = read(U / 'AUDIT.json')
s = read(U / 'SCOPE.json')
recovery = read(U / 'RECOVERY.json')
assert sha(U / 'SCOPE.json') == a['scope_sha256'] == recovery['scope_sha256']
assert sha(U / 'audit.py') == recovery['corrected_code_sha256']
assert recovery['excluded_clips'] == 0 and recovery['condition_change'] is False
assert set(s['clips']) == {r['clip_id'] for r in a['records']}
assert len(a['records']) == 36
for p, expected in a['pins'].items():
    assert sha(p) == expected, p
protected = read(OPS / 'word100_frame_interval_20261017/FINAL_CHECK.json')['protected_pins']
for p, expected in protected.items():
    assert sha(p) == expected, p

totals = {}
checked_coordinates = 0
flag_keys = set()
for view in ('native', 'sensor'):
    counts = Counter()
    for r in a['records']:
        assert 1 <= r['signer_id'] <= 9
        v = r['views'][view]
        with np.load(v['raw_path']) as raw, np.load(v['assignment_path']) as assignment, np.load(v['candidate_path']) as obs:
            assert np.array_equal(raw['frame_id'], obs['frame_id'])
            assert np.array_equal(raw['timestamp_us'], obs['timestamp_us'])
            core = np.where((raw['timestamp_us'] >= r['annotation'][0] * 1e6) &
                            (raw['timestamp_us'] <= r['annotation'][1] * 1e6))[0]
            assert core.tolist() == [f['frame'] for f in v['frames']]
            trace = read(v['trace_path'])
            for f in core:
                counts['core_frames'] += 1
                n = int(raw['raw_count'][f])
                sides = assignment['sides'][f, :n]
                assert len(set(sides[sides >= 0])) == sum(sides >= 0)
                counts[f'raw{n}'] += 1
                counts[f'assigned{sum(sides >= 0)}'] += 1
                for hand in (0, 1):
                    points = np.array([0, 5, 9, 13, 17]) + 33 + 21 * hand
                    if obs['usable'][f, points].all():
                        counts['observed_hand_slots'] += 1
                    else:
                        counts['missing_hand_slots'] += 1
                        if hand in sides:
                            counts['assigned_but_palm_unusable'] += 1
                        elif np.any(sides < 0):
                            counts['missing_with_unassigned_raw'] += 1
                        else:
                            counts['raw_absence_slots'] += 1
                for index, hand in enumerate(sides):
                    if hand < 0:
                        assert trace[f]['reasons'][index] == 'same_side_alternative'
                        counts['withheld_raw'] += 1
                        flag_keys.add((r['clip_id'], view, int(f), 'two_raw_withheld'))
                        continue
                    selected = obs['xy_px'][f, 33 + 21 * hand:54 + 21 * hand]
                    expected = raw['hand_xy_px'][f, index].astype(np.float32)
                    valid = np.isfinite(selected).all(1)
                    assert np.array_equal(selected[valid], expected[valid]), (r['clip_id'], view, f)
                    checked_coordinates += int(valid.sum()) * 2
                    if hand != raw['raw_label'][f, index]:
                        counts['raw_label_disagreement_detections'] += 1
                        flag_keys.add((r['clip_id'], view, int(f), 'raw_label_disagreement'))
    assert counts['observed_hand_slots'] + counts['missing_hand_slots'] == 2 * counts['core_frames']
    for k, v in a['summary'][view]['counts'].items():
        assert counts[k] == v, (view, k)
    assert counts['assigned_but_palm_unusable'] == 0
    totals[view] = dict(counts)
expected_flags = {(r['clip_id'], r['view'], f, r['kind']) for r in a['flags'] for f in r['frames']}
assert expected_flags == flag_keys

stored_scores = {}
score_pins = {}
for view in ('masked', 'sensor'):
    path = OPS / 'word100_frame_interval_20261017' / f'PRED_{view}_fps30.json'
    pred = read(path)
    assert pred['model_sha256'] == '283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab'
    rows = [r for r in pred['rows'] if r['clip_id'] in s['clips']]
    assert len(rows) == 36
    ranks = {}
    for r in rows:
        assert r['role'] == 'train_candidate' and r['signer_id'] <= 9
        order = sorted(range(len(r['logits'])), key=lambda i: (-r['logits'][i], i))
        ranks[r['clip_id']] = order.index(r['label_index']) + 1
    stored_scores[view] = {'top1': sum(x == 1 for x in ranks.values()), 'clips': 36, 'ranks': ranks,
                          'meaning': 'Reused TRAIN predictions; not new accuracy or generalization evaluation.'}
    score_pins[str(path)] = sha(path)

result = {'status': 'PASS', 'at': datetime.now().astimezone().isoformat(timespec='seconds'),
          'operator': 'Codex root; independent recount code, not another agent or expert',
          'source_pins_verified': len(a['pins']), 'protected_pins': protected,
          'summary': totals, 'exact_coordinate_channels_checked': checked_coordinates,
          'flag_frame_kind_pairs': len(flag_keys), 'stored_train_scores': stored_scores,
          'score_pins': score_pins, 'audit_sha256': sha(U / 'AUDIT.json'),
          'new_inference': 0, 'new_detection': 0, 'new_training': 0, 'user_recording_reads': 0}
with (U / 'RECOUNT.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in result.items() if k not in ('protected_pins', 'stored_train_scores', 'score_pins')}, ensure_ascii=False, indent=2))
