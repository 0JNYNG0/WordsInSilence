"""TRAIN-only legacy calibration forecast; no detector or model execution."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from collections import Counter

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
LAB = U.parents[1]
OLD = LAB / 'ops/word100_training_20261013'
DATA = Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
SRC = LAB / 'ops/word100_stable_rest_roi_20261017/stable_rest.py'
sys.path.insert(0, str(LAB / 'src'))
import numpy as np
from wis_lab.core.observation import Observation
spec = importlib.util.spec_from_file_location('frozen_stable_legacy_audit', SRC)
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)


def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def write(p, value):
    assert p.parent == U and p.name.startswith('root_audit_legacy_preflight')
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def diagnose(obs):
    stop = int(np.searchsorted(obs.timestamp_us, obs.timestamp_us[0] + stable.FIRST_US, side='right'))
    ts = obs.timestamp_us[:stop]
    xy = obs.xy_px[:stop].astype(np.float64)
    req = (11, 12, 13, 14, 33, 54)
    names = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hand_wrist', 'right_hand_wrist')
    valid = obs.observed[:stop, req] & obs.usable[:stop, req] & np.isfinite(xy[:, req]).all(-1)
    allvalid = valid.all(1)
    funnel = Counter()
    first_rejection = Counter()
    geometry_failures = Counter()
    nearest_stability = None
    for end in range(stable.MIN_FRAMES - 1, stop):
        for start in range(end - stable.MIN_FRAMES + 2):
            if int(ts[end]) - int(ts[start]) < stable.DWELL_US: continue
            funnel['duration_and_frame_count'] += 1
            if (np.diff(ts[start:end + 1]) > stable.MAX_GAP_US).any():
                first_rejection['gap_over100ms'] += 1; continue
            funnel['plus_gap'] += 1
            if not allvalid[start:end + 1].all():
                first_rejection['required_point_missing_or_unusable'] += 1; continue
            funnel['plus_required_coverage'] += 1
            w = xy[start:end + 1]
            scale = float(np.median(np.linalg.norm(w[:, 11] - w[:, 12], axis=1)))
            if not math.isfinite(scale) or scale <= 0:
                first_rejection['shoulder_scale'] += 1; continue
            funnel['plus_valid_scale'] += 1
            arm_ok = True
            for side, s, e, h in stable._SIDES:
                upper = (w[:, e, 1] - w[:, s, 1]) / scale
                fore = (w[:, h, 1] - w[:, e, 1]) / scale
                horiz = np.abs(w[:, h, 0] - w[:, e, 0]) / scale
                for rule, failed in [('upper_drop_under0.20S', (upper < .20).any()),
                                     ('forearm_drop_under0.20S', (fore < .20).any()),
                                     ('forearm_horizontal_over_half_vertical', (horiz > .5 * fore).any())]:
                    if failed:
                        geometry_failures[f'{side}:{rule}'] += 1
                        arm_ok = False
            if not arm_ok:
                first_rejection['arm_down_geometry'] += 1; continue
            funnel['plus_arms_down'] += 1
            movement = float(np.linalg.norm(w[:, stable._STABLE] - np.median(w[:, stable._STABLE], axis=0), axis=-1).max()) / scale
            if nearest_stability is None or movement < nearest_stability['movement_ratio']:
                nearest_stability = dict(start_frame=int(obs.frame_id[start]), end_frame=int(obs.frame_id[end]), movement_ratio=movement)
            if movement > stable.STABILITY_RATIO:
                first_rejection['stability_over0.05S'] += 1; continue
            funnel['plus_stability'] += 1
            bottom = math.floor(min(float(np.median(w[:, 33, 1])), float(np.median(w[:, 54, 1]))) - stable.BOTTOM_MARGIN_RATIO * scale)
            if not 0 < bottom <= int(obs.sidecar['height']):
                first_rejection['boundary_outside_source'] += 1; continue
            funnel['qualifying_windows'] += 1
    longest = 0
    runstart = None
    for i, yes in enumerate(allvalid):
        if not yes:
            runstart = None; continue
        if runstart is None or (i and ts[i] - ts[i - 1] > stable.MAX_GAP_US): runstart = i
        longest = max(longest, int(ts[i] - ts[runstart]))
    return dict(first_second_frames=stop,
                valid_frames_by_required_point={n: int(valid[:, i].sum()) for i, n in enumerate(names)},
                all_required_valid_frames=int(allvalid.sum()), longest_required_valid_contiguous_us=longest,
                window_funnel=dict(funnel), first_rejection_window_counts=dict(first_rejection),
                arm_geometry_failed_window_counts=dict(geometry_failures), nearest_stable_arms_down_window=nearest_stability,
                limitation='Counts refer to overlapping candidate windows, not independent failures; legacy assignment may differ from new joint_v3')


def main():
    started = time.perf_counter()
    final = U / 'root_audit_legacy_preflight_RESULT.json'
    assert not final.exists(), 'Read existing forecast instead of repeating it'
    rows = [r for r in read(OLD / 'MANIFEST.json') if r['role'] == 'train_candidate']
    selected = []
    for code in sorted({r['word_code'] for r in rows}):
        selected.append(min((r for r in rows if r['word_code'] == code), key=lambda r: (r['signer_id'], r['clip_id'])))
    assert len(selected) == 100 and all(r['role'] == 'train_candidate' for r in selected)
    pins = {str(p): sha(p) for p in [OLD / 'MANIFEST.json', OLD / 'INPUT_CHECK.json', SRC, Path(__file__)]}
    assert pins[str(SRC)] == '366b0cf653a9a41136f716ff218a2da67dc12a13128556c561fc6df9ef8f910b'
    checked = read(OLD / 'INPUT_CHECK.json')
    write(U / 'root_audit_legacy_preflight_PLAN.json', dict(status='FROZEN_BEFORE_FORECAST',
          observation_source='legacy_preflight, original production pose_wrist_nearest assignment; not joint_v3',
          selection_rule='Each fixed word code: lowest eligible TRAIN signer ID, then lexical clip_id; exactly100 representatives',
          selected=[dict(clip_id=r['clip_id'], word_code=r['word_code'], signer_id=r['signer_id']) for r in selected],
          source_pins=pins, detector_execution=0, classifier_execution=0, training=0, heldout_or_user_access=0))
    results = []
    for r in selected:
        rp = DATA / 'records' / f"{r['clip_id']}.json"
        assert sha(rp) == checked['record_hashes'][r['clip_id']]
        rec = read(rp)
        op = Path(rec['observation'])
        sp = op.with_suffix('.sidecar.json')
        assert sha(op) == rec['observation_sha256']
        pins.update({str(rp): sha(rp), str(op): sha(op), str(sp): sha(sp)})
        obs = Observation.load(str(op))
        cal = stable.calibrate_rest(obs, int(obs.sidecar['width']), int(obs.sidecar['height']))
        detail = diagnose(obs)
        assert (cal['status'] == 'CALIBRATED') == (detail['window_funnel'].get('qualifying_windows', 0) > 0)
        results.append(dict(clip_id=r['clip_id'], label=r['canonical_label'], signer_id=r['signer_id'],
                            observation=str(op), observation_sha256=rec['observation_sha256'],
                            observation_policy=obs.sidecar.get('hand_assignment_policy'), legacy_preflight=True,
                            calibration=cal, diagnostics=detail))
    failures = [r for r in results if r['calibration']['status'] != 'CALIBRATED']
    assert all(sha(p) == h for p, h in pins.items()), 'Read-only source changed'
    report = dict(status='LEGACY_PREFLIGHT_COMPLETE_NOT_CANDIDATE_READINESS', representatives=100,
                  calibrated=100-len(failures), not_calibrated=len(failures),
                  failure_clip_ids=[r['clip_id'] for r in failures],
                  meaning='Exact frozen calibration on legacy observations only. Failure predicts review priority, not confirmed joint_v3 failure; no rule changed.',
                  next='Check newjoint raw/assigned observation and first-second source video for each listed case; no later-pose fallback, threshold relaxation or silent removal.',
                  elapsed_seconds=time.perf_counter()-started, results=results, source_pins=pins,
                  all_source_pins_unchanged=True, detector_execution=0, classifier_execution=0, training=0, heldout_or_user_access=0)
    write(final, report)
    print(json.dumps({k: report[k] for k in ('status','representatives','calibrated','not_calibrated','failure_clip_ids','elapsed_seconds')}, ensure_ascii=False))


if __name__ == '__main__': main()
