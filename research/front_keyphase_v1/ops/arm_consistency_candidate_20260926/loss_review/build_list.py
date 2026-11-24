"""Keyed list (clip, frame, side) of slots the FROZEN arm candidate lacks vs the operating baseline,
with existing verdicts attached, plus a decision trace of why each slot is empty (instrumentation only,
decisions asserted unchanged). Candidate code is not modified."""
import json, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).resolve().parent
ARM = H.parent
REC = ARM.parent / 'temporal_recovery_candidate_20260926'
sys.path[:0] = [str(ARM), str(REC)]
import slot_review as sr                                  # noqa: E402
from arm_landmarker import ArmLandmarker                  # noqa: E402

class Trace(ArmLandmarker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.trace = {}
    def _assign_sides(self, hand_res, pose_xy, pose_usable, w, h):
        before = dict(self.reasons)
        n = len(hand_res.hand_landmarks)
        wr = [np.array([hand_res.hand_landmarks[i][0].x * w, hand_res.hand_landmarks[i][0].y * h]) for i in range(n)]
        sh_ok = bool(pose_usable[11] and pose_usable[12])
        sw = float(np.linalg.norm(pose_xy[11] - pose_xy[12])) if sh_ok else None
        out = super()._assign_sides(hand_res, pose_xy, pose_usable, w, h)
        self.trace[round(self._t_s, 4)] = {
            'shoulders_usable': sh_ok, 'pose_wrist_usable': {'L': bool(pose_usable[15]), 'R': bool(pose_usable[16])},
            'n_det': n, 'out': {i: ('LR'[s] if s is not None else None) for i, (s, _) in out.items()},
            'wrist_d': {f'{i}{"LR"[s]}': round(float(np.linalg.norm(wr[i] - pose_xy[idx]) / sw), 3)
                        for i in range(n) for s, idx in ((0, 15), (1, 16)) if sw and pose_usable[idx]},
            'reasons': {k: self.reasons[k] - before.get(k, 0) for k in self.reasons if self.reasons[k] != before.get(k, 0)},
            'tracks_after': sorted('LR'[s] for s in self.tracks)}
        return out

rv = json.loads((sr.SA / 'RUN_REVIEW.json').read_text(encoding='utf-8'))['rows']
known = {}
for r in rv:
    for f in range(r['run'][0], r['run'][-1] + 1):
        known[(r['clip'], r['side'], f)] = {'source': 'side_assignment RUN_REVIEW (run-level)', 'category': r['category'],
                                            'frame_displayed': f in r['frames_displayed_inside_the_run']}
for c, f, s in sr.ERRORS:
    known[(c, 'LR'[s], f)] = {'source': 'FROZEN_ERROR (frame displayed)', 'category': 'WRONG_ASSIGNMENT', 'frame_displayed': True}
known[('NIKL_JOESONGHADA_01', 'L', 15)] = {'source': 'wrist_gap CLASSIFICATION + LOSS_DIAGNOSIS.md §3',
    'category': 'WRONG_SIDE_BINDING_REMOVED (real right hand had been bound to L; baseline assignment was already wrong)',
    'frame_displayed': True}

rows = []
for p in sorted(sr.CACHE.glob('*.npz')):
    clip = p.stem
    z = np.load(p, allow_pickle=False)
    src = json.loads((sr.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
    with Trace(sr.CONTRACT) as lm:
        a = sr.extract(src, lm, clip); tr = lm.trace
    ref, _ = sr.run(ArmLandmarker, src, clip)
    assert np.array_equal(a.usable, ref.usable) and np.array_equal(np.nan_to_num(a.xy_px), np.nan_to_num(ref.xy_px))
    _, lost, _ = sr.decompose(z['xy_px'], z['usable'], a.xy_px, a.usable)
    fids = list(map(int, z['frame_id']))
    for r, s in sorted(lost):
        k = (clip, 'LR'[s], fids[r])
        t = round(float(a.timestamp_us[r]) / 1e6, 4)
        rows.append({'clip': clip, 'frame': fids[r], 'side': 'LR'[s], 'existing': known.get(k),
                     'other_side_filled': {'baseline': bool(z['usable'][r, sr._rc.SLOT[1 - s]].any()),
                                           'arm': bool(a.usable[r, sr._rc.SLOT[1 - s]].any())},
                     'trace': tr.get(t, 'NO_TRACE (shoulders unusable -> operating fallback path)')})
todo = [r for r in rows if r['existing'] is None]
doc = {'lost_vs_baseline': len(rows), 'with_existing_verdict': len(rows) - len(todo), 'to_review': len(todo),
       'why_31_vs_33': 'recovery lacked 59 slots vs baseline: 24 in existing wrong-assignment verdicts + 35 without. '
                       'SLOT_REVIEW (previous unit) judged 2 of the 35 on video (죄송하다 R65, R71) -> 33 unreviewed for recovery. '
                       'The arm candidate then holds 4 of the 35 (R63, R64, R65, R71) -> 55 lacked = 24 + 31. '
                       'The "33 unchanged" wording in arm RESULT.md/REVIEW.json carried the recovery-era number and was wrong; 31 is correct. '
                       'Of the 31, 죄송하다 L15 already has an existing verdict, so this list reviews the rest.',
       'rows': rows}
(H / 'LOSS_LIST.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
print('lost', len(rows), 'existing', len(rows) - len(todo), 'to_review', len(todo))
for r in todo:
    t = r['trace']
    print(r['clip'][5:], r['side'], r['frame'], 'other', r['other_side_filled'],
          t if isinstance(t, str) else (t['n_det'], t['out'], t['pose_wrist_usable'], t['wrist_d'], t['reasons'], t['tracks_after']))
