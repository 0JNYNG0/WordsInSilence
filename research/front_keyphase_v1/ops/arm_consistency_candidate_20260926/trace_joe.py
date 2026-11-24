"""Instrumentation only: log track state, costs and the deciding branch for 죄송하다 f58-75 under the
RECOVERY candidate. The candidate's decision is asserted unchanged (we call the real method)."""
import json, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).resolve().parent
REC = H.parent / 'temporal_recovery_candidate_20260926'
sys.path[:0] = [str(REC)]
import slot_review as sr                                         # noqa: E402
from recovery_landmarker import RecoveryLandmarker, TRACK_GATE, RATIO, INIT_GATE  # noqa: E402

class Trace(RecoveryLandmarker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.trace = []
    def _assign_sides(self, hand_res, pose_xy, pose_usable, w, h):
        t = self._t_s
        pre = {s: (tr.predict(t, 1.0).tolist(), round(t - tr.t, 3)) for s, tr in self.tracks.items()}
        n = len(hand_res.hand_landmarks)
        palms = [np.array([[hand_res.hand_landmarks[i][j].x * w, hand_res.hand_landmarks[i][j].y * h]
                           for j in (0, 5, 9, 13, 17)]) for i in range(n)]
        sw = float(np.linalg.norm(pose_xy[11] - pose_xy[12])) if pose_usable[11] and pose_usable[12] else None
        cost = {}
        if sw:
            for i in range(n):
                for s, tr in self.tracks.items():
                    cost[f'{i}{"LR"[s]}'] = round(float(np.linalg.norm(palms[i] - tr.predict(t, sw), axis=1).mean() / sw), 3)
            dw = {f'{i}{"LR"[s]}': round(float(np.linalg.norm(palms[i][0] - pose_xy[idx]) / sw), 3)
                  for i in range(n) for s, idx in ((0, 15), (1, 16)) if pose_usable[idx]}
        else:
            dw = {}
        before = dict(self.reasons)
        out = super()._assign_sides(hand_res, pose_xy, pose_usable, w, h)
        why = {k: self.reasons[k] - before.get(k, 0) for k in self.reasons if self.reasons[k] != before.get(k, 0)}
        self.trace.append({'t': round(t, 4), 'tracks_before': sorted('LR'[s] for s in pre), 'age_s': {('LR'[s]): v[1] for s, v in pre.items()},
                           'cost': cost, 'wrist_d': dw, 'out': {i: ('LR'[s] if s is not None else None) for i, (s, _) in out.items()},
                           'refusals': why, 'sw_ok': sw is not None})
        return out

clip = 'NIKL_JOESONGHADA_01'
z = np.load(sr.CACHE / f'{clip}.npz')
src = json.loads((sr.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
with Trace(sr.CONTRACT) as lm:
    o = sr.extract(src, lm, clip); tr = lm.trace
c, _ = sr.run(RecoveryLandmarker, src, clip)
assert np.array_equal(o.usable, c.usable)             # instrumentation did not change decisions
tmap = {round(float(x) / 1e6, 4): int(f) for x, f in zip(z['timestamp_us'], z['frame_id'])}
rows = [dict(e, frame=tmap.get(e['t'])) for e in tr if tmap.get(e['t']) is not None and 58 <= tmap[e['t']] <= 75]
for e in rows:
    print(e['frame'], 'tracks', e['tracks_before'], e['age_s'], 'cost', e['cost'], 'wd', e['wrist_d'], 'out', e['out'], e['refusals'])
(H / 'TRACE_JOE.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
