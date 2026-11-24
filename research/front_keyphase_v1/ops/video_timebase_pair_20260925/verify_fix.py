"""Arm C = the fixed web path: Landmarker continued from an earlier recording (clamp armed) but
fed a SESSION-continuous clock, so real spacing survives. Must reproduce arm A exactly.
"""
import hashlib, json, pathlib, sys, time
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
from wis_lab.capture.landmarker import Landmarker          # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
OFFSET_US = 1_000_000_000            # 1000 s: the session was already running


class SessionClock(Landmarker):
    """What camera.py now does: the frame's SESSION time reaches MediaPipe, not the take's."""
    def process(self, rgb, timestamp_us, out, row):
        return super().process(rgb, timestamp_us + OFFSET_US, out, row)


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    src = pathlib.Path(L / 'src' / 'wis_lab' / 'service' / 'camera.py').read_text(encoding='utf-8')
    assert 'lm.process(rgb, now_ns // 1000, row, 0)' in src, 'camera.py fix not applied'
    started = time.time()
    rows, clamp_fired = [], 0
    for rel in plan['material']['landmark_level_all_6_videos']:
        v = L / rel
        with Landmarker(CONTRACT) as a:
            oa = extract(str(v), a, v.stem)
        with SessionClock(CONTRACT) as c:
            c._last_ms = OFFSET_US // 1000 - 1        # continued session, clamp armed
            before = c._last_ms
            oc = extract(str(v), c, v.stem)
            spacing = c._last_ms - before
        same_xy = bool(np.array_equal(np.nan_to_num(oa.xy_px, nan=-1.0),
                                      np.nan_to_num(oc.xy_px, nan=-1.0)))
        r = {'clip_id': v.stem, 'frames': int(oa.xy_px.shape[0]),
             'xy_identical': same_xy,
             'observed_identical': bool(np.array_equal(oa.observed, oc.observed)),
             'usable_identical': bool(np.array_equal(oa.usable, oc.usable)),
             'timestamps_identical': bool(np.array_equal(oa.timestamp_us, oc.timestamp_us)),
             'mediapipe_ms_span': int(spacing)}
        rows.append(r)
        print(f"{r['clip_id']:<26} frames {r['frames']:>4}  xy={r['xy_identical']!s:<5} "
              f"observed={r['observed_identical']!s:<5} usable={r['usable_identical']!s:<5} "
              f"ts={r['timestamps_identical']!s:<5} mp_span={r['mediapipe_ms_span']} ms", flush=True)

    ok = all(r['xy_identical'] and r['observed_identical'] and r['usable_identical']
             and r['timestamps_identical'] for r in rows)
    out = {'schema': 'wis_video_timebase_fix_verification_v1', 'at_kst': '2026-09-25',
           'fix': 'wis_lab/service/camera.py::_record_frame now hands MediaPipe now_ns//1000 '
                  '(session-continuous) instead of the per-recording ts_us. The STORED '
                  'timestamp_us, the dt channel and build_channels are unchanged.',
           'applies_to': 'every user identically; no word-specific exception',
           'arm_C': 'Landmarker continued from an earlier recording with the clamp armed, fed a '
                    'session-continuous clock',
           'rows': rows,
           'verdict': 'FIX_REPRODUCES_REFERENCE_EXACTLY' if ok else 'FIX_DOES_NOT_REPRODUCE',
           'meaning': 'the absolute time origin does not matter to the extractor; only the '
                      'spacing does, and the fix restores it' if ok else 'offset alone is not '
                      'sufficient; a per-recording reset is also required',
           'training_runs': 0, 'downloads': 0, 'model_changed': 0, 'user_data': 0,
           'seconds': round(time.time() - started, 1)}
    with (H / 'FIX_VERIFICATION.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print('\nverdict:', out['verdict'], '|', out['seconds'], 's')


if __name__ == '__main__':
    main()
