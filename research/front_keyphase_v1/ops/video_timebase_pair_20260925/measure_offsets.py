"""How much does the ABSOLUTE time origin matter, once spacing is correct?
Arm A is the reference (origin 0). Each offset arm keeps real spacing and shifts the origin.
"""
import json, pathlib, sys, time
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
from wis_lab.capture.landmarker import Landmarker          # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402

CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
OFFSETS_S = [0, 10, 60, 1000]


def shifted(off_us):
    class C(Landmarker):
        def process(self, rgb, timestamp_us, out, row):
            return super().process(rgb, timestamp_us + off_us, out, row)
    return C


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    vids = [L / r for r in plan['material']['landmark_level_all_6_videos'][:3]]
    started, rows = time.time(), []
    for v in vids:
        with Landmarker(CONTRACT) as lm:
            ref = extract(str(v), lm, v.stem)
        rec = {'clip_id': v.stem, 'frames': int(ref.xy_px.shape[0]), 'offsets': {}}
        for s in OFFSETS_S:
            off = s * 1_000_000
            cls = shifted(off)
            with cls(CONTRACT) as lm:
                lm._last_ms = off // 1000 - 1
                o = extract(str(v), lm, v.stem)
            seen = ref.observed & o.observed
            d = np.abs(ref.xy_px[seen] - o.xy_px[seen])
            rec['offsets'][f'{s}s'] = {
                'identical': bool(np.array_equal(ref.xy_px[seen], o.xy_px[seen])),
                'mean_abs_dx_px': round(float(d.mean()), 6),
                'max_abs_dx_px': round(float(d.max()), 5),
                'usable_flag_differences': int((ref.usable != o.usable).sum())}
        rows.append(rec)
        line = '  '.join(f"{k}: mean {v['mean_abs_dx_px']:.4f} max {v['max_abs_dx_px']:.3f}"
                         for k, v in rec['offsets'].items())
        print(f"{rec['clip_id']:<26} {line}", flush=True)
    out = {'schema': 'wis_timebase_offset_sensitivity_v1', 'at_kst': '2026-09-25',
           'reference': 'arm A: fresh Landmarker, origin 0, real spacing',
           'note': 'offset 0s with the clamp armed is the exact fixed behaviour for the FIRST '
                   'recording; later recordings carry a real session offset',
           'rows': rows, 'training_runs': 0, 'downloads': 0, 'model_changed': 0, 'user_data': 0,
           'seconds': round(time.time() - started, 1)}
    with (H / 'OFFSET_SENSITIVITY.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print('\n', out['seconds'], 's')


if __name__ == '__main__':
    main()
