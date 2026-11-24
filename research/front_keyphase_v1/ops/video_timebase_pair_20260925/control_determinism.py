"""Control: same video, same timestamps, two fresh Landmarkers. Any difference here is
run-to-run extractor noise, not the time base. Without this the A/B numbers cannot be read.
"""
import json, pathlib, sys, time
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
from wis_lab.capture.landmarker import Landmarker          # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402

CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    started, rows = time.time(), []
    for rel in plan['material']['landmark_level_all_6_videos']:
        v = L / rel
        obs = []
        for _ in range(2):
            with Landmarker(CONTRACT) as lm:
                obs.append(extract(str(v), lm, v.stem))
        a, b = obs
        seen = a.observed & b.observed
        d = np.abs(a.xy_px[seen] - b.xy_px[seen])
        r = {'clip_id': v.stem, 'frames': int(a.xy_px.shape[0]),
             'xy_identical': bool(np.array_equal(a.xy_px[seen], b.xy_px[seen])),
             'max_abs_dx_px': round(float(d.max()), 5), 'mean_abs_dx_px': round(float(d.mean()), 5),
             'usable_flag_differences': int((a.usable != b.usable).sum())}
        rows.append(r)
        print(f"{r['clip_id']:<26} identical={r['xy_identical']!s:<5} "
              f"mean|dx| {r['mean_abs_dx_px']:>8.3f} px  max|dx| {r['max_abs_dx_px']:>9.3f} px  "
              f"usable-diff {r['usable_flag_differences']:>4}", flush=True)
    det = all(r['xy_identical'] for r in rows)
    out = {'schema': 'wis_extractor_determinism_control_v1', 'at_kst': '2026-09-25',
           'question': 'Is the extractor bit-reproducible across runs on identical input?',
           'rows': rows, 'deterministic': det,
           'consequence': ('bit-identity is a valid criterion; the A/B difference is the time base'
                           if det else
                           'bit-identity is NOT a valid criterion. Run-to-run noise exists, so the '
                           'A/B difference must be read against THIS noise floor, and any part of '
                           'it not exceeding the floor is not attributable to the time base.'),
           'training_runs': 0, 'downloads': 0, 'model_changed': 0, 'user_data': 0,
           'seconds': round(time.time() - started, 1)}
    with (H / 'CONTROL_DETERMINISM.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print('\ndeterministic:', det, '|', out['seconds'], 's')


if __name__ == '__main__':
    main()
