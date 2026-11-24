"""Scores under the FIXED path for a non-first recording (session offset, real spacing),
side by side with the reference (A) and the defective behaviour (B). Model frozen.
"""
import hashlib, json, pathlib, sys, time
import numpy as np, torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
import tools_prep_union93_candidate as u                    # noqa: E402
from pilot.data import build_channels, collate             # noqa: E402
from pilot.web_input import to_native_like                 # noqa: E402
from wis_lab.capture.landmarker import Landmarker          # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
OFF_US = 60 * 1_000_000          # a recording one minute into the session


class Fixed(Landmarker):
    def process(self, rgb, timestamp_us, out, row):
        return super().process(rgb, timestamp_us + OFF_US, out, row)


def score(model, classes, obs, s, idx18):
    rel = (obs.timestamp_us - obs.timestamp_us[0]) / 1e6
    sel = np.flatnonzero((rel >= s['window_s'][0]) & (rel < s['window_s'][1]))
    nat, pres = to_native_like(obs.xy_px[sel], obs.usable[sel])
    clip = build_channels(nat, pres, obs.timestamp_us[sel])
    xx, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(xx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
    yi = classes.index(s['label'])
    out = {}
    for scope, inds in [('full93', list(range(93))), ('project18', idx18)]:
        order = sorted(inds, key=lambda j: -float(logit[j]))
        rival = next(j for j in order if j != yi)
        out[scope] = {'pred': classes[order[0]], 'correct': bool(order[0] == yi),
                      'true_rank': order.index(yi) + 1,
                      'margin': round(float(logit[yi] - logit[rival]), 5)}
    return out


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    prev = json.loads((H / 'RESULT.json').read_text(encoding='utf-8'))
    ckpt = L / plan['model']['path']
    assert sha(ckpt) == plan['model']['sha256']
    model, classes = u.rebuild_from_payload(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    from wis_lab.service.neural import PROJECT_18
    idx18 = [i for i, c in enumerate(classes) if c in set(PROJECT_18.values())]
    byid = {r['clip_id']: r for r in prev['rows']}
    started, rows = time.time(), []
    for s in plan['material']['scored_subset_4_clips']:
        v = next(L / r for r in plan['material']['landmark_level_all_6_videos']
                 if pathlib.Path(r).stem == s['clip_id'])
        with Fixed(CONTRACT) as lm:
            lm._last_ms = OFF_US // 1000 - 1
            obs = extract(str(v), lm, v.stem)
        fx = score(model, classes, obs, s, idx18)
        old = byid[s['clip_id']]['scored']
        r = {'clip_id': s['clip_id'], 'label': s['label'],
             'A_reference': old['A']['full93'], 'B_defective': old['B']['full93'],
             'C_fixed': fx['full93'],
             'margin_change_fixed_minus_defective': round(
                 fx['full93']['margin'] - old['B']['full93']['margin'], 5),
             'gain': bool(fx['full93']['correct'] and not old['B']['full93']['correct']),
             'regression': bool(old['B']['full93']['correct'] and not fx['full93']['correct'])}
        rows.append(r)
        print(f"{r['clip_id']:<26} {s['label']:<6} A {old['A']['full93']['margin']:+8.3f} | "
              f"B(defect) {old['B']['full93']['margin']:+8.3f} | C(fixed) {fx['full93']['margin']:+8.3f} "
              f"| delta {r['margin_change_fixed_minus_defective']:+7.3f} "
              f"| {'GAIN' if r['gain'] else 'REGRESSION' if r['regression'] else 'same answer'}")
    out = {'schema': 'wis_timebase_after_fix_scores_v1', 'at_kst': '2026-09-25',
           'model': plan['model'], 'arms': {
               'A_reference': 'fresh extractor, origin 0 - the offline/training time base',
               'B_defective': 'the web path from the 2nd recording of a session onward, 1 ms spacing',
               'C_fixed': 'the patched web path, 60 s into a session, real spacing'},
           'rows': rows,
           'gains': sum(r['gain'] for r in rows), 'regressions': sum(r['regression'] for r in rows),
           'all_correct_in_every_arm': all(r[k]['correct'] for r in rows
                                           for k in ('A_reference', 'B_defective', 'C_fixed')),
           'honest_reading': 'no prediction changed on these 4 clips in any arm. The fix is '
                             'justified because it removes a confirmed mismatch with the time base '
                             'the model was built on, NOT because a recognition gain was measured.',
           'training_runs': 0, 'downloads': 0, 'model_changed': 0, 'user_data': 0,
           'seconds': round(time.time() - started, 1)}
    with (H / 'AFTER_FIX_SCORES.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(f"\ngains {out['gains']} | regressions {out['regressions']} | {out['seconds']} s")


if __name__ == '__main__':
    main()
