"""A/B on the SAME video: real time base (offline/training) vs the collapsed 1 ms time base
the web path produces from the second recording of a session onward.

Only the timestamp handed to MediaPipe differs. Model frozen, build_channels unmodified.
"""
import hashlib, json, pathlib, sys, time
import numpy as np
import torch

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
COLLAPSED_LAST_MS = 10_000_000          # far past any clip: the clamp fires on every frame


def run(video, collapse):
    with Landmarker(CONTRACT) as lm:
        if collapse:
            lm._last_ms = COLLAPSED_LAST_MS
        obs = extract(str(video), lm, pathlib.Path(video).stem)
    return obs


def window(obs, start_s, end_s):
    rel = (obs.timestamp_us - obs.timestamp_us[0]) / 1e6
    sel = np.flatnonzero((rel >= start_s) & (rel < end_s))
    nat, present = to_native_like(obs.xy_px[sel], obs.usable[sel])
    return build_channels(nat, present, obs.timestamp_us[sel]), int(sel.size)


def score(model, classes, clip, label, idx18):
    xx, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(xx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
    yi = classes.index(label)
    out = {}
    for scope, inds in [('full93', list(range(93))), ('project18', idx18)]:
        order = sorted(inds, key=lambda j: -float(logit[j]))
        rival = next(j for j in order if j != yi)
        out[scope] = {'pred': classes[order[0]], 'correct': bool(order[0] == yi),
                      'true_rank': order.index(yi) + 1,
                      'margin': round(float(logit[yi] - logit[rival]), 5)}
    return out, logit


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FROZEN_BEFORE_EXECUTION'
    started = time.time()

    ckpt = L / plan['model']['path']
    assert sha(ckpt) == plan['model']['sha256']
    model, classes = u.rebuild_from_payload(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval(); torch.set_num_threads(2)
    from wis_lab.service.neural import PROJECT_18
    idx18 = [i for i, c in enumerate(classes) if c in set(PROJECT_18.values())]
    scored = {s['clip_id']: s for s in plan['material']['scored_subset_4_clips']}

    rows = []
    for rel in plan['material']['landmark_level_all_6_videos']:
        v = L / rel
        cid = v.stem
        a, b = run(v, False), run(v, True)
        assert a.xy_px.shape == b.xy_px.shape
        seen = a.observed & b.observed            # NaN fills unobserved slots in both arms
        d = np.abs(a.xy_px[seen] - b.xy_px[seen])
        r = {'clip_id': cid, 'frames': int(a.xy_px.shape[0]),
             'landmark': {
                 'identical': bool(d.size and np.array_equal(a.xy_px[seen], b.xy_px[seen])),
                 'points_observed_in_both': int(seen.sum()),
                 'max_abs_dx_px': round(float(d.max()), 5) if d.size else 0.0,
                 'mean_abs_dx_px': round(float(d.mean()), 5) if d.size else 0.0,
                 'observed_flag_differences': int((a.observed != b.observed).sum()),
                 'usable_flag_differences': int((a.usable != b.usable).sum()),
                 'usable_entries': int(a.usable.size)},
             'hand_rate': {
                 'A_left': round(float(a.usable[:, 33:54].any(axis=1).mean()), 4),
                 'B_left': round(float(b.usable[:, 33:54].any(axis=1).mean()), 4),
                 'A_right': round(float(a.usable[:, 54:75].any(axis=1).mean()), 4),
                 'B_right': round(float(b.usable[:, 54:75].any(axis=1).mean()), 4)}}
        if cid in scored:
            s = scored[cid]
            ca, na = window(a, *s['window_s'])
            cb, nb = window(b, *s['window_s'])
            sa, la = score(model, classes, ca, s['label'], idx18)
            sb, lb = score(model, classes, cb, s['label'], idx18)
            r['scored'] = {'label': s['label'], 'window_s': s['window_s'],
                           'window_frames': {'A': na, 'B': nb},
                           'A': sa, 'B': sb,
                           'same_prediction': sa['full93']['pred'] == sb['full93']['pred'],
                           'max_logit_difference': round(float(np.max(np.abs(la - lb))), 5)}
        rows.append(r)
        m = r['landmark']
        print(f"{cid:<26} frames {r['frames']:>4}  identical={m['identical']!s:<5} "
              f"max|dx| {m['max_abs_dx_px']:>9.4f} px  usable-diff {m['usable_flag_differences']:>6}",
              flush=True)
        if 'scored' in r:
            g = r['scored']
            print(f"    A {g['A']['full93']['pred']:<8} rank {g['A']['full93']['true_rank']:>2} "
                  f"margin {g['A']['full93']['margin']:+8.3f} | "
                  f"B {g['B']['full93']['pred']:<8} rank {g['B']['full93']['true_rank']:>2} "
                  f"margin {g['B']['full93']['margin']:+8.3f}")

    any_diff = any(not r['landmark']['identical'] for r in rows)
    out = {'schema': 'wis_video_timebase_pair_result_v1', 'at_kst': '2026-09-25',
           'framing': plan['framing'], 'plan_sha256': sha(H / 'FROZEN_PLAN.json'),
           'model': plan['model'], 'rows': rows,
           'any_landmark_difference': any_diff,
           'verdict': ('DEFECT_HAS_MEASURED_EFFECT' if any_diff
                       else 'DEFECT_CONFIRMED_IN_CODE_NO_MEASURED_EFFECT'),
           'training_runs': 0, 'downloads': 0, 'model_changed': 0, 'user_data': 0,
           'reserved_signers': 0, 'reshoots': 0,
           'limits': plan['limits_stated_in_advance'], 'seconds': round(time.time() - started, 1)}
    with (H / 'RESULT.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print('\nverdict:', out['verdict'], '|', out['seconds'], 's')


if __name__ == '__main__':
    main()
