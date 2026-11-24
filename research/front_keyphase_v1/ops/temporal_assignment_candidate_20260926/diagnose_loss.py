"""Separate the causes of the assignment drop and the slower reconnection.

No new candidate, no coefficient search, no interpolation. One instrumented pass of the SAME
candidate; the baseline comes from the stored cache, which is bit-deterministic.
"""
import hashlib, json, pathlib, sys, time
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
sys.path.insert(0, str(H))
from wis_lab.capture.video_source import extract          # noqa: E402
from temporal_landmarker import TemporalLandmarker        # noqa: E402

CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
CACHE = L / 'cache' / '2683eadbc5f1c588'
SLOT = {0: slice(33, 54), 1: slice(54, 75)}
PALM = (0, 5, 9, 13, 17)


def palm_of(xy, side):
    b = 33 if side == 0 else 54
    return xy[[b + j for j in PALM]]


def main():
    stored = json.loads((H / 'RESULT.json').read_text(encoding='utf-8'))
    by_clip = {r['clip']: r for r in stored['nikl']}
    started = time.time()
    tot = {'lost': 0, 'new': 0, 'swapped': 0, 'unchanged': 0}
    reasons = {}
    rows = []
    mismatch = []

    for p in sorted(CACHE.glob('*.npz')):
        z = np.load(p, allow_pickle=False)
        src = json.loads((CACHE / f'{p.stem}.sidecar.json').read_text(encoding='utf-8'))['source_path']
        with TemporalLandmarker(CONTRACT) as lm:
            c = extract(src, lm, p.stem)
            rs, stt = dict(lm.reasons), dict(lm.stats)
        b_us, c_us = z['usable'], c.usable
        assert b_us.shape == c_us.shape
        # behaviour check: the instrumented pass must reproduce the stored aggregate exactly
        got = int(np.stack([c_us[:, 33:54].any(1), c_us[:, 54:75].any(1)], 1).sum())
        want = by_clip[p.stem]['sides_assigned']['candidate']
        if got != want:
            mismatch.append((p.stem, got, want))

        lost = new = swap = unchanged = 0
        swap_frames = []
        for r in range(b_us.shape[0]):
            for s in (0, 1):
                bb = bool(b_us[r, SLOT[s]].any())
                cc = bool(c_us[r, SLOT[s]].any())
                if bb and not cc:
                    # is the SAME detection now on the other side? then it is a swap, not a loss
                    o = 1 - s
                    if c_us[r, SLOT[o]].any() and not b_us[r, SLOT[o]].any():
                        if np.allclose(palm_of(z['xy_px'][r], s), palm_of(c.xy_px[r], o), atol=1.0):
                            swap += 1; swap_frames.append([int(z['frame_id'][r]), 'LR'[s]]); continue
                    lost += 1
                elif cc and not bb:
                    o = 1 - s
                    if b_us[r, SLOT[o]].any() and not c_us[r, SLOT[o]].any():
                        if np.allclose(palm_of(c.xy_px[r], s), palm_of(z['xy_px'][r], o), atol=1.0):
                            continue          # already counted as a swap from the other side
                    new += 1
                elif bb and cc:
                    unchanged += 1
        for k, v in (('lost', lost), ('new', new), ('swapped', swap), ('unchanged', unchanged)):
            tot[k] += v
        for k, v in rs.items():
            reasons[k] = reasons.get(k, 0) + v
        rows.append({'clip': p.stem, 'lost': lost, 'new': new, 'swapped': swap,
                     'unchanged': unchanged, 'swap_frames': swap_frames, 'reasons': rs,
                     'stats': stt})
        print(f'{p.stem:<24} lost {lost:>3}  new {new:>3}  swapped {swap:>3}  kept {unchanged:>4}',
              flush=True)

    print('\n--- 1. what the -223 actually is ---')
    print(f"  lost (baseline had it, candidate does not) : {tot['lost']}")
    print(f"  new  (candidate has it, baseline did not)  : {tot['new']}")
    print(f"  swapped sides (same detection, other side) : {tot['swapped']}")
    print(f"  unchanged                                  : {tot['unchanged']}")
    print(f"  net = lost - new = {tot['lost'] - tot['new']}")
    print('\n--- 2. which condition refused ---')
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f'  {k:<34}{v:>5}')
    if mismatch:
        print('\n!! instrumented pass did not reproduce the stored aggregate:', mismatch)
    else:
        print('\nbehaviour check: instrumented pass reproduces the stored per-clip totals exactly')

    out = {'schema': 'wis_temporal_loss_diagnosis_v1', 'at_kst': '2026-09-26',
           'note': 'diagnosis only. The candidate is unchanged and stays FAILED and unapplied. No '
                   'coefficient search, no new candidate, no interpolation.',
           'baseline_source': 'the stored deterministic cache extraction',
           'instrumentation': 'reason counters and a per-frame log were ADDED; no decision changed, '
                              'verified by reproducing the stored per-clip candidate totals',
           'decomposition': tot, 'refusal_reasons': reasons, 'per_clip': rows,
           'aggregate_check': {'reproduced': not mismatch, 'mismatches': mismatch},
           'seconds': round(time.time() - started, 1)}
    with (H / 'LOSS_DIAGNOSIS.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(f"\nwritten LOSS_DIAGNOSIS.json | {out['seconds']} s")


if __name__ == '__main__':
    main()
