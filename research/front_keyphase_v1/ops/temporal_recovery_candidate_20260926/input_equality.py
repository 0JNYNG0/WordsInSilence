"""Which recovery TCN inputs (coords + mask + active span + timestamps, i.e. build_channels output)
are identical to an already-scored input? Only inputs equal to neither need scoring."""
import json, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import slot_review as sr
rc = sr._rc
res = {r['clip']: r for r in json.loads((H / 'RESULT.json').read_text(encoding='utf-8'))['nikl']}
old = {r['clip']: r for r in json.loads((sr.OLD / 'RESULT.json').read_text(encoding='utf-8'))['nikl']}


def eq(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')


rows = []
for p in sorted(sr.CACHE.glob('*.npz')):
    clip = p.stem
    z = np.load(p, allow_pickle=False)
    src = json.loads((sr.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
    o, _ = sr.run(sr.TemporalLandmarker, src, clip)
    c, _ = sr.run(sr.RecoveryLandmarker, src, clip)
    B = rc.channels(z['xy_px'], z['usable'], z['timestamp_us'])
    O = rc.channels(o.xy_px, o.usable, o.timestamp_us)
    C = rc.channels(c.xy_px, c.usable, c.timestamp_us)
    r = {'clip': clip, 'label': res[clip]['label'], 'recovery_eq_baseline': eq(C, B),
         'recovery_eq_original': eq(C, O), 'original_eq_baseline': eq(O, B),
         'recovery_scored_in_RESULT': 'score_recovery' in res[clip]}
    if r['label'] is None:
        r['score_source'] = 'NOT_SCORABLE_label_not_in_vocabulary'
    elif r['recovery_eq_baseline']:
        r['score_source'] = 'baseline (identical input)'; r['recovery_score'] = old[clip]['baseline']
    elif r['recovery_eq_original']:
        r['score_source'] = 'original (identical input)'; r['recovery_score'] = old[clip]['candidate']
    elif r['recovery_scored_in_RESULT']:
        r['score_source'] = 'RESULT.json score_recovery'; r['recovery_score'] = res[clip]['score_recovery']
    else:
        r['score_source'] = 'UNSCORED'
    r['baseline_score'] = old[clip].get('baseline')
    rows.append(r)
    print(clip, r['recovery_eq_baseline'], r['recovery_eq_original'], r['score_source'], flush=True)
(H / 'INPUT_EQUALITY.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
