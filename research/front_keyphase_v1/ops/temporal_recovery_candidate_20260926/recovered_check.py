"""The 135 slots recovery holds with BASELINE content: baseline can be wrong too. Cross-check each
against known-wrong frames and flag any whose wrist sits closer to the OTHER arm (reference only)."""
import json, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import slot_review as sr
rv = json.loads((sr.SA / 'RUN_REVIEW.json').read_text(encoding='utf-8'))['rows']
wrong = {(r['clip'], r['side'], f) for r in rv for f in range(r['run'][0], r['run'][-1] + 1)} | {(c, 'LR'[s], f) for c, f, s in sr.ERRORS}
res = json.loads((H / 'RESULT.json').read_text(encoding='utf-8'))['nikl']
out, flags = [], []
for rec in res:
    if not rec['recovered_with_baseline_content']:
        continue
    clip = rec['clip']
    z = np.load(sr.CACHE / f'{clip}.npz', allow_pickle=False)
    src = json.loads((sr.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
    o, _ = sr.run(sr.TemporalLandmarker, src, clip)
    c, _ = sr.run(sr.RecoveryLandmarker, src, clip)
    bxy, bus = z['xy_px'], z['usable']
    _, lost_ob, _ = sr.decompose(bxy, bus, o.xy_px, o.usable)
    fids = list(map(int, z['frame_id']))
    n = 0
    for r, s in sorted(lost_ob):
        bc = sr.content(bxy, bus, r, s)
        if not sr.same(bc, sr.content(c.xy_px, c.usable, r, s)):
            continue
        n += 1
        g = sr.geom(bc, bxy[r], bus[r]) or {}
        own, oth = g.get('LR'[s], {}).get('d_wrist'), g.get('LR'[1 - s], {}).get('d_wrist')
        e = {'clip': clip, 'frame': fids[r], 'side': 'LR'[s], 'd_own': own, 'd_other': oth,
             'known_wrong': (clip, 'LR'[s], fids[r]) in wrong}
        if e['known_wrong'] or (own is not None and oth is not None and oth < own) or own is None:
            flags.append(e)
        out.append(e)
    print(clip, n, flush=True)
doc = {'recovered_slots': len(out), 'flagged': flags,
       'd_own_max': max(e['d_own'] for e in out if e['d_own'] is not None), 'slots': out}
(H / 'RECOVERED_CHECK.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
print('recovered', len(out), 'flagged', len(flags), flags, 'd_own_max', doc['d_own_max'])
