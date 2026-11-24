"""죄송하다 f60-80: which detection sits in which slot, in all three arms (reference only)."""
import json, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import slot_review as sr
clip = 'NIKL_JOESONGHADA_01'
z = np.load(sr.CACHE / f'{clip}.npz', allow_pickle=False)
src = json.loads((sr.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
o, olog = sr.run(sr.TemporalLandmarker, src, clip)
c, clog = sr.run(sr.RecoveryLandmarker, src, clip)
arms = {'B': (z['xy_px'], z['usable']), 'O': (o.xy_px, o.usable), 'C': (c.xy_px, c.usable)}
fids = list(map(int, z['frame_id']))
rows = []
for r, f in enumerate(fids):
    if not 60 <= f <= 80:
        continue
    t = round(float(c.timestamp_us[r]) / 1e6, 4)
    dets = [np.array(p) for p in clog[t]['palms']] if t in clog else []
    slots = {a: {'LR'[s]: sr.det_id(sr.content(xy, us, r, s), dets) for s in (0, 1)} for a, (xy, us) in arms.items()}
    g = [sr.geom(d, z['xy_px'][r], z['usable'][r]) for d in dets]
    rows.append({'frame': f, 'slots': slots, 'detections': g})
    print(f, slots, [(k, x['L'].get('d_wrist'), x['R'].get('d_wrist'), x['R'].get('cos_forearm')) for k, x in enumerate(g) if x])
(H / 'JOE_WINDOW.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
