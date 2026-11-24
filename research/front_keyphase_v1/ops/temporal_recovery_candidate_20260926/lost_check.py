"""The 59 slots recovery still lacks vs baseline: which are ALREADY-confirmed wrong baseline
assignments (existing video verdicts), and which remain unreviewed. No new judgement is made here."""
import json, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import slot_review as sr
rv = json.loads((sr.SA / 'RUN_REVIEW.json').read_text(encoding='utf-8'))['rows']
wrong = {(r['clip'], r['side'], f): 'RUN_REVIEW' for r in rv for f in range(r['run'][0], r['run'][-1] + 1)}
wrong.update({(c, 'LR'[s], f): 'FROZEN_ERROR' for c, f, s in sr.ERRORS})
res = json.loads((H / 'RESULT.json').read_text(encoding='utf-8'))['nikl']
rows = []
for rec in res:
    if not rec['recov_vs_base']['LOST']:
        continue
    clip = rec['clip']
    z = np.load(sr.CACHE / f'{clip}.npz', allow_pickle=False)
    src = json.loads((sr.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
    c, _ = sr.run(sr.RecoveryLandmarker, src, clip)
    _, lost, _ = sr.decompose(z['xy_px'], z['usable'], c.xy_px, c.usable)
    fids = list(map(int, z['frame_id']))
    for r, s in lost:
        k = (clip, 'LR'[s], fids[r])
        rows.append({'clip': clip, 'frame': fids[r], 'side': 'LR'[s], 'existing_verdict': wrong.get(k)})
known = [r for r in rows if r['existing_verdict']]
doc = {'lost_vs_baseline': len(rows), 'already_confirmed_wrong_baseline_assignment': len(known),
       'unreviewed': len(rows) - len(known), 'rows': rows}
(H / 'LOST_CHECK.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
by = {}
for r in rows:
    by.setdefault((r['clip'], bool(r['existing_verdict'])), []).append(f"{r['side']}{r['frame']}")
for k, v in by.items():
    print(k, v)
print('lost', len(rows), 'known_wrong', len(known))
