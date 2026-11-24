"""Instrumentation only: SEN0354 rows 36-46 under the recovery candidate - costs, wrist distances, outcome."""
import json, pathlib, sys
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import trace_joe as tj                                     # reuses the Trace class (runs JOE once on import)
import numpy as np
SEN = tj.sr._rc.SEN
with tj.Trace(tj.sr.CONTRACT) as lm:
    o = tj.sr.extract(str(SEN), lm, SEN.stem); tr = lm.trace
tmap = {round(float(x) / 1e6, 4): r for r, x in enumerate(o.timestamp_us)}
rows = [dict(e, row=tmap.get(e['t'])) for e in tr if tmap.get(e['t']) is not None and 36 <= tmap[e['t']] <= 46]
for e in rows:
    print('row', e['row'], e['tracks_before'], 'cost', e['cost'], 'wd', e['wrist_d'], 'out', e['out'], e['refusals'])
(H / 'TRACE_SEN0354.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
