"""Render SEN0354 rows around a new assignment: unmarked | arm output (L orange, R cyan) + arm chains.
Grey thin skeleton = a detection the recovery candidate had in that slot but arm does not."""
import pathlib, sys
import cv2, numpy as np
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import run_compare as rcmp
rc, sr = rcmp.rc, rcmp.sr
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
runs = {}
for tag, cls in (('recovery', sr.RecoveryLandmarker), ('arm', rcmp.ArmLandmarker)):
    with cls(rc.CONTRACT) as lm:
        runs[tag] = rc.extract(str(rc.SEN), lm, rc.SEN.stem)
cap = cv2.VideoCapture(str(rc.SEN))
for row in [int(x) for x in sys.argv[1:]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, row); ok, img = cap.read()
    a, c = runs['arm'], runs['recovery']
    xy, us = a.xy_px[row], a.usable[row]
    pts = np.array([xy[i] for i in range(11, 17) if us[i]] + [xy[j] for j in range(33, 75) if us[j]] +
                   [c.xy_px[row][j] for j in range(33, 75) if c.usable[row][j]])
    x0, y0 = np.maximum(pts.min(0) - 80, 0).astype(int); x1, y1 = (pts.max(0) + 80).astype(int)
    S = 1.5
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=S, fy=S); mk = crop.copy()
    P = lambda p: (int((p[0] - x0) * S), int((p[1] - y0) * S))
    for (sh, el, wr), col in (((11, 13, 15), (0, 220, 0)), ((12, 14, 16), (220, 0, 220))):
        ch = [i for i in (sh, el, wr) if us[i]]
        for u, v in zip(ch, ch[1:]): cv2.line(mk, P(xy[u]), P(xy[v]), col, 2)
    for hb, col, name in ((33, (0, 150, 255), 'L'), (54, (255, 200, 0), 'R')):
        if not c.usable[row][hb] or (us[hb] and np.allclose(c.xy_px[row][hb], xy[hb])):
            pass
        else:
            for u, v in EDGES: cv2.line(mk, P(c.xy_px[row][hb + u]), P(c.xy_px[row][hb + v]), (160, 160, 160), 1)
        if us[hb]:
            for u, v in EDGES: cv2.line(mk, P(xy[hb + u]), P(xy[hb + v]), col, 2)
            cv2.putText(mk, 'arm ' + name, P(xy[hb + 12]), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
    out = np.hstack([crop, np.full((crop.shape[0], 6, 3), 255, np.uint8), mk])
    cv2.putText(out, f'SEN0354 row {row}', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    cv2.imwrite(str(H / 'frames' / f'SEN0354_row{row:03d}.png'), out)
    print('row', row, 'arm L/R', bool(us[33]), bool(us[54]), 'recovery L/R', bool(c.usable[row][33]), bool(c.usable[row][54]))
