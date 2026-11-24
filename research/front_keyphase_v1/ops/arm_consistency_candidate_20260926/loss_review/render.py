"""Per target frame: unmarked | baseline skeletons (L orange, R cyan; the lacked slot drawn thick and
labelled) + shoulders/elbows/pose wrists where usable. Contact sheets per clip."""
import json, pathlib, sys
import cv2, numpy as np
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent.parent / 'temporal_recovery_candidate_20260926'))
import slot_review as sr
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
rows = [r for r in json.loads((H / 'LOSS_LIST.json').read_text(encoding='utf-8'))['rows'] if r['existing'] is None]
by = {}
for r in rows:
    by.setdefault(r['clip'], []).append(r)
for clip, items in by.items():
    z = np.load(sr.CACHE / f'{clip}.npz')
    src = json.loads((sr.CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
    fids = list(map(int, z['frame_id']))
    cap = cv2.VideoCapture(src)
    tiles = []
    for it in items:
        f = it['frame']; r = fids.index(f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        xy, us = z['xy_px'][r], z['usable'][r]
        h0 = int(img.shape[0] * 0.12)
        crop = img[h0:]; mk = crop.copy()
        P = lambda p: (int(p[0]), int(p[1] - h0))
        for a, b, c, col in ((11, 13, 15, (0, 220, 0)), (12, 14, 16, (220, 0, 220))):
            ch = [i for i in (a, b, c) if us[i]]
            for u, v in zip(ch, ch[1:]): cv2.line(mk, P(xy[u]), P(xy[v]), col, 2)
            for i in ch: cv2.circle(mk, P(xy[i]), 4, col, -1)
        for s, hb, col in (('L', 33, (0, 150, 255)), ('R', 54, (255, 200, 0))):
            if not us[hb]: continue
            th = 3 if s == it['side'] else 1
            for u, v in EDGES: cv2.line(mk, P(xy[hb + u]), P(xy[hb + v]), col, th)
            cv2.putText(mk, ('>>' if th == 3 else '') + 'base ' + s, (P(xy[hb])[0] - 30, P(xy[hb])[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
        pair = np.hstack([crop, np.full((crop.shape[0], 4, 3), 255, np.uint8), mk])
        pair = cv2.resize(pair, (960, int(pair.shape[0] * 960 / pair.shape[1])))
        cv2.rectangle(pair, (0, 0), (959, 24), (0, 0, 0), -1)
        cv2.putText(pair, f"{clip[5:]} f{f} lacked {it['side']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        tiles.append(pair)
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((hmax - t.shape[0], 960, 3), np.uint8)]) for t in tiles]
    for k in range(0, len(tiles), 4):
        chunk = tiles[k:k + 4]
        cv2.imwrite(str(H / 'frames' / f'{clip}_{k // 4}.png'), np.vstack(chunk))
    print(clip, [i['frame'] for i in items])
