"""Render each reviewed frame: unmarked crop | both detections + both arm chains.

Detections come from the BASELINE cache (it holds every detection of these frames in full).
Orange = detection baseline put in L, cyan = detection baseline put in R. Letters after the
colour tag say where original (O) and recovery (C) put that same detection. Distances are not drawn.
"""
import json, pathlib, sys
import cv2, numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
C = L / 'cache' / '2683eadbc5f1c588'
OUT = H / 'slot_frames'; OUT.mkdir(exist_ok=True)
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
ARM = {'L': (11, 13, 15, (0, 220, 0)), 'R': (12, 14, 16, (220, 0, 220))}
COL = {'L': (0, 150, 255), 'R': (255, 200, 0)}
S = 2.0


def where(slots, det):
    return ''.join(s for s in 'LR' if slots.get(s) == det) or '-'


def frame_panel(img, xy, us, row_info, title):
    pts = [xy[i] for i in (11, 12, 13, 14, 15, 16) if us[i]]
    for hb in (33, 54):
        if us[hb]:
            pts += list(xy[hb:hb + 21])
    pts = np.array(pts)
    x0, y0 = np.maximum(pts.min(0) - 60, 0).astype(int)
    x1, y1 = np.minimum(pts.max(0) + 60, [img.shape[1], img.shape[0]]).astype(int)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    raw, mk = crop.copy(), crop.copy()
    P = lambda p: (int(round((p[0] - x0) * S)), int(round((p[1] - y0) * S)))
    for s, (a, b, c, col) in ARM.items():
        chain = [i for i in (a, b, c) if us[i]]
        for u, v in zip(chain, chain[1:]):
            cv2.line(mk, P(xy[u]), P(xy[v]), col, 2, cv2.LINE_AA)
        if us[c]:
            cv2.drawMarker(mk, P(xy[c]), col, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(mk, 'p' + s, (P(xy[c])[0] + 6, P(xy[c])[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, .55, col, 2)
    for s, hb in (('L', 33), ('R', 54)):
        if not us[hb]:
            continue
        det = row_info['slots']['baseline'][s]
        tag = f"B{s} O:{where(row_info['slots']['original'], det)} C:{where(row_info['slots']['recovery'], det)}"
        kept = where(row_info['slots']['recovery'], det) != '-'
        for u, v in EDGES:
            cv2.line(mk, P(xy[hb + u]), P(xy[hb + v]), COL[s], 3 if kept else 1, cv2.LINE_AA)
        tip = P(xy[hb + 12])
        cv2.putText(mk, tag, (tip[0] - 40, tip[1] - 10 - (18 if s == 'R' else 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, COL[s], 2, cv2.LINE_AA)
    pair = np.hstack([raw, np.full((raw.shape[0], 6, 3), 255, np.uint8), mk])
    bar = np.full((28, pair.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, pair])


def main():
    frames = json.loads((H / 'SLOT_REVIEW.json').read_text(encoding='utf-8'))['frames']
    joe = json.loads((H / 'JOE_WINDOW.json').read_text(encoding='utf-8'))
    todo = {(f['clip'], f['frame']): f for f in frames}
    for j in joe:
        if 63 <= j['frame'] <= 73 and j['frame'] != 66:
            todo.setdefault(('NIKL_JOESONGHADA_01', j['frame']), {'slots': {
                'baseline': j['slots']['B'], 'original': j['slots']['O'], 'recovery': j['slots']['C']}})
    by_clip = {}
    for (clip, f), info in sorted(todo.items()):
        by_clip.setdefault(clip, []).append((f, info))
    for clip, items in by_clip.items():
        z = np.load(C / f'{clip}.npz', allow_pickle=False)
        src = json.loads((C / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']
        fids = list(map(int, z['frame_id']))
        cap = cv2.VideoCapture(src)
        for f, info in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                print('read fail', clip, f); continue
            r = fids.index(f)
            p = frame_panel(img, z['xy_px'][r], z['usable'][r], info, f'{clip} f{f}')
            cv2.imwrite(str(OUT / f'{clip}_f{f:03d}.png'), p)
        cap.release()
        print(clip, [f for f, _ in items])


if __name__ == '__main__':
    main()
