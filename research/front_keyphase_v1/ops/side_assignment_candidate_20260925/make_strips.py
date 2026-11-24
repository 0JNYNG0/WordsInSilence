"""One contact sheet per unassignment run: the frames just before, across, and just after.

Both hand landmark sets, both pose wrists and both shoulders are drawn, so the physical hands can
be tracked by eye across the run. Distance is printed for reference only - it is never the basis
for the judgement.
"""
import json, pathlib
import cv2, numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
C = L / 'cache' / '2683eadbc5f1c588'
OUT = H / 'strips'; OUT.mkdir(exist_ok=True)
SIDE = {'L': (33, 15), 'R': (54, 16)}
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
SCALE = 1.6
CROP = (0.12, 0.92)      # drop the letterbox, keep the torso


def panel(img, z, row, fid, side, marked, dy=0):
    xy, usa = z['xy_px'][row], z['usable'][row]
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    P = lambda p: (int(round(p[0] * SCALE)), int(round((p[1] + dy) * SCALE)))
    for s, (hb, _) in SIDE.items():
        if not usa[hb]:
            continue
        col = (0, 200, 255) if s == 'L' else (255, 160, 0)
        for a, b in EDGES:
            cv2.line(img, P(xy[hb + a]), P(xy[hb + b]), col, 1, cv2.LINE_AA)
        cv2.circle(img, P(xy[hb]), 6, col, 2)
        cv2.putText(img, s, (P(xy[hb])[0] + 8, P(xy[hb])[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    for lbl, idx in (('pL', 15), ('pR', 16), ('sL', 11), ('sR', 12)):
        if usa[idx]:
            c = (0, 255, 0) if lbl.startswith('p') else (190, 190, 190)
            cv2.drawMarker(img, P(xy[idx]), c, cv2.MARKER_CROSS, 14, 2)
            cv2.putText(img, lbl, (P(xy[idx])[0] + 6, P(xy[idx])[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    hb, pw = SIDE[side]
    d = None
    if usa[hb] and usa[pw] and usa[11] and usa[12]:
        sw = float(np.linalg.norm(xy[11] - xy[12]))
        d = float(np.linalg.norm(xy[hb] - xy[pw]) / sw)
    bar = np.full((26, img.shape[1], 3), 25, np.uint8)
    txt = f"f{fid}" + ("  [UNASSIGNED]" if marked else "") + (f"  d={d:.2f}" if d is not None else "")
    cv2.putText(bar, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (80, 80, 255) if marked else (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    runs = json.loads((H / 'RUNS.json').read_text(encoding='utf-8'))
    todo = json.loads((H / 'FROZEN_REVIEW.json').read_text(encoding='utf-8'))['runs_to_review']
    want = {(r['clip'], r['side'], r['frames'][0]) for r in todo}
    made = []
    for r in runs:
        key = (r['clip'], r['side'], r['frames'][0])
        if key not in want:
            continue
        z = np.load(C / f"{r['clip']}.npz", allow_pickle=False)
        src = json.loads((C / f"{r['clip']}.sidecar.json").read_text(encoding='utf-8'))['source_path']
        fids = list(map(int, z['frame_id']))
        run = r['frames']
        inner = [run[0]] if len(run) <= 2 else [run[0], run[len(run) // 2], run[-1]]
        picks = [f for f in [run[0] - 2] if f in fids] + inner + [f for f in [run[-1] + 2] if f in fids]
        cap = cv2.VideoCapture(src)
        panels = []
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            h0, h1 = int(img.shape[0] * CROP[0]), int(img.shape[0] * CROP[1])
            panels.append(panel(img[h0:h1], z, fids.index(f), f, r['side'], f in run, -h0))
        cap.release()
        if not panels:
            continue
        wid = min(p.shape[1] for p in panels)
        sheet = np.vstack([p[:, :wid] for p in panels])
        head = np.full((30, sheet.shape[1], 3), 15, np.uint8)
        cv2.putText(head, f"{r['clip']}  side {r['side']}  run {run[0]}-{run[-1]}  n={len(run)}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        name = f"{r['clip']}_{r['side']}_{run[0]}-{run[-1]}.png"
        cv2.imwrite(str(OUT / name), np.vstack([head, sheet]))
        made.append(name)
        print(name, '->', len(panels), 'panels')
    print(f'\n{len(made)} strips written to {OUT}')


if __name__ == '__main__':
    main()
