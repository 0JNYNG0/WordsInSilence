"""Select frames by the FROZEN rule (blind to recognition) and render both wrist markers."""
import json, pathlib, sys
import cv2, numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
C = L / 'cache' / '2683eadbc5f1c588'
OUT = H / 'frames'; OUT.mkdir(exist_ok=True)
SIDE = {'L': (33, 15), 'R': (54, 16)}
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
              (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]


def pool():
    out = []
    for p in sorted(C.glob('*.npz')):
        z = np.load(p, allow_pickle=False)
        xy, usa, conf = z['xy_px'], z['usable'], z['conf']
        sw = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=1)
        sh = usa[:, 11] & usa[:, 12]
        for s, (hb, pw) in SIDE.items():
            ok = np.flatnonzero(usa[:, hb] & usa[:, pw] & sh)
            for i in ok:
                out.append({'clip': p.stem, 'side': s, 'row': int(i),
                            'frame_id': int(z['frame_id'][i]),
                            'd': float(np.linalg.norm(xy[i, hb] - xy[i, pw]) / sw[i]),
                            'pose_wrist_conf': round(float(conf[i, pw]), 3),
                            'shoulder_conf': [round(float(conf[i, 11]), 3), round(float(conf[i, 12]), 3)],
                            'other_side_hand_usable': bool(usa[i, SIDE['R' if s == 'L' else 'L'][0]])})
    return out


def draw(rec):
    z = np.load(C / f"{rec['clip']}.npz", allow_pickle=False)
    side_cfg = json.loads((C / f"{rec['clip']}.sidecar.json").read_text(encoding='utf-8'))
    cap = cv2.VideoCapture(side_cfg['source_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec['frame_id'])
    ok, img = cap.read(); cap.release()
    if not ok:
        return None
    i = rec['row']; xy = z['xy_px'][i]; usa = z['usable'][i]
    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    P = lambda p: (int(round(p[0] * 1.6)), int(round(p[1] * 1.6)))
    # both hands, so a side-assignment problem is visible
    for s, (hb, _pw) in SIDE.items():
        col = (0, 200, 255) if s == 'L' else (255, 160, 0)
        if not usa[hb]:
            continue
        for a, b in HAND_EDGES:
            cv2.line(img, P(xy[hb + a]), P(xy[hb + b]), col, 1, cv2.LINE_AA)
        cv2.circle(img, P(xy[hb]), 7, col, 2)
        cv2.putText(img, f'hand {s}', (P(xy[hb])[0] + 9, P(xy[hb])[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    for lbl, idx, col in (('pose L', 15, (0, 255, 0)), ('pose R', 16, (0, 255, 0)),
                          ('sh L', 11, (200, 200, 200)), ('sh R', 12, (200, 200, 200))):
        if usa[idx]:
            cv2.drawMarker(img, P(xy[idx]), col, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, lbl, (P(xy[idx])[0] + 8, P(xy[idx])[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    hb, pw = SIDE[rec['side']]
    if usa[hb] and usa[pw]:
        cv2.line(img, P(xy[hb]), P(xy[pw]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(img, f"{rec['clip']}  side {rec['side']}  frame {rec['frame_id']}  d={rec['d']:.3f}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    name = f"{rec['group']}_{rec['d']:.3f}_{rec['clip']}_{rec['side']}_{rec['frame_id']}.png"
    cv2.imwrite(str(OUT / name), img)
    return name


def main():
    plan = json.loads((H / 'FROZEN_SELECTION.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FROZEN_BEFORE_LOOKING_AT_RECOGNITION'
    p = pool()
    med = float(np.median([r['d'] for r in p]))
    best = {}
    for r in p:
        k = (r['clip'], r['side'])
        if k not in best or r['d'] > best[k]['d']:
            best[k] = r
    large = sorted(best.values(), key=lambda r: -r['d'])[:8]
    used = {(r['clip'], r['side']) for r in large}
    near = {}
    for r in p:
        k = (r['clip'], r['side'])
        if k in used:
            continue
        if k not in near or abs(r['d'] - med) < abs(near[k]['d'] - med):
            near[k] = r
    small = sorted(near.values(), key=lambda r: abs(r['d'] - med))[:8]
    for g, rs in (('LARGE', large), ('SMALL', small)):
        for r in rs:
            r['group'] = g
    sel = large + small
    for r in sel:
        r['image'] = draw(r)
        print(f"{r['group']:<6}{r['clip']:<24}{r['side']}  frame {r['frame_id']:>3}  d={r['d']:.3f}  "
              f"poseWristConf={r['pose_wrist_conf']:.2f}  otherHand={r['other_side_hand_usable']}  -> {r['image']}")
    with (H / 'SELECTED.json').open('w', encoding='utf-8') as f:
        json.dump({'schema': 'wis_wrist_gap_selected_v1', 'at_kst': '2026-09-25',
                   'plan_sha256': __import__('hashlib').sha256((H / 'FROZEN_SELECTION.json').read_bytes()).hexdigest(),
                   'pool_frames': len(p), 'pool_median_d': round(med, 4),
                   'selection_blind_to_recognition': True, 'selected': sel},
                  f, ensure_ascii=False, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
