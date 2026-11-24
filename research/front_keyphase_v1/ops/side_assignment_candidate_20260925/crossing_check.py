"""Crossed-hands check on a SMALL set of held public clips not yet examined for crossing.

A crossing CANDIDATE frame is one where the subject-left hand sits left of the subject-right hand
in the image. That is a geometric shortlist only - screen position alone is never the verdict. Each
shortlisted episode is rendered before / during / after so the physical hands can be tracked by eye.

No NIKL rescan, no ZIP listing, no download. Baseline and candidate extraction of 4 clips only.
"""
import json, pathlib, sys, time
import cv2, numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
from wis_lab.capture.landmarker import Landmarker          # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402
sys.path.insert(0, str(H))
from run_candidate import MutualNearest                    # noqa: E402

CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
OUT = H / 'crossing'; OUT.mkdir(exist_ok=True)
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
SCALE, CROP = 1.6, (0.10, 0.95)

CLIPS = [
    ('runs/greeting_source_review_v1/raw/NIA_SL_SEN0254_REAL01_F.mp4', '반갑다 - hands may meet'),
    ('runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4', '안녕하세요 - hands may meet'),
    ('runs/apology_source_review_v1/raw/NIA_SL_SEN0356_REAL01_F.mp4', '미안합니다 - one hand over the other'),
    ('runs/apology_source_review_v1/raw/NIA_SL_SEN0356_REAL02_F.mp4', '미안합니다 - one hand over the other'),
]


def panel(img, obs, row, marked, dy):
    xy, usa = obs.xy_px[row], obs.usable[row]
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    P = lambda p: (int(round(p[0] * SCALE)), int(round((p[1] + dy) * SCALE)))
    for s, hb in (('L', 33), ('R', 54)):
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
    bar = np.full((26, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, f"row {row}" + ("  [CROSSED in image]" if marked else "  [not crossed]"),
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (80, 200, 255) if marked else (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    started, report = time.time(), []
    for rel, why in CLIPS:
        v = L / rel
        with Landmarker(CONTRACT) as lm:
            b = extract(str(v), lm, v.stem)
        with MutualNearest(CONTRACT) as lm:
            c = extract(str(v), lm, v.stem)
        both = b.usable[:, 33] & b.usable[:, 54]
        crossed = np.flatnonzero(both & (b.xy_px[:, 33, 0] < b.xy_px[:, 54, 0]))
        hb = np.stack([b.usable[:, 33:54].any(1), b.usable[:, 54:75].any(1)], 1)
        hc = np.stack([c.usable[:, 33:54].any(1), c.usable[:, 54:75].any(1)], 1)
        lost = np.argwhere(hb & ~hc)
        rec = {'clip': v.stem, 'why_shortlisted': why, 'frames': int(b.xy_px.shape[0]),
               'frames_with_both_hands': int(both.sum()),
               'crossing_candidate_frames': [int(i) for i in crossed],
               'candidate_unassignments': int(len(lost)),
               'unassigned_rows': [[int(x), int(y)] for x, y in lost]}
        # render episodes of contiguous crossing candidates
        eps = []
        if crossed.size:
            cur = [int(crossed[0])]
            for i in crossed[1:]:
                if int(i) - cur[-1] <= 2:
                    cur.append(int(i))
                else:
                    eps.append(cur); cur = [int(i)]
            eps.append(cur)
        rec['episodes'] = [[e[0], e[-1]] for e in eps]
        cap = cv2.VideoCapture(str(v))
        for e in eps[:3]:
            picks = [e[0] - 3, e[0], e[len(e) // 2], e[-1], e[-1] + 3]
            panels = []
            for r in picks:
                if not (0 <= r < b.xy_px.shape[0]):
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(b.frame_id[r]))
                ok, img = cap.read()
                if not ok:
                    continue
                h0, h1 = int(img.shape[0] * CROP[0]), int(img.shape[0] * CROP[1])
                panels.append(panel(img[h0:h1], b, r, r in e, -h0))
            if panels:
                w = min(p.shape[1] for p in panels)
                sheet = np.vstack([p[:, :w] for p in panels])
                head = np.full((30, sheet.shape[1], 3), 15, np.uint8)
                cv2.putText(head, f"{v.stem}  crossing episode rows {e[0]}-{e[-1]}", (8, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(OUT / f"{v.stem}_{e[0]}-{e[-1]}.png"), np.vstack([head, sheet]))
        cap.release()
        report.append(rec)
        print(f"{v.stem:<26} frames {rec['frames']:>4} bothHands {rec['frames_with_both_hands']:>4} "
              f"crossingCandidates {len(crossed):>3} episodes {len(eps):>2} "
              f"candidateUnassign {rec['candidate_unassignments']}", flush=True)

    out = {'schema': 'wis_crossing_check_v1', 'at_kst': '2026-09-25',
           'scope': 'a SMALL shortlist of held public clips not yet examined for crossing. No NIKL '
                    'rescan, no ZIP listing, no download, no new capture.',
           'shortlist_rule': 'a crossing CANDIDATE is a frame where the subject-left hand sits left '
                             'of the subject-right hand in the image. Screen position alone is '
                             'never the verdict - each episode is reviewed on video.',
           'clips': report,
           'total_crossing_candidate_frames': sum(len(r['crossing_candidate_frames']) for r in report),
           'seconds': round(time.time() - started, 1)}
    with (H / 'CROSSING_CHECK.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(f"\ntotal crossing candidate frames {out['total_crossing_candidate_frames']} | {out['seconds']} s")


if __name__ == '__main__':
    main()
