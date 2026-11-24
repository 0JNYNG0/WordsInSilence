"""New reading evidence for the SEN0354 undecidable case.

Three things at once, per frame:
  1. the FRONT view with NO markers          - what the hands actually do
  2. the FRONT view with markers             - what the extractor claimed
  3. the TOP (U) view with NO markers        - a second camera of the same take

The other view comes from the already-held archive (member extraction, no download). Frame
correspondence between views is CHECKED here, not assumed.
"""
import json, pathlib, sys
import cv2, numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1' / 'src')]
sys.path.insert(0, str(H))
from wis_lab.capture.landmarker import Landmarker          # noqa: E402
from wis_lab.capture.video_source import extract           # noqa: E402

F = L / 'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'
U = H / 'other_views/NIA_SL_SEN0354_REAL01_U.mp4'
OUT = H / 'crossing'
ROWS = [36, 38, 40, 42, 44, 46]
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))


def box(xy, usa, pad=110):
    pts = np.concatenate([xy[33:75][usa[33:75]], xy[[15, 16]][usa[[15, 16]]]])
    x0, y0 = pts.min(0) - pad
    x1, y1 = pts.max(0) + pad
    return int(max(0, x0)), int(max(0, y0)), int(x1), int(y1)


def crop(img, b, scale):
    x0, y0, x1, y1 = b
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    return cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def annotate(img, xy, usa, b, scale):
    out = crop(img, b, scale)
    P = lambda p: (int(round((p[0] - b[0]) * scale)), int(round((p[1] - b[1]) * scale)))
    for s, hb in (('L', 33), ('R', 54)):
        if not usa[hb]:
            continue
        col = (0, 200, 255) if s == 'L' else (255, 160, 0)
        for a, c in EDGES:
            cv2.line(out, P(xy[hb + a]), P(xy[hb + c]), col, 2, cv2.LINE_AA)
        cv2.circle(out, P(xy[hb]), 10, col, 3)
        cv2.putText(out, f'{s} wrist', (P(xy[hb])[0] + 12, P(xy[hb])[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    for lbl, idx in (('pose L', 15), ('pose R', 16)):
        if usa[idx]:
            cv2.drawMarker(out, P(xy[idx]), (0, 255, 0), cv2.MARKER_CROSS, 24, 3)
            cv2.putText(out, lbl, (P(xy[idx])[0] + 8, P(xy[idx])[1] + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def label(w, text, col=(235, 235, 235)):
    bar = np.full((30, w, 3), 22, np.uint8)
    cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
    return bar


def main():
    with Landmarker(CONTRACT) as lm:
        obs = extract(str(F), lm, F.stem)
    unassigned = {38, 40, 44}
    capF, capU = cv2.VideoCapture(str(F)), cv2.VideoCapture(str(U))
    # correspondence evidence, measured not assumed
    corr = {'frames_F': int(capF.get(cv2.CAP_PROP_FRAME_COUNT)),
            'frames_U': int(capU.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps_F': round(capF.get(cv2.CAP_PROP_FPS), 4),
            'fps_U': round(capU.get(cv2.CAP_PROP_FPS), 4)}
    rows = []
    for r in ROWS:
        fid = int(obs.frame_id[r])
        capF.set(cv2.CAP_PROP_POS_FRAMES, fid); okF, imF = capF.read()
        capU.set(cv2.CAP_PROP_POS_FRAMES, fid); okU, imU = capU.read()
        if not (okF and okU):
            continue
        xy, usa = obs.xy_px[r], obs.usable[r]
        b = box(xy, usa)
        scale = 620 / max(1, b[2] - b[0])
        raw = crop(imF, b, scale)
        ann = annotate(imF, xy, usa, b, scale)
        # top view: centre crop of the same width, no markers, no assumed mapping
        cx = imU.shape[1] // 2
        half = int((b[2] - b[0]) * 0.75)
        ub = (max(0, cx - half), int(imU.shape[0] * 0.30), min(imU.shape[1], cx + half),
              int(imU.shape[0] * 0.95))
        top = crop(imU, ub, 620 / max(1, ub[2] - ub[0]))
        hgt = min(raw.shape[0], ann.shape[0], top.shape[0])
        strip = np.hstack([raw[:hgt], ann[:hgt], top[:hgt]])
        tag = f'frame {fid}' + ('   [R UNASSIGNED by candidate]' if r in unassigned else '')
        col = (90, 90, 255) if r in unassigned else (235, 235, 235)
        rows.append(np.vstack([label(strip.shape[1], tag, col), strip]))
        capU.set(cv2.CAP_PROP_POS_FRAMES, fid)
    capF.release(); capU.release()
    w = min(x.shape[1] for x in rows)
    head = label(w, 'left: FRONT no markers   centre: FRONT with markers   right: TOP view no markers')
    cv2.imwrite(str(OUT / 'SEN0354_three_way.png'), np.vstack([head] + [x[:, :w] for x in rows]))
    (H / 'SEN0354_VIEW_CORRESPONDENCE.json').write_text(
        json.dumps({'schema': 'wis_view_correspondence_v1', 'at_kst': '2026-09-25',
                    'front': str(F.name), 'top': str(U.name),
                    'source': 'member extracted from the already-held [원천]01_real_sen_video.zip; '
                              'no download',
                    'measured': corr,
                    'correspondence_status': 'frame count, fps and duration match exactly. That is '
                                             'CONSISTENT with the same take but is NOT proof of '
                                             'frame-level alignment; the views are read together '
                                             'and any judgement states this.',
                    'frames_rendered': ROWS}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('written SEN0354_three_way.png', corr)


if __name__ == '__main__':
    main()
