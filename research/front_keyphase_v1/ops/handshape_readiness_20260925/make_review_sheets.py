"""Contact sheets from the held demonstration videos, for human/AI visual review.

Reads the videos already in the project and the cached observation to locate the hand.
No download, no training, no model inference, no user footage. Frames are decoded and
cropped only; nothing is generated or altered.
"""
import hashlib, json, pathlib, sys
import cv2
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
OUT = H / 'review_sheets'
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.core.observation import Observation, LEFT_HAND_SLICE, RIGHT_HAND_SLICE  # noqa: E402

read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
TARGETS = ['GAMSAHAMNIDA', 'NA', 'NEO', 'MOREUDA', 'SILTA', 'ALDA', 'JADA', 'JOTA', 'JOESONGHADA']


def crop(frame, pts, ok, pad=1.9, size=320):
    """Square crop centred on the observed hand landmarks."""
    if ok.sum() < 5:
        return None
    p = pts[ok]
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    r = max(float(np.abs(p[:, 0] - cx).max()), float(np.abs(p[:, 1] - cy).max())) * pad
    r = max(r, 40.0)
    h, w = frame.shape[:2]
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_CUBIC)


def main():
    manifest = read(L / 'config' / 'source_manifest.json')
    clips = {c['word_id']: c for c in manifest['clips']}
    OUT.mkdir(exist_ok=True)
    record = []
    for wid in TARGETS:
        c = clips[wid]
        assert sha(c['absolute_path']) == c['sha256'], wid
        obs = Observation.load(str(L / c['observation_npz']))
        lo, hi = c['quality']['active_span_first_frame'], c['quality']['active_span_last_frame']
        picks = [lo + int((hi - lo) * f) for f in (0.15, 0.40, 0.65, 0.90)]
        cap = cv2.VideoCapture(c['absolute_path'])
        panels, used = [], []
        for row in picks:
            fid = int(obs.frame_id[row])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            got, frame = cap.read()
            if not got:
                continue
            full = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
            col = [full]
            for side, sl in (('R', RIGHT_HAND_SLICE), ('L', LEFT_HAND_SLICE)):
                cut = crop(frame, obs.xy_px[row, sl], obs.usable[row, sl])
                col.append(cut if cut is not None else np.full((320, 320, 3), 40, np.uint8))
            pad = max(x.shape[0] for x in col)
            col = [np.vstack([x, np.full((pad - x.shape[0], 320, 3), 25, np.uint8)]) if x.shape[0] < pad else x
                   for x in col]
            panels.append(np.vstack(col))
            used.append({'row': row, 'frame_id': fid})
        cap.release()
        if not panels:
            continue
        sheet = np.hstack(panels)
        label = np.full((34, sheet.shape[1], 3), 15, np.uint8)
        cv2.putText(label, f'{wid}  rows {[u["row"] for u in used]}  (top=full, mid=RIGHT hand, bottom=LEFT hand)',
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        sheet = np.vstack([label, sheet])
        path = OUT / f'{wid}.png'
        cv2.imwrite(str(path), sheet)
        record.append({'word_id': wid, 'clip_id': c['clip_id'], 'sheet': str(path.relative_to(L)),
                       'sheet_sha256': sha(path), 'video_sha256': c['sha256'],
                       'active_span': [lo, hi], 'sampled': used})
        print(f'{wid:<13} rows {[u["row"] for u in used]} -> {path.name}')
    with (H / 'REVIEW_SHEETS.json').open('x', encoding='utf-8') as f:
        json.dump({'schema': 'wis_handshape_review_sheets_v1', 'at_kst': '2026-09-25',
                   'purpose': 'visual review material only; decoding and cropping, nothing generated',
                   'sheets': record, 'models_run': 0, 'training_runs': 0, 'downloads': 0,
                   'user_footage': 0}, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('sheets', len(record))


if __name__ == '__main__':
    main()
