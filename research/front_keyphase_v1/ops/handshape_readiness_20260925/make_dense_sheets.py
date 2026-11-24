"""Dense hand crops across the active span, for items four frames could not settle.

Same held videos, same cached observation. Decode and crop only: no model, no training,
no download, no user footage.
"""
import hashlib, json, pathlib, sys
import cv2
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
OUT = H / 'dense_sheets'
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.core.observation import Observation, LEFT_HAND_SLICE, RIGHT_HAND_SLICE  # noqa: E402

read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

# word -> which hand row to draw, and why this clip needs a dense look
PLAN = {
    'JOTA': ('R', 'thumb-up claim vs code 1000: is the thumb extended throughout or only briefly'),
    'JOESONGHADA': ('B', 'flat-hand claim vs code 0005, and the frame the second hand joins'),
    'NEO': ('R', 'index points along the camera axis; check the shape before it turns toward the lens'),
    'JADA': ('R', 'the V closes at the end; find where the change starts and how far it goes'),
}
N = 10


def crop(frame, pts, ok, pad=1.9, size=260):
    if ok.sum() < 5:
        return np.full((size, size, 3), 38, np.uint8)
    p = pts[ok]
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    r = max(float(np.abs(p[:, 0] - cx).max()), float(np.abs(p[:, 1] - cy).max())) * pad
    r = max(r, 40.0)
    h, w = frame.shape[:2]
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return np.full((size, size, 3), 38, np.uint8)
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_CUBIC)


def main():
    clips = {c['word_id']: c for c in read(L / 'config' / 'source_manifest.json')['clips']}
    OUT.mkdir(exist_ok=True)
    record = []
    for wid, (which, why) in PLAN.items():
        c = clips[wid]
        assert sha(c['absolute_path']) == c['sha256'], wid
        obs = Observation.load(str(L / c['observation_npz']))
        lo, hi = c['quality']['active_span_first_frame'], c['quality']['active_span_last_frame']
        rows = [lo + round((hi - lo) * i / (N - 1)) for i in range(N)]
        cap = cv2.VideoCapture(c['absolute_path'])
        sides = ['R', 'L'] if which == 'B' else [which]
        bands, used = {s: [] for s in sides}, []
        for row in rows:
            fid = int(obs.frame_id[row])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            got, frame = cap.read()
            if not got:
                continue
            for s in sides:
                sl = RIGHT_HAND_SLICE if s == 'R' else LEFT_HAND_SLICE
                tile = crop(frame, obs.xy_px[row, sl], obs.usable[row, sl])
                tag = np.full((22, tile.shape[1], 3), 18, np.uint8)
                cv2.putText(tag, f'{s} r{row}', (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (230, 230, 230), 1, cv2.LINE_AA)
                bands[s].append(np.vstack([tag, tile]))
            used.append({'row': row, 'frame_id': fid})
        cap.release()
        stack = [np.hstack(bands[s]) for s in sides if bands[s]]
        if not stack:
            continue
        sheet = np.vstack(stack)
        head = np.full((30, sheet.shape[1], 3), 12, np.uint8)
        cv2.putText(head, f'{wid}  dense {N} rows over active span {lo}-{hi}', (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        sheet = np.vstack([head, sheet])
        path = OUT / f'{wid}_dense.png'
        cv2.imwrite(str(path), sheet)
        record.append({'word_id': wid, 'reason': why, 'sheet': str(path.relative_to(L)),
                       'sheet_sha256': sha(path), 'video_sha256': c['sha256'],
                       'active_span': [lo, hi], 'sampled': used, 'hands_shown': sides})
        print(f'{wid:<13} rows {[u["row"] for u in used]} -> {path.name}')
    with (H / 'DENSE_SHEETS.json').open('x', encoding='utf-8') as f:
        json.dump({'schema': 'wis_handshape_dense_sheets_v1', 'at_kst': '2026-09-25',
                   'purpose': 'continuous-interval visual check for items four frames could not settle',
                   'sheets': record, 'models_run': 0, 'training_runs': 0, 'downloads': 0,
                   'user_footage': 0}, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('dense sheets', len(record))


if __name__ == '__main__':
    main()
