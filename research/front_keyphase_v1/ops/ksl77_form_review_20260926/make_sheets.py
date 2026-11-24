"""Contact sheets for the form review: each row = one clip, 8 frames from 5% to 95% of its length.
The user-confirmed prefix-00 clip is the first row of every sheet (reference, not re-reviewed).
No identity judgement is made from these images; they are for the sign movement only."""
import hashlib, json, pathlib
import cv2, numpy as np
H = pathlib.Path(__file__).resolve().parent
RAW = H.parents[1] / 'runs/ksl77_source_review_v1/raw'
W, HH, N, PER = 240, 135, 8, 6
def row(path, tag):
    c = cv2.VideoCapture(str(path)); n = int(c.get(7)); fps = c.get(5)
    tiles = []
    for k in range(N):
        idx = int(round((0.05 + 0.90 * k / (N - 1)) * (n - 1)))
        c.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = c.read()
        f = cv2.resize(f, (W, HH)) if ok else np.zeros((HH, W, 3), np.uint8)
        cv2.putText(f, f'{idx}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
        tiles.append(f)
    bar = np.full((HH, 110, 3), 30, np.uint8)
    for i, t in enumerate(tag.split('|')):
        cv2.putText(bar, t, (4, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return np.hstack([bar] + tiles), {'frames': n, 'fps': round(fps, 2)}
meta = {}
for cls, word in (('16', 'meokda'), ('48', 'gongbu')):
    files = sorted((RAW / cls).glob('*.MP4'))
    ref = [f for f in files if f.name.startswith('00_')][0]
    todo = [f for f in files if f != ref]
    ref_row, m = row(ref, f'{ref.stem}|REFERENCE|user-confirmed')
    meta[ref.name] = dict(m, role='reference', sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
    for s in range(0, len(todo), PER):
        rows = [ref_row]
        for f in todo[s:s + PER]:
            r, m = row(f, f'{f.stem}|review')
            meta[f.name] = dict(m, role='review', sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            rows.append(r)
        cv2.imwrite(str(H / 'sheets' / f'{word}_{s // PER}.png'), np.vstack(rows))
(H / 'SHEET_INDEX.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
print(len(meta), 'clips;', sorted(p.name for p in (H / 'sheets').glob('*.png')))
