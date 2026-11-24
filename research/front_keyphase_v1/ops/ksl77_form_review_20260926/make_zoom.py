"""Enlarged 12-frame strips for clips that were unclear on the contact sheets (movement only)."""
import pathlib, cv2, numpy as np
H = pathlib.Path(__file__).resolve().parent
RAW = H.parents[1] / 'runs/ksl77_source_review_v1/raw'
N, W, HH = 12, 320, 240
def strip(p):
    c = cv2.VideoCapture(str(p)); n = int(c.get(7)); t = []
    for k in range(N):
        idx = int(round((0.03 + 0.94 * k / (N - 1)) * (n - 1)))
        c.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = c.read()
        f = f[40:680, 320:960] if ok else np.zeros((640, 640, 3), np.uint8)   # centre crop, upper body
        f = cv2.resize(f, (W, HH)); cv2.putText(f, f'{p.stem} f{idx}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        t.append(f)
    return np.vstack([np.hstack(t[:6]), np.hstack(t[6:])])
for cls, word, ids in (('16', 'meokda', ['00', '13', '14', '18']), ('48', 'gongbu', ['00', '13', '14', '18'])):
    rows = [strip(RAW / cls / f'{i}_{cls}.MP4') for i in ids]
    cv2.imwrite(str(H / 'sheets' / f'zoom_{word}.png'), np.vstack(rows))
print('ok')
