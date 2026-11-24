"""Draw the stored photo of each differing slot that HAS one: operating (primary) hands, candidate
(shadow) hands and the pose arm chains, plus the pose-wrist usability at that row. Frames without a
stored photo are not drawn (they stay UNDECIDABLE)."""
import json, pathlib, sys, zipfile
import cv2, numpy as np
H = pathlib.Path(__file__).resolve().parent
L = H.parents[2]
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.core.observation import Observation
SID = sys.argv[1]
RUN = L / 'runs' / SID
OUT = H / f'session_{SID}'; OUT.mkdir(exist_ok=True)
E = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
pair = json.loads((H / f'SESSION_{SID}_PAIRED.json').read_text(encoding='utf-8'))
info = []
for t in pair['trials']:
    for dif in t['differences']:
        tid, row = t['trial_id'], dif['row']
        prim = Observation.load(str(RUN / 'trials' / f'{tid}.npz'))
        shad = Observation.load(str(RUN / 'trials' / f'{tid}.shadow_start_provisional.npz'))
        u = prim.usable[row]
        rec = {'trial_id': tid, 'row': row, 'side': dif['side'], 'kind': dif['kind'],
               'pose_usable': {'L_shoulder': bool(u[11]), 'R_shoulder': bool(u[12]), 'L_wrist': bool(u[15]), 'R_wrist': bool(u[16])},
               'operating_hands': {'L': bool(u[33:54].any()), 'R': bool(u[54:75].any())},
               'candidate_hands': {'L': bool(shad.usable[row, 33:54].any()), 'R': bool(shad.usable[row, 54:75].any())},
               'photo': None}
        if dif['rgb_sample']:
            with zipfile.ZipFile(RUN / 'trials' / f'{tid}.capture.zip') as z:
                man = json.loads(z.read('manifest.json'))
                e = next(f for f in man['frames'] if f['observation_row'] == row)
                img = cv2.imdecode(np.frombuffer(z.read(e['image']), np.uint8), cv2.IMREAD_COLOR)
            sx, sy = e['width'] / e['source_width'], e['height'] / e['source_height']
            raw, mk = img.copy(), img.copy()
            P = lambda p: (int(p[0] * sx), int(p[1] * sy))
            for (a, b, c), col in (((11, 13, 15), (0, 220, 0)), ((12, 14, 16), (220, 0, 220))):
                ch = [i for i in (a, b, c) if u[i]]
                for i, j in zip(ch, ch[1:]): cv2.line(mk, P(prim.xy_px[row][i]), P(prim.xy_px[row][j]), col, 2)
                for i in ch: cv2.circle(mk, P(prim.xy_px[row][i]), 5, col, -1)
            for o, tag, th in ((prim, 'op', 3), (shad, 'cand', 1)):
                for hb, col, s in ((33, (0, 150, 255), 'L'), (54, (255, 200, 0), 'R')):
                    if o.usable[row, hb]:
                        for i, j in E: cv2.line(mk, P(o.xy_px[row][hb + i]), P(o.xy_px[row][hb + j]), col, th)
                        cv2.putText(mk, f'{tag} {s}', (P(o.xy_px[row][hb])[0] + 5, P(o.xy_px[row][hb])[1] + (20 if tag == 'op' else 40)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
            img2 = np.hstack([raw, np.full((raw.shape[0], 4, 3), 255, np.uint8), mk])
            name = f'{tid}_row{row:03d}.png'
            cv2.imwrite(str(OUT / name), img2)
            rec['photo'] = name
        info.append(rec); print(rec)
(OUT / 'DIFF_FRAMES.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
