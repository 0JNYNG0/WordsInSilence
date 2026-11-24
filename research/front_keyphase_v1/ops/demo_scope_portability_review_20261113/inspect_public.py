"""Read-only source inventory and small FIT RGB preview; no model or training."""
import collections
import hashlib
import json
from pathlib import Path
import re
import zipfile

import cv2
from PIL import Image, ImageDraw, ImageFont

U = Path(__file__).resolve().parent
OPS = U.parent
SOURCE = Path('W:/WordsInSilence/data/AIHub_103')
TARGETS = {'WORD0088': '딸기', 'WORD0099': '사과', 'WORD0775': '인사법',
           'WORD0815': '우유1', 'WORD1173': '평안', 'WORD1243': '우유2',
           'WORD1290': '감사', 'WORD2633': '숫자 2', 'WORD2635': '숫자 4', 'WORD2637': '숫자 6'}


def main():
    rows = [json.loads(s) for s in (OPS/'word_acquisition_inventory_20261011/FRONT_MANIFEST.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    manifest = {x['clip_id']: x for x in rows}
    current = json.loads((OPS/'word100_rtmw_joint_learning_20261108/FROZEN.json').read_text(encoding='utf-8-sig'))['labels']
    found = collections.defaultdict(list)
    archives = sorted((SOURCE/'videos').rglob('*.zip'))
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                stem = Path(info.filename).stem
                m = re.fullmatch(r'NIA_SL_(WORD\d{4})_REAL(\d{2})_F', stem)
                if m and m[1] in TARGETS:
                    row = manifest.get(stem)
                    found[m[1]].append({'clip_id': stem, 'record_id': int(m[2]),
                        'archive': str(archive), 'member': info.filename, 'bytes': info.file_size,
                        'crc32': f'{info.CRC:08x}', 'metadata': row,
                        'current100_label': bool(row and any(x in current for x in row['labels']))})
        print('INDEX', archive.name, flush=True)
    report = {'purpose': 'inventory/visual FIT inspection only; no performance evaluation',
              'archives': [{'path': str(p), 'bytes': p.stat().st_size} for p in archives],
              'exact_greeting_metadata': [x for x in rows if any('안녕' in l for l in x['labels'])],
              'targets': dict(found), 'previews': []}
    (U/'INVENTORY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (U/'media').mkdir(exist_ok=True)
    (U/'panels').mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 17)
    for code, candidates in found.items():
        # Inspect only public FIT record01. Validation/held-out media remain unopened.
        selected = [x for x in candidates if x['record_id']==1]
        if len(selected) != 1:
            raise RuntimeError((code, 'Expected exactly one record01', len(selected)))
        row = selected[0]
        with zipfile.ZipFile(row['archive']) as z:
            raw = z.read(row['member'])  # CRC is checked for this selected member.
        path = U/'media'/f'{row["clip_id"]}.mp4'
        path.write_bytes(raw)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start, end = row['metadata']['intervals'][0]
        times = [start+(end-start)*i/7 for i in range(8)]
        board = Image.new('RGB', (1360, 904), 'white')
        d = ImageDraw.Draw(board)
        d.text((16, 10), f'{TARGETS[code]} · {code} · 공개 FIT 기록01 · 원본 동작 구간', fill='#172b49', font=font)
        frames = []
        for k, t in enumerate(times):
            f = min(n-1, round(t*fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError((code, f, 'decode failed'))
            h, w = frame.shape[:2]
            # Fixed upper-body crop for readability; this changes no measurement or label.
            crop = frame[int(h*.06):int(h*.91), int(w*.24):int(w*.76)]
            im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            im.thumbnail((330, 374))
            x, y = 10+(k%4)*340, 52+(k//4)*420
            board.paste(im, (x+(330-im.width)//2, y))
            d.text((x+12, y+380), f'{t:.2f}초 / 원본 {f}프레임', fill='#172b49', font=small)
            frames.append(f)
        cap.release()
        panel = U/'panels'/f'{code}.jpg'
        board.save(panel, quality=92)
        report['previews'].append({'code': code, 'clip_id': row['clip_id'], 'video': str(path),
            'sha256': hashlib.sha256(raw).hexdigest(), 'panel': str(panel), 'fps': fps,
            'total_frames': n, 'sampled_frames': frames, 'source_interval': [start,end],
            'member_crc_verified': True, 'no_inference': True})
        print('PREVIEW', code, n, fps, flush=True)
    (U/'INVENTORY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for code, items in found.items():
        print(code, TARGETS[code], 'records', sorted(x['record_id'] for x in items), 'current100', items[0]['current100_label'])


if __name__ == '__main__':
    main()
