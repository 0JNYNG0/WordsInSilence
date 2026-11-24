"""Visualize frozen comparison bounds on source frames; no detection or training."""
from pathlib import Path
import hashlib
import json
import av
from PIL import Image, ImageDraw, ImageFont

U = Path(__file__).resolve().parent
PAIR = U.parent / 'word100_fullframe_roi_pair_20261016'
plan = json.loads((PAIR / 'PLAN.json').read_text(encoding='utf-8-sig'))
items = {x['clip_id']: x for x in plan['items']}
FONT = 'C:/Windows/Fonts/malgun.ttf'
BOLD = 'C:/Windows/Fonts/malgunbd.ttf'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def make(cid, indices, filename):
    row = items[cid]
    assert sha(row['video_path']) == row['source_sha256']
    box = row['bottom_box']
    w, h = row['original_size']
    assert box[:3] == [0, 0, w]
    bottom = box[3]
    frames = {}
    times = {}
    with av.open(row['video_path']) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in indices:
                frames[i] = f.to_image().convert('RGBA')
                times[i] = float(f.pts * f.time_base)
            if i == max(indices):
                break
    assert set(frames) == set(indices)
    # Each panel preserves the complete 16:9 original, including excluded area.
    panel_w = 1200
    panel_h = round(panel_w * h / w)
    scale = panel_w / w
    cw = panel_w*len(indices) + 48*(len(indices)+1)
    canvas = Image.new('RGB', (cw, panel_h + 284), '#f3f6fa')
    d = ImageDraw.Draw(canvas)
    d.text((48, 20), f"{row['label']} | 이번 비교에서 사용한 정확한 입력 범위", font=ImageFont.truetype(BOLD, 38), fill='#152a43')
    d.text((48, 76), f'원본 전체 {w} × {h}px에서 검출 → 노란 경계 안의 관절만 입력에 유지', font=ImageFont.truetype(FONT, 28), fill='#304c69')
    for col, idx in enumerate(indices):
        im = frames[idx]
        assert im.size == (w, h)
        overlay = Image.new('RGBA', im.size)
        od = ImageDraw.Draw(overlay)
        od.rectangle((0, bottom, w-1, h-1), fill=(30, 40, 54, 180))
        im = Image.alpha_composite(im, overlay)
        idraw = ImageDraw.Draw(im)
        idraw.rectangle((4, 4, w-5, bottom-1), outline='#ffe14c', width=8)
        for x in range(0, w, 70):
            idraw.line((x, bottom, min(w-1, x+45), bottom), fill='#ffe14c', width=6)
        im = im.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        x = 48 + col*(panel_w+48)
        y = 166
        d.text((x, 129), f'{"시작 장면" if idx == 0 else "동작 중 장면"}  ·  {times[idx]:.2f}초', font=ImageFont.truetype(BOLD, 25), fill='#152a43')
        canvas.paste(im.convert('RGB'), (x, y))
        # Labels are placed in side background, never covering body/hands.
        d.rounded_rectangle((x+15, y+15, x+470, y+62), radius=8, fill='#152a43')
        d.text((x+27, y+22), f'입력 유지: {w} × {bottom}px', font=ImageFont.truetype(BOLD, 24), fill='#ffe14c')
        boundary_y = y + round(bottom*scale)
        d.text((x+20, boundary_y+16), f'하단 {h-bottom}px: 입력 제외', font=ImageFont.truetype(BOLD, 25), fill='white')
    yy = 166 + panel_h + 20
    d.text((48, yy), f'상단·좌우는 원본 그대로  |  아래 경계 y = {bottom}px  |  영상 도중 경계 이동 없음', font=ImageFont.truetype(BOLD, 27), fill='#152a43')
    d.text((48, yy+42), '입력 준비 후보의 범위 설명입니다. 이 조건으로 새 모델을 학습한 상태는 아닙니다.', font=ImageFont.truetype(FONT, 24), fill='#304c69')
    out = U / filename
    assert not out.exists()
    canvas.save(out)
    return dict(clip_id=cid, label=row['label'], source=row['video_path'], source_sha256=row['source_sha256'], source_size=[w,h], roi=box, frame_indices_zero_based=indices, pts_seconds=times, output=filename, output_sha256=sha(out), output_size=list(canvas.size))

if __name__ == '__main__':
    outputs = [
        make('NIA_SL_WORD0106_REAL01_F', [0, 52], 'coldcoffee_exact_extent.png'),
        make('NIA_SL_WORD1545_REAL01_F', [0, 36], 'swimming_exact_extent.png'),
    ]
    result = dict(status='SOURCE_FRAME_ROI_VISUALIZATION_ONLY', plan_sha256=sha(PAIR/'PLAN.json'), outputs=outputs,
                  bottom_range_20_clips=[min(x['bottom_box'][3] for x in plan['items']), max(x['bottom_box'][3] for x in plan['items'])],
                  boundary_rule='Frozen legacy per-clip bottom: round(median shoulder center y + .75 * (median hip center y - median shoulder center y)); full width/top retained. Not hand-following; not measured navel.',
                  new_detection=False, new_training=False, live_camera=False)
    (U/'PROVENANCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
