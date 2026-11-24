"""Render exact saved coordinate plots with Pillow, not generated photographs."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from prepare_fixtures import U, read, write_new, sha, stamp
from palm_control import CHAINS, HAND_START, PROTECTED

BG = '#f4f6f8'
INK = '#182b41'
ORIGINAL = '#1976c9'
SYNTHETIC = '#e97026'
ANCHOR = '#12834a'
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)] + [(0, c[0]) for c in CHAINS] + [(a, b) for c in CHAINS for a, b in zip(c, c[1:])]
FONT = 'C:/Windows/Fonts/arial.ttf'


def render_panel(canvas, box, row, saved, index):
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=10, fill='white', outline='#cad4df', width=1)
    small, tiny = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 14)
    start = HAND_START[row['side']]
    before = saved['original_xy_px'][index, start:start + 21].astype(float)
    after = saved['xy_px'][index, start:start + 21].astype(float)
    applied = next(r['applied'] for r in row['frame_report'] if r['index'] == index)
    draw.text((left + 12, top + 10), f"{row['word_id']}  P{row['person']:02d}  {row['side']}  frame {int(saved['frame_id'][index])}", font=small, fill=INK)
    draw.text((left + 12, top + 33), f"{saved['timestamp_us'][index]/1e6:.3f}s | " + ('APPLIED' if applied else 'REJECTED / unchanged'), font=tiny, fill=INK)
    joint = np.concatenate([before, after])
    good = np.isfinite(joint).all(1)
    points = joint[good]
    if not len(points):
        draw.text((left + 12, top + 75), 'No finite hand coordinates', font=small, fill=INK)
        return
    lo, hi = points.min(0), points.max(0)
    # Identical before/after viewport; minimum40px prevents unstable extreme zoom.
    span = np.maximum(hi - lo, 40.)
    scale = min((right - left - 65) / span[0], (bottom - top - 100) / span[1])
    centre = (lo + hi) / 2
    offset = np.array([(left + right) / 2, (top + bottom + 40) / 2])
    def transform(hand):
        return (hand - centre) * scale + offset
    b, a = transform(before), transform(after)
    for hand, color, width in ((b, ORIGINAL, 4), (a, SYNTHETIC, 2)):
        for p, q in EDGES:
            if np.isfinite(hand[[p, q]]).all():
                draw.line([tuple(hand[p]), tuple(hand[q])], fill=color, width=width)
        for point in hand:
            if np.isfinite(point).all():
                x, y = point; draw.ellipse((x-2,y-2,x+2,y+2), fill=color)
    for j in PROTECTED:
        if np.isfinite(a[j]).all():
            x, y = a[j]; draw.ellipse((x-4,y-4,x+4,y+4), fill=ANCHOR)
    draw.text((left + 12, bottom - 26), 'Blue source | Orange synthetic | Green fixed palm/thumb', font=tiny, fill=INK)


def make_sheet(rows, title, filename):
    panel_w, panel_h, gap, header = 510, 300, 12, 102
    image = Image.new('RGB', (3 * panel_w + 4 * gap, len(rows) * (panel_h + gap) + header + 12), BG)
    draw = ImageDraw.Draw(image)
    draw.text((gap, 12), title, fill=INK, font=ImageFont.truetype(FONT, 24))
    draw.text((gap, 45), 'SYNTHETIC COORDINATE ATTRIBUTE CONTROL - not RGB, not real incorrect KSL, not expert truth', fill=INK, font=ImageFont.truetype(FONT, 19))
    draw.text((gap, 73), 'Fixed first/middle/last requested times. Original palm, wrist, thumb and body coordinates stay unchanged.', fill=INK, font=ImageFont.truetype(FONT, 17))
    for y, row in enumerate(rows):
        assert sha(row['path']) == row['sha256']
        with np.load(row['path'], allow_pickle=False) as z:
            for x, index in enumerate(row['visual_indices']):
                left, top = gap + x * (panel_w + gap), header + y * (panel_h + gap)
                render_panel(image, (left, top, left + panel_w, top + panel_h), row, z, index)
    path = U / filename
    assert not path.exists()
    image.save(path)
    return dict(path=str(path), sha256=sha(path), panels=3 * len(rows), width=image.width, height=image.height)


def main():
    assert not (U / 'VISUALS.json').exists()
    result = read(U / 'FIXTURE_RESULTS.json')
    rows = result['rows']
    overview = [next(r for r in rows if r['person'] == 2 and r['word_id'] == word and r['side'] == side)
                for side in ('R', 'L') for word in ('WORD0739', 'WORD1133')]
    visuals = [make_sheet(overview, 'Palm-preserving controls: fixed P02 overview, both expressions and hands', 'CONTROL_OVERVIEW.png')]
    for word in ('WORD0739', 'WORD1133'):
        for side in ('L', 'R'):
            selected = sorted([r for r in rows if r['word_id'] == word and r['side'] == side], key=lambda r: r['person'])
            visuals.append(make_sheet(selected, f'{word} {side}: all six public FIT people, fixed three timepoints', f'{word}_{side}_CONTACT_SHEET.png'))
    write_new(U / 'VISUALS.json', dict(at=stamp(), visuals=visuals, plots_only=True, rgb_images_modified=False,
        source='Saved exact original/synthetic arrays; no image synthesis, no private input',
        root_visual_review='PENDING', no_model_inference=True))
    print('Rendered', len(visuals), 'sheets;', sum(v['panels'] for v in visuals), 'panels incl overview repeats', flush=True)


if __name__ == '__main__':
    main()
