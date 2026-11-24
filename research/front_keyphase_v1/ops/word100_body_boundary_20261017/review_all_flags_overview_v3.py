"""Immutable review aid: one row per window-condition, never semantic approval.

Unlike v1/v2, a window with multiple side/comparison flags receives multiple
rows. The detailed temporal sheets remain the source for ambiguous motion.
"""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_all_flags_build as b
from review_all_flags_overview_v2 import tile
from PIL import Image, ImageDraw


def condition_peak(material, condition):
    side = 'L' if condition['side'] == 'left' else 'R'
    old, new = ('native', 'masked') if condition['comparison'] == 'native_to_mask' else ('masked', 'sensor')
    focus = set(condition['focus_frame_ids'])
    candidates = [f for f in material['selected_frames'] if f['frame_id'] in focus]
    assert candidates, 'A frozen condition has no displayed focus frame'
    def loss(f):
        return max(f['valid_hand_counts'][old][side] - f['valid_hand_counts'][new][side], 0)
    peak = max(candidates, key=lambda f: (loss(f), -f['frame_id']))
    return peak['frame_id'], loss(peak)


def build():
    jobs = b.plan_windows(b.read(b.U / 'review_flag_coverage_RESULT.json'))
    outputs = []
    for batch in (1, 2, 3):
        entries = [(j, ci, c) for j in jobs if j['batch'] == batch
                   for ci, c in enumerate(j['conditions'], 1)]
        for at in range(0, len(entries), 5):
            group = entries[at:at + 5]
            number = at // 5 + 1
            paths = {j['window_id']: b.OUT / (j['window_id'] + '.json') for j, _, _ in group}
            if not all(p.exists() for p in paths.values()):
                continue
            try:
                materials = {wid: b.read(p) for wid, p in paths.items()}
            except json.JSONDecodeError:
                continue  # Main writer has not committed this window yet.
            assert all(materials[j['window_id']]['job'] == j for j, _, _ in group)
            metadata = b.OUT / f'batch_{batch}_overview_v3_{number:03d}.json'
            image = b.OUT / f'batch_{batch}_overview_v3_{number:03d}.jpg'
            if metadata.exists():
                r = b.read(metadata)
                assert r['generator_sha256'] == b.sha(__file__)
                assert b.sha(image) == r['image_sha256']
                assert all(b.sha(p) == h for p, h in r['material_pins'].items())
                outputs.append(str(metadata))
                continue
            assert not image.exists(), 'Uncommitted overview image preserved'
            canvas = Image.new('RGB', (1920, 70 + 280 * len(group)), (14, 21, 31))
            d = ImageDraw.Draw(canvas)
            d.text((10, 5), f'묶음 {batch} / 조건별 개요 {number:03d} / 원본 시작 · 조건 최대손실 · native · masked · sensor · 원본 끝', font=b.font(24), fill='white')
            d.text((10, 39), '같은 W번호도 조건별로 여러 행입니다. 모든 조건을 확인하세요. 작거나 동작이 애매하면 상세/추가 프레임 필수. 생성≠검토완료.', font=b.font(21), fill=(255, 219, 130))
            rows = []
            for row, (j, ci, condition) in enumerate(group):
                m = materials[j['window_id']]
                selected = m['selected_frames']
                start, end = selected[0]['frame_id'], selected[-1]['frame_id']
                peak, loss = condition_peak(m, condition)
                y = 70 + 280 * row
                cond = ('NM' if condition['comparison'] == 'native_to_mask' else 'MS') + ':' + condition['side']
                key = f"{j['window_id']}_C{ci:02d}"
                title = f"{key} {j['label']} {j['clip_id']} f{j['window']['start_frame_id']}–{j['window']['end_frame_id']} {cond} loss={loss}"
                d.text((8, y + 1), title, font=b.font(22), fill='white')
                specs = [(start, 0, 'full', 'RGB 시작'), (peak, 0, 'full', 'RGB 조건 peak'),
                         (peak, 1, 'zoom', 'native 확대'), (peak, 2, 'zoom', 'masked 확대'),
                         (peak, 3, 'zoom', 'sensor 확대'), (end, 0, 'full', 'RGB 끝')]
                for col, (frame, sourcecol, kind, label) in enumerate(specs):
                    stamp = next(f for f in selected if f['frame_id'] == frame)
                    canvas.paste(b.letterbox(tile(m, frame, sourcecol, kind), 320, 200), (col * 320, y + 33))
                    d.text((col * 320 + 5, y + 238), f'{label} f{frame} {stamp["source_relative_pts_us"] / 1e6:.3f}s', font=b.font(18), fill=(215, 225, 237))
                rows.append(dict(condition_review_id=key, window_id=j['window_id'],
                    condition_index=ci, condition=condition, clip_id=j['clip_id'],
                    record_sha256=m['record_sha256'], selected_peak_frame_id=peak,
                    peak_flagged_side_point_loss=loss,
                    overview_displayed_frame_ids=sorted(set([start, peak, end])),
                    actual_overlay_compared_frame_ids=[peak],
                    source_only_context_frames=sorted(set([start, end])),
                    detail_material=str(paths[j['window_id']]), detail_images=m['images'],
                    review_status='PENDING', semantic_pass=False))
            canvas.save(image, quality=92, subsampling=0)
            result = dict(status='CONDITION_COMPLETE_OVERVIEW_CREATED_PENDING_ACTUAL_VIEW',
                batch=batch, page=number, image=str(image), image_sha256=b.sha(image),
                material_pins={str(p): b.sha(p) for p in paths.values()}, rows=rows,
                training_allowed=False, semantic_pass=False, all_window_frames_reviewed=False,
                peak_selection='Each separate side/comparison condition has its own maximum positive count-loss focus frame. Detailed material contains the exact per-condition peak among all focus frames.',
                note='Only sparse overview samples displayed. Ambiguous motion/support-hand interpretation requires temporal detailed sheets or explicit extra frames. Complete every condition before closing its window.',
                generator_sha256=b.sha(__file__))
            b.write_new(metadata, result)
            outputs.append(str(metadata))
            print(json.dumps(dict(batch=batch, page=number, conditions=[r['condition_review_id'] for r in rows], image=str(image)), ensure_ascii=False), flush=True)
    print(json.dumps(dict(compact_metadata_files=len(outputs)), ensure_ascii=False))


def self_test():
    frames = []
    for fid, counts in [(43, (21, 21, 0)), (58, (21, 0, 0)), (73, (21, 21, 0))]:
        frames.append(dict(frame_id=fid, valid_hand_counts={k: {'L': v, 'R': 21} for k, v in zip(('native', 'masked', 'sensor'), counts)}))
    material = {'selected_frames': frames}
    assert condition_peak(material, dict(side='left', comparison='native_to_mask', focus_frame_ids=[58])) == (58, 21)
    assert condition_peak(material, dict(side='left', comparison='mask_to_sensor', focus_frame_ids=[73])) == (73, 21)
    assert condition_peak(material, dict(side='right', comparison='mask_to_sensor', focus_frame_ids=[58, 73])) == (58, 0)
    try:
        condition_peak(material, dict(side='left', comparison='native_to_mask', focus_frame_ids=[999]))
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing condition coverage accepted')
    print('PASS: 4 pure condition-selection checks; no RGB/detector/classifier execution')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    self_test() if '--self-test' in sys.argv else build()
