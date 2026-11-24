"""Original public DEV13 visual diagnosis only. No learned model or relabeling."""
from pathlib import Path
import json
import hashlib
import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

U = Path(__file__).resolve().parent
OPS = U.parents[2]
R = U / 'dev13_rgb_diagnosis'


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    R.mkdir(exist_ok=True)
    assert not (R / 'MANIFEST.json').exists()
    cid = 'NIA_SL_WORD1133_REAL13_F'
    source_index = OPS / 'word100_rtmw_joint_learning_20261108/SOURCES.json'
    source = next(r for r in read(source_index) if r['clip_id'] == cid)
    assert source['role'] == 'development' and source['person'] == 13
    video = Path(source['feedback_source'])
    assert sha(video) == source['source_sha256']
    channel = OPS / f'word100_rtmw_joint_learning_20261108/channels/{cid}.npz'
    runtime = OPS / f'word100_candidate_role_learning_20261111/evaluation_inputs/{cid}_full_normal.npz'
    with np.load(channel, allow_pickle=False) as z:
        native = z['native67'].copy()
        available = z['native_available'].copy()
        full = z['full'].copy()
        frame_ids = z['frame_id'].copy()
    with np.load(runtime, allow_pickle=False) as z:
        assert np.array_equal(full, z['x'])
    assert np.array_equal(frame_ids, np.arange(len(frame_ids)))
    frames = [55, 60, 67, 68, 69, 70, 71, 72, 73]
    crop = (720, 250, 1320, 850)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    sheets = {name: Image.new('RGB', (1350, 1500), '#e5e9ef') for name in ('raw', 'points')}
    for name, sheet in sheets.items():
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), cid + ' | source RGB; frames55,60,67..73 | ' + name, font=font, fill='#15243a')
        draw.text((12, 36), 'Red=left elbow  Orange=body wrist  Blue=hand-root0  Green=right palm0+MCP mean', font=small, fill='#15243a')
        draw.text((12, 58), 'Exposed DEV diagnosis only. Coordinates are estimates, not independent truth.', font=small, fill='#15243a')
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 117
    records = []
    for index, frame in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        assert ok
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        assert rgb.size == (1920, 1080)
        points = {'left_elbow': native[frame, 6], 'body_left_wrist': native[frame, 7],
                  'left_hand_root': native[frame, 25], 'right_palm': native[frame, [46, 51, 55, 59, 63]].mean(0)}
        if frame in (55, 67, 70, 73):
            rgb.save(R / f'frame{frame}_full.jpg', quality=96)
        for name, sheet in sheets.items():
            im = rgb.crop(crop).resize((450, 450))
            if name == 'points':
                d = ImageDraw.Draw(im)
                positions = {key: ((float(p[0]) - crop[0]) * .75, (float(p[1]) - crop[1]) * .75) for key, p in points.items()}
                d.line([positions['left_elbow'], positions['body_left_wrist']], fill='#db5b20', width=3)
                d.line([positions['left_elbow'], positions['left_hand_root']], fill='#1770e8', width=2)
                for key, color in [('left_elbow', '#e52f3c'), ('body_left_wrist', '#ed951d'), ('left_hand_root', '#1770e8'), ('right_palm', '#07956c')]:
                    x, y = positions[key]
                    d.ellipse((x-5, y-5, x+5, y+5), fill=color, outline='white', width=1)
            tile = Image.new('RGB', (450, 470), 'white')
            tile.paste(im, (0, 20))
            ImageDraw.Draw(tile).text((10, 0), f'Original frame {frame}', font=font, fill='#15243a')
            sheet.paste(tile, ((index % 3) * 450, 90 + (index // 3) * 470))
        records.append(dict(frame=frame, points={key: p.tolist() for key, p in points.items()},
            point_available={key: bool(available[frame, i]) for key, i in [('left_elbow', 6), ('body_left_wrist', 7), ('left_hand_root', 25)]},
            body_wrist_to_hand_root_pixels=float(np.linalg.norm(points['body_left_wrist'] - points['left_hand_root']))))
    cap.release()
    for name, sheet in sheets.items():
        sheet.save(R / f'{name}.jpg', quality=95)
    # Same point roles; compare both anchor definitions without model scoring.
    native51 = native[:, [0, 5, 2, 6, 3, 7, 4, 12, 9] + list(range(25, 67))].astype(float)
    shoulder_scale = np.median(np.linalg.norm(native51[:, 1] - native51[:, 2], axis=1))
    palm = native[:, [46, 51, 55, 59, 63]].mean(1).astype(float)
    rel = (palm - native[:, 6]) / shoulder_scale
    anchor_results = []
    for start, end in [(54, 71), (55, 70)]:
        for anchor_name, wrist_index in [('body_wrist', 7), ('hand_root0', 25)]:
            arm = (native[:, wrist_index].astype(float) - native[:, 6]) / shoulder_scale
            length = np.linalg.norm(arm, axis=1)
            projected = (rel * arm).sum(1) / np.maximum(length, 1e-8)
            fraction = projected / np.maximum(length, 1e-8)
            anchor_results.append(dict(span=[start, end], anchor=anchor_name, projected_delta=float(projected[end]-projected[start]),
                fraction_delta=float(fraction[end]-fraction[start]), anchor_length_start=float(length[start]), anchor_length_end=float(length[end]),
                endpoint_length_rule_pass=bool(length[start] > .05 and length[end] > .05),
                note='Same left elbow retained. No contact label, detector substitution or evaluation verdict.'))
    pins = {str(p): sha(p) for p in [Path(__file__), source_index, video, channel, runtime, R / 'raw.jpg', R / 'points.jpg']}
    manifest = dict(at=datetime.datetime.now().astimezone().isoformat(), clip_id=cid, role='previously exposed development', person_record_id=13,
        source=str(video), source_sha256=source['source_sha256'], original_frame_ids=frames, crop=crop,
        runtime_full_channels_exact=True, records=records, anchor_formula_comparison=anchor_results, pins=pins,
        raw_sheet=str(R / 'raw.jpg'), point_sheet=str(R / 'points.jpg'), reviewed=False,
        private_access=False, final8=False, model_inference=False, no_relabeling=True, no_model_selection=True)
    with (R / 'MANIFEST.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('Public DEV13 diagnostic sheets prepared. No model inference.', flush=True)


if __name__ == '__main__':
    main()
