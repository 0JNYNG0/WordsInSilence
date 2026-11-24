"""FIT anchor diagnostics and source overlays, no repair or learned inference."""
from pathlib import Path
import json
import hashlib
import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

U = Path(__file__).resolve().parent
H = U.parent
OPS = H.parent.parent
D = H / 'partial_direction_contract'


def rd(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert (U / 'PLAN_FROZEN.json').exists() and not (U / 'RESULTS.json').exists()
    reviews = rd(D / 'rgb_review/VISUAL_REVIEW.json')['rows']
    assert len(reviews) == 7
    source_index = OPS / 'word100_rtmw_joint_learning_20261108/SOURCES.json'
    sources = {r['clip_id']: r for r in rd(source_index) if r['role'] == 'fit' and r['person'] in range(1, 8)}
    ready_index = OPS / 'word100_shared_motion_learning_20261109/READY.json'
    ready = {r['clip_id']: r for r in rd(ready_index)['rows'] if r['role'] == 'fit' and r['person'] in range(1, 8)}
    pins = {str(p): sha(p) for p in [Path(__file__), U / 'PLAN_FROZEN.json', D / 'rgb_review/VISUAL_REVIEW.json', source_index, ready_index]}
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    cache = {}
    cases = []
    unique = {}
    for review in reviews:
        cid = review['clip_id']
        sr = sources[cid]
        rr = ready[cid]
        assert sr['source_sha256'] == rr['source_sha256'] == review['source_sha256']
        if cid not in cache:
            video = Path(sr['feedback_source'])
            assert sha(video) == sr['source_sha256']
            channel = Path(rr['channel'])
            assert sha(channel) == rr['channel_sha256']
            pins[str(video)] = sr['source_sha256']
            pins[str(channel)] = rr['channel_sha256']
            with np.load(channel, allow_pickle=False) as z:
                native = z['native67'].copy().astype(float)
                present = z['native_available'].copy().astype(bool)
                ts = z['timestamp_us'].copy().astype(float)
                assert np.array_equal(z['frame_id'], np.arange(len(native)))
            shoulder_ok = present[:, 5] & present[:, 2]
            scale = float(np.median(np.linalg.norm(native[:, 5]-native[:, 2], axis=1)[shoulder_ok]))
            assert scale > 0
            points = dict(shoulder=native[:, 5], elbow=native[:, 6], body_wrist=native[:, 7],
                hand_root=native[:, 25], mcp_centroid=native[:, [30, 34, 38, 42]].mean(1),
                moving_palm=native[:, [46, 51, 55, 59, 63]].mean(1))
            masks = dict(shoulder=present[:, 5], elbow=present[:, 6], body_wrist=present[:, 7],
                hand_root=present[:, 25], mcp_centroid=present[:, [30, 34, 38, 42]].all(1),
                moving_palm=present[:, [46, 51, 55, 59, 63]].all(1))
            residuals = {}
            residual_valid = {}
            for name, xy in points.items():
                v = np.full(len(native), np.nan)
                good = np.zeros(len(native), bool)
                for t in range(1, len(native)-1):
                    if masks[name][t-1:t+2].all():
                        alpha = (ts[t]-ts[t-1])/(ts[t+1]-ts[t-1])
                        prediction = xy[t-1]*(1-alpha)+xy[t+1]*alpha
                        v[t] = np.linalg.norm(xy[t]-prediction)/scale
                        good[t] = True
                residuals[name] = v
                residual_valid[name] = good
            cache[cid] = dict(video=video, native=native, present=present, ts=ts, scale=scale,
                points=points, masks=masks, residuals=residuals, residual_valid=residual_valid)
        c = cache[cid]
        if 'source_ranges' in review:
            frames = [f for a, b in review['source_ranges'] for f in range(a, b+1)]
        else:
            frames = review['source_frames']
        assert all(0 < f < len(c['native'])-1 for f in frames)
        # Fixed central half-width contains torso, both hands and both arms.
        crop = (480, 0, 1440, 1080)
        columns = 5
        tile_w, tile_h = 320, 390
        image = Image.new('RGB', (columns*tile_w, 70+((len(frames)+columns-1)//columns)*tile_h), '#e6eaf0')
        draw = ImageDraw.Draw(image)
        draw.text((10, 5), review['id'], font=font, fill='#14243a')
        draw.text((10, 28), 'RED elbow / ORANGE body wrist / BLUE hand root / CYAN MCP centre / GREEN moving palm', font=small, fill='#14243a')
        draw.text((10, 47), 'FIT original RGB; points are observations, not anatomical gold. No coordinate correction.', font=small, fill='#14243a')
        cap = cv2.VideoCapture(str(c['video']))
        assert cap.isOpened()
        frame_rows = []
        for index, f in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, bgr = cap.read()
            assert ok
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            assert rgb.size == (1920, 1080)
            rgb = rgb.crop(crop).resize((320, 360))
            dd = ImageDraw.Draw(rgb)
            locations = {name: ((p[f, 0]-crop[0])/3, p[f, 1]/3) for name, p in c['points'].items()}
            dd.line([locations['elbow'], locations['body_wrist']], fill='#d35a1e', width=2)
            dd.line([locations['elbow'], locations['hand_root']], fill='#226dde', width=1)
            for name, color in [('elbow', '#e63440'), ('body_wrist', '#e59c1c'), ('hand_root', '#206fe5'), ('mcp_centroid', '#08c3d7'), ('moving_palm', '#119b68')]:
                if not c['masks'][name][f]:
                    continue
                px, py = locations[name]
                dd.ellipse((px-3, py-3, px+3, py+3), fill=color, outline='white', width=1)
            tile = Image.new('RGB', (tile_w, tile_h), 'white')
            tile.paste(rgb, (0, 25))
            ImageDraw.Draw(tile).text((8, 3), 'Original frame '+str(f), font=font, fill='#14243a')
            image.paste(tile, ((index % columns)*tile_w, 70+(index//columns)*tile_h))
            p = c['points']
            body_root = float(np.linalg.norm(p['body_wrist'][f]-p['hand_root'][f]))
            palm_length = float(np.linalg.norm(p['mcp_centroid'][f]-p['hand_root'][f]))
            elbow_residual = c['residuals']['elbow'][f]
            peers = [c['residuals'][name][f] for name in ['shoulder', 'body_wrist', 'hand_root', 'mcp_centroid'] if c['residual_valid'][name][f]]
            row = dict(clip_id=cid, frame=f, person=sr['person'],
                points_px={name: p[f].tolist() for name, p in c['points'].items()},
                masks={name: bool(v[f]) for name, v in c['masks'].items()},
                body_wrist_root_discrepancy_shoulder=body_root/c['scale'],
                body_wrist_root_discrepancy_palm=body_root/palm_length if palm_length > 0 else None,
                forearm_body_length_shoulder=float(np.linalg.norm(p['body_wrist'][f]-p['elbow'][f])/c['scale']),
                forearm_handroot_length_shoulder=float(np.linalg.norm(p['hand_root'][f]-p['elbow'][f])/c['scale']),
                interpolation_residuals={name: float(v[f]) if c['residual_valid'][name][f] else None for name, v in c['residuals'].items()},
                elbow_excess_over_peer_median=float(elbow_residual-np.median(peers)) if np.isfinite(elbow_residual) and peers else None,
                anchor_valid_label=None, interpolation_used_as_replacement=False)
            frame_rows.append(row)
            unique[cid, f] = row
        cap.release()
        dest = U / (review['id']+'.jpg')
        image.save(dest, quality=95)
        pins[str(dest)] = sha(dest)
        cases.append(dict(id=review['id'], clip_id=cid, source_sha256=sr['source_sha256'],
            frames=frames, original_review=review['anchor'], overlay=str(dest), measurements=frame_rows,
            visual_reviewed=False))
    assert len(unique) == 58 and len(cache) == 5
    for path, digest in pins.items():
        assert sha(path) == digest
    result = dict(at=datetime.datetime.now().astimezone().isoformat(), cases=cases,
        unique_frames=len(unique), clips=len(cache), all_unique_frame_measurements=list(unique.values()), pins=pins,
        decision='No reliability threshold fitted and no anchor valid/invalid ground truth manufactured. Review overlays then specify repair contract.',
        no_coordinate_replacement=True, no_model_inference=True, no_DEV=True, no_training=True, no_private=True)
    write = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    with (U / 'RESULTS.json').open('x', encoding='utf-8') as f:
        f.write(write+'\n')
    print('FIT anchor evidence:', len(cache), 'clips,', len(unique), 'unique frames; no corrections or models.', flush=True)


if __name__ == '__main__':
    main()
