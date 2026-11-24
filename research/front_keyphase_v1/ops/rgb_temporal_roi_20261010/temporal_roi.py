"""Isolated, offline RGB region preparation; never reconstruct hand landmarks."""
from __future__ import annotations
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

UNIT = Path(__file__).resolve().parent
BASE = UNIT.parents[1]
OUT = BASE / 'runs/rgb_temporal_roi_v1'
HANDS = (slice(33, 54), slice(54, 75))
SIDES = ('left', 'right')
sys.path[:0] = [str(BASE / 'ops/handgap_identity_gate_20261004'),
               str(BASE / 'ops/partial_hand_reconstruction_20261003/implementation')]
import gate_candidate as gate


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def identity(xy3, usable3, side):
    # Minimal adapter for the frozen gate. No unused native node is available.
    xy = np.full((3, 67, 2), np.nan, dtype=np.float64)
    present = np.zeros((3, 67), bool)
    for dest, source in [(7, 15), (4, 16), (slice(25, 46), HANDS[0]),
                         (slice(46, 67), HANDS[1])]:
        xy[:, dest] = xy3[:, source]
        present[:, dest] = usable3[:, source]
    target = slice(25, 46) if side == 0 else slice(46, 67)
    xy[1, target] = np.nan
    present[1, target] = False
    return gate._decision(xy, present, 1, SIDES[side])


def propose(before, after, timestamps):
    b, a = np.asarray(before, np.float64), np.asarray(after, np.float64)
    ts = np.asarray(timestamps, np.int64)
    if not (ts[0] < ts[1] < ts[2]) or ts[2] - ts[0] > 66668:
        raise ValueError('Outside frozen one-frame time bound')
    alpha = float(ts[1] - ts[0]) / float(ts[2] - ts[0])
    center = (1-alpha)*(b[:2]+b[2:])/2 + alpha*(a[:2]+a[2:])/2
    side = math.ceil((1-alpha)*(b[2]-b[0]) + alpha*(a[2]-a[0]))
    if side < 2:
        raise ValueError('Invalid endpoint square')
    origin = np.floor(center-side/2).astype(np.int64)
    return np.concatenate((origin, origin+side)).tolist()


def measures(box, truth_box, points):
    box, truth = np.asarray(box, float), np.asarray(truth_box, float)
    inside = ((points >= box[:2]) & (points < box[2:])).all(1)
    area = float(np.prod(box[2:]-box[:2]))
    truth_area = float(np.prod(truth[2:]-truth[:2]))
    inter = float(np.prod(np.maximum(0, np.minimum(box[2:], truth[2:])-np.maximum(box[:2], truth[:2]))))
    return {'all21': bool(inside.all()), 'points_inside': int(inside.sum()),
            'iou': inter/(area+truth_area-inter),
            'normalized_center_error': float(np.linalg.norm((box[:2]+box[2:]-truth[:2]-truth[2:])/2)/(truth[2]-truth[0])),
            'area_ratio': area/truth_area}


def crop(rgb, box):
    x0, y0, x1, y1 = map(int, box)
    h, w = rgb.shape[:2]
    size = x1-x0
    if size != y1-y0 or not 2 <= size <= 3*max(h, w):
        raise ValueError('Invalid proposed square')
    canvas = np.zeros((size, size, 3), np.uint8)
    valid = np.zeros((size, size), np.uint8)
    lx, ty, rx, by = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if rx <= lx or by <= ty:
        raise ValueError('Proposed region wholly outside image')
    canvas[ty-y0:by-y0, lx-x0:rx-x0] = rgb[ty:by, lx:rx]
    valid[ty-y0:by-y0, lx-x0:rx-x0] = 255
    return (np.asarray(Image.fromarray(canvas).resize((128, 128), Image.Resampling.BILINEAR)),
            np.asarray(Image.fromarray(valid).resize((128, 128), Image.Resampling.NEAREST)) > 0)


def aggregate(events):
    n = len(events)
    result = {'n': n}
    for method in ('proposal', 'previous'):
        m = [e[method+'_metrics'] for e in events]
        result[method] = {'all21': sum(x['all21'] for x in m),
                          'all21_percent': 100*sum(x['all21'] for x in m)/n if n else None,
                          'point_containment_percent': 100*sum(x['points_inside'] for x in m)/(21*n) if n else None}
        for k in ('iou', 'normalized_center_error', 'area_ratio'):
            result[method]['mean_'+k] = float(np.mean([x[k] for x in m])) if n else None
    result['paired_coverage_improved'] = sum(e['proposal_metrics']['all21'] and not e['previous_metrics']['all21'] for e in events)
    result['paired_coverage_regressed'] = sum(e['previous_metrics']['all21'] and not e['proposal_metrics']['all21'] for e in events)
    return result


def run():
    OUT.mkdir(exist_ok=False)
    plan = read(UNIT/'FROZEN_PLAN.json')
    paths = {
        str(BASE/plan['inputs']['prepared_manifest']): plan['inputs']['manifest_sha256'],
        str(BASE/plan['inputs']['identity_gate']): plan['inputs']['identity_gate_sha256'],
        str(BASE/'ops/partial_hand_reconstruction_20261003/implementation/handgap.py'): plan['inputs']['handgap_dependency_sha256'],
        str(BASE/'ops/natural_handgap_audit_20261004/REVIEW.json'): plan['inputs']['prior_visual_review_sha256'],
        str(UNIT/'FROZEN_PLAN.json'): sha(UNIT/'FROZEN_PLAN.json'),
        str(Path(__file__).resolve()): sha(__file__)}
    manifest = read(BASE/plan['inputs']['prepared_manifest'])
    for r in manifest:
        paths[r['output_path']] = r['output_sha256']
        for k in ('raw', 'observation', 'input'):
            paths[r['source'][k+'_path']] = r['source'][k+'_sha256']
    for p, expected in paths.items():
        if sha(p) != expected:
            raise ValueError('Source pin mismatch: '+p)
    write(OUT/'SOURCE_PINS.json', paths)
    prior = read(BASE/'ops/natural_handgap_audit_20261004/REVIEW.json')['events']
    known = {(e['sample_id'], e['side'], e['frame_ids'][1]): e for e in prior}
    hidden, natural, source_map, observed_cache = [], [], {}, {}
    exclusions = Counter()
    invalid_slots = 0
    for r in sorted(manifest, key=lambda r:r['sample_id']):
        sid = r['sample_id']
        source_map[sid] = r
        z = load(r['output_path'])
        op = r['source']['observation_path']
        if op not in observed_cache:
            observed_cache[op] = load(op)
        obs = observed_cache[op]
        ids, ts = z['frame_id'], z['timestamp_us']
        if not np.array_equal(ts, obs['timestamp_us'][ids]):
            raise ValueError('Frame/time mismatch')
        invalid_slots += int((~z['crop_valid']).sum())
        for j in range(len(ids)):
            for side, sl in enumerate(HANDS):
                missing = not bool(z['crop_valid'][j,side])
                reason = None
                if j == 0 or j+1 == len(ids):
                    reason = 'edge'
                elif not np.array_equal(np.diff(ids[j-1:j+2]), [1,1]):
                    reason = 'nonadjacent'
                elif not (0 < ts[j]-ts[j-1] and 0 < ts[j+1]-ts[j] and ts[j+1]-ts[j-1] <= 66668):
                    reason = 'time'
                elif not bool(z['crop_valid'][[j-1,j+1],side].all()):
                    reason = 'invalid_endpoint'
                elif missing and (obs['usable'][ids[j],sl].any() or obs['observed'][ids[j],sl].any()):
                    reason = 'not_wholly_missing'
                if reason:
                    if missing:
                        exclusions[reason] += 1
                    continue
                rows = ids[j-1:j+2]
                decision = identity(obs['xy_px'][rows], obs['usable'][rows], side)
                b = z['bbox_xyxy'][j-1,side].tolist()
                a = z['bbox_xyxy'][j+1,side].tolist()
                box = propose(b, a, ts[j-1:j+2])
                event = {'sample_id':sid, 'clip_id':r['clip_id'], 'side':SIDES[side],
                         'row':j, 'frame_id':int(ids[j]), 'timestamp_us':int(ts[j]),
                         'endpoint_frames':[int(rows[0]),int(rows[2])],
                         'before_box':b, 'after_box':a, 'proposed_box':box,
                         'gate':decision}
                if missing:
                    old = known.get((sid, SIDES[side], int(ids[j])))
                    event.update(prior_event=old['event'] if old else None,
                                 prior_classification=old['classification'] if old else 'not_previously_reviewed')
                    natural.append(event)
                else:
                    truth = z['bbox_xyxy'][j,side]
                    points = z['hand_xy_px'][j,side]
                    event.update(truth_box=truth.tolist(),
                                 proposal_metrics=measures(box, truth, points),
                                 previous_metrics=measures(b, truth, points))
                    hidden.append(event)
    write(OUT/'HIDDEN_EVENTS.json', hidden)
    write(OUT/'NATURAL_EVENTS.json', natural)
    accepted = [e for e in hidden if e['gate']['accepted']]
    selected_natural = [e for e in natural if e['gate']['accepted']]
    by_key = {(e['sample_id'],e['side'],e['frame_id']):e for e in natural}
    known_check = []
    for old in prior:
        e = by_key.get((old['sample_id'],old['side'],old['frame_ids'][1]))
        known_check.append({'event':old['event'], 'classification':old['classification'],
                            'eligible': e is not None, 'accepted': bool(e and e['gate']['accepted']),
                            'reasons':e['gate']['reasons'] if e else ['outside_new_roi_eligibility']})
    a = aggregate(accepted)
    criteria = {
        'known7_wrong_none_accepted': sum(x['accepted'] for x in known_check if x['classification']=='wrong_endpoint') == 0,
        'coverage_not_worse_than_previous': bool(accepted) and a['proposal']['all21'] >= a['previous']['all21'],
        'mean_center_error_lower': bool(accepted) and a['proposal']['mean_normalized_center_error'] < a['previous']['mean_normalized_center_error'],
        'retains_at_least_half': bool(hidden) and len(accepted)*2 >= len(hidden)}
    write(OUT/'GEOMETRY_RESULT.json', {
        'hidden_ungated':aggregate(hidden), 'hidden_accepted':a,
        'hidden_rejected':len(hidden)-len(accepted),
        'hidden_rejection_reasons':dict(Counter(reason for e in hidden for reason in e['gate']['reasons'])),
        'natural_eligible':len(natural), 'natural_accepted':len(selected_natural),
        'natural_rejected':len(natural)-len(selected_natural), 'natural_exclusions':dict(exclusions),
        'original_invalid_slots':invalid_slots, 'remaining_without_proposed_roi':invalid_slots-len(selected_natural),
        'prior_identity_cases':known_check, 'geometry_criteria':criteria,
        'geometry_pass':all(criteria.values()), 'independent_verification_pending':True})
    print(json.dumps({'geometry':criteria, 'hidden':len(hidden),'accepted':len(accepted),
                      'natural':len(selected_natural)}), flush=True)
    # No repeated data extraction or model: original pixels only at selected times.
    (OUT/'crops').mkdir()
    (OUT/'visual').mkdir()
    grouped = defaultdict(list)
    for idx, e in enumerate(selected_natural):
        e = dict(e, crop_index=idx)
        grouped[source_map[e['sample_id']]['source']['raw_path']].append(e)
    rendered, pixel_records = [], []
    for raw, events in sorted(grouped.items()):
        wanted = defaultdict(list)
        for e in events:
            wanted[e['frame_id']].append(e)
        with av.open(raw, mode='r') as con:
            stream = con.streams.video[0]
            if float(stream.metadata.get('rotate','0')) % 360 or (stream.sample_aspect_ratio is not None and stream.sample_aspect_ratio != 1):
                raise ValueError('Unreviewed source orientation')
            previous_us = -1
            found = set()
            for fid, frame in enumerate(con.decode(stream)):
                if frame.pts is None:
                    raise ValueError('Missing video PTS')
                us = int(round(frame.pts*stream.time_base*1_000_000))
                if (fid == 0 and us != 0) or us <= previous_us or getattr(frame,'rotation',0)%360:
                    raise ValueError('Source timing/orientation mismatch')
                previous_us = us
                if fid in wanted:
                    rgb = frame.to_ndarray(format='rgb24')
                    for e in wanted[fid]:
                        if abs(us-e['timestamp_us']) > 1:
                            raise ValueError('Same-time RGB mismatch')
                        cr, mask = crop(rgb, e['proposed_box'])
                        target = OUT/'crops'/f"{e['crop_index']:03d}.npz"
                        np.savez_compressed(target, rgb=cr, pixel_valid=mask,
                                            bbox_xyxy=np.asarray(e['proposed_box'],np.int32),
                                            frame_id=np.int64(fid), timestamp_us=np.int64(e['timestamp_us']),
                                            region_estimated=np.bool_(True), hand_landmarks_observed=np.bool_(False))
                        pixel_records.append({**e, 'path':str(target), 'sha256':sha(target),
                                              'source_path':raw, 'decoded_pts_us':us,
                                              'raw_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                                              'padding_pixels':int((~mask).sum())})
                        panel = Image.new('RGB',(900,390),'white')
                        draw = ImageDraw.Draw(panel)
                        draw.text((8,5),f"#{e['crop_index']} prior={e['prior_event']} {e['side']} fid={fid}",fill='black')
                        draw.text((8,22),e['sample_id'],fill='black')
                        draw.text((8,39),e['prior_classification']+' / estimated region; real pixels',fill='black')
                        thumb = Image.fromarray(rgb).resize((576,324))
                        td = ImageDraw.Draw(thumb)
                        td.rectangle(tuple(v*576/rgb.shape[1] for v in e['proposed_box']),outline='red',width=2)
                        panel.paste(thumb,(8,60))
                        panel.paste(Image.fromarray(cr).resize((288,288)),(600,66))
                        rendered.append((e['crop_index'],panel))
                    found.add(fid)
                if len(found)==len(wanted):
                    break
            if found != set(wanted):
                raise ValueError('Missing requested RGB frame')
    write(OUT/'PIXEL_RECORDS.json', sorted(pixel_records,key=lambda e:e['crop_index']))
    rendered.sort(key=lambda item:item[0])
    for start in range(0,len(rendered),6):
        page = Image.new('RGB',(1800,1170),'white')
        for i,(_,panel) in enumerate(rendered[start:start+6]):
            page.paste(panel,((i%2)*900,(i//2)*390))
        page.save(OUT/'visual'/f'page_{start//6+1:02d}.jpg',quality=94)
    changed = [p for p,s in paths.items() if sha(p)!=s]
    write(OUT/'SOURCE_PRESERVATION.json', {'files':len(paths),'changed':changed})
    if changed:
        raise ValueError('Frozen inputs changed')
    print('RGB_EXTRACTION_COMPLETE',len(pixel_records),flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if OUT.is_dir() and not (OUT/'FIRST_FAILURE.json').exists():
            write(OUT/'FIRST_FAILURE.json', {'type':type(exc).__name__,'message':str(exc)})
        raise
