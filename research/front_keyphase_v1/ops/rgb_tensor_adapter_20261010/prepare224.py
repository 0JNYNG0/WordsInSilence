"""Derive224 directly from local video, reusing frozen128 crop geometry."""
import json
from pathlib import Path
from collections import defaultdict, Counter
import av
import numpy as np
from rgb_input import crop_original, sha, validate, require

UNIT = Path(__file__).resolve().parent
BASE = UNIT.parents[1]
OUT = BASE/'runs/rgb_tensor_adapter_v1'
INDEX = BASE/'ops/rgb_comparison_contract_20261010/INPUT_INDEX.json'
MANIFEST = BASE/'ops/rgb_pose_readiness_20261010/PAIRED_INPUT_MANIFEST.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def run():
    OUT.mkdir(exist_ok=False)
    try:
        execute()
    except Exception as e:
        write(OUT/'FIRST_FAILURE.json', {'type': type(e).__name__, 'message': str(e)})
        raise


def execute():
    index, manifest = read(INDEX), read(MANIFEST)
    by_id = {m['sample_id']: m for m in manifest}
    pins = dict(index['source_sha256'])
    pins.update({str(p.resolve()): sha(p) for p in [INDEX, MANIFEST, UNIT/'SCOPE.json', UNIT/'rgb_input.py', Path(__file__)]})
    for m in manifest:
        for k in ('raw', 'observation'):
            pins[m['source'][k+'_path']] = m['source'][k+'_sha256']
    require(all(sha(p) == h for p, h in pins.items()), 'Frozen source hash changed')
    write(OUT/'SOURCE_PINS.json', pins)
    groups = defaultdict(list)
    for s in index['samples']:
        groups[by_id[s['sample_id']]['source']['raw_path']].append(s)
    results, counts = [], Counter()
    for group_no, (video, samples) in enumerate(sorted(groups.items()), 1):
        buffers, wanted, frame_digests = {}, defaultdict(list), {}
        for s in samples:
            sid = s['sample_id']; n = len(s['rows'])
            pose, old = load(s['shared_pose_path']), load(s['observed_rgb_path'])
            require(np.array_equal(old['frame_id'], pose['frame_id']) and np.array_equal(old['timestamp_us'], pose['timestamp_us']), 'Old timeline changed')
            a = dict(pose_x=pose['x'].copy(), frame_id=pose['frame_id'].copy(), timestamp_us=pose['timestamp_us'].copy(),
                     rgb=np.zeros((n, 2, 224, 224, 3), np.uint8), pixel_valid=np.zeros((n, 2, 224, 224), bool),
                     available=np.zeros((n, 2), bool), observed_rgb=old['crop_valid'].copy(),
                     region_estimated=np.zeros((n, 2), bool),
                     landmark_observed_count=old['hand_usable'].sum(2).astype(np.int32),
                     bbox_xyxy=np.full((n, 2, 4), -1, np.int32),
                     source_width_height=old['source_width_height'].copy(),
                     shoulder_relative_size=old['shoulder_relative_size'].copy(),
                     frame_relative_size=old['frame_relative_size'].copy(),
                     padding_fraction=np.full((n, 2), np.nan, np.float64))
            for row in s['rows']:
                j = row['row']
                require(j < n and row['frame_id'] == int(a['frame_id'][j]) and row['timestamp_us'] == int(a['timestamp_us'][j]), 'Index timeline mismatch')
                for side, hand in enumerate(row['hands']):
                    require(hand['side'] == ('left', 'right')[side], 'Side order changed')
                    if hand['observed_rgb'] is not None:
                        ref = hand['observed_rgb']
                        require(ref['row'] == j and ref['side_index'] == side and ref['path'] == s['observed_rgb_path'], 'Wrong observed link')
                        a['available'][j, side] = True
                        a['bbox_xyxy'][j, side] = old['bbox_xyxy'][j, side]
                    elif hand['estimated_region_rgb'] is not None:
                        p = load(hand['estimated_region_rgb']['path'])
                        require(int(p['frame_id']) == row['frame_id'] and int(p['timestamp_us']) == row['timestamp_us'], 'Wrong estimated time')
                        a['available'][j, side] = a['region_estimated'][j, side] = True
                        a['bbox_xyxy'][j, side] = p['bbox_xyxy']
                        # Do not invent shoulder-relative hand size during a landmark gap.
                        a['shoulder_relative_size'][j, side] = np.nan
                        a['frame_relative_size'][j, side] = (p['bbox_xyxy'][2]-p['bbox_xyxy'][0])/old['source_width_height'][j, 0]
                wanted[row['frame_id']].append((sid, j))
            buffers[sid] = a
            frame_digests[sid] = []
        got = set()
        with av.open(video, mode='r') as con:
            stream = con.streams.video[0]
            require(float(stream.metadata.get('rotate', '0')) % 360 == 0, 'Rotated stream')
            require(stream.sample_aspect_ratio is None or stream.sample_aspect_ratio == 1, 'Non-square pixels')
            previous = -1
            for fid, frame in enumerate(con.decode(stream)):
                require(frame.pts is not None and stream.time_base is not None, 'Missing PTS')
                ts = int(round(frame.pts * stream.time_base * 1_000_000))
                require(ts > previous and (fid != 0 or ts == 0), 'Nonmonotonic/origin PTS')
                previous = ts
                if fid in wanted:
                    rgb = frame.to_ndarray(format='rgb24')
                    digest = __import__('hashlib').sha256(rgb.tobytes()).hexdigest()
                    require(getattr(frame, 'rotation', 0) % 360 == 0, 'Rotated frame')
                    for sid, j in wanted[fid]:
                        a = buffers[sid]
                        require(ts == int(a['timestamp_us'][j]), 'PTS mismatch')
                        require(np.array_equal(a['source_width_height'][j], [frame.width, frame.height]), 'Dimension mismatch')
                        for side in range(2):
                            if a['available'][j, side]:
                                box = a['bbox_xyxy'][j, side]
                                a['rgb'][j, side], a['pixel_valid'][j, side] = crop_original(rgb, box)
                                x0, y0, x1, y1 = map(int, box)
                                area = (min(frame.width, x1)-max(0,x0))*(min(frame.height,y1)-max(0,y0))
                                a['padding_fraction'][j, side] = 1-area/(x1-x0)**2
                        frame_digests[sid].append({'frame_id': fid, 'timestamp_us': ts, 'raw_rgb_sha256': digest})
                    got.add(fid)
                if fid >= max(wanted):
                    break
        require(got == set(wanted), 'Incomplete decode')
        for s in samples:
            sid = s['sample_id']; a = buffers[sid]; n = validate(a)
            dest = OUT/(sid+'.npz')
            with dest.open('xb') as f:
                np.savez_compressed(f, **a)
            results.append({'sample_id': sid, 'cache_path': str(dest.resolve()), 'cache_sha256': sha(dest),
                            'source_video': video, 'source_pose': s['shared_pose_path'],
                            'frames': n, 'frame_digests': frame_digests[sid],
                            'scope': 'development_fixture_only_no_accuracy_claim'})
            counts.update(rows=n, slots=n*2, observed=int(a['observed_rgb'].sum()),
                          estimated=int(a['region_estimated'].sum()), available=int(a['available'].sum()))
        if group_no % 20 == 0:
            print(json.dumps({'videos_completed': group_no, 'videos_total': len(groups)}), flush=True)
    require(len(results) == 113 and dict(counts) == {'rows':2789,'slots':5578,'observed':5315,'estimated':78,'available':5393}, 'Counts differ')
    unchanged = all(sha(p) == h for p,h in pins.items())
    require(unchanged, 'Sources changed')
    write(OUT/'CACHE_MANIFEST.json', results)
    write(OUT/'BUILD_CHECK.json', {'status':'PASS','segments':len(results),'videos':len(groups),'counts':dict(counts),
                                 'source_files_unchanged':len(pins), 'resolution_source':'original_frame_direct224',
                                 'trained_or_inferred':False})
    print(json.dumps({'status':'PASS','counts':dict(counts)}), flush=True)


if __name__ == '__main__':
    run()
