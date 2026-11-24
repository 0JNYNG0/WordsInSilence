"""Extract exact public RGB frames around every saved normal-warning peak.

No recognition, model, feature extraction or alignment invocation.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
OLD = OUT.parent.parent / 'word100_multipeople_live_feedback_20261101'
PINS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p = Path(p).resolve()
    PINS[str(p)] = sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))


def arr(p):
    p = Path(p).resolve()
    PINS[str(p)] = sha(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write(p, value):
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    public = read(OLD / 'PUBLIC_INPUTS.json')
    rows = {r['clip_id']: r for r in public['rows']}
    bank = read(OLD / 'REFERENCE_BANK.json')
    brows = {r['clip_id']: r for r in bank['rows']}
    pred = read(OLD / 'BANK_PREDICTIONS.json')
    prows = {r['clip_id']: r for r in pred['rows']}
    evaluation = read(OLD / 'FEEDBACK_EVALUATION.json')
    labels = {'오른손 모양(학습한 항목)': 'R_shape', '손목에서 손가락 뿌리로 향하는 방향': 'hand_axis',
              '손·팔의 이동과 위치': 'arm_path'}
    indices = {'R_shape': [0, 1, 3], 'hand_axis': [25, 26, 53, 54], 'arm_path': [2, 3, 4, 5, 30, 31, 32, 33]}
    cache = {}
    def sequence(cid):
        if cid not in cache:
            row = rows[cid]
            data = arr(row['input'])
            data['probabilities'] = arr(prows[cid]['probability_file'])['probabilities']
            cache[cid] = data
        return cache[cid]
    warnings = []
    requests = defaultdict(set)
    for ci in evaluation['case_index']:
        if ci['kind'] != 'same_expression':
            continue
        case = read(ci['path'])
        guard = read(OLD / 'guard_cases' / (ci['id'] + '.json'))
        for si, stage in enumerate(case['modes']['all9']['stages']):
            if stage['status'] != 'DIFFERENCE_CANDIDATE':
                continue
            assert guard['modes']['all9']['stages'][si]['status'] == stage['status']
            wid = 'W' + str(len(warnings) + 1).zfill(2)
            w = dict(id=wid, task_id=ci['id'], query=ci['query'], stage_index=si + 1,
                     canonical_stage_id=stage['canonical_stage_id'], aggregate_stage=stage,
                     guard_stage_status_unchanged=True, peak_pairs=[])
            for f in stage['features']:
                if f['status'] != 'DIFFERENCE_CANDIDATE':
                    continue
                family = labels[f['name']]
                for rid, peak in zip(f['reference_ids'], f['peak_at']):
                    pairptr = next(p for p in case['comparisons'] if p['reference'] == rid)
                    pair = read(pairptr['path'])
                    qi, ri = peak['query_frame'], peak['reference_frame']
                    assert [qi, ri] in pair['path']
                    d = dict(id=wid + '_' + rid.split('REAL')[-1].replace('_F', ''), reference=rid,
                             family=family, peak=peak, saved_pair=pairptr,
                             reference_stage=brows[rid]['stages'][si],
                             saved_stage=pair['stages'][si], source_description=stage['source_description'],
                             raw_windows={})
                    for side, cid, center in [('query', ci['query'], qi), ('reference', rid, ri)]:
                        a = sequence(cid)
                        window = list(range(max(0, center - 3), min(len(a['frame_id']), center + 4)))
                        images = [window[0], center, window[-1]]
                        requests[cid].update(images)
                        if family == 'R_shape':
                            values = a['probabilities'][:, indices[family]]
                            masks = np.isfinite(values)
                            names = [pred['targets'][j] for j in indices[family]]
                        else:
                            values = a['x'][:, indices[family]]
                            masks = a['mask'][:, indices[family]].astype(bool) & np.isfinite(values)
                            names = [public['feature_names'][j] for j in indices[family]]
                        vals = [[float(v) if np.isfinite(v) else None for v in row] for row in values[window]]
                        d['raw_windows'][side] = dict(clip_id=cid, indices=window,
                            frame_ids=a['frame_id'][window].tolist(), timestamp_us=a['timestamp_us'][window].tolist(),
                            feature_names=names, values=vals, masks=masks[window].tolist(), image_indices=images)
                    w['peak_pairs'].append(d)
            warnings.append(w)
    assert len(warnings) == 7 and sum(len(w['peak_pairs']) for w in warnings) == 14
    sources = {}
    for cid in requests:
        row = rows[cid]
        assert sha(row['source']) == row['source_sha256']
        PINS[row['source']] = row['source_sha256']
        sources[cid] = dict(path=row['source'], sha256=row['source_sha256'],
                            indices=sorted(requests[cid]), input=row['input'], input_sha256=row['input_sha256'])
    scope = dict(status='FIXED_FROM_ALL_SEVEN_SAVED_WARNINGS_BEFORE_RGB_VIEW',
                 warnings=warnings, source_pins=dict(PINS), sources=sources,
                 sampling='Every saved offending-feature peak for both selected whole references; center plus -3/+3 native frames, clipped only at source edges.',
                 video_sources='Exact public original MP4s, read-only; no private or camera source.',
                 model_calls=0, DTW_runs=0, thresholds_modified=False,
                 visual_review_is_human_KSL_ground_truth=False)
    write(OUT / 'WARNING_SCOPE.json', scope)
    (OUT / 'frames').mkdir()
    (OUT / 'sheets').mkdir()
    actual = {}
    for cid, req in requests.items():
        a = sequence(cid)
        requested = {int(a['frame_id'][i]): i for i in req}
        records = {}
        with av.open(rows[cid]['source']) as container:
            for fid, frame in enumerate(container.decode(video=0)):
                if fid not in requested:
                    continue
                ix = requested[fid]
                us = round(float(frame.pts * frame.time_base) * 1e6)
                assert us == int(a['timestamp_us'][ix]), (cid, fid, us, a['timestamp_us'][ix])
                p = OUT / 'frames' / f'{cid}_f{fid:04}.png'
                frame.to_image().save(p)
                records[ix] = dict(path=str(p), sha256=sha(p), frame_id=fid, timestamp_us=us)
                if len(records) == len(req):
                    break
        assert len(records) == len(req)
        actual[cid] = records
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 22)
    except OSError:
        font = ImageFont.load_default()
    sheets = []
    for w in warnings:
        for d in w['peak_pairs']:
            sheet = Image.new('RGB', (1800, 1100), 'white')
            draw = ImageDraw.Draw(sheet)
            for rownum, side in enumerate(['query', 'reference']):
                v = d['raw_windows'][side]
                for col, ix in enumerate(v['image_indices']):
                    record = actual[v['clip_id']][ix]
                    with Image.open(record['path']) as im:
                        box = (360, 30, min(1560, im.width), min(1050, im.height))
                        view = im.crop(box).resize((600, 510))
                        sheet.paste(view, (col * 600, rownum * 550 + 40))
                    label = f"{side} {v['clip_id'].split('WORD')[-1]} f{record['frame_id']} {record['timestamp_us']/1e6:.3f}s"
                    draw.text((col * 600 + 8, rownum * 550 + 8), label, fill='black', font=font)
            p = OUT / 'sheets' / (d['id'] + '.png')
            sheet.save(p)
            sheets.append(dict(id=d['id'], path=str(p), sha256=sha(p), crop=[360, 30, 1560, 1050],
                               scale='600x510 each image; original full frames retained', order='query top/reference bottom; -3/peak/+3 native frames'))
    assert all(sha(p) == v for p, v in PINS.items())
    write(OUT / 'EXTRACTION_MANIFEST.json', dict(status='PASS', public_clips=len(sources),
        exact_frames=sum(len(v) for v in actual.values()), sheets=sheets, frames=actual,
        source_pins=PINS, source_hashes_unchanged=True, model_calls=0, feature_extraction=0,
        DTW_runs=0, original_results_unchanged=True, script_sha256=sha(__file__)))
    print(json.dumps(dict(status='PASS', warnings=7, peak_pairs=14, clips=len(sources), frames=sum(len(v) for v in actual.values()))))


if __name__ == '__main__':
    main()
