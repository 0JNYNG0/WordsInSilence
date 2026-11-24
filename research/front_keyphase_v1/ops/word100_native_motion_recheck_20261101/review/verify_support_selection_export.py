"""Independent finite-support selection and native pixel verification.

No producer script is imported or rerun. Only the final verification JSON is
written. Existing coordinate arrays are read without labels/RGB model features.
"""
from pathlib import Path
from fractions import Fraction
import collections
import datetime
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import av
import numpy as np
from PIL import Image, ImageDraw

N = Path(__file__).resolve().parent.parent
S = N / 'support_inventory'
U = N.parent / 'word100_core_feature_split_20261101'
W = N.parent / 'word100_interval_motion_20261101'
OUT = N / 'review/SUPPORT_SELECTION_EXPORT_VERIFICATION.json'
checks, read_pins = [], {}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ck(group, name, okay):
    checks.append(dict(group=group, name=name, passed=bool(okay)))


def pin(group, path, expected):
    p = str(path)
    actual = sha(p)
    ck(group, p, actual == expected)
    if p in read_pins:
        ck('pin_consistency', p, read_pins[p] == actual)
    read_pins[p] = actual


def png(p):
    with Image.open(p) as im:
        ck('png_encoding', str(p), im.format == 'PNG' and im.mode == 'RGB')
        return np.array(im)


def main():
    if OUT.exists():
        raise FileExistsError('Reuse immutable completed verification.')
    # Fail before doing any decoding if the original export has not completed.
    manifest, blind = read(S / 'MANIFEST.json'), read(S / 'BLIND_SUPPORT.json')
    plan, selection = read(S / 'PLAN_FROZEN.json'), read(S / 'SELECTION.json')
    export_freeze = read(S / 'EXPORT_FROZEN.json')
    prior_check = read(N / 'review/DIAGNOSIS_VERIFICATION.json')
    protected = dict(prior_check['protected_pins'])
    for origin in [plan['pins'], selection['pins'], export_freeze['pins'], prior_check['source_pins']]:
        for p, h in origin.items():
            ck('pin_consistency', p, p not in protected or protected[p] == h)
            protected[p] = h
    for p, h in protected.items():
        pin('protected_sha', p, h)
    for p in [Path(__file__), S / 'MANIFEST.json', S / 'BLIND_SUPPORT.json', S / 'EXPORT_FROZEN.json']:
        read_pins[str(p)] = sha(p)
    inputs = {r['alias']: r for r in read(U / 'INPUTS.json')['rows'] if r['role'] == 'fit'}
    visuals = {r['alias']: r['windows'] for r in read(U / 'VISUAL_INDEX.json')['rows']}
    old_rows = read(W / 'PILOT_MANIFEST.json')['rows']
    old_ids, old_clips = {r['sample_id'] for r in old_rows}, {r['clip_id'] for r in old_rows}
    candidates = selection['candidates']
    saved = {(r['sample_id'], r['hand']): r for r in candidates}
    ck('pool', 'exact60 fit clips', len(inputs) == 60)
    ck('pool', 'exact360 candidate rows, unique keys', len(candidates) == len(saved) == selection['opportunities'] == 360)
    recomputed = {}
    sample_ids = set()
    exclusions = collections.Counter()
    reason_combinations = collections.Counter()
    max_step_error = 0.0
    for alias, inp in sorted(inputs.items()):
        pin('fit_input_sha', inp['input'], inp['input_sha256'])
        ck('pool', alias + '/public fit lineage', inp['clip_id'].startswith('NIA_SL_WORD') and inp['source_method'] in
           ('existing_exact_public_cache', 'existing_public_archive_exact_sha'))
        with np.load(inp['input'], allow_pickle=False) as z:
            x, mask, ids, times = z['x'], z['mask'], z['frame_id'], z['timestamp_us']
            ck('input_schema', alias, x.shape == mask.shape and len(x) == len(ids) == len(times) and mask.dtype == bool and
               len(set(ids.tolist())) == len(ids) and np.all(np.diff(ids) > 0) and np.all(np.diff(times) > 0))
            names = inp['feature_names']
            index = {int(fid): pos for pos, fid in enumerate(ids)}
            ck('pool', alias + '/three fixed windows', len(visuals[alias]) == 3)
            for visual in visuals[alias]:
                sid = visual['sample_id']
                sample_ids.add(sid)
                locs = [index[f] for f in visual['frame_ids']]
                ck('window_frame_pts', sid, len(locs) == 13 and visual['frame_ids'] == list(range(visual['frame_ids'][0], visual['frame_ids'][0] + 13)) and
                   times[locs].tolist() == visual['timestamps_us'])
                for hand in 'RL':
                    key = (sid, hand)
                    row = saved[key]
                    cols = [names.index(hand + '_' + suffix) for suffix in ('hand_wrist_x', 'hand_wrist_y', 'palm_center_x', 'palm_center_y')]
                    points = x[np.ix_(locs, cols)]
                    valid = mask[np.ix_(locs, cols)]
                    reasons = []
                    if sid in old_ids:
                        reasons.append('previous24window')
                    if not (bool(np.all(valid)) and bool(np.all(np.isfinite(points)))):
                        reasons.append('incomplete_observed_fixedROI_position')
                    # Separately compute each Euclidean step in scalar float32
                    # arithmetic, rather than producer's vector norm/diff call.
                    steps = []
                    if not reasons:
                        for start in (0, 2):
                            for t in range(12):
                                dx = points[t + 1, start] - points[t, start]
                                dy = points[t + 1, start + 1] - points[t, start + 1]
                                steps.append(float(np.sqrt(dx * dx + dy * dy)))
                    score = min(steps) if steps else None
                    expected = dict(sample_id=sid, alias=alias, clip_id=inp['clip_id'], person=inp['person'], word_id=inp['word_id'], hand=hand,
                        eligible=not reasons, exclude_reasons=reasons, score=score, steps=steps if steps else None,
                        frame_ids=visual['frame_ids'], timestamps_us=visual['timestamps_us'], source=inp['source'], source_sha256=inp['source_sha256'],
                        input=inp['input'], input_sha256=inp['input_sha256'], roi_bottom=inp['box'][3])
                    ck('candidate_metadata', sid + '/' + hand, {k: v for k, v in row.items() if k not in ('score', 'steps')} ==
                       {k: v for k, v in expected.items() if k not in ('score', 'steps')})
                    ck('candidate_scalar_scores', sid + '/' + hand, row['score'] == score and row['steps'] == expected['steps'])
                    if steps:
                        max_step_error = max(max_step_error, max(abs(a - b) for a, b in zip(steps, row['steps'])))
                    recomputed[key] = expected
                    exclusions.update(reasons)
                    reason_combinations['+'.join(reasons) if reasons else 'eligible'] += 1
    ck('pool', 'all180 fixed windows and360 opportunities retained', len(sample_ids) == selection['windows'] == 180 and set(recomputed) == set(saved))
    ck('pool', 'old24 excludes both hands48 times', exclusions['previous24window'] == 48 and old_ids <= sample_ids)

    chosen, used, shortfalls = [], set(), []
    for roundno in range(2):
        for person in (1, 2, 3):
            for hand in 'RL':
                options = [r for r in recomputed.values() if r['eligible'] and r['person'] == person and r['hand'] == hand and r['clip_id'] not in used]
                options.sort(key=lambda r: (-r['score'], r['sample_id']))
                if not options:
                    shortfalls.append(dict(round=roundno, person=person, hand=hand))
                    continue
                choice = dict(options[0], selection_round=roundno, support_alias=f'C{len(chosen) + 1:02}')
                chosen.append(choice)
                used.add(choice['clip_id'])
    for i, row in enumerate(chosen):
        ck('selection_rank_order', row['support_alias'], selection['selected'][i] == row)
    balance = collections.Counter((r['person'], r['hand']) for r in chosen)
    ck('selection_contract', 'exact12 unique clips balanced3IDs2hands2rounds', len(chosen) == len(selection['selected']) == len(used) == 12 and
       balance == collections.Counter({(p, h): 2 for p in (1, 2, 3) for h in 'RL'}))
    ck('selection_contract', 'shortfalls exact', selection['shortfalls'] == shortfalls == [])
    ck('selection_contract', 'no old24 window selected', not old_ids.intersection(r['sample_id'] for r in chosen))
    ck('selection_contract', 'no model truth/denominator promotion', selection['model_scores_read'] == 0 and selection['coordinate_rank_not_truth'] is True and selection['original_test_denominator_unchanged'] is True)
    ck('manifest_scope', 'exact12/156 output', manifest['status'] == 'COMPLETE' and manifest['windows'] == 12 and manifest['native_frames'] == 156 and len(manifest['rows']) == 12 and
       manifest['selection_sha256'] == sha(S / 'SELECTION.json'))
    ck('blind_scope', 'exact pinned output/rubric', blind['status'] == 'COMPLETE' and blind['manifest_sha256'] == sha(S / 'MANIFEST.json') and
       blind['rubric'] == str(W / 'RUBRIC.json') and blind['rubric_sha256'] == sha(W / 'RUBRIC.json') and len(blind['rows']) == 12)
    exported = []
    for i, (row, chosen_row, blind_row) in enumerate(zip(manifest['rows'], chosen, blind['rows']), 1):
        alias, sid = chosen_row['support_alias'], chosen_row['sample_id']
        ck('export_identity', alias, row['support_alias'] == alias and row['sample_id'] == sid and row['target_hand'] == chosen_row['hand'] and
           row['source'] == chosen_row['source'] and row['source_sha256'] == chosen_row['source_sha256'] and row['person'] == chosen_row['person'] and
           row['word_id'] == chosen_row['word_id'] and row['role'] == 'fit' and row['roi_bottom'] == chosen_row['roi_bottom'])
        ck('blind_scope', alias, blind_row == {k: v for k, v in row.items() if k not in ['source', 'source_sha256', 'person', 'word_id', 'role']})
        pin('source_video_sha', row['source'], chosen_row['source_sha256'])
        expected_ids, expected_times = chosen_row['frame_ids'], chosen_row['timestamps_us']
        ck('export_frame_ids', alias, len(row['frames']) == 13 and [f['frame_id'] for f in row['frames']] == expected_ids)
        saved_frames = {f['frame_id']: f for f in row['frames']}
        decoded, frames_seen, source_pts = {}, [], []
        with av.open(row['source'], mode='r') as video:
            for fid, frame in enumerate(video.decode(video=0)):
                if fid in saved_frames:
                    metadata = saved_frames[fid]
                    rgb = frame.to_ndarray(format='rgb24')
                    height, width = rgb.shape[:2]
                    ts = round(Fraction(frame.pts) * Fraction(frame.time_base) * 1000000)
                    ck('absolute_source_pts', alias + '/' + str(fid), ts == metadata['timestamp_us'] and abs(ts - expected_times[expected_ids.index(fid)]) <= 1)
                    box = (round(Fraction(width, 5)), round(Fraction(2 * height, 25)), round(Fraction(4 * width, 5)), round(Fraction(19 * height, 20)))
                    ck('native_crop_geometry', alias + '/' + str(fid), tuple(metadata['crop_box']) == box and metadata['source_size'] == [width, height])
                    x0, y0, x1, y1 = box
                    crop = rgb[y0:y1, x0:x1]
                    for kind, expected_pixels in [('full', rgb), ('fixed', crop)]:
                        path = Path(metadata[kind])
                        ck('export_paths', alias + '/' + kind + str(fid), path == S / 'visuals' / alias / f'{kind}_{fid:04}.png')
                        pin('export_image_sha', path, metadata[kind + '_sha256'])
                        ck(kind + '_pixels_vs_source', alias + '/' + str(fid), np.array_equal(png(path), expected_pixels))
                    decoded[fid] = crop.copy()
                    frames_seen.append(fid)
                    source_pts.append(dict(frame_id=fid, pts=frame.pts, time_base=str(frame.time_base), absolute_timestamp_us=ts))
                if fid >= max(expected_ids):
                    break
        ck('decoded_full13', alias, frames_seen == expected_ids and len(decoded) == 13)
        ck('fixed_all13_geometry', alias, len({tuple(f['crop_box']) for f in row['frames']}) == 1 and len({tuple(f['source_size']) for f in row['frames']}) == 1)
        ck('pair_count', alias, len(row['pairs']) == 7)
        for j, pair in enumerate(row['pairs']):
            ids = expected_ids[j * 2:j * 2 + 2]
            ck('pair_ids_paths', alias + '/' + str(j), pair['frame_ids'] == ids and Path(pair['file']) == S / 'visuals' / alias / f'pair_{j:02}.png')
            pin('export_image_sha', pair['file'], pair['sha256'])
            pixels = png(pair['file'])
            ch, cw = decoded[ids[0]].shape[:2]
            ck('pair_dimensions', alias + '/' + str(j), pixels.shape == (ch + 36, cw * 2, 3))
            header = Image.new('RGB', (cw * 2, 36), (32, 37, 43))
            draw = ImageDraw.Draw(header)
            for panel, fid in enumerate(ids):
                ck('pair_pixels_vs_source', alias + '/' + str(fid), np.array_equal(pixels[36:, panel * cw:(panel + 1) * cw], decoded[fid]))
                text = f'{alias} f{fid} {expected_times[expected_ids.index(fid)]/1e6:.6f}s native1:1'
                draw.text((panel * cw + 12, 8), text, fill='white')
            ck('pair_headers', alias + '/' + str(j), np.array_equal(pixels[:36], np.array(header)))
            if len(ids) == 1:
                ck('last_blank_panel', alias, np.all(pixels[:, cw:] == np.array([32, 37, 43], dtype=np.uint8)))
        expected_files = {f'{kind}_{fid:04}.png' for fid in expected_ids for kind in ('full', 'fixed')} | {f'pair_{j:02}.png' for j in range(7)}
        ck('export_file_set', alias, {p.name for p in (S / 'visuals' / alias).iterdir()} == expected_files)
        exported.append(dict(alias=alias, sample_id=sid, target_hand=row['target_hand'], source_sha256=row['source_sha256'], absolute_source_pts=source_pts))
        print(json.dumps(dict(verified_support_windows=i, total=12, alias=alias)), flush=True)
    for path, digest in read_pins.items():
        ck('read_set_unchanged', path, sha(path) == digest)
    native = read(N / 'NATIVE_DIAGNOSIS.json')
    ck('earlier_results', 'native9 remains incomplete readiness5/6', native['status'] == 'SOURCE_DIAGNOSIS_COMPLETE_TRAINING_NOT_READY' and native['left_six_resolved'] == 5 and native['left_six_unresolved'] == ['N008_W1'] and native['training_allowed'] is False)
    ck('earlier_results', 'failed24 still failed', read(W / 'PRESENCE_PILOT_AUDIT.json')['status'] == 'REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING')
    groups = {}
    for group in sorted({r['group'] for r in checks}):
        values = [r for r in checks if r['group'] == group]
        groups[group] = dict(checks=len(values), passed=sum(r['passed'] for r in values), failed=sum(not r['passed'] for r in values))
    failures = [r for r in checks if not r['passed']]
    report = dict(status='PASS_INDEPENDENT_SUPPORT_SELECTION_EXPORT' if not failures else 'FAIL_SUPPORT_SELECTION_EXPORT',
        at_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        reviewer='Separate Codex verifier; no selection/export producer imported or executed',
        counts=dict(fit_clips=60, windows=180, hand_opportunities=360, eligible=sum(r['eligible'] for r in recomputed.values()),
                    excluded=sum(not r['eligible'] for r in recomputed.values()), selected_windows=12, selected_unique_clips=12,
                    native_frames=156, full_pngs=156, fixed_pngs=156, pair_pngs=84, blank_panels=12,
                    protected_files=len(protected), protected_production_model_baseline_files=32, rehashed_read_files=len(read_pins), checks=len(checks), failed=len(failures)),
        exclusion_reasons=dict(exclusions), exclusion_reason_combinations=dict(reason_combinations),
        selection_balance={f'person{p}_{h}': n for (p,h),n in sorted(balance.items())},
        selected_order=[dict(alias=r['support_alias'], sample_id=r['sample_id'], clip_id=r['clip_id'], person=r['person'], hand=r['hand'], score=r['score']) for r in chosen],
        existing_failed_pilot_clip_overlap=sum(r['clip_id'] in old_clips for r in chosen),
        maximum_scalar_step_error=max_step_error,
        check_groups=groups, checks=checks, failures=failures, exported_windows=exported,
        protected_pins=protected,
        source_pins={str(p):sha(p) for p in [Path(__file__), S/'PLAN_FROZEN.json', S/'SELECTION.json', S/'MANIFEST.json', S/'BLIND_SUPPORT.json', S/'EXPORT_FROZEN.json']},
        static_review=['The selector reads only existing x/mask/frame_id/timestamp_us arrays and public fit metadata; no stored feature/RGB/classifier score array is read.',
                       'Pinned motion_observer defines these position columns using contemporaneous shoulder centre/width and observed/usable masks; this run does not recompute features.',
                       'Score is min of24 adjacent native wrist/palm steps, tie by sample_id, with fixed2-round/person1,2,3/handR,L selection and unique clips.',
                       'No target/criterion change or replacement after viewing; direct RGB annotation is still required for actual hold support.'],
        limitations=['Observed coordinate masks and nonzero steps cannot establish real continuous movement or ROI eligibility; jitter/identity/normalizer effects still require direct RGB.',
                     'Selected windows may share public clips with earlier pilot windows; they are a support search, not fresh validation.',
                     'Pixel verification uses separate logic with the same PyAV/Pillow decoding dependencies.',
                     'No binary-label/readiness/learned-performance claim follows from selection/export PASS.'],
        execution=dict(selector_rerun=False, exporter_rerun=False, original_input_reextraction=0, training_runs=0, inference_runs=0,
                       private_recording_access=0, production_changes=False, root_documents_changed=False))
    with OUT.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('status', 'counts', 'exclusion_reasons', 'exclusion_reason_combinations')}, ensure_ascii=False))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
