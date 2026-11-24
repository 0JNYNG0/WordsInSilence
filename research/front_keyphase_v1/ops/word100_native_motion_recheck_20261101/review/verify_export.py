"""Independent read-only native export verification; never imports/runs prepare.py.

Only the JSON report beside this file is written. Original videos, earlier
annotations, exported PNGs, models and production files are never written.
"""
from __future__ import annotations

import collections
import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import av
import numpy as np
from PIL import Image, ImageDraw

N = Path(__file__).resolve().parent.parent
O = N.parent
W = O / 'word100_interval_motion_20261101'
U = O / 'word100_core_feature_split_20261101'
OUT = N / 'review' / 'EXPORT_VERIFICATION.json'
EXPECTED = ['N007_W1', 'N008_W1', 'N022_W2', 'N047_W1', 'S002_W1',
            'S004_W2', 'S005_W1', 'S020_W0', 'S039_W0']
checks = []
pins = {}
pin_origins = collections.defaultdict(list)
read_hashes = {}
windows = []


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check(group, name, passed, detail=None):
    item = {'group': group, 'name': name, 'passed': bool(passed)}
    if not passed and detail is not None:
        item['detail'] = detail
    checks.append(item)


def add_pins(origin, values):
    for path, expected in values.items():
        if path in pins:
            check('pin_consistency', f'{origin}: {path}', pins[path] == expected)
        pins[path] = expected
        pin_origins[path].append(origin)


def png(path):
    with Image.open(path) as im:
        check('png_encoding', str(path), im.format == 'PNG' and im.mode == 'RGB')
        return np.array(im)


def main():
    if OUT.exists():
        raise FileExistsError('Completed verification is immutable; review it instead of overwriting.')
    plan = load(N / 'PLAN_FROZEN.json')
    manifest = load(N / 'MANIFEST.json')
    blind = load(N / 'BLIND_NATIVE.json')
    prior = load(W / 'PILOT_MANIFEST.json')
    complete = load(W / 'COMPLETION.json')
    audit = load(W / 'PRESENCE_PILOT_AUDIT.json')
    old_frozen = load(W / 'PILOT_FROZEN.json')
    old_verify = load(W / 'review' / 'PRESENCE_VERIFICATION.json')
    inputs = load(U / 'INPUTS.json')
    baseline = load(U / 'SELECTION.json')['protected']
    prior_rows = {row['sample_id']: row for row in prior['rows']}
    input_rows = {row['alias']: row for row in inputs['rows']}
    by_blind = {row['sample_id']: row for row in blind['rows']}

    for origin, values in [('native plan', plan['pins']), ('native manifest', manifest['pins']),
                           ('old completion', complete['source_pins']), ('old audit', audit['source_pins']),
                           ('old frozen', old_frozen['pins']), ('old verifier', old_verify['source_pins']),
                           ('protected production/model baseline', baseline)]:
        add_pins(origin, values)
    for path, expected in pins.items():
        actual = sha(path)
        read_hashes[path] = actual
        check('protected_frozen_hash', path, actual == expected)
    for path in [N / 'PLAN_FROZEN.json', N / 'MANIFEST.json', N / 'BLIND_NATIVE.json', Path(__file__)]:
        read_hashes[str(path)] = sha(path)

    check('scope', 'frozen nine identities', plan['union_nine'] == EXPECTED)
    check('scope', 'protected production/model baseline contains32 pins', len(baseline) == 32)
    check('scope', 'six left cases retain old ambiguity identities',
          plan['six_left'] == complete['left_down_uncertainty_mismatch_samples'])
    check('scope', 'four hold support identities',
          plan['hold_support'] == ['N022_W2', 'S039_W0', 'S020_W0', 'S002_W1'])
    check('scope', 'exact union, no added windows', sorted(set(plan['six_left'] + plan['hold_support'])) == EXPECTED)
    check('scope', 'manifest completed counts', manifest['status'] == 'COMPLETE' and
          manifest['windows'] == 9 and manifest['source_frames'] == 117 and len(manifest['rows']) == 9)
    check('scope', 'ordered unique manifest windows', [r['sample_id'] for r in manifest['rows']] == EXPECTED)
    check('scope', 'blind exact identities', len(blind['rows']) == 9 and sorted(by_blind) == EXPECTED)
    check('scope', 'exact source rubric', blind['rubric'] == plan['rubric'] == str(W / 'RUBRIC.json') and
          blind['rubric_sha256'] == plan['rubric_sha256'] == sha(W / 'RUBRIC.json'))
    check('old_failure', 'old 24-window readiness failure preserved',
          complete['status'] == 'VALIDATION_COMPLETE_TRAINING_NOT_READY' and complete['windows'] == 24 and
          audit['status'] == 'REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING' and
          audit['all_windows'] == 24 and audit['all_predicate_cells'] == 96 and audit['training_allowed'] is False)
    check('old_failure', 'hold counterexamples still absent in both hands',
          all(complete['counts'][hand + '_hold']['agreed_negative'] == 0 for hand in ['R', 'L']))
    check('old_failure', 'six left unknown mismatches still counted',
          complete['counts']['L_down']['known_unknown_mismatch'] == 6)
    check('old_failure', 'zero new model execution in old completion',
          complete['training_runs'] == complete['actual_model_inference_runs'] == 0)

    for index, row in enumerate(manifest['rows'], 1):
        sid = row['sample_id']
        old = prior_rows[sid]
        inp = input_rows[sid.rsplit('_W', 1)[0]]
        source = Path(inp['source'])
        expected_ids = old['frame_ids']
        expected_times = dict(zip(expected_ids, old['timestamps_us']))
        frames = row['frames']
        native_rows = {f['frame_id']: f for f in frames}
        check('scope', sid + ' public fit input lineage', row['role'] == old['role'] == inp['role'] == 'fit' and
              inp['clip_id'].startswith('NIA_SL_WORD') and inp['source_method'] in
              ['existing_exact_public_cache', 'existing_public_archive_exact_sha'])
        check('identity', sid + ' alias/person/expression/ROI/source',
              row['alias'] == f'R{index:02}' and row['person'] == old['person'] == inp['person'] and
              row['word_id'] == old['word_id'] == inp['word_id'] and row['roi_bottom'] == inp['box'][3] and
              row['source'] == inp['source'])
        source_before = sha(source)
        check('source_sha', sid, source_before == inp['source_sha256'] == row['source_sha256'])
        read_hashes[str(source)] = source_before
        check('identity', sid + ' old montage path/hash', row['old_montage'] == old['image'] and
              row['old_montage_sha256'] == old['image_sha256'] == sha(old['image']))
        read_hashes[old['image']] = row['old_montage_sha256']
        check('frame_id_set', sid, [f['frame_id'] for f in frames] == expected_ids and len(frames) == 13 and
              expected_ids == list(range(expected_ids[0], expected_ids[0] + 13)))
        check('blind_consistency', sid, by_blind[sid] == {k: v for k, v in row.items() if k not in
              ['source', 'source_sha256', 'old_montage', 'old_montage_sha256', 'person', 'word_id', 'role']})

        decoded_crops = {}
        decoded_ids = []
        raw_pts = []
        max_time_error = 0
        dimensions = set()
        source_boxes = set()
        # Enumerate from source frame zero. No seek, elapsed-time normalization,
        # crop tracking, detector, resampling, or producer function is used.
        with av.open(str(source), mode='r') as container:
            stream = container.streams.video[0]
            stream_start_time = stream.start_time
            stream_time_base = str(stream.time_base)
            for ordinal, frame in enumerate(container.decode(stream)):
                if ordinal in expected_times:
                    saved = native_rows[ordinal]
                    rgb = frame.to_ndarray(format='rgb24')
                    height, width = rgb.shape[:2]
                    exact_us = Fraction(frame.pts) * Fraction(frame.time_base) * 1000000
                    absolute_us = round(exact_us)
                    err = abs(absolute_us - expected_times[ordinal])
                    max_time_error = max(max_time_error, err)
                    check('absolute_source_pts', f'{sid}/{ordinal}',
                          saved['timestamp_us'] == absolute_us and err <= 1,
                          {'source_absolute_us': absolute_us, 'prior_us': expected_times[ordinal],
                           'manifest_us': saved['timestamp_us']})
                    check('source_dimensions', f'{sid}/{ordinal}', saved['source_size'] == [width, height])
                    # Rational arithmetic derives the frozen central60% / y8%-95%.
                    box = (round(Fraction(width, 5)), round(Fraction(2 * height, 25)),
                           round(Fraction(4 * width, 5)), round(Fraction(19 * height, 20)))
                    check('fixed_crop_bounds', f'{sid}/{ordinal}', tuple(saved['crop_box']) == box)
                    x0, y0, x1, y1 = box
                    expected_crop = rgb[y0:y1, x0:x1]
                    full_file, fixed_file = Path(saved['full']), Path(saved['fixed'])
                    check('export_path', f'{sid}/{ordinal}', full_file == N / 'visuals' / sid / f'full_{ordinal:04}.png'
                          and fixed_file == N / 'visuals' / sid / f'fixed_{ordinal:04}.png')
                    for path, expected_sha in [(full_file, saved['full_sha256']), (fixed_file, saved['fixed_sha256'])]:
                        actual_sha = sha(path)
                        read_hashes[str(path)] = actual_sha
                        check('export_png_sha', str(path), actual_sha == expected_sha)
                    full = png(full_file)
                    crop = png(fixed_file)
                    check('full_pixels_vs_source', f'{sid}/{ordinal}', np.array_equal(full, rgb))
                    check('fixed_pixels_vs_source', f'{sid}/{ordinal}', np.array_equal(crop, expected_crop))
                    decoded_crops[ordinal] = expected_crop.copy()
                    decoded_ids.append(ordinal)
                    dimensions.add((width, height))
                    source_boxes.add(box)
                    raw_pts.append({'frame_id': ordinal, 'pts': frame.pts, 'time_base': str(frame.time_base),
                                    'absolute_timestamp_us': absolute_us})
                if ordinal >= expected_ids[-1]:
                    break
        check('decoded_coverage', sid, decoded_ids == expected_ids and len(decoded_crops) == 13)
        check('fixed_window_geometry', sid, len(dimensions) == len(source_boxes) == 1 and
              len({tuple(f['crop_box']) for f in frames}) == 1)
        check('source_sha_after_decode', sid, sha(source) == source_before)
        check('pair_count', sid, len(row['pairs']) == 7)
        for pair_index, pair in enumerate(row['pairs']):
            frame_ids = expected_ids[pair_index * 2:(pair_index + 1) * 2]
            check('pair_frame_order', f'{sid}/{pair_index}', pair['frame_ids'] == frame_ids)
            path = Path(pair['file'])
            check('pair_path', f'{sid}/{pair_index}', path == N / 'visuals' / sid / f'pair_{pair_index:02}.png')
            actual_sha = sha(path)
            read_hashes[str(path)] = actual_sha
            check('pair_sha', str(path), actual_sha == pair['sha256'])
            sheet = png(path)
            ch, cw = decoded_crops[frame_ids[0]].shape[:2]
            check('pair_native_dimensions', f'{sid}/{pair_index}', sheet.shape == (ch + 36, cw * 2, 3))
            for panel, fid in enumerate(frame_ids):
                actual_panel = sheet[36:, panel * cw:(panel + 1) * cw]
                check('pair_actual_pixels_vs_source', f'{sid}/{pair_index}/{fid}',
                      np.array_equal(actual_panel, decoded_crops[fid]))
            # Verify IDs/time labels separately from body pixels, and keep the
            # unused last right panel truly blank (including its header).
            header = Image.new('RGB', (cw * 2, 36), (32, 37, 43))
            draw = ImageDraw.Draw(header)
            for panel, fid in enumerate(frame_ids):
                label = f'{sid}  f{fid}  {expected_times[fid]/1e6:.6f}s  native1:1 fixed view'
                draw.text((panel * cw + 12, 8), label, fill='white')
            check('pair_header_labels', f'{sid}/{pair_index}', np.array_equal(sheet[:36], np.array(header)))
            if len(frame_ids) == 1:
                check('last_blank_panel', sid, np.all(sheet[:, cw:] == np.array([32, 37, 43], dtype=np.uint8)))
        actual_files = {p.name for p in (N / 'visuals' / sid).iterdir() if p.is_file()}
        expected_files = {f'{kind}_{fid:04}.png' for fid in expected_ids for kind in ['full', 'fixed']}
        expected_files.update(f'pair_{i:02}.png' for i in range(7))
        check('export_file_set', sid, actual_files == expected_files)
        windows.append({'sample_id': sid, 'role': inp['role'], 'clip_id': inp['clip_id'],
                        'source_method': inp['source_method'], 'source_sha256': source_before,
                        'frame_count': len(decoded_ids), 'pair_count': len(row['pairs']),
                        'source_sizes': [list(d) for d in sorted(dimensions)],
                        'fixed_crop_bounds': [list(b) for b in sorted(source_boxes)],
                        'source_stream_start_time': stream_start_time, 'source_stream_time_base': stream_time_base,
                        'maximum_prior_timestamp_error_us': max_time_error, 'absolute_source_pts': raw_pts})
        print(json.dumps({'verified_windows': index, 'total_windows': 9, 'sample_id': sid}), flush=True)

    # Each protected evidence file is hashed again after all read-only decoding.
    # This covers source videos, old FAIL/labels/input metadata, frozen PNGs and code.
    for path, before in read_hashes.items():
        check('read_set_unchanged', path, sha(path) == before)
    groups = {}
    for group in sorted({c['group'] for c in checks}):
        group_checks = [c for c in checks if c['group'] == group]
        groups[group] = {'checks': len(group_checks), 'passed': sum(c['passed'] for c in group_checks),
                         'failed': sum(not c['passed'] for c in group_checks)}
    failures = [c for c in checks if not c['passed']]
    report = {
        'status': 'PASS_INDEPENDENT_NATIVE_EXPORT_VERIFICATION' if not failures else 'FAIL_NATIVE_EXPORT_VERIFICATION',
        'at_kst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        'verifier': str(Path(__file__).resolve()), 'verifier_sha256': sha(__file__),
        'author_relation': 'Separate Codex subagent implemented verifier; no producer code imported or executed. Same AI system, not independent human semantic truth.',
        'scope': 'Original nine public fit windows; exact native source frame IDs, absolute PTS, RGB pixels, fixed crop, pair pixels/labels/blank panel, frozen prior failure and hashes.',
        'counts': {'windows': len(windows), 'source_videos': len(windows), 'native_frames': sum(w['frame_count'] for w in windows),
                   'full_pngs': sum(w['frame_count'] for w in windows), 'fixed_pngs': sum(w['frame_count'] for w in windows),
                   'pair_pngs': sum(w['pair_count'] for w in windows), 'blank_right_panels': len(windows),
                   'protected_frozen_files': len(pins), 'read_set_files_rehashed': len(read_hashes),
                   'protected_production_model_baseline_files': len(baseline),
                   'checks': len(checks), 'failed': len(failures)},
        'check_groups': groups, 'failures': failures, 'checks': checks, 'windows': windows,
        'protected_pins': pins, 'pin_origins': dict(pin_origins),
        'source_pins': {str(N / name): sha(N / name) for name in ['PLAN_FROZEN.json', 'MANIFEST.json', 'BLIND_NATIVE.json']},
        'execution': {'prepare_rerun': False, 'original_input_reextraction': 0, 'native_exports_rewritten': 0,
                      'training_runs': 0, 'model_inference_runs': 0, 'reannotation': False,
                      'private_recording_access': 0, 'models_loaded': 0, 'production_code_executed': 0,
                      'production_model_baseline_hash_reads': len(baseline),
                      'written_files': [str(OUT)], 'root_documents_changed': False},
        'limitations': ['Export integrity is not a movement-label or readiness pass.',
                       'C/T3F and protected production files were read only as bytes to verify32 supplied baseline hashes; no model was loaded or production code executed.',
                       'PyAV/libav and Pillow are also producer dependencies; this is independent selection/check logic, not a different video decoding implementation.'],
    }
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status': report['status'], 'counts': report['counts']}, ensure_ascii=False), flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
