"""Freeze AO public cache compatibility before any new AH/order inference."""
import ast
from collections import Counter
import datetime
import sys
import traceback
import numpy as np
from input_data import (AO, O, Y, AH, AJ, AN, M, R, MODEL_SHA, read, sha, require,
                        array_sha, npz, selected_y_rows, prepare_915, prepare_915_from_arrays,
                        aux_from_prediction)

G = O / 'word100_learned_feedback_integration_20261102'
PHASE = O / 'word100_learned_phase_sequence_20261102'
FLOW = O / 'word100_learned_flow_role_20261102'
PR = O / 'word100_learned_action_process_20261102'


def main():
    require(not (AO / 'INPUT_PLAN.json').exists(), 'Completed AO preparation is immutable')
    checks = []
    pins = {}
    historical_verified = {}

    def check(name, ok):
        require(bool(ok), name)
        checks.append(name)

    y_frozen = read(Y / 'FROZEN.json')
    ah = read(AH / 'DATASET_FROZEN.json')
    ah_eval = read(AH / 'EVAL_RESULT.json')
    ah_train = read(AH / 'TRAIN_RESULT.json')
    aj = read(AJ / 'INPUT_FROZEN.json')
    aj_result = read(AJ / 'RESULT.json')
    an = read(AN / 'reverse_inputs.json')
    an_result = read(AN / 'REVERSE_RESULT.json')
    phase = read(PHASE / 'REVERSE_FIT_MANIFEST.json')
    gap_plan = read(G / 'FROZEN.json')
    gap_records = read(G / 'SENSOR_GAP_RESULTS.json')
    protocol = read(Y / 'EVALUATION_PROTOCOL.json')
    trusted = {}
    for doc in [y_frozen, ah, aj, an, protocol]:
        for path, digest in doc['pins'].items():
            if path in trusted:
                check('historical agreement:' + path, trusted[path] == digest)
            trusted[path] = digest

    def pin(path, expected=None):
        path = str(__import__('pathlib').Path(path).resolve())
        digest = sha(path)
        if expected is not None:
            check('declared:' + path, digest == expected)
        if path in trusted:
            check('historical:' + path, digest == trusted[path])
            historical_verified[path] = digest
        if path in pins:
            check('stable:' + path, pins[path] == digest)
        pins[path] = digest
        return dict(path=path, sha256=digest)

    for path in [AO / 'input_data.py', AO / 'prepare_inputs.py', Y / 'FROZEN.json',
                 Y / 'EVALUATION_PROTOCOL.json', Y / 'readout_model.py',
                 Y / 'INPUT_CONTRACT_REVIEW.json', FLOW / 'flow_model.py',
                 FLOW / 'FLOW_INPUTS.json', PR / 'PREPARED.json',
                 AH / 'DATASET_FROZEN.json', AH / 'EVAL_RESULT.json', AH / 'TRAIN_RESULT.json',
                 AH / 'data.py', AH / 'local_events.py', AH / 'SAVED_VERIFIED.json',
                 AJ / 'INPUT_FROZEN.json', AJ / 'RESULT.json', AJ / 'aj_data.py',
                 AN / 'reverse_inputs.json', AN / 'REVERSE_RESULT.json', AN / 'reverse_data.py',
                 PHASE / 'REVERSE_FIT_MANIFEST.json', PHASE / 'prepare_reverse_fit.py',
                 G / 'FROZEN.json', G / 'test_sensor_gaps.py', G / 'SENSOR_GAP_RESULTS.json',
                 M / 'motion_model.py', M / 'NORMALIZATION.json', R / 'rgb_model.py',
                 R / 'RGB_NORMALIZATION.json',
                 O / 'word100_process_contract_alignment_20261102/process_rgb_support.py',
                 O / 'word100_learned_guide_feedback_20261102/live_adapter.py',
                 O / 'word100_signrep_feasibility_20261029/common.py']:
        pin(path)
    check('fixed AH model metadata', ah_train['model_sha256'] == MODEL_SHA ==
          aj_result['model_sha256'] == an_result['model_sha256'])
    pin(ah_train['model'], MODEL_SHA)  # Byte hash only, never torch.load.
    geo, rgb = read(M / 'NORMALIZATION.json'), read(R / 'RGB_NORMALIZATION.json')
    check('FIT-only unchanged stats', geo['fit_people'] == list(range(1, 10)) and
          rgb['fit_only'] and ah['normalization']['refit'] is False)
    check('five supported frozen AH targets', ah['supported_targets'] == list(range(5)))

    originals = {r['clip_id']: r for r in ah['rows']}
    native_predictions = {}
    for r in ah_eval['rows']:
        if r['clip_id'] in native_predictions:
            check('same native prediction:' + r['clip_id'], native_predictions[r['clip_id']] == r['prediction'])
        native_predictions[r['clip_id']] = r['prediction']
    aj_rows = {r['case_id']: r for r in aj['rows']}
    aj_predictions = {r['case_id']: r['prediction'] for r in aj_result['rows']}
    an_rows = {r['word_id']: r for r in an['rows']}
    an_predictions = {r['word_id']: r['prediction'] for r in an_result['rows']}
    reverse_rows = {r['clip_id']: r for r in phase['rows']}
    gap_sources = {r['word_id']: r for r in gap_plan['sources']}
    gaps = {r['case']: {k: r[k] for k in ['case', 'word', 'center', 'kind', 'frames']}
            for r in gap_records['rows']}
    check('gap inventory36', len(gaps) == 36)
    pure = {'np': np}
    for path, names in [(M / 'motion_model.py', {'features', 'normalize'}),
                        (R / 'rgb_model.py', {'rgb_input'})]:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        check('pure arithmetic definitions:' + str(path), len(nodes) == len(names))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), pure)

    rows_by_split = {}
    excluded = []
    for split in ['fit', 'dev']:
        yrows = selected_y_rows(split)
        out = []
        for r in yrows:
            key, cid, word, kind = r['flow_key'], r['clip_id'], r['word_id'], r['kind']
            check(key + ':role identity', r['person'] in (range(1, 10) if split == 'fit' else [11, 12, 13]) and
                  cid == f'NIA_SL_{word}_REAL{r["person"]:02d}_F')
            for field, hfield in [('input', 'input_sha256'), ('cache', 'cache_sha256'),
                                  ('base_path', 'base_sha256'), ('axis_path', 'axis_sha256')]:
                if field in r:
                    pin(r[field], r[hfield])
            for p, h in r.get('extra_pins', []):
                pin(p, h)
            row = dict(flow_key=key, split=split, source_group_id=cid,
                       y_row={k: r[k] for k in ['clip_id', 'person', 'word_id', 'kind', 'input', 'input_sha256']},
                       source_frame_mapping=np.asarray(r['mapping']).tolist(),
                       y_available=np.asarray(r['available'], bool).tolist(),
                       y_array_hashes={k: array_sha(r[k]) for k in
                                       ['x', 'rgb', 'z', 'm', 'current', 'physical', 'episode', 'role', 'available']},
                       training_queries=r.get('order_queries', []),
                       eval_queries=[q for q in protocol['query_manifest'][split] if q['flow_key'] == key])
            binding = None
            prediction = None
            prepared = None
            gap_case = None
            if kind in {'original', 'native'}:
                old = originals[cid]
                source = pin(old['source'], old['source_sha256'])
                saved_input = pin(old['input'], old['input_sha256'])
                for field in ['shape_observation', 'relation_observation']:
                    pin(old[field]['path'], old[field]['sha256'])
                a = npz(r['input'])
                original_a = npz(old['input'])
                base_path = __import__('pathlib').Path(r['input']).parent / 'RESULT.json'
                base = read(base_path)
                pin(base_path)
                check(key + ':same source bytes', base['source_sha256'] == source['sha256'])
                check(key + ':same native axes across distinct snapshots', all(np.array_equal(a[k], original_a[k])
                      for k in ['frame_id', 'timestamp_us']))
                check(key + ':original image availability not changed',
                      np.array_equal(np.asarray(a.get('image_available', np.ones(len(a['frame_id']), bool))),
                                     np.asarray(original_a.get('image_available', np.ones(len(a['frame_id']), bool)))))
                # Original AH predictions are reusable ONLY for exact915 input
                # equality. Same source/time alone is insufficient.
                original_shape = npz(old['shape_observation']['path'])
                original_relation = npz(old['relation_observation']['path'])
                check(key + ':AH own auxiliary axis', all(np.array_equal(original_a[k], o[k])
                      for o in [original_shape, original_relation] for k in ['frame_id', 'timestamp_us']))
                original_a['retained_probability'] = original_shape['probabilities']
                original_a['relation_probability'] = original_relation['probability']
                original_a.setdefault('image_available', np.ones(len(original_a['frame_id']), bool))
                original_prepared = prepare_915_from_arrays(original_a, ah['geometry_valid'])
                input_binding = dict(route='same_analysis', analysis=pin(r['input'], r['input_sha256']))
                prepared = prepare_915(input_binding, ah['geometry_valid'])
                exact = np.array_equal(prepared['x'], original_prepared['x'])
                equality = {k: np.array_equal(a[k], original_a[k], equal_nan=True) for k in ['x', 'mask', 'rgb']}
                binding = dict(status='REUSE_SAVED_AH' if exact else 'NEEDS_FROZEN_AH_FORWARD',
                               origin='AH_native_exact915_reuse' if exact else 'native_new_exact_Y_snapshot',
                               input_binding=input_binding, input_snapshot=input_binding['analysis'], input_snapshot_same_as_Y=True,
                               source=source, model_sha256=MODEL_SHA,
                               completed_AH_snapshot=saved_input, completed_AH_prediction=native_predictions[cid],
                               completed_AH_915_sha256=array_sha(original_prepared['x']), completed_AH_915_equal=bool(exact),
                               completed_AH_raw_equality=equality, completed_AH_rerun=False,
                               rgb_lineage='Actual saved Y full sensor snapshot and matching7shape/3relation observations; existing encoder recipe only',
                               scope='Reuse only exact915 equality. Otherwise one new forward on the distinct Y915 snapshot; never rerun completed AH input.')
                if exact:
                    prediction = native_predictions[cid]
                    binding['prediction'] = prediction
                ids, ts = prepared['ids'], prepared['ts']
            elif kind == 'actual_reversed_pixels':
                old = reverse_rows[cid]
                check(key + ':exact reverse manifest', old['input'] == r['input'] and old['analysis_sha256'] == r['input_sha256'])
                check(key + ':exact reverse source map', old['source_frame_indices'] == row['source_frame_mapping'])
                source = pin(old['path'], old['video_sha256'])
                pin(old['source'], old['source_sha256'])
                input_binding = dict(route='same_analysis', analysis=pin(r['input'], r['input_sha256']))
                prepared = prepare_915(input_binding, ah['geometry_valid'])
                base_path = __import__('pathlib').Path(r['input']).parent / 'RESULT.json'
                base = read(base_path)
                pin(base_path)
                check(key + ':actual reverse physical origin', base['source_sha256'] == source['sha256'] and
                      base['word_id'] == word and base['relation_condition'] == 'rgb_geometry' and base['relation_seed'] == 42)
                binding = dict(status='NEEDS_FROZEN_AH_FORWARD', origin='FIT_actual_RGB_reverse',
                               input_binding=input_binding, source=source, model_sha256=MODEL_SHA,
                               rgb_lineage='Pinned original reverse generator -> actual pixel reverse -> saved full extraction;16/stride2 recipe, window vectors absent',
                               probabilities_reversed_from_original=False, new_sensor_or_RGB_required=False)
                ids, ts = prepared['ids'], prepared['ts']
            elif kind == 'actual_reverse_main':
                old = an_rows[word]
                check(key + ':AN exact input', old['analysis']['path'] == r['input'] and old['analysis']['sha256'] == r['input_sha256'])
                source = pin(old['video']['path'], old['video']['sha256'])
                prediction = an_predictions[word]
                binding = dict(status='REUSE_SAVED_AH', origin='AN_actual_reverse_main', prediction=prediction,
                               input_snapshot=old['analysis'], input_snapshot_same_as_Y=True,
                               source=source, model_sha256=MODEL_SHA,
                               normalized_915_sha256=old['normalized_915_sha256'])
                ids, ts = np.asarray(old['native_frame_ids']), np.asarray(old['timestamp_us'])
                masks = dict(motion_anchor=old['motion_anchor'], evidence_available=old['evidence_available'],
                             rgb_available=old['image_available'])
            elif kind.startswith(('pose_only_', 'pose_and_image_')):
                # The original generator is exactly test_sensor_gaps.py. No
                # decoder/model or generic historical module is imported here.
                suffix, center = kind.rsplit('_f', 1)
                gap_case = f'{word}_f{center}_{suffix}'
                gr = gaps[gap_case]
                source_record = gap_sources[word]
                source = pin(source_record['source'], source_record['source_sha256'])
                original = aj_rows[word + '_reencoded_normal']
                check(key + ':gap source equality', source_record['input'] == original['analysis']['path'] and
                      source['sha256'] == original['video']['sha256'])
                check(key + ':gap exact source coordinates', gr['center'] == int(center) and gr['kind'] == suffix and
                      gr['frames'] == list(range(int(center) - len(gr['frames'])//2, int(center) + len(gr['frames'])//2 + 1)))
                input_binding = dict(route='sensor_gap_reconstruction', analysis=pin(original['analysis']['path'], original['analysis']['sha256']),
                                     gap_prediction=pin(r['input'], r['input_sha256']), gap_kind=suffix, gap_frames=gr['frames'],
                                     generator=str(G / 'test_sensor_gaps.py'), generator_lines=[23,24,25,26,30,31,38])
                prepared = prepare_915(input_binding, ah['geometry_valid'])
                binding = dict(status='NEEDS_FROZEN_AH_FORWARD', origin='DEV_exact_sensor_gap_recipe', input_binding=input_binding,
                               source=source, model_sha256=MODEL_SHA, new_sensor_or_RGB_required=False,
                               original_AH_predictions_reused=False,
                               rgb_lineage='Same saved normal RGB windows; selected representation rows zeroed only for pose_and_image. Neighbor window context still contains original pixels. This is feature/sensor missingness, not actual black-video re-extraction.')
                ids, ts = prepared['ids'], prepared['ts']
                a = prepared['raw']
                check(key + ':gap Y image mask exact', np.array_equal(a['image_available'], r['available']))
                check(key + ':gap source rows unchanged', np.array_equal(r['mapping'], ids))
                check(key + ':gap Y latent exact', np.array_equal(a['latent'], r['x']))
                check(key + ':gap masked raw geometry exact', np.all(a['x'][gr['frames']] == 0) and not a['mask'][gr['frames']].any())
            else:
                old = aj_rows[word + '_' + kind]
                check(key + ':AJ exact input', old['analysis']['path'] == r['input'] and old['analysis']['sha256'] == r['input_sha256'])
                source = pin(old['video']['path'], old['video']['sha256'])
                prediction = aj_predictions[old['case_id']]
                pin(old['relation']['path'], old['relation']['sha256'])
                check(key + ':AJ source frame mapping', row['source_frame_mapping'] == old['source_ledger']['source_frame_indices'])
                binding = dict(status='REUSE_SAVED_AH', origin='AJ_actual_pixel_control', prediction=prediction,
                               input_snapshot=old['analysis'], input_snapshot_same_as_Y=True,
                               source=source, model_sha256=MODEL_SHA)
                ids, ts = np.asarray(old['native_frame_ids']), np.asarray(old['timestamp_us'])
                saved = npz(prediction['path'])
                masks = {k: saved[k].tolist() for k in ['motion_anchor', 'evidence_available', 'rgb_available']}
                check(key + ':AJ saved observation mask', masks['evidence_available'] == old['evidence_available'])
            if prepared is not None:
                a = prepared['raw']
                z, mask = pure['features'](a, a['retained_probability'], a['relation_probability'])
                mask &= np.isfinite(z) & a['image_available'][:, None]
                independent = np.concatenate([pure['normalize'](np.where(mask, z, 0), mask, geo), pure['rgb_input'](a, rgb)], 1)
                check(key + ':exact independent AH915 arithmetic', np.array_equal(independent, prepared['x']))
                check(key + ':all915finite', prepared['x'].shape == (len(ids), 915) and np.isfinite(prepared['x']).all())
                binding['normalized_915_sha256'] = array_sha(prepared['x'])
                binding['raw_component_hashes'] = {k: array_sha(a[k]) for k in
                       ['frame_id', 'timestamp_us', 'x', 'mask', 'rgb', 'image_available', 'retained_probability', 'relation_probability']}
                masks = dict(motion_anchor=prepared['motion_anchor'].tolist(), evidence_available=prepared['evidence_available'].tolist(),
                             rgb_available=prepared['available'].tolist())
            check(key + ':native frame indexing', np.array_equal(ids, np.arange(len(r['x']))) and np.all(np.diff(ts) > 0))
            if 'times' in r:
                check(key + ':Y exact input PTS', np.array_equal(np.rint(np.asarray(r['times']) * 1e6).astype(np.int64), ts))
            row.update(native_frame_ids=ids.tolist(), timestamp_us=ts.tolist(), aux_masks=masks, aux_binding=binding)
            if prediction is not None:
                pin(prediction['path'], prediction['sha256'])
                aux, av = aux_from_prediction(npz(prediction['path']), row)
                check(key + ':aux finite8', aux.shape == (len(ids), 8) and np.isfinite(aux).all())
                binding['aux_sha256'] = array_sha(aux)
            # Verify old label-witness binding without using any candidate score.
            for q in row['eval_queries']:
                check(key + ':fixed first witnesses', q['first_witnesses'] == np.flatnonzero(r['episode'] == q['first']).tolist())
                check(key + ':fixed second witnesses', q['second_witnesses'] == np.flatnonzero(r['episode'] == q['second']).tolist())
            out.append(row)
        rows_by_split[split] = out
    # Exact existing Y total-reverse DEV12 is outside the prospectively selected
    # main-reverse experiment; unlike gap36, it is not required for its mask gate.
    excluded.append(dict(split='dev', kinds=['actual_reverse'], rows=12,
                         reason='Whole-clip reverse is outside the prescribed main-range reverse4; AH output absent. Excluded before candidate scores, not counted as passed protection.'))
    missing = [r for rr in rows_by_split.values() for r in rr if r['aux_binding']['status'] == 'NEEDS_FROZEN_AH_FORWARD']
    native_missing = sum(r['aux_binding']['origin'] == 'native_new_exact_Y_snapshot' for r in missing)
    check('bounded missing inventory', len(missing) == 72 + native_missing and 0 <= native_missing <= 48)
    check('FIT original/reverse paired', all(v == 2 for v in Counter(r['y_row']['clip_id'] for r in rows_by_split['fit']).values()))
    denominators = {}
    for split, rr in rows_by_split.items():
        qs = [q for r in rr for q in r['eval_queries']]
        tq = [q for r in rr for q in r['training_queries']]
        denominators[split] = dict(rows=len(rr), original_sources=len({r['source_group_id'] for r in rr}),
                                  kinds=dict(Counter(r['y_row']['kind'] for r in rr)), unswapped_queries=len(qs),
                                  known=sum(q['target'] >= 0 for q in qs), held=sum(q['target'] < 0 for q in qs),
                                  source_relations=len({(r['source_group_id'], q['first'], q['second']) for r in rr for q in r['eval_queries']}),
                                  training_queries=len(tq), training_known=sum(q['target'] >= 0 for q in tq))
    check('FIT known180', denominators['fit']['training_known'] == 180)
    # Pin actual old project Python import closure, but never import sensor/media/model runtimes.
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = __import__('pathlib').Path(file).resolve()
            if path.suffix == '.py' and path.is_relative_to(O):
                pin(path)
    check('no media native imports', not any(k in sys.modules for k in ['av', 'cv2', 'mediapipe', 'openvino']))
    for path, digest in pins.items():
        check('final pin:' + path, sha(path) == digest)
    result = dict(status='FROZEN_CACHE_INPUT_PLAN_PENDING_MISSING_AH',
                  at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows_by_split,
                  denominators=denominators, pins=pins, historical_pins_verified=historical_verified,
                  model=dict(path=ah_train['model'], sha256=MODEL_SHA), geometry_valid=ah['geometry_valid'],
                  auxiliary_channels=['centered_probability0', 'centered_probability1', 'centered_probability2',
                                      'centered_probability3', 'centered_probability4', 'motion_anchor', 'evidence_available', 'RGB_available'],
                  new_AH_forward_request=dict(total=len(missing), fit_actual_reversed_pixels=36, dev_reconstructed_sensor_gaps=36,
                                              native_distinct_Y_snapshots=native_missing,
                                              flow_keys=[r['flow_key'] for r in missing], new_sensor=0, new_RGB=0,
                                              new_DTWs=0, native_saved_reinference=0, AJ_saved_reinference=0, AN_saved_reinference=0),
                  reused_saved_predictions=76-native_missing, native_snapshot_policy='Completed AH native predictions reused only if full915 exact; mismatched snapshots require new precisely bound AH forward. No completed AH snapshot rerun.',
                  short_contrary_status='NOT_EVALUATED_NO_EXISTING_LOCAL_SHORT_DEV_GOLD; retain synthetic contract test separately',
                  exclusions=excluded, metadata_forbidden_to_train=['person', 'word_id', 'kind', 'source_frame_mapping', 'edit_metadata', 'truth', 'timestamps'],
                  unknown_is_not_negative=True, source_grouped_pairs=True, fit_registered_ids=list(range(1,10)), dev_registered_ids=[11,12,13],
                  model_loads=0, model_forwards=0, training=0, media_decodes=0, old_results_regraded=False, whole_goal_complete=False)
    check('exact reused inventory', sum(r['aux_binding']['status'] == 'REUSE_SAVED_AH' for rr in rows_by_split.values() for r in rr) == 76-native_missing)
    with (AO / 'INPUT_PLAN.json').open('x', encoding='utf-8') as f:
        __import__('json').dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    report = dict(status='PASS_CACHE_COMPATIBILITY_PENDING_FROZEN_AH_FORWARDS', checks_count=len(checks), checks=checks,
                  input_plan_sha256=sha(AO / 'INPUT_PLAN.json'), denominators=denominators,
                  new_AH_forward_request=result['new_AH_forward_request'], reused_saved_predictions=76-native_missing,
                  distinct_native_observer_snapshots=native_missing, historical_pin_count=len(historical_verified),
                  model_loads=0, model_forwards=0, training=0, media_decodes=0, old_results_regraded=False,
                  limitations=['Native AH predictions are not reused when915 differs; new exact Y snapshots are separate input evaluations, not completed AH input reruns.',
                               'Sensor-gap36 reconstructs representation masking, not actual RGB-video blackout.',
                               'AH scores are weak learned local evidence, never dense event or order labels.',
                               'No actual local/short contrary DEV gold exists in selected Y inventory; synthetic behavior is separate.'],
                  approved_for_pair_training_now=False)
    with (AO / 'INPUT_COMPATIBILITY.json').open('x', encoding='utf-8') as f:
        __import__('json').dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(dict(status=report['status'], checks=len(checks), denominators=denominators,
               input_plan_sha256=report['input_plan_sha256']), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        path = AO / 'PREPARATION_FIRST_FAILURE.txt'
        if not path.exists():
            path.write_text(traceback.format_exc(), encoding='utf-8')
        raise
