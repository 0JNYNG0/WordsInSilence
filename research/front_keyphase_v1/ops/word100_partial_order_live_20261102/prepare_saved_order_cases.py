"""Join exact saved public inputs/results; never run a model or rescore a pair."""
from pathlib import Path
import sys
import json
import hashlib
import copy
import datetime
import traceback
import numpy as np

AA = Path(__file__).resolve().parent
O = AA.parent
Y = O / 'word100_direct_occurrence_order_20261102'
Z = O / 'word100_partial_order_feedback_20261102'
X = O / 'word100_learned_component_extent_20261102'
G = O / 'word100_learned_feedback_integration_20261102'
RET = O / 'word100_learned_feature_retention_20261102'
V = O / 'word100_learned_body_relation_20261102'
YDATA = Path('W:/WordsInSilence_derived/word100_direct_occurrence_order_20261102')
WORDS = ['WORD0739', 'WORD1133', 'WORD1567', 'WORD2523']
KINDS = {'normal': 'reencoded_normal', 'cut': 'action_cut',
         'order': 'actual_reverse_main', 'gap': 'main_blackout'}
MEDIA = {'normal': 'public', 'cut': 'control', 'order': 'order', 'gap': 'gap'}
FORBIDDEN = {'target', 'truth_reason', 'first_witnesses', 'second_witnesses',
             'raw_prediction', 'confirmed_prediction', 'person', 'kind',
             'clip_id', 'flow_key', 'word_id', 'key'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert not (AA / 'SAVED_CASES_MANIFEST.json').exists()
    # Isolated process: get_rows('dev') loads cached metadata/features only.
    # Do not construct a model or call observe/infer/evaluate.
    sys.path.insert(0, str(Y))
    import readout_model as model
    sys.path.insert(0, str(Z))
    from order_feedback import attach_partial_order
    sys.path.insert(0, str(X))
    from source_bridge import SourceBridge

    protocol_path = Y / 'EVALUATION_PROTOCOL.json'
    evaluation_path = YDATA / 'evaluations/tied_readout_dev.json'
    protocol = read(protocol_path)
    evaluation = read(evaluation_path)
    context_path = Z / 'RUNTIME_CONTEXT.json'
    context = read(context_path)
    assert context['graphs'] == protocol['graphs']
    rows = model.get_rows('dev')
    assert len(rows) == 88 and {r['person'] for r in rows} == {11, 12, 13}
    selected = {(r['word_id'], r['kind']): r for r in rows if r['kind'] in KINDS.values()}
    assert len(selected) == 16
    x_replay_path = X / 'PUBLIC_REPLAY_VERIFIED.json'
    x_replays = read(x_replay_path)['cases']
    x_amend_path = X / 'REVIEW_FIXES_VERIFIED.json'
    x_amended = read(x_amend_path)['rows']
    controls_path = RET / 'VIDEO_CONTROLS.json'
    controls = read(controls_path)['rows']
    order_path = G / 'MAIN_ORDER_RESULTS.json'
    order_rows = read(order_path)['rows']
    bridge = SourceBridge()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        if str(path) in pins:
            assert pins[str(path)] == digest, str(path)
        pins[str(path)] = digest
        return digest

    for path in [Path(__file__), protocol_path, evaluation_path, context_path,
                 Z / 'order_feedback.py', X / 'source_bridge.py', x_replay_path,
                 x_amend_path, controls_path, order_path,
                 O / 'word100_predicate_supervision_20261102/CLAUSE_COMPONENT_LINKS.json',
                 O / 'word100_source_component_connector_20261102/DIRECT_TARGETS.json']:
        pin(path)
    cases = {word: {} for word in WORDS}
    checks = []
    for word in WORDS:
        for short, kind in KINDS.items():
            r = selected[word, kind]
            assert r['person'] == 12
            inp = Path(r['input']).resolve()
            base_eval_path = Path(r['base_path']).resolve()
            pin(inp, r['input_sha256'])
            pin(base_eval_path, r['base_sha256'])
            pin(r['axis_path'], r['axis_sha256'])
            assert protocol['pins'][str(inp)] == r['input_sha256']
            a = dict(np.load(inp, allow_pickle=False))
            pred_path = YDATA / 'predictions/tied_readout/dev' / (r['flow_key'] + '.npz')
            pin(pred_path)
            prediction = dict(np.load(pred_path, allow_pickle=False))
            n = len(a['frame_id'])
            times = a['timestamp_us'] / 1e6
            assert n == len(r['x']) == len(prediction['physical'])
            assert np.all(np.diff(a['frame_id']) > 0) and np.all(np.diff(times) > 0)
            assert np.array_equal(times, r['times']) and np.array_equal(times, prediction['times'])
            assert np.array_equal(a['image_available'], r['available'])
            assert np.array_equal(a['image_available'], prediction['image_available'])
            assert np.array_equal(r['mapping'], prediction['mapping'])
            assert np.array_equal(r['current'], prediction['current'])
            assert np.array_equal(a['latent'], r['x'])
            old_eval_base = read(base_eval_path)
            supplemental = []
            if short == 'order':
                mapping = next(q for q in order_rows if q['word'] == word)
                video = Path(mapping['video']).resolve()
                assert Path(mapping['input']).resolve() == inp
                assert Path(mapping['result']).resolve() == base_eval_path
                video_hash = pin(video, mapping['video_sha256'])
                assert np.array_equal(mapping['source_frame_indices'], r['mapping'])
            else:
                mapping = next(q for q in controls if q['word_id'] == word and q['kind'] == kind)
                video = Path(mapping['path']).resolve()
                video_hash = pin(video, mapping['sha256'])
                assert np.array_equal(mapping['source_frame_indices'], r['mapping'])
            assert old_eval_base['word_id'] == word and old_eval_base['frames'] == n
            assert old_eval_base['source_sha256'] == video_hash
            if short != 'gap':
                x_kind = 'reverse_main' if short == 'order' else kind
                xr = next(q for q in x_replays if q['word'] == word and q['kind'] == x_kind)
                xm = next(q for q in x_amended if q['word'] == word and q['kind'] == x_kind)
                assert Path(xr['input']).resolve() == inp and xr['input_sha256'] == r['input_sha256']
                x_parent = read(xr['source'])
                pin(xr['source'], xr['source_sha256'])
                base_path = Path(xm['output']).resolve()
                pin(base_path, xm['sha256'])
                base = read(base_path)
                assert all(base[k] == value for k, value in x_parent.items())
                # X's additive source bridge names the exact saved input and G result.
                assert np.array_equal(base['path'], old_eval_base['path'])
                assert np.array_equal(np.asarray(base['overlay']['times']), times)
                source_mode = 'EXACT_EXISTING_X_RESULT_V2'
            else:
                base_path = base_eval_path
                base = old_eval_base
                relation_path = V / 'control_predictions' / f'rgb_geometry_42_{word}_{kind}.npz'
                relation_feedback = V / 'control_feedback' / f'rgb_geometry_42_{word}_{kind}.json'
                pin(relation_path)
                pin(relation_feedback)
                pin(V / 'evaluate_relations.py')
                relation = dict(np.load(relation_path, allow_pickle=False))
                relation_base = read(relation_feedback)
                assert np.array_equal(relation['frame_id'], a['frame_id'])
                assert np.array_equal(relation['timestamp_us'], a['timestamp_us'])
                assert relation_base['source_sha256'] == video_hash and relation_base['word_id'] == word
                assert np.array_equal(relation_base['path'], base['path'])
                assert relation['probability'].shape == (n, 3)
                source_input = dict(a, relation_probability=relation['probability'])
                linked = bridge.attach(base, source_input)
                assert all(linked[k] == value for k, value in base.items())
                base = linked
                supplemental = [str(relation_path.resolve()), str(relation_feedback.resolve())]
                source_mode = 'PURE_X_SOURCE_BRIDGE_ON_EXACT_RET_BASE_AND_SAVED_V_RELATIONS'
            assert base['word_id'] == word and base['frames'] == n
            assert base['source_sha256'] == video_hash
            assert np.array_equal(base['path'], old_eval_base['path'])

            saved_pairs = [q for q in evaluation['pairs'] if q['flow_key'] == r['flow_key']]
            graph = context['graphs'][word]
            assert {(q['first'], q['second']) for q in saved_pairs} == {tuple(v) for v in graph['precedences']}
            pairs = []
            for q in saved_pairs:
                assert q['word_id'] == word and q['kind'] == kind and q['comparison_frames'] == n
                assert q['clip_id'] == r['clip_id']
                assert q['unobserved_frames'] == np.flatnonzero(~np.asarray(a['image_available'], bool)).tolist()
                for prefix in ['first_evidence', 'second_evidence', 'raw_physical_first', 'raw_physical_second']:
                    assert np.array_equal(times[q[prefix + '_frames']], np.asarray(q[prefix + '_seconds']))
                pairs.append({k: copy.deepcopy(v) for k, v in q.items() if k not in FORBIDDEN})
            merged = attach_partial_order(base, pairs, context['families'], graph)
            assert all(merged[k] == value for k, value in base.items())
            if short == 'gap':
                assert all(merged[k] == value for k, value in old_eval_base.items())
            if not graph['precedences']:
                assert merged['partial_order_feedback']['status'] == 'NO_DISTINCT_ORDER_QUERY'
                assert merged['partial_order_feedback']['repeat_count'] is None
            payload = merged['partial_order_feedback']
            assert payload['whole_sign_correctness'] is None and payload['complete_action_verified'] is False
            assert payload['missing_action_verified'] is False and payload['completion_unknown'] is True
            for item in payload['items']:
                assert item['complete_action_verified'] is False and item['missing_action_verified'] is False
            media_url = f'/media/{MEDIA[short]}/{word}'
            merged['saved_order_case_provenance'] = dict(
                source_input=str(inp), source_input_sha256=r['input_sha256'],
                original_base_result=str(base_path), original_base_result_sha256=sha(base_path),
                saved_tied_prediction=str(pred_path), saved_tied_prediction_sha256=sha(pred_path),
                served_video_sha256=video_hash, saved_evaluation=str(evaluation_path),
                saved_evaluation_sha256=sha(evaluation_path), source_mode=source_mode,
                cached_result=True, new_inference=0, new_training=0, rescored=False,
                end_to_end_latency_measured_this_request=False,
                overlay_available='overlay' in base,
                overlay_notice=None if 'overlay' in base else '이 저장된 이미지 누락 사례에는 관절 오버레이가 없습니다. 영상과 확인된 시간 근거만 표시합니다.',
                limitation='부분 관측의 순서이며 전체 완료·빠진 동작·필수 반복 횟수 판정이 아닙니다.')
            dest = AA / 'saved_cases' / word / short / 'RESULT.json'
            if dest.exists():
                assert read(dest) == merged, 'Existing prepared result differs: ' + str(dest)
            else:
                save(dest, merged)
            cases[word][short] = dict(
                word_id=word, case_kind=short, saved_kind=kind, flow_key=r['flow_key'],
                video=str(video), video_sha256=video_hash, media_url=media_url,
                result=str(dest), result_sha256=sha(dest), analysis=str(inp), analysis_sha256=r['input_sha256'],
                base_result=str(base_path), base_result_sha256=sha(base_path),
                y_base_result=str(base_eval_path), y_base_result_sha256=r['base_sha256'],
                source_sha256=base['source_sha256'], base_source_sha_matches_video=True,
                prediction=str(pred_path), prediction_sha256=sha(pred_path),
                evaluation=str(evaluation_path), evaluation_sha256=sha(evaluation_path),
                source_mode=source_mode, supplemental_sources=supplemental,
                frames=n, native_frame_ids=a['frame_id'].tolist(), native_times_seconds=times.tolist(),
                source_frame_mapping=np.asarray(r['mapping']).tolist(), image_unavailable_frames=np.flatnonzero(~np.asarray(a['image_available'], bool)).tolist(),
                native_axis_exact=True, existing_base_fields_preserved=True,
                overlay_available='overlay' in base, query_count=len(pairs),
                saved_pair_statuses=[q['status'] for q in saved_pairs],
                review_status='PASS_EXACT_SAVED_SOURCE_AND_AXIS_BINDING',
                new_inference=0, new_training=0, rescored=False)
            checks.append(dict(word=word, kind=short, frames=n, queries=len(pairs),
                               video_source_exact=True, native_axes_exact=True, base_fields_exact=True))
            print(word, short, 'PASS', len(pairs), 'saved pairs', flush=True)
    for path, digest in pins.items():
        assert sha(path) == digest, path
    manifest = dict(status='PASS_16_EXACT_PUBLIC_SAVED_CASES',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=cases,
        preparation_script=str(Path(__file__).resolve()), pins=pins, checks=checks,
        evaluation='Saved pair status display only; no changed scores, labels, thresholds or inference.',
        new_inference=0, new_training=0, old_evaluation_rescored=False,
        no_private_inputs=True, no_holdout_access=True, whole_goal_complete=False)
    save(AA / 'SAVED_CASES_MANIFEST.json', manifest)
    save(AA / 'SAVED_CASES_PREPARED.json', dict(status=manifest['status'],
         manifest=str(AA / 'SAVED_CASES_MANIFEST.json'), manifest_sha256=sha(AA / 'SAVED_CASES_MANIFEST.json'),
         cases=16, saved_pairs=sum(c['queries'] for c in checks), new_inference=0, new_training=0,
         gap_overlay_available=False, source_video_hash_matches_base_all=True,
         no_completion_or_omission_verdict_added=True, whole_goal_complete=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure = AA / 'SAVED_CASES_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf8')
        raise
