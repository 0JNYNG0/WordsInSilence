"""Read-only FIT lineage audit; never invokes a neural model or old evaluator.

Checks the actual get_rows('fit') loader, all query source mappings and saved
input hashes, then one predeclared FIT01 example per condition including pixel
order and RGB-context provenance. Does not load DEV/H/private data.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import sys, json, hashlib, traceback
import numpy as np

Y = Path(__file__).resolve().parent
O = Y.parent
G = O / 'word100_guided_event_order_20261102'
F = O / 'word100_learned_flow_role_20261102'
N = O / 'word100_learned_motion_invariance_20261102'
S = O / 'word100_learned_phase_sequence_20261102'
T = O / 'word100_learned_state_context_balance_20261102'
L = O / 'word100_seated_fit_transfer_20261102'
B = O / 'word100_learned_action_process_20261102'
D = O / 'word100_action_episode_supervision_20261102'
E = O / 'word100_guide_conditioned_episode_20261102'
P = O / 'word100_learned_guide_feedback_20261102'
CID = 'NIA_SL_WORD0739_REAL01_F'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dest = Y / 'INPUT_CONTRACT_REVIEW.json'
    assert not dest.exists()
    inputs = read(G / 'ORDER_SUPERVISION.json')
    qrows = inputs['rows']
    assert len(qrows) == 630
    assert {r['recorded_id'] for r in qrows} == set(range(1, 10))
    assert sum(r['target'] >= 0 for r in qrows) == 576
    assert sum(r['target'] == -1 for r in qrows) == 54
    sources = read(B / 'PREPARED.json')['rows'] + read(L / 'PAIR_MANIFEST.json')['conditions']['seated_variation']
    bysource = {(r['clip_id'], r['kind']): r for r in sources}
    assert len(sources) == len(bysource) == 252
    original_specs = read(N / 'INPUT_MANIFEST.json')['rows']
    reverse_specs = read(S / 'REVERSE_FIT_MANIFEST.json')['rows']
    gap_specs = read(S / 'SHORT_GAP_FIT_MANIFEST.json')['rows']
    local_specs = read(T / 'LOCAL_FIT_MANIFEST.json')['rows']
    seated_specs = read(L / 'INPUT_MANIFEST.json')['rows']
    specmap = {(r['clip_id'], r['kind']): r for r in original_specs + reverse_specs + gap_specs + local_specs}
    specmap.update({(r['clip_id'], 'seated_variation'): r for r in seated_specs})
    assert len(specmap) == 252

    # Prohibit model execution while exercising the real FIT-only loader.
    import torch
    oldcall, oldload = torch.nn.Module._call_impl, torch.load
    def forbidden(*args, **kwargs):
        raise AssertionError('Model execution/weight loading forbidden during input-contract audit')
    torch.nn.Module._call_impl = forbidden
    torch.load = forbidden
    observed_opens = set()
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = str(args[0]).replace('\\', '/').lower()
        if p.endswith(('.json', '.npz', '.mp4', '.pt', '.pkl')):
            assert not any(s in p for s in ['/private_sessions/', '/mentor/', 'real14', 'real15', 'real16', 'dev_action_targets', 'development_feedback', '/dev_', '/heldout/']), p
            observed_opens.add(str(args[0]))
    sys.addaudithook(hook)
    try:
        sys.path.insert(0, str(F))
        import flow_model
        rr = flow_model.get_rows('fit')
    finally:
        torch.nn.Module._call_impl, torch.load = oldcall, oldload
    assert len(rr) == 252 and {r['person'] for r in rr} == set(range(1, 10))
    loaded = {r['flow_key']: r for r in rr}
    flow_metadata = read(F / 'FLOW_INPUTS.json')
    flowfit = {k: v for k, v in flow_metadata['items'].items() if k.startswith('fit_')}
    assert set(loaded) == set(flowfit)
    hash_checks, axis_checks = 0, 0
    for key, r in loaded.items():
        meta = bysource[(r['clip_id'], r['kind'])]
        fm = flowfit[key]
        assert (r['clip_id'], r['word_id'], r['kind'], r['person']) == (fm['clip_id'], fm['word_id'], fm['kind'], fm['person'])
        assert sha(meta['cache']) == meta['cache_sha256']
        assert sha(meta['input']) == meta['input_sha256']
        assert sha(fm['path']) == fm['sha256']
        hash_checks += 3
        with np.load(fm['path'], allow_pickle=False) as a:
            assert np.array_equal(a['source_frame_id'], r['mapping'])
            assert np.array_equal(a['query_frame_id'], np.arange(len(r['x'])))
            assert np.array_equal(a['image_available'], r['available'])
        spec = specmap[(r['clip_id'], r['kind'])]
        assert np.array_equal(r['mapping'], spec['source_frame_indices'])
        axis_checks += 1

    direct = {r['clip_id']: {p['frame']: p['action_family'] for p in r['points'] if p['mask']}
              for r in read(D / 'ACTION_TARGETS.json')['rows']}
    contexts = {r['clip_id']: {p['frame']: p['action_family'] for p in r['points'] if p['mask']}
                for r in read(E / 'CONTEXT_TARGETS.json')['rows']}
    for q in qrows:
        r = loaded[q['flow_key']]
        assert q['frames'] == len(r['x']) and q['clip_id'] == r['clip_id']
        mp = np.asarray(r['mapping'])
        a, b = q['first_witnesses'], q['second_witnesses']
        for side, target in [('first', q['asked_before']), ('second', q['asked_after'])]:
            idx = q[side + '_witnesses']
            assert mp[idx].tolist() == q['source_' + side + '_witnesses']
            assert all(direct[r['clip_id']][int(mp[i])] == target for i in idx)
            assert all(r['episode'][i] == target for i in idx)
        assert all(contexts[r['clip_id']][int(mp[i])] >= 8 for i in q['reviewed_nonmain'])
        truth = 1 if a and b and max(a) < min(b) else 0 if a and b and max(b) < min(a) else -1
        assert q['target'] == truth
        known = {int(f): 0 for f in q['reviewed_nonmain']}
        known.update({int(f): 1 for f in a}); known.update({int(f): 2 for f in b})
        assert q['occurrence_targets'] == [dict(frame=f, label=known[f]) for f in sorted(known)]
    for q, swapped in zip(qrows[::2], qrows[1::2]):
        assert q['flow_key'] == swapped['flow_key']
        assert q['asked_before'] == swapped['asked_after'] and q['first_witnesses'] == swapped['second_witnesses']
        assert (q['target'] < 0 and swapped['target'] < 0) or q['target'] + swapped['target'] == 1

    # Representative actual RGB/latent identity, axis and pixel transform evidence.
    reps = {r['kind']: r for r in rr if r['clip_id'] == CID}
    assert len(reps) == 7
    original = specmap[(CID, 'original')]
    with np.load(original['input'], allow_pickle=False) as a:
        base_rgb = a['rgb'].copy(); base_latent = a['latent'].copy(); base_times = a['timestamp_us'].copy()
    import av
    def decode_small(path):
        frames, pts = [], []
        with av.open(str(path)) as c:
            vs = c.streams.video[0]
            for f in c.decode(vs):
                pts.append(int(round(float(f.pts * vs.time_base) * 1e6)))
                frames.append(f.reformat(width=192, height=108).to_ndarray(format='rgb24'))
        times = np.asarray(pts, np.int64); times -= times[0]
        return np.stack(frames), times
    assert sha(original['source']) == original['source_sha256']
    original_pixels, original_pts = decode_small(original['source'])
    representative = []
    for kind, r in sorted(reps.items()):
        spec = specmap[(CID, kind)]
        input_hash = spec.get('analysis_sha256', spec.get('input_sha256'))
        assert sha(spec['input']) == input_hash
        with np.load(spec['input'], allow_pickle=False) as a:
            assert np.array_equal(r['x'], a['latent'])
            ids, times = a['frame_id'].copy(), a['timestamp_us'].copy()
            actual_rgb = a['rgb'].copy() if 'rgb' in a else None
        assert np.array_equal(ids, np.arange(len(ids))) and np.all(np.diff(times) > 0)
        mapping = np.asarray(r['mapping'], int)
        report = dict(kind=kind, clip_id=CID, recorded_id=1, fit_only=True,
            cache=r['cache'], cache_sha256=sha(r['cache']), input=spec['input'], input_sha256=input_hash,
            frames=len(ids), mapping_sha256=hashlib.sha256(mapping.tobytes()).hexdigest(),
            playback_timestamp_sha256=hashlib.sha256(times.tobytes()).hexdigest(),
            playback_time_first_last_us=[int(times[0]), int(times[-1])],
            latent_exactly_from_matching_input=True)
        if kind == 'pose_only_3':
            assert actual_rgb is None and np.array_equal(r['rgb'], base_rgb)
            assert np.array_equal(ids, mapping) and np.array_equal(times, base_times)
            gap = spec['gap_frames']
            assert not np.asarray(r['m'])[gap].any() and np.asarray(r['available'])[gap].all()
            assert not np.array_equal(r['x'], base_latent)
            report.update(transform_type='Sensor-only artificial3-frame pose gap; same actual RGB and native times, newly computed contextual latent.',
                gap_frames=gap, RGB_reextracted=False, RGB_preserved_exactly=True, image_still_available=True,
                not_an_actual_missing_action_or_video_blackout=True)
        else:
            assert np.array_equal(r['rgb'], actual_rgb)
            video = spec.get('path', spec.get('video'))
            assert sha(video) == spec['video_sha256']
            resultpath = Path(spec.get('result', str(Path(spec['input']).parent / 'RESULT.json')))
            result = read(resultpath)
            assert result['source_sha256'] == spec['video_sha256']
            pixels, pts = (original_pixels, original_pts) if kind == 'original' else decode_small(video)
            assert len(pixels) == len(ids) and np.max(np.abs(pts - times)) <= 2
            report.update(video=video, video_sha256=spec['video_sha256'], result=str(resultpath), result_sha256=sha(resultpath),
                stored_context_equals_matching_video_analysis=True, playback_pts_verified=True,
                RGB_reextracted_from_corresponding_pixels=True, not_cached_feature_reordering=True)
            if kind == 'seated_variation':
                report.update(transform_type='Actual cropped RGB camera-framing variation, no resized/shifted feature surrogate.',
                    actual_sensor_extent=spec['actual_sensor_extent'], cropped_action_evidence=spec['cropped_action_evidence'])
                assert not np.array_equal(actual_rgb, base_rgb)
            else:
                expected = original_pixels[mapping]
                diff = np.abs(pixels.astype(np.int16) - expected.astype(np.int16))
                report.update(pixel_mapping_mean_absolute_difference=float(diff.mean()),
                    pixel_mapping_max_frame_mean_difference=float(diff.mean((1, 2, 3)).max()),
                    transform_type={'original':'Original public full sensor input', 'speed_0.8':'Actual RGB frame repetition/re-encoding at30fps; slower playback',
                        'speed_1.25':'Actual RGB frame dropping/re-encoding at30fps; faster playback',
                        'actual_reversed_pixels':'Entire original RGB sequence reversed, then context re-extracted',
                        'actual_local_reorder':'Local original RGB frames reordered, then context re-extracted'}[kind])
                assert diff.mean((1, 2, 3)).max() < 6, 'Declared source pixel mapping does not match encoded video'
                if kind != 'original':
                    assert not np.array_equal(actual_rgb, base_rgb[mapping])
                    report['RGB_equals_simple_original_feature_remap'] = False
                    report['mean_RGB_difference_from_naive_feature_remap'] = float(np.abs(actual_rgb - base_rgb[mapping]).mean())
        representative.append(report)

    relations = {(r['clip_id'], min(r['asked_before'], r['asked_after']), max(r['asked_before'], r['asked_after'])) for r in qrows}
    clips = {r['clip_id'] for r in qrows}
    assert len(relations) == 45 and len(clips) == 27
    files = [Y/'SCOPE.json', G/'ORDER_SUPERVISION.json', F/'FLOW_INPUTS.json', F/'flow_model.py',
        B/'PREPARED.json', B/'process_model.py', D/'episode_model.py', E/'CONTEXT_TARGETS.json', D/'ACTION_TARGETS.json',
        L/'PAIR_MANIFEST.json', L/'transfer_data.py', N/'prepare_real_fit.py', S/'prepare_reverse_fit.py',
        S/'prepare_short_gap_fit.py', T/'prepare_inputs.py', L/'extract_seated.py', L/'prepare_seated.py',
        O/'word100_learned_body_relation_20261102/demo_adapter.py', P/'live_adapter.py', Path(__file__)]
    result = dict(status='PASS_FIT_INPUT_REUSE_WITH_EXPLICIT_CONTROL_SCOPE',
        at_utc=datetime.now(timezone.utc).isoformat(),
        actual_fit_loader_called=True, model_forward_or_weight_loading_during_loader_forbidden=True,
        new_model_inference=0, training=0, old_evaluation_rerun=0, development_private_mentor_H_access=False,
        input_variants=252, original_fit_clips=36, recorded_ids=list(range(1,10)),
        query_records=630, known_queries=576, held_queries=54, unique_order_source_clips=27,
        unique_original_unordered_relations=45, independent_630_cases=False,
        input_hash_checks=hash_checks, all_saved_flow_frame_maps_checked=axis_checks,
        all630_source_witness_and_mask_mappings_preserved=True,
        counts=[dict(kind=k[0], target=k[1], queries=v) for k,v in sorted(Counter((r['kind'],r['target']) for r in qrows).items())],
        representative_input_checks=representative,
        recommendation='Reuse existing FIT inputs for the fixed tied/direct structural comparison. No new backbone extraction is needed by this contract.',
        mandatory_limits=[
            '576 known queries derive from45 source relations in27 FIT clips, with seven transforms and swapped question pairs; group reporting by original clip/relation.',
            'pose_only_3 is an artificial detector-gap robustness input, not RGB occlusion, actual learner omission, or an independently performed video.',
            'Actual reverse/local reorder must stay physical-order controls; neither is automatically a linguistically incorrect sign.',
            'speed1.25 drops some original frames and cannot establish short-action preservation universally; only surviving direct witnesses retain supervision.',
            '54 local-reorder interleaved/missing-witness queries remain target=-1. Unlabelled frames/other action families do not become occurrence absence.',
            'Local edit seams have only the existing limited witness exclusions; newly extracted16-frame RGB context may still include synthetic joins. Report editing-domain limitations.',
            'DTW canonical coverage is an input hint and never proof of action completion or a dense occurrence label.',
            'This FIT audit does not validate DEV contract automatically; comparison must freeze and verify its own input mapping without changing labels or thresholds from outcome.',
        ],
        exclusions_required_by_input_failure=[],
        inspected_data_open_paths=sorted(observed_opens),
        pins={str(p):sha(p) for p in files},
        independent_humanexpert_truth=False, whole_goal_complete=False)
    with dest.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(result['status'], 'hashes', hash_checks, 'queries630 / relations45 / clips27', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        p=Y/'INPUT_CONTRACT_FIRST_FAILURE.txt'
        if not p.exists(): p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
