"""Build the fixed public observer bank, then calibrate FIT-only ranges once.

Commands are separate: build (predictions + canonical stage projection),
calibrate (36 FIT queries x8 other-person references), selftest (synthetic only).
The root executes build/calibrate after reviewing the final engine. Existing
completed predictions are reused only with exact input identity. No detector,
encoder, label fitting, target-rule creation, private video, or download occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parent
SEM = O/'word100_semantic_feature_cycle_20261101'
OLD = O/'word100_stage_feedback_cycle_20261031'
DEMO = O/'word100_public_feedback_demo_20261101'
DESCRIPTIONS = O/'word100_description_evidence_feedback_20261101/REPORT_DATA.json'
CASE_INPUTS = O/'word100_action_feedback_casebook_20261101/INPUTS.json'
TARGETS = ('R_index_only', 'R_open_four', 'L_open_four', 'R_closed_four',
           'L_closed_four', 'R_higher', 'L_higher')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value, compact=False):
    path = Path(path).resolve()
    if not path.is_relative_to(U):
        raise ValueError('New unit output only')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False,
                  indent=None if compact else 2, separators=(',', ':') if compact else None)


def module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def load_input(row, names):
    if sha(row['input']) != row['input_sha256']:
        raise ValueError('Public input changed: ' + row['clip_id'])
    with np.load(row['input'], allow_pickle=False) as z:
        a = {k: z[k].copy() for k in ('x', 'mask', 'rgb', 'frame_id', 'timestamp_us')}
    n = len(a['frame_id'])
    if (a['x'].shape != (n, 69) or a['mask'].shape != (n, 69) or a['mask'].dtype != bool
            or a['rgb'].shape != (n, 768) or not np.isfinite(a['rgb']).all()
            or not np.isfinite(a['x'][a['mask']]).all() or (np.diff(a['frame_id']) <= 0).any()
            or (np.diff(a['timestamp_us']) <= 0).any()):
        raise ValueError('Invalid public native input')
    a['feature_names'] = names
    return a


def check_probability_axes(path, a, expected_sha=None):
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError('Stored probabilities changed')
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z['frame_id'], a['frame_id']) or not np.array_equal(z['timestamp_us'], a['timestamp_us']):
            raise ValueError('Probability/native exact frame/PTS mismatch')
        if 'targets' in z and tuple(z['targets'].tolist()) != TARGETS:
            raise ValueError('Probability target order mismatch')
        p = z['probabilities'].copy()
    if p.shape != (len(a['frame_id']), 7) or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError('Invalid saved posterior schema')
    return p


class BaselineSources:
    """Prefer exact completed full results, then old3 and saved new4 centers."""
    def __init__(self, training, pins):
        self.pins = pins
        self.case = read(CASE_INPUTS)['rows']
        dense = read(DEMO/'DENSE_MANIFEST.json')
        if dense['seed'] != 42 or tuple(dense['targets']) != TARGETS:
            raise ValueError('Historical demo observer route changed')
        self.dense = {self.case[r['alias']]['clip_id']: r for r in dense['rows']}
        self.dense_inputs = {r['clip_id']: r for r in self.case.values()}
        base = next(r for r in training['protected_baseline'] if r['seed'] == 42)
        self.new_dev = {r['clip_id']: r for r in base['full_outputs']}
        self.old_info = read(OLD/'LEARNING_INPUTS.json')
        self.old_rows = {r['clip_id']: (i, r) for i,r in enumerate(self.old_info['rows'])}
        self.native_path = OLD/'runs/pose_plus_signrep_42/probabilities.npy'
        manifest = read(OLD/'runs/pose_plus_signrep_42/RUN.json')
        if sha(self.native_path) != manifest['prediction_sha256']:
            raise ValueError('Historical native probabilities changed')
        self.native = np.load(self.native_path, allow_pickle=False)
        self.old_cols = [self.old_info['targets'].index(t) for t in TARGETS[:3]]
        self.sem_info = read(SEM/'DATASET.json')
        splitpath = SEM/'runs/protected_shape_split_features_42/probabilities.npy'
        splitmeta = read(SEM/'runs/protected_shape_split_features_42/COMPLETE.json')
        if sha(splitpath) != splitmeta['probabilities_sha256']:
            raise ValueError('Historical center probabilities changed')
        self.split = np.load(splitpath, allow_pickle=False)
        self.centers = {(r['clip_id'], r['center_frame']): (i, r) for i,r in enumerate(self.sem_info['rows'])}
        for path in (CASE_INPUTS, DEMO/'DENSE_MANIFEST.json', OLD/'LEARNING_INPUTS.json',
                     self.native_path, OLD/'runs/pose_plus_signrep_42/RUN.json', SEM/'DATASET.json',
                     splitpath, SEM/'runs/protected_shape_split_features_42/COMPLETE.json'):
            pins[str(path)] = sha(path)

    def predict(self, row, a, adapter, train):
        clip = row['clip_id']; provenance = []
        fingerprint = train.array_fingerprint(a)
        if clip in self.dense and self.dense_inputs[clip]['input_sha256'] == row['input_sha256']:
            source = self.dense[clip]; path = Path(source['path'])
            p = check_probability_axes(path, a, source['sha256'])
            self.pins[str(path)] = source['sha256']
            return p, dict(route='reused_completed_public_demo_full7', new_old3_frames=0, new4_centers=0,
                           reused_full_frames=len(p), sources=[dict(path=str(path), sha256=source['sha256'])])
        if clip in self.new_dev:
            source = self.new_dev[clip]
            if source['array_fingerprint'] != fingerprint:
                raise ValueError('Development baseline cache has different native input')
            path = Path(source['path']); p = check_probability_axes(path, a, source['sha256'])
            self.pins[str(path)] = source['sha256']
            return p, dict(route='reused_new_development_baseline_full7', new_old3_frames=0, new4_centers=0,
                           reused_full_frames=len(p), sources=[dict(path=str(path), sha256=source['sha256'])])
        runtime = adapter._head_runtime(); adapter._frozen_heads(runtime)
        import torch
        n = len(a['frame_id']); p = np.full((n, 7), np.nan, np.float32)
        old_calls = 0
        if clip in self.old_rows:
            oi, old = self.old_rows[clip]
            if old['input_sha256'] != row['input_sha256']:
                raise ValueError('Known completed old3 input differs; do not silently rerun it')
            p[:, :3] = self.native[oi, :n][:, self.old_cols]
            provenance.append(dict(kind='original_fullclip_old3', path=str(self.native_path), row=oi,
                                   input_sha256=old['input_sha256']))
        else:
            xx, zz = adapter.normalize(a, runtime['old_stats'])
            with torch.inference_mode():
                pred = runtime['old'](torch.from_numpy(xx[None]), torch.from_numpy(zz[None]),
                                      torch.ones(1, n)).sigmoid().numpy()[0]
            p[:, :3] = pred[:, runtime['old_columns']]; old_calls = n
            provenance.append(dict(kind='previously_uninferred_fullclip_old3', frames=n))
        reused = []; todo = []
        for i in adapter.window_centers(a['frame_id']):
            cached = self.centers.get((clip, int(a['frame_id'][i])))
            if cached is not None:
                ci, original = cached
                if original['input_sha256'] != row['input_sha256'] or original['center_timestamp_us'] != int(a['timestamp_us'][i]):
                    raise ValueError('Known completed center input differs')
                if not np.array_equal(p[i, :3], self.split[ci, :3]):
                    raise ValueError('Protected old3 differs from completed center baseline')
                p[i, 3:] = self.split[ci, 3:]; reused.append(int(i))
            else:
                todo.append(int(i))
        if todo:
            xx, zz = adapter.normalize(a, runtime['new_stats'])
            with torch.inference_mode():
                for k in range(0, len(todo), 64):
                    ids = todo[k:k+64]
                    x = torch.from_numpy(np.stack([xx[i-6:i+7] for i in ids]))
                    z = torch.from_numpy(np.stack([zz[i-6:i+7] for i in ids]))
                    p[ids, 3:] = runtime['new'](x, z).sigmoid().numpy()
        provenance.append(dict(kind='new4_exact13_centers', reused_center_indices=reused, new_center_indices=todo))
        return p, dict(route='protected_old3_plus_new4_with_exact_reuse', new_old3_frames=old_calls,
                       new4_centers=len(todo), reused_new4_centers=len(reused), sources=provenance)


def canonical_stages(report, reference_alias, word):
    choices = [(i, c) for i,c in enumerate(report['cases']) if c['reference'] == reference_alias]
    if not choices:
        raise ValueError('Canonical reference has no saved stage description')
    ci, case = choices[0]
    identity = [(s['id'], s['reference_core'], s['reference_rgb_description']) for s in case['stages']]
    if any([(s['id'], s['reference_core'], s['reference_rgb_description']) for s in c['stages']] != identity for _,c in choices):
        raise ValueError('Canonical own stages disagree across saved cases')
    output = []
    for j, s in enumerate(case['stages']):
        cards = []
        for k, card in enumerate(s['source_cards']):
            # Query links from that historical pair are not runtime inputs.
            keys = ('clause_id', 'text', 'url', 'source', 'scope', 'restrictions', 'limitations',
                    'named_shape_verified', 'source_action_verified', 'full_span_semantic_verified')
            cards.append({key: card.get(key) for key in keys} | dict(
                canonical_reference_anchor_frames=card.get('reference_anchor_frames', []),
                saved_card_pointer=f'/cases/{ci}/stages/{j}/source_cards/{k}', context_only=True))
        output.append(dict(id=s['id'], canonical_stage_id=f'{word}:stage{j+1:02}',
            title=f'주요 구간 {j+1}', canonical_core_frame_ids=s['reference_core'],
            description=s['reference_rgb_description'], source_cards=cards,
            source_url=next((c['url'] for c in cards if c.get('url')), None),
            saved_stage_pointer=f'/cases/{ci}/stages/{j}',
            description_scope='Saved canonical reference observation; other-reference application is estimated context, not new truth',
            source_cards_are_hard_rules=False))
    return output


def project_stages(stages, path, canonical_ids, reference_ids, direct=False):
    """Path=(other_reference_index, canonical_index); preserve every occurrence."""
    path = np.asarray(path, dtype=np.int64); n, m = len(reference_ids), len(canonical_ids)
    if (path.ndim != 2 or path.shape[1] != 2 or not np.array_equal(path[0], [0, 0])
            or not np.array_equal(path[-1], [n-1, m-1]) or (np.diff(path, axis=0) < 0).any()
            or set(path[:,0]) != set(range(n)) or set(path[:,1]) != set(range(m))):
        raise ValueError('Stage projection requires complete monotonic axes')
    lookup = {int(f): i for i,f in enumerate(canonical_ids)}; out = []
    for stage in stages:
        lo, hi = stage['canonical_core_frame_ids']; ci = [lookup[int(lo)], lookup[int(hi)]]
        picks = path[(path[:,1] >= ci[0]) & (path[:,1] <= ci[1]), 0]
        if not len(picks):
            raise ValueError('Canonical stage cannot disappear')
        bounds = [int(picks.min()), int(picks.max())]
        out.append(dict(stage, frames=bounds,
            source_frame_ids=[int(reference_ids[bounds[0]]), int(reference_ids[bounds[1]])],
            canonical_core_indices=ci, projected_unique_reference_frames=int(len(np.unique(picks))),
            overlaps_previous_projected_stage=bool(out and bounds[0] <= out[-1]['frames'][1]),
            boundary_provenance='canonical01_saved_reviewed_span' if direct else 'inferred_from_whole_sequence_DTW_not_truth',
            stage_semantic_correctness=None, mandatory_repetition_verified=False))
    return out


def reference_rest_evidence(stages, calibration, observation):
    """Reference-only near-rest wrist evidence, never query-missing inference.

    The old anchors were created from HAND wrists33/54, not pose wrists15/16.
    Use exactly those sites before the ROI mask, with original observed/usable
    quality. Missing frames remain in the entire projected-stage denominator.
    This is a wrist-posture proxy, not proof of linguistic irrelevance or still
    fingers. No coordinate, mask, model posterior or interval is changed.
    """
    xy = np.asarray(observation['xy_px'], dtype=np.float64)
    observed = np.asarray(observation['observed']); usable = np.asarray(observation['usable'])
    ids = np.asarray(observation['frame_id']); ts = np.asarray(observation['timestamp_us'])
    n = len(ids)
    if (xy.shape != (n,75,2) or observed.shape != (n,75) or usable.shape != (n,75)
            or observed.dtype != bool or usable.dtype != bool or len(ts) != n
            or (np.diff(ids) <= 0).any() or (np.diff(ts) <= 0).any()):
        raise ValueError('Original full75 observation/time schema required')
    calibrated = (calibration.get('status') == 'CALIBRATED'
                  and calibration.get('reason') == 'STABLE_ARMS_DOWN_WINDOW')
    if calibrated:
        radius = calibration['parameters']['anchor_radius_normalized']
        if radius != .08:
            raise ValueError('Frozen rest anchor radius differs from .08')
        anchors = {side: np.asarray(calibration['wrist_anchors_normalized'][name], float)
                   for side,name in [('L','left'),('R','right')]}
        if any(v.shape != (2,) or not np.isfinite(v).all() for v in anchors.values()):
            raise ValueError('Invalid saved original hand-wrist anchor')
    else:
        radius = None; anchors = {}
    out = []
    for stage in stages:
        lo, hi = stage['frames']
        if not (0 <= lo <= hi < n):
            raise ValueError('Projected stage lies outside original native observations')
        ix = np.arange(lo,hi+1); possible = len(ix); sides = {}; confirmed = []
        for side, wrist in [('L',33),('R',54)]:
            valid = (observed[ix][:,[11,12,wrist]] & usable[ix][:,[11,12,wrist]]
                     & np.isfinite(xy[ix][:,[11,12,wrist]]).all(-1)).all(1)
            scale = np.linalg.norm(xy[ix,11]-xy[ix,12], axis=1)
            valid &= np.isfinite(scale) & (scale > 0)
            distance = np.full(possible, np.nan, float)
            if calibrated:
                center = (xy[ix,11]+xy[ix,12])*.5
                rel = (xy[ix[valid],wrist]-center[valid])/scale[valid,None]
                distance[valid] = np.linalg.norm(rel-anchors[side], axis=1)
            near = np.isfinite(distance) & (distance <= .08)
            count = int(near.sum()); ratio = count/possible
            is_rest = calibrated and ratio >= .8
            if is_rest:
                confirmed.append(side)
            sides[side] = dict(status='CONFIRMED_REFERENCE_REST_WRIST_POSTURE' if is_rest else
                'UNKNOWN_NO_STABLE_REST_CALIBRATION' if not calibrated else 'NOT_CONFIRMED',
                possible_stage_frames=possible, raw_usable_same_site_frames=int(valid.sum()),
                raw_unavailable_frames=int((~valid).sum()), comparable_anchor_frames=int(np.isfinite(distance).sum()),
                near_anchor_frames=count, fraction_of_whole_stage=ratio if calibrated else None,
                anchor_normalized=anchors[side].tolist() if calibrated else None,
                radius_normalized=radius, required_fraction_of_whole_stage=.8,
                native_wrist_landmark=wrist, pose_wrist_fallback_used=False,
                frame_ids=ids[ix].tolist(), timestamp_us=ts[ix].tolist(),
                near_anchor=near.tolist(), raw_usable=valid.tolist(),
                anchor_distance_normalized=[float(v) if np.isfinite(v) else None for v in distance])
        out.append(dict(stage, confirmed_rest_sides=confirmed,
            reference_rest_evidence=dict(sides=sides, calibration_status=calibration.get('status'),
                calibration_reason=calibration.get('reason'), full_stage_denominator=True,
                source='original reference full75 before ROI; same hand-wrist anchors and original quality masks',
                query_values_or_missing_used=False, source_observations_modified=False,
                linguistic_irrelevance_verified=False, finger_motion_or_whole_hand_rest_verified=False)))
    return out


def build():
    for name in ('BANK_BUILD_STARTED.json', 'BANK_PREDICTIONS.json', 'REFERENCE_BANK.json',
                 'REST_REFERENCE_AUDIT.json', 'bank_predictions', 'stage_mappings'):
        if (U/name).exists():
            raise FileExistsError('Preserve existing/partial bank: ' + name)
    frozen = read(U/'FROZEN.json'); training = read(U/'TRAINING_REPORT.json'); public = read(U/'PUBLIC_INPUTS.json')
    if training['status'] != 'COMPLETE_DEVELOPMENT_COMPARISON' or public['frozen_sha256'] != sha(U/'FROZEN.json'):
        raise ValueError('Completed fixed model choice and same public freeze required')
    chosen = training['adoption']['chosen_condition']
    if chosen not in ('expanded_nine_people', 'protected_baseline'):
        raise ValueError('Unknown fixed observer choice')
    rows = public['rows']
    if (len(rows) != 48 or len({r['clip_id'] for r in rows}) != 48
            or Counter((r['word_id'], r['person']) for r in rows) != {
                (w,p): 1 for w in frozen['word_ids'] for p in list(range(1,10))+[11,12,13]}):
        raise ValueError('Fixed48 source bank differs')
    pins = {str(p): sha(p) for p in [Path(__file__), U/'FROZEN.json', U/'TRAINING_REPORT.json',
           U/'PUBLIC_INPUTS.json', U/'engine.py', U/'adapter.py', U/'train_expansion.py',
           U/'TRAIN_PREPARED.json', U/'TRAIN_READY.json', U/'data_audit/SOURCES.json',
           O/'word100_stable_rest_roi_20261017/stable_rest.py', DESCRIPTIONS, CASE_INPUTS]}
    rest_sources = {r['clip_id']: r for r in read(U/'data_audit/SOURCES.json')['rows'] if r['person'] <= 9}
    for row in rows:
        if row['person'] <= 9:
            source = rest_sources[row['clip_id']]
            if source['native'] != row['observation'] or source['native_sha256'] != row['observation_sha256']:
                raise ValueError('Reference original observation source differs')
            pins[source['native']] = source['native_sha256']
            pins[source['record']['path']] = source['record']['sha256']
    pins.update(read(U/'TRAIN_PREPARED.json')['protected_pins'])
    if chosen == 'expanded_nine_people':
        directory = U/'trainruns/expanded_nine_people_42'
        pins.update({str(directory/name): sha(directory/name) for name in
                     ('COMPLETE.json', 'model.pt', 'norm.json', 'probs.npy')})
    pins.update({r['input']: r['input_sha256'] for r in rows})
    for path, h in pins.items():
        if sha(path) != h:
            raise ValueError('Changed frozen bank source: ' + path)
    engine = module('_live_bank_engine', U/'engine.py')
    adapter = module('_live_bank_adapter', U/'adapter.py')
    train = module('_live_bank_training', U/'train_expansion.py')
    base = BaselineSources(training, pins) if chosen == 'protected_baseline' else None
    save(U/'BANK_BUILD_STARTED.json', dict(observer=chosen, source_pins=pins, public_rows=48,
                                          scope='new bank observer data; completed probabilities reused'))
    start = time.perf_counter(); enriched = []; arrays = {}; prediction_rows = []
    folder = U/'bank_predictions'; folder.mkdir()
    for row in rows:
        a = load_input(row, public['feature_names'])
        if chosen == 'protected_baseline':
            p, route = base.predict(row, a, adapter, train)
        else:
            p = train.infer_candidate(a)
            route = dict(route='expanded_seed42_dense_new_centers_with_exact_DEV_center_reuse',
                         input_fingerprint=train.array_fingerprint(a),
                         no_prior_model_substitution=True, seed=42)
        if p.shape != (len(a['frame_id']), 7):
            raise ValueError('Posterior full axis changed')
        path = folder/f"{row['clip_id']}.npz"
        with path.open('xb') as f:
            np.savez_compressed(f, probabilities=p, posterior_available=np.isfinite(p),
                                frame_id=a['frame_id'], timestamp_us=a['timestamp_us'], targets=np.asarray(TARGETS))
        a['probabilities'] = p; arrays[row['clip_id']] = a
        result = dict(row, probability_file=str(path), probability_sha256=sha(path), observer=chosen,
                      input_fingerprint=train.array_fingerprint(a), prediction_route=route)
        enriched.append(result); prediction_rows.append(result)
        print(json.dumps(dict(stage='bank_prediction', clip_id=row['clip_id'], completed=len(enriched), route=route['route'])), flush=True)
    report = read(DESCRIPTIONS); cases = read(CASE_INPUTS)['rows']; refs = []
    mappings = U/'stage_mappings'; mappings.mkdir()
    for word, alias in zip(frozen['word_ids'], frozen['reference_aliases']):
        canonical = next(r for r in enriched if r['word_id'] == word and r['person'] == 1)
        if cases[alias]['clip_id'] != canonical['clip_id'] or cases[alias]['input_sha256'] != canonical['input_sha256']:
            raise ValueError('Canonical alias/input changed')
        ca = arrays[canonical['clip_id']]; stages = canonical_stages(report, alias, word)
        for row in [r for r in enriched if r['word_id'] == word and r['person'] <= 9]:
            ar = arrays[row['clip_id']]; direct = row['person'] == 1
            if direct:
                path = np.stack([np.arange(len(ca['frame_id']))]*2, 1)
            else:
                main = engine.core_mask(canonical, ca['timestamp_us']/1e6)
                path = engine.path_dtw(engine.matrix(engine.channels(ar), engine.channels(ca), main))
            projected = project_stages(stages, path, ca['frame_id'], ar['frame_id'], direct)
            rest_source = rest_sources[row['clip_id']]
            record = read(rest_source['record']['path'])
            with np.load(rest_source['native'], allow_pickle=False) as z:
                observation = {k:z[k].copy() for k in ('xy_px','observed','usable','frame_id','timestamp_us')}
            if (not np.array_equal(observation['frame_id'], ar['frame_id'])
                    or not np.array_equal(observation['timestamp_us'], ar['timestamp_us'])):
                raise ValueError('Reference rest/native input exact axes differ')
            projected = reference_rest_evidence(projected, record['calibration'], observation)
            for stage in projected:
                stage['reference_rest_evidence']['sources'] = dict(
                    native=dict(path=rest_source['native'], sha256=rest_source['native_sha256']),
                    calibration=dict(**rest_source['record'], json_pointer='/calibration'))
            mp = mappings/f"{row['clip_id']}.json"
            save(mp, dict(reference=row['clip_id'], canonical=canonical['clip_id'], path=path.tolist(),
                path_axes=['other_reference_native_index', 'canonical_native_index'],
                canonical_frame_id=ca['frame_id'].tolist(), reference_frame_id=ar['frame_id'].tolist(),
                canonical_timestamp_us=ca['timestamp_us'].tolist(), reference_timestamp_us=ar['timestamp_us'].tolist(),
                stages=projected, engine_sha256=pins[str(U/'engine.py')], input_sha256=row['input_sha256'],
                canonical_input_sha256=canonical['input_sha256'], query_stage_truth_used=False), compact=True)
            refs.append(dict(row, stages=projected, canonical_clip_id=canonical['clip_id'],
                             canonical_alias=alias, stage_mapping=str(mp), stage_mapping_sha256=sha(mp),
                             description_source=dict(path=str(DESCRIPTIONS), sha256=pins[str(DESCRIPTIONS)])))
    if len(refs) != 36 or sum(len(r['stages']) for r in refs) != 117:
        raise ValueError('Expected36references and117 projected stages')
    for path, h in pins.items():
        if sha(path) != h:
            raise ValueError('Source changed while building bank: ' + path)
    save(U/'BANK_PREDICTIONS.json', dict(status='COMPLETE', observer=chosen, targets=list(TARGETS), rows=prediction_rows,
        source_pins=pins, training_or_decoder_calls=0, seconds=time.perf_counter()-start,
        completed_old_inference_repeated=0, development_rows_excluded_from_reference_bank=12))
    rest_rows = [dict(clip_id=r['clip_id'], person=r['person'], word_id=r['word_id'],
                     canonical_stage_id=s['canonical_stage_id'], frames=s['frames'],
                     confirmed_rest_sides=s['confirmed_rest_sides'], evidence=s['reference_rest_evidence'])
                 for r in refs for s in r['stages']]
    save(U/'REST_REFERENCE_AUDIT.json', dict(status='REFERENCE_ONLY_POSTURE_EVIDENCE_SAVED',
        stages=117, side_stage_opportunities=234, rows=rest_rows,
        confirmed_side_stage_counts=dict(Counter(side for r in rest_rows for side in r['confirmed_rest_sides'])),
        source_pins=pins, anchor_site='hand wrists33/54; no pose15/16 substitution',
        radius_normalized=.08, whole_stage_fraction=.8, query_or_DEV_observation_used=False,
        limitation='Near the calibrated rest wrist posture does not prove still fingers or linguistic irrelevance'))
    save(U/'REFERENCE_BANK.json', dict(status='COMPLETE_PUBLIC_REFERENCE_BANK', observer=chosen, rows=refs,
        targets=list(TARGETS), feature_names=public['feature_names'], source_pins=pins,
        stages=117, people_per_expression=9, canonical_boundary_truth='Saved reference01 review only; all other bounds inferred',
        description_understanding_learned=False, educational_correctness_validated=False,
        reference_rest_audit_sha256=sha(U/'REST_REFERENCE_AUDIT.json'),
        probability_manifest_sha256=sha(U/'BANK_PREDICTIONS.json')))


def calibrate():
    for name in ('CALIBRATION_STARTED.json', 'NORMAL_RANGES.json', 'calibration_pairs'):
        if (U/name).exists():
            raise FileExistsError('Preserve existing/partial calibration: ' + name)
    bank = read(U/'REFERENCE_BANK.json'); frozen = read(U/'FROZEN.json'); rows = bank['rows']
    if len(rows) != 36 or {r['person'] for r in rows} != set(range(1,10)) or any(r['role'] != 'fit' for r in rows):
        raise ValueError('Calibration uses36 FIT sequences only')
    if (frozen['multi_reference']['coverage_min'] != .65 or frozen['multi_reference']['normal_support_min'] != 2):
        raise ValueError('Coverage/support contract changed')
    for path, h in bank['source_pins'].items():
        if sha(path) != h:
            raise ValueError('Frozen bank implementation/input changed before calibration: ' + path)
    engine = module('_live_calibration_engine', U/'engine.py')
    arrays = {}
    for r in rows:
        a = load_input(r, bank['feature_names'])
        a['probabilities'] = check_probability_axes(r['probability_file'], a, r['probability_sha256'])
        arrays[r['clip_id']] = a
    save(U/'CALIBRATION_STARTED.json', dict(reference_bank_sha256=sha(U/'REFERENCE_BANK.json'),
        engine_sha256=sha(U/'engine.py'), builder_sha256=sha(__file__), queries=36, candidates_per_query=8,
        pair_opportunities=288, selection='same two nearest complete reference sequences for every family/stage',
        minimum_coverage=.65, range_quantile=.95, development_used=False))
    dest = U/'calibration_pairs'; dest.mkdir()
    values = {f: [] for f in engine.FAMILIES}; ledger = []; pair_index = []; start = time.perf_counter()
    for query in rows:
        candidates = [r for r in rows if r['word_id'] == query['word_id'] and r['person'] != query['person']]
        if len(candidates) != 8 or query['person'] in {r['person'] for r in candidates}:
            raise ValueError('Leave-person-out identity exclusion failed')
        comparisons = []
        for reference in candidates:
            result = engine.compare_one(arrays[query['clip_id']], arrays[reference['clip_id']], reference)
            result['query_id'] = query['clip_id']; result['query_person'] = query['person']
            p = dest/f"{query['clip_id']}__{reference['clip_id']}.json"
            save(p, result, compact=True)
            pair_index.append(dict(query=query['clip_id'], reference=reference['clip_id'], path=str(p), sha256=sha(p)))
            comparisons.append(result)
        chosen = sorted(comparisons, key=lambda c: c['distance'])[:2]
        for rank, comparison in enumerate(chosen, 1):
            for si, stage in enumerate(comparison['stages']):
                for family in engine.FAMILIES:
                    v = stage['families'][family]
                    usable = (v.get('comparison_required', True) and v['coverage'] is not None and v['coverage'] >= .65
                              and v['distance'] is not None and v.get('peak_distance') is not None)
                    if usable:
                        values[family].append((float(v['distance']), float(v['peak_distance'])))
                    ledger.append(dict(query=query['clip_id'], person=query['person'], word_id=query['word_id'],
                        selected_rank=rank, reference=comparison['id'], canonical_stage_id=stage['canonical_stage_id'],
                        stage_index=si, family=family, usable_for_range=usable, distance=v['distance'],
                        comparison_required=v.get('comparison_required', True),
                        peak_distance=v.get('peak_distance'), coverage=v['coverage'], expected=v['expected'],
                        observed=v['observed'], query_expected=v['query_expected'], query_observed=v['query_observed'],
                        pair_json_pointer=f'/stages/{si}/families/{family}',
                        pair_path=str(dest/f"{query['clip_id']}__{comparison['id']}.json")))
        print(json.dumps(dict(stage='fit_calibration', query=query['clip_id'], completed=len(pair_index), total=288)), flush=True)
    if len(pair_index) != 288 or len(ledger) != 117*2*len(engine.FAMILIES):
        raise ValueError('Fixed calibration opportunity denominator changed')
    ranges = {}; counts = {}
    for family in engine.FAMILIES:
        vv = np.asarray(values[family], float).reshape(-1, 2)
        ranges[family] = dict(p90=float(np.quantile(vv[:,0], .95)) if len(vv) else None,
                              peak=float(np.quantile(vv[:,1], .95)) if len(vv) else None)
        opportunities = [v for v in ledger if v['family'] == family]
        counts[family] = dict(possible=len(opportunities), supported=len(vv),
                              not_required_reference_rest=sum(not v['comparison_required'] for v in opportunities),
                              insufficient=sum(v['comparison_required'] and not v['usable_for_range'] for v in opportunities),
                              expected_cells=sum(v['expected'] for v in opportunities),
                              observed_cells=sum(v['observed'] for v in opportunities),
                              query_expected_cells=sum(v['query_expected'] for v in opportunities),
                              query_observed_cells=sum(v['query_observed'] for v in opportunities))
    for path, h in bank['source_pins'].items():
        if sha(path) != h:
            raise ValueError('Source changed during calibration: ' + path)
    save(U/'NORMAL_RANGES.json', dict(status='COMPLETE_FIT_ONLY_DESCRIPTIVE_RANGES', ranges=ranges,
        counts=counts, query_clips=36, query_people=list(range(1,10)), references_exclude_query_person=True,
        complete_pair_opportunities=288, selected_pair_opportunities=72, stage_family_opportunities=len(ledger),
        pair_index=pair_index, selected_stage_family_ledger=ledger, seconds=time.perf_counter()-start,
        reference_bank_sha256=sha(U/'REFERENCE_BANK.json'), engine_sha256=sha(U/'engine.py'),
        builder_sha256=sha(__file__), observer=bank['observer'], development_used=False, user_data_used=False,
        selection='Two nearest whole reference sequences; no family/point-specific best-reference selection',
        quantiles=dict(within_stage_distance=.9, across_all_supported_fit_stage_distances=.95,
                       across_all_supported_fit_stage_peaks=.95),
        range_scope='Pooled across four expressions, one cutoff per family; empirical observed FIT range, not linguistic tolerance',
        per_reference_support_missing_is_not_zero=True, independent_sample_accuracy_claim=False))


def selftest():
    checks = []
    def check(name, good):
        if not good:
            raise AssertionError(name)
        checks.append(name)
    stages = [dict(id='a', canonical_stage_id='w:1', canonical_core_frame_ids=[12,13]),
              dict(id='b', canonical_stage_id='w:2', canonical_core_frame_ids=[15,15])]
    path = [[0,0],[1,1],[2,2],[2,3],[3,4],[4,5],[5,5]]
    projected = project_stages(stages, path, np.arange(10,16), np.arange(20,26))
    check('nonzero original frameIDs mapped to local indices', projected[0]['frames'] == [2,2]
          and projected[0]['source_frame_ids'] == [22,22])
    check('many query endpoints retained', projected[1]['frames'] == [4,5])
    check('all canonical occurrences retained', [p['canonical_stage_id'] for p in projected] == ['w:1','w:2'])
    check('projected ranges not promoted to truth', all(p['boundary_provenance'].endswith('not_truth') for p in projected))
    try:
        project_stages(stages, path[:-1], np.arange(10,16), np.arange(20,26))
    except ValueError:
        checks.append('missing final source axis rejected')
    else:
        raise AssertionError('Incomplete axis accepted')
    # Quantile includes only known values, with denominator retained separately.
    evidence = [dict(coverage=.7, distance=1., peak_distance=2.), dict(coverage=.2, distance=100., peak_distance=100.)]
    used = [r for r in evidence if r['coverage'] >= .65]
    check('unknown cannot become zero cutoff', len(evidence) == 2 and len(used) == 1
          and np.quantile([r['distance'] for r in used], .95) == 1)
    # Separate original HAND wrist sites from pose wrists; no ROI or query data.
    xy = np.full((5,75,2), np.nan)
    xy[:,11] = [-.5,0]; xy[:,12] = [.5,0]
    xy[:,33] = [0,1.5]; xy[:,54] = [1,1.5]
    obs = dict(xy_px=xy, observed=np.isfinite(xy).all(2), usable=np.isfinite(xy).all(2),
               frame_id=np.arange(100,105), timestamp_us=np.arange(5)*33333)
    cal = dict(status='CALIBRATED', reason='STABLE_ARMS_DOWN_WINDOW',
               parameters=dict(anchor_radius_normalized=.08),
               wrist_anchors_normalized=dict(left=[0,1.5], right=[0,1.5]))
    stage = [dict(id='test', frames=[0,4])]
    original = {k:v.copy() for k,v in obs.items()}
    rest = reference_rest_evidence(stage, cal, obs)[0]
    check('reference rest side independent from active opposite hand', rest['confirmed_rest_sides'] == ['L'])
    check('whole stage denominator exact', rest['reference_rest_evidence']['sides']['L']['possible_stage_frames'] == 5)
    check('rest audit does not modify original observations', all(np.array_equal(v,obs[k],equal_nan=True) for k,v in original.items()))
    obs['usable'][0,33] = False
    rest = reference_rest_evidence(stage, cal, obs)[0]
    check('four of five actual supported frames meets .8', rest['confirmed_rest_sides'] == ['L']
          and rest['reference_rest_evidence']['sides']['L']['fraction_of_whole_stage'] == .8)
    obs['usable'][1,33] = False
    rest = reference_rest_evidence(stage, cal, obs)[0]
    check('missing frames cannot shrink rest denominator', rest['confirmed_rest_sides'] == []
          and rest['reference_rest_evidence']['sides']['L']['fraction_of_whole_stage'] == .6)
    obs['usable'][:,33] = True; obs['xy_px'][:,33,0] = .08
    check('original .08 radius inclusive', reference_rest_evidence(stage,cal,obs)[0]['confirmed_rest_sides'] == ['L'])
    obs['xy_px'][:,33,0] = .080001
    check('outside original radius is not rest', reference_rest_evidence(stage,cal,obs)[0]['confirmed_rest_sides'] == [])
    obs['xy_px'][:,15] = [0,1.5]; obs['observed'][:,15] = True; obs['usable'][:,15] = True
    obs['observed'][:,33] = False
    check('pose wrist cannot replace missing original hand wrist', reference_rest_evidence(stage,cal,obs)[0]['confirmed_rest_sides'] == [])
    obs = original
    fallback = dict(cal, reason='COMMON_TRAIN_MEDIAN_BODY_RATIO')
    test = reference_rest_evidence(stage, fallback, obs)[0]
    check('body ratio fallback has unknown rest', test['confirmed_rest_sides'] == []
          and test['reference_rest_evidence']['sides']['L']['status'] == 'UNKNOWN_NO_STABLE_REST_CALIBRATION')
    test = reference_rest_evidence(stage, dict(cal,status='NOT_CALIBRATED'), obs)[0]
    check('uncalibrated remains unknown', test['confirmed_rest_sides'] == [])
    broken = {k:v.copy() for k,v in original.items()}; broken['usable'][:,11] = False
    check('missing shoulder support cannot confirm rest', reference_rest_evidence(stage,cal,broken)[0]['confirmed_rest_sides'] == [])
    try:
        reference_rest_evidence(stage,dict(cal,parameters=dict(anchor_radius_normalized=.09)), original)
    except ValueError:
        checks.append('changed anchor coefficient rejected')
    else:
        raise AssertionError('Changed rest coefficient accepted')
    return dict(status='PASS', checks=len(checks), results=checks, code_sha256=sha(__file__),
                real_alignment_or_model_calls=0, public_or_user_data_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'calibrate', 'selftest'])
    args = parser.parse_args()
    try:
        if args.action == 'build':
            build()
        elif args.action == 'calibrate':
            calibrate()
        else:
            print(json.dumps(selftest(), ensure_ascii=False, indent=2))
    except BaseException:
        if args.action != 'selftest':
            failure = U/('BANK_BUILD_FIRST_FAILURE.txt' if args.action == 'build' else 'CALIBRATION_FIRST_FAILURE.txt')
            if not failure.exists():
                failure.write_text(traceback.format_exc(), encoding='utf8')
        raise
