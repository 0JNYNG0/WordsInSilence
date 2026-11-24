"""Frozen local-event head on a fresh, already-extracted native timeline.

API: LocalEventRuntime().load() optionally loads the frozen CPU head (no forward).
analyze(base_feedback, analysis_path, job_contract, output_dir) adds one field,
saving LOCAL_EVENT_* files exclusively. It never decodes a video, invokes an
encoder, recomputes DTW, trains, or imports historical generic worker modules.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import re
import threading
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
AH = OPS / 'word100_local_event_learning_20261103'
MODEL = AH / 'runs/local_42/model.pt'
PINS = {
    'word100_local_event_learning_20261103/TRAIN_RESULT.json': '2936f12dbb1b8bca0d3f7eac68f2b4135821164b2acaaece460996cf81b357cb',
    'word100_local_event_learning_20261103/local_events.py': '4456fc1825718d4223a4e7131935be617585cb3c30ffafd654461f73b0ca3211',
    'word100_local_event_learning_20261103/runs/local_42/model.pt': '27aced2983a25de0a980d511e912c6a6bfaa3c9383ecf75fecf06bf316b5d9c6',
    'word100_learned_motion_order_20261102/NORMALIZATION.json': '6be4dba2180d299faeb967377d8583a41711731dc38cf442d6c6f74d03da68ae',
    'word100_learned_rgb_motion_20261102/RGB_NORMALIZATION.json': '2b9305bc6df3941d729e22dce2ed8b1d6840f99278a78a61827b4b2872d389cb',
    'word100_learned_motion_invariance_20261102/TRAINING_CONTRACT.json': 'cc032c675116ec3bba3fde3daa98cb15eb928d959752fc40584c938ff309f693',
    'word100_local_event_guide_adapter_20261103/adapter.py': '89ac39586b48aaba8da5f44afe7a4b5ec8a532051efaf7ac4ae5126b382b2f51',
    'word100_guide_feature_supervision_20261102/COMPARISON_CONTRACT.json': '4d7c83eb77b75521ba66067925ed5e266cf3251fd8162b66d7d159a63f98901a',
    'word100_learned_component_extent_20261102/source_bridge.py': '185e77138296f1e41c238ff4660fb3911da751887ee31208364e37f49c1d0f4a',
    'word100_predicate_supervision_20261102/CLAUSE_COMPONENT_LINKS.json': '33c8f48fc9fc20d05615c97ba1fef411823ac839edfa74330507cb2bd10ba04f',
    'word100_source_component_connector_20261102/DIRECT_TARGETS.json': '03a1fa077bc350672e92fbb4c586af618be9cf9d6f1a149e997374b2462b7315',
    'word100_process_contract_alignment_20261102/process_rgb_support.py': 'c58ec8595c8ed93a72f0a8eadb8f74197c0eb563651495ba15c40890ce7ccb05',
    'word100_learned_body_relation_20261102/demo_adapter.py': 'bff7184bb23c717cf48b0cd996cdba40b80591e15e1ca8a90cd481823cf4c018',
    'word100_learned_guide_feedback_20261102/live_adapter.py': 'f8419575a1fa153073571e16417c5e29565194216321035e76cbea4a1636bada',
    'word100_seated_web_framing_20261102/adapter.py': '913e58e8711f166c3e542e4d18fb97feb6301edfc0e6666510d0c95cc4a749df',
    'word100_signrep_feasibility_20261029/common.py': 'cf4d31ab77aeef6486e84d423c062d8662274c3d11d89dee5ec2606bfd9390d7',
    'word100_learned_motion_order_20261102/MOTION_GUIDE_SUPPORT.json': '44acb9b022898015b062f38fed987eedfd76ad772064d94966c3dd4d26e4b072',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def axis_sha(ids, ts):
    return hashlib.sha256(np.asarray(ids).tobytes() + np.asarray(ts).tobytes()).hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def isolated_module(name, path):
    # These modules have no relative/generic project imports. Never change
    # sys.path or reuse common/data/train/adapter names from the order worker.
    spec = importlib.util.spec_from_file_location('_ak_local_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_arrays(a, stats, rgb_stats, geometry_valid, support_module):
    """Frozen M.features/normalize + R.rgb_input arithmetic, no observer calls."""
    required = ['x', 'mask', 'retained_probability', 'relation_probability', 'rgb',
                'frame_id', 'timestamp_us', 'image_available']
    require(all(k in a for k in required), 'Missing native geometry/shape/relation/RGB evidence')
    ids, ts = a['frame_id'], a['timestamp_us']
    n = len(ids)
    require(n >= 16, 'Existing live encoder contract requires at least16 native rows')
    for v in [ids, ts]:
        require(v.shape == (n,) and v.dtype.kind in 'iu' and (v >= 0).all()
                and (np.diff(v.astype(np.int64)) > 0).all(), 'Native axes must be integer and strictly increasing')
    # Frame IDs may be nonconsecutive after invalid PTS removal. Never renumber.
    raw, mask, image = np.asarray(a['x']), np.asarray(a['mask']), np.asarray(a['image_available'])
    shape, relation, rgb = a['retained_probability'], a['relation_probability'], a['rgb']
    require(raw.shape == mask.shape == (n, 69) and mask.dtype.kind == 'b', 'Expected native69 geometry and boolean masks')
    require(image.shape == (n,) and image.dtype.kind == 'b', 'Explicit image availability must accompany native rows')
    require(shape.shape == (n, 7) and relation.shape == (n, 3) and rgb.shape == (n, 768), 'Auxiliary native dimensions differ')
    for v in [shape, relation]:
        finite = np.isfinite(v)
        require(((v[finite] >= 0) & (v[finite] <= 1)).all(), 'Auxiliary probabilities outside[0,1]')
    valid_keep = np.asarray(geometry_valid, dtype=np.float32)
    require(valid_keep.shape == (146,) and np.isin(valid_keep, [0, 1]).all()
            and np.array_equal(valid_keep[:73], valid_keep[73:]), 'Invalid frozen geometry feature mask')
    v = raw[:, 2:4] - raw[:, 4:6]
    delta = raw[:, 36:38] - raw[:, 4:6]
    den = (v * v).sum(1)
    good = mask[:, [2, 3, 4, 5, 36, 37]].all(1) & np.isfinite(den) & (den > .0025)
    along = (delta * v).sum(1) / np.maximum(den, 1e-6)
    side = (delta[:, 0] * v[:, 1] - delta[:, 1] * v[:, 0]) / np.maximum(den, 1e-6)
    z = np.column_stack([raw[:, :61], along, side, shape, relation])
    valid = np.column_stack([mask[:, :61] & np.isfinite(raw[:, :61]), good, good,
                             np.isfinite(shape), np.isfinite(relation)])
    valid &= image[:, None]
    z = np.where(valid, z, 0).astype(np.float32)
    valid &= np.isfinite(z)
    z = np.where(valid, z, 0)
    gm, gs = np.asarray(stats['mean']), np.asarray(stats['std'])
    rm, rs = np.asarray(rgb_stats['mean']), np.asarray(rgb_stats['std'])
    require(gm.shape == gs.shape == (73,) and rm.shape == rs.shape == (768,), 'Frozen normalization dimensions')
    require(all(np.isfinite(v).all() for v in [gm, gs, rm, rs]) and (gs > 0).all() and (rs > 0).all(), 'Invalid normalization')
    geo = np.concatenate([np.where(valid, (z - gm) / gs, 0), valid], 1).astype(np.float32)
    available = image & np.isfinite(rgb).all(1)
    video = np.concatenate([np.where(available[:, None], (rgb - rm) / rs, 0), available[:, None]], 1).astype(np.float32)
    x = np.concatenate([geo, video], 1)
    require(x.shape == (n, 915) and np.isfinite(x).all(), 'Prepared native915 must be finite')
    observed = available | valid[:, valid_keep[:73].astype(bool)].any(1)
    edge = np.diff(np.r_[False, observed, False].astype(np.int8))
    spans = [[int(s), int(e) - 1] for s, e in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]
    anchor = np.zeros(n, dtype=bool)
    for lo, hi in spans:
        if hi > lo:
            anchor[lo:hi + 1] = True
    support = support_module.build_rgb_support(ts, image, ids, rgb_finite=np.isfinite(rgb).all(1))
    return dict(x=x, ids=ids.copy(), ts=ts.copy(), mask=valid, available=available,
                image_available=image.copy(), evidence_available=observed, motion_anchor=anchor,
                observed_fragments=spans, geometry_valid=valid_keep, support=support,
                rgb_lineage='PINNED_LIVE_FULLFRAME_16_STRIDE2_NATIVE_TIME_INTERPOLATION',
                rgb_window_values_independently_reconstructed=False)


def composed_support(p, indices):
    lo, hi = indices
    nominal = list(range(lo, hi + 1)); tcn = set()
    for start, end in p['observed_fragments']:
        for i in range(max(lo, start), min(hi, end) + 1):
            if p['motion_anchor'][i]:
                tcn.update(range(max(start, i - 7), min(end, i + 7) + 1))
    tcn = sorted(tcn)
    rgb = sorted({j for i in tcn for j in p['support']['row_consumed_support'][i]})
    outside = sorted(set(rgb) - set(nominal)); ids, ts = p['ids'], p['ts']
    return dict(nominal_indices=indices, nominal_native_frames=ids[nominal].tolist(),
        nominal_timestamp_us=ts[nominal].tolist(), tcn_input_indices=tcn,
        tcn_native_frames=ids[tcn].tolist(), rgb_input_indices=rgb,
        rgb_native_frames=ids[rgb].tolist(), rgb_timestamp_us=ts[rgb].tolist(),
        rgb_outside_nominal_native_frames=ids[outside].tolist(),
        tcn_radius_rows=7, rgb_window_rows=16, rgb_stride_rows=2,
        scope='Structural maximum support only; not causal attribution, exact onset or nominal truth.')


def verify_rgb_windows(a, windows, p):
    """Join the producer's actual windows to native values, with no encoding."""
    required = ['features', 'starts', 'center_timestamp_us', 'prelinear_windows', 'frame_id', 'timestamp_us']
    require(all(k in windows for k in required), 'Actual RGB window provenance is incomplete')
    require('prelinear' in a, 'Native prelinear values missing from extraction')
    ids, ts = p['ids'], p['ts']; n = len(ids)
    require(np.array_equal(windows['frame_id'], ids) and np.array_equal(windows['timestamp_us'], ts), 'RGB window/native axes differ')
    expected = np.asarray([w['slot_rows'][0] for w in p['support']['windows']])
    starts = windows['starts']; centers = windows['center_timestamp_us']
    require(starts.dtype.kind in 'iu' and np.array_equal(starts, expected), 'Actual RGB windows differ from16/stride2 recipe')
    expected_centers = (ts[starts] + ts[starts + 15]) / 2
    require(np.array_equal(centers, expected_centers), 'RGB window centers differ from original PTS')
    for field, native in [('features', 'rgb'), ('prelinear_windows', 'prelinear')]:
        values = windows[field]
        require(values.shape == (len(starts), 768) and np.isfinite(values).all(), 'Invalid stored window features')
        require(a[native].shape == (n, 768), 'Native RGB/prelinear shape mismatch')
        restored = np.stack([np.interp(ts, centers, values[:, j]) for j in range(768)], 1).astype(np.float32)
        require(np.array_equal(restored, a[native]), 'Actual window/native interpolation mismatch: ' + field)
    return dict(status='ACTUAL_WINDOW_VALUES_AND_NATIVE_INTERPOLATION_EXACT', windows=len(starts),
                native_rows=n, features_and_prelinear_reconstructed=True, new_encoder_forwards=0,
                native_axis_sha256=axis_sha(ids, ts), structural_support_is_dense_truth=False)


class LocalEventRuntime:
    def __init__(self):
        self.verify_resources()
        self._adapter = isolated_module('additive', OPS / 'word100_local_event_guide_adapter_20261103/adapter.py')
        self._support = isolated_module('support', OPS / 'word100_process_contract_alignment_20261102/process_rgb_support.py')
        bridge = isolated_module('source_catalog', OPS / 'word100_learned_component_extent_20261102/source_bridge.py').SourceBridge()
        self.nominal = copy.deepcopy(bridge.nominal)
        constraints = read(OPS / 'word100_guide_feature_supervision_20261102/COMPARISON_CONTRACT.json')['constraints']
        selected = [c for c in constraints if c['clause_id'] == 'well_brush']
        require(len(selected) == 1, 'Ambiguous source association')
        self.source_constraint = copy.deepcopy(selected[0])
        c = self.source_constraint
        require(c['condition_relation'] == 'RELATED_ONLY' and c['entry_id'] == 'NIKL_legacy:9543'
                and c['component_id'] == 'right_hand_traverses_left_arm_toward_wrist', 'Changed source semantic contract')
        require(all(bridge.catalog['well_brush'][k] == c[k] for k in ['text', 'url', 'source']), 'Source catalog mismatch')
        require(sha(c['source']['path']) == c['source']['sha256'], 'Source citation content changed')
        self.motion_guides = read(OPS / 'word100_learned_motion_order_20261102/MOTION_GUIDE_SUPPORT.json')
        self.stats = read(OPS / 'word100_learned_motion_order_20261102/NORMALIZATION.json')
        self.rgb_stats = read(OPS / 'word100_learned_rgb_motion_20261102/RGB_NORMALIZATION.json')
        keep = read(OPS / 'word100_learned_motion_invariance_20261102/TRAINING_CONTRACT.json')['motion_value_indices']
        self.geometry_valid = np.zeros(146, dtype=np.float32)
        self.geometry_valid[keep] = 1; self.geometry_valid[np.asarray(keep) + 73] = 1
        self.model = None; self._torch = None; self.model_loads = 0; self.forwards = 0
        self._used_jobs = set(); self._lock = threading.Lock()

    @staticmethod
    def verify_resources():
        for rel, digest in PINS.items():
            require(sha(OPS / rel) == digest, 'Frozen runtime resource changed: ' + rel)

    def load(self):
        """Explicit warmup of one CPU head; no dummy/real sequence forward."""
        if self.model is None:
            self.verify_resources()
            import torch
            model_module = isolated_module('frozen_head', AH / 'local_events.py')
            model = model_module.LocalEventNet()
            model.load_state_dict(torch.load(MODEL, map_location='cpu', weights_only=True))
            require(torch.equal(model.geometry_valid, torch.tensor(self.geometry_valid)), 'Checkpoint geometry mask differs')
            model.eval(); model.requires_grad_(False)
            self.model = model; self._torch = torch; self.model_loads += 1
        return dict(model_path=str(MODEL), model_sha256=PINS['word100_local_event_learning_20261103/runs/local_42/model.pt'],
                    device='cpu', model_loads=self.model_loads, forwards=self.forwards, training=False)

    def prepare(self, arrays):
        return prepare_arrays(arrays, self.stats, self.rgb_stats, self.geometry_valid, self._support)

    def make_proposals(self, base, p):
        """Use fixed source/reference metadata plus the already-computed path."""
        require('local_event_feedback' not in base, 'Local event output already exists')
        require(base.get('path_axes') == ['query_native_index', 'learned_canonical_index'], 'Unexpected learned-path axes')
        path = np.asarray(base['path']); ids, ts = p['ids'], p['ts']
        require(path.ndim == 2 and path.shape[1] == 2 and len(path) and path.dtype.kind in 'iu'
                and (path >= 0).all() and path[:, 0].max() < len(ids) and (np.diff(path, axis=0) >= 0).all(), 'Invalid existing native path')
        stages = {s['guide_id']: s for s in base['stages']}
        require(len(stages) == len(base['stages']), 'Duplicate stage identity')
        expected = {m['target']: m for m in self.motion_guides if m['word_id'] == base['word_id']}
        motions = base.get('motion_observations', [])
        require(expected and len(motions) == len(expected) and {m['target'] for m in motions} == set(expected), 'Unsupported or incomplete reference-motion contract')
        proposals = []
        for old in motions:
            j = old['target']; fixed = expected[j]
            require(type(j) is int and old['guide_id'] == fixed['guide_id'] and old['canonical_frames'] == fixed['canonical_frames'], 'Motion identity differs from frozen catalog')
            require(old['guide_id'] in stages, 'Missing guide stage')
            stage = stages[old['guide_id']]; legacy = old['frame_ids']
            legacy_rows = np.flatnonzero(np.isin(ids, legacy))
            require(len(legacy_rows) and ids[legacy_rows].tolist() == legacy and np.array_equal(legacy_rows, np.arange(legacy_rows[0], legacy_rows[-1] + 1)), 'Legacy proposal is not retained native rows')
            clo, chi = old['canonical_frames']
            old_hit = path[(path[:, 1] >= clo) & (path[:, 1] <= chi), 0]
            require(len(old_hit) and [int(old_hit.min()), int(old_hit.max())] == [int(legacy_rows[0]), int(legacy_rows[-1])], 'Legacy proposal/path binding mismatch')
            canonical = None; source = None
            if j == 0:
                entry = self.nominal[(base['word_id'], 'right_hand_traverses_left_arm_toward_wrist')]
                require('well_brush' in entry['clauses'], 'Source nominal lacks cited clause')
                canonical = [min(entry['frames']), max(entry['frames'])]
                hit = path[(path[:, 1] >= canonical[0]) & (path[:, 1] <= canonical[1]), 0]
                require(len(hit), 'Source proposal has no native match')
                lo, hi = int(hit.min()), int(hit.max())
                source = copy.deepcopy(self.source_constraint)
                family = 'existing_source_component_nominal_projected_by_existing_path'
            else:
                lo, hi = int(legacy_rows[0]), int(legacy_rows[-1])
                family = 'existing_legacy_motion_native_proposal'
            proposals.append(dict(target_index=j, guide_id=old['guide_id'], scope_family=family,
                canonical_extent=canonical, indices=[lo, hi], native_frame_ids=ids[lo:hi + 1].tolist(),
                timestamp_us=ts[lo:hi + 1].tolist(), source_constraint=source,
                source_component_id=source['component_id'] if source else None,
                source_binding_relation='RELATED_ONLY' if source else 'REFERENCE_OBSERVATION_ONLY',
                other_scope_metadata=dict(legacy_motion_frame_ids=copy.deepcopy(legacy),
                    guide_stage_frame_ids=copy.deepcopy(stage['frame_ids']),
                    canonical_legacy_motion_frames=copy.deepcopy(old['canonical_frames']), not_alternative_score_candidates=True),
                selection='Fixed from existing metadata/path before the fresh local head forward.',
                event_boundary_truth=None, source_variant_applicability_verified=None))
        return proposals

    @staticmethod
    def validate_job(base, analysis_path, job, output_dir):
        require(isinstance(job, dict) and job.get('source_role') == 'public_runtime_inference', 'Only authorized public runtime inference jobs in this unit')
        require(all(job.get(k) is False for k in ['training', 'tuning', 'candidate_selection']), 'Inference-only privacy contract required')
        require(isinstance(job.get('job_id'), str) and re.fullmatch('[A-Za-z0-9_.-]+', job['job_id']), 'Invalid job identity')
        for k in ['source_sha256', 'input_sha256', 'native_axis_sha256', 'base_value_sha256', 'rgb_support_sha256']:
            require(isinstance(job.get(k), str) and re.fullmatch('[0-9a-f]{64}', job[k]), 'Missing job hash: ' + k)
        require(job.get('word_id') == base.get('word_id') and job['source_sha256'] == base.get('source_sha256'), 'Job and existing source feedback differ')
        require(job['base_value_sha256'] == value_sha(base), 'Hash must identify the actual order-attached base passed to local runtime')
        require(Path(job['input']).resolve() == analysis_path and sha(analysis_path) == job['input_sha256'], 'Job input path/hash mismatch')
        require(output_dir == analysis_path.parent and output_dir.is_dir(), 'Additive outputs must stay beside their job analysis')
        support_path = Path(job['rgb_support_path']).resolve()
        require(support_path == output_dir / 'RGB_NATIVE_SUPPORT.npz' and sha(support_path) == job['rgb_support_sha256'], 'Job RGB support path/hash mismatch')
        return job['job_id']

    def analyze(self, base_feedback, analysis_path, job_contract, output_dir):
        """One fresh input forward, exclusive durable marker, then additive output."""
        with self._lock:
            path = Path(analysis_path).resolve(); directory = Path(output_dir).resolve()
            base = copy.deepcopy(base_feedback); job = copy.deepcopy(job_contract)
            jid = self.validate_job(base, path, job, directory)
            names = ['INPUT_CONTRACT.json', 'STARTED.json', 'PREDICTIONS.npz', 'FEEDBACK.json', 'FINISHED.json', 'FIRST_FAILURE.txt']
            require(jid not in self._used_jobs and not any((directory / ('LOCAL_EVENT_' + n)).exists() for n in names), 'Job already attempted; inspect first result instead of repeating')
            self.verify_resources()
            with np.load(path, allow_pickle=False) as f:
                arrays = {k: f[k].copy() for k in f.files}
            p = self.prepare(arrays)
            with np.load(job['rgb_support_path'], allow_pickle=False) as f:
                windows = {k: f[k].copy() for k in f.files}
            rgb_validation = verify_rgb_windows(arrays, windows, p)
            p['rgb_window_values_independently_reconstructed'] = True
            require(axis_sha(p['ids'], p['ts']) == job['native_axis_sha256'], 'Native frame/PTS identity differs from job')
            p['input_identity'] = dict(clip_id=jid, word_id=job['word_id'], source_sha256=job['source_sha256'], input_sha256=job['input_sha256'])
            proposals = self.make_proposals(base, p)
            support = {q['target_index']: composed_support(p, q['indices']) for q in proposals}
            preflight = dict(status='FROZEN_BEFORE_LOCAL_FORWARD', job=job, proposals=proposals,
                proposals_value_sha256=value_sha(proposals), base_value_sha256=value_sha(base),
                native_frame_ids=p['ids'].tolist(), timestamp_us=p['ts'].tolist(),
                evidence_available=p['evidence_available'].tolist(), motion_anchor=p['motion_anchor'].tolist(),
                image_available=p['image_available'].tolist(), rgb_available=p['available'].tolist(),
                geometry_valid=self.geometry_valid.tolist(), structural_support=support,
                dependency_pins={str(OPS / k): v for k, v in PINS.items()}, implementation_sha256=sha(__file__),
                input_dim=915, source_video_read=False, new_DTW=0, new_training=0,
                rgb_window_validation=rgb_validation,
                RGB_support_scope='Actual saved live16/stride2 window values exactly reinterpolated on the native axis')
            write(directory / 'LOCAL_EVENT_INPUT_CONTRACT.json', preflight)
            self._used_jobs.add(jid)
            started = time.perf_counter(); before_loads = self.model_loads
            try:
                write(directory / 'LOCAL_EVENT_STARTED.json', dict(job_id=jid, source_sha256=job['source_sha256'], at=time.time(), planned_forwards=1))
                self.load(); torch = self._torch
                forward_start = time.perf_counter()
                with torch.inference_mode():
                    prediction = self.model(torch.from_numpy(p['x'])[None], torch.tensor([len(p['ids'])]))
                head_seconds = time.perf_counter() - forward_start
                self.forwards += 1
                pred = dict(logits=prediction['logits'][0].numpy(), probabilities=prediction['logits'][0].sigmoid().numpy(),
                    motion_anchor=prediction['motion_anchor'][0].numpy(), evidence_available=prediction['evidence_available'][0].numpy(),
                    rgb_available=prediction['rgb_available'][0].numpy(), frame_id=p['ids'], timestamp_us=p['ts'])
                for k, source in [('motion_anchor', 'motion_anchor'), ('evidence_available', 'evidence_available'), ('rgb_available', 'available')]:
                    require(np.array_equal(pred[k], p[source]), 'Frozen model/input observation-mask mismatch: ' + k)
                destination = directory / 'LOCAL_EVENT_PREDICTIONS.npz'
                with destination.open('xb') as f:
                    np.savez_compressed(f, **pred)
                envelope = dict(**p['input_identity'], AH_prediction_sha256=sha(destination),
                    AH_prediction=str(destination), AH_model_path=str(MODEL), AH_model_sha256=PINS['word100_local_event_learning_20261103/runs/local_42/model.pt'],
                    base_value_sha256=value_sha(base), proposals_value_sha256=value_sha(proposals),
                    native_frame_ids=p['ids'].tolist(), timestamp_us=p['ts'].tolist(),
                    source_hash_verification='Validated producer job/source feedback identity; this module does not reread video.')
                out = self._adapter.attach(base, envelope, p, pred, proposals, support)
                addition = out['local_event_feedback']
                addition.update(new_model_loads=self.model_loads - before_loads, new_model_forwards=1,
                    attachment_phase_model_forwards=0, prediction_path=str(destination),
                    fresh_job_contract=job, native_axis_sha256=axis_sha(p['ids'], p['ts']),
                    runtime_device='cpu', head_forward_seconds=head_seconds,
                    local_runtime_seconds=time.perf_counter() - started,
                    timing_is_end_to_end_latency=False, new_encoder_forwards=0, source_video_reads=0,
                    rgb_window_validation=rgb_validation)
                require(value_sha({k: v for k, v in out.items() if k != 'local_event_feedback'}) == value_sha(base), 'Existing output changed')
                require(sha(path) == job['input_sha256'], 'Native input changed during inference')
                require(sha(job['rgb_support_path']) == job['rgb_support_sha256'], 'RGB window input changed during inference')
                self.verify_resources()
                write(directory / 'LOCAL_EVENT_FEEDBACK.json', out)
                write(directory / 'LOCAL_EVENT_FINISHED.json', dict(status='COMPLETE', job_id=jid,
                    result_sha256=sha(directory / 'LOCAL_EVENT_FEEDBACK.json'), prediction_sha256=sha(destination),
                    model_forwards=1, new_DTW=0, new_training=0, at=time.time()))
                return out
            except BaseException:
                failure = directory / 'LOCAL_EVENT_FIRST_FAILURE.txt'
                if not failure.exists():
                    with failure.open('x', encoding='utf-8') as f:
                        f.write(traceback.format_exc())
                raise
