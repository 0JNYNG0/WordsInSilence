"""Additional pose correspondence evidence; never a new sign-correctness score.

Import is inert. The baseline engine/results remain unchanged. Only a copy of an
existing SIMILAR status can be downgraded when extra query opportunities lack
support. The auxiliary arm-only path is an estimate, not a phase annotation.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

U = Path(__file__).resolve().parent
EXPECTED_ENGINE_SHA = '68a94384350353db6c73025cba7c99ee406be593186c198f371d41b26fd75c11'
THRESHOLD = .65
SIMILAR = 'SIMILAR_REFERENCE_RANGE'
INSUFFICIENT = 'INSUFFICIENT_EVIDENCE'
GUARD_MESSAGE = ('손·팔 위치로 따로 추정한 대응 시간도 함께 확인하면 관측 근거가 부족합니다. '
                 '두 경로 모두 추정이며, 이 보류는 동작 오류를 뜻하지 않습니다.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _base():
    p = U / 'engine.py'
    if sha(p) != EXPECTED_ENGINE_SHA:
        raise ValueError('Frozen engine code differs; guard contract requires new review')
    spec = importlib.util.spec_from_file_location('_guard_frozen_engine', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _array_hash(h, value):
    a = np.ascontiguousarray(value)
    h.update(str(a.shape).encode())
    h.update(a.dtype.str.encode())
    h.update(a.tobytes())


def _arm_indices(a):
    return [a['feature_names'].index(s + '_' + part + '_' + d)
            for s in ['L', 'R'] for part in ['pose_wrist', 'elbow'] for d in ['x', 'y']]


def arm_key(a, r):
    """Only exact raw pose arrays/masks/native axes, never labels/posteriors."""
    h = hashlib.sha256(b'arm-path-guard-v1' + EXPECTED_ENGINE_SHA.encode() +
                       sha(U / 'GUARD_FROZEN.json').encode())
    for seq in [a, r]:
        cols = _arm_indices(seq)
        for value in [seq['x'][:, cols], seq['mask'][:, cols], seq['frame_id'], seq['timestamp_us']]:
            _array_hash(h, value)
    return h.hexdigest()


def validate_path(path, n, m):
    a = np.asarray(path, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 1:
        raise ValueError('Invalid saved path shape')
    if not np.array_equal(a[0], [0, 0]) or not np.array_equal(a[-1], [n - 1, m - 1]):
        raise ValueError('Path must preserve both complete source axes')
    steps = np.diff(a, axis=0)
    if not np.all((steps >= 0) & (steps <= 1)) or np.any(steps.sum(1) == 0):
        raise ValueError('Invalid monotone full path')
    return a


class ArmPathCache:
    """Run-local cache. Optional persistence is only used by the public replay."""
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else None
        self.records = {}
        self.requests = []
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)

    def get(self, a, r, *, query_id=None, reference_id=None):
        key = arm_key(a, r)
        hit = key in self.records
        if not hit:
            start = time.perf_counter()
            qx, qm = _base().channels(a)['arm_path']
            rx, rm = _base().channels(r)['arm_path']
            valid = qm[:, None, :] & rm[None, :, :]
            count = valid.sum(2)
            cost = np.sqrt((((qx[:, None, :] - rx[None, :, :]) ** 2) * valid).sum(2) /
                           np.maximum(count, 1))
            cost[count == 0] = 1.
            cost = np.minimum(cost, 3.)
            path = _base().path_dtw(cost)
            validate_path(path, len(qx), len(rx))
            value = dict(key=key, path=path.tolist(), query_frames=len(qx), reference_frames=len(rx),
                         seconds=time.perf_counter() - start,
                         source='8 pose wrist/elbow XY coordinates only; frozen baseline DP',
                         phase_correctness_verified=False, semantic_path_truth=None,
                         query_frame_ids=np.asarray(a['frame_id']).tolist(),
                         reference_frame_ids=np.asarray(r['frame_id']).tolist(),
                         query_timestamp_us=np.asarray(a['timestamp_us']).tolist(),
                         reference_timestamp_us=np.asarray(r['timestamp_us']).tolist())
            if self.directory is not None:
                target = self.directory / (key + '.json')
                target.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
                value['saved'] = dict(path=str(target), sha256=sha(target))
            self.records[key] = value
        self.requests.append(dict(key=key, cache_hit=hit, query_id=query_id, reference_id=reference_id))
        return self.records[key]


def _stage_links(path, bounds):
    p = np.asarray(path, dtype=np.int64)
    return p[(p[:, 1] >= bounds[0]) & (p[:, 1] <= bounds[1])]


def _support(qm, rm, links, bounds, shape=False):
    """Each unique query-frame/attribute once, complete reference axis retained."""
    qm, rm = np.asarray(qm, bool), np.asarray(rm, bool)
    links = np.asarray(links, np.int64)
    qq = np.unique(links[:, 0])
    rr = np.arange(bounds[0], bounds[1] + 1)
    dim_re = rm[rr].sum(0)
    dim_ro = np.zeros(rm.shape[1], np.int64)
    dim_qe = np.zeros(rm.shape[1], np.int64)
    dim_qo = np.zeros(rm.shape[1], np.int64)
    query_need = []
    query_good = []
    for j in rr:
        q = links[links[:, 1] == j, 0]
        dim_ro += qm[q].any(0) & rm[j]
    for i in qq:
        need = rm[links[links[:, 0] == i, 1]].any(0)
        good = need & qm[i]
        dim_qe += need
        dim_qo += good
        query_need.append(need.tolist())
        query_good.append(good.tolist())
    re, ro, qe, qo = map(int, [dim_re.sum(), dim_ro.sum(), dim_qe.sum(), dim_qo.sum()])
    rc = ro / re if re else None
    qc = qo / qe if qe else None
    cov = min(rc, qc) if rc is not None and qc is not None else None
    dc = np.minimum(dim_ro / np.maximum(dim_re, 1), dim_qo / np.maximum(dim_qe, 1))
    if shape and re and qe:
        cov = float(dc.min())
    return dict(reference_expected=re, reference_observed=ro, query_expected=qe, query_observed=qo,
                reference_coverage=rc, query_coverage=qc, coverage=cov,
                dimension_reference_expected=dim_re.tolist(), dimension_reference_observed=dim_ro.tolist(),
                dimension_query_expected=dim_qe.tolist(), dimension_query_observed=dim_qo.tolist(),
                dimension_coverage=dc.tolist(), query_indices=qq.tolist(),
                query_expected_mask=query_need, query_observed_mask=query_good,
                possible_query_cells=int(len(qq) * rm.shape[1]),
                possible_reference_cells=int(len(rr) * rm.shape[1]),
                expected_from_reference_only=True)


def _eligible(value):
    return value['coverage'] is not None and value['coverage'] >= THRESHOLD


def pair_audit(a, r, row, baseline, cache):
    base = _base()
    qch, rch = base.channels(a), base.channels(r)
    primary = validate_path(baseline['path'], len(a['x']), len(r['x']))
    aux = cache.get(a, r, query_id=baseline.get('query_id'), reference_id=row['clip_id'])
    apath = np.asarray(aux['path'], np.int64)
    scope, rules = base.reference_scope(row, rch)
    stages = []
    for i, stage in enumerate(row['stages']):
        bounds = stage['frames']
        first = _stage_links(primary, bounds)
        second = _stage_links(apath, bounds)
        union = np.unique(np.vstack([first, second]), axis=0)
        # Auxiliary-only pose support cannot be rescued by the primary path.
        arm = _support(qch['arm_path'][1], rch['arm_path'][1] & scope['arm_path'], second, bounds)
        fams = {}
        for family in base.FAMILIES:
            qm, rm = qch[family][1], rch[family][1]
            scoped = rm & scope[family]
            shape = family.endswith('_shape')
            support = {name: _support(qm, scoped, p, bounds, shape)
                       for name, p in [('original', first), ('auxiliary', second), ('union', union)]}
            required = bool(scope[family][bounds[0]:bounds[1] + 1].any())
            reasons = []
            if required and not _eligible(support['union']):
                reasons.append('UNION_QUERY_OR_REFERENCE_FEATURE_SUPPORT_BELOW_0_65')
            if required and not _eligible(arm):
                reasons.append('AUXILIARY_POSE_CORRESPONDENCE_SUPPORT_BELOW_0_65')
            fams[family] = dict(support, comparison_required=required,
                                sufficient=required and not reasons, reasons=reasons,
                                reference_scope_excluded_expected=int(rm[bounds[0]:bounds[1] + 1].sum() - scoped[bounds[0]:bounds[1] + 1].sum()))
        stages.append(dict(stage_index=i, id=stage['id'], canonical_stage_id=stage.get('canonical_stage_id'),
                           reference_bounds=list(bounds), original_links=first.tolist(),
                           auxiliary_links=second.tolist(), union_links=union.tolist(),
                           query_frame_ids=np.asarray(a['frame_id'])[np.unique(union[:, 0])].tolist(),
                           query_timestamp_us=np.asarray(a['timestamp_us'])[np.unique(union[:, 0])].tolist(),
                           auxiliary_arm_support=arm, families=fams,
                           phase_correctness_verified=False))
    return dict(reference_id=row['clip_id'], arm_path_key=aux['key'],
                arm_path_source=aux.get('saved'), auxiliary_path=aux['path'],
                reference_scope_rules=rules, stages=stages)


def _change_stage(stage, feature_audits):
    original_status = stage.get('status')
    changes = []
    for feature in stage.get('features', []):
        audit = feature_audits[feature['name']]
        old = feature['status']
        if old == SIMILAR and not audit['sufficient']:
            feature['status'] = INSUFFICIENT
            feature['guard_previous_status'] = old
            feature['guard_message'] = GUARD_MESSAGE
            changes.append(feature['name'])
        # Original coverage/peak/distance fields are not rewritten.
        feature['guard_support'] = audit
    if original_status == SIMILAR and changes:
        stage['status'] = INSUFFICIENT
        stage['guard_previous_status'] = original_status
        stage['guard_previous_message'] = stage.get('message')
        stage['message'] = GUARD_MESSAGE
    stage['guard_changed_features'] = changes
    stage['guard_correspondence_verified'] = False
    return dict(before=original_status, after=stage.get('status'), changed_features=changes)


def guard_result(a, references_rows, reference_arrays, base_result, *, cache=None):
    """Return a new result with guard_audit; never mutate any caller input.

    ``references_rows`` accepts a list or id->row mapping. ``reference_arrays``
    is id->69-feature array dict. The base result is engine.evaluate_arrays's
    output. A sparse result without individual feature statuses is also allowed
    for saved development case replay; its raw reference families stay untouched.
    """
    out = deepcopy(base_result)
    if not out.get('references'):
        out['guard_audit'] = dict(status='NO_REFERENCE_COMPARISON', new_inference=0)
        return out
    cache = cache if cache is not None else ArmPathCache()
    rows = references_rows if isinstance(references_rows, dict) else {r['clip_id']: r for r in references_rows}
    selected = list(base_result['selected_whole_references'])
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError('Guard requires exactly the two preselected whole references')
    original_ids = [c['id'] for c in base_result['references']]
    if len(original_ids) != len(set(original_ids)) or any(r not in original_ids for r in selected):
        raise ValueError('Invalid base reference identities')
    audits = {}
    for comparison in base_result['references']:
        rid = comparison['id']
        audits[rid] = pair_audit(a, reference_arrays[rid], rows[rid], comparison, cache)
    base = _base()
    aggregate_changes = []
    for si, stage in enumerate(out['stages']):
        feature_audits = {}
        for family in base.FAMILIES:
            records = [audits[rid]['stages'][si]['families'][family] for rid in selected]
            expected_id = stage.get('canonical_stage_id')
            if any(audits[rid]['stages'][si]['canonical_stage_id'] != expected_id for rid in selected):
                raise ValueError('Selected whole-reference stages differ')
            feature_audits[base.LABELS[family]] = dict(
                reference_ids=selected, sufficient=all(v['sufficient'] for v in records),
                union_coverage=[v['union']['coverage'] for v in records],
                auxiliary_pose_coverage=[audits[rid]['stages'][si]['auxiliary_arm_support']['coverage'] for rid in selected],
                reasons=[v['reasons'] for v in records], threshold=THRESHOLD,
                unchanged_original_coverage=True, phase_correctness_verified=False)
        aggregate_changes.append(_change_stage(stage, feature_audits))
    individual_changes = []
    for comparison in out['references']:
        rid = comparison['id']
        for si, stage in enumerate(comparison['stages']):
            if 'features' not in stage:
                continue
            feature_audits = {}
            for family in base.FAMILIES:
                record = audits[rid]['stages'][si]['families'][family]
                feature_audits[base.LABELS[family]] = dict(
                    reference_ids=[rid], sufficient=record['sufficient'],
                    union_coverage=[record['union']['coverage']],
                    auxiliary_pose_coverage=[audits[rid]['stages'][si]['auxiliary_arm_support']['coverage']],
                    reasons=[record['reasons']], threshold=THRESHOLD,
                    unchanged_original_coverage=True, phase_correctness_verified=False)
            individual_changes.append(dict(reference_id=rid, stage_index=si,
                                           **_change_stage(stage, feature_audits)))
    out['guard_audit'] = dict(status='APPLIED_CONSERVATIVE_SUPPORT_GUARD',
        candidate='pose-arm alternate-path union support', selected_whole_references=selected,
        reference_selection_unchanged=True, original_paths_unchanged=True,
        original_feature_values_and_metrics_unchanged=True, inferred_pose_path_is_not_truth=True,
        correction_advice=None, linguistic_correctness=None, new_inference=0, threshold=THRESHOLD,
        pairs=audits, aggregate_changes=aggregate_changes, individual_changes=individual_changes,
        frozen_contract_sha256=sha(U / 'GUARD_FROZEN.json'), engine_sha256=EXPECTED_ENGINE_SHA)
    return out
