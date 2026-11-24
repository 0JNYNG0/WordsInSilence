"""Copy-only explanation of stored differences on an existing correspondence.

No alignment, inference, classification or threshold choice. Instantaneous raw
residuals above an existing quantile *number* are not a new p90 statistic. A one
sample episode has zero first-to-last PTS span, but contains one observation.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

U = Path(__file__).resolve().parent
OLD = U.parent / 'word100_multipeople_live_feedback_20261101'
ENGINE_SHA = '68a94384350353db6c73025cba7c99ee406be593186c198f371d41b26fd75c11'


@lru_cache(maxsize=1)
def _engine():
    p = OLD / 'engine.py'
    if hashlib.sha256(p.read_bytes()).hexdigest() != ENGINE_SHA:
        raise ValueError('Protected engine changed')
    spec = importlib.util.spec_from_file_location('_detail_frozen_channels', p)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def _nullable(a):
    return [float(x) if np.isfinite(x) else None for x in a]


def _axis(a, index):
    result = dict(index=int(index), frame_id=int(a['frame_id'][index]),
                  timestamp_us=int(a['timestamp_us'][index]))
    for key in ['original_source_index', 'original_source_frame_id', 'original_source_timestamp_us']:
        if key in a:
            result[key] = int(a[key][index])
    return result


def _interpretation(family):
    if family.endswith('_shape'):
        return dict(kind='LEARNED_ATTRIBUTE_PROBABILITY_DIFFERENCE',
                    text='학습한 제한 항목의 확률 차이입니다. 전체 손모양이나 교육적 정오의 확인값이 아닙니다.',
                    measurement_unit='probability', whole_handshape_verified=False)
    if family == 'hand_axis':
        return dict(kind='PROJECTED_2D_HAND_AXIS_DIFFERENCE',
                    text='저장된 손목에서 중지 뿌리로 향하는 2D 투영 축의 차이입니다. 실제 3D 회전이나 손목 꺾임으로 확정할 수 없습니다.',
                    measurement_unit='unit_image_axis_component', actual_rotation_interpretation='UNRESOLVED',
                    source_point_quality_verified=False, palm_surface_normal=False)
    if family == 'arm_path':
        return dict(kind='BODY_NORMALIZED_POSE_LOCATION_DIFFERENCE',
                    text='어깨 중심·너비를 기준으로 한 손목과 팔꿈치의 2D 위치 차이입니다. 그 차이의 언어적 중요성은 확인되지 않았습니다.',
                    measurement_unit='shoulder_width', linguistic_importance=None)
    if family.endswith('_angles'):
        return dict(kind='PROJECTED_FINGER_ANGLE_DIFFERENCE',
                    text='개별 손가락의 2D 투영 각도 차이입니다. 실제 3D 관절각이나 교육적 오류로 확정하지 않습니다.',
                    measurement_unit='angle_divided_by_pi', actual_3d_joint_angle=False)
    return dict(kind='BODY_NORMALIZED_HAND_RELATION_DIFFERENCE',
                text='저장된 양손의 2D 상대 위치 차이입니다. 정확한 접촉이나 언어적 오류를 확정하지 않습니다.',
                measurement_unit='shoulder_width', exact_contact_verified=False)


def _episodes(indices, frame_ids):
    result = []
    for i in indices:
        i = int(i)
        if result and i == result[-1][-1] + 1 and frame_ids[i] == frame_ids[result[-1][-1]] + 1:
            result[-1].append(i)
        else:
            result.append([i])
    return result


def trace_family(a, r, qchannel, rchannel, scope, path, bounds, cutoffs, family, original):
    qx, qm = qchannel
    rx, rm0 = rchannel
    rm = rm0 & scope
    pairs = np.asarray(path, dtype=np.int64)
    pairs = pairs[(pairs[:, 1] >= bounds[0]) & (pairs[:, 1] <= bounds[1])]
    qq = np.unique(pairs[:, 0])
    dim = qx.shape[1]
    data = {}
    for qi in qq:
        ri = np.unique(pairs[pairs[:, 0] == qi, 1])
        need = rm[ri].any(0)
        valid = rm[ri] & qm[qi]
        delta = np.abs(rx[ri] - qx[qi])
        maxima = np.full(dim, np.nan)
        for d in range(dim):
            if valid[:, d].any():
                maxima[d] = np.max(delta[valid[:, d], d])
        peak = float(np.nanmax(maxima)) if np.isfinite(maxima).any() else None
        witnesses = []
        if peak is not None:
            for row, d in np.argwhere(valid & (delta == peak)):
                witnesses.append(dict(query=_axis(a, qi), reference=_axis(r, ri[row]),
                                      coordinate=int(d), abs_residual=peak))
        data[int(qi)] = dict(query=_axis(a, qi), reference_indices=ri.tolist(),
                             own_observed=qm[qi].tolist(), expected_from_reference=need.tolist(),
                             shared_observed=valid.any(0).tolist(),
                             per_dimension_max_abs_residual=_nullable(maxima),
                             max_abs_residual=peak, max_witnesses=witnesses)
    def context(index):
        if not 0 <= index < len(qx):
            return None
        return dict(query=_axis(a, index), belongs_to_this_stage_correspondence=index in data,
                    own_values=_nullable(np.where(qm[index], qx[index], np.nan)), own_observed=qm[index].tolist(),
                    paired_max_abs_residual=data[index]['max_abs_residual'] if index in data else None)
    criteria = {}
    for name in ['p90', 'peak']:
        limit = cutoffs.get(name) if isinstance(cutoffs, dict) else cutoffs if name == 'p90' else None
        selected = [i for i in qq if limit is not None and data[int(i)]['max_abs_residual'] is not None
                    and data[int(i)]['max_abs_residual'] > limit]
        episodes = []
        for ix in _episodes(selected, a['frame_id']):
            maximum = max(data[i]['max_abs_residual'] for i in ix)
            dimensions = sorted({d for i in ix for d, v in enumerate(data[i]['per_dimension_max_abs_residual'])
                                 if v is not None and v > limit})
            episodes.append(dict(start=_axis(a, ix[0]), end=_axis(a, ix[-1]), query_indices=ix,
                sample_count=len(ix), source_pts_span_us=int(a['timestamp_us'][ix[-1]] - a['timestamp_us'][ix[0]]),
                singleton=len(ix) == 1,
                singleton_meaning='One observed sample, not absence or zero-duration action' if len(ix) == 1 else None,
                dimensions=dimensions, max_abs_residual=maximum,
                max_witnesses=[v for i in ix if data[i]['max_abs_residual'] == maximum for v in data[i]['max_witnesses']],
                context_before=context(ix[0] - 1), context_after=context(ix[-1] + 1)))
        criteria[name + '_number_exceedance'] = dict(threshold=limit, comparison='strict greater than',
            meaning='Instantaneous raw shared-coordinate residual exceeds the saved FIT ' + name +
                    ' numerical cutoff; not a recomputed quantile, new classification or error duration.',
            above_query_samples=len(selected), episodes=episodes)
    all_ref = np.arange(bounds[0], bounds[1] + 1)
    values = [v['max_abs_residual'] for v in data.values() if v['max_abs_residual'] is not None]
    derived_peak = max(values) if values else None
    # This is a preservation assertion, never a replacement of original metrics.
    old_peak = original.get('peak_distance')
    if old_peak is None:
        if derived_peak is not None:
            raise AssertionError('New trace must not invent support absent from old summary')
    elif derived_peak is None or abs(old_peak - derived_peak) > 1e-12:
        raise AssertionError('Stored raw peak not preserved in detailed trace')
    return dict(interpretation=_interpretation(family), source='unchanged existing path and measured/learned channels',
        query_frame_opportunities=len(qq), query_dimension_opportunities=len(qq) * dim,
        reference_frame_opportunities=len(all_ref), reference_dimension_opportunities=len(all_ref) * dim,
        reference_original_expected=int(rm0[all_ref].sum()), reference_scoped_expected=int(rm[all_ref].sum()),
        reference_scope_excluded=int(rm0[all_ref].sum() - rm[all_ref].sum()),
        shared_query_dimension_cells=sum(sum(v['shared_observed']) for v in data.values()),
        fully_unshared_query_samples=sum(v['max_abs_residual'] is None for v in data.values()),
        trace=[data[int(i)] for i in qq], exceedances=criteria,
        original_peak_distance=old_peak, original_peak_at=deepcopy(original.get('peak_at')),
        original_peak_preserved_in_trace=True, unknown_is_not_zero=True,
        every_short_peak_retained=True, classification_unchanged=True, educational_correctness=None)


def enrich(a, reference_rows, reference_arrays, result, ranges):
    """Return copy with extra details; every existing key/value is immutable."""
    out = deepcopy(result)
    if 'detail_enrichment' in out:
        raise ValueError('Already enriched; do not overwrite existing detail fields')
    if not out.get('references'):
        out['detail_enrichment'] = dict(status='NO_REFERENCE_COMPARISON', original_values_unchanged=True)
        return out
    engine = _engine()
    rows = reference_rows if isinstance(reference_rows, dict) else {r['clip_id']: r for r in reference_rows}
    qch = engine.channels(a)
    ids = set()
    for ci, comparison in enumerate(out['references']):
        rid = comparison['id']
        if rid in ids:
            raise ValueError('Duplicate reference identity')
        ids.add(rid)
        ref = reference_arrays[rid]
        rch = engine.channels(ref)
        scope, rules = engine.reference_scope(rows[rid], rch)
        for si, stage in enumerate(comparison['stages']):
            if 'detail_enrichment' in stage:
                raise ValueError('Existing detail field would be overwritten')
            source_stage = rows[rid]['stages'][si]
            if stage.get('canonical_stage_id') != source_stage.get('canonical_stage_id'):
                raise ValueError('Source stage identity mismatch')
            details = {f: trace_family(a, ref, qch[f], rch[f], scope[f], comparison['path'],
                       source_stage['frames'], ranges[f], f, stage['families'][f]) for f in engine.FAMILIES}
            stage['detail_enrichment'] = dict(families=details, reference_scope_rules=rules,
                original_query_range_unchanged=True, correspondence_is_estimated=True,
                actual_3d_rotation_verified=False, linguistic_importance=None)
    chosen = list(out['selected_whole_references'])
    for si, stage in enumerate(out['stages']):
        for feature in stage.get('features', []):
            if 'detail_enrichment' in feature:
                raise ValueError('Existing feature detail would be overwritten')
            family = next(k for k, v in engine.LABELS.items() if v == feature['name'])
            pointers = []
            for rid in chosen:
                ci = next(i for i, c in enumerate(out['references']) if c['id'] == rid)
                pointers.append(dict(reference_id=rid, json_pointer=f'/references/{ci}/stages/{si}/detail_enrichment/families/{family}'))
            feature['detail_enrichment'] = dict(interpretation=_interpretation(family), references=pointers,
                original_warning_and_numbers_preserved=True, elapsed_span_is_not_action_duration=True)
    out['detail_enrichment'] = dict(status='ADDED_SAVED_CORRESPONDENCE_EVIDENCE',
        observer_calls=0, new_paths=0, thresholds_modified=False, status_changes=0,
        original_values_unchanged=True, all_original_peaks_retained=True,
        source_pts_span='Last minus first source timestamp; singleton0 span still contains one observation.',
        no_interpolation=True, no_educational_truth=True)
    return out
