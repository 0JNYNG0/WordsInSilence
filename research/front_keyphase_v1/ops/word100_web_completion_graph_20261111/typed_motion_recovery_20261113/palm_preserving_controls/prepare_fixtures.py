"""Freeze and prepare public FIT coordinate controls; never imports a model.

Two explicit invocations: `python -B prepare_fixtures.py freeze`, then `... run`.
All new files stay in this directory. Historical code and caches are read-only.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import ast
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
from types import SimpleNamespace
import traceback
import numpy as np

from palm_control import (CHAINS, DISTAL, HAND_START, PROTECTED, make_control,
                          point_available, rebuild_channel_pair)

U = Path(__file__).resolve().parent
H, GRAPH, OPS = U.parent, U.parent.parent, U.parent.parent.parent
ANNOTATIONS = [GRAPH / 'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',
               GRAPH / 'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']
MOTION = OPS / 'word100_motion_components_20261109/motion.py'
BODY = GRAPH / 'candidate_description_supervision/projected_direction_core.py'
PALM = H / 'partial_direction_contract/palm_projection_core.py'
ADAPTER = OPS / 'word100_rtmw_shared_input_20261108/adapter.py'


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def write_new(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(U.resolve()) and not path.exists()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')


def verify_pins(pins):
    for path, expected in pins.items():
        assert sha(path) == expected, ('pin mismatch', path)


def pack(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in ('xy_px', 'score', 'timestamp_us', 'frame_id', 'source_width', 'source_height')}


def resolve_frame(frame_ids, frame):
    found = np.flatnonzero(frame_ids == frame)
    assert len(found) == 1, ('frame must resolve exactly once', frame)
    return int(found[0])


def freeze():
    assert not (U / 'FIXTURE_FROZEN.json').exists()
    proposal = read(U / 'PROPOSED_INPUT_PINS.json')
    verify_pins(proposal['pins'])
    annotated = {}
    for path in ANNOTATIONS:
        for row in read(path)['rows']:
            if row['clip_id'] in {s['clip_id'] for s in proposal['proposed_sources']}:
                assert row['clip_id'] not in annotated
                annotated[row['clip_id']] = (row, path)
    sources = []
    for source in proposal['proposed_sources']:
        row, annotation_path = annotated[source['clip_id']]
        assert row['role'] == source['role'] == 'fit' and row['person'] == source['person']
        assert row['source_sha256'] == source['source_sha256']
        raw = pack(source['raw_path'])
        events = []
        indices = set()
        for event in row['events']:
            low, high = event['onset'][0], event['offset'][-1]
            a, b = resolve_frame(raw['frame_id'], low), resolve_frame(raw['frame_id'], high)
            assert a < b and b - a == high - low
            indices.update(range(a, b + 1))
            events.append(dict(event_id=event['event_id'], axis=event['axis'], family=event['family'],
                               original_onset=event['onset'], original_offset=event['offset'],
                               source_frame_span=[low, high], array_span=[a, b],
                               direct_witness_frames=event['progression_witness_frames']))
        requested = sorted(indices)
        assert requested
        visual_indices = [requested[0], requested[(len(requested) - 1) // 2], requested[-1]]
        sources.append(dict(**source, annotation_path=str(annotation_path), events=events,
                            frozen_indices=requested, visual_indices=visual_indices,
                            visual_frame_ids=[int(raw['frame_id'][i]) for i in visual_indices],
                            width=int(raw['source_width']), height=int(raw['source_height'])))
    donor = dict(proposal['donor'])
    raw = pack(donor['raw_path'])
    index = resolve_frame(raw['frame_id'], donor['frame_id'])
    donor.update(array_index=index, timestamp_us=int(raw['timestamp_us'][index]),
                 width=int(raw['source_width']), height=int(raw['source_height']))
    labels = read(donor['attribute_source'])['rows']
    label = next(a for r in labels if r['clip_id'] == donor['clip_id'] for a in r['shape'] if a['frame'] == donor['frame_id'])
    assert label['attributes']['1'] == 0 and label['attributes']['3'] == 1
    pins = dict(proposal['pins'])
    for path in [Path(__file__), U / 'make_contact_sheets.py', U / 'PROPOSED_INPUT_PINS.json', *ANNOTATIONS, MOTION, BODY, PALM]:
        pins[str(path)] = sha(path)
    contract = dict(at=stamp(), status='FROZEN_BEFORE_COORDINATE_GENERATION',
        author='Codex palm_fixture_contract_1113', parent_review='Root reports 12 toy contracts passed before execution authorization',
        source_count=12, intervention_sides=['L', 'R'], expected_control_count=24,
        sources=sources, donor=donor, pins=pins,
        selection='Union of existing directly RGB-reviewed source event OUTER intervals; no unannotated gaps. Each event retained for measurement comparison. No model predictions read.',
        visual_selection='Three fixed requested indices: first, lower-middle, last; no replacement/rejection/outcome-based frame selection. All12 sources and both intervention sides shown.',
        geometry_checks='Exact new palm_projection_core.geometry g58/eligibility/info and current projected body-wrist geometry, plus legacy palm features; axis2, both moving sides, each source event, forward and reversed measurement. Formula functions AST-extracted unchanged; imports/classes/model loading excluded.',
        reconstruction='Canonical full adapter channels, no new quarantine; same original raw observations and timestamps',
        thresholds='Inherited helper numeric degeneracy and image-bounds only; no parameter/seed/donor search',
        skipped='Reject whole requested frame and leave source untouched; rejected remains denominator. No fallback donor or frame selection.',
        measurement_tolerance='Protected coordinates/channels/masks/time and g58 exact; original projected finger lengths atol=0.001px rtol=0.0001 only for float32 rounding',
        interpretation='Synthetic distal-four-finger attribute controls; no actual incorrect KSL, expert truth, semantic completion, visibility guarantee or new accuracy',
        private_access=False, final8=False, model_inference=False, training=False, web_changed=False, git_changed=False)
    verify_pins(pins)
    write_new(U / 'FIXTURE_FROZEN.json', contract)
    print('Frozen 12 sources, 24 side-specific controls; donor frame/index', donor['frame_id'], index, flush=True)


def extracted(path, names, environment):
    """Execute only named pure function/constant AST nodes; never source imports."""
    tree = ast.parse(Path(path).read_text('utf-8-sig'), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    expected = set(names)
    actual = {node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id for node in selected}
    assert actual == expected, (path, actual, expected)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), environment)
    return SimpleNamespace(**{name: environment[name] for name in names})


def geometry_functions():
    motion = extracted(MOTION, ['KINDS', 'SIDES', 'body_points', 'series', 'features'], {'np': np})
    prior = SimpleNamespace(motion=motion)
    body = extracted(BODY, ['geometry'], {'np': np, 'prior': prior})
    old = SimpleNamespace(prior=prior, geometry=body.geometry)
    palm = extracted(PALM, ['PALM_OFFSETS', 'SCHEMA', 'geometry'], {'np': np, 'old': old})
    return motion, body, palm


def canonical_adapter():
    spec = importlib.util.spec_from_file_location('palm_fixture_adapter', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    assert not (U / 'FIXTURE_RESULTS.json').exists() and not (U / 'fixtures').exists()
    frozen = read(U / 'FIXTURE_FROZEN.json')
    verify_pins(frozen['pins'])
    adapter = canonical_adapter()
    motion, body, palm = geometry_functions()
    donor_source = pack(frozen['donor']['raw_path'])
    di = resolve_frame(donor_source['frame_id'], frozen['donor']['frame_id'])
    assert di == frozen['donor']['array_index']
    donor = donor_source['xy_px'][di, 112:133]
    dav = point_available(donor, frozen['donor']['width'], frozen['donor']['height'])
    assert dav[[0, *range(5, 21)]].all(), 'Fixed donor unavailable: stop, no fallback'
    (U / 'fixtures').mkdir()
    rows, geometry_rows = [], []
    for source in frozen['sources']:
        raw = pack(source['raw_path'])
        original_hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in raw.items()}
        for side in frozen['intervention_sides']:
            result = make_control(raw['xy_px'], raw['score'], raw['timestamp_us'], raw['frame_id'], donor,
                side=side, donor_side='R', frame_indices=source['frozen_indices'],
                width=source['width'], height=source['height'], donor_available=dav)
            pair = rebuild_channel_pair(raw['xy_px'], result, convert=adapter.convert)
            before, after = pair['baseline']['tcn280'], pair['control']['tcn280']
            start = HAND_START[side]
            applied = np.array([r['index'] for r in result['frame_report'] if r['applied']], dtype=int)
            assert result['xy_px'][:, start + np.array(PROTECTED)].tobytes() == raw['xy_px'][:, start + np.array(PROTECTED)].tobytes()
            lengths = []
            for chain in CHAINS:
                ids = start + np.array(chain)
                before_lengths = np.linalg.norm(np.diff(raw['xy_px'][applied][:, ids].astype(float), axis=1), axis=2)
                after_lengths = np.linalg.norm(np.diff(result['xy_px'][applied][:, ids].astype(float), axis=1), axis=2)
                np.testing.assert_allclose(after_lengths, before_lengths, rtol=1e-4, atol=1e-3)
                lengths.extend(np.abs(after_lengths - before_lengths).ravel().tolist())
            for event in source['events']:
                a, b = event['array_span']
                for moving in ('L', 'R'):
                    anchor = 'L' if moving == 'R' else 'R'
                    for reverse in (False, True):
                        for name, call in [
                            ('new_palm_projection_g58', lambda x: palm.geometry(x, raw['timestamp_us'], a, b, 2, moving, anchor, reverse)),
                            ('current_body_wrist_projection_g58', lambda x: body.geometry(x, raw['timestamp_us'], a, b, 2, moving, anchor, reverse)),
                            ('legacy_palm_forearm_fraction_g58', lambda x: motion.features(x, raw['timestamp_us'], a, b, 'along_other_forearm', moving, anchor, reverse))]:
                            ga, oka, ia = call(before)
                            gb, okb, ib = call(after)
                            assert ga.tobytes() == gb.tobytes() and bool(oka) == bool(okb) and ia == ib
                            geometry_rows.append(dict(clip_id=source['clip_id'], intervention_side=side, event_id=event['event_id'],
                                source_span=event['source_frame_span'], moving=moving, anchor=anchor, reverse=reverse,
                                measurement=name, eligible_before=bool(oka), eligible_after=bool(okb), exact=True,
                                coverage=ia['coverage'], observed=ia['observed'], signed_change=ia.get('signed_change')))
            path = U / 'fixtures' / (source['clip_id'] + '_' + side + '_distal.npz')
            assert not path.exists()
            np.savez_compressed(path, **{k: result[k] for k in ('xy_px', 'score', 'timestamp_us', 'frame_id', 'synthetic_point')},
                source_width=source['width'], source_height=source['height'],
                original_xy_px=raw['xy_px'], original_tcn280=before, control_tcn280=after,
                original_body_relative_xy=pair['baseline']['body_relative_xy'],
                control_body_relative_xy=pair['control']['body_relative_xy'],
                original_hand_local_xy=pair['baseline']['hand_local_xy'],
                control_hand_local_xy=pair['control']['hand_local_xy'],
                point_available=pair['control']['point_available'], hand_local_available=pair['control']['hand_local_available'],
                body_relative_available=pair['control']['body_relative_available'],
                native_available=pair['control']['native_available'], native67=pair['control']['native67'],
                visibility_evidence=pair['control']['visibility_evidence'], allowed_tcn_change=pair['allowed_tcn_channel_change'],
                requested_indices=np.array(source['frozen_indices'], np.int64),
                schema=np.array('synthetic_distal_four_finger_control_palm_invariant_v1'),
                educational_correctness=np.array('not_defined_synthetic_attribute_control'))
            row = dict(clip_id=source['clip_id'], person=source['person'], word_id=source['word_id'], side=side,
                path=str(path), sha256=sha(path), raw_source_sha256=source['raw_sha256'],
                requested_frames=result['requested_frame_count'], applied_frames=result['applied_frame_count'],
                rejected_frames=result['requested_frame_count'] - result['applied_frame_count'],
                rejections=dict(Counter(r['reason'] for r in result['frame_report'] if not r['applied'])),
                frame_report=result['frame_report'], visual_indices=source['visual_indices'], visual_frame_ids=source['visual_frame_ids'],
                max_projected_segment_length_rounding_px=max(lengths, default=0.),
                invariant_coordinates_and_channels=True, source_metadata_identical=True,
                nonzero_distal_displacement_points=int(np.any(result['xy_px'] != raw['xy_px'], axis=2)[result['synthetic_point']].sum()),
                actual_generated_points=int(result['synthetic_point'].sum()))
            rows.append(row)
            print(source['clip_id'], side, 'applied', row['applied_frames'], '/', row['requested_frames'], 'rejected', row['rejected_frames'], flush=True)
        assert original_hashes == {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in raw.items()}
    verify_pins(frozen['pins'])
    result = dict(at=stamp(), status='PREPARED_AND_COORDINATE_CONTRACT_VERIFIED_NO_MODEL',
        rows=rows, geometry_checks=geometry_rows, checks=len(geometry_rows),
        eligible_geometry_checks=sum(r['eligible_before'] for r in geometry_rows),
        unavailable_geometry_checks=sum(not r['eligible_before'] for r in geometry_rows),
        requested_frames=sum(r['requested_frames'] for r in rows), applied_frames=sum(r['applied_frames'] for r in rows),
        rejected_frames=sum(r['rejected_frames'] for r in rows), geometry_exact_all=True,
        same_source_frame_count_counted_once_per_intervention_side=True,
        pins_verified_before_after=True, frozen_path=str(U / 'FIXTURE_FROZEN.json'), frozen_sha256=sha(U / 'FIXTURE_FROZEN.json'),
        model_inference=False, training=False, real_error_accuracy=False, private_access=False, web_changed=False,
        interpretation='12 public originals, 24 synthetic side interventions; repeated measurements/sides are not independent participants or new accuracy',
        reviewer='Implementing Codex arithmetic; root review of saved results and skeleton sheets remains pending')
    write_new(U / 'FIXTURE_RESULTS.json', result)
    print('Saved', len(rows), 'controls;', len(geometry_rows), 'exact measurement checks', flush=True)


if __name__ == '__main__':
    try:
        {'freeze': freeze, 'run': run}[sys.argv[1]]()
    except BaseException:
        failure = U / 'FIRST_PREPARATION_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), 'utf-8')
        raise
