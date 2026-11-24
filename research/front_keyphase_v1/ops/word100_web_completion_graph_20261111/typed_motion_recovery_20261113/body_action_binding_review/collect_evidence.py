"""Read-only saved-score/coordinate arithmetic for a design review, no models."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parent
H, G = U.parent, U.parent.parent
O = G.parent


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def describe(p, available, a, b):
    """Descriptive projected measurements, no threshold-derived action label."""
    arm = p[:, 6] - p[:, 4]
    length = np.linalg.norm(arm, axis=1)
    good = available[:, [1, 2, 4, 6]].all(1) & np.isfinite(p[:, [1, 2, 4, 6]]).all((1, 2)) & (length > .05)
    # Do not bridge missing coordinates. Gaps stay in the observation count.
    edge = good[a:b] & good[a+1:b+1]
    unit = arm / np.maximum(length[:, None], 1e-5)
    d = np.diff(p[a:b+1, 6], axis=0)[edge]
    u, v = unit[a:b][edge], unit[a+1:b+1][edge]
    angle = np.degrees(np.arctan2(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0], (u*v).sum(1)))
    step = np.linalg.norm(d, axis=1)
    endpoints = bool(good[a] and good[b])
    return dict(span=[a, b], frames=b-a+1, observed=int(good[a:b+1].sum()), valid_adjacent_steps=int(edge.sum()),
        wrist_delta_xy=(p[b, 6]-p[a, 6]).tolist() if endpoints else None,
        elbow_delta_xy=(p[b, 4]-p[a, 4]).tolist() if endpoints else None,
        forearm_upright_delta=float(unit[a, 1]-unit[b, 1]) if endpoints else None,
        wrist_path=float(step.sum()), max_wrist_step=float(step.max()) if len(step) else None,
        largest_step_fraction_of_path=float(step.max()/step.sum()) if len(step) and step.sum() else 0.,
        signed_turn_degrees=float(angle.sum()), absolute_turn_degrees=float(abs(angle).sum()),
        max_turn_step_degrees=float(abs(angle).max()) if len(angle) else None,
        interpretation='2D descriptive geometry; no action/correctness label or decision threshold')


def from_x(x):
    p = x[:, :102].astype(float).reshape(-1, 51, 2)[:, :9]
    mask = x[:, 186:195] > .5
    shoulders = mask[:, 1] & mask[:, 2]
    scale = np.median(np.linalg.norm(p[shoulders, 1]-p[shoulders, 2], axis=1))
    assert scale > 0
    return (p-(p[:, 1]+p[:, 2])[:, None]/2)/scale, mask


def main():
    assert not (U / 'EVIDENCE.json').exists()
    paths = {
        'association': H/'hand_transition_learning/ASSOCIATION_DESIGN_REVIEW.json',
        'dense': H/'typed_shape_transition_review/REVIEW_COMPLETE.json',
        'local': G/'candidate_description_supervision/local_rank_learning/evaluation/COMPARISON.json',
        'diagnosis': G/'candidate_description_supervision/local_rank_learning/FAILURE_DIAGNOSIS.json',
        'body': H/'body_progression_learning/evaluation/COMPARISON.json',
        'partial': H/'body_progression_learning/local_evaluation/COMPARISON.json',
        'static': H/'body_progression_learning/static_evaluation/COMPARISON.json',
        'composition': H/'composition_audit/RESULTS.json',
        'fixture_inputs': H/'palm_preserving_controls/FIXTURE_FROZEN.json',
        'features': O/'word100_guide_requirement_learning_20261111/requirement_core.py',
        'cards': O/'word100_action_feedback_demo_20261108/CARDS.json',
        'local_encoder': G/'candidate_description_supervision/transfer_local_timefix_core.py',
        'local_loss': G/'candidate_description_supervision/train_local_rank.py',
    }
    pins = {str(p): sha(p) for p in paths.values()}
    values = {name: read(path) for name, path in paths.items() if path.suffix == '.json'}
    comp, diag = values['local'], values['diagnosis']
    lookup = {(r['seed'], r['clip_id'], r['kind']): r for r in comp['predictions']}
    failures = []
    for row in diag['rows']:
        if row['condition'] != 'local_full' or row['all_witness']:
            continue
        source = lookup[row['seed'], row['clip_id'], row['kind']]
        assert sha(source['source']) == source['source_sha256']
        pins[source['source']] = source['source_sha256']
        with np.load(source['source'], allow_pickle=False) as z:
            p, mask = from_x(z['x'])
            chosen = describe(p, mask, *row['selected'])
            good = describe(p, mask, *row['best_source_witness_candidate']['span'])
        failures.append(dict(**row, selected_geometry=chosen, admissible_witness_geometry=good))
    source_inputs = {r['clip_id']: r for r in values['fixture_inputs']['sources']}
    dense_segments = []
    for row in values['dense']['rows']:
        source = source_inputs[row['clip_id']]
        assert sha(source['raw_path']) == source['raw_sha256']
        pins[source['raw_path']] = source['raw_sha256']
        with np.load(source['raw_path'], allow_pickle=False) as z:
            p = z['xy_px'][:, [0, 5, 6, 7, 8, 9, 10, 11, 12]].astype(float)
            mask = np.isfinite(p).all(2) & (p[..., 0] >= 0) & (p[..., 0] < int(z['source_width'])) & (p[..., 1] >= 0) & (p[..., 1] < int(z['source_height']))
            scale = np.median(np.linalg.norm(p[:, 1]-p[:, 2], axis=1))
            p = (p-(p[:, 1]+p[:, 2])[:, None]/2)/scale
            frame_lookup = {int(f): i for i, f in enumerate(z['frame_id'])}
        records = row['frame_records']
        runs = []
        for record in records:
            if not runs or record['body_motion'] != runs[-1][-1]['body_motion'] or record['frame'] != runs[-1][-1]['frame'] + 1:
                runs.append([])
            runs[-1].append(record)
        for run in runs:
            if len(run) < 2 or run[0]['body_motion'] == 'unknown':
                continue
            a, b = frame_lookup[run[0]['frame']], frame_lookup[run[-1]['frame']]
            dense_segments.append(dict(clip_id=row['clip_id'], person=row['person'], source_frames=[run[0]['frame'], run[-1]['frame']],
                existing_rgb_body_label=run[0]['body_motion'], existing_hand_states=dict(Counter(r['hand_state'] for r in run)),
                geometry=describe(p, mask, a, b)))
    partial = values['partial']['partial_all_families']
    bycondition = {}
    for name in ('local', 'body', 'composition'):
        bycondition[name] = [s for s in values[name]['summary'] if s['kind'] in ('full_normal', 'speed075', 'speed125', 'dynamic_complete_blocks', 'reversed_core_auto_order')]
    result = dict(status='READ_ONLY_DESIGN_EVIDENCE_NO_MODELS_NO_NEW_LABELS',
        local_full_failure_count=len(failures), saved_failures_with_descriptive_geometry=failures,
        dense_counts=values['dense']['counts'], dense_contiguous_segments=dense_segments,
        association_aggregate=values['association']['aggregate'],
        saved_local_diagnosis_summary=diag['summary'], body_partial_all_families=partial,
        body_static_overall=values['static']['overall'], saved_selected_summaries=bycondition,
        scenario_count=len({(r['clip_id'], r['kind']) for r in comp['rows']}),
        scope='Known exposed public DEV failures for mechanism diagnosis only; no tuning/labels selected from these scores. Dense FIT labels remain original206.',
        no_new_model_inference=True, no_new_training=True, no_labels_created=True, pins=pins)
    for path, expected in pins.items():
        assert sha(path) == expected
    (U / 'EVIDENCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', 'utf-8')
    print('Recorded', len(failures), 'saved failures and', len(dense_segments), 'existing dense-label segments; no model/label change')
    for r in failures:
        a, b = r['selected_geometry'], r['admissible_witness_geometry']
        print(r['seed'], r['clip_id'], r['card'], 'selected', a['span'], 'maxstep fraction', round(a['largest_step_fraction_of_path'],3), 'turn',round(a['absolute_turn_degrees'],1), 'vs source witness', b['span'],round(b['largest_step_fraction_of_path'],3),round(b['absolute_turn_degrees'],1))
    for r in dense_segments:
        if r['existing_rgb_body_label'] in ('wrist_lowering', 'forearm_raising'):
            m=r['geometry']; print(r['clip_id'],r['existing_rgb_body_label'],r['source_frames'],'dy',round(m['wrist_delta_xy'][1],3),'upright',round(m['forearm_upright_delta'],3),'turn',round(m['signed_turn_degrees'],1))
    print('body partial summaries', json.dumps(partial,ensure_ascii=False))


if __name__ == '__main__':
    main()
