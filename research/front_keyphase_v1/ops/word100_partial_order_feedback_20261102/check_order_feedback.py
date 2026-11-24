"""Display-contract tests only: saved Y pairs, no inference/re-evaluation.

Prints the report to stdout; the parent owns scope/integration/report files.
"""
from pathlib import Path
from collections import defaultdict, Counter
from copy import deepcopy
import json
import hashlib
from order_feedback import attach_partial_order, FIELD

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
Y = OPS / 'word100_direct_occurrence_order_20261102'
DATA = Path('W:/WordsInSilence_derived/word100_direct_occurrence_order_20261102/evaluations/tied_readout_dev.json')
ACTION = OPS / 'word100_action_episode_supervision_20261102/ACTION_TARGETS.json'
FORBIDDEN = {'target', 'truth_reason', 'first_witnesses', 'second_witnesses', 'raw_prediction', 'confirmed_prediction',
             'person', 'kind', 'clip_id', 'flow_key', 'word_id', 'canonical_reference_anchor_frames'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in FORBIDDEN, key
            if key in ('whole_sign_correctness', 'education_error', 'repeat_count'):
                assert item is None
            if key in ('complete_action_verified', 'missing_action_verified', 'repeat_count_verified',
                       'repeated_sequence_verified', 'source_clause_verified', 'full_span_semantic_verified'):
                assert item is False
            if key == 'completion_unknown':
                assert item is True
            yield key, item
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def formatted(base, pairs, families, graph):
    snapshots = deepcopy((base, pairs, families, graph))
    result = attach_partial_order(base, pairs, families, graph)
    assert (base, pairs, families, graph) == snapshots
    assert set(result) == set(base) | {FIELD}
    assert {k: result[k] for k in base} == base
    assert result['sequence_verdict'] == base['sequence_verdict']
    assert result['point_evidence'] == base['point_evidence']
    assert result['rgb_motion_evidence'] == base['rgb_motion_evidence']
    list(walk(result[FIELD]))
    # Evaluation annotations cannot affect display or leak to the client.
    poisoned = deepcopy(pairs)
    for pair in poisoned:
        for key in ('target', 'truth_reason', 'first_witnesses', 'second_witnesses', 'raw_prediction', 'confirmed_prediction'):
            pair[key] = 'EVALUATION_TRUTH_POISON_DO_NOT_DISPLAY'
        pair['person'] = 'DO_NOT_DISPLAY_PERSON'; pair['kind'] = 'DO_NOT_DISPLAY_EDIT_KIND'
    assert attach_partial_order(base, poisoned, families, graph) == result
    text = json.dumps(result[FIELD], ensure_ascii=False)
    assert 'EVALUATION_TRUTH_POISON' not in text and 'DO_NOT_DISPLAY' not in text
    assert not any(phrase in text for phrase in ('정확히 수행했습니다', '빠진 동작이 없습니다', '수어를 완성했습니다', '정답입니다'))
    return result[FIELD]


def fixture(template, status, first, second, missing=(), gaps=()):
    pair = deepcopy(template)
    pair.update(first=0, second=1, comparison_frames=8, timestamp_kind='native_input_PTS',
                status=status, unobserved_frames=list(missing), gaps_inside_evidence=list(gaps),
                overlap_frames=sorted(set(first) & set(second)))
    for prefix, frames in [('first_evidence', first), ('second_evidence', second),
                           ('raw_physical_first', first), ('raw_physical_second', second)]:
        pair[prefix + '_frames'] = list(frames)
        pair[prefix + '_seconds'] = [0.017 + frame * 0.037 for frame in frames]
    return pair


def main():
    files = [DATA, ACTION, Y / 'EVALUATION_PROTOCOL.json', Y / 'FROZEN.json', Y / 'SAVED_READOUT_VERIFIED.json',
             HERE / 'order_feedback.py', Path(__file__).resolve()]
    before = {str(p): digest(p) for p in files}
    verified = load(Y / 'SAVED_READOUT_VERIFIED.json')
    assert verified['status'] == 'PASS_SAVED_ARITHMETIC_CANDIDATE_GATES_NOT_MET'
    assert verified['new_inference'] == verified['new_training'] == 0
    protocol = load(Y / 'EVALUATION_PROTOCOL.json'); families = load(ACTION)['families']
    data = load(DATA); assert data['condition'] == 'tied_readout' and data['split'] == 'dev'
    pairs = data['pairs']; assert len(pairs) == 110
    base = dict(sequence_verdict={'code': 'LEGACY_UNCHANGED', 'details': [1, 2]},
                point_evidence={'coordinates': [0.1, None, 0.3]},
                rgb_motion_evidence={'observed': True, 'source_clause_correctness': None},
                arbitrary_old_field=['retain', {'nested': True}])
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair['flow_key']].append(pair)
    checked = 0; statuses = Counter(); directions = Counter(); cards = 0; global_missing_examples = 0
    for key, rows in grouped.items():
        graph = protocol['graphs'][rows[0]['word_id']]
        assert all(r['word_id'] == rows[0]['word_id'] for r in rows)
        result = formatted(base, rows, families, graph)
        assert len(result['items']) == len(rows)
        assert result['global_missing_image_frames'] == rows[0]['unobserved_frames']
        global_missing_examples += bool(result['global_missing_image_frames'])
        for pair, item in zip(rows, result['items']):
            assert item['first_evidence']['input_frames'] == pair['first_evidence_frames']
            assert item['first_evidence']['native_seconds'] == pair['first_evidence_seconds']
            assert item['second_evidence']['native_seconds'] == pair['second_evidence_seconds']
            assert item['gaps_inside_evidence_frames'] == pair['gaps_inside_evidence']
            assert item['overlap_frames'] == pair['overlap_frames']
            assert item['completion_unknown'] and item['limit_message']
            statuses[item['observation_status']] += 1; directions[str(item['observed_direction'])] += 1
            for side in ('first', 'second'):
                for desc in item[side]['reference_descriptions']:
                    for card in desc['source_cards']:
                        assert card['context_only'] and not card['source_clause_verified']
                        cards += 1
            checked += 1
        if key == 'dev_034':
            # This saved action-cut case has no direct first-family gold witness.
            # That fact is tested here only; it is absent from the formatter.
            assert rows[0]['target'] == -1 and not rows[0]['first_witnesses']
            item = result['items'][0]
            assert item['observation_status'] == 'OBSERVED_A_BEFORE_B'
            assert item['first_evidence']['input_frames'] and item['second_evidence']['input_frames']
            assert item['completion_unknown'] and not item['complete_action_verified']
            action_cut_preserved = True
    assert checked == 110 and statuses == Counter(r['status'] for r in pairs)
    assert directions['A_BEFORE_B'] > 0 and directions['B_BEFORE_A'] > 0 and global_missing_examples > 0
    no_edges = protocol['graphs']['WORD1567']
    no_result = formatted(base, [], families, no_edges)
    assert no_result['status'] == 'NO_DISTINCT_ORDER_QUERY' and no_result['query_count'] == 0
    assert no_result['repeated_sequence_verified'] is False and no_result['repeat_count'] is None
    assert no_result['global_missing_image_frames'] is None and not no_result['global_image_availability_reported']
    # The same behavior follows the graph, not a WORD1567 branch.
    assert formatted(base, [], families, dict(no_edges, source='another reference graph'))['status'] == 'NO_DISTINCT_ORDER_QUERY'
    graph = dict(groups=[0, 1], precedences=[[0, 1]], source='Fixture reference context')
    fixtures = [
        ('short_reverse', 'OBSERVED_B_BEFORE_A', [4], [3], (), ()),
        ('gap', 'PAIR_OBSERVATION_GAP', [1], [5], (3,), (3,)),
        ('outer_gap', 'OBSERVED_A_BEFORE_B', [1], [5], (0, 7), ()),
        ('overlap', 'PAIR_OVERLAPPING_EVIDENCE', [1, 2], [2, 3], (), ()),
        ('interleaving', 'PAIR_INTERLEAVED_EVIDENCE', [1, 4], [2, 3], (), ()),
        ('not_observed', 'PAIR_OCCURRENCE_UNCONFIRMED', [], [3], (), ()),
        ('both_not_observed', 'PAIR_OCCURRENCE_UNCONFIRMED', [], [], (0, 1, 2, 3, 4, 5, 6, 7), ()),
        ('readout_disagreement', 'PAIR_READOUT_DISAGREEMENT', [1], [4], (), ()),
        ('retained_disagreement', 'RETAINED_PHYSICAL_ORDER_DISAGREEMENT', [1], [4], (), ()),
    ]
    for name, status, first, second, missing, gaps in fixtures:
        pair = fixture(pairs[0], status, first, second, missing, gaps)
        report = formatted(base, [pair], families, graph)
        item = report['items'][0]
        assert item['observation_status'] == status and item['completion_unknown']
        if name == 'short_reverse':
            assert item['order_difference_candidate'] and item['education_error'] is None
        if name == 'not_observed':
            assert item['first_evidence']['observation'] == 'NOT_CONFIRMED' and not item['missing_action_verified']
    rejected = 0
    bad = fixture(pairs[0], 'OBSERVED_A_BEFORE_B', [1], [5], (3,), (3,))
    for b, pp, gg in [
        (dict(base, partial_order_feedback={}), [], graph),
        (base, [bad], graph),
        (base, [fixture(pairs[0], 'OBSERVED_B_BEFORE_A', [1], [5])], graph),
        (base, [dict(fixture(pairs[0], 'OBSERVED_A_BEFORE_B', [1], [5]), timestamp_kind='DTW_REFERENCE_TIME')], graph),
        (base, [fixture(pairs[0], 'OBSERVED_A_BEFORE_B', [1], [5])], no_edges),
    ]:
        try:
            attach_partial_order(b, pp, families, gg)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid observation must fail closed.')
    assert rejected == 5
    after = {str(p): digest(p) for p in files}; assert before == after
    output = dict(status='PASS_DISPLAY_CONTRACT_ONLY', saved_pairs=checked, saved_input_groups=len(grouped),
        status_counts=dict(statuses), observed_directions=dict(directions), source_context_cards_checked=cards,
        groups_with_global_missing_images=global_missing_examples, fixture_cases=len(fixtures), invalid_inputs_rejected=rejected,
        action_cut_dev034_partial_observation_preserved=action_cut_preserved,
        no_graph_edges_not_repeat_verified=True, evaluation_annotation_poison_invariance=True,
        old_result_and_argument_deep_preservation=True, native_evidence_times_exact=True,
        source_descriptions_not_full_clause_truth=True, native_gap_times_not_invented=True,
        inputs_unchanged=True, pins=before, new_model_inference=0, training=0,
        old_results_recomputed=False, whole_goal_complete=False,
        limits=['Formatter implementation tests by its author; parent performs separate review.',
                'Saved repeated-development cases and fixtures are not independent educational validation.',
                'No full-sign correctness, completion, exact boundary, or omitted-action determination.'])
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
