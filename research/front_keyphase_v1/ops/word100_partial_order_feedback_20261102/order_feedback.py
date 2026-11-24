"""Pure display adapter for frozen, observed partial event order.

No model, score calculation, evaluation label, or word-specific correction is
used here. ``graph`` is one selected FIT reference graph. The caller supplies
observations from Y's frozen tied readout, never a framewise truth annotation.
"""
from copy import deepcopy
import math


FIELD = 'partial_order_feedback'
STATUSES = {
    'OBSERVED_A_BEFORE_B': 'OBSERVED_REFERENCE_ORDER',
    'OBSERVED_B_BEFORE_A': 'OBSERVED_REVERSE_ORDER_CANDIDATE',
    'PAIR_OCCURRENCE_UNCONFIRMED': 'OCCURRENCE_NOT_CONFIRMED',
    'PAIR_OBSERVATION_GAP': 'OBSERVATION_GAP',
    'PAIR_OVERLAPPING_EVIDENCE': 'OVERLAPPING_EVIDENCE',
    'PAIR_INTERLEAVED_EVIDENCE': 'INTERLEAVED_EVIDENCE',
    'PAIR_READOUT_DISAGREEMENT': 'READOUT_DISAGREEMENT',
    'RETAINED_PHYSICAL_ORDER_DISAGREEMENT': 'RETAINED_PHYSICAL_DISAGREEMENT',
}
MESSAGES = {
    'OBSERVED_A_BEFORE_B': '표시한 두 동작의 관측 근거는 참고 자료에서 본 순서로 나타났습니다.',
    'OBSERVED_B_BEFORE_A': '표시한 두 동작의 관측 근거가 참고 자료에서 본 순서와 반대로 나타났습니다. 순서 차이 후보이며 수어 오답 판정은 아닙니다.',
    'PAIR_OCCURRENCE_UNCONFIRMED': '한쪽 또는 양쪽 동작의 관측 근거가 충분하지 않아 순서를 확인하지 못했습니다. 해당 동작을 하지 않았다는 뜻은 아닙니다.',
    'PAIR_OBSERVATION_GAP': '두 동작의 관측 근거 사이에 이미지가 없는 순간이 있어 순서를 확정하지 않았습니다.',
    'PAIR_OVERLAPPING_EVIDENCE': '같은 순간에 두 동작의 근거가 겹쳐 어느 쪽이 먼저인지 확정하지 않았습니다.',
    'PAIR_INTERLEAVED_EVIDENCE': '두 동작의 근거가 섞여 나타나 한 번의 앞뒤 순서로 정리하지 않았습니다. 반복 횟수도 판단하지 않았습니다.',
    'PAIR_READOUT_DISAGREEMENT': '관측 시점의 앞뒤 관계와 순서 판정부의 결과가 충분히 일치하지 않아 확정하지 않았습니다.',
    'RETAINED_PHYSICAL_ORDER_DISAGREEMENT': '기존 동작 관측과 새 순서 근거가 서로 반대여서 기존 근거를 보존하고 판단을 보류했습니다.',
}
LIMIT = '여기서는 보인 일부 동작의 앞뒤 관계만 표시합니다. 전체 행동의 완성 여부와 빠진 동작은 아직 확인하지 못했습니다.'


def _frame_list(value, n):
    if not isinstance(value, (list, tuple)):
        raise ValueError('Frame evidence must be a list.')
    result = list(value)
    if any(type(v) is not int or v < 0 or v >= n for v in result):
        raise ValueError('Invalid input-frame index.')
    if result != sorted(set(result)):
        raise ValueError('Frame evidence must be unique and ordered.')
    return result


def _evidence(pair, prefix, n):
    frames = _frame_list(pair[prefix + '_frames'], n)
    seconds = list(pair[prefix + '_seconds'])
    if len(seconds) != len(frames) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in seconds):
        raise ValueError('Native timestamps must accompany every observed frame.')
    if any(a >= b for a, b in zip(seconds, seconds[1:])):
        raise ValueError('Native evidence timestamps must increase.')
    return dict(input_frames=frames, native_seconds=seconds,
                first_observed_second=seconds[0] if seconds else None,
                last_observed_second=seconds[-1] if seconds else None,
                span_is_exact_action_boundary=False,
                observation='EVIDENCE_PRESENT' if frames else 'NOT_CONFIRMED')


def _family_context(family):
    descriptions = []
    for source in family.get('source_descriptions', []):
        cards = []
        for card in source.get('source_cards', []):
            # No canonical-frame labels, evaluation targets, or arbitrary fields
            # are forwarded. Source text is context, not an observed full clause.
            clean = {k: deepcopy(card[k]) for k in ('clause_id', 'text', 'url', 'restrictions', 'limitations') if k in card}
            if 'source' in card:
                clean['source'] = {k: deepcopy(card['source'][k]) for k in ('path', 'sha256', 'json_pointer') if k in card['source']}
            clean.update(scope='REFERENCE_CONTEXT_ONLY', context_only=True,
                         source_clause_verified=False, full_span_semantic_verified=False)
            cards.append(clean)
        descriptions.append(dict(text=source.get('description', ''),
            scope='REFERENCE_OBSERVATION_DESCRIPTION_NOT_CURRENT_FULL_CLAUSE', source_cards=cards))
    return dict(id=int(family['id']), name=family.get('name', ''),
                reference_descriptions=descriptions, guide_ids=deepcopy(family.get('guide_ids', [])),
                source_clause_verified=False)


def attach_partial_order(base_result, pairs, families, graph):
    """Return a deep copy plus a whitelisted, non-educational display layer.

    Existing point/RGB motion/sequence verdict fields are never rewritten.
    The graph selects query pairs only; no repeated decoding or forced order.
    Completion is unknown even when the observed order matches the reference.
    """
    if not isinstance(base_result, dict) or FIELD in base_result:
        raise ValueError('Expected an unmodified base result without this layer.')
    family_map = {int(f['id']): f for f in families}
    if len(family_map) != len(families):
        raise ValueError('Duplicate family ID.')
    edges = [tuple(map(int, edge)) for edge in graph.get('precedences', [])]
    if len(set(edges)) != len(edges) or any(len(edge) != 2 or edge[0] == edge[1] for edge in edges):
        raise ValueError('Invalid reference graph.')
    groups = list(map(int, graph.get('groups', [])))
    if any(k not in family_map for k in groups) or any(a not in groups or b not in groups for a, b in edges):
        raise ValueError('Graph references unknown physical groups.')
    items = []; seen = set(); global_missing = None; total_frames = None
    for pair in pairs:
        key = (int(pair['first']), int(pair['second']))
        if key not in edges or key in seen:
            raise ValueError('An observation must belong to one selected reference query.')
        seen.add(key)
        status = pair['status']
        if status not in STATUSES:
            raise ValueError('Unsupported observed-pair status.')
        if pair.get('timestamp_kind', 'native_input_PTS') != 'native_input_PTS':
            raise ValueError('Require native input time, not DTW reference time.')
        n = pair['comparison_frames']
        if type(n) is not int or n < 0:
            raise ValueError('Invalid comparison-frame count.')
        missing = _frame_list(pair['unobserved_frames'], n)
        if total_frames is not None and (n != total_frames or missing != global_missing):
            raise ValueError('Pairs must refer to the same input and availability.')
        total_frames = n; global_missing = missing
        first = _evidence(pair, 'first_evidence', n)
        second = _evidence(pair, 'second_evidence', n)
        raw_first = _evidence(pair, 'raw_physical_first', n)
        raw_second = _evidence(pair, 'raw_physical_second', n)
        overlap = _frame_list(pair['overlap_frames'], n)
        gaps = _frame_list(pair['gaps_inside_evidence'], n)
        if not set(gaps).issubset(missing) or set(overlap) != set(first['input_frames']) & set(second['input_frames']):
            raise ValueError('Inconsistent missing or overlapping evidence.')
        if set(missing) & (set(first['input_frames']) | set(second['input_frames'])):
            raise ValueError('Unavailable images cannot be confirmed occurrence evidence.')
        is_observed = status in ('OBSERVED_A_BEFORE_B', 'OBSERVED_B_BEFORE_A')
        if is_observed:
            if not first['input_frames'] or not second['input_frames'] or gaps or overlap:
                raise ValueError('Observed order requires two distinct visible evidence sets.')
            before = first['input_frames'][-1] < second['input_frames'][0]
            after = second['input_frames'][-1] < first['input_frames'][0]
            if not (before if status == 'OBSERVED_A_BEFORE_B' else after):
                raise ValueError('Saved order disagrees with saved evidence times.')
        item = dict(first=_family_context(family_map[key[0]]), second=_family_context(family_map[key[1]]),
            observation_status=status, display_state=STATUSES[status],
            message=MESSAGES[status], limit_message=LIMIT,
            first_evidence=first, second_evidence=second,
            retained_physical_context=dict(first=raw_first, second=raw_second,
                role_qualified=False, note='기존 모양 근거이며 이것만으로 주요 동작 구간을 확정하지 않습니다.'),
            input_missing_frames=missing, gaps_inside_evidence_frames=gaps, overlap_frames=overlap,
            missing_frame_seconds=None, missing_time_note='프레임 번호만 제공된 누락 시점에 임의의 시간값을 붙이지 않습니다.',
            observed_direction='A_BEFORE_B' if status == 'OBSERVED_A_BEFORE_B' else 'B_BEFORE_A' if status == 'OBSERVED_B_BEFORE_A' else None,
            reference_order=[key[0], key[1]], scope='PARTIAL_OBSERVED_PAIR_ONLY',
            order_difference_candidate=status == 'OBSERVED_B_BEFORE_A',
            education_error=None, whole_sign_correctness=None, complete_action_verified=False,
            missing_action_verified=False, completion_unknown=True, repeat_count=None,
            repeat_count_verified=False, source_clause_verified=False)
        items.append(item)
    unobserved_queries = [list(edge) for edge in edges if edge not in seen]
    payload = dict(schema_version=1, producer='frozen_tied_readout_display_only',
        scope='PARTIAL_OBSERVED_PAIR_ONLY', items=items,
        reference_groups=[_family_context(family_map[k]) for k in groups],
        reference_graph_source=graph.get('source', ''),
        query_count=len(edges), supplied_query_count=len(items), unobserved_queries=unobserved_queries,
        comparison_frames=total_frames, global_missing_image_frames=global_missing,
        global_image_availability_reported=global_missing is not None,
        timestamp_kind='native_input_PTS',
        status='NO_DISTINCT_ORDER_QUERY' if not edges else 'PARTIAL_ORDER_REPORT',
        message='서로 다른 두 동작의 앞뒤를 비교하는 정의가 없습니다. 반복 과정과 횟수는 아직 검증하지 않았습니다.' if not edges else LIMIT,
        whole_sign_correctness=None, education_error=None, complete_action_verified=False,
        missing_action_verified=False, completion_unknown=True,
        repeated_sequence_verified=False, repeat_count_verified=False, repeat_count=None,
        source_clause_verified=False, full_span_semantic_verified=False,
        training=False, new_inference=False, evaluation_truth_used=False)
    result = deepcopy(base_result)
    result[FIELD] = payload
    return result
