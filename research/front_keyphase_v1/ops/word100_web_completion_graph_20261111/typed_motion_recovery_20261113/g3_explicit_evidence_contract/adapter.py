"""Pure display adapter for already-computed, explicitly scoped evidence.

No file access, model execution, phase selection, DTW, or educational verdict.
The caller verifies source bytes against the frozen registry before calling.
"""
import copy
import math
import re


class ContractError(ValueError):
    pass


STYLE = {
    'supported_features': {'color': '#243B53', 'pattern': 'solid', 'label': '확인된 특징'},
    'difference_evidence': {'color': '#B45309', 'pattern': 'outline', 'label': '차이 근거'},
    'observation_unknown': {'color': '#64748B', 'pattern': 'dotted', 'label': '관측 부족'},
    'association_unknown': {'color': '#64748B', 'pattern': 'dashed', 'label': '연결 불확실'},
    'uncertain_evidence': {'color': '#64748B', 'pattern': 'dashed', 'label': '판단 보류'},
    'unsupported': {'color': '#64748B', 'pattern': 'none', 'label': '미검사·미지원'},
}
STATE_CATEGORY = {
    'supported': 'supported_features', 'difference': 'difference_evidence',
    'opposite_direction': 'difference_evidence', 'observation_unknown': 'observation_unknown',
    'association_unknown': 'association_unknown', 'uncertain': 'uncertain_evidence',
    'movement_not_established': 'uncertain_evidence', 'sequence_required': 'unsupported',
    'not_checked': 'unsupported',
}


def require(condition, message):
    if not condition:
        raise ContractError(message)


def integer(value):
    return type(value) is int


def sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def span(value, n, nullable=False):
    if value is None:
        require(nullable, 'missing inspection span')
        return None
    require(isinstance(value, list) and len(value) == 2 and all(integer(v) for v in value), 'span schema')
    require(0 <= value[0] <= value[1] < n, 'span outside source or reversed')
    return list(value)


def probability(value):
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1, 'invalid probability')
    return value


def artifact(value, contract):
    require(isinstance(value, dict) and isinstance(value.get('path'), str), 'artifact schema')
    require(sha256(value.get('sha256')), 'artifact hash schema')
    require(contract['pins'].get(value['path']) == value['sha256'], 'artifact not in frozen registry')


def pts_interval(value, pts):
    return None if value is None else [pts[value[0]], pts[value[1]]]


def shape_state(item):
    if not item['observation_eligible']:
        return 'observation_unknown'
    if not item['association_qualified']:
        return 'association_unknown'
    p = item['probability']
    return 'supported' if p >= .65 else 'difference' if p <= .35 else 'uncertain'


def direction_state(item):
    if not item['observation_eligible']:
        return 'observation_unknown'
    p, desired = item['probabilities'], item['desired']
    if desired == 1:
        return 'supported' if p[1] >= .65 and max(p[0], p[2]) < .65 else 'uncertain'
    opposite = 2 - desired
    if p[desired] >= .65 and p[opposite] < .65 and p[1] < .65:
        return 'supported'
    if p[opposite] >= .65 and p[desired] < .35 and p[1] < .65:
        return 'opposite_direction'
    return 'movement_not_established' if p[1] >= .65 else 'uncertain'


def message(descriptor, state):
    label = descriptor['label']
    if state == 'supported':
        return label + ' 특징이 보입니다.', None
    if state in ('difference', 'opposite_direction'):
        # A low open-hand score is NOT evidence of a fist or any alternative shape.
        text = label + ('과 반대 방향의 움직임 근거가 있습니다.' if state == 'opposite_direction' else ' 특징과 차이 근거가 있습니다.')
        return text, descriptor['review_action']
    if state == 'observation_unknown':
        return label + '을 확인할 관측이 부족합니다.', '강조한 부위가 보이도록 다시 확인해 주세요.'
    if state == 'association_unknown':
        return label + '을 이 설명과 연결할 근거가 부족합니다.', None
    if state == 'movement_not_established':
        return label + ' 움직임이 뚜렷하지 않아 판단을 보류합니다.', None
    if state == 'uncertain':
        return label + ' 판단이 아직 불확실합니다.', None
    return label + '은 아직 검사하지 못했습니다.', None


def _adapt(request, contract):
    """Return immutable-in-meaning payload; input objects are never modified.

    An inclusive frame interval is two observed timestamps, NOT a video duration
    or a claim that every internal frame passed. The exact inspection interval
    and eligible count remain distinct from the selected action interval.
    """
    require(isinstance(request, dict) and request.get('schema') == 'g3_explicit_evidence_v1', 'request schema')
    require(request.get('completion') is None and request.get('whole_sign_correctness') is None, 'educational verdict forbidden')
    require(request.get('research_only') is True, 'research scope required')
    source = request['source']
    require(source.get('artifact_kind') == 'observation_npz', 'source hash must identify observation artifact')
    require(source.get('frame_indexing') == 'decoded_observation_0_based', 'frame indexing mismatch')
    artifact(source, contract)
    require(source['path'] in contract['sources'], 'source outside frozen public manifest')
    expected_source = contract['sources'][source['path']]
    require(source.get('id') == expected_source['clip_id'] and source.get('scenario') == expected_source['kind'], 'source identity mismatch')
    pts = source['pts_us']
    require(isinstance(pts, list) and len(pts) > 0 and all(integer(t) and t >= 0 for t in pts), 'PTS must be integer microseconds')
    require(all(a < b for a, b in zip(pts, pts[1:])), 'PTS must be strictly increasing')
    # The digest of the PTS list is checked without I/O; stale/different timelines cannot be attached.
    import hashlib
    import json
    digest = hashlib.sha256(json.dumps(pts, separators=(',', ':')).encode('ascii')).hexdigest()
    require(digest == expected_source['pts_sha256'], 'PTS not from pinned source')
    word = request['word']
    require(word in contract['cards'] and word == expected_source['word'], 'target/source word mismatch')
    require(type(request.get('seed')) is int and request['seed'] in contract['seeds'], 'selector seed')
    require(request.get('condition') in contract['conditions'], 'selector condition')
    artifact(request['semantic_result'], contract)
    require(request['semantic_result'] == contract['semantic_result'], 'semantic result provenance mismatch')
    artifact(request['selection_result'], contract)
    require(request['selection_result'] == contract['selection_result'], 'selection result provenance mismatch')
    observed = request['cards']
    require(isinstance(observed, list) and all(isinstance(c, dict) for c in observed), 'cards schema')
    ids = [c.get('id') for c in observed]
    require(len(ids) == len(set(ids)), 'duplicate card')
    specs = contract['cards'][word]
    require(set(ids) <= {c['id'] for c in specs}, 'unregistered card')
    supplied = {c['id']: c for c in observed}
    cards = []
    for spec in specs:
        c = supplied.get(spec['id'])
        selected = None
        items = {}
        selection = None
        legacy_status = None
        if c is not None:
            require(c.get('completion') is None, 'card completion forbidden')
            require(spec['shadow_inspected'], 'card lacks frozen observer scope')
            selected = span(c['selected_span'], len(pts), nullable=True)
            legacy_status = c.get('saved_status')
            profile = f"{request['condition']}:{request['seed']}:{spec['family']}"
            require(c.get('selector_profile') == profile, 'selector profile mismatch')
            selection = copy.deepcopy(contract['selectors'][profile])
            for a in selection['models']:
                artifact(a, contract)
            require(isinstance(c.get('inspections'), list), 'inspection schema')
            for item in c['inspections']:
                key = item.get('descriptor')
                require(key in spec['descriptors'] and key not in items, 'unknown/duplicate descriptor')
                require(selected is not None, 'inspection without selected span')
                require(item.get('source_sha256') == source['sha256'], 'inspection source mismatch')
                artifact(item['observer'], contract)
                require(item['observer'] == contract['observer'], 'observer identity mismatch')
                state = item.get('state')
                require(state in STATE_CATEGORY, 'unknown semantic state')
                descriptor = contract['descriptors'][key]
                scope = descriptor['task']
                inspected = span(item.get('inspection_span'), len(pts), nullable=scope == 'sequence')
                if scope == 'sequence':
                    require(state == 'sequence_required' and inspected is None, 'sequence has no inspected evidence in this frozen observer')
                    require(item.get('probabilities') is None, 'uncomputed sequence probability')
                else:
                    a, b = selected
                    expected_span = [a + (b - a) // 2, b] if scope == 'shape' and spec['shape_scope'] == 'late_half' else [a, b]
                    require(inspected == expected_span, 'inspection scope changed or phase reselected')
                    require(type(item.get('observation_eligible')) is bool, 'missing observation eligibility')
                    if scope == 'shape':
                        require(integer(item.get('feature')) and integer(item.get('expected')) and item['feature'] == descriptor['feature'] and item['expected'] == spec['descriptors'][key], 'shape identity mismatch')
                        require(type(item.get('association_qualified')) is bool, 'association qualification schema')
                        require(item['association_qualified'] == spec['shape_qualification'][key], 'association qualification altered')
                        count = item.get('observed_count')
                        require(integer(count) and 0 <= count <= inspected[1] - inspected[0] + 1, 'observed count outside inspected range')
                        require(item['observation_eligible'] == (count / (inspected[1] - inspected[0] + 1) >= .8), 'observability conflicts with frozen .8 coverage rule')
                        if count:
                            probability(item.get('probability'))
                        else:
                            require(item.get('probability') is None, 'unobserved shape must not carry probability')
                        require(state == shape_state(item), 'shape state contradicts frozen semantic rule')
                    else:
                        require(integer(item.get('axis')) and integer(item.get('desired')) and item['axis'] == descriptor['axis'] and item['desired'] == spec['descriptors'][key], 'direction identity mismatch')
                        require(isinstance(item.get('probabilities'), list) and len(item['probabilities']) == 3, 'direction probability schema')
                        for p in item['probabilities']:
                            probability(p)
                        require(state == direction_state(item), 'direction state contradicts frozen semantic rule')
                items[key] = item
        output_items = []
        for key in spec['descriptors']:
            descriptor = contract['descriptors'][key]
            item = items.get(key)
            state = item['state'] if item else 'association_unknown' if c is not None and selected is None else 'not_checked'
            category = STATE_CATEGORY[state]
            text, action = message(descriptor, state)
            inspected = item['inspection_span'] if item else None
            output_items.append({
                'descriptor': key, 'label': descriptor['label'], 'state': state, 'category': category,
                'style': dict(STYLE[category]), 'text': text, 'review_action': action,
                'inspection_span': copy.deepcopy(inspected), 'inspection_pts_us': pts_interval(inspected, pts),
                'highlight': {'parts': descriptor['parts'], 'span': copy.deepcopy(inspected), 'extent': 'inspection_range_only'},
                'evidence': copy.deepcopy(item), 'performance_certified': False,
            })
        gaps = []
        for gap in spec['unverified_components']:
            gaps.append(dict(gap, state='not_checked', category='unsupported', style=dict(STYLE['unsupported']),
                             inspection_span=None, inspection_pts_us=None, completion=None))
        categories = sorted({i['category'] for i in output_items} | ({'unsupported'} if gaps else set()))
        cards.append({
            'id': spec['id'], 'guide_title': spec['title'], 'guide_text': spec['instruction'],
            'guide_text_is_full_verdict': False, 'required': spec['required'],
            'selected_span': selected, 'selected_pts_us': pts_interval(selected, pts), 'shape_scope': spec['shape_scope'],
            'selection_provenance': selection, 'saved_status_not_full_verdict': legacy_status,
            'items': output_items, 'unverified_components': gaps, 'categories': categories,
            'completion': None, 'whole_action_correctness': None, 'performance_certified': False,
            'availability_is_accuracy': False, 'selection_available': selected is not None,
        })
    return {
        'schema': 'g3_display_payload_v1', 'source': copy.deepcopy(source), 'word': word,
        'condition': request['condition'], 'seed': request['seed'], 'cards': cards,
        'semantic_result': copy.deepcopy(request['semantic_result']),
        'selection_result': copy.deepcopy(request['selection_result']),
        'observer': copy.deepcopy(contract['observer']), 'research_only': True,
        'web_connected': False, 'selector_adopted': False, 'G2_pass': None, 'G3_pass': None,
        'completion': None, 'whole_sign_correctness': None, 'performance_certified': False,
        'timestamp_semantics': 'Inclusive observed endpoints; not interval duration, all-frame correctness, or original-video timestamps.',
    }


def adapt(request, contract):
    try:
        return _adapt(request, contract)
    except (KeyError, TypeError, IndexError) as exc:
        raise ContractError('Malformed or incomplete evidence schema') from exc
