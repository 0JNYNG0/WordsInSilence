"""One frozen offline contract test. Stored data only; no learned code imports."""
from pathlib import Path
import collections
import copy
import hashlib
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def key(row):
    return row['seed'], row['clip_id'], row['kind']


def run():
    assert sys.argv[1:] == ['run-reviewed'], 'Root review required before execution.'
    assert not (U / 'TEST_STARTED.json').exists(), 'No repeated execution; preserve result/failure.'
    cfg = read(U / 'FROZEN.json')
    for path, expected in cfg['pins'].items():
        assert sha(path) == expected, path
    protections = read(U.parent.parent / 'PROTECTIONS.json')
    for path, expected in protections.items():
        assert sha(path) == expected, path
    write(U / 'TEST_STARTED.json', dict(at=time.time(), frozen_sha256=sha(U / 'FROZEN.json'), no_model_forward=True))
    from adapter import adapt, ContractError
    from offline_fixture import request_from_saved
    saved = read(cfg['semantic_result']['path'])
    selected = {key(r): r for r in read(cfg['selection_result']['path'])['rows']}
    checks = 0
    def check(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(label)
    check(len(saved['rows']) == len(selected) == 246, 'saved row denominator')
    cache, requests, payloads, states, categories = {}, [], [], collections.Counter(), collections.Counter()
    examples = {}
    inspected_cards = absent_cards = 0
    for row in saved['rows']:
        for condition in cfg['conditions']:
            req = request_from_saved(row, selected[key(row)], condition, cfg, cache)
            before = copy.deepcopy(req)
            out = adapt(req, cfg)
            check(before == req, 'pure adapter mutated input')
            check(out['completion'] is None and out['whole_sign_correctness'] is None, 'whole-sign promotion')
            check(out['research_only'] and not out['web_connected'] and not out['selector_adopted'], 'research status')
            check(out['source'] == req['source'], 'source/PTS changed')
            check(len(out['cards']) == len(cfg['cards'][req['word']]), 'guide cards silently omitted')
            byid = {c['id']: c for c in out['cards']}
            supplied = {c['id']: c for c in req['cards']}
            for spec in cfg['cards'][req['word']]:
                actual = byid[spec['id']]
                check(actual['required'] == spec['required'], 'optional action became required')
                check(actual['completion'] is None and actual['whole_action_correctness'] is None, 'card completion promoted')
                check(not actual['performance_certified'] and not actual['availability_is_accuracy'], 'qualification promoted to accuracy')
                if spec['id'] not in supplied:
                    absent_cards += 1
                    check(actual['selected_span'] is None and actual['selected_pts_us'] is None, 'reference span fabricated')
                    check(all(i['state'] == 'not_checked' for i in actual['items']), 'absent evidence invented')
                    check('unsupported' in actual['categories'], 'absent card lacks unsupported status')
                else:
                    inspected_cards += 1
                    src = supplied[spec['id']]
                    check(actual['selected_span'] == src['selected_span'], 'phase reselected')
                    bydescriptor = {i['descriptor']: i for i in actual['items']}
                    for inp in src['inspections']:
                        item = bydescriptor[inp['descriptor']]
                        check(item['evidence'] == inp and item['state'] == inp['state'], 'saved evidence changed')
                        check(item['inspection_span'] == inp['inspection_span'], 'actual inspection scope changed')
                        check(item['highlight']['span'] == inp['inspection_span'], 'highlight exceeds inspected scope')
                        if inp['inspection_span'] is not None:
                            a, b = inp['inspection_span']
                            check(item['inspection_pts_us'] == [req['source']['pts_us'][a], req['source']['pts_us'][b]], 'wrong evidence timestamp')
                        else:
                            check(item['inspection_pts_us'] is None, 'uncomputed sequence time invented')
                        states[item['state']] += 1
                        categories[item['category']] += 1
                        if item['state'] == 'observation_unknown':
                            check(item['category'] == 'observation_unknown' and item['style']['color'] != '#B45309', 'masked evidence marked wrong')
                        if item['state'] == 'sequence_required':
                            check(item['category'] == 'unsupported' and item['inspection_span'] is None and item['review_action'] is None, 'alternation promoted')
                        if item['state'] == 'association_unknown':
                            check(item['category'] == 'association_unknown', 'unknown association ignored')
                        if item['descriptor'] in ('shape:1', 'shape:2') and item['state'] == 'difference':
                            check('주먹' not in item['text'] + (item['review_action'] or ''), 'low open invented fist')
                        if item['state'] not in examples:
                            examples[item['state']] = copy.deepcopy(out)
            if row['kind'] == 'all_hand_missing':
                for card in out['cards']:
                    for item in card['items']:
                        if item['evidence'] and item['descriptor'].startswith('shape:'):
                            check(item['state'] == 'observation_unknown', 'allmissing shape passed')
            requests.append(req)
            payloads.append(out)
    check(len(requests) == 492 and len(cache) == 82 and inspected_cards == 636, 'fixed data size changed')
    check(set(states) == {'supported', 'difference', 'opposite_direction', 'observation_unknown',
        'association_unknown', 'uncertain', 'movement_not_established', 'sequence_required'}, 'missing saved branch coverage')

    # Fixed schema/misrepresentation counterexamples, not model/threshold experiments.
    base = next(r for r in requests if any(c['id'] == 'close_chest' and c['selected_span'] for c in r['cards']))
    malformed = []
    def rejected(name, original, change):
        req = copy.deepcopy(original)
        change(req)
        try:
            adapt(req, cfg)
        except ContractError as exc:
            malformed.append(dict(test=name, rejected=True, reason=str(exc)))
        else:
            raise AssertionError('accepted invalid request: ' + name)
    rejected('PTS repeated', base, lambda r: r['source']['pts_us'].__setitem__(1, r['source']['pts_us'][0]))
    rejected('PTS fractional', base, lambda r: r['source']['pts_us'].__setitem__(1, 1.5))
    rejected('PTS foreign monotonic', base, lambda r: r['source'].__setitem__('pts_us', [x + 1 for x in r['source']['pts_us']]))
    rejected('wrong source hash', base, lambda r: r['source'].__setitem__('sha256', '0' * 64))
    rejected('wrong source identity', base, lambda r: r['source'].__setitem__('id', 'unregistered'))
    rejected('span out of bounds', base, lambda r: r['cards'][0].__setitem__('selected_span', [0, len(r['source']['pts_us'])]))
    rejected('span reversed', base, lambda r: r['cards'][0].__setitem__('selected_span', [9, 2]))
    rejected('span floating', base, lambda r: r['cards'][0].__setitem__('selected_span', [0.0, 2]))
    rejected('late scope drift', base, lambda r: r['cards'][0]['inspections'][0]['inspection_span'].__setitem__(0, r['cards'][0]['inspections'][0]['inspection_span'][0] + 1))
    rejected('foreign inspection source', base, lambda r: r['cards'][0]['inspections'][0].__setitem__('source_sha256', '0' * 64))
    rejected('foreign observer', base, lambda r: r['cards'][0]['inspections'][0]['observer'].__setitem__('sha256', '0' * 64))
    rejected('NaN probability', base, lambda r: r['cards'][0]['inspections'][0].__setitem__('probability', float('nan')))
    rejected('probability bool', base, lambda r: r['cards'][0]['inspections'][0].__setitem__('probability', True))
    rejected('whole-sign promotion', base, lambda r: r.__setitem__('completion', True))
    rejected('card completion promotion', base, lambda r: r['cards'][0].__setitem__('completion', True))
    rejected('missing provenance', base, lambda r: r.pop('semantic_result'))
    rejected('duplicate card', base, lambda r: r['cards'].append(copy.deepcopy(r['cards'][0])))
    rejected('wrong selector profile', base, lambda r: r['cards'][0].__setitem__('selector_profile', 'anything'))
    rejected('phase reselect with old inspection', base, lambda r: r['cards'][0]['selected_span'].__setitem__(0, r['cards'][0]['selected_span'][0] - 1))
    masked = next(r for r in requests if any(i['state'] == 'observation_unknown' for c in r['cards'] for i in c['inspections']))
    def mask_to_difference(r):
        next(i for c in r['cards'] for i in c['inspections'] if i['state'] == 'observation_unknown')['state'] = 'difference'
    rejected('missing observation as error', masked, mask_to_difference)
    unqualified = next(r for r in requests if any(i['state'] == 'association_unknown' for c in r['cards'] for i in c['inspections']))
    def unknown_to_support(r):
        next(i for c in r['cards'] for i in c['inspections'] if i['state'] == 'association_unknown')['state'] = 'supported'
    rejected('unqualified association promoted', unqualified, unknown_to_support)
    sequence = next(r for r in requests if any(i['state'] == 'sequence_required' for c in r['cards'] for i in c['inspections']))
    def sequence_to_support(r):
        next(i for c in r['cards'] for i in c['inspections'] if i['state'] == 'sequence_required')['state'] = 'supported'
    rejected('uncomputed alternation promoted', sequence, sequence_to_support)
    partial_coverage = copy.deepcopy(base)
    shape = partial_coverage['cards'][0]['inspections'][0]
    n = shape['inspection_span'][1] - shape['inspection_span'][0] + 1
    shape.update(observed_count=1, observation_eligible=False, state='observation_unknown')
    check(n > 1 and adapt(partial_coverage, cfg)['cards'][1]['items'][0]['category'] == 'observation_unknown', 'some visible frames do not bypass original .8 rule')
    rejected('partial coverage promoted', partial_coverage, lambda r: r['cards'][0]['inspections'][0].update(observation_eligible=True, state='supported'))

    # Missing feature evidence remains unknown instead of copying another feature's support.
    incomplete = copy.deepcopy(base)
    incomplete['cards'][0]['inspections'].pop(0)
    inc = adapt(incomplete, cfg)
    check(inc['cards'][1]['items'][0]['state'] == 'not_checked', 'missing feature replaced by another feature')
    check(all(c['completion'] is None for c in inc['cards']), 'incomplete record marked complete')
    check('torch' not in sys.modules and 'onnxruntime' not in sys.modules, 'model runtime imported')
    for path, expected in cfg['pins'].items():
        check(sha(path) == expected, 'frozen input changed: ' + path)
    for path, expected in protections.items():
        check(sha(path) == expected, 'protected original/web changed: ' + path)
    write(U / 'EXAMPLE_PAYLOADS.json', dict(scope='One saved example per state for root inspection, not outcome-selected model adoption.', examples=examples))
    write(U / 'PAYLOADS.json', dict(scope='Fixed offline L saved results; not live webcam or educational validation.', rows=payloads))
    write(U / 'VERIFICATION.json', dict(at=time.time(), status='OFFLINE_PAYLOAD_CONTRACT_VERIFIED_ONLY', checks=checks,
        schema_rejection_tests=malformed, requests=len(requests), source_scenarios=len(cache), selector_seeds=cfg['seeds'],
        saved_inspected_cards=inspected_cards, absent_cards_retained_unknown=absent_cards,
        saved_semantic_states=dict(states), mapped_categories=dict(categories),
        no_model_forward=True, no_training=True, no_new_scores=True, no_phase_reselection=True,
        protected_files_unchanged=len(protections), no_web=True, no_final8=True, no_private=True,
        completion=None, whole_sign_correctness=None, G2_pass=None, G3_pass=None,
        limitation='Pure presentation plumbing and semantic consistency only; not accuracy or real-user feedback validation.',
        frozen_sha256=sha(U / 'FROZEN.json'), payload_sha256=sha(U / 'PAYLOADS.json')))
    print(json.dumps({'checks': checks, 'schema_rejections': len(malformed), 'requests': len(requests), 'saved_cards': inspected_cards, 'unknown_cards': absent_cards}))


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        (U / f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(), 'utf-8')
        raise
