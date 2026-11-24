"""Freeze metadata and static display rules only; never run a model or the adapter."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
H = U.parent
G = H.parent
O = G.parent
L = H / 'full_bag_rate_rank_learning'
REQ = O / 'word100_guide_requirement_learning_20261111'
MODES = ['full_bag_rate_rank', 'old_local_full_plus_fullbag_half']


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p):
    return {'path': str(Path(p)), 'sha256': sha(p)}


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


# These are measurement descriptors, not labels inferred from a word name.
DESCRIPTORS = {
    'shape:1': dict(task='shape', feature=1, label='오른손 네 손가락 펼침', parts=['R:fingers'], review_action='이 구간의 오른손 네 손가락 펼침을 참고 모습과 비교해 보세요.'),
    'shape:2': dict(task='shape', feature=2, label='왼손 네 손가락 펼침', parts=['L:fingers'], review_action='이 구간의 왼손 네 손가락 펼침을 참고 모습과 비교해 보세요.'),
    'shape:3': dict(task='shape', feature=3, label='오른손 네 손가락 접힘', parts=['R:fingers'], review_action='이 구간의 오른손 네 손가락 접힘을 참고 모습과 비교해 보세요.'),
    'shape:4': dict(task='shape', feature=4, label='왼손 네 손가락 접힘', parts=['L:fingers'], review_action='이 구간의 왼손 네 손가락 접힘을 참고 모습과 비교해 보세요.'),
    'direction:0': dict(task='direction', axis=0, label='손목을 아래로 내리는 방향', parts=['R:wrist', 'R:forearm'], review_action='손목을 아래로 내리는 방향을 비교해 보세요.'),
    'direction:1': dict(task='direction', axis=1, label='팔뚝을 세우는 방향', parts=['R:elbow', 'R:wrist'], review_action='팔뚝을 세우는 방향을 비교해 보세요.'),
    'direction:2': dict(task='direction', axis=2, label='반대 팔의 손목 쪽으로 이동하는 방향', parts=['R:hand', 'L:forearm'], review_action='반대 팔의 손목 쪽으로 이어가는 방향을 비교해 보세요.'),
    'direction:3': dict(task='direction', axis=3, label='양손 사이가 벌어지는 방향', parts=['R:hand', 'L:hand'], review_action='두 손을 서로 벌리는 방향을 비교해 보세요.'),
    'sequence:4': dict(task='sequence', axis=4, label='양손이 번갈아 움직이는 순서', parts=['R:hand', 'L:hand'], review_action=None),
}
# Source-backed capability declarations. They never change scores or select frames.
GAPS = {
    'open_front': [('face_relation', '얼굴 앞의 정확한 부위 관계는 아직 검사하지 못했습니다.')],
    'close_chest': [('closing_transition', '손이 실제로 닫히는 과정은 아직 검사하지 못했습니다.'), ('chest_relation', '가슴 부위 관계는 아직 검사하지 못했습니다.')],
    'raise_forearm': [('inter_action_order', '앞 동작과의 순서 및 동작 완료는 별도 확인이 필요합니다.')],
    'brush_arm': [('physical_contact', '손이 팔에 실제로 닿았는지는 확인하지 못했습니다.')],
    'reach_wrist': [('arrival_completion', '손목 도착이나 유지 시간을 완료 조건으로 판단하지 않습니다.')],
    'alternating': [('thumb_shape', '엄지 모양은 아직 검사하지 못했습니다.')],
    'gather_hands': [('fingertip_convergence', '손끝 모으기는 아직 검사하지 못했습니다.'), ('physical_contact', '손끝이 실제로 맞닿았는지는 확인하지 못했습니다.')],
    'spread_hands': [],
}


def freeze():
    if (U / 'FROZEN.json').exists():
        raise RuntimeError('Contract already frozen; preserve and amend instead of overwriting.')
    cards_path = O / 'word100_action_feedback_demo_20261108/CARDS.json'
    families_path = O / 'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json'
    audit_path = H / 'G3_CAPABILITY_SCOPE_AUDIT.json'
    sem_path = L / 'changed_span_shadow/RESULTS.json'
    rank_path = L / 'evaluation/COMPARISON.json'
    sf = read(L / 'changed_span_shadow/FROZEN.json')
    cards, families = read(cards_path)['words'], read(families_path)['words']
    audits = {(r['word'], r['card']): r for r in read(audit_path)['rows']}
    capabilities = {(r['word'], r['card'], r['feature']): r for r in read(REQ / 'GUIDE_CAPABILITY.json')['rows']}
    catalog = {}
    for word, entries in cards.items():
        catalog[word] = []
        for c in entries:
            f = next(v for v in families[word] if v['id'] == c['id'])
            a = audits[word, c['id']]
            ds = {f'shape:{j}': int(y) for j, y in c['features'].items()}
            qual = {f'shape:{j}': capabilities[word, c['id'], int(j)]['qualified'] for j in c['features']}
            if f['axis'] is not None:
                key = f"sequence:{f['axis']}" if f['sign'] is None else f"direction:{f['axis']}"
                ds[key] = None if f['sign'] is None else int(f['sign'] + 1)
            assert set(ds) <= set(DESCRIPTORS)
            gaps = [dict(component=k, text=t) for k, t in GAPS[c['id']]]
            for gap in c.get('coverage_requirements', []):
                gaps.append(dict(component=gap['component'], text=gap['label'] + '은 아직 확인하지 못했습니다.',
                                 applicability=gap['source_applicability'], requested_scope=gap['scope']))
            catalog[word].append(dict(id=c['id'], title=c['title'], instruction=c['instruction'],
                required=c['required'], family=f['families'][0], descriptors=ds, shape_qualification=qual,
                shape_scope='late_half' if c.get('shape_scope') == 'late' else 'selected_span',
                shadow_inspected=a['research_K_shadow_has_card'], unverified_components=gaps,
                exact_repetition_required=False, fixed_hold_required=False,
                guide_provenance=ref(cards_path), capability_provenance=ref(audit_path)))
    observer = ref(O / 'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt')
    sources = {}
    pins = {}
    for s in sf['sources']:
        assert s['clip_id'].startswith('NIA_SL_WORD') and s['clip_id'].split('_')[3] in ('REAL11', 'REAL12', 'REAL13')
        assert sha(s['source']) == s['sha256']
        with np.load(s['source'], allow_pickle=False) as z:
            ts = z['ts'].copy()
        assert ts.ndim == 1 and np.issubdtype(ts.dtype, np.integer) and len(ts) and (ts >= 0).all() and (np.diff(ts) > 0).all()
        pts = list(map(int, ts))
        sources[s['source']] = dict(s, word=s['clip_id'].split('_')[2], frames=len(pts),
            pts_sha256=hashlib.sha256(json.dumps(pts, separators=(',', ':')).encode('ascii')).hexdigest())
        pins[s['source']] = s['sha256']
    selectors = {}
    for mode in MODES:
        for seed in (42, 43, 44):
            old = ref(G / f'candidate_description_supervision/local_rank_learning/runs/local_full/{seed}/model.pt')
            new = ref(L / f'runs/{seed}/model.pt')
            for family in (2, 3, 4, 5, 7):
                changed_family = family in (2, 3)
                models = [new] if changed_family and mode == MODES[0] else [old, new] if changed_family else [old]
                weights = [.5, .5] if len(models) == 2 else [1.0]
                selectors[f'{mode}:{seed}:{family}'] = dict(condition=mode, seed=seed, family=family,
                    models=models, rank_logit_weights=weights, original_presence_gate_preserved=True,
                    selected_span_recomputed=False, hand_qualified_means='observation availability only; not accuracy', adopted=False)
                for model in models:
                    pins[model['path']] = model['sha256']
    files = [cards_path, families_path, audit_path, G / 'CAPABILITY_CONTRACT.json', sem_path, rank_path,
        L / 'changed_span_shadow/FROZEN.json', L / 'changed_span_shadow/VERIFICATION.json',
        L / 'evaluation/FROZEN.json', L / 'evaluation/EXECUTION_FROZEN.json', L / 'shadow_components.py',
        REQ / 'requirement_core.py', REQ / 'GUIDE_CAPABILITY.json', G / 'PROTECTIONS.json',
        U / 'adapter.py', U / 'prepare_contract.py', U / 'offline_fixture.py', U / 'verify_offline.py', U / 'README.txt']
    for p in files:
        pins[str(p)] = sha(p)
    pins[observer['path']] = observer['sha256']
    for p in U.glob('*.py'):
        ast.parse(p.read_text('utf-8'))
    contract = dict(schema='g3_explicit_evidence_contract_v1', frozen_at=time.time(),
        scope='Pure research display adapter; unadopted saved L results; not web/runtime validation or accuracy evaluation.',
        cards=catalog, descriptors=DESCRIPTORS, sources=sources, selectors=selectors, seeds=[42, 43, 44], conditions=MODES,
        observer=observer, semantic_result=ref(sem_path), selection_result=ref(rank_path), pins=pins,
        rules={'phase_reselection': False, 'shape_scope': 'exact selected interval or existing late half; never late third',
               'completion': None, 'whole_sign_correctness': None, 'qualified_is_accuracy': False,
               'reference_spans_are_user_spans': False, 'sequence_required_is_supported': False,
               'four_open_low_is_fist': False, 'unobserved_is_wrong': False,
               'semantic_thresholds': 'Unchanged saved .65/.35; verification only; no threshold selection.',
               'PTS': 'integer microseconds from pinned observation NPZ; no original-video mapping implied'},
        validation_plan={'saved_inputs': '82 scenarios x 3 selector seeds x 2 fixed conditions, no inference',
            'negative_contracts': ['nonmonotonic/float/foreign PTS', 'out-of-bounds/reversed span', 'foreign source/model hash',
                'late-third or phase reselection', 'unqualified-as-supported', 'masked-as-difference',
                'sequence-as-supported', 'missing provenance', 'whole-sign completion'],
            'display_counterexamples': ['open-low does not name fist', 'sequence unknown without temporal highlight',
                'missing cards unknown', 'supported subset with unsupported components', 'no completion promotion']},
        no_model_forward=True, no_training=True, no_web=True, no_new_pose=True, no_final8=True, no_private=True,
        G2_pass=None, G3_pass=None, tests_executed=False)
    write(U / 'FROZEN.json', contract)
    print(json.dumps({'status': 'FROZEN_AWAITING_ROOT_CODE_REVIEW', 'cards': sum(map(len, catalog.values())), 'sources': len(sources), 'pins': len(pins)}))


if __name__ == '__main__':
    freeze()
