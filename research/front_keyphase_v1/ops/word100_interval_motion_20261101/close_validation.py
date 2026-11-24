"""Close the failed readiness check without changing labels, gates or model scores."""
from pathlib import Path
import json, hashlib, datetime

W = Path(__file__).resolve().parent
O = W.parent
ROOT = O.parents[2]
V = O / 'word100_center_motion_pilot_20261101'
U = O / 'word100_core_feature_split_20261101'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write_new(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def replace_with_backup(p, obj):
    p = Path(p)
    backup = W / 'before' / ('validation_close_' + p.name)
    if backup.exists():
        raise FileExistsError(backup)
    backup.write_bytes(p.read_bytes())
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    audit = read(W / 'PRESENCE_PILOT_AUDIT.json')
    verification = read(W / 'review/PRESENCE_VERIFICATION.json')
    assert audit['status'] == 'REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING'
    assert verification['status'] == 'PASS_SEPARATE_PRESENCE_DECODER_AND_IMMUTABILITY'
    assert verification['checks_count'] == 425 and not verification['failures']
    assert not audit['training_allowed']
    for p, h in verification['source_pins'].items():
        assert sha(p) == h, p
    protected = read(U / 'SELECTION.json')['protected']
    for p, h in protected.items():
        assert sha(p) == h, p
    for folder in (V, W):
        assert not (folder / 'TRAIN_FROZEN.json').exists()
        assert not (folder / 'runs').exists()
    a = read(W / 'review/PRESENCE_A.json')
    b = read(W / 'review/PRESENCE_B.json')
    assert a['status'] == b['status'] == 'COMPLETE'
    assert len(a['rows']) == len(b['rows']) == 24
    # A simple root recount independent of both audit implementations.
    ar = {r['sample_id']: r for r in a['rows']}
    br = {r['sample_id']: r for r in b['rows']}
    recount = {}
    mismatch_ids = []
    for hand in ('R', 'L'):
        for pred in ('down', 'hold'):
            key = hand + '_' + pred
            c = dict(both_known=0, both_unknown=0, known_unknown_mismatch=0,
                     known_conflicts=0, agreed_positive=0, agreed_negative=0)
            for sid in ar:
                x, y = ar[sid]['presence'][hand][pred], br[sid]['presence'][hand][pred]
                if x is None and y is None:
                    c['both_unknown'] += 1
                elif x is None or y is None:
                    c['known_unknown_mismatch'] += 1
                    if key == 'L_down':
                        mismatch_ids.append(sid)
                else:
                    c['both_known'] += 1
                    if x != y:
                        c['known_conflicts'] += 1
                    else:
                        c['agreed_positive' if x == 1 else 'agreed_negative'] += 1
            assert all(audit['metrics'][key][k] == n for k, n in c.items()), key
            recount[key] = c
    assert sum(r['both_known'] for r in recount.values()) == 77
    assert sum(r['known_unknown_mismatch'] for r in recount.values()) == 13
    assert sum(r['both_unknown'] for r in recount.values()) == 6
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    result = dict(status='VALIDATION_COMPLETE_TRAINING_NOT_READY', at=now,
        goal_completed=False, windows=24, native_frames_per_window=13, people=3, expressions=15,
        physical_predicate_opportunities=96, both_known_agree=77,
        one_reviewer_unknown=13, both_reviewers_unknown=6,
        counts=recount, failed_checks=[k for k,v in audit['checks'].items() if not v],
        left_down_uncertainty_mismatch_samples=mismatch_ids,
        completed=['24 original public RGB windows read by two fresh-context Codex reviewers',
                   'ordered direct down/hold intervals retained independently',
                   'normal hold, crossing, rotation and brief-change examples retained',
                   'first centre-state and interval-presence failures preserved',
                   '425-check separate-decoder audit plus root raw recount'],
        blockers=['No agreed hold-absent cases for either hand in the pilot; binary learner lacks counterexamples.',
                  'Left down known-versus-unknown differs in6/24 windows; cannot dilute with easy cases or convert unknown to0.'],
        limitations=['Same-Codex reproducibility is not independent human/KSL ground truth.',
                     'Occurrence labels do not establish correct event order or educational correctness.',
                     'Normal/crossing/brief-change coverage is not a learned non-regression result.',
                     'Centre-state and presence targets differ; their known counts are not a model-performance comparison.'],
        training_runs=0, actual_model_inference_runs=0, completed_runs_rerun=0,
        model_performance_improvement=None, html_created=False,
        original_input_reextraction=0, protected_hashes_verified=len(protected),
        private_recording_access=0, production_changed=False, active_execution=None,
        next_action='Fit-only bounded diagnosis: enlarge fixed torso/forearm RGB evidence at the same timestamps for the six left-down ambiguities; independently inventory genuine continuous-motion windows for hold-negative support. Keep the failed24-window test unchanged; no broad288 relabel or training before a separately reviewed readiness repair.',
        source_pins={str(p):sha(p) for p in [W/'PRESENCE_PILOT_AUDIT.json', W/'review/PRESENCE_VERIFICATION.json',
            W/'review/PRESENCE_A.json', W/'review/PRESENCE_B.json', W/'RUBRIC.json', W/'PILOT_GATES.json',
            W/'PILOT_FROZEN.json', W/'AUDIT_CODE_FROZEN.json', V/'PILOT_AUDIT.json', U/'DATASET.json', Path(__file__)]})
    write_new(W/'COMPLETION.json',result)
    report = '''움직임 정답 기준 검증 결과 — 2026-11-01

검증은 마쳤지만 학습 준비는 통과하지 못했습니다. 새 학습·모델 성능 비교는 실행하지 않았습니다.

한 구간에서 손을 유지하다 내려가면, 둘 중 하나로 뭉뚱그리지 않고 실제로 보인 유지 구간과 내려간 구간을 각각 기록하도록 보완했습니다. 정상 손 교차·회전·짧은 변화도 지우지 않았고, 안 보이는 부분을 ‘움직임 없음’으로 바꾸지 않았습니다.

공개15표현·수행자ID3개의24장면을 두 새 컨텍스트 Codex 검토자가 각각 원본13프레임으로 확인했습니다. 같은 자료24개를 두 번 보았다는 뜻이며 독립적인 사람 전문가 정답은 아닙니다.

96판단항목 중77개는 둘 다 확인 가능하다고 판단하고 같은 답을 냈습니다.13개는 한쪽만 확정했고6개는 둘 다 보류했습니다. 이것은 모델 정확도가 아닙니다.
오른손 내려감:15 있음/4 없음/나머지5 미합의·보류.
왼손 내려감:8 있음/8 없음/나머지8 미합의·보류.
잠깐 유지하는 구간:양손 각각21 있음/0 없음/나머지3 미합의·보류.

남은 두 문제:
1. 유지 구간이 ‘없다’고 확실히 확인한 반대 사례가 없습니다. 이 상태로 학습하면 항상 ‘있다’고 답해도 좋아 보일 수 있습니다.
2. 왼손의 작은 내려감을 확정할 수 있는지6/24장면에서 차이가 남았습니다.주로 작은 위치 변화와 회전·자세 흔들림을 구분하는 문제입니다. 이를 정상 또는 오류로 강제하지 않았습니다.

앞서 중심5프레임에 전환이 있으면 전부 보류하던 기준은 움직임을 지나치게 버려 실패했습니다. 이번 구간별 기록은 그 정보를 남기지만, 두 기준의 과제와 분모가 다르므로 정확도가 높아졌다고 비교할 수 없습니다. 현재77개 일치는 세부 동작 학습의 준비 근거이며 수어 순서 이해나 피드백 완성을 뜻하지 않습니다.

별도 계산기로425항목을 검산하고 root도 원시 답을 다시 집계했습니다. 최초 실패와 미확인 항목은 보존했습니다. 정상 손·교차 예시는 포함했지만 새 모델 비회귀 성능은 아직 검증하지 않았습니다.

다음은 전체 자료를 또 주석 처리하거나 학습하는 대신, 왼손6장면의 같은 시점을 고정된 큰 몸통·팔 화면으로 확인하고, 계속 움직여 실제 유지 구간이 없는 학습용 사례의 지원 여부만 좁혀 확인하는 것입니다. 애매한 것을 억지로 정답으로 정하거나 쉬운 사례를 추가해서 실패 비율만 낮추지 않습니다.

기존 C·T3-F, 운영 웹·DTW, 기존 입력·저장 점수는 보존했습니다. 사용자·멘토 촬영에 접근하거나 학습·튜닝에 쓰지 않았습니다. 새 학습0회, 새 모델 추론0회, 재채점0회입니다. 명확한 모델 성능 개선이 없어 HTML은 만들지 않았습니다.
'''
    with (W/'RESULT_KO.txt').open('x',encoding='utf-8') as f:
        f.write(report)
    for folder in (W, V):
        old = folder/'CURRENT_EXECUTION.json'
        backup = W/'before'/('validation_close_'+folder.name+'_CURRENT_EXECUTION.json')
        assert not backup.exists()
        backup.write_bytes(old.read_bytes())
        old.write_text(json.dumps(dict(status='VALIDATION_COMPLETE_TRAINING_NOT_READY',
            at=now, active_execution=None, active_agents=[],training_started=False,
            result=str(folder/'PRESENCE_PILOT_AUDIT.json' if folder==W else folder/'PILOT_AUDIT.json')),
            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'], protected=len(protected),both_known=77,
        unresolved=19, training=0, report=str(W/'RESULT_KO.txt')),ensure_ascii=False))

if __name__ == '__main__':
    main()
