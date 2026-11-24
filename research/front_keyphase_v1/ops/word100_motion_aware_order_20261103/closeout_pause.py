"""Record the user-requested pause after cache/input preparation only.

No media/model imports, inference, training, downloads or service mutation.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
O = U.parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def new_json(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    assert ROOT.name == 'WordsInSilence'
    assert not (U / 'PAUSE_CLOSEOUT.json').exists()
    plan = read(U / 'INPUT_PLAN.json')
    compatibility = read(U / 'INPUT_COMPATIBILITY.json')
    review_path = U / 'INPUT_REVIEW.json'
    review = read(review_path) if review_path.exists() else None
    assert compatibility['status'].startswith('PASS')
    if review is not None:
        assert review['status'].startswith('PASS'), review['status']
    synthetic_path = U / 'SYNTHETIC_CHECKS_v3.json'
    assert synthetic_path.exists()
    synthetic = read(synthetic_path)
    assert synthetic['status'] == 'PASS'
    ah_final = read(O / 'word100_local_event_learning_20261103' / 'FINAL.json')
    assert ah_final['dev_native_positive_high'] == [15, 15]
    assert ah_final['dev_additional_local_positive_high'] == [2, 2]
    assert ah_final['dev_local_negative_false_high'] == [0, 13]
    for name in ['REAL_PREFLIGHT.json', 'FROZEN.json', 'TRAIN_STARTED.json',
                 'TRAIN_RESULT.json', 'AUXILIARY_PREDICTIONS.json', 'EVALUATION.json']:
        assert not (U / name).exists(), 'Unexpected execution artifact: ' + name
    assert not (U / 'runs').exists()
    protocol = read(U / 'EVALUATION_PROTOCOL.json')
    assert protocol['training_allowed'] is False
    for path, digest in plan['pins'].items():
        assert sha(path) == digest, 'Preparation input changed: ' + path
    now = datetime.now(timezone(timedelta(hours=9))).isoformat()
    fit, dev = plan['denominators']['fit'], plan['denominators']['dev']
    missing = plan['new_AH_forward_request']['total']
    native = plan['new_AH_forward_request']['native_distinct_Y_snapshots']
    reused = plan['reused_saved_predictions']
    review_sentence = ('별도 에이전트의 입력 검토도 완료했다. ' if review is not None else
                       '작성자의 입력 검사와 root의 파일해시·분모 확인은 완료했으나 별도 에이전트의 최종 입력 검토는 미완료로 남겼다. ')
    assert (fit['rows'], fit['original_sources'], fit['training_known']) == (72, 36, 180)
    assert (dev['rows'], dev['original_sources'], dev['unswapped_queries'], dev['known'], dev['held']) == (76, 12, 95, 88, 7)
    rows = [r for split in plan['rows'].values() for r in split]
    assert len(rows) == 148 and len({r['flow_key'] for r in rows}) == 148
    assert sum(r['aux_binding']['status'] == 'REUSE_SAVED_AH' for r in rows) == reused == 28
    assert sum(r['aux_binding']['status'] == 'NEEDS_FROZEN_AH_FORWARD' for r in rows) == missing == 120
    assert native == 48
    summary = (
        f'사용자 요청으로 현재 작은 입력 준비 단위를 마친 뒤 일시중단했다. '
        f'AO 공개 FIT {fit["rows"]}입력({fit["original_sources"]}원영상, 등록ID01~09), '
        f'DEV {dev["rows"]}입력({dev["original_sources"]}원영상, 등록ID11~13)의 '
        f'영상출처·원시간·배열·감독정답 연결을 캐시만으로 확인했다. '
        f'기존값 정확재사용 {reused}, 신규동결AH 계산이 필요한 입력 {missing}'
        f'(그중 native의 별도 Y snapshot {native})을 분리했다. '
        '새 AH 추론/실제 모델 preflight/두조건 학습/평가는 모두0이며 성능향상 결과가 아니다. '
        '모델코드와 합성검사는 준비됐으나 실제입력 preflight와 최종학습승인은 미완료다. '
        + review_sentence +
        '최초 import 검사실패는 기록보존 후 실제 decode/model 호출금지 검사로 정정했다. '
        '학습/평가 프로세스는 종료·미실행 상태이고, 기존 AM8810 데모 서버는 대기 상태로 보존한다. '
        '사용자·멘토자료 미사용, C·T3-F·원본·기존데모 보존. 전체목표 미완료·사용자재개 전 진행금지.'
    )
    next_action = (
        '사용자의 명시적 재개 요청까지 작업하지 않는다. 재개하면 AO/INPUT_PLAN.json·'
        'INPUT_COMPATIBILITY.json·PAUSE_CLOSEOUT.json과 존재한다면 INPUT_REVIEW.json을 먼저 읽는다. '
        '미완료인 별도 입력 검토를 먼저 마치고 실제학습 승인으로 혼동하지 않는다. '
        '완료된 준비검사와 과거 평가를 반복하지 않는다. 불일치 native를 기존 AH 결과로 '
        '대체하지 않고, 새 exact snapshot에 필요한 AH 추론 예산을 평가계약에 명시해 '
        '검토·동결한 뒤 필요한 입력에만1회 계산한다. 이후 실제2FIT preflight, 최종 '
        'CODE_REVIEW/FROZEN, 동일72입력·초기값·480step의 두조건 부분미세조정, '
        '고정DEV76 및 별도저장검산이 남아 있다. 평가 실행코드는 아직 미작성이다. '
        '같은 주요맥락의 잘못된손모양 정답0, 실제짧은반대동작 평가부족, 전체완료/반복/접촉 '
        '미검증, 전체처리 약11초 문제를 해결된것으로 바꾸지 않는다. '
        '시작·끝 질문의 설명은 이미 답했으므로 재개시 반복하지 않는다.'
    )
    report = f'''작업 일시중단 보고 ({now})

현재 목표
공개 한국수어의 설명과 여러 사람의 영상을 학습한 신경망이, 새 사용자의 행동에서 중요한 손모양·신체관계·움직임·순서를 확인하고 해당 설명 문구와 사용자 영상 시간에 연결된 피드백을 제공하는 것이다. 단어를 맞추는 기존 TCN과는 별개의 목표이며, 현재 전체 기능은 완성되지 않았다.

지금까지 실제로 한 일
1. 손·팔 관절과 SignRep 이미지 특징을 함께 사용하고, 설명의 행동 항목·주요 구간·원영상 시간을 연결했다. SignRep 본체는 고정하고 작은 판정부 및 일부 투영 가중치를 지도학습·미세조정했다.
2. 기존 순서 판정부 비교에서는 관측과 주요동작 정보를 함께 쓰는 조건이 개발 관계103개 중82개를 올바른 방향으로 확정했다. 의존을 제거한 후보는42개로 줄어 채택하지 않았다. 103개는 독립영상103개가 아니라15개 원래 관계의 변형을 포함한다.
3. 시간별 움직임 판정부는 공개4표현·학습 등록ID01~09의36영상으로80회차/320단계를 실제 학습했다. 다른 개발ID11~13의12영상에서 기존 확인 이동15구간과 추가 국소이동2구간을 찾았고, 검토한 국소 비발생13구간의 오확인은0이었다. 반복 노출된 개발자료에서의 제한된 결과이며 새 사용자 정답률이 아니다.
4. 별도 데모8810에는 이 학습형 판정부와 설명문구·원영상시간 표시를 연결했다. 여러 참고영상9개를 매번 비교하는 대신 학습된 기준 흐름과 DTW1회를 사용한다. 공개영상1개의 전체 실행은10.93초였고 최초 준비 포함29.00초였다. 아직3초 안팎 목표에는 도달하지 못했다.
5. 최근 정상·일부삭제·역순·가림16조건의 결과를 함께 정리했다. 실제 역순4영상의 비교가능한 앞뒤 관계5쌍 중2쌍은 역순을 확인했고3쌍은 근거가 섞여 보류했다. 움직임이 보인다는 사실을 전체 동작의 성공으로 바꾸지 않았다.

이번 중단 직전에 끝낸 작은 작업
{summary}
개발비교에는 {dev['unswapped_queries']}질의(정답이 있는 {dev['known']}, 보류 {dev['held']})가 있으며 변형끼리 원영상을 공유한다. 이 수는 새 사람이나 새 독립영상 수가 아니다.
검사 프로그램의 import 자체를 실제 영상분석 실행으로 혼동한 첫 검사오류를 고쳤고, 원오류와 수정 전 코드는 보존했다. 추가로 중복 학습 호출이 원실행의 상태기록을 덮지 않게 코드 수준에서 보완했다. 이는 연구 성능향상과 구분한다.

남은 핵심 문제
- 동작의 존재는 찾더라도 두 동작의 근거가 겹치거나 섞여 정확한 순서를 확정하지 못하는 경우가 있다.
- 같은 주요 동작에서 실제로 잘못된 손모양을 수행한 정답 사례가 최근 감사범위에0개다. 올바른 주요 특징52행, 정상적인 다른단계의 반대특징12행, 불확실3행을 오류 정답과 혼동하지 않았다. 따라서 중요한 손모양 오류 탐지 성능은 아직 검증하지 못했다.
- 짧은 주요동작 전체 누락, 필요한 반복횟수, 정확한 접촉, 전체 수어의 교육적 정오는 아직 확인하지 못한다.
- 개발자료를 반복 사용했다. 더 넓은 새 사람에 대한 일반화와 정상 동작의 오경고 감소가 충분히 입증되지 않았다.
- 전체 분석시간 약11초가 남아 있다. 작은 판정부의 몇 밀리초 계산시간과 전체 앱 대기시간은 다르다.

재개할 경우의 남은 순서 (지금 실행하지 않음)
{next_action}

현재 화면 안내
8800은 과거 비교화면이며 최신 학습형 실험데모는8810이다. 기존 화면을 자동 교체하지 않았다. 데모 서버는 기존 대기 상태로 보존하되 새 촬영·분석·학습을 자동 실행하지 않는다.

검증 범위
구현자와 다른 에이전트의 코드/산술 검산을 구분해 기록했다. 모두 Codex 팀의 검토이며 독립 한국수어 전문가의 정오 검토는 아니다. 사용자·멘토 촬영을 학습·튜닝·후보선택에 사용하지 않았고 C·T3-F를 교체하지 않았다. 새 성능개선이 없으므로 새 HTML은 만들지 않았다.
'''
    (U / 'RESULT_KO.txt').write_text(report, encoding='utf-8')
    evidence = [U / n for n in ['INPUT_PLAN.json', 'INPUT_COMPATIBILITY.json',
                               'EVALUATION_PROTOCOL.json', 'SYNTHETIC_CHECKS_v3.json', 'RESULT_KO.txt']]
    if review_path.exists():
        evidence.append(review_path)
    evidence += [O / directory / name for directory, name in [
        ('word100_direct_occurrence_order_20261102', 'RESULT_KO.txt'),
        ('word100_local_event_learning_20261103', 'FINAL.json'),
        ('word100_local_event_learning_20261103', 'SAVED_VERIFIED.json'),
        ('word100_local_event_demo_20261103', 'RESULT_KO.txt'),
        ('word100_feedback_discrimination_20261103', 'RESULT_KO.txt'),
        ('word100_feedback_discrimination_20261103', 'SAVED_COMPOSITION_VERIFIED.json')]]
    status = dict(status='PAUSED_BY_USER_AFTER_INPUT_PREPARATION', at=now,
                  directory=str(U), summary_ko=summary, next=next_action,
                  fit=fit, dev=dev, reused_saved_predictions=reused,
                  pending_AH_forward_request=plan['new_AH_forward_request'],
                  actual_new_model_preflight=0, new_AH_forwards=0, new_training=0,
                  new_model_evaluation=0, whole_goal_complete=False,
                  independent_input_review_complete=review is not None,
                  existing_demo=dict(url='http://127.0.0.1:8810/', exec_session=19318,
                      ready=True, busy=False, verified='Read-only GET /api/state and live session poll in root turn; no analyze/camera call'),
                  auto_resume_allowed=False, pins={str(p): sha(p) for p in evidence})
    new_json(U / 'PAUSE_CLOSEOUT.json', status)
    context = ROOT / 'WordsInSilence_PROJECT_CONTEXT.md'
    state = O / 'STATE.json'
    handoff = ROOT / 'NEW_CHAT_HANDOFF.md'
    backup = U / 'before_pause_closeout'
    backup.mkdir(exist_ok=False)
    for p in [context, state, handoff]:
        shutil.copy2(p, backup / p.name)
    text = context.read_text(encoding='utf-8-sig')
    h8 = '## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
    h9 = '## 9. 다음 단일 작업'
    assert text.count(h8) == text.count(h9) == 1
    text = text.replace(h8, h8 + '\n\n### 2026-11-03 사용자 요청 일시중단 — 입력 준비 마감\n\n' + summary, 1)
    text = text.replace(h9, h9 + '\n\n### 2026-11-03 사용자 요청 일시중단 — 재개 전 작업 금지\n\n' + next_action, 1)
    context.write_text(text, encoding='utf-8')
    s = read(state)
    s['updated_date_kst'] = now
    s['current_stage'] = 'PAUSED_BY_USER_AFTER_MOTION_ORDER_INPUT_PREPARATION'
    s['next_single_action'] = next_action
    s[U.name] = status
    state.write_text(json.dumps(s, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n' + now + '\n\n' + summary +
                       '\n\n다음: ' + next_action + '\n\n근거: ' + str(U) +
                       '\n\nAM8810 서버 exec19318/PID18696는 기존 대기 상태로 보존. '
                       '재개 때 실제 핸들 확인 후 중복 실행하지 않는다.\n'
                       '직전 작업: 입력 준비 및 코드 합성검사 진행. 현재: 사용자 요청 pause, 전체목표 미완료.\n', encoding='utf-8')
    assert read(state)['current_stage'] == 'PAUSED_BY_USER_AFTER_MOTION_ORDER_INPUT_PREPARATION'
    assert next_action in context.read_text(encoding='utf-8') and next_action in handoff.read_text(encoding='utf-8')
    assert all(sha(p) == h for p, h in status['pins'].items())
    new_json(U / 'PAUSE_DOCS_VERIFIED.json', dict(status='PASS', at=now,
             docs={str(p): sha(p) for p in [context, state, handoff]},
             report_sha256=sha(U / 'RESULT_KO.txt'), whole_goal_complete=False))
    print(json.dumps(dict(status=status['status'], fit=fit, dev=dev,
                          reused=reused, new_AH_pending=missing, training=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
