"""Root evidence closeout only; no model imports or rescoring."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, re, shutil, statistics

U = Path(__file__).resolve().parent
O = U.parent
AI = O / 'word100_local_event_guide_adapter_20261103'
ROOT = O.parents[2]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def prepend(text, section, entry):
    match = re.search(r'(?m)^## ' + str(section) + r'\.[^\n]*\n', text)
    assert match
    return text[:match.end()] + '\n' + entry + '\n\n' + text[match.end():].lstrip('\r\n')

def main():
    result = read(U / 'RESULT.json')
    verified = read(U / 'SAVED_VERIFIED.json')
    adapter = read(AI / 'REPLAY_RESULT.json')
    adapter_verified = read(AI / 'SAVED_VERIFIED.json')
    assert verified['status'].startswith('PASS') and verified['failed_checks'] == []
    assert adapter_verified['status'].startswith('PASS') and adapter_verified['failed_checks'] == []
    assert verified['summary'] == result['summary']
    assert sha(U / 'RESULT.json') == read(U / 'RUN_FINISHED.json')['result_sha256']
    normal = [r for r in result['summary'] if r['normal_preservation_expectation']]
    assert sum(r['scopes'] for r in normal) == 20
    assert sum(r['high_occurrence'] for r in normal) == 20
    blackout = next(r for r in result['summary'] if r['kind'] == 'main_blackout')
    cut = next(r for r in result['summary'] if r['kind'] == 'action_cut')
    assert (blackout['high_occurrence'], blackout['insufficient'], blackout['scopes_with_gaps']) == (2, 3, 5)
    assert cut['high_occurrence'] == 3
    assert adapter['primary_scopes'] == 15 and adapter_verified['all_original_outputs_unchanged'] is True
    median_ms = 1000 * statistics.median(r['head_seconds'] for r in result['rows'])
    at = datetime.now(timezone(timedelta(hours=9))).isoformat()
    next_action = (
        'AH 학습/native평가, AI 저장연결, AJ24 입력 추론/검산은 완료되어 반복하지 않는다. '
        '다음은 별도 연구 실행부에 동결 AH 시간별 판정부와 AI의 출처 연결 계약을 함께 붙이는 작업이다. '
        '새 공개 입력 1개를 관측한 뒤 학습된 prototype과 DTW 1회만 수행하고, 전체 시계열 판정부 1회로 '
        '손모양·신체관계·움직임 근거와 원시각을 반환한다. 9개 참고영상의 실행시 비교는 하지 않는다. '
        '이전 웹/8804/8808/8809와 C/T3F는 보존한다. 새 입력용 출처·native축 검증을 먼저 마련하고 '
        '기존 고정 학습/평가의 재실행 없이 합성 입력·공개 전용 새 실행 시험으로 연결을 확인한다. '
        '검사 범위는 점수로 선택하지 않고, 짧은 움직임 high를 문구 전체/반복완료/수어정답으로 승격하지 않는다. '
        'RGB 문맥이 범위 밖 또는 가린 시점을 포함할 수 있음을 남기고 관측 부족은 확인불가로 표시한다. '
        '원본대기시간 포함 계측은 새 실행에서 별도로 하고 캐시 head ms를 전체속도로 보고하지 않는다. '
        '기존 사용자·멘토/잠금평가자료 접근·학습·튜닝·선택, 새 다운로드/외부전송/Git/운영교체 금지. '
        '사용자 촬영은 명시적 새 시험 때 결과제공만 가능하며 학습/선택 금지. 전체 goal active·미완료.')
    summary = (
        '시간별 움직임 판정부 AH의 저장 결과를 AI 격리 adapter로 공개 DEV12영상·고정15구간에 연결했고, '
        '새 움직임 high15/15와 기존 모든 출력 보존12/12를 별도819검산했다. 기존 방식도 같은15구간은 high였으므로 '
        '이 연결 자체는 추가 정확도 향상이 아니다. 이어 AJ에서 등록 ID12의 공개4표현×6실제편집조건24입력을 '
        '학습된 AH 후보로 각1회 추론했다. 원영상/0.8배속/1.25배속/시작대기제거의 고정범위20/20에서 '
        '움직임 high가 유지됐다(전체흐름 진단20/20 별도). 이는 기존 개발자료의 보존 시험이며 새 사용자 정확도 아니다. '
        '동작 일부삭제5구간 중 high3/낮음2는 발생정답 미확정이다. 남은 짧은 움직임을 전체동작 완료로 판단하지 않는다. '
        '가림5구간 모두 공백 유지: high+공백2/관측부족3, 전부 부재확정 불가. 전체흐름 진단 high5/5라도 '
        '주요구간 복구나 정상으로 승격하지 않았다. AJ 준비787·코드456·별도 저장2912검사 PASS, '
        '모델/원본/옛결과 해시 보존. 추가학습/재추출/새DTW/웹교체0. '
        f'캐시 입력의 작은 판정부 중앙값 {median_ms:.2f}ms는 전체 앱 지연이 아니다. '
        '정확한 동작 경계/반복횟수/접촉/전체행동의 충분성/교육적 오류와 웹개선은 아직 미검증이다. '
        '같은 Codex의 구현·별도 검산이며 독립 수어전문가 정답이 아니다. '
        '사용자 질문에 설명한 시작·끝은 검사구간의 양끝이며, 양끝 유사성이 기존 넓은구간 실패의 단독원인인지는 미확인이다. '
        '전체 학습형 피드백 목표 active·미완료.')
    limits = [
        'Repeated development controls with one registered source ID12, not an untouched generalization test.',
        'Normal preservation is a fixed metamorphic expectation, not independently reviewed exact-scope dense truth.',
        'Cut and blackout occurrence labels remain null; no educational error accuracy is established.',
        'Temporal and RGB context extends beyond nominal scopes; maxima do not certify exact event boundaries.',
        'No live web path changed. Cache-only head timing excludes extraction and other pipeline work.',
        'Same Codex implementation/review/arithmetic, not independent KSL expert annotations.'
    ]
    adapter_final = dict(status='SAVED_GUIDE_ADAPTER_VERIFIED', at=at, clips=12, primary_scopes=15,
        high_occurrence=15, new_accuracy_improvement_claimed=False, checks=819,
        original_fields_preserved=True, new_model_forwards=0, new_DTW=0, training=0,
        live_changed=False, whole_goal_complete=False, next=next_action,
        pins={str(AI/n):sha(AI/n) for n in ['FROZEN.json','CODE_REVIEW.json','REPLAY_RESULT.json','SAVED_VERIFIED.json']})
    final = dict(status='PUBLIC_SPEED_GAP_CONTROLS_VERIFIED', at=at, inputs=24, source_registered_ids=[12],
        model_sha256=result['model_sha256'], new_model_forwards=24, training=0, new_extraction=0, new_DTW=0,
        fixed_scopes=30, diagnostic_scopes=30, normal_preserved=[20,20],
        cut_high=3, cut_unknown_truth=5, blackout_high_with_gap=2, blackout_insufficient=3,
        all_blackout_scopes_keep_gap=True, saved_checks=verified['checks_count'],
        median_cached_head_ms=median_ms, whole_pipeline_latency=None, live_changed=False,
        whole_goal_complete=False, limitations=limits, next=next_action,
        pins={str(U/n):sha(U/n) for n in ['INPUT_FROZEN.json','PLAN_FROZEN.json','PREP_CHECKS.json',
            'CODE_REVIEW.json','EXECUTION_FROZEN.json','RESULT.json','RUN_FINISHED.json','SAVED_VERIFIED.json']})
    # Korean report is kept separate from immutable scored artifacts.
    report = ('공개 영상의 속도·대기·가림 변화 확인 결과\n\n'
        '1. 이미 학습한 작은 판정부를 공개 영상 24개에 한 번씩 적용했습니다. 새 학습은 하지 않았습니다.\n'
        '2. 원본 조건, 0.8배속, 1.25배속, 시작 대기 제거에서 확인 대상 20개 모두 움직임 근거가 유지됐습니다.\n'
        '3. 동작 일부를 잘라낸 경우에도 움직임 일부가 남을 수 있습니다. 5개 중 3개에서 높은 점수가 나왔으나, 전체 동작을 잘했다는 뜻으로 쓰지 않습니다.\n'
        '4. 주요 부분을 가린 5개는 모두 관측 공백을 남겼습니다. 2개는 남은 움직임+공백, 3개는 관측 부족입니다. 영상 다른 곳의 높은 점수로 가린 부분을 통과시키지 않습니다.\n'
        '5. 별도 검산을 마쳤습니다. 기존 C·T3-F, 웹 테스트와 사용자·멘토 자료는 변경하거나 학습에 쓰지 않았습니다.\n'
        f'6. 작은 판정부만의 계산 중앙값은 {median_ms:.2f}ms입니다. 영상 관측부터 결과 화면까지의 대기시간은 아닙니다.\n\n'
        '아직 모르는 것: 새로운 사용자의 다양한 동작, 정확한 경계, 반복/접촉/전체 행동의 충분성과 교육적 오류 판단. 이번 자료는 반복 사용한 개발자료이며 독립 수어전문가 정답이 아닙니다.\n\n'
        '다음 작업: '+next_action+'\n')
    with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f: f.write(report)
    write(AI/'FINAL.json',adapter_final)
    write(U/'FINAL.json',final)
    before=U/'before'; before.mkdir(exist_ok=True)
    docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
    for p in docs:
        assert not (before/p.name).exists()
        shutil.copy2(p,before/p.name)
    heading='2026-11-03 시간별 움직임 연결·속도·관측누락 공개 검증 완료'
    context=docs[0].read_text(encoding='utf-8-sig'); assert heading not in context
    context=prepend(context,8,'### '+heading+'\n\n'+summary)
    context=prepend(context,9,'### '+heading+' 이후 다음 단일 작업\n\n'+next_action)
    docs[0].write_text(context,encoding='utf-8')
    docs[1].write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+next_action+
        '\n\n근거: '+str(AI)+'\n'+str(U)+'\n\n모델 실행 AJ 종료코드0, 별도 저장검산 종료코드0. 진행 중 모델 프로세스 없음. 완료 추론을 반복하지 않는다.\n',encoding='utf-8')
    state=read(docs[2]); state['updated_date_kst']='2026-11-03'
    state['current_stage']='LOCAL_EVENT_ADAPTER_SPEED_GAP_VERIFIED_RESEARCH_RUNTIME_NEXT'
    state['next_single_action']=next_action
    state[AI.name]=dict(adapter_final,directory=str(AI))
    state[U.name]=dict(final,directory=str(U),summary_ko=summary)
    docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    assert read(docs[2])[U.name]['normal_preserved']==[20,20]
    assert summary in docs[1].read_text(encoding='utf-8')
    write(U/'DOCS_SYNCED.json',dict(at=at,files={str(p):sha(p) for p in docs},whole_goal_complete=False))
    print('AI and AJ verified closeout; context/state/handoff synchronized; overall goal active')

if __name__ == '__main__':
    main()
