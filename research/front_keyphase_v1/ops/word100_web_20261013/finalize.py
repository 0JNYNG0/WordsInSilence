"""Record verified web work; no inference, device access, or training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib

U = Path(__file__).resolve().parent
B = U.parents[1]
R = B.parents[1]
now = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

check = json.loads((U/'CHECK.json').read_text('utf-8'))
assert check['status'] == 'PASS' and len(check['checks']) == 22
assert (U/'PRACTICE_2X2.jpg').is_file()
ui = dict(status='PASS', at_kst=now, driver='Codex in-app browser through computer-use',
    camera='No real camera opened; localhost8876 fake session API only',
    checks=['Mock connect/countdown/result flow', 'Countdown cancellation returns to idle',
            'Requested expression disabled during countdown', 'Participant confirmation stored',
            'Word change clears prior result', 'Mock camera close returns to connect button',
            'Only one repeat capture button after result', 'Actual reference video playback',
            'No browser warning/error in completed mock flow', 'Real practice page MODEL READY'],
    layouts=[dict(width=1280, height=720, four_card_bottom=688.78125, horizontal_overflow=False,
                  state='result and confirmation controls'),
             dict(width=319, height=594, four_card_bottom=578.7890625, horizontal_overflow=False,
                  state='result and saved confirmation')],
    viewport_override_reset=True, fixture_pid=18072, fixture_stopped=True,
    screenshot='PRACTICE_2X2.jpg', real_live_user_accuracy_verified=False)
write(U/'UI_CHECK.json', ui)
changed = ['src/wis_lab/service/word100.py', 'src/wis_lab/service/app.py',
           'web/index.html', 'web/practice.html', 'web/practice.css', 'web/practice.js',
           'config/word100_practice.json']
record = dict(status='COMPLETE_VERIFIED_CAMERA_FREE', at_kst=now,
    implementer='Codex local', verifier='Codex independent input/logit comparison, API mocks and browser inspection; not another expert/agent',
    purpose='AIHub100 opt-in practice with 12 whole-front training references and one-screen2x2 UI',
    checkpoint_sha256='e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f',
    full_classifier_classes=100, practice_reference_count=12, reference_bytes=54133702,
    api_checks='CHECK.json', browser_checks='UI_CHECK.json', reference_check='REFERENCE_CHECK.json',
    url='http://127.0.0.1:8765/practice.html', real_server_pid=62560,
    server_environment='require_escalated; camera opens only on user connect',
    camera_opened=False, training=False, final_heldout_evaluation=False,
    production_default_replaced=False, git_changes=False,
    limits=['Camera-free verification is not live recognition accuracy.',
            '100-way top1 selection does not validate correct handshape or non-sign rejection.',
            'Legacy18 DTW intentionally not applied to new WORD IDs.',
            'W drive must remain connected for reference videos.',
            'Collapsed extra information or extremely small windows may require scrolling.'],
    files={p: hashlib.sha256((B/p).read_bytes()).hexdigest() for p in changed})
write(U/'COMPLETION.json',record)

p=R/'WordsInSilence_PROJECT_CONTEXT.md'
text=p.read_text('utf-8')
entry8='''### 2026-10-13 100단어 연구용 연습 웹 연결·2×2 화면 완료
`ops/word100_web_20261013/COMPLETION.json` COMPLETE_VERIFIED_CAMERA_FREE. Codex 로컬 구현, 입력/출력 별도 계산·API 모의·브라우저 검증 완료; 다른 에이전트/수어 전문가 검수 아님.
사용자 최신 요청(이전18개 대신 새100단어 중 일상 표현, 다운로드 영상 참고, 깔끔한 화면, 결과 포함2×2)에 따라 `http://127.0.0.1:8765/practice.html` 연결. 참고 동작/내 동작/인식 결과/촬영·기록 네 칸, 추가 설명은 접음.
12표현: 수어·마음에들다·멀다·잘하다·믿다·즐겁다·감사·수영·공항·별·사계절·냉커피. 빈도순위/점수기반 선정 아님. ID01 학습 정면 전체 영상12개54,133,702바이트, W의 word100_web_references_20261013에 보존·해시 검산.
모델 e39d8ac4(whole_context)를 명시적으로 연결, 예측은12개 제한 없이100개 전체. 요청 단어는 예측 후 비교, 기존18개 DTW 미적용. 단어 인식과 동작 정확성은 구분하며 수형 정답 판정 없음. 기존 기본 모델/학습 결과 보존.
22개 API/입력 검산 PASS(원 입력·logit 차이0, 빈손 거절, 취소·저장·세션소유권 포함). 브라우저 모의 촬영/확인 기록/표현 변경 PASS. 결과가 나온 상태1280×720 및319×594에서 네 칸 표시/가로 넘침 없음. 실제 카메라/새 사용자 정확도 미검증; 성능 개선 주장 없음.
일반환경 실제 서버 PID62560/8765 유지, 카메라는 사용자 연결 버튼에서만 시작. UI 모의 서버18072 종료. 원본·기존 모델·Git 유지. 새 학습/보류평가0. 발표용 보고서는 계속 유예.
'''
entry9='''### 2026-10-13 다음 — 100단어 연습 화면에서 실제 촬영 연결 확인
카메라 없는 웹 연결 검증은 완료. 사용자가 `http://127.0.0.1:8765/practice.html`에서 표현 선택→필요시 다시보기 저장 체크→카메라 연결→촬영 시작→촬영 종료→확인 기록을 수행한다. 모의 결과를 실사용 정확도로 세지 않는다.
진행 중 새 학습 없음. 서버62560/8765는 재개 때 확인 후 재사용. 외장 W의 참고영상12개 연결 필요. 완료한 추출/학습/개발 채점은 반복하지 않는다. 사용자 자료로 튜닝/후보 선택하지 않으며, 손모양·DTW/거절 검증은 별도 범위. 기존 기본모델 교체·보류시험·어휘추가학습을 자동 시작하지 않는다.
'''
for heading, entry in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry8), ('## 9. 다음 단일 작업',entry9)]:
    assert entry.split('\n')[0] not in text
    text=text.replace(heading+'\n',heading+'\n\n'+entry+'\n',1)
p.write_text(text,encoding='utf-8')

p=B/'ops/STATE.json'
state=json.loads(p.read_text('utf-8'))
state['updated_date_kst']=now
state['current_stage']='WORD100_WEB_PRACTICE_READY_CAMERA_FREE_VERIFIED'
state['next_single_action']='새100단어 연습 화면의 사용자 실제 촬영 연결 확인. 학습/채점 반복·사용자 자료 튜닝·기본모델 교체·보류시험 자동시작 없음.'
state['word100_web_20261013']=record
write(p,state)

(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계
갱신 {now}. AGENTS / CONTEXT 최신 §3·§8·§9 / WORKFLOW를 먼저 읽는다.

## 마지막 완료
`ops/word100_web_20261013` COMPLETE_VERIFIED_CAMERA_FREE. 새100단어 모델의 명시적 연습 화면을 만들고 사용자의2×2 요청까지 반영했다.
URL http://127.0.0.1:8765/practice.html — 참고영상/내 동작/인식 결과/촬영·기록. 참고 표현12개, 인식은100개 전체. 기존18단어 DTW는 새 어휘에 사용하지 않음.
ID01 학습 정면영상12개54,133,702바이트: W:/WordsInSilence/data/AIHub_103/word100_web_references_20261013. 선택은 일상 적합성 판단, 빈도/점수 순위 아님.
모델 whole_context e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 요청 단어와 상관없이100-way 추론. 정확한 수형/수어 아닌 동작 거절/새 사람 일반화는 미검증.
Codex 로컬 구현 + 입력/logit 별도검산(차이0)·API22항목·브라우저 모의촬영/확인기록/반응형PASS. UI_CHECK/CHECK/REFERENCE_CHECK/COMPLETION 참조. 다른 전문가/에이전트 검수 아님. 실제 카메라 미개방.

## 실행 상태 / 다음
실제 서버 PID62560, localhost8765, 일반환경(require_escalated). SERVER_PROCESS.json과 server_stdout/stderr.log가 위 작업 폴더에 있음. 중복 시작 전에 실제 프로세스 확인. 카메라는 사용자가 연결 버튼 누를 때 시작.
UI 모의8876/PID18072 종료. 추가 학습/추출/평가 실행 없음. 연습 탭4 유지, 브라우저 viewport override 해제.
다음: 사용자 실제 촬영 연결 확인. 표현 선택→다시보기 저장(선택)→카메라 연결→촬영 시작/종료→확인 기록. 좌표·인식 결과는 자동 저장, 영상 장면은 옵션. 사용자 기록 학습/튜닝/후보선택 금지.
기존 기본 모델 교체/보류평가/추가튜닝 자동시작 없음. 발표보고서는 사용자 지시대로 마지막으로 유예.

## 완료한 학습 (반복하지 않음)
`ops/word100_training_20261013` COMPLETE_VERIFIED. AIHub-only fresh MaskedSignTCN100클래스/280채널/37,636매개변수.
학습899(ID01~09), 개발300(ID11~13), 전체1,199영상150,476프레임. 전체277/300=92.33%, 전체·핵심 혼합290/300=96.67%, +4.33pp(개선17/회귀4).
개발자료로 epoch도 선택했으므로 독립 최종시험 아님. 기존93클래스 점수와 직접 비교 금지. NIKL/KSL/사용자/멘토·ID10/14~18 사용 안 함.
원본 관절과 전체/dev를 보존하고 context 끝1프레임 경계오류3개만 학습 전 복구·검산. 학습/검산 exit0, MODEL 기본 해시 유지. 캐시 W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013.
''',encoding='utf-8')
assert json.loads((B/'ops/STATE.json').read_text('utf-8'))['word100_web_20261013']['status']==record['status']
print(json.dumps({'status':'RECORDED','at_kst':now,'api_checks':22,'ui':'PASS'},ensure_ascii=False))
