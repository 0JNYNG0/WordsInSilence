"""Record verified completion and synchronize current handoff (no experiment execution)."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json

UNIT = Path(__file__).resolve().parent
BASE = UNIT.parents[1]
ROOT = BASE.parents[1]
STAMP = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
KEY = UNIT.name


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(name, data):
    with (UNIT / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


assert not (UNIT / 'RESULT.json').exists(), 'Already closed; do not overwrite'
for name in ['SYNTHETIC_CHECK', 'CONNECTION_CHECK', 'WEB_FUNCTIONAL_CHECK', 'EXISTING_UI_CHECK']:
    result = read_json(UNIT / (name + '.json'))
    assert result['status'] == 'PASS', (name, result)
web = read_json(UNIT / 'WEB_FUNCTIONAL_CHECK.json')
assert digest(BASE / 'web/app.js') == web['app_sha256']
assert digest(BASE / 'web/index.html') == web['html_sha256']
weights = read_json(UNIT / 'WEIGHTS.json')
assert digest(Path(weights['path'])) == weights['sha256']
assert digest(BASE / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt') == '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'

next_action = ('Acquire the actual Modu release guide/annotation sample/video listing, verify gloss-time-video-person '
               'linkage and person-separated data availability, then freeze an A/B/C fusion training comparison. '
               'Model connection is complete; do not rerun preparation/tests or treat zero-initialized parity as '
               'accuracy improvement. No implementation-role approval is pending. No training starts from this closure.')
entry = {
    'status': 'COMPLETE_CONNECTION_AND_WEB_FUNCTIONAL_VERIFICATION_NOT_TRAINED',
    'updated_at_kst': STAMP,
    'artifact_root': str(UNIT.relative_to(ROOT)).replace('\\', '/'),
    'report': 'RESULT_KO.txt', 'usage': 'USAGE.txt',
    'authorization': 'AUTHORIZATION.json; user approved local model/web implementation, official small weights and role instruction cleanup',
    'author_and_verification': 'Codex local implementation, separate hook/official-backbone reference and mocked-client verification; no external expert or agent review',
    'weights': weights,
    'bridge': {'parent': 'TCN832', 'rgb_features_per_hand': 576, 'additional_parameters': 73984,
               'method': 'zero-initialized bias-free RGB projection to existing width64 before frozen masked temporal blocks',
               'trained_for_sign_recognition': False, 'web_registered': False, 'default_model_replaced': False},
    'verification': {'synthetic_tests': 8, 'public_connection_segments': 3, 'public_images': 153,
                     'initial_logit_max_abs_error': 0.0, 'official_encoder_reference_max_abs_error': 0.0,
                     'web_functional_tests': 8, 'selected_existing_ui_assertions': 38,
                     'actual_browser_rendering': 'not verified: CUA initialization failed to write kernel assets (os error 3)',
                     'camera': 'not opened'},
    'web': 'Opt-in remains off; checked requires explicit model, name/ID shown before start; capture and replay lock choice; recorded result identity preserved. No untrained RGB option.',
    'limits': 'Functional connection, not trained RGB accuracy. Prepared113 only known training people01/02; independent paired RGB/pose evaluation still missing. CPU float32 only, no Unity export verification.',
    'role_rule': 'AGENTS latest: Codex may implement already-approved work locally without repeat author-role exceptions. External tool approval remains separate; no denied external resend.',
    'scope_preserved': ['no sign training/accuracy rescore', 'no user data', 'no camera/device', 'no corpus download', 'no operating default replacement', 'no Git mutation'],
    'next': next_action,
    'active_processes': [],
}
save_new('RESULT.json', entry)

p = ROOT / 'WordsInSilence_PROJECT_CONTEXT.md'
text = p.read_text(encoding='utf-8-sig')
updates = {
    '## 3. 운영 역할': '''

- 2026-10-10 최신 사용자 승인: “승인할테니 빨리 진행해줘. … 불필요한 문구면 그냥 수정해서 다음에 이런 승인을 따로 안받아도되게”. **승인된 작업 범위에서는 Codex가 구현·실행·검증·문서를 직접 수행할 수 있다.** Claude 한도·인증·외부 전송 거절이 있더라도 로컬로 수행 가능한 같은 작업의 작성자 변경만으로 다시 승인받지 않는다. AGENTS/CLAUDE/WORKFLOW에 반영했다. 과거의 좁은 역할 예외와 아래 승인 대기 기록은 이 최신 승인을 제한하지 않는다. 외부 도구의 전송 승인 판단은 별도이며 이를 우회하지 않는다.
- 이번 승인 범위 `ops/rgb_pose_bridge_20261010/AUTHORIZATION.json`: 관절+RGB 격리 연결부, 웹 연구 모델 명시 선택, 공식 MobileNetV3-Small 약10MB 확보. 로컬 구현·검증 완료. 학습 실험·사용자 자료 튜닝·카메라·운영 기본 모델 교체·Git 변경으로 확대하지 않았다.
''',
    '## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인': '''

### 2026-10-10 관절+RGB 연결·웹 선택 수정 완료 — 수어 학습 전, 역할 승인 대기 해소

최신 근거는 `ops/rgb_pose_bridge_20261010/RESULT_KO.txt`·`RESULT.json`이다. 공식 MobileNetV3-Small 10,306,551바이트를 확보하고 전체 SHA256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`로 고정했다. 실제 손 사진→손당576차원→원시각/좌우 배치→추가64차원 투영→기존TCN832 시간 처리로 연결했다. 기존280관절 입력·관측·크기·시간을 유지한다. 사진 누락과 추정 영역 표시를 구분하며 사진/관절을 합성하지 않는다.

합성8검사, 공개 준비3구간/153사진 실제 연결, 별도 hook/공식 인코더 계산 대조 PASS. 공식 특징 오차와 초기93개 점수 오차 모두0.0이다. **연결부를0으로 초기화했으므로 기존 점수와 같은 것이며 RGB 인식률 향상이 아니다.** 인코더/부모TCN 고정, 추가73,984파라미터만 학습 가능하게 준비했다. 실제 학습 계약은 미확정이며 수어 학습·체크포인트 저장·정답률 계산을 하지 않았다. 원모델·사용 캐시 해시/통계 불변.

웹은 연구 비교 기본 꺼짐을 유지하고 켰을 때 모델을 직접 고르도록 했다. 미선택이면 촬영/재생 요청 차단, 시작 전 이름/ID 표시, 촬영·재생 동안 선택 잠금. 과거 결과는 기록된 모델 정체성을 유지한다. 실제 app.js 모의 HTTP 기능8검사·기존 UI38확인·문법검사 PASS. 브라우저 도구가 초기화 오류(os error3)로 실패해 실제 화면 렌더링/실제 촬영은 미검증이다. 서버 코드/운영기본을 바꾸지 않았으며 미학습 RGB는 웹 목록에 없다.

AGENTS/CLAUDE/WORKFLOW의 Claude 전담·반복 역할 예외 요구를 최신 사용자 승인으로 정리했다. 도구의 외부 전송 승인 판단을 제거한 것은 아니다. 현재 실행 종료. 아래 과거 기록의 가중치 미확보/역할 답변 대기는 이번 완료로 해소됐다. 대응 RGB를 가진 새 사람 평가 자료는 여전히 미확보다.
''',
    '## 9. 다음 단일 작업': '''

### 2026-10-10 다음 — 실제 배포 주석·영상·사람 연결 확인 후 RGB 비교 학습 계약 확정

관절+RGB 연결과 웹 명시 선택 구현은 완료했다(`ops/rgb_pose_bridge_20261010/RESULT_KO.txt`). **역할 승인 질문은 남아 있지 않다.** 입력224 생성·완료 검사·기존 학습 비교는 재실행하지 않는다. 현재 준비113구간은 학습 사람01/02이며 새 사람의 RGB 효과 검증 자료가 아니다.

사용자가 모두의 말뭉치 배포 설명서/주석 샘플/영상 목록을 확보하면 `docs/mentoring_audio_20261010/MODU_APPLICATION_AND_HANDSHAPE_GUIDE.txt`와 기존 DATA_ACQUISITION_PLAN에 따라 단어·사람·시간·원영상 연결 및 고유 표본 수부터 점검한다. 실제 배포 파일 전에는 top100·사람 분할·최종 학습 예산을 추정으로 채우지 않는다. 신청 본인 인증/약정은 사용자 단계다.

대응 자료가 준비되면 같은 새 학습 기회의 관절만A/정상RGB추가B/같은 조건의 공백시점 실제RGB추가C 비교를 고정한다. 학습 층·분할·성공/회귀 기준을 사전에 정하고 기존832는 별도 기준으로 보고한다. 이번 연결의0초기화 점수 동일성은 성능 결과가 아니다. 실제 브라우저 렌더링 검사는 도구 복구 시 작은 후속으로 남는다. 운영 기본 교체·사용자 자료 학습·카메라·Git 변경을 자동 시작하지 않는다.
''',
}
for heading, block in updates.items():
    assert text.count(heading) == 1, heading
    text = text.replace(heading, heading + block, 1)
p.write_text(text, encoding='utf-8')

p = BASE / 'ops/STATE.json'
state = read_json(p)
state['current_stage'] = 'RGB_POSE_CONNECTION_AND_WEB_SELECTION_VERIFIED_AWAITING_PAIRED_EVALUATION_DATA'
state['updated_date_kst'] = '2026-10-10'
state['next_single_action'] = next_action
# Keep the newest unit near the top without dropping any historical keys.
items = list(state.items())
state = dict(items[:6] + [(KEY, entry)] + items[6:])
p.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

handoff = f'''# WordsInSilence 최신 인수인계
갱신: {STAMP}. 현재 웹 STATE는 `research/front_keyphase_v1/ops/STATE.json`이다.
시작: AGENTS.md, WordsInSilence_PROJECT_CONTEXT.md 최신 §3·§8·§9, WORKFLOW_HANDOFF.md를 읽는다.

## 현재 목표와 마지막 완료
- 목표: 여러 사람의 수어 단어 인식과 이후 구체적 수행 피드백. 사용자 자료를 학습/튜닝/후보 선택에 쓰지 않는다.
- 최신 완료: `ops/{KEY}/RESULT_KO.txt`·`RESULT.json`. 관절+RGB 연결 부품과 기존 웹의 명시 모델 선택을 Codex가 로컬 구현·검증했다. 역할 승인 대기는 해소됐다.
- 공식 MobileNetV3-Small 약10MB 확보. 전체 해시와 위치는 WEIGHTS.json. 다른 자료 다운로드나 외부 프로젝트 전송은 없다.
- 손당576차원 RGB 특징을 원래 시각/좌우에 배치해 추가64차원 투영 후 기존TCN832에 더한다. 기존280관절·관측·시간·몸 대비 크기는 유지한다. 추정 영역을 관절 관측으로 바꾸지 않는다.
- 인코더/부모TCN 고정, 추가73,984파라미터만 학습 가능하게 준비. 연결부0초기화로 기존93점수와 완전히 같다. 수어 학습을 하지 않았으며 RGB 성능 향상을 입증한 결과가 아니다. 미학습 RGB를 웹에 등록하지 않았다.
- 합성8/공개3구간153사진 연결/공식 인코더·별도hook 대조 PASS. 특징/초기점수 최대오차0.0. 원체크포인트/캐시/정규화 통계 불변.
- 웹: 연구 비교 기본 꺼짐. 켜면 모델을 직접 골라야 시작/재생 가능, 이름·ID 표시, 처리 중 선택 잠금. 저장된 결과의 정체성 유지. 실제 app.js 모의 기능8·기존 UI38확인·문법 PASS. 실제 화면은 브라우저 도구 초기화 실패(os error3)로 미검증. 카메라/서버/운영기본 변경 없음.

## 다음 한 작업
실제 모두의 말뭉치 배포 설명서/주석 샘플/영상 목록을 확보하면 단어·사람·시각·영상 연결과 고유 표본 수를 확인한다. 자료 신청은 아직 미신청/미확보다.
이후 사람 분리와 같은 학습 기회의 A관절만/B정상사진 추가/C공백 시점의 실제사진 추가 비교를 동결한다. 연결 완료를 새 사람 정확도로 설명하지 않는다. 기존113구간은 알려진 학습 사람01/02다.
현재 구현 승인 질문은 없다. 입력 준비/기존 실험/통과 검사를 반복하지 않는다. 실제 브라우저 렌더링은 도구 복구 시 작은 후속으로 남긴다. 새 학습은 자동 시작하지 않았다.

## 역할
- 2026-10-10 사용자 승인으로 승인된 범위에서는 Codex가 구현·실행·검증·문서 갱신 가능. Claude 사용 불가 시 같은 작업의 작성자 변경만으로 다시 묻지 않는다. AGENTS/CLAUDE/WORKFLOW 동기화 완료.
- 외부 전송 도구의 승인 판단은 별도. 이전 Claude 명세 전송 거절을 우회하거나 재전송하지 않았다. 실제 작성자/검증 방법을 기록한다.
- 원본/동결 결과·학습/평가 분리 유지. 운영기본교체·사용자자료튜닝·카메라/장치·새 말뭉치 다운로드·Git 변경은 이번 범위가 아니다. 이번 가중치 파일만 명시 승인으로 확보했다.
- 이번 모든 실행 종료. 자동 학습/서버/카메라 시작 없음. 예전 웹8765 생존 여부 미확인. st-gcn heartbeat는 과거 완료 후 PAUSED.

## 완료 입력 근거 — 재실행하지 않음
- `ops/rgb_tensor_adapter_20261010/RESULT_KO.txt`:105영상/113구간에서 원영상 직접224사진5393장(정상5315+추정78),185칸 누락 유지. 14합성·전수사진/시각/정규화 검산 PASS,523입력파일 불변. pack/scatter로 관절·좌우·시각·표시 보존.
- `ops/rgb_comparison_contract_20261010/`:113구간/2789행/5578손칸 A/B/C 입력색인, 인코더 전처리/사용조건. 공식 중심크롭과 달리 원정사각 영역 전체224 사용. 당시 미확보였던 가중치는 최신 단위에서 확보했다.
- `ops/rgb_temporal_roi_20261010/RESULT_KO.txt`:관절 누락 시 앞뒤 손 영역을 연결해 같은시각의 실제사진을 읽는 도구. 손가락/사진 합성 아님. 정상관측숨김4456통과사례에서21점전체포함97.35→99.91%(+2.56pp), 개선116/후퇴2. 단어인식률 아님.
- 실제 한프레임 누락88 중78사진 확보/10거절. 잘못된 손7개 거절, 동일손74/77통과, 불확실4개는 불확실 유지. 흐림/가림은 남는다. 원자료 보존/별도 전수검산 완료.

## 자료 확보와 전체 계획
- 사용자: 모두의 말뭉치 아직 신청·확보 안 함. 연구실 Seagate ST3000DM001 3TB 2개는 연결제품 없어 미연결. 건강/공간/포트 미확인. 장치 접근 권한으로 보지 않는다.
- 안내: `docs/mentoring_audio_20261010/MODU_APPLICATION_AND_HANDSHAPE_GUIDE.txt`. 공식2024한국어-한국수어 병렬 구축보고서에 gloss/시간/영상/수행자 필드가 있다. 실제 배포 스키마·단어별 사람 수·영상 포함은 미확인. 병렬을 문장번역만 있는 자료로 제외하지 않는다.
- 주석/파일목록부터 고유단어·원본·사람 수와 필요한 영상 묶음을 확인한다. 양손 주석 중복/문장번역 빈도/연도별 동일인 혼동 금지. 영상 포함 자료를 원시·병렬·주석으로 중복 다운로드하지 않는다.
- 수형 사전 분류는 특징/평가 정의의 참고다. 프레임별 정답·다인 학습자료가 아니다. 기존18참조의 분류 대조는 아직 제안이다.
- 전체계획: `docs/mentoring_audio_20261010/DATA_SELECTION_AND_SPLIT_FOLLOWUP.txt`, `ops/data_pretraining_inventory_20261010/DATA_ACQUISITION_PLAN.txt`. 단어당10건/top100/7:2:1·8:2는 제안이며 일반화 보장/확정 비율 아님.
- 현재2346학습행/1882원본/93단어. AIHub81단어 수행자01~09, 나머지12는 NIKL시범1개씩/신원미확인. MP113은832·bed 미학습구간이나 수행자는01/02. 44단어2명각1구간/25단어1명1구간. 새사람평가 부족.
- 기존699/74평가는11/12/13좌표이며 대응RGB 경로 미확인. 조사한 비압축NIA127영상은01/02만 확인(전체디스크 부재 주장 아님). KSL36 사람 미확인; 공부/먹다는 준비69단어에 없음.

## 모델과 멘토 현장 시험
- 기준TCN832: `runs/partial_encoder_pair_v1/checkpoint_control_final.pt`, SHA83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91.
- 기하bed: `runs/tcn_geometry_pair_v1/checkpoint_treatment_final.pt`, SHAbed406d931cb5a747ae55f347a42c91a9ea55a38798dd8fbb2f3cdfed5a6ae36. 832→bed691/699유지(98.86%),KSL22/34→23/34. 먹다개선은대조군도맞혀기하만의효과아님. bed웹미연결. STGCNlocalshape662/699·8/34미채택. 완료실험반복금지.
- 멘토2명5회는 `docs/mentoring_audio_20261010/SESSION_RECORD_REVIEW_KO.txt`. 정답확인0/사람ID혼재, ses_33578ccdf974 연구모델union93/e7c9로832·bed아님. 새사람정확도 집계 불가. 이 혼동을 줄이기 위해 최신 웹을 명시 선택으로 바꿨다.
- 웹DTW/task_match 구현, 수행 합격/거절 기준 UNCONFIGURED. top1차이는 목표로 인식되지 않음이며 top2/3이 경미한 수형오류를 뜻하지 않는다. 단어 인식과 수행 피드백 검증을 구분한다.
- 멘토녹음 로컬전사/중요구간 재전사 완료. `docs/mentoring_audio_20261010/MENTORING_REVIEW.html`, `output/pdf/WordsInSilence_멘토링_피드백과_실행계획_20261010.pdf`. 불확실성 유지, 외부전송0, 반복전사 불필요.
'''
(ROOT / 'NEW_CHAT_HANDOFF.md').write_text(handoff, encoding='utf-8')

paths = ['AGENTS.md', 'CLAUDE.md', 'WORKFLOW_HANDOFF.md', 'WordsInSilence_PROJECT_CONTEXT.md',
         'NEW_CHAT_HANDOFF.md', 'research/front_keyphase_v1/ops/STATE.json',
         'research/front_keyphase_v1/web/app.js', 'research/front_keyphase_v1/web/index.html',
         'research/front_keyphase_v1/src/wis_lab/service/neural.py']
save_new('CLOSURE_CHECK.json', {'status': 'PASS', 'at_kst': STAMP,
    'files': {p: digest(ROOT / p) for p in paths},
    'json_readback': read_json(BASE / 'ops/STATE.json')[KEY]['status'],
    'browser_limitation': 'CUA initialization error3; actual render not verified',
    'unfinished': ['Actual browser render/live camera check', 'Paired person-separated data and frozen training comparison'],
    'role_questions_pending': False})
print('Completed documentation sync:', KEY)
