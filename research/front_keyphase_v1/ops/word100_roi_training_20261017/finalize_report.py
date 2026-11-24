"""Summarize completed evidence and update project handoff; no model execution."""
from pathlib import Path
import json
import hashlib
import html
from datetime import datetime
from zoneinfo import ZoneInfo

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
STAMP = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')

def read(name):
    return json.loads((UNIT / name).read_text(encoding='utf-8-sig'))

def write(path, text):
    path.write_text(text, encoding='utf-8', newline='\n')

def dump(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

records = read('INPUT_RECORDS_INDEPENDENT.json')
recovery = read('RECOVERY_AND_PROBE_INDEPENDENT.json')
forecast = read('root_audit_legacy_preflight_CAUSE_SUMMARY.json')
visual = read('root_probe_audit_VISUAL_REVIEW.json')
assert records['prepared_records'] == 20 and records['matched_feature_values'] == 4258240
assert records['context_pins_unchanged'] == 5119
assert recovery['existing_records_preserved'] == 12 and recovery['source_path_only_patches'] == 8
assert all(x['independent_calibration_status'] == 'NOT_CALIBRATED' for x in recovery['calibration_probe_independent'])
assert forecast['all1199']['failure'] == 12
assert len(visual['findings']) == 2

evidence_names = [
    'AUTHORIZATION.json', 'PREP_CONTEXT.json', 'PREP_FIRST_FAILURE.json',
    'PREP_RECOVERY_CONTEXT.json', 'PREP_RECOVERY_RESULT.json',
    'INPUT_RECORDS_INDEPENDENT.json', 'RECOVERY_AND_PROBE_INDEPENDENT.json',
    'baseline_RESULT.json', 'BASELINE_SAVED_INDEPENDENT.json',
    'root_audit_legacy_preflight_CAUSE_SUMMARY.json', 'CALIBRATION_PROBE_SCOPE.json',
    'CALIBRATION_PROBE_RESULT.json', 'root_probe_audit_RESULT.json',
    'root_probe_audit_VISUAL_REVIEW.json', 'root_probe_audit_FIRST_ERROR.json',
    'root_probe_audit_SECOND_ERROR.json', 'EARLY_CACHE_INDEPENDENT.json',
    'root_probe_audit_NIA_SL_WORD2732_REAL02_F.png',
    'root_probe_audit_NIA_SL_WORD0500_REAL03_F.png',
]
result = {
    'status': 'PARTIAL_INPUTS_VALIDATED_CALIBRATION_BLOCKED_NO_TRAINING',
    'updated_kst': STAMP,
    'implementation': ['Codex root', 'roi_data_budget'],
    'independent_review': ['roi_missing_protocol', 'roi_training_protocol'],
    'requested_records': 1199, 'prepared_records': 20,
    'input_numeric_contract': 'PASS_FOR_PREPARED20_NOT_ALL1199',
    'matched_feature_values': 4258240, 'hidden_points_perturbed': 101589,
    'protected_context_pins_verified': 5119,
    'legacy_calibration_forecast': forecast['all1199'],
    'new_full_frame_detector_runs': 2,
    'new_probe_confirmed_calibration_failures': 2,
    'new_probe_clip_ids': [x['clip_id'] for x in recovery['calibration_probe_independent']],
    'remaining_legacy_flags_unconfirmed_by_new_detector': 10,
    'baseline_reproduction': {'evaluated_development_clips': 300, 'top1_correct': 290,
        'top1_accuracy_percent': 100 * 290 / 300, 'top5_correct': 299,
        'logits_max_abs_difference': 0, 'training': 0},
    'candidate_training_runs': 0, 'new_candidate_accuracy': None,
    'extra_rest_wrist_filter': 'OFF', 'user_data_used': False,
    'production_changes': 0, 'scientific_conditions_changed': False,
    'readiness': 'FAIL_CALIBRATION_RULE_NOT_APPLICABLE_TO_ALL_REQUIRED_INPUTS',
    'remaining_semantic_review': 'Geometry loss and actual-crop detector loss need separate review; missingness alone is not an invalid training example.',
    'process_status': 'All preparation/probe/audit commands finished. No new training or bulk extraction running.',
    'next_single_action': 'Review a common boundary initialization rule for short or absent initial rest, preserving moving hands; validate on TRAIN cases and motion-preservation cases before revising frozen conditions. No per-word override or silent exclusion.',
    'evidence_sha256': {n: sha(UNIT / n) for n in evidence_names},
}
dump(UNIT / 'RESULT.json', result)

report = '''상반신 입력 준비·검증 결과

판정: 일부 입력의 계산 검증은 통과했지만, 화면 하단을 정하는 초기 자세 조건이 일부 원본 영상에 맞지 않아 학습은 시작하지 않았다.

1. 완료 범위
- 기존 공개 TRAIN 20영상의 관측을 재사용하여 전체 범위·전체 검출 후 상반신 마스크·실제 RGB 잘림 후 검출 및 기존 모델 비교용 입력을 준비했다. 요청된 전체 1,199개 준비 완료가 아니다.
- 준비된 20개 입력의 4,258,240개 채널값을 독립 계산과 비교하여 모두 일치했다. 영역 밖 101,589점의 값을 바꿔도 모델 입력이 변하지 않았다. 내부 미관측 구간·시간·부분 손 처리도 검산했다. 이 통과는 모든 필수 동작의 의미 보존을 보장하지 않는다.
- 추가 휴식 손목 제거 필터는 껐다. 원본·동결 조건 등 5,119개 보호 파일 해시가 유지됐다.
- 기존 모델의 원래 개발 검증을 재현했다. 1위 정답 290/300=96.67%, 상위5개 299/300. 저장된 출력과 차이 0. 기존 결과의 재현이며, 새 후보의 성능이나 실웹캠 정확도가 아니다.

2. 학습 전에 확인한 문제
정해 둔 하단 경계 규칙은 처음 1초 안에 양팔을 내리고 최소 0.3초 안정된 구간이 있어야 한다. 일부 원본은 시작 직후 이미 수어를 시작해 이 조건이 성립하지 않는다.
- 기존 저장 관측 1,199개 전수 점검: 1,187개 보정 성공, 12개 보정 불가 예상(학습10·개발2). 단어별 대표 100개만 봤을 때는 모두 통과했으므로 대표만으로 전수 준비 성공을 판단할 수 없었다.
- 예상 사례 중 학습 자료 2개를 전체 원본 화면에서 새로 검출하고, 저장 원후보 재생·독립 계산·원본 사진 확인까지 수행했다. 둘 다 동일한 보정 실패를 확인했다. 나머지 10개는 새 검출로 확정한 실패가 아니다.
- 백 REAL02: 첫 프레임부터 오른팔을 들기 시작하고 0.3초에는 손이 얼굴 가까이에 있다. 조건에 맞는 양팔 내림 구간이 없다.
- 마음에들다 REAL03: 시작에는 손이 아래에 있지만 곧 움직인다. 가장 안정적인 0~9프레임도 오른손목 이동이 어깨너비의 약10.94%로 고정 허용5%를 넘는다. 양어깨 변화는 약0.36% 미만이므로 실제 손 이동이 주요 원인이다.
이 두 사례는 수어가 잘못됐거나 학습에 쓸 수 없는 영상이라는 뜻이 아니다. 우리가 정한 자동 경계 설정 가정이 영상 시작 방식과 맞지 않는 것이다. 좌우 손 배정 고장으로 설명할 증거도 없었다. 이 선별된 두 사례로 전체 실패율을 추정하지 않는다.

3. 손이 잘리거나 빠지는 문제의 남은 검토
전체 검출 후 좌표를 제외하는 경우와 실제 RGB 화면을 자른 뒤 검출하는 경우를 따로 기록했다. 실제 잘림 조건의 전멸·수영 일부 장면은 손이 범위 안에 있어도 검출기가 한 손을 놓치는 기존 현상이 남는다. 예를 들어 전멸 일부 원후보는 같은 손에 겹쳐 검출됐다. 화면 경계가 손을 잘라서 없앤 경우와 다르다.
이런 미관측 자체는 실웹캠 환경을 학습에 반영하려는 목적에 포함된다. 따라서 미관측이 있으면 무조건 잘못된 자료라는 판정은 하지 않는다. 다만 의미 구별에 필요한 손 정보까지 장시간 없어지는지, 다른 손·몸·시간 정보로 구별 가능한지는 별도 검토가 남아 있다. 모든 필수 동작이 보존됐다고 아직 결론내리지 않았다.

4. 실행 오류와 복구 이력
- 준비 도구가 기존 영상 경로 8개를 찾지 못해 12개 작성 뒤 중단됐다. 최초 오류와 동결 코드를 보존하고, 같은 원본 해시·크기·ZIP CRC가 확인된 경로만 별도 회복 명세로 연결했다. 기존12개 불변·추가8개 준비 완료. 이 회복에서는 새 검출·새 영상 해제 0이다.
- 독립 검산 도구의 반환형 해석 오류와 기존 사진 보호로 인한 중단도 최초 코드·오류를 보존하고 검산 코드만 수정했다. 최종 검산 exit0. 원자료·보정 규칙·검출 결과는 바꾸지 않았다.
- 이 실행 오류들은 복구됐다. 현재 학습을 막는 문제는 위의 초기 자세 보정 조건이다.

5. 다음 한 작업
초기 준비 자세가 짧거나 없는 영상에도 적용할 공통 경계 결정안을 검토한다. 우선 학습 자료의 예상 실패10개와 기존 동작 보존 사례를 함께 확인한다. 개발2개는 고정안의 검증 대상으로 남긴다. 초기 정지 손목에만 의존하지 않는 몸 기준 경계 등 대안을 비교하되, 현재 발견한 실패만 통과하도록 임계값을 올리거나 단어별 예외를 만들지 않는다.
경계의 근거·결정 시점·실제 화면 잘림과의 일치·동작손 보존을 정리하여 사용자에게 보고한 후 조건을 수정한다. 입력 준비가 통과하면 기존 A/B/C 각3회와 동결 M0 비교 계획을 이어간다. 현재는 조건 수정이나 새 학습을 실행하지 않았다.

현재 실행 상태: 새 전체 화면 검출2개, 새로운 TCN 학습0회, 새 후보 성능 비교 미실행. 대규모 추출 및 학습 프로세스 없음. 기존 운영 모델·웹·DTW·사용자 기록 및 보류 자료는 변경하지 않았다.
담당: Codex root/roi_data_budget 구현·실행, roi_missing_protocol/roi_training_protocol 독립 검산·영상 확인, root 종합 검토.
'''
write(UNIT / 'RESULT_KO.txt', report)
sections = report.strip().split('\n\n')
cards = '\n'.join('<section><p>' + html.escape(s).replace('\n', '<br>') + '</p></section>' for s in sections[1:])
page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>상반신 입력 준비 검증 결과</title>
<style>body{font:17px/1.75 "Malgun Gothic",sans-serif;background:#f3f5f8;color:#192333;margin:0}main{max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:32px;line-height:1.4}section,figure{margin:18px 0;background:white;padding:24px 30px;border-radius:12px;border:1px solid #dce2e9}p{margin:0}img{width:100%;height:auto}figcaption{margin-top:12px}a{color:#1564b7}.status{background:#fff1d7;border-left:6px solid #c67c00;padding:16px 24px;font-weight:700}</style>
<main><h1>상반신 입력 준비·검증 결과</h1><div class="status">입력 계산 검증 통과 · 초기 경계 보정 문제 확인 · 학습 미시작</div>'''
page += cards
page += '''<figure><img src="root_probe_audit_NIA_SL_WORD2732_REAL02_F.png" alt="백 원본 시작 구간"><figcaption><b>백:</b> 처음부터 오른팔을 들어 안정된 양팔 내림 구간이 없다. 전체 원본 화면을 표시했다.</figcaption></figure>
<figure><img src="root_probe_audit_NIA_SL_WORD0500_REAL03_F.png" alt="마음에들다 원본 시작 구간"><figcaption><b>마음에들다:</b> 0.3초 이전부터 손이 움직여 정지 조건을 충족하지 않는다. 손 움직임 자체는 정상적인 수어 수행의 일부다.</figcaption></figure>
<section><p><a href="RESULT.json">종합 검증 기록</a> · <a href="INPUT_RECORDS_INDEPENDENT.json">입력 독립 검산</a> · <a href="RECOVERY_AND_PROBE_INDEPENDENT.json">복구 및 보정 실패 검산</a> · <a href="root_probe_audit_VISUAL_REVIEW.json">원본 장면 검토</a></p></section></main></html>'''
write(UNIT / 'RESULT_KO.html', page)

state_path = ROOT / 'research/front_keyphase_v1/ops/STATE.json'
state = json.loads(state_path.read_text(encoding='utf-8-sig'))
state['updated_date_kst'] = STAMP
state['current_stage'] = 'WORD100_ROI_PARTIAL_INPUTS_CALIBRATION_BLOCKED_NO_TRAINING'
state['word100_roi_training_20261017'].update({
    'status': result['status'], 'updated_kst': STAMP,
    'summary': '20입력의 계산·마스크·시간 계약 독립검산PASS. 기존1199관측 중 초기보정12개 불가예상, 그중TRAIN2개 새전체검출·독립사진검증으로 실패확인. 초기0.3초 안정 양팔내림 가정 불일치. 학습0·전수준비보류.',
    'next_single_action': '초기 준비 자세가 짧거나 없는 TRAIN 영상에도 적용할 공통 경계 결정안을 검토하고 동작손 보존과 함께 보고한다. 조건 수정·학습 전 사용자와 문제 확인. 단어별 예외·임계값 임의완화·실패자료 조용한 제외 금지.',
    'prepared_records': 20, 'requested_records': 1199,
    'legacy_calibration_flags': 12, 'confirmed_new_calibration_failures': 2,
    'remaining_legacy_flags_unconfirmed': 10,
    'new_detector_runs': 2, 'training': 0, 'candidate_comparisons': 0,
    'baseline_top1_correct': 290, 'baseline_development_clips': 300,
    'baseline_top1_percent': 100 * 290 / 300,
    'result': str(UNIT / 'RESULT.json'), 'report': str(UNIT / 'RESULT_KO.html'),
    'implementation': result['implementation'], 'reviewers': result['independent_review'],
    'process_status': result['process_status'],
})
# Keep the prior plan entry as historical evidence; the current task above supersedes it.
dump(state_path, state)

ctx_path = ROOT / 'WordsInSilence_PROJECT_CONTEXT.md'
ctx = ctx_path.read_text(encoding='utf-8-sig')
entry8 = '''### 2026-10-17 상반신 입력 20개 검산 완료 — 초기 보정 문제 확인, 학습 미시작
사용자 승인에 따라 입력 준비·선행 검증을 수행했다. `ops/word100_roi_training_20261017/RESULT.json`, `RESULT_KO.html` 참조. 기존 TRAIN20 입력 준비, 4,258,240채널값 독립 일치·영역밖101,589점 교란 불변·5,119보호핀 불변. 기존899학습/300개발 총1199관측 전수 사전점검에서 초기보정12개 불가예상(학습10/개발2); 그중 TRAIN 백REAL02·마음에들다REAL03를 새로 전체 검출해 둘 다 실패확정. 첫1초 안0.3초 안정양팔내림 조건이 실제 영상시작과 맞지 않음. 나머지10예상실패는 새검출 미확인. 두사례를 전체실패율로 해석하지 않는다. 원본사진·저장원후보 재생·독립조건검산 확인.
기존M0 개발300 재현290/300=96.67%,Top5 299/300,저장출력차이0. 새 후보 성능 아님. 준비도구 기존영상경로8개 오류는 첫증거보존 후 동일원본핀으로 복구하여 기존12+추가8개 준비; 검산도구 오류2건도 원자료/조건 변경없이 복구. 현재 문제는 초기경계보정. 실제잘림 전멸·수영의 영역내 검출누락 등 필수동작 보존검토도 남음; 미관측 자체를 잘못된학습자료로 단정하지 않음. 추가손목필터OFF. 새검출2개·새학습0회·후보비교미실행, 대량추출/학습프로세스없음. 구현root/roi_data_budget, 독립roi_missing_protocol/roi_training_protocol. 사용자자료·보류자료·운영·동결조건 보존.

'''
entry9 = '''### 2026-10-17 다음 — 초기 안정 자세가 없는 영상의 공통 경계 설정 검토
입력 준비 통과 전 문제 발견으로 학습을 시작하지 않았다. `ops/word100_roi_training_20261017/RESULT_KO.html`의 두 확정 사례와 나머지 TRAIN 보정불가 예상 사례를 토대로, 초기 정지 손목에만 의존하지 않는 공통 경계 결정안을 검토한다. 개발2개는 고정안의 검증으로 남긴다. 단어별 예외·사후 임계값 완화·899자료 조용한 감소 금지. 실제잘림 관측누락과 기하학적 잘림을 분리하고 필요한 동작/받침손 보존을 함께 검증한다. 문제와 조건 변경안을 사용자와 확인한 후 입력준비를 재개하며, 준비 통과 시 이미 승인된 A/B/C 각3회와 M0 비교를 이어간다. 현재 조건변경·학습·운영교체 미실행. 종료된 probe/검산 중복실행 금지.

'''
for header, entry in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인', entry8), ('## 9. 다음 단일 작업', entry9)]:
    assert ctx.count(header) == 1
    ctx = ctx.replace(header + '\n\n', header + '\n\n' + entry, 1)
write(ctx_path, ctx)

handoff = f'''# WordsInSilence 최신 인수인계

갱신 {STAMP}. 현재 목표는 전체 영상 검출 후 상반신 범위만 사용하고 실제 화면 밖 미검출을 반영한 TCN 비교다. 사용자는 입력 검증이 문제없으면 학습·비교까지 진행, 문제 발견 시 즉시 보고를 승인했다.

## 현재 판정
**일부 입력 계산 검증 완료, 초기 경계 보정 문제 확인, 학습 미시작.**
현재 단위 `research/front_keyphase_v1/ops/word100_roi_training_20261017/`의 `RESULT.json`, `RESULT_KO.html`이 결과 근거다. 이전 계획 단위 `word100_roi_training_plan_20261017`의 동결조건은 그대로 보존했다. 과거 계획 승인대기는 최신 직접 승인으로 해소됐지만 이번 준비 문제는 해결 전이다.

## 완료와 미완료
- TRAIN20개 준비, 4,258,240입력값 독립 일치, 영역밖101,589점 교란 불변, 보호파일5,119개 불변. 전체1,199개 준비 완료 아님.
- 기존 관측1199(학습899/개발300) 사전점검: 초기보정12개 불가예상(학습10/개발2). 대표100개만 검사했을 때는 모두통과했다.
- 그중 TRAIN 백REAL02·마음에들다REAL03만 새 전체원본 검출2개 실행. 초기1초 안0.3초 안정된 양팔내림 구간을 요구하는 조건이 성립하지 않아 두사례 실패 확정. 저장원후보148프레임/290손 재생·독립윈도우계산·원본사진 확인. 나머지10개는 새검출미확인, 전체실패율 추정금지.
- 기존M0 원래개발300 재현 Top1 290/300=96.67%,Top5 299/300, 저장출력차이0. 새후보나웹캠정확도아님.
- 실제잘림 전멸·수영 일부영역내 검출누락 남음. 미관측은 후보C의 의도된 환경변형일수있으므로 자동자료오류로 단정하지 않는다. 필수동작 의미보존 검토는 별도로 남음.
- 추가휴식손목제거필터OFF. 새TCN학습0·후보성능비교미실행. 모든 이번 실행 종료, 대량추출/학습프로세스없음.

## 복구 이력
prep.py는 기존 영상 경로8개를 못찾아12records뒤중단. PREP_FIRST_FAILURE/PREP_STATUS 원본보존. prep_recovery.py가 동일원본해시·크기·ZIPCRC인 경로만 보완해20records완료. 현재회복결과는 PREP_RECOVERY_RESULT.json. 기존12개·PREP_CONTEXT불변, 회복새검출/영상해제0. 독립검산도구 API해석/기존사진보호 오류2건도 최초증거보존 후 검산코드만복구,최종exit0. 이들은현재장애가아니며 초기보정조건이실질문제다.

## 다음 한 작업
초기 정지 자세가 짧거나 없는 TRAIN 영상에도 적용할 공통 경계 결정안을 검토한다. TRAIN예상실패10개와 동작보존사례를 함께 사용하고 개발2개는 고정안검증으로남긴다. 초기정지손목에만의존하지않는몸기준경계등대안의 근거·결정시점·실제잘림일치·필수손보존을보고한다. 현재 조건을 임의완화하거나 단어별예외/실패자료제외로학습하지 않는다. 사용자와 문제/조건변경을 확인한뒤 입력준비통과→실행조건동결→A/B/C각3회 비교를이어간다.

## 유지할 실험 조건
AIHub100단어, 학습ID01~09 899개/개발ID11~13 300개. 원본영상전체검출→joint_v3→상반신마스크와 실제RGB잘림→독립검출 두보기. 내부미관측/PTS/부분손 유지. 동일280채널TCN, A전체/B마스크/C두보기50:50, 각3고정seed 계획. 세부동결조건은 PROPOSED_EXPERIMENT.json 참조. 조건 수정은 별도버전으로 기록한다.
M0 `research/front_keyphase_v1/runs/word100_tcn_v1/whole_context/best.pt` SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 기존운영TCN/DTW/웹·원본·sources·과거동결결과 유지. 사용자/멘토/NIKL/KSL/말뭉치 학습·튜닝·후보선택 없음. ID10미배정/ID14~18보류. 카메라·다운로드·운영교체·Git작업 없음.

## 담당과 검증
Codex root/roi_data_budget 구현·실행. roi_missing_protocol 독립입력/복구/보정계산; roi_training_protocol 기존모델재현/전수사전점검/원본사진·원후보재생. ROOT도 두 원본 비교사진 확인. INPUT_RECORDS_INDEPENDENT.json, RECOVERY_AND_PROBE_INDEPENDENT.json, root_probe_audit_VISUAL_REVIEW.json을 재사용하며 불필요한 재추출/재학습은 하지 않는다.
'''
write(ROOT / 'NEW_CHAT_HANDOFF.md', handoff)
print(json.dumps({'status': result['status'], 'report': str(UNIT / 'RESULT_KO.html'), 'updated_kst': STAMP}, ensure_ascii=False))
