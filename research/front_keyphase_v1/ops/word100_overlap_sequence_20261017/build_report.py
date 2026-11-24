"""Package completed analysis and visual review; no model execution."""
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

U = Path(__file__).resolve().parent
OPS = U.parent
ROOT = OPS.parents[2]


def read(p):
    return json.loads(Path(p).read_text('utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, d):
    with p.open('x', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


v = read(U / 'VISUAL_MANIFEST.json')
recount = read(U / 'RECOUNT.json')
audit = read(U / 'AUDIT.json')
assert recount['status'] == 'PASS'
stamp = datetime.now().astimezone().isoformat(timespec='seconds')
files = [x['file'] for x in v['flag_pages']] + [f for x in v['gap_pages'] for f in x['files']] + [x['file'] for x in v['sequence_pages']]
assert len(files) == 20 and all((U / f).exists() for f in files)
review = {
    'at': stamp, 'reviewer': 'Codex root visual inspection; not expert annotation or independent human review',
    'all_selected_flag_frames_reviewed': True,
    'flag_intervals': len(audit['flags']), 'flag_frame_view_pairs': v['flag_unique_frames'],
    'gap_context_frame_view_pairs': 2 * sum(len(x['reviewed_context_frames']) for x in v['gap_pages']),
    'detailed_sequence_images': sum(len(x['frames']) for x in v['sequence_pages']),
    'earlier_all36_contact_sheets_reviewed': ['WORD0500_CORE_TRAIN9.jpg', 'WORD0003_CORE_TRAIN9.jpg', 'WORD1642_CORE_TRAIN9.jpg', 'WORD1884_CORE_TRAIN9.jpg'],
    'earlier_contact_sheet_samples': 360,
    'not_claimed': ['all frames visually labelled', 'ground-truth landmark accuracy', 'user-error causality', 'new recognition improvement'],
    'raw_label_disagreements': {
        'native': 38, 'sensor': 44,
        'finding': 'Reviewed assignments are consistent with visible arm/hand identity; no clear new anatomical swap confirmed in flagged sample. Raw handedness disagreement is not a remaining swap.'},
    'held_detections': [
        {'clip': 'NIA_SL_WORD1642_REAL02_F', 'view': 'native', 'frames': [50, 51, 52], 'finding': 'Two landmark clouds on the same visible right hand; other left hand occluded.'},
        {'clip': 'NIA_SL_WORD1884_REAL05_F', 'view': 'native', 'frames': [97, 99], 'finding': 'Two clouds near right hand/face; actual left hand lower at chest not recovered.'},
        {'clip': 'NIA_SL_WORD1884_REAL06_F', 'view': 'native', 'frames': [158], 'finding': 'Two clouds on same lowered left hand; other right hand not recovered.'},
        {'clip': 'NIA_SL_WORD1884_REAL06_F', 'view': 'sensor', 'frames': [158], 'finding': 'Same duplicate-left-hand interpretation.'}],
    'gap_findings': [
        {'clip': 'NIA_SL_WORD0500_REAL06_F', 'frames': [55, 68], 'sensor_left_missing_frames': 14,
         'interpretation': 'Inside crop, hands close/overlapping at chest. Raw detector returns right only. Assignment has no left candidate to restore.',
         'native_left_present_frames': [55, 56, 61, 62, 63, 64, 65, 66, 67, 68],
         'native_also_missing_frames': [57, 58, 59, 60],
         'limitation': 'RGB shows partial occlusion; does not isolate detector crop/tracking causes or prove classifier causality.'},
        {'clip': 'NIA_SL_WORD0003_REAL01_F', 'frames': [35, 35], 'interpretation': 'Visible moving hand, brief raw loss; motion blur plausible, not experimentally isolated.'},
        {'clip': 'NIA_SL_WORD1642_REAL01_F', 'frames': [83, 98], 'interpretation': 'Left rests below crop during right-face stage; expected out-of-view absence, not evidence of active hand loss.'},
        {'clip': 'NIA_SL_WORD1884_REAL03_F', 'frames': [85, 98], 'interpretation': 'Left rests below crop during right-face stage; expected out-of-view absence.'}],
    'display_note': 'Full-source RGB backdrop for BOTH native and sensor overlay. Sensor landmarks were detected only inside the frozen upper-body crop. Full backdrop is for diagnosis, not the actual sensor input extent.',
    'evidence_pins': {f: sha(U / f) for f in files},
    'manifest_sha256': sha(U / 'VISUAL_MANIFEST.json')
}
write_json(U / 'VISUAL_REVIEW.json', review)

sequence = [
    ('마음에 들다', '턱·아래 얼굴 근처의 오른손 → 가슴 앞에서 두 손이 모이는 동작', 'WORD0500'),
    ('수어', '가슴 앞에서 양손이 번갈아 위치를 바꾸는 동작', 'WORD0003'),
    ('가평군수어통역센터', '앞쪽 양손 동작 → 번갈아 움직임 → 오른손 얼굴 근처·왼손 내림 → 가슴 앞 두 손 동작', 'WORD1642'),
    ('강진군수어통역센터', '초반 손 모양·양손 관계가 가평과 다름 → 일부 비슷한 중간·후반 흐름', 'WORD1884')
]
plan = [
    {'priority': 1, 'name': '손이 안 보이는 이유를 구분하는 공통 관측 품질 처리',
     'description': '영상의 모든 손을 억지로 채우지 않는다. 화면 밖/내려놓음, 화면 안에서 가려짐·검출 실패, 좌우 판정 불확실을 분리하고 판단 불가 상태를 둔다. 팔 연결·앞뒤 관측은 근거로 쓰되 없어진 손가락 좌표를 만들어 넣지 않는다.',
     'gate': '공개 장면의 수동 판독표와 비교. 화면 밖 휴식 손을 정상 복원으로 세거나 다른 손을 복제하는 경우가 없어야 함. 마스크·시간 보존 검산. 현재 프로그램에는 아직 적용하지 않음.'},
    {'priority': 2, 'name': '앞·중간·뒤 정보를 따로 남기는 TCN 단일 후보',
     'description': '현재 TCN은 시간 흐름을 학습하지만 끝에서 전체 평균으로 요약한다. 앞·중간·뒤 특징을 따로 요약해 이어 붙이는 후보는 부분 동작이 비슷해도 그 앞에 무엇을 했는지 구분하도록 돕는 가설이다. 임의의 단어별 규칙이나 센터명 제외는 사용하지 않는다.',
     'gate': '사용자 검토 후에만 계획을 확정. 구조 변경 전 공개 TRAIN의 순서 단서에 대한 고정모델 진단을 사전 고정하며, 인위적인 순서 변경은 정답률 시험이 아닌 민감도 검사로만 보고. 후보 학습 시 같은 입력·분할·예산·3seed, 공개 TRAIN 내부 선택/반복 DEV300 비교. 새 사람 개선으로 부르지 않음.'},
    {'priority': 3, 'name': '재검출·보간을 무작정 추가하지 않음',
     'description': '2026-10-04 pose ROI 재검출 후보는 고유 회복 0, 불량 적합 2, 판단 불가 4로 미채택이었다. 이번에 crop 크기나 임계값을 바꾸어 재탐색하지 않는다. 완전히 가린 손은 확대만으로 관찰할 수 없다. 긴 겹침을 직선으로 메우면 손 모양 변화가 지워질 수 있다.',
     'gate': '새 검출 방법은 과거와 무엇이 다른지, 올바른 실제 손 회복을 어떻게 판정할지 별도 근거가 있을 때만 제안.'}
]
result = {
    'status': 'ANALYSIS_COMPLETE_AWAITING_USER_REVIEW_BEFORE_TRAINING', 'at': stamp,
    'public_train_clips': 36, 'words': 4, 'signers': 9, 'core_frames_per_view': 2072,
    'summary': recount['summary'], 'sequence_observations': sequence, 'proposed_common_methods': plan,
    'visual_review': str(U / 'VISUAL_REVIEW.json'), 'recount': str(U / 'RECOUNT.json'),
    'stored_training_prediction_reuse': {'masked_top1': 36, 'sensor_top1': 36, 'denominator_each': 36,
       'limitation': 'Already trained clips and saved scores; not new generalization/performance test.'},
    'conclusion': 'Confirmed public in-frame raw observation gap; existing side correction/duplicate gating should be preserved. Shared partial motion with distinct full sequence is observed. User-error cause and gain from proposed changes remain unproven.',
    'new_training': 0, 'new_inference': 0, 'new_detection': 0, 'user_recording_reads': 0,
    'production_changes': 0, 'active_process': None,
    'implementation_and_verification': 'Codex root, separate arithmetic code and visual inspection; no other agent/expert review.',
    'first_failure': str(U / 'FIRST_ERROR.txt'), 'recovery': str(U / 'RECOVERY.json'),
    'supporting_pins': {
        str(OPS / 'pose_roi_probe_20261004/RESULT_KO.md'): sha(OPS / 'pose_roi_probe_20261004/RESULT_KO.md'),
        str(ROOT / 'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py'): sha(ROOT / 'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py')}
}
write_json(U / 'RESULT.json', result)

text = '''공개 손 겹침·좌우 배정·동작 순서 분석
2026-10-17 / 분석 완료, 새 학습 전 사용자 검토

결론
현재 근거는 “좌우를 더 과감하게 바꾸자”보다 “실제로 못 본 손과 쉬고 있는 손을 구별하고, 전체 동작 흐름을 보존하자”를 지지한다.
공개 자료에서 손 겹침 구간의 관절 누락은 확인했다. 하지만 이것이 사용자 촬영 오답의 확정 원인은 아니며, 개선 정확도는 아직 측정하지 않았다.

1. 무엇을 확인했나
마음에 들다·수어·가평군수어통역센터·강진군수어통역센터 4표현, TRAIN 촬영자 ID01~09의 36영상. 사용자 촬영은 읽지 않았다.
주석 핵심구간 총2,072프레임을 전체 원본 검출(native)과 상반신 화면 검출(sensor)에서 대조했다. 두 보기는 같은 영상이므로 72개의 독립 영상이 아니다.
관절 원검출→좌우 배정→저장 관측을 별도로 검산했다. 좌우 이름 불일치와 검출 보류32구간의 89프레임·보기 조합 전부, 각 표현의 선택한 최장 누락 구간 앞뒤122장, 4표현의 상세 순서48장을 시각 확인했다. 이전36영상의360장 표본도 확인했다. 모든 영상의 모든 프레임에 정답 관절을 붙인 검사는 아니다.

2. 마음에 들다: 왼손 후보 자체가 없던 구간
TRAIN06의 손이 가슴 앞에서 겹치는 구간에서 sensor 왼손이14프레임(약0.47초) 연속 빠졌다. 55~68프레임 모두 원검출이 오른손 하나만 출력했다. 왼손이 검출됐는데 좌우 배정 과정에서 버린 상황이 아니다.
이 손은 상반신 영역 안에 있다. 하단 경계를 더 내리거나 중복 필터를 느슨하게 해도 없는 왼손 후보를 되살릴 수 없다.
같은14프레임 중 전체 화면 검출에서는10프레임에 왼손이 있었지만4프레임은 함께 빠졌다. 화면 문맥/검출 경로에 따른 차이와 가림이 관련될 가능성이 있지만 원인을 하나씩 분리한 실험은 아니다. 전체 화면에서 검출하면 항상 해결된다는 뜻도 아니다.

3. 좌우 배정·중복 방지는 이번 표본에서 유지하는 편이 타당
원검출의 Left/Right 이름과 최종 배정이 달랐던 검출은 native38개, sensor44개였다. 원본 영상과 팔 연결을 대조했을 때 기존 배정이 잘못된 이름을 교정하는 장면들이었다. 검토한 flag에서 새로 확정한 해부학적 좌우 교환 오류는 없었다. 전수 정답 주석이 없으므로 오류0%라는 뜻은 아니다.
원검출 두 개 중 하나를 보류한 것은 native6프레임, sensor1프레임이었다. 7개 모두 같은 실제 손 위에 관절 묶음이 중복된 장면으로 보였다. 이를 빈 반대손에 강제로 넣으면 다른 손을 만들어내는 오류가 생긴다.
MediaPipe handedness score는 좌우 분류 점수이며 손가락 좌표가 정확하다는 점수가 아니다.

4. 손 미관측 수를 오류 수로 읽으면 안 됨
각 보기의2,072프레임×2손=4,144개의 손·프레임 자리 중, native 미관측129(원검출 부재123/배정 보류 관련6), sensor 미관측379(원검출 부재378/보류 관련1)였다.
이 숫자는 손가락 전체 정답 오차나 수어 실패 횟수가 아니다. 손바닥5점이 함께 유효한지의 관측 집계이며, 같은 한 번의 누락이 여러 프레임으로 세어진다.
가평TRAIN01 왼손16프레임, 강진TRAIN03 왼손14프레임 누락은 오른손이 얼굴 근처에서 동작하는 동안 왼손을 내려놓아 화면 밖에 있던 구간이었다. 사용하지 않는 손을 보이지 않게 두는 정상 상황이 포함된다. 수어TRAIN01은 움직이는 왼손이1프레임 잠깐 빠진 별도 사례다.

5. 혼동된 표현들의 전체 흐름
마음에 들다: 턱·아래 얼굴의 오른손 → 가슴 앞에서 두 손이 모이는 동작.
수어: 가슴 앞에서 양손이 번갈아 위치를 바꾸는 동작.
가평군수어통역센터: 앞쪽 양손 동작 → 번갈아 움직임 → 오른손 얼굴 근처·왼손 내림 → 가슴 앞 두 손 동작.
강진군수어통역센터: 초반 손 모양·양손 관계가 가평과 다르며, 중간·후반 일부 흐름은 비슷하다.
이것은 영상에서 관찰한 설명이며 수어 전문가가 확정한 어휘·형태소 분해가 아니다. 얼굴에서 가슴 앞으로 이어지는 일부 장면만 보면 비슷해질 수 있지만 그 앞의 동작과 전체 구성은 다르다. 단순히 같은 동작의 순서만 뒤집힌 표현이라는 뜻도 아니다.
현재 TCN도 시간 순서를 이미 처리한다. 마지막에 전체 시간 평균으로 요약하는 구조는 확인했지만, 이 구조 때문에 오답이 났다는 것은 아직 가설이다.

6. 이번 분석에서 성능 개선을 주장할 수 없는 이유
기존 C 모델의 저장 점수를 다시 계산해 순위만 확인했을 때, 이36개 공개 TRAIN 영상은 masked/sensor 각각36/36 정답1위였다. 새로 채점하지 않았다. 이 영상들은 이미 학습에 사용했으므로 새로운 사람 정확도100%라는 뜻이 아니다.
따라서 “관절 누락이 존재한다”는 확인과 “그 누락 때문에 틀렸다”는 인과 입증은 다르다. 사용자 촬영을 읽거나 점수에 맞춰 조건을 바꾸지 않았다.

7. 제안하는 공통 보완 순서 — 아직 미적용
첫째, 시간별·손별로 화면 밖/휴식과 화면 안 가림·검출 실패, 좌우 불확실을 구별하는 관측 품질 처리를 검증한다. 애매하면 판단 불가로 남기고 관측 마스크·시간을 보존한다. 없어진 손가락 좌표나 임의의 정답률은 만들지 않는다. 모든 프레임에서 양손을 요구하지 않는다.
둘째, 전체 평균 하나 대신 앞·중간·뒤 특징을 나눠 보존하는 TCN 후보를 검토한다. 쉽게 말해 한 장면만 비슷한지보다 앞에서 무엇을 한 뒤 그 장면에 도달했는지를 남기는 방식이다. 시간 순서를 새로 처음 배우게 한다는 뜻은 아니다. 공개 TRAIN의 순서 단서 진단과 실행 조건을 먼저 고정하고, 사용자 검토 후 한 변경씩 비교한다.
학습한다면 같은100단어·자료분할·학습예산·3seed로 비교하고, 공개 TRAIN 내부에서 조건을 정한다. 반복 사용한 DEV300은 개발 검증으로 표기한다. 전체 성능과 대상표현·단손·양손의 회귀를 함께 보고, 최종 일반화 평가는 별도로 남긴다. 사용자·멘토 촬영은 학습/튜닝/후보선택에 사용하지 않는다.
셋째, 겹친 손을 확대하거나 긴 빈 구간을 직선으로 메우는 방법은 우선 채택하지 않는다. 10월4일 pose ROI 검사는 고유 회복0, 불량 적합2, 판단 불가4로 미채택이었다. 같은 자료에 crop/임계값만 바꿔 반복하지 않는다. 완전히 가려진 손의 실제 모양은 현재 이미지에서 확인할 수 없다.
촬영 간격 보강은 사용자 요청대로 보류한다. 이번 결과와 보완 방향을 함께 확인한 다음 학습 여부를 정한다.

8. 검증 및 보존
Codex가 분석과 시각 검토를 수행했고, 별도 검산 코드로360개 자료 해시, 보호된10개 파일, 관절 채널326,760개 정확 일치, 미관측 집계,89개 flag 및 저장 점수 순위를 확인했다. 다른 에이전트나 수어 전문가의 독립 검토는 아니다.
첫 실행의 경로 오류(raw와 배정 결과가 다른 폴더에 저장된 사례)는 FIRST_ERROR.txt에 보존했다. 경로 해석만 수정해 동일36개 전체를 처리했으며 조건 변경·조용한 제외0.
새 학습0, 새 분류0, 새 검출0, 사용자 촬영 읽기0, 웹/TCN/DTW 적용 변경0. 모든 실행 완료. 기존C/M0와 보호 코드의 해시 유지.
사진 설명: 주황=해부학적 왼손, 하늘=오른손, 분홍=보류. 글자 L/R>는 원래 이름에서 최종 배정으로 바뀐 관계. sensor도 원본 전체 사진 위에 관절을 표시한 것은 잘린 영역 밖 상황을 함께 설명하기 위함이며 실제 검출 입력은 고정 상반신 범위였다.
공식 문서: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
'''
with (U / 'RESULT_KO.txt').open('x', encoding='utf-8') as f:
    f.write(text)


def photo(file, caption):
    return f'<figure><a href="{file}" target="_blank"><img loading="lazy" src="{file}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)} · 클릭하면 원본 크기</figcaption></figure>'


body = '''<header><span class="tag">공개자료 진단 · 학습 미실행</span><h1>손이 안 보이는 이유와<br>동작 순서를 나눠 확인했습니다.</h1><p>2026.10.17 · 4표현 × TRAIN 9명 = 36영상 · 사용자 촬영 읽기 0</p></header>
<main><section><h2>이번 결론</h2><p class="lead">좌우를 더 과감하게 바꾸기보다, <b>실제 검출 누락과 쉬고 있는 손을 구별</b>하고 <b>전체 동작 흐름을 보존</b>하는 방향이 타당합니다.</p><p>공개 영상에서 겹침 중 관절 누락을 확인했습니다. 다만 사용자 오답의 확정 원인이나 보완 후 성능 향상은 아직 입증하지 않았습니다.</p></section>
<section><h2>1. ‘마음에 들다’에서 확인한 누락</h2><div class="stat"><b>14프레임 · 약 0.47초</b><span>가슴 앞 두 손이 겹칠 때, 왼손 관절이 연속 미관측</span></div><p>상반신 검출 경로에서는 오른손 하나만 나왔습니다. 왼손을 잘못 배정해서 버린 상황이 아니므로, 좌우 배정 필터를 완화해도 복구할 후보가 없습니다. 손은 영역 안에 있어 하단 경계를 더 내리는 것으로 해결할 문제도 아닙니다.</p><p>같은 구간의 전체 화면 검출은 14프레임 중 10프레임에서 왼손을 찾았지만, 4프레임은 함께 놓쳤습니다. 전체 화면 검출도 완전한 해결책은 아닙니다.</p>'''
body += photo('WORD0500_gap_02.jpg', '마음에 들다 TRAIN06 · 같은 장면에서 native와 sensor의 원검출 대조')
body += '''<details><summary>누락 시작과 회복 장면도 보기</summary>'''
body += photo('WORD0500_gap_01.jpg', '누락 전·초반') + photo('WORD0500_gap_03.jpg', '누락 후반·회복') + '</details></section>'
body += '''<section><h2>2. 기존 좌우 보정은 보존할 근거가 있습니다</h2><div class="cols"><article><h3>원래 좌우 이름과 다름: 82개 검출</h3><p>native 38 + sensor 44. 영상의 팔 연결과 대조하면 기존 배정이 잘못된 원래 이름을 교정하는 장면들이었습니다. 검토한 flag에서 새로 확정한 좌우 뒤바뀜은 없었습니다.</p></article><article><h3>보류된 중복 후보: 7프레임·보기</h3><p>native 6 + sensor 1. 같은 실제 손 위에 관절이 두 묶음 찍힌 장면들이었습니다. 빈 반대손 칸에 강제로 넣으면 손을 잘못 만들어내게 됩니다.</p></article></div><p>정답 관절 주석이 없어 “좌우 오류 0%”라고 말할 수는 없습니다. 모든 좌우 불일치·두 손 중 보류 flag 32구간, 89프레임·보기 조합을 시각 검토했습니다.</p><details><summary>전체 flag 장면 6장 보기</summary>'''
body += ''.join(photo(x['file'], f"좌우 불일치·중복 보류 근거 {i+1}") for i, x in enumerate(v['flag_pages'])) + '</details></section>'
body += '''<section><h2>3. 안 보이는 손이 모두 문제인 것은 아닙니다</h2><p>가평·강진 센터 표현은 오른손을 얼굴 근처에서 움직이는 동안 왼손을 내려놓습니다. 이때 화면 밖 왼손이 안 잡히는 것은 사용 환경과 맞는 상황입니다.</p><table><thead><tr><th>핵심구간 2,072프레임 × 2손</th><th>전체 화면 검출</th><th>상반신 화면 검출</th></tr></thead><tbody><tr><td>손 미관측 자리</td><td>129 / 4,144</td><td>379 / 4,144</td></tr><tr><td>그중 원검출 부재</td><td>123</td><td>378</td></tr><tr><td>그중 배정 보류 관련</td><td>6</td><td>1</td></tr></tbody></table><p><b>실패 횟수나 손동작 오차율이 아닙니다.</b> 손바닥 5점의 관측 여부이며, 정상 휴식과 여러 프레임에 걸친 같은 누락을 포함합니다. 전체 프레임에서 양손 검출을 강제할 근거로 쓰면 안 됩니다.</p><details><summary>정상 손 내림과 짧은 검출 누락의 대조</summary>'''
for g in v['gap_pages'][1:]:
    for file in g['files']:
        body += photo(file, f"{g['clip_id']} · 선택한 누락 구간과 앞뒤")
body += '</details></section><section><h2>4. 일부 장면은 닮았어도 전체 흐름은 다릅니다</h2><table><thead><tr><th>표현</th><th>관찰한 흐름</th></tr></thead><tbody>'
body += ''.join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(desc)}</td></tr>' for name, desc, _ in sequence)
body += '''</tbody></table><p>센터 표현의 중간·후반에는 얼굴 근처에서 가슴 앞 두 손 동작으로 이어지는 일부 비슷한 장면이 있습니다. 하지만 앞부분과 전체 구성은 다릅니다. 단순히 같은 동작을 거꾸로 한 단어들이라는 뜻은 아닙니다.</p><p>현재 TCN도 순서를 학습합니다. 끝에서 전체 시간 평균으로 요약하는 구조는 확인했지만, 이 구조가 이번 오답의 원인이라는 결론은 아직 낼 수 없습니다. 아래는 영상 관찰 설명이며 전문적인 수어 형태소 분석이 아닙니다.</p>'''
for name, desc, code in sequence:
    body += f'<details><summary>{name} — 시간순 장면 보기</summary>' + photo(f'{code}_sequence.jpg', name + ' TRAIN01 · 왼쪽→오른쪽, 위→아래 순서') + '</details>'
body += '''</section><section><h2>5. 성능이 개선됐다는 결과는 아닙니다</h2><p class="lead">이 36개 공개 학습 영상은 기존 C가 이미 <b>36/36개를 1위</b>로 맞혔습니다.</p><p>masked·sensor 각각의 저장 점수만 검산했습니다. 이미 학습한 영상이므로 새로운 사람에 대한 정확도 100%가 아닙니다. 관절 누락이 존재한다는 사실만으로 오분류의 원인을 확정할 수 없습니다.</p></section><section><h2>6. 제안하는 공통 보완 순서</h2>'''
for p in plan:
    body += f'<article class="plan"><h3>{p["priority"]}. {p["name"]}</h3><p>{p["description"]}</p><p><b>검증 조건:</b> {p["gate"]}</p></article>'
body += '''<p class="lead">촬영 간격 보강은 보류합니다. 이번 결과를 함께 검토한 뒤 학습 여부와 비교 조건을 확정합니다.</p></section><section><h2>검증 기록과 범위</h2><p>별도 검산 코드: 360개 자료 해시, 보호된 10개 파일, 326,760개 관절 좌표 채널 일치, 미관측 집계·89개 flag·저장 순위 PASS. Codex의 별도 코드 검산이며, 다른 에이전트나 전문가의 독립 검토는 아닙니다.</p><p>처음에는 재사용 자료의 raw와 assignment 폴더가 달라 경로 오류가 났습니다. 최초 오류를 보존하고 경로 해석만 수정했으며, 36개 모두 포함했습니다.</p><p><b>새 학습 0 · 새 분류 0 · 새 검출 0 · 사용자 촬영 읽기 0 · 운영 적용 변경 0.</b> 기존 C·M0 및 보호 코드 해시가 유지됐습니다.</p><p>사진의 주황=해부학적 왼손, 하늘=오른손, 분홍=보류. L/R&gt;는 원래 이름→최종 배정입니다. sensor도 진단을 위해 원본 전체 RGB 위에 표시했으며, 실제 검출 입력은 고정된 상반신 범위였습니다.</p><p>MediaPipe의 handedness 점수는 좌우 분류 점수입니다. 관절 정확도 점수와 다릅니다. <a href="https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md">공식 문서</a></p><nav><a href="RESULT_KO.txt">전체 설명</a><a href="RESULT.json">결과 JSON</a><a href="RECOUNT.json">별도 검산</a><a href="VISUAL_REVIEW.json">시각 검토 기록</a><a href="FIRST_ERROR.txt">최초 오류</a></nav></section></main>'''
page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 손 겹침·순서 분석</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#172c40;font:17px/1.75 'Malgun Gothic',sans-serif}header,main{max-width:1120px;margin:auto}header{padding:60px 32px 34px}h1{font-size:40px;line-height:1.35;margin:18px 0}h2{font-size:25px;margin:0 0 20px}h3{font-size:19px;margin:0 0 10px}p{margin:10px 0 18px}.tag{color:#086b7a;font-weight:700}section{background:white;border:1px solid #dce5ee;border-radius:18px;padding:32px;margin:0 18px 24px}.lead{font-size:21px}.stat{background:#e8f5f6;border-left:5px solid #148495;padding:20px;display:grid;margin-bottom:20px}.stat b{font-size:28px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{padding:20px;background:#f2f6fa;border-radius:10px}.plan{margin:18px 0}table{border-collapse:collapse;width:100%;margin:22px 0}th,td{text-align:left;padding:14px;border-bottom:1px solid #d9e2eb}th{background:#eff5f9}figure{margin:22px 0}img{display:block;width:100%;border:1px solid #cbd6df}figcaption{font-size:14px;color:#41576b;padding:8px 0}summary{cursor:pointer;font-weight:700;padding:14px 0}details{border-top:1px solid #d9e2eb}a{color:#096e9b}nav{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:700px){h1{font-size:29px}.cols{grid-template-columns:1fr}section{padding:22px}table{font-size:14px}th,td{padding:8px}}
</style>''' + body + '</html>'
with (U / 'RESULT_KO.html').open('x', encoding='utf-8') as f:
    f.write(page)
print(json.dumps({'result': str(U / 'RESULT_KO.html'), 'status': result['status'], 'review_images': len(files)}, ensure_ascii=False))
