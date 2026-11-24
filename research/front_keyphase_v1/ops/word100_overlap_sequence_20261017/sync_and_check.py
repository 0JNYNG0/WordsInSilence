"""Save completed diagnosis in current handoff; preserve prior task history."""
import hashlib
import json
import shutil
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

U = Path(__file__).resolve().parent
OPS = U.parent
ROOT = OPS.parents[2]
UNIT = U.name


def read(p):
    return json.loads(Path(p).read_text('utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


result = read(U / 'RESULT.json')
recount = read(U / 'RECOUNT.json')
visual = read(U / 'VISUAL_REVIEW.json')
assert recount['status'] == 'PASS' and visual['all_selected_flag_frames_reviewed']
assert visual['flag_frame_view_pairs'] == 89
assert visual['gap_context_frame_view_pairs'] == 122
assert visual['detailed_sequence_images'] == 48
assert recount['summary']['sensor']['missing_hand_slots'] == 379
assert recount['summary']['native']['missing_hand_slots'] == 129
assert all(x['top1'] == 36 for x in recount['stored_train_scores'].values())
for p, expected in recount['protected_pins'].items():
    assert sha(p) == expected, p
for p, expected in recount['score_pins'].items():
    assert sha(p) == expected, p
for p, expected in visual['evidence_pins'].items():
    assert sha(U / p) == expected, p
for p, expected in result['supporting_pins'].items():
    assert sha(p) == expected, p


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ('href', 'src') and not v.startswith(('http', '#')):
                self.links.append(v)


parser = Links()
parser.feed((U / 'RESULT_KO.html').read_text('utf-8'))
assert all((U / p).is_file() for p in parser.links)
stamp = datetime.now().astimezone().isoformat(timespec='seconds')
before = U / 'before'
before.mkdir(exist_ok=False)
for p in (ROOT / 'WordsInSilence_PROJECT_CONTEXT.md', ROOT / 'NEW_CHAT_HANDOFF.md', OPS / 'STATE.json'):
    shutil.copy2(p, before / p.name)

next_action = '사용자와 공개 겹침·좌우·순서 분석 결과 및 공통 보완안 검토. 촬영간격 보강 보류. 다음 후보는 시간별 손 관측 품질 구분, 이후 조건부 앞/중간/뒤 요약 보존 TCN. 자동학습·운영변경 금지; 구체 조건을 함께 확인한 뒤 실행.'
state = read(OPS / 'STATE.json')
assert UNIT not in state
state['updated_date_kst'] = stamp
state['current_stage'] = 'WORD100_OVERLAP_SEQUENCE_ANALYSIS_COMPLETE_REVIEW_BEFORE_TRAINING'
state['next_single_action'] = next_action
state['resume_command'] = 'Read latest PROJECT_CONTEXT sections3/8/9, NEW_CHAT_HANDOFF and word100_overlap_sequence_20261017. Reuse completed audit/recount/visual evidence. No new training until user review. Frame-interval augmentation remains on hold.'
entry = dict(result)
entry.update({'unit': 'ops/' + UNIT, 'updated_kst': stamp,
              'report': str(U / 'RESULT_KO.html'), 'text_report': str(U / 'RESULT_KO.txt'),
              'result': str(U / 'RESULT.json'), 'next_single_action': next_action,
              'final_check': str(U / 'FINAL_CHECK.json')})
state = {**{k: v for k, v in list(state.items())[:6]}, UNIT: entry, **{k: v for k, v in list(state.items())[6:]}}
with (OPS / 'STATE.json').open('w', encoding='utf-8') as f:
    json.dump(state, f, ensure_ascii=False, indent=2)
    f.write('\n')

sec8 = '''### 2026-10-17 公開手겹침·좌우·순서 분석 완료 — 관측 누락 확인, 새 학습 전 검토
사용자 승인 범위는 촬영간격 보강 보류·공개 손겹침/좌우/순서 분석 후 보고. ops/word100_overlap_sequence_20261017 완료. 공개4표현(마음/수어/가평센터/강진센터)×TRAIN01~09=36영상, 주석핵심2072프레임의 native/sensor 원검출→배정→저장관측 검산. 사용자촬영읽기0·새분류0·새검출0·학습0·운영변경0.
sensor 미관측379/4144손프레임자리(원검출부재378/보류1), native129(부재123/보류6). 정상휴식/화면밖손을포함하므로오류율아님. 마음TRAIN06 왼손55~68프레임14개≈0.47초 영역내겹침중sensor원검출R하나만존재,배정보류아님. native동일14중10관측/4누락. 좌우임계완화·하단경계하향만으로회복할문제아님.
원좌우라벨불일치native38/sensor44검출은flag시각검토에서기존팔연결보정과일관;새확정좌우교환없음(전수정답오류0뜻아님). 두raw중보류native6/sensor1프레임은같은실제손중복으로판독,현재중복gate유지근거. 모든32flag구간89프레임보기·최장누락앞뒤122장·4순서48장및기존36영상360표본검토. 가평01/강진03최장왼손누락은얼굴근처R동작중L내림/화면밖휴식,무조건보완할누락아님.
마음은턱→가슴앞양손,센터표현은앞선다른동작뒤얼굴→가슴앞양손의부분공유. 전문가어휘분해아님. 현재TCN도순서를학습하며마지막전체평균의영향은가설. 저장C TRAIN masked/sensor각36/36Top1재검산,재채점0. 이미학습한자료이므로새사용자성능100%아님;관측누락만으로사용자오답인과확정불가.
별도코드360핀/보호10파일/326760좌표채널정확일치/집계·flag·저장순위PASS. Codex동일담당별도코드+시각검토,다른에이전트·전문가검토아님. 최초경로오류raw/assignment재사용폴더불일치는FIRST_ERROR/RECOVERY보존,동일36자료경로만복구·조건변경/제외0. 전작업종료.
근거 RESULT_KO.html/RESULT_KO.txt/RESULT.json/AUDIT.json/RECOUNT.json/VISUAL_REVIEW.json/FINAL_CHECK.json. 다음은관측품질(화면밖휴식/영역내가림/판단불가)분리와조건부순서보존후보검토. 과거poseROI고유회복0실패재탐색금지. 학습은사용자와결과확인후.
'''.replace('公開手겹침', '공개 손겹침')
sec9 = '''### 2026-10-17 공개 겹침·순서 진단 결과 검토 — 학습 전 단계
ops/word100_overlap_sequence_20261017의완료결과를사용자와확인. 촬영간격증강보류유지. 다음제안1은시간별·손별관측품질분리(화면밖/휴식,영역내가림·실패,배정불확실,판단불가)이며가상좌표복원/양손강제금지. 기존팔연결/중복gate를느슨하게하지않음. 새단어별예외금지.
그다음후보는현재TCN의전체평균외앞·중간·뒤특징보존. 현재TCN이순서를모른다는주장금지;모델원인가설은아직미검증. 구조변경전에공개TRAIN고정모델순서단서민감도조건고정(순서교란은정답률아님),사용자검토후한변경씩동일자료/예산3seed검증. TRAIN내부선택·반복DEV300회귀검사·보류14~18보존,사용자/멘토자료학습·튜닝·후보선택금지. 10월4일poseROI후보고유회복0/불량2/판단불가4미채택을재사용,동일crop계수재탐색안함. 자동학습/웹기본교체없음.
'''
context_path = ROOT / 'WordsInSilence_PROJECT_CONTEXT.md'
context = context_path.read_text('utf-8')
for heading, block in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인', sec8), ('## 9. 다음 단일 작업', sec9)]:
    assert context.count(heading) == 1
    context = context.replace(heading, heading + '\n\n' + block, 1)
context_path.write_text(context, encoding='utf-8')
handoff = f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

## 현재 목표·마지막 완료
사용자: 촬영간격 보강은 보류하고 공개 손겹침/누락/좌우/혼동표현 순서 분석, 공통보완 보고 후 함께 확인하고 학습.
완료 단위 research/front_keyphase_v1/ops/{UNIT}. RESULT_KO.html/RESULT_KO.txt/RESULT.json/AUDIT.json/RECOUNT.json/VISUAL_REVIEW.json/FINAL_CHECK.json 재사용. 새학습 승인아님.

## 확인한 결과
4표현×TRAIN01~09=36영상, 핵심2072프레임×native/sensor. sensor 미관측379/4144손프레임자리(원검출부재378/보류1),native129(123/6). 정상화면밖휴식포함,오류율아님.
마음TRAIN06 왼손55~68=14프레임약0.47초: 영역내겹침에서sensor원검출R하나뿐. 배정에서왼손을버린것아님. native는동일14중10관측/4누락. 단순경계하향/좌우gate완화로복구불가.
원라벨불일치native38/sensor44검출은검토한flag에서팔연결보정과일관,새확정해부학좌우오류없음(전수정답정확도주장금지). 중복보류native6/sensor1은동일실제손중복,유지타당. 가평01/강진03의긴L누락은R얼굴동작중L내려놓은화면밖휴식.
마음은턱→가슴앞양손,센터명은앞선동작후얼굴→가슴앞양손일부공유. 전체흐름다르며전문가어휘분해아님. TCN이미시간순서사용;마지막전체평균의영향은미확정가설.
기존저장C점수TRAIN36개masked/sensor각36/36Top1검산. 새추론아니며이미학습한자료,새사람정확도100%아님. 관측누락이사용자오답원인이라는인과확정못함.

## 검증·실행·보존
32flag구간89프레임보기전부,선택최장누락앞뒤122장,순서48장,기존36영상360표본시각확인. 전프레임전문가관절주석아님.
별도코드360자료핀/보호10파일/326760관절채널정확일치/집계/flag/저장순위PASS. Codex동일담당별도코드;다른에이전트검토아님.
최초audit실패raw와candidate저장폴더다름: FIRST_ERROR.txt보존,경로해석만수정RECOVERY,제외0·조건변경0.
audit완료,render exec86613종료0,recount exec64304종료0. 실행중작업없음. 새학습/분류/검출/사용자촬영읽기/운영변경0. 재추출·재채점불필요.

## 다음 한 작업
{next_action}
품질분류는모름상태유지,화면밖손억지복구/좌우교환/단어특례없음. 순서보존은효과가설,현재TCN도순서처리함. 학습시동일분할/예산3seed·TRAIN내부선택·반복DEV300회귀비교·보류14~18유지. 고정모델순서교란진단은정답률시험으로오해금지.
과거pose_roi_probe_20261004고유회복0/불량2/판단불가4로미채택;동일ROI크기/회전/임계값재탐색금지. 누락장구간직선보간자동적용금지.
원본/W:/sources/과거동결자료읽기전용. 카메라/다운로드/외부전송/Git/운영기본교체금지. 사용자·멘토자료학습/튜닝/후보선택금지.

## 기존 웹과 모델
C http://127.0.0.1:8765/practice.html?model=c / M0 ?model=m0 / 기본M0 유지.
마지막알려진서버PID38708/8765;이번서버상태점검·재시작·카메라접근없음.
C SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab;M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f 해시보존.
필요손DTW는ops/word100_role_dtw_20261017 완료상태유지. st-gcn자동화삭제유지.
'''
(ROOT / 'NEW_CHAT_HANDOFF.md').write_text(handoff, encoding='utf-8')

saved_state = read(OPS / 'STATE.json')
assert saved_state[UNIT]['new_training'] == 0
assert UNIT in (ROOT / 'NEW_CHAT_HANDOFF.md').read_text('utf-8')
old_state = read(before / 'STATE.json')
for key in old_state:
    if key not in ('updated_date_kst', 'current_stage', 'next_single_action', 'resume_command'):
        assert old_state[key] == saved_state[key], key
final = {'status': 'PASS', 'at': stamp, 'local_html_targets_checked': len(parser.links),
         'protected_pins': recount['protected_pins'], 'recount_sha256': sha(U / 'RECOUNT.json'),
         'result_sha256': sha(U / 'RESULT.json'), 'report_sha256': sha(U / 'RESULT_KO.html'),
         'text_sha256': sha(U / 'RESULT_KO.txt'), 'visual_review_sha256': sha(U / 'VISUAL_REVIEW.json'),
         'handoff_sha256': sha(ROOT / 'NEW_CHAT_HANDOFF.md'),
         'project_context_sha256': sha(context_path), 'state_sha256': sha(OPS / 'STATE.json'),
         'prior_state_entries_preserved': True, 'new_training': 0, 'new_inference': 0,
         'new_detection': 0, 'user_recording_reads': 0, 'production_changes': 0,
         'operator': 'Codex root; separate verification code, no agent/expert cross-review'}
with (U / 'FINAL_CHECK.json').open('x', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(json.dumps({'status': 'PASS', 'html_targets': len(parser.links), 'state': saved_state['current_stage'], 'new_training': 0}, ensure_ascii=False))
