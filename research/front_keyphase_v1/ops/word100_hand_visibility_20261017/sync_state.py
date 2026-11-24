"""Synchronize current handoff after report; preserve previous documents."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
U = Path(__file__).resolve().parent
R = U.parents[3]
assert (R/'AGENTS.md').is_file(), R
result = json.loads((U/'RESULT.json').read_text('utf-8'))
check = json.loads((U/'INDEPENDENT_CHECK.json').read_text('utf-8'))
assert check['status'] == 'PASS'
at = datetime.now().astimezone().isoformat(timespec='seconds')
before = U/'before'
before.mkdir(exist_ok=False)
context_path = R/'WordsInSilence_PROJECT_CONTEXT.md'
handoff_path = R/'NEW_CHAT_HANDOFF.md'
state_path = U.parent/'STATE.json'
for p in (context_path,handoff_path,state_path):
    (before/p.name).write_bytes(p.read_bytes())
next_action = '사용자와 손 미관측 공통 구분의 부분 검증 결과 확인. 다음 후보는 손목 한 점 대신 손 영역·이동 불확실성을 함께 쓰는 고정 검증 계획. 겹침 판정 범위 부족으로 자동 학습·DTW 제외·운영 연결하지 않음. 학습은 결과 확인 후.'
stage = 'WORD100_HAND_VISIBILITY_DIAGNOSTIC_PARTIAL_VALIDATION_REVIEW_BEFORE_TRAINING'
state = json.loads(state_path.read_text('utf-8'))
assert 'word100_hand_visibility_20261017' not in state
entry = {**result, 'report':str(U/'RESULT_KO.html'), 'text_report':str(U/'RESULT_KO.txt'),
         'independent_check':str(U/'INDEPENDENT_CHECK.json'), 'final_check':str(U/'FINAL_CHECK.json'),
         'running_processes': [], 'replay_status':'completed0; reuse saved evidence; do not rerun',
         'next_action':next_action}
ordered = {}
for k,v in state.items():
    ordered[k] = v
    if k=='current_stage': ordered['word100_hand_visibility_20261017'] = entry
ordered['updated_date_kst']=at
ordered['current_stage']=stage
ordered['next_single_action']=next_action
ordered['resume_command']='Read latest PROJECT_CONTEXT sections3/8/9, NEW_CHAT_HANDOFF and word100_hand_visibility_20261017. Reuse SCOPE, PREDICTIONS, VISUAL_LABELS, ASSERTION_REVIEW and INDEPENDENT_CHECK. Diagnostic partial validation, no production adoption. No training until user review; do not tune thresholds to current error examples.'
state_path.write_text(json.dumps(ordered,ensure_ascii=False,indent=2)+'\n','utf-8')

s8 = '''
### 2026-10-17 손 미검출 공통 구분 구현·공개 검증 완료 — 위치 구분 일부 유효, 겹침 판정 부족
최신 승인: 화면 밖 손과 동작 중 겹침 미검출 구분 보완·공개 검증 후 보고, 학습은 결과 확인 후. ops/word100_hand_visibility_20261017 완료. visibility.py는 같은 sensor 시야의 현재 팔 연결·과거 실제 손 관측만 사용하는 인과적 진단 부품. 관측됨/부분관측/경계·화면밖추정/화면안미관측/겹침의심/배정불확실/판단불가. 단어·사람ID·native숨긴정보·미래프레임 미사용, 좌표·마스크·배정 변경/복원0. 운영 미연결.
공개TRAIN16표현·9명·48영상(기존4표현36+결과전고정추가12표현12), 전체6352프레임/12704손상태. 추가12는새사람아님. 눈판독을출력보기전고정: 기존45미관측큰위치구분45일치; 추가26미관측21일치/1불일치/4보류, 정상관측16유지. 1표본=프레임의한손, 연속상관/선별표본이므로인식률·새사용자정확도주장금지. 추가보류는팔근거1/좌우배정보류3.
기존마음TRAIN06 L55~68의14프레임모두화면안미관측,겹침특정/좌표회복안됨. 겹침눈판독19중겹침의심1/일반화면안미관측18: 겹침원인분류범위부족. R016유성구센터L70은손목안쪽·손가락경계걸침을화면안으로오판. 손목한점/중심거리와실제손영역불일치,긴누락·빠른이동과이력만료한계. 결과후계수완화·단어특례없음. 경계추정은손일부잘림포함,휴식/비필수손확정아님.
판정후핵심확신35구간(경계32/겹침3)의시작중간끝94장면(경계89/겹침5)안전확인:이표본내온전한영역내동작손의명백한화면밖오판없음. 전프레임정확성보장아님. 합성23PASS·별도scalar코드12704상태일치·자료192핀/보호10파일해시PASS. Codex동일담당별도코드+시각검토이며타에이전트/전문가검토아님.
새학습/분류추론/MediaPipe재실행/사용자촬영읽기/운영TCN·DTW·웹변경0. 원본·과거동결결과보존. 실행모두종료0. RESULT_KO.html/RESULT_KO.txt/RESULT.json/SCOPE.json/PREDICTIONS.json/VISUAL_LABELS.json/ASSERTION_REVIEW.json/INDEPENDENT_CHECK.json/FINAL_CHECK.json 재사용. 현재는설명용진단의부분검증이며자동손제외/DTW무시/학습연결준비완료아님.

'''
s9 = '''
### 2026-10-17 손 미관측 구분 결과 함께 확인 — 자동 학습 전 단계
ops/word100_hand_visibility_20261017 완료결과를사용자와확인. 추가26장면21큰위치일치/1오판/4보류,겹침눈판독19중1만특정이므로현진단을자동학습손제외·DTW무시로연결하지않는다. 기존C/M0·웹·DTW유지. 촬영간격보강보류. 사용자/멘토자료학습·튜닝·후보선택금지.
다음제안은손목한점대신마지막실제관측의손영역과이동불확실성·팔연결을함께쓰는공통품질판단계획. 경계걸침/오래된이력은모름유지,겹침은중심거리만으로확정금지. 현재실패예시를맞추는계수완화·단어특례·새가상관절복원금지. 새조건먼저고정하고별도공개장면으로검증한후학습여부확인. 학습은현재요청에서승인되지않음.
완료검출/재생/검산반복불필요. SCOPE와PREDICTIONS/시각판독/별도검산/보고서재사용. 다음실행전최신사용자지시와이항목을확인한다.

'''
context = context_path.read_text('utf-8')
assert '### 2026-10-17 손 미검출 공통 구분 구현·공개 검증 완료' not in context
for heading, block in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',s8),('## 9. 다음 단일 작업',s9)]:
    assert context.count(heading)==1
    context = context.replace(heading+'\n',heading+'\n'+block,1)
context_path.write_text(context,'utf-8')

handoff = f'''# WordsInSilence 최신 인수인계

갱신 {at}

## 현재 목표·마지막 완료
사용자: 화면 밖 손과 동작 중 겹침 미검출 구분 공통 보완, 공개 검증 후 보고. 사용자 촬영 학습/튜닝 금지. 학습은 결과 확인 후.
ops/word100_hand_visibility_20261017의 진단 부품 구현·공개 검증 완료. 물리 원인 판정은 부분 검증이며 운영 미적용. RESULT_KO.html/RESULT_KO.txt/RESULT.json/FINAL_CHECK.json 재사용.

## 확인한 결과
공개TRAIN16표현·9명·48영상(기존36+다른표현12). 같은sensor시야의6352프레임,양손12704상태. 추가12는새사람아님.
현재팔연결+직전실제손관측→관측/경계·밖추정/화면안미관측/겹침의심/보류. 단어·사람ID·native숨긴정보·미래정보미사용. 좌표·마스크·배정변경/복원0.
출력전시각판독고정: 기존45미관측위치구분45일치. 추가26미관측21일치/1오판/4보류, 정상관측16유지. 한표본은한프레임의한손;연속상관·선별표본으로인식률/새사용자정확도아님.
마음TRAIN06 L55~68=14모두화면안미관측. 겹침특정/복구아님. 겹침눈판독19중겹침의심1/일반화면안미관측18:범위부족.
R016유성구센터L70은손목안쪽·손가락경계걸침을화면안으로오판. 손목점/중심거리와손전체영역차이,오래된이력·빠른움직임의한계. 계수완화·단어예외없음.
판정후35확신구간의시작중간끝94장면(경계89/겹침5)을안전검토. 표본내온전히영역내있는동작손의명백한화면밖오판없음,전프레임보장아님. 경계상태는손일부잘림포함,완전화면밖/휴식/비필수손확정아님.

## 구현·검증·보존
visibility.py SHA790fa3162e5f56846b3feace64aa75d8e384b1967c5836cc10efb661189ac88c. SCOPE.json에결과전조건/표본/핀고정.
합성23PASS. verify_saved.py의별도scalar계산12704상태일치·자료192핀/보호10파일해시PASS. 같은Codex담당별도코드,다른에이전트나전문가검토아님. 생물학적/물리적정답성은시각선별표본한계.
SYNTHETIC_TESTS.json/VISUAL_LABELS.json/PREDICTIONS.json/ASSERTION_REVIEW.json/INDEPENDENT_CHECK.json 완료근거재사용. 모든실행종료0·진행중프로세스없음. 새학습/TCN추론/MediaPipe실행/사용자촬영읽기/운영변경0.
public_videos12개는기존W:ZIP에서CRC·크기·해시검증로컬해제,다운로드아님. 원본/W:/sources/과거동결자료읽기전용,Git변경없음.

## 다음 한 작업
{next_action}
현재상태만으로학습에서손제외/DTW무시금지. 장기누락직선보간·가상손복원·단어특례·임계값재탐색금지. 새조건고정후별도공개검증부터. 촬영간격보강보류,일반화성능미확인.

## 기존 웹과 모델
C http://127.0.0.1:8765/practice.html?model=c / M0 ?model=m0 / 기본M0 유지. 이번서버점검·재시작·카메라접근없음.
C SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab; M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f 유지.
필요손DTW는ops/word100_role_dtw_20261017 완료상태유지. st-gcn자동화삭제유지.
'''
handoff_path.write_text(handoff,'utf-8')

# Lightweight close-out: reports/links/JSON/state, no repeated detector or training.
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(U/'visibility.py') == check['evidence_sha256']['visibility.py']
for path,digest in check['protected_pins'].items(): assert sha(Path(path)) == digest
for name in ['RESULT.json','INDEPENDENT_CHECK.json','ASSERTION_REVIEW.json','SCOPE.json','PREDICTIONS.json']:
    json.loads((U/name).read_text('utf-8'))
report = (U/'RESULT_KO.html').read_text('utf-8')
links = re.findall(r'(?:href|src)="([^"]+)"',report)
assert links and all((U/p).is_file() for p in links)
after = json.loads(state_path.read_text('utf-8'))
assert after['current_stage']==stage and after['next_single_action']==next_action
assert after['word100_hand_visibility_20261017']['status']==result['status']
assert all('word100_hand_visibility_20261017' in p.read_text('utf-8') for p in (context_path,handoff_path))
final = {'status':'PASS','at':at,'implementation_status':result['status'],
         'report_links_checked':len(links),'json_parse':'PASS','root_documents_synchronized':True,
         'candidate_code_preserved':True,'protected_model_runtime_pins':10,
         'running_processes':[], 'new_training':0,'user_recording_reads':0,
         'next_single_action':next_action,
         'document_sha256':{str(p):sha(p) for p in (context_path,handoff_path,state_path,U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'RESULT.json')}}
with (U/'FINAL_CHECK.json').open('x',encoding='utf-8') as f:json.dump(final,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'PASS','updated_at':at,'report_links':len(links),'stage':stage},ensure_ascii=False))
