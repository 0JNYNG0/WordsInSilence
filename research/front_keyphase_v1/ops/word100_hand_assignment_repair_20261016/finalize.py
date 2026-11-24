"""Record verified input-repair results; training requires subsequent user review."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
verify=read(U/'INDEPENDENT_VERIFICATION.json');audit=read(U/'CHANGE_AUDIT.json')
assert verify['independent_behavior_tests']==31 and verify['failures']==verify['errors']==0
assert audit['previous_nine_all_preserved'] and audit['named_recovery_fixtures_all_pass']
for name in ['CODE_REVIEW.md','EMPIRICAL_REVIEW.md','RESULT_KO.txt']:
    assert (U/name).is_file(),name
for p,h in read(U/'PROTECTED_FILES.json').items():assert sha(Path(p))==h,p
for p,h in read(U/'REPLAY_CONTEXT.json')['pins'].items():assert sha(Path(p))==h,p
summary=('정상 손 거절·중복 선택 수정 후보 구현/검증 완료. 같은 공개학습12영상24조건·2,976프레임·3,910검출 재생. '
 '공항 원본63–65/crop78 정상R 복원, 마음에들다47·멀다88 pose 근거 raw 선택 복원. 이전 좌우 교정9/9·믿다49 추가L 유지. '
 '기존 운영 배정 대비 제외31→8, 좌우 변경10/새배정2. 남은8은 모두 오류 제거가 아님: 겹친 실제 받침손과 경계 진입 손의 보류 포함. '
 '독립검사31/31,3190파일 보존, 검출순서 역전24조건 물리배정변경0. 특정 회귀 수정 통과이며 전체 관측/인식 성능 통과 아님. '
 '학습·분류추론·새추출·운영/웹/DTW 교체0, 사용자 검토 대기.')
next_action=('이번 수정 결과와 남은 겹침·경계 손의 보류 사례를 사용자와 검토한다. '
 '사용자 확인 전 전체 재추출·학습으로 넘어가지 않는다. 다음 구현을 승인하면 남은 가시손 보류 정책을 별도 조건으로 확정하거나, '
 '허용한 한계와 평가조건을 고정한 상반신 자료 준비 단계로 진행한다. 기존 계수를 결과에 맞춰 낮추지 않는다.')
decision={'status':'TARGETED_REPAIR_VERIFIED_RESIDUAL_ABSTENTIONS_AWAITING_USER_REVIEW',
 'updated_kst':now,'summary':summary,'next_single_action':next_action,
 'known_target_repairs_pass':True,'all_visible_hands_preserved':False,
 'training_ready':False,'production_adopted':False,'user_review':'PENDING',
 'training':0,'classifier_inference':0,'new_detector_extraction':0,'running_processes':[],
 'implementation':'Codex root',
 'independent_reviewers':['assignment_verify: tests and conservation','raw_capture: saved-array numeric audit','assignment_review: code and visual review'],
 'limitations':['Known TRAIN engineering fixtures, same REAL01 performer; not unseen-user recognition evaluation.',
  'Handedness confidence and nearest pose distance do not prove all finger coordinates are accurate.',
  'Overlapping hands without distinct reliable identity evidence may be withheld.',
  'Partial-edge detection recovery is not full 21-point hand recovery.'],
 'execution_notes':['Process command-line CIM query unavailable; Get-Process found no Python process before work.',
  'A read-only inspection failed under default cp949; retried with explicit UTF-8, no output mutation.',
  'Independent reviewer synthetic first fixture wrongly assumed 4px close-palms; corrected fixture to2px with first failure preserved. Candidate unchanged.']}
new(U/'DECISION.json',decision)
entry={**decision,'unit':'ops/'+U.name,'report':'ops/'+U.name+'/RESULT_KO.txt',
 'conditions':24,'frames':2976,'raw_hand_detections':3910,'counts_vs_production':audit['counts_vs_baseline'],
 'previous_side_corrections_preserved':9,'behavior_tests_passed':31,'protected_file_hashes':3190,
 'completion':'ops/'+U.name+'/COMPLETION.json'}
sp=ROOT/'research/front_keyphase_v1/ops/STATE.json';state=read(sp)
state['updated_date_kst']=now;state['current_stage']='WORD100_HAND_ASSIGNMENT_REPAIR_VERIFIED_RESIDUALS_USER_REVIEW'
state['next_single_action']=next_action
# Keep previous operation records unchanged; the current key is placed near top.
state={k:v for k,v in state.items() if k!=U.name}
out={}
for k,v in state.items():
    out[k]=v
    if k=='current_stage':out[U.name]=entry
sp.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=cp.read_text(encoding='utf-8')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
text=text.replace(h8,h8+'\n### 2026-10-16 좌우 배정 후속 수정 완료 — 특정 회귀 복구, 잔여 보류 검토 대기\n'+summary+'\n근거 `ops/'+U.name+'/RESULT_KO.txt`, `DECISION.json`, `COMPLETION.json`. 이전 후보/원시 캐시 불변. 구현 Codex root, 독립 검증3에이전트.\n',1)
text=text.replace(h9,h9+'\n### 2026-10-16 다음 — 수정 결과와 남은 관측 보류 상호 검토\n'+next_action+'\n이번 사용자 요청은 두 결함 수정·보고까지 승인이며 학습 자동 승인이 아니다.\n',1)
cp.write_text(text,encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}. 근거리 웹캠의 상반신·화면 밖 손 조건에 맞춘 입력 보완 중.

## 사용자 진행 방식
기능별 작업 → 결과·문제 보고 → 함께 검토 → 다음 단계. 최신 요청은 정상 손 제외와 중복 검출 선택 보완 후 보고, 확인한 다음 학습.

## 마지막 완료
{summary}
새 단위 `ops/{U.name}`. `SCOPE.json`에 구조 수정·수치 고정 후 이전 unit의 raw cache만 재사용했다. 이전 결과와 원본·TCN 보호. 구현 root, 검증 assignment_verify/raw_capture/assignment_review.
높은 handedness의 pose 거절을 제거하고, 같은 side 후보 경쟁을 좌우 모호성과 분리해 현재 pose 손목에 가까운 후보를 남긴다. 별개 temporal/pose 양손 근거는 보존. 부분 경계는 기존5% 관측 계약과 일치, 없는 좌표 보간/생성0.
남은 제외8건 중 실제 받침손/경계손 보류가 있으므로 '정상 손 누락 모두 해결' 또는 '학습 준비 완료'로 말하지 않는다. 신규 수어crop32는 일부 손점 관측이며 완전한21점복구가 아니다. 31→8은 배정 제외 건수이지 인식률이 아니다.
근거: `RESULT_KO.txt`, `EMPIRICAL_REVIEW.md`, `CODE_REVIEW.md`, `INDEPENDENT_VERIFICATION.json`, `CHANGE_AUDIT.json`, `DECISION.json`, `COMPLETION.json`.

## 다음 한 작업
{next_action}
사용자자료 학습·튜닝·후보선택·임계값 조정 금지. 동결 후보 결과를 덮어쓰지 않는다.

## 작업 목록
`Docs/WEBCAM_IMPROVEMENT_TODO.md`. 1단계 완료, 2단계 후속 수정·검증 완료/잔여 보류 검토 대기. 3단계 자료준비·4단계 학습 미실행. DTW 필수손과 웹 재시험도 별도 기능 단위.

## 보존 기준
TCN `runs/word100_tcn_v1/whole_context/best.pt`, SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 학습9명899/개발3명300,96.67%는 개발검증. 보류10/14–18 미사용. 사용자12회 Top1 6/12,Top5 8/12는 개발참여자1인 실사용.
운영 웹/DTW/모델, 원본ZIP/관절, 이전 동결 산출물 보존. 프로세스 종료. 새학습/분류추론/카메라/장치/다운로드/외부전송/Git 변경0. 발표PDF 완료상태 유지.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
tp=ROOT/'Docs/WEBCAM_IMPROVEMENT_TODO.md';todo=tp.read_text(encoding='utf-8')
todo=todo.replace('| 구현·비교 완료 / 회귀로 채택 보류 |','| 후속 수정·검증 완료 / 잔여 보류 상호 검토 대기 |',1)
todo+='\n\n## 2026-10-16 2단계 후속 수정 결과\n\n'+summary+'\n\n다음: '+next_action+'\n\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`.\n'
tp.write_text(todo,encoding='utf-8')
artifacts=[p for p in U.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'before' not in p.parts]
docfiles=[cp,sp,tp,ROOT/'NEW_CHAT_HANDOFF.md']
new(U/'COMPLETION.json',{**decision,'artifacts_sha256':{str(p):sha(p) for p in artifacts},'documents_sha256':{str(p):sha(p) for p in docfiles}})
for p in [sp,U/'DECISION.json',U/'COMPLETION.json']:read(p)
for p,h in read(U/'PROTECTED_FILES.json').items():assert sha(Path(p))==h,p
print('Completion and four state documents recorded:',now)
