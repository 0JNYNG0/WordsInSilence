"""Close diagnostic-only implementation; no training authorization inferred."""
import json,hashlib,html
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
c=read(U/'CONTEXT.json');s=read(U/'SUMMARY.json');v=read(U/'INDEPENDENT.json');seq=read(U/'ASSIGNMENT_SEQUENCE.json');assert v['status']=='PASS'
at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
review={'status':'REVIEWED_WITH_REMAINING_ISSUE','reviewer':'Codex local, same operator; no separate human/agent ground-truth annotation',
 'reused_source_reviews':'Previous public RGB review12panels/10unique scenes/7clips; matched new EXAMPLES sidecar values. No new detector.',
 'new_boundary_panels':6,'new_sequence_panels':6,
 'findings':[
 'Existing observed-hand counts and old side allocation stay unchanged, including possible pre-existing errors. Numerical preservation does not certify anatomical correctness.',
 'Primary palm5 observed plus inferred boundary/near cases: masked172, sensor287. Of sensor287,283 are near-boundary and4 have the Pose wrist actually beyond frame. These overlapping fields are not automatically contradictions or hand observation errors.',
 'All four sensor outside-plus-palm cases were visually reviewed. Three have only7/9/10 remaining usable points near the image boundary, with Pose/Hand21 wrist estimates offset; no automatic exclusion or point correction is appropriate.',
 'Fourth case NIA_SL_WORD2885_REAL06_F frame45 has21 points in the left slot while its Pose left wrist is down/beyond border. RGB frames43-48 show the right hand rising and left arm remaining down. At45 two detector hypotheses lie near the right hand; one was placed in left slot by label_without_close_pose, the other in right by temporal_and_pose. Previous44 and next46 allocate the single remaining candidate to right. This is an existing duplicate/misassignment case revealed by separated evidence, not caused or fixed by this change.',
 'A scalar same-frame arm estimate can remain unknown for a projected forearm below the frozen geometry minimum, even when the human sees a hand. No threshold changed to accommodate that example.'
 ],'production_side_assignment_changed':False,'new_hand_identity_accuracy_measured':False}
save(U/'VISUAL_REVIEW.json',review)
extra_pins={seq['source_trace_path']:seq['source_trace_sha256']}
for name in ['assignment.py','SCOPE.json','POINT_CONTRACT.json']:
 p=O/'word100_hand_assignment_joint_20261016'/name;extra_pins[str(p)]=sha(p)
save(U/'ADDITIONAL_READONLY_EVIDENCE.json',extra_pins)
plan={'status':'PROPOSED_AWAITING_USER_REVIEW_NO_TRAINING',
 'next_unit':'Public-only audit of duplicate candidate and label-only fallback assignment conflicts before considering new learning.',
 'reason':'Separation preserves information but does not recover Hand21 or repair existing identity errors; one concrete old duplicate/misassignment was found.',
 'requirements':['Select public TRAIN cases by current observation/assignment conditions, not classification errors, user recordings or word names.',
 'Inspect label_without_close_pose assignments and competing candidates near the same currently supported arm; quantify coverage before choosing a common rule.',
 'Keep already supported correct assignments, preserve genuine crossed hands, do not force two hands or split identity by screen x half.',
 'If proposing a rule change, freeze it before evaluation, measure withheld/new errors and preserved cases, and never correct source records silently.',
 'Keep current separated diagnostics isolated until reviewed. Existing C/TCN/DTW/web stays unchanged. Training remains pending user result review and a fixed comparison plan.'],
 'user_mentor_data_used':False,'training':0}
save(U/'NEXT_PLAN.json',plan)
report='''손 관측·팔 위치·좌우 배정 불확실성 분리 보완 결과

결론
요청한 세 정보의 분리 구현과 공개 자료 검증을 완료했다. 기존 C의 손 관절·관측 마스크·시간·배정·TCN·DTW·웹은 변경하지 않았다. 새 구조는 별도 진단 결과이며 학습 입력이나 운영 화면에 연결하지 않았다. 사용자·멘토 촬영, DEV 및 보류 촬영자14~18 자료는 사용하지 않았다. 새 학습0회, 모델 추론0회다.

1. 왼손·오른손은 처음부터 알 수 있지 않나?
MediaPipe는 Handedness라는 좌우 추정값을 제공한다. 하지만 정답이 보장된 신분표는 아니다. 오른손을 왼쪽 어깨로 가져가도 오른손이며, 화면 중앙을 넘었다고 왼손이 되지 않는다. 정면 카메라/미러 표시에서는 화면 좌우와 사람의 좌우도 다를 수 있다.
현재 공개 입력을 만든 배정 코드도 MediaPipe 좌우 라벨만 쓰지 않고 Pose 손목과의 거리, 이전 관측 연결 등을 참고한다. 다만 근거가 약할 때 라벨만 사용하는 경로가 남아 있다. 겹침·미검출·중복 후보 뒤에도 최초 좌우를 영구적으로 확신할 수는 없다.
'배정 보류'는 잘못 배정됐다는 확정이 아니라 어느 손인지 확신하지 못해 후보 사용을 보류했다는 뜻이다. 이번 검토에서는 이와 별개로 실제 중복/오배정 사례도 한 장면 확인했다.
공식 문서: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker

2. 이번에 바뀐 점
기존: 한 위치 상태 칸에서 관측됨/일부관측/안쪽추정/경계·밖추정/좌우보류/모름 중 하나를 고름.
보완: 다음 세 항목을 독립적으로 기록함.
① 실제 손 관측: 사용 가능한 관절 수0~21개, 손바닥5점 여부,21점 전부 여부.
② 팔 기반 위치 추정: 어깨·팔꿈치·손목 근거로 안쪽/경계·밖/근거부족을 기록. 화면 밖 손목은 추정으로 표시하며 실제 Hand21로 승격하지 않음.
③ 좌우 배정 정보: 해당 손에 배정된 근거와 미배정 후보 존재를 별도로 기록. 미배정 후보 표시는 프레임 전체 정보이며, 특정 손이 틀렸다는 정답으로 취급하지 않음.
masked 보기에서 허용되지 않는 원본 후보 정보는 계속 차단하고 '정보 없음'으로 남긴다. sensor에서 미배정 후보가 없다는 것과 masked에서 확인할 수 없다는 것을 구분한다.
기존 위치 진단도 감사용으로 그대로 보존했다. 정상 관절을 지우거나 손의 좌우를 새로 바꾸지 않았다. 기존 공통 계수와 관측 구간도 그대로다.

3. 공개 검증 결과
AI Hub TRAIN899영상/100단어/촬영자1~9. 저장된 전체 프레임을 두 보기로 처리한452,724손-프레임에 대해 기존 관측 개수와 위치 진단을 보존했다. 이미 손바닥 관측으로 분류되던162,952건과 기존 팔 위치 추정112,699건도 보존했다. 이것은 기존 관측의 보존 수이며 사람의 정답 손 위치로 검증된 건수나 인식 정답률이 아니다.
서로 동일 시점인116,640손-프레임을 비교한 sensor 구간에서 기존 좌우 배정 보류1,814건은 다음처럼 정보를 함께 표시할 수 있게 됐다.
  - 손목 안쪽 추정533건 + 좌우 배정 불확실성 유지
  - 손목 경계·밖 추정1,207건 + 좌우 배정 불확실성 유지
  - 팔 위치도 여전히 근거 부족74건
따라서1,740건에서 가려졌던 기존 팔 근거가 별도 항목으로 드러났다.1,740개 손을 새로 검출하거나 좌우 정답을 고친 것이 아니다.
손바닥5점은 보이지만21점 전체가 보이지 않는 masked3,779건/sensor4,247건을 별도로 구분했다. '손바닥이 보임'을 '모든 손가락을 확실히 봄'으로 잘못 설명하지 않게 했다.

4. 검산
별도 벡터 계산으로 새 팔 위치452,724건과 손 관측452,724건을 재계산하여 일치했다. 손 교차·좌우 반전·관측 보존·배정 보류·화면 밖 추정·숨긴 정보 차단 등12개 계약 검사를 통과했다. 실제 자료899영상의2,697프레임에서 숨긴 좌표/후보/배정 정보를 바꿔도 masked 출력이 같았다. 미래 프레임 변경에 앞선3,548프레임도 보존됐다. 원본·기존 입력/코드10,799개 해시가 유지됐다.
검산은 같은 담당자가 별도 계산 방법으로 수행했다. 다른 사람이나 에이전트가 실제 좌우 정답을 전수 주석한 검증은 아니다. 원본 RGB 검토와 수치 검산을 구분한다.

5. 남은 문제와 추가로 발견한 사례
손가락 미검출은 그대로 남는다. 팔을 통해 손목 위치를 추정할 수 있어도 손가락 모양을 복원할 수는 없다.2D 팔 길이가 짧게 투영되거나 Pose 신뢰도가 낮으면 위치도 근거 부족으로 남는다.
손바닥 관측과 팔의 경계 추정이 함께 나온 구간은 masked172/sensor287손-프레임이었다. sensor 중283건은 손목이 경계 근처인 경우이므로 손 일부가 보이는 것과 모순이 아니다. 실제 Pose 손목이 경계 밖인 나머지4건을 모두 원본 RGB로 확인했다. 그 중3건은 손 일부7/9/10점이 남아 있고 Pose와 Hand21 손목 추정이 어긋난 경우였다. 하나의 근거로 다른 근거를 자동 삭제하지 않았다.
나머지1건은 '이만오천' 공개 촬영자06의45프레임이다. 오른손을 올리는 중 오른손 부근에서 검출 후보가2개 생겼고, 하나는 이전 관측·Pose로 오른손에, 다른 하나는 가까운 Pose 근거 없이 MediaPipe 좌우 라벨만으로 왼손 칸에 들어갔다(label_without_close_pose). 실제 영상에서 왼팔은 아래에 있다. 앞뒤43·44·46·47·48프레임에서는 단일 후보가 오른손으로 배정됐다. 당시 왼손 라벨 점수가 높아도 실제 좌우를 보장하지 못했다.
이 중복/오배정은 기존 저장 관측에 이미 있던 문제다. 새 분리 구조가 만들거나 고친 것이 아니다. 보완 과정에서 두 근거를 나란히 검토하여 드러났다. 한 사례를 전체 좌우 오류율로 일반화하지 않는다.

6. 다음 권장 작업
새 학습 전에 공개 자료에서 중복 후보와 '라벨만으로 배정'하는 경로의 충돌을 조사하는 것이 우선이다. 특정 단어가 아니라 동일 조건 전체에서 검토하고, 올바른 손 교차·두 손 겹침·기존 정상 배정을 보존하는 공통 보완안을 고정해서 비교해야 한다. 화면을 반으로 나누거나 무조건 두 손으로 채우는 방식은 쓰지 않는다.
이번 분리 기능은 진단 정보 보존 검증을 통과했다. 손 인식률/TCN 정확도 향상은 아직 평가하지 않았고 보장할 수 없다. 결과를 사용자와 확인한 뒤 다음 보완 및 학습 비교 계획을 결정한다.

실제 변경: 현재 작업 폴더 axes.py/run.py/verify.py 및 진단 결과·보고서, 기준문서 최신 상태. 기존 C/웹/DTW/학습 파일 수정 없음.
담당: Codex 로컬 작성·실행·원본 시각 검토·별도 계산 검산. 학습/사용자자료/카메라/다운로드/외부 전송/Git 변경 없음.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 근거 분리 보완 결과</title><style>
body{margin:0;background:#edf2f7;color:#172b40;font:17px/1.7 'Malgun Gothic',sans-serif}main{max-width:1080px;margin:40px auto;padding:0 24px}h1{font-size:34px}h2{font-size:25px}.box{background:white;border-radius:14px;padding:28px;margin:24px 0}.axes{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.axes>div{background:#eaf4fb;border-radius:9px;padding:18px}strong{color:#0b6688}img{width:100%;border-radius:10px}pre{font:16px/1.8 'Malgun Gothic',sans-serif;white-space:pre-wrap}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #dde5ee}@media(max-width:700px){.axes{grid-template-columns:1fr}}</style><main>
<p>WordsInSilence · 공개899영상 · 기존 C 유지 · 새 학습 없음</p><h1>손이 보이는지, 어디에 있을지,<br>어느 쪽 손인지 따로 기록</h1>
<div class="box"><h2>이전에는 ‘배정 보류’ 하나만 보이던 장면</h2><div class="axes"><div>실제 손 관측<br><strong>사용 가능 관절0개</strong><p>손가락 모양을 모름</p></div><div>팔 기반 위치<br><strong>손목은 화면 안쪽 추정</strong><p>기존 팔 근거를 보존</p></div><div>좌우 배정<br><strong>미배정 후보 있음</strong><p>불확실성을 그대로 유지</p></div></div><p>손가락을 복원하거나 좌우를 강제로 바꾼 것이 아니다. 기존 정보가 서로 가리지 않게 했다.</p></div>
<div class="box"><h2>보존과 개선</h2><table><tr><th>확인 항목</th><th>결과</th></tr><tr><td>기존 손 관측·위치 진단</td><td>452,724손-프레임 보존</td></tr><tr><td>배정 보류 중 가려졌던 팔 위치 단서</td><td>1,740건 함께 기록 가능</td></tr><tr><td>분리 후에도 팔 위치가 불확실한 경우</td><td>74건 그대로 유지</td></tr><tr><td>새 학습·운영 모델 변경</td><td>없음</td></tr></table><p>건수는 프레임별 손 기록이다. 인식 정답률이나 새로 복원한 손의 수가 아니다.</p></div>
<div class="box"><h2>남은 중요 문제: 같은 손에 후보가 두 개</h2><p>공개 ‘이만오천’ 영상45프레임에서 오른손 부근의 두 후보 중 하나가 라벨만으로 왼손 칸에 배정됐다. 기존 관측에 있던 문제이며 이번에 자동 수정하지 않았다.</p><img src="evidence/assignment_sequence.jpg" alt="43~48프레임 손 후보 및 좌우 배정 비교"><p>다음에는 공개 자료의 중복 후보·라벨 단독 배정 경로를 공통 조건으로 검토하는 것이 우선이다.</p></div>
<div class="box"><h2>전체 결과와 검증 범위</h2><pre>'''+html.escape(report)+'</pre></div></main></html>'
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
body=f'ops/{U.name}. 사용자 승인으로 실제Hand21관측/팔위치추정/좌우배정불확실성 분리 sidecar 구현·공개TRAIN899 검증 완료. 기존관측/legacy452,724손프레임 보존, 기존palm관측162,952/팔위치112,699보존(정답인증아님). 공통sensor배정보류1,814 중 안쪽533/경계밖1,207 근거를별도보존,74모름유지. 독립계산452,724건/계약12/숨긴정보2,697/미래변경prefix3,548 PASS. 새학습·추론0, 기존C/DTW/웹유지, 사용자/멘토/DEV/보류읽기0. 팔/관측경계병존 sensor287 중 Pose밖4개시각검토:3개부분관측+1개기존중복오배정발견(WORD2885_REAL06_F f45 label_without_close_pose). 자동재배정하지않음. RESULT_KO.txt/html 및 VISUAL_REVIEW/INDEPENDENT 참조.'
nextbody=f'{U.name}/RESULT_KO.txt와NEXT_PLAN.json 결과를사용자와확인. 세근거분리구현은완료했으나학습미승인/운영미연결. 다음제안은공개TRAIN의중복후보와라벨단독배정충돌경로감사,정상교차/관측보존공통보완안검토. 발견한기존오배정을특정단어예외로고치지않고C그대로유지. 사용자/멘토자료학습·튜닝·선택금지. 학습은추후결과검토와조건동결후.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text(encoding='utf-8-sig')
for heading,title,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','2026-10-18 손 관측·팔 위치·좌우 불확실성 분리 구현·공개 검증 완료',body),('## 9. 다음 단일 작업','2026-10-18 분리 결과와 기존 중복 후보 오배정 검토',nextbody)]:
 assert txt.count(heading)==1;txt=txt.replace(heading,heading+'\n\n### '+title+'\n'+b+'\n',1)
p.write_text(txt,encoding='utf-8')
state=read(O/'STATE.json');state['current_stage']='WORD100_EVIDENCE_AXES_VERIFIED_WITH_REMAINING_ASSIGNMENT_ISSUE';state['updated_date_kst']=at;state['next_single_action']=nextbody
state[U.name]={'at':at,'status':'COMPLETE_AWAITING_REVIEW','train_videos':899,'observations_preserved':452724,'previously_blocked_arm_evidence_revealed':1740,'unresolved_arm':74,'independent':'PASS','new_training':0,'user_mentor_heldout_dev_used':False,'production_changed':False,'remaining_assignment_issue':{'clip_id':'NIA_SL_WORD2885_REAL06_F','frame':45,'reason':'label_without_close_pose duplicate candidate'},'result':str(U/'RESULT_KO.txt'),'next_plan':str(U/'NEXT_PLAN.json')}
(O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+at+'\n\n### 현재 목표·마지막 완료\n'+body+'\n\n### 다음 단일 작업\n'+nextbody+'\n\n현재 실행 중 명령·잠금 없음. st-gcn 자동화 삭제 상태 유지. 원본/W:/sources/과거동결결과 읽기전용. 코드 담당Codex로컬, 동일담당자별도계산검산이며타에이전트/독립인물주석검증아님.\n',encoding='utf-8')
save(U/'STATUS.json',{'stage':'COMPLETE_VERIFIED_WITH_REMAINING_ASSIGNMENT_ISSUE','at':at,'training':0,'production_changed':False})
for p,h in {**c['pins'],**extra_pins}.items():assert sha(p)==h,p
pins={str(p):sha(p) for p in U.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'}
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:pins[str(p)]=sha(p)
save(U/'FINAL_AUDIT.json',{'status':'PASS_WITH_DOCUMENTED_REMAINING_ISSUE','at':at,'output_and_document_pins':pins,'verification':str(U/'INDEPENDENT.json'),'visual_review':str(U/'VISUAL_REVIEW.json'),'training':0,'production_changed':False})
print('Verified diagnostic-only completion recorded; existing assignment issue reported, no training.')
