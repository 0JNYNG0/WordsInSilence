from common import *
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

assert read(U/'VERIFIED.json')['status']=='PASS'
at=datetime.now().astimezone().isoformat()
historic=read(U/'HISTORIC_SOURCE_AUDITED.json')['metrics'];diag=read(U/'v2_diagnostic_RESULTS.json')['metrics'];confirm=read(U/'v2_confirmation_RESULTS.json')['metrics']
decision=dict(at=at,status='COMPLETE_RESEARCH_NOT_ADMITTED',web_changed=False,web_version='G6-finger-20261115',training_runs=9,protected_files=518,private_data_used=False,
 implementation='Codex',verification='Codex separate saved-data/source/hash checks; not independent expert',
 result='Source-audited historical966: existing910correct/1wrong/55unknown, candidate912/2/52. Fresh diagnostic144:122/6/16 to125/6/13. Frozen fill candidate confirmation143 unchanged97/3/43. Predeclared override ablation99/3/41; not promoted. Brief true index extension WORD2551p12f35 still wrong.',
 source_correction='Original9673newdisagreements:2 due wrong both-hands-open annotation. Correct middle/ring and exclude unclear little symmetrically; preserve original. Remaining1newerror; do not select based on posthoc relabeling.',
 gates=dict(evidence_integrity=True,public122_contract_preserved=True,no_new_confident_errors=False,brief_extension_recovery=False,physical_webcam_acceptance=False),
 admitted_component='None from this round; previous limited supplement remains',
 server_observed='8821 not reachable at close preparation; no server or camera started in this round',
 complete_steps=['A public source/side annotation review','B separate state/change training and predeclared whole-hand corroboration','C historical/diagnostic/reserved comparisons','D public122 regression, source audit and separate verification'],
 reason='Do not trade new corrections for confident mistakes; change detector is not reliable proof of center state. Complete bounded experiment without declaring goal solved.')
write(U/'DECISION.json',decision)

report='''손가락 현재 모양·시간 변화 분리 — 이번 연구 완료, 새 후보는 웹 미적용

한눈에 보는 결론
공개 원본 검토 → 분리 학습 → 기존/새 자료 비교 → 공개 재생 검증까지 끝냈다.
일부 손가락은 더 맞혔지만, 짧게 실제로 편 검지를 놓치는 문제는 해결하지 못했다.
새 오판도 남아 새 후보를 웹에 넣지 않았다. 현재 웹의 제한된 손가락 보조는 유지한다.

무엇을 가르쳤나
1. 현재 프레임: 지금 손가락이 펴졌는지, 굽혀졌는지.
2. 앞뒤 흐름: 모양을 유지하고 있는지, 바꾸고 있는지.
두 출력을 분리했다. 시간 변화가 있다는 이유만으로 현재 모양을 맞혔다고 믿지 않도록
손 전체와 개별 손가락 관측이 일치하는 보완도 비교했다. 단어별 예외는 없다.
기존 공개 학습에 새 정답을 더해 22단어·109영상·9명의626손시점으로 현재 모양,
153손시점으로 변화를 학습했다. 손시점은 한 영상에서 고른 특정 프레임의 한 손이다.
현재모양 두 구조는 기존 학습 가중치에서 미세조정, 변화 구조는 별도 지도학습했다.
각3개 고정 초기값으로 총9회 소규모 학습. 공통100단어/C·T3-F는 재학습·교체하지 않았다.

검증에서 알아낸 것
새로 확보한12개 정면 공개영상과 같은 촬영의 측면 영상을 검토했다. 학습6/개발3/확인3.
측면은 정답 검토에만 쓰고 모델 입력은 정면 관절값이다. 수행자는 과거TCN에 노출된
사람이므로 프로젝트 전체에서 처음 본 사람이라고 주장하지 않는다.
기존 정답표의 한 장면에 양손 모두 폄이 잘못 붙어 있었다. 왼손 중지·약지는 굽힘으로
정정하고 새끼는 가림으로 제외했다. 원본표와 최초 결과는 보존하고 양쪽 모델을 똑같이
재집계했다. 정답표 정정이 모델 개선은 아니다. 전문가 검수가 아닌 Codex 원본 검토다.

비교 단위: 특정 프레임의 손가락1개. 영상 수/수어 단어 정답률이 아니다.
자료                       기존:맞음/틀림/보류       후보:맞음/틀림/보류
기존 자료·원본 재검토 후966  910 / 1 / 55             912 / 2 / 52
새 개발144                 122 / 6 / 16             125 / 6 / 13
예약 확인143               97 / 3 / 43              97 / 3 / 43
후보는 동결한 dual_change_fill. 별도 미리 정한 override 비교는 확인99/3/41이나
기존 추가 오판은 같아서 채택하지 않았다. 유리한 방식/장면만 고른 성공 보고가 아니다.

공개122조건 재생: 정상64조건 새 경고0, 기존 손가락 안내20개 유지,
방향·순서·기존 행동 구간도 유지. 정상속도/교차/순서/누락 등의 대조를 포함한다.
이122는 이미 사용한 원본20영상에서 만든 시험 조건이며 실제 사용자122명 시험이 아니다.

남은 원인
손을 돌리며 같은 모양을 유지해도 정면 투영이 달라져 '모양 변화'로 읽는 사례가 있다.
따라서 변화 감지만으로 현재 모양을 확정할 수 없다. 실제짧은폄 사례의 독립 확인 근거도
아직 적다. 목표였던 WORD2551 REAL12 f35의 검지폄도 현재 보완으로 복구되지 않았다.
후속은 모양유지+회전과 실제폄/굽힘을 구별하는 정답·관측품질부터 보강해야 한다.
캡스톤 자료 마감 전에는 이 연구를 반복하기보다 검증된 웹 기능과 한계를 확정한다.

보호/검산
984개 저장값·분리·불변성 검사 통과. 이는 성능984개 통과라는 뜻이 아니다.
C·T3-F/현재웹 등518파일 동일. 개인 촬영 열람·학습·튜닝 없음.
학습·원본 확인·별도 재집계 담당 Codex, 독립 수어 전문가 검수 아님.
웹캠은 시작하지 않았다. 마지막 확인 시8821서버는 응답하지 않아 실행 중이라고 하지 않는다.

근거: DECISION.json, VERIFIED.json, SOURCE_TRUTH_AUDIT.json,
HISTORIC_SOURCE_AUDITED.json, v2_*_RESULTS.json, REPLAY_RESULTS.json.
제출/시연 계획: CAPSTONE_READINESS.txt, DELIVERY_PLAN.json, DELIVERY_GRAPH.png.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')

plan=dict(at=at,goal='캡스톤 자료3일 미만: 현재 구현·검증 사실을 제출 가능한 자료로 고정. 실제 발표12월1일경: 제한4단어 웹 시연 안정화. S-CURT:세부판정 확장.',
 deadline_source='User: materials due in less than3days, actual presentation around Dec1; exact submission time and live-demo requirement not confirmed. Dates below are work targets, not measured completion estimates.',
 scope=dict(recognition=100,shared_feedback_learning=100,detailed_feedback_validation=['조상','이집트','순종','잘하다'],new_quest_words='안녕하세요/사과/감사 추가는 본 제출의 필수 완료 범위에서 제외; 현재4단어 지원과 혼동 금지',out_of_scope=['new model replacement','LLM','Unity migration','all100 detailed correctness','avatar expansion']),
 stages=[
 dict(id='P0',window='지금~24시간',task='현 버전·발표 주장 고정',deliverable='구현/공개대조/사용자미확인 구분한 주장표 + 정상/교정/확인필요 화면 + 구조도',gate='모든 화면·수치 출처가 존재하고 실제촬영/편집시험 표시. 새 후보 미적용 상태 명시',on_fail='근거 없는 주장 삭제, 후속 과제로 이동'),
 dict(id='P1',window='다음24~48시간, 자료 제출 전',task='발표자료 구성과 검수',deliverable='목표→관절재사용→TCN/피드백역할→설명주석/DTW→실제화면→제한/계획 6장 안팎',gate='100단어 인식과4단어 피드백, 과거실측/최신버전, 합성오류/사용자오류를 섞지 않음',on_fail='미검증 기능의 완료 표현 제거, 현재 화면으로 대체'),
 dict(id='P2',window='자료 제출 후 우선, 늦어도11월21일 목표',task='실제 웹캠 전체 경로 수용시험',deliverable='한손순종/양손잘하다 정상과 중간겹침, 내려서종료, 피드백표시/마지막프레임→화면 지연 기록',gate='사용자가 시작한 촬영으로 확인; 모델은 동결, 개인자료는 학습/튜닝/후보선택 금지; 원치 않는 중간종료나 UI멈춤은 실행결함으로 해결',on_fail='포착/종료/표시 문제를 공개 재현으로 분리 보완, 판정불확실은 성공/오답으로 덮지 않음'),
 dict(id='P3',window='11월22~27일 목표',task='시연 시나리오와 실행환경 정리',deliverable='재시도 포함 짧은 데모, 로컬필수모델/예시/아바타 점검, HDD없음·서버재시작 확인, 예비녹화',gate='W의존성 실제 전체경로 검사. 공개통제시험은 설명을 붙여 시연하고 사용자실수인척하지 않음',on_fail='미해결 기능은 확인필요 표시, HDD가 필요하면 필수장비로 명시하고 예비공개결과 재생 준비'),
 dict(id='P4',window='11월28~30일 목표 / 실제발표12월1일경',task='시연 버전 고정과 리허설',deliverable='모델교체 없는 반복 실행, 발표PC/해상도/카메라/서버 준비, 실패시 복구경로',gate='정상·교정·보류 화면을 설명할 수 있음; 라이브와 저장결과 구별. 정확발표일 확인 시 일정 보정',on_fail='녹화/저장 공개결과를 보조 자료로 사용 가능 여부 확인, 실시간 성공으로 표시하지 않음'),
 dict(id='S1',window='S-CURT 후속',task='세부판정 연구 보완',deliverable='현재모양/회전중유지/실제변화 정답 구별, 가림품질, 다양한 왼손/약지/새끼/짧은폄, 신규공개예약평가',gate='현재정상 보존 + 실제 짧은폄 회복 + 새오판비증가; 기존확인자료는 이제개발노출자료',on_fail='원본정답→관측품질→모양추론→구간연결 순서로 원인 귀속, 동일시험으로 무한 재선택 금지')],
 implementation_remaining=['G6 physical capture acceptance','false-positive/missed-meaningful-error coverage beyond controlled public cases','HDD-independent full run and startup packaging','presentation/evidence/backup rehearsal'],
 user_input_needed='실제 웹캠 수용시험 시 사용자 촬영만 필요; 지금 자료 정리·공개 근거 정리에는 추가 답변 불필요.',
 safety='No public deployment, no download, no private training, keep protected C/T3F/shared100. New experiment not applied.',
 evidence=[str(U/'DECISION.json'),str(O/'word100_live_g6_20261114/RESULTS_REVIEW.json'),str(P/'RESULT_KO.txt'),str(O/'word100_public_g4_20261114/RESULT_KO.txt')])
write(U/'DELIVERY_PLAN.json',plan)
readiness='''캡스톤 제출과 실제 시연을 나눈 마무리 계획

지금 상태
뼈대부터 새로 만드는 단계는 지났다. RTMW 관절 추출을 촬영 중에 수행하고,
TCN 단어 인식, DTW 비교, 설명과 연결한 세부판정, 손·팔 강조/교정문장/확인필요
표시까지 웹에 연결돼 있다. 공개자료 입력으로 전체 연결은 확인했다.
TCN은100단어, 공통학습도100단어지만 세부피드백의 우선 검증은4단어다.
모든 손모양·반복·접촉·전체 수어 정오를 판단하는 완제품은 아니다.
G0~G5는 제한된 범위 완료. G6는 실제 사용자 웹캠 확인이 남아 있다.

남은 큰 조각
① 실제 카메라에서도 제대로 끝나고 결과가 나오는지 확인.
② 중요한 오류를 찾으면서 정상 차이를 오답으로 지적하지 않는 범위 확보.
   현재는 일부 손모양/방향/순서/부분수행에 근거가 있고, 손가락·접촉·반복전체는 제한.
③ HDD 없이 켜서 시연하고 실패 때 복구할 실행 묶음 점검.
④ 발표용 화면·실측·한계 정리. 기존 그림/화면을 재사용하면 새 모델을 기다릴 필요가 없다.
따라서 근거 없는 완성률% 대신 '기능 연결됨 / 제한범위 공개검증됨 /
실제촬영 수용시험과 범위 확대는 남음'으로 설명하는 것이 정확하다.

3일 미만 자료 제출: 오늘 모델 범위 고정, 24시간 안에 핵심화면·구조도 정리,
다음24~48시간에 발표자료 제작·검수. 사용자 제작을 돕는6장 구성:
1 목표: 단어 맞히기에서 어느 순간을 어떻게 고칠지 안내하기로 확장.
2 처리 구조: 촬영중RTMW → TCN단어인식 / 상대관절 시간합성곱+GRU → DTW연결 → 화면.
3 학습 정답: 설명을 손모양/신체관계/움직임 조건으로 나누고 공개 시점과 연결.
4 결과: 정상 / 확인된 차이 / 판단보류의 실제 저장 화면. 대조 입력임을 캡션으로 표시.
5 검증: 공개 원본과 속도·역방향·부분수행 대조. 신뢰가능한 범위와 아직 안되는 것.
6 이후: 실제촬영·HDD독립 시연 → S-CURT에서 세부손가락/반복/접촉 보완.

속도 발표 주의
현재 연결 계보 G6의 공개영상 브라우저 재생에서 순종2.87초·잘하다1.55초를 기록했다.
이는 마지막프레임부터화면확인, 물리웹캠이 아니며 손가락 보조 추가 전 G6 버전이다.
최신손가락보조 웹의 공개저장관절1회2.13초는 검출/촬영/화면표시 제외다.
두 값을 같은 조건의 최신속도 측정처럼 섞거나 '모든촬영2초'라 말하면 안 된다.
과거촬영후/촬영중 비교는12영상×3반복, 완료35쌍 평균5.39→2.57초였으나
촬영중36회 중1회 큐초과 중단도 있었다. 과거9~10초 체감과 직접 같은 실험이 아니다.

실제시연(12월1일경): 자료 제출 후 먼저 한손·양손 실제 촬영을 확인한다.
잘되면 재시도와HDD독립, 서버 재시작, 다른앱 부하를 점검한다.
안되면 입력/자동종료/판정/표시 중 어느 부분인지 공개 재현으로 고친다.
11월28일경부터 모델 교체 없이 리허설. 날짜는 작업 목표이며 완료 보장이 아니다.
사용자가 촬영을 미룬 상태를 존중하며 자동으로 카메라를 켜지 않는다.

시연에서 보여줄 범위
잘하다: 손모양과 팔을 따라가는 방향의 근거/강조. 순종: 한손 동작과 행동순서.
조상·이집트: 기존지원범위 유지. 반복전체/이집트마지막세밀한손끝 검증완료로 확대하지 않는다.
오류예시는 실제로 확보된 공개 대조 결과를 사용. 개인촬영에서 검증되지 않은 오류를
이미 잘잡았다고 말하거나 대조영상을 실제사용자오류라고 보여주지 않는다.
지원 안된 인사·사과 퀘스트 추가/Unity병합/LLM/새3D연출은 제출 필수에서 뺀다.

S-CURT: 이번 원본감사에서 드러난 정답오류부터 추적하고, 회전하면서손모양유지와
실제폄/굽힘을 나눠 학습한다. 정면가림·왼손조합·짧은폄·엄지/접촉/반복은 각각
근거를 갖춰 통과한 것부터 확장한다. 단어 수만 늘리는 방법으로 대신하지 않는다.
'''
(U/'CAPSTONE_READINESS.txt').write_text(readiness,'utf-8')

evidence=[dict(use='손·팔 강조와 교정문장',path=str(P/'WEB_FINGER_CHECK.jpg'),caption='공개 관절값을 변형한 대조시험의 저장 판정. 실제 사용자 오답/DTW 재생 성능 시험은 아님.'),dict(use='차이 시각화',path=str(O/'word100_visual_feedback_g5_20261114/BROWSER_DIFFERENCE.jpg'),caption='공개 영상 역재생 대조의 실제 분석 결과. 새 사용자 촬영이 아님.'),dict(use='판단보류 시각화',path=str(O/'word100_visual_feedback_g5_20261114/BROWSER_UNCERTAIN.jpg'),caption='관측·판정이 충돌한 경우 확인필요 표시. 오답으로 확정하지 않음.'),dict(use='DTW 원리와 대응점',path=str(R/'Docs/final_check_20261112/발표흐름_추천9장/07_dtw_pdf_three_moments.png'),caption='기존 발표용 대응점 설명. 현재 손가락 후보의 정확도 개선 증거로 쓰지 않음.'),dict(use='현재 진행과 제출계획',path=str(U/'DELIVERY_GRAPH.png'),caption='현재 기록 기반 작업상태. 미래 단계는 계획.')]
write(U/'PRESENTATION_EVIDENCE.json',dict(items=evidence,scope='Existing evidence map, not a new slide deck or camera demo. Original assets unchanged.'))

im=Image.new('RGB',(1600,1050),'#f5f7fa');d=ImageDraw.Draw(im);navy='#17324f';muted='#53677d';line='#a9b6c5'
font=lambda s,b=False:ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf'if b else'C:/Windows/Fonts/malgun.ttf',s)
d.text((70,42),'현재 완성 범위와 제출까지 남은 작업',font=font(43,True),fill=navy)
d.text((70,108),'기능은 연결됨 · 실제 촬영 확인과 세부 판정 보완은 남음',font=font(26),fill=muted)
def box(x,y,w,h,title,body,filled=False):
 d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill=navy if filled else 'white',outline=navy,width=2)
 d.text((x+24,y+21),title,font=font(27,True),fill='white'if filled else navy)
 for j,t in enumerate(body):d.text((x+24,y+67+34*j),t,font=font(23),fill='#e1e8ef'if filled else muted)
def arrow(x,y):
 d.line((x,y,x+35,y),fill=navy,width=3);d.polygon([(x+35,y),(x+25,y-7),(x+25,y+7)],fill=navy)
box(70,190,460,165,'G0~G2 · 학습 기반', ['설명 정리 → 공개 시점 정답 → 학습','100단어 공통학습 유지'],True)
arrow(540,270)
box(585,190,420,165,'G3~G5 · 판단과 화면', ['우선 4단어 · 제한 범위 검증','차이 강조 / 교정 / 판단보류'],True)
arrow(1015,270)
box(1060,190,470,165,'G6 · 실제 촬영 확인 남음', ['공개 입력의 웹 연결은 확인','사용자 웹캠 · 종료 · 전체 대기시간'])
d.text((70,390),'이번 손가락 연구: 분리 학습·검증 완료 / 추가 오판이 남아 새 후보 미적용',font=font(25,True),fill=navy)
d.line((70,455,1530,455),fill=line,width=2)
d.text((70,490),'마감별로 목표 분리',font=font(30,True),fill=navy)
box(70,555,460,260,'캡스톤 자료 · 3일 미만', ['현재 검증된 화면·구조도 고정','정상 / 교정 / 보류 예시 정리','공개 대조와 실제 촬영 구별','새 연구 결과를 기다리지 않음'])
box(585,555,420,260,'실제 발표 · 12월 1일경', ['한손·양손 웹캠 수용시험','자동 종료·재시도 안정화','HDD 없이 실행 확인','버전 고정·녹화·리허설'])
box(1060,555,470,260,'S-CURT · 후속 고도화', ['회전 중 유지 / 실제 변화 구분','짧은 폄·가림·손가락 조합','반복·접촉 등 미검증 범위','새 공개자료로 통과 후 확장'])
d.text((70,865),'완료 기준',font=font(27,True),fill=navy)
d.text((70,916),'관측한 차이만 설명하고, 모르는 부분은 보류하며, 촬영부터 결과까지 시연 가능',font=font(27),fill=navy)
d.text((70,986),'공개 검증은 제한 범위 · 전체 수어 정오 판단의 완성을 뜻하지 않음',font=font(21),fill=muted)
im.save(U/'DELIVERY_GRAPH.png')
print('decision, report, delivery plan and graph saved')
