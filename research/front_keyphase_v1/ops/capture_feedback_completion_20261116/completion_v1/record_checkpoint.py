from train_shape import *
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image,ImageDraw,ImageFont
R=O.parents[2];now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();unit=V.parent
report='''촬영후 피드백 그래프 — 감사 1차 연결 완료 / 전체 작업은 계속

이번에 실제 달라진 점
1. 감사 설명의 손모양·손면·상대 도착 위치·두 번의 흐름을 함께 확인하는 완료 판정부를 연결했다. 단어 이름만 맞히거나 경고가 없다고 통과시키지 않는다.
2. 공유 손모양 모델을 공개 FIT1~9 자료로 보강했다. 정면으로 굽혀 길게 보이는 손은 기존 판단을 유지하고, 검수한 세운 손의 미확정 열린 모양만 보조한다. 원본 검수에서 DEV 정답1개가 전환 중 잘못 붙어 있음을 정정했다. 정정된 공통237 비교227맞음/1오판/9보류→228맞음/1오판/8보류; 기존 별도17시점12맞음/5보류→17맞음. 영상 전체 정확도가 아니다.
3. 첫 도달 시점을 동작의 끝으로 취급해 실제 끝자세를 잘라내던 연결을 수정했다. 다음 움직임/관측누락으로 넘어가지 않는 실제 유지구간까지 확인한다. 빠른 동작을 고정 시간길이로 거부하지 않는다.
4. 관절1프레임이 빠질 때 7프레임 이동창 전체가 사라지는 문제를, 실제 남은 관측6개로 확인하도록 제한 보완했다. 좌표·손가락 정답을 채워 넣지 않는다. 공개6개중5개에서 과제완료 유지, 18번은 여전히 확인 필요. 긴 누락은 통과시키지 않는다.
5. 웹8821에 정상 확인·손모양 교정·부분 수행 확인필요 및 감사 퀘스트 완료를 연결했다. 부분 촬영을 참고 전체 끝까지 늘리던 DTW 재생도 실제 확인한 참고 동작까지만 연결한다. 이 재생 연결은 판정 근거가 아니다.

확인 근거
- 감사 정상 공개6명(10,14~18): 기존 웹의 관절 검역과 TCN/판정부 경로에서6개 모두 필수6항목지원, TCN감사1위. 10번만 이번 후보 FIT/선택 밖이며 나머지는 기존 개발/진단이다.
- 고정35정상/속도시간변경/크기변경/부분/정지/누락 조건35통과. 속도변경은 타임스탬프 변환이며 실제 빠른 카메라의 성능 주장 아님.
- 공개 관절에서 손모양/손면/위치를 바꾼40진단조건 통과. 실제 사람이 일부러 틀리게 수행한40영상이 아니다. 뒤집은 반복은 여전히 유효할 수 있어 오답 정답으로 사용하지 않았다.
- 실제 다른 공개 단어(조상)를 감사 목표로 검사: 양손 굽힌 모양 차이 표시, 통과금지. 이것은 학습자 오류 연기 영상이 아니다.
- 기존4단어의 이전 소비기와 새 분기: TCN3순위·카드·문장·DTW 결과 동일.
- 새 게임계약6 + 기존계약6 검사통과. 공개 재생은 실제 퀘스트를 진행시키지 않는다.
- 공개 영상 카메라 대체입력2조건(전체640x360,상체640x480): 3초카운트다운·자동종료·프레임재검출0·정상완료 확인.126/125프레임 모두추출. 종료→계산0.277/0.229초. 실제 사용자 웹캠/브라우저 전체 대기와 구분한다.
- 실제 브라우저에서 정상·다른 손모양·첫동작 부분영상 확인. 지원되지 않던mp4v를H264로 수정했고, 부분 결과에서는 확인필요 항목을 먼저 보여준다.

남은 일
실제 사용자 촬영의 자동종료/이해가능성/화면까지 대기 확인. 사과의 검지→반대손 닦기 완료정답 및 판정 연결. 승인된안녕하세요는참고1개만있어다인일반화미검증. 긴 가림·정확3D손등접촉·다양한 실제오류전반은 보장하지 않는다. 이 단어의 정면 관측 가능한 학습 과제를 확인한 것이며 전체 한국수어 정답률은 아니다.

학습/검증 담당 Codex root. 독립 수어전문가/새에이전트검수아님. C·T3-F 및100공통동결모델보존. 사용자 촬영 학습·튜닝·선택0. 새외부다운로드0. 실패한무제한손모양/끝상태학습후보는격리보존. 현재통과한제한모양회복·손면관측만사용.
다음 단일 작업: 사과 기존공개FIT·개발자료의 검지/반대손닦기 구간과 출처를 확인해 범용행동정답으로연결. 감사 촬영요청은비동기대기, 작업중단아님.
'''
write(V/'CHECKPOINT.json',dict(at=now,status='GRATITUDE_WEB_CONNECTED_NEXT_APPLE',report=str(V/'RESULT_KO.txt'),
 actor='Codex root',training_runs=15,production_new_heads=6,normal_public_videos=6,old_four_regression=True,
 source_labels_corrected=1,user_training=False,C_T3F_unchanged=True,
 web_url='http://127.0.0.1:8821/',server_pid_file=str(V/'web/server_pid.txt'),
 browser_images=['web/browser_correction.jpg','web/browser_partial.jpg'],
 next='Apple clause/source completion; await optional user webcam confirmation without blocking public work.'))
(V/'RESULT_KO.txt').write_text(report,encoding='utf-8')
plan=read(unit/'plan.json');plan['updated_at']=now;plan['status']='IN_PROGRESS';plan['next']='사과 검지→반대손 닦기 공개 정답과 완료 연결. 감사 실제웹캠확인은비동기대기.'
for n in plan['nodes']:
 if n['id']in['B','C','D','E']:n['status']='DONE_FOR_GRATITUDE_PENDING_OTHER_DEMO_WORDS';n['evidence']=[str(V/'CHECKPOINT.json'),str(V/'web/ADMISSION.json')]
 if n['id']=='F':n['status']='PUBLIC_CAPTURE_PASSED_USER_WEBCAM_PENDING'
plan['remaining']='사과/안녕하세요의 필수행동 연결 및 공개검증, 실제웹캠 확인. 감사 제한완료 연결됨. 전체완료 아님.'
write(unit/'plan.json',plan)
s=read(O/'STATE.json');s['current_stage']=unit.name;s['next_single_action']=plan['next'];s[unit.name].update(status='IN_PROGRESS',updated_at=now,next_single_action=plan['next'],training_runs=15,web_admitted=True,web_scope='WORD1290 visible lesson only',completion_checkpoint=str(V/'CHECKPOINT.json'),whole_word_accuracy=None);write(O/'STATE.json',s)
context=R/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8');h='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';text=text.replace(h,h+f'\n\n### {now[:10]} 감사 정상완료·교정 웹 연결 / 사과 다음\n공개 정상6개 필수6항목과TCN감사1위 확인. 공유손모양제한회복·실제끝자세연결·단일관절누락보완,35회귀/40좌표진단 및 기존4소비기동일확인.8821에감사퀘스트완료/교정/확인부족연결,실제브라우저확인.공개카메라대체2조건자동종료/재검출0/종료후계산0.23~0.28초.실제웹캠미확인.사과/안녕하세요미완료로전체작업계속.작은판정부15학습실험,새웹사용6헤드(제한범위),100공통/C·T3-F보존,사용자자료학습0.근거capture_feedback_completion_20261116/completion_v1/CHECKPOINT.json·RESULT_KO.txt.\n',1);h9='## 9. 다음 단일 작업';text=text.replace(h9,h9+f'\n\n### {now[:10]} 감사 후 사과 필수행동 연결\n사과 공개FIT·개발·확인 자료의 검지→반대손닦기 원문/원본구간을확인하고학습정답·판정부·정상완료를연결한다.감사실제촬영확인은비동기질문대기하되독립공개작업계속.안녕하세요는승인된SEN0354만,다인자료미확보.현재8821서버PID는completion_v1/web/server_pid.txt확인(과거값가정금지).\n',1);context.write_text(text,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}
## 현재 작업: 재개 승인 유효, 계속 진행
감사 촬영후 피드백 B~E의 제한 시연 범위 연결. 전체 프로젝트 완료 아님. 다음은 사과 검지→반대손닦기 원문/원본 정답과 완료 판단 연결. 사용자에게 감사 웹캠 확인 비동기질문을 보냈으며 답변 없어도 공개 작업 진행.
## 읽을 파일
research/front_keyphase_v1/ops/capture_feedback_completion_20261116/completion_v1/CHECKPOINT.json
같은폴더 RESULT_KO.txt, web/ADMISSION.json, web/CAPTURE_CHECK.json, web/DISPLAY_CHECK.json
상위 plan.json. 기존PAUSE.json은이전이력,현재지시아님.
## 실제 상태
8821에감사정상통과·손모양교정·부분수행확인필요를연결하고브라우저확인했다.공개정상6개모두필수항목지원.공개실제RTMW촬영루프2조건자동종료·종료후계산0.23~0.28초.물리웹캠/브라우저전체대기는아직미검증.
web_adapter.py 새감사분기,기존4는이전소비기그대로;4단어TCN/카드/문장/DTW동일확인.C·T3-F·100공통동결가중치미변경.사용자촬영학습/튜닝/선택금지.
기존코드백업completion_v1/before_web. 제한모양회복/손면관측/도착구간으로최종6필수항목합성.정확3D접촉및전수어일반화보장아님.공통237비교의정답1개는원본RGB확인후정정,원본보존.전체15작은학습실험중실패후보보존.세부수치는RESULT_KO.txt.
## 프로세스
8821실제서버일반환경실행,카메라는사용자버튼만. PID web/server_pid.txt 확인. CUA공개검증완료선언했고현재탭deliverable.새촬영미실행.공개capture fixture이번종료여부는통합exec세션81487확인후정리.
## 남은 문제
1프레임누락6중5회복,18은회색.사과뒤닦기완료정답미구축.안녕하세요SEN0354참고1개외다인자료없음.기초게임1단어(감사)만정상완료연결,기존4를최종시연에임의대체안함.실시간8822확장보류,LLM/새3D확장없음.
## 다음 한 작업
basic_demo_feedback_20261115/acquisition_motion_followup/confirmation/SOURCE_TRUTH.json과사과원문/데이터를읽고필수구간정답준비.기존100+사과classifier후보는보존되어있으나웹미연결.단어별임계값/미관측정답채우기금지.
''',encoding='utf-8')
font=lambda n,b=False:ImageFont.truetype('C:/Windows/Fonts/malgun'+('bd'if b else'')+'.ttf',n)
im=Image.new('RGB',(1400,1400),'white');d=ImageDraw.Draw(im);navy='#192f49';gray='#586779'
d.text((70,45),'촬영 후 피드백 · 현재 작업 흐름',font=font(43,True),fill=navy)
d.text((70,110),'감사: 정상 통과·교정 연결 완료  /  사과·안녕하세요: 이어서 준비',font=font(26),fill=gray)
items=[('A','기존 기반과 빈칸 확인','완료','RTMW 촬영 중 추출 · TCN · 시간 합성곱+GRU · DTW 유지'),('B','설명과 실제 행동의 정답 연결','감사 완료','양손 모양 · 손면 · 상대 위치 · 두 번의 흐름'),('C','공통 판단·구간 연결 보완','감사 완료','펴진 손 확인 · 실제 끝자세 보존 · 짧은 누락 보완'),('D','정상 / 차이 / 확인 부족 검증','공개 검증 완료','정상 공개6명 · 좌표 변경 진단 · 기존4단어 유지'),('E','화면과 퀘스트 연결','감사 연결 완료','초록=확인 · 주황=교정 · 회색=확인 필요'),('F','실제 촬영과 최종 시연','사용자 확인 남음','공개 촬영루프 자동 종료 통과 · 실제 웹캠은 확인 전')]
for i,(num,title,state,desc)in enumerate(items):
 y=190+i*162;d.rounded_rectangle((80,y,1310,y+125),14,fill='#f1f4f8',outline=navy,width=2);d.text((105,y+25),num,font=font(40,True),fill=navy);d.text((180,y+18),title,font=font(31,True),fill=navy);d.text((1000,y+23),state,font=font(24,True),fill=gray);d.text((180,y+73),desc,font=font(24),fill=gray)
 if i<5:d.line((695,y+126,695,y+153),fill=navy,width=3);d.polygon([(687,y+148),(703,y+148),(695,y+159)],fill=navy)
d.text((80,1210),'다음: 사과의 검지 → 닦기 동작을 연결하고 같은 검증 과정 적용',font=font(28,True),fill=navy)
d.text((80,1270),'문제가 나오면 정답·관측·구간 연결 중 원인 단계로 돌아가 보완',font=font(25),fill=gray)
d.text((80,1320),'사용자 촬영 학습 금지 · C·T3-F 유지 · 전체 기능 완성을 뜻하는 그래프 아님',font=font(22),fill=gray)
im.save(V/'TASK_GRAPH.png')
print('CHECKPOINT_SAVED',now)
