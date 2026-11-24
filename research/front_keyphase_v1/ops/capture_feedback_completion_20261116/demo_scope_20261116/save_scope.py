from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image,ImageDraw,ImageFont
import json,shutil
U=Path(__file__).resolve().parent;G=U.parent;O=G.parent;R=O.parents[2]
read=lambda p:json.loads(p.read_text('utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();v=read(U/'VERIFICATION.json');srv=read(O/'web_tutorial_live_integration_20261113/SERVER.json')
scope=dict(at=now,decision='안녕하세요는 현재 시연에서 제외; 사과·감사 연습/실전 통일. 과거 인사 연구는 보존.',user_capture='사용자 나중에 촬영할게 응답: 실제촬영 대기, 개발 중단 요청 아님',runtime='apple_web_adapter -> gratitude -> preserved G6; hello worker not loaded',verification=v,training=False,models_changed=False,server=srv,launcher=str(R/'START_WEB_DEMO.cmd'),next='사용자가 촬영할 때 사과·감사 정상→교정→재시도/퀘스트와 실제 화면 대기시간 확인. 그 전 추가 학습 근거 없이 모델 변경하지 않음.')
save(U/'CHECKPOINT.json',scope)
for name in ['CHECKPOINT.json','plan.json','TASK_GRAPH.png']:
 p=G/name;dst=U/'before'/name
 if p.exists()and not dst.exists():shutil.copyfile(p,dst)
c=read(G/'CHECKPOINT.json');c.update(at=now,status='TWO_WORD_DEMO_READY_USER_CAPTURE_DEFERRED',server=srv,scope=dict(quest=['WORD0099','WORD1290'],practice=['WORD0099','WORD1290'],greeting='Excluded from current demo by user discussion; archive preserved',old_four='Retained research regression options, not quest'),next=scope['next'],current_scope_record=str(U/'CHECKPOINT.json'));save(G/'CHECKPOINT.json',c)
p=read(G/'plan.json');p.update(updated_at=now,status='TWO_WORD_DEMO_READY_USER_CAPTURE_DEFERRED',next=scope['next'],remaining='실제 웹캠 확인은 사용자 요청으로 나중에 진행. 안녕하세요는 현 시연 범위에서 제외.',scope_decision=str(U/'CHECKPOINT.json'));p['scope']['demo_priority']=['WORD0099 사과','WORD1290 감사'];p['scope']['parked']='SEN0354 안녕하세요: 연구 보관, 시연·실행·추가학습 제외'
for n in p['nodes']:
 if n['id']in'ABCDE':n['status']='DONE_FOR_APPLE_GRATITUDE'
 else:n['status']='PUBLIC_CAPTURE_PASSED_USER_CAPTURE_DEFERRED'
save(G/'plan.json',p)
s=read(O/'STATE.json');s.update(current_stage=G.name,next_single_action=scope['next']);s[G.name].update(status=c['status'],updated_at=now,next_single_action=scope['next'],scope_checkpoint=str(U/'CHECKPOINT.json'),report=str(U/'RESULT_KO.txt'),web_scope=c['scope']);save(O/'STATE.json',s)
report='''현재 시연은 사과·감사로 통일

연습: 공개 참고 → 3초 준비 → 촬영 중 RTMW 추출 → 손 내림 자동 종료 → TCN 인식·세부 피드백 → 재시도.
실전: 사과 요청 → 감사 인사. 단어 인식과 필요한 행동 근거가 모두 있어야 통과.
안녕하세요는 연습 메뉴·공개 예제·API·전용 분석기 로딩에서 제외. 연구 파일과 결과는 보관. 조상·이집트·순종·잘하다는 기존 회귀 확인용으로 보존하며 실전 단어가 아님.

확인: 새 서버의 사과·감사 공개 정상 입력에서 단어1위와 필수항목 통과 유지. 안녕하세요 세 경로404, 메뉴·시연계획·예제 없음. 공개시험으로 퀘스트 진행하지 않음. 기존 모델 파일 해시 유지, 계약6검사 통과. 새 학습0·사용자 촬영0.
실행: 프로젝트 루트 START_WEB_DEMO.cmd를 두 번 클릭. 서버가 있으면 재사용하고 없으면 숨겨진 서버를 준비한 뒤 브라우저를 열도록 구성. 카메라는 사용자가 화면에서 켠다. --prepare-only로 서버 새 시작을 시험했고 --check-only로 준비상태를 확인했다. .cmd 더블클릭 자체는 사용자 미시험.
현재 주소: http://127.0.0.1:8821/

남음: 실제 사용자 촬영은 사용자가 나중에 하기로 했다. 실제 앉은 자세에서 자동 종료·정상 통과·교정·재시도와 화면 대기시간은 그때 확인. 공개자료 검증만으로 새 사용자 정확도나 전체 기능 완성을 선언하지 않는다.
HDD: 이전 C: 실행 경로/W: Python 읽기차단 검사 통과. 실제 HDD를 분리한 촬영 시연은 미확인.
이번 변경은 시연 범위와 실행 준비 정리이며 모델 정확도 향상을 주장하지 않는다. 기존 C·T3-F·공통100 모델 및 사용자의 학습/튜닝/후보선택 금지 유지.
''';(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
navy='#17304F';gray='#52657D';font=lambda sz,b=False:ImageFont.truetype('C:/Windows/Fonts/malgun'+('bd'if b else'')+'.ttf',sz)
im=Image.new('RGB',(1440,1190),'white');d=ImageDraw.Draw(im)
d.text((72,38),'사과·감사 시연 · 작업 현황',font=font(44,True),fill=navy);d.text((75,110),'연습 → 교정·재시도 → 사과 요청 → 감사 인사',font=font(27),fill=gray)
rows=[('A','기존 기능·빈칸 확인','완료'),('B','설명과 공개 정답 연결','완료'),('C','원인별 보완·필요한 학습','완료'),('D','정상·오류·미관측 비교','공개 검사 완료'),('E','화면·퀘스트 연결','사과·감사 연결'),('F','실제 촬영·최종 시연','사용자 촬영은 나중에')]
for i,(key,title,status)in enumerate(rows):
 y=182+i*133;d.rounded_rectangle((95,y,1345,y+100),14,fill='#F0F4F8',outline=navy,width=2);d.text((123,y+24),key,font=font(33,True),fill=navy);d.text((200,y+23),title,font=font(30,True),fill=navy);d.text((980,y+29),status,font=font(24),fill=gray)
 if i<5:d.line((715,y+101,715,y+122),fill=navy,width=3);d.polygon([(707,y+115),(723,y+115),(715,y+127)],fill=navy)
d.text((95,1015),'안녕하세요: 현재 시연에서 제외 · 연구 결과는 보관',font=font(27,True),fill=navy);d.text((95,1075),'기존 C·T3-F와 공통 모델 유지 · 사용자 촬영 학습·튜닝 금지',font=font(25),fill=gray);im.save(G/'TASK_GRAPH.png')
f=R/'WordsInSilence_PROJECT_CONTEXT.md';t=f.read_text('utf-8');h='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';t=t.replace(h,h+'\n\n### 2026-11-16 시연 사과·감사 통일 / 안녕하세요 제외\n사용자 통일을위해안녕하세요일단제외 의견 반영.8821의인사메뉴/예제/API/worker로딩제외,연구원본보존.사과·감사공개정상인식/필수항목유지·모델해시동일·계약6검사.추가학습0/개인촬영0.루트START_WEB_DEMO.cmd+demo_launcher.py로서버준비/재사용확인.사용자실제촬영은나중에하겠다고응답.근거capture_feedback_completion_20261116/demo_scope_20261116/CHECKPOINT.json.\n',1);h='## 9. 다음 단일 작업';t=t.replace(h,h+'\n\n### 2026-11-16 사과·감사 시연 유지 / 사용자 촬영 대기\n안녕하세요추가학습·시연재연결하지않음.사용자는실제촬영을나중에하겠다고함.8821사과·감사촬영후피드백/퀘스트준비유지.촬영시자동종료·정상/교정/재시도·실제대기시간확인,문제는공개자료로재현해보완.현재범위는demo_scope_20261116/CHECKPOINT.json.\n',1);f.write_text(t,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}
## 현재 결정
사용자: 감사촬영은나중에/작업계속, 통일을위해안녕하세요일단사용하지말자 의견.사과·감사연습/실전시연으로정리. 안녕하세요메뉴/예제/API/worker연결제외,hello_capture_v1연구는보존. 재연결/새학습임의진행금지.
## 근거
research/front_keyphase_v1/ops/capture_feedback_completion_20261116/demo_scope_20261116/CHECKPOINT.json·VERIFICATION.json·RESULT_KO.txt, 상위plan.json·TASK_GRAPH.png.
## 실행
8821 일반환경 listener{srv['pid']},카메라꺼짐. 실제PID/state항상확인.현재server.py는apple_web_adapter→감사→기존G6.사과/감사공개정상TCN1위·필수동작통과,인사3경로404.기존C·T3-F/100가중치및판정핀유지.모델재학습0.로그demo_scope_20261116/launcher_stdout.log·launcher_stderr.log.
프로젝트루트START_WEB_DEMO.cmd:Python서버없으면준비/있으면재사용후브라우저.카메라는사용자만시작.실행로직--prepare-only/--check-only확인,실제.cmd더블클릭사용자미확인.
## 다음
A~E사과·감사범위완료,F공개촬영루프완료·실제사용자촬영은나중에.정상→차이교정→정상재시도/퀘스트·실제화면시간확인.문제시공개재현후보완.개인촬영학습/튜닝/후보선택금지.실제HDD분리웹캠미확인.공통회귀기존4단어유지,최종시연으로바꾸지않음.과거미사용16/17재진단으로최종독립평가아님.
''',encoding='utf-8')
with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8')as f:f.write(f'\n[{now}] 시연일관성과준비수준으로안녕하세요제외.한참고단계연습과TCN+필수항목실전의서로다른범위를섞지않고사과·감사로통일.인사연구삭제없음/모델재학습없음.자동촬영실제사용자검증은나중에.근거{U}/CHECKPOINT.json.\n')
print('Scope/checkpoint/graph/history updated')
