from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];P=U/'apple_v1';V=U/'completion_v1';H=U/'hello_capture_v1'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();server=read(O/'web_tutorial_live_integration_20261113/SERVER.json')
record=dict(at=now,status='A_TO_E_BOUNDED_DEMO_CONNECTED_F_PUBLIC_CAPTURE_DONE_USER_WEBCAM_PENDING',actor='Codex root implementation and separate execution/source checks; no new independent expert',server=server,scope=dict(quest=['WORD0099','WORD1290'],guided_practice='SEN0354 single-public-person prototype, no word TCN, no quest pass',old_four='preserved regression, not final demo replacement'),apple=dict(normal_fit=8,normal_development_confirmation=7,coordinate_controls=70,all_words_accuracy=None,final_heldout_independent=False,capture=str(P/'web/capture_role_v2/CAPTURE_CHECK.json')),gratitude=dict(normal_development_confirmation=6,capture=str(V/'web/CAPTURE_CHECK.json')),hello=dict(public_people=1,controls=24,normal_crop_conditions=3,training=False,known_limit='Reversed sweep and closed sweeping hand can remain unknown rather than a specific correction',capture=str(H/'web/capture/CAPTURE_CHECK.json')),data_rules=dict(user_training=False,user_tuning=False,user_selection=False,C_T3F_preserved=True,shared_100_backbone_preserved=True,apple_classifier='original100 frozen logits plus separately trained apple output'),next='User-started actual webcam verification of apple/gratitude normal -> correction -> retry/quest. Reproduce any failures on public data before changing models. Hello remains guided prototype pending multi-person evidence and word-recognition decision.',evidence=[str(H/'web/API_CHECK.json'),str(P/'web/EXAMPLES_CHECK.json'),str(H/'web/EXAMPLES_CHECK.json'),str(P/'web/PORTABILITY_CHECK.json')])
save(U/'CHECKPOINT.json',record)
save(H/'CHECKPOINT.json',dict(at=now,status='LIMITED_GUIDED_PRACTICE_CONNECTED',validation=str(H/'web/ADMISSION.json'),webcam_pending=True,training=False,private_data=False,unadmitted_candidates=['whole-clip single-pass encoder changed direction: 23/24','initial shape grouping included preparation: 16/24'],scope=record['scope']['guided_practice']))
plan=read(U/'plan.json');plan.update(updated_at=now,status='BOUNDED_DEMO_READY_ACTUAL_WEBCAM_PENDING',next=record['next'],remaining='실제 웹캠의 정상·교정·재시도 확인. 안녕하세요는 단계 연습이며 다인 검증·TCN 인식 미완료.',completed_this_turn=['감사 필수6항목·정상완료·교정 연결','사과 공개정답/작은판정부 학습/7필수항목/역방향교정/정상완료','사과 상반신 화면의 불필요한 반대손 의존성 수정','안녕하세요 기존 손모양·Conv+GRU를 촬영후 단계연습으로 재사용','3표현×2공개화면 촬영루프: 카운트다운/자동종료/관절재사용','브라우저 정상·교정·부분 회색/DTW 재생 확인','사과→감사 퀘스트와 공개재생 분리','C: 모델·참고 의존성 점검'])
for n in plan['nodes']:
 if n['id']in'ABCDE':n['status']='DONE_FOR_APPLE_GRATITUDE_WITH_LIMITED_HELLO_PRACTICE';n['scope_note']='공개 확인한 보이는 필수특징의 시연 범위. 전수어 정확도/실제 사용자 완성 선언 아님.'
 if n['id']=='F':n['status']='PUBLIC_CAPTURE_PASSED_USER_WEBCAM_PENDING'
plan['stop_boundary']='사용자가 촬영을 시작해야 하는 최종 확인만 대기. 자율 진행 가능한 공개 검증·연결·기록은 완료.'
save(U/'plan.json',plan)
report='''촬영 후 피드백 시연 — 현재 결과

실행: http://127.0.0.1:8821/ (02 연습 / 03 실전)
사과·감사: 촬영 중 RTMW → 자동 종료 → TCN 인식 + 필수동작 검사 → 교정/확인부족/완료 → 재시도·퀘스트 연결.
안녕하세요: 승인된 참고 1명의 두 단계 연습을 촬영후 경로에 연결. 기존 학습 손모양과 시간 합성곱+GRU 방향 관측, 명시적 위치·반복 규칙을 조합한다. 새 단어 TCN과 다인 정확도는 검증하지 않았으므로 실전 퀘스트 통과에 사용하지 않는다.

이전과 달라진 핵심
1. TCN이 단어를 맞혔다는 이유만으로 성공하지 않는다. 사과7개·감사6개 필수특징의 실제 관측 근거가 모두 있어야 퀘스트가 진행된다.
2. 사과에서 손목 밖으로 사라진 반대손 때문에 입 주변 동작도 놓치던 입력 의존성을 수정했다. 상반신 화면의 정상 동작을 유지했다.
3. 안녕하세요의 손을 주먹으로 바꾸는 준비 구간을 반복 동작과 섞어 경고하던 문제를 분리했다. 앞 행동의 학습 방향 근거를 그대로 보존하고 다음 행동 준비를 오답으로 만들지 않는다.
4. 차이가 난 순간을 주황색 손·팔과 짧은 교정 문장으로 표시한다. 미관측·미완료는 회색이며 근거 없이 오답/성공으로 만들지 않는다. DTW는 해당 행동끼리 비교 재생하는 데 쓰며 누락 동작을 채우지 않는다.

공개 검증
사과: 학습8명+개발/확인7명 정상 유지. 5개 공개 영상의 70개 좌표·시간 변형 검사 통과. 일부만 수행/반대 방향/손모양 차이/순서 등 시험이며 실제 학습자70명의 정확도가 아니다. 16/17은 최초 확인 후 문제 진단에 재사용했으므로 최종 미사용 평가로 부르지 않는다.
감사: 개발/확인6명 정상, 기존 손모양·공통경로 회귀 유지. 상세 completion_v1/CHECKPOINT.json.
안녕하세요: 하나의 공개 영상에서 만든24조건과 화면3조건 유지. 새 수행자 일반화 증거가 아니다. 반대 쓸기·닫힌 손 쓸기는 구체적 교정보다 확인 부족으로 남는 경우가 있다.
공개 영상을 카메라 대신 넣어 실제 RTMW·영상저장·자동종료까지 수행한 검사: 사과 전체/상반신 약0.25~0.30초, 감사 약0.23~0.28초, 안녕하세요 약0.16~0.17초(종료 후 계산). 촬영 중 추출은 이미 수행되며 이 시간은 실제 웹캠·브라우저 표시시간과 다르다. 매 촬영마다 여러 원영상을 읽어 일대일 비교하는 구조가 아니다.

시연 준비
02 연습: 사과 또는 감사 선택 → 공개 참고 보기 → 카메라 켜기 → 손 내린 자세 → 3초 뒤 촬영 → 수행 후 손 내림 → 결과.
초록은 확인한 특징, 주황은 관측된 차이, 회색은 확인 부족. 주황 순간 버튼을 누르면 해당 손·팔과 교정 방법을 볼 수 있다.
03 실전: 사과 → 감사. 단어 인식과 필수특징 판정이 모두 맞아야 진행. 공개 예시 재생은 퀘스트에 반영하지 않는다.
안녕하세요는 연습 목록의 '단계 연습'만 사용. TCN 후보를 임의로 만들어 표시하지 않는다.

남은 최종 확인
실제 사용자가 앉은 상태에서 사과·감사를 정상 수행하고, 잘못된 방향 또는 손모양을 한번 시험한 뒤 다시 정상 수행한다. 자동종료, 교정 부위, 정상 재시도와 실제 화면 대기시간을 확인해야 한다. 사용자 촬영은 학습/튜닝/후보선택에 사용하지 않는다. 문제가 보이면 공개 자료로 재현 후 보완한다.
HDD: 사과·감사 모델/참고는 C:에서 읽는 실행 확인을 했다. Python W: 접근 차단 검사 통과. 실제 HDD 분리·물리웹캠 시연은 아직 미확인이며 네이티브 전체 파일접근 차단을 주장하지 않는다.
발표에서 '모든 수어를 이해하거나 모든 오류를 잡는다'고 말하지 않는다. 현재 시연은 선택한 기초 단어의 확인 가능한 손모양·위치·방향·순서를 판정하는 제한 범위다. 정확한3D접촉/숨은손가락/새 사용자 모든 오류는 검증 완료가 아니다.

보존: 기존 C·T3-F, 100단어 공통 신경망 가중치와 기존4 회귀. 사과 TCN만 기존100 로짓을 보존한 추가1출력 연구경로. 안녕하세요 추가학습0. 사용자자료 학습/튜닝/선택0.
구현 및 검증: Codex root. 별도 수어 전문가나 새 독립 에이전트 검증은 하지 않았다.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
f=U/'TASK_GRAPH.png'
if f.exists()and not(U/'TASK_GRAPH_before_resume.png').exists():shutil.copyfile(f,U/'TASK_GRAPH_before_resume.png')
navy='#17304F';gray='#52657D';light='#F0F4F8';font=lambda size,b=False:ImageFont.truetype('C:/Windows/Fonts/malgun'+('bd'if b else'')+'.ttf',size)
im=Image.new('RGB',(1440,1270),'white');d=ImageDraw.Draw(im)
d.text((72,38),'촬영 후 피드백 · 작업 현황',font=font(44,True),fill=navy)
d.text((74,108),'사과·감사: 웹 시연 연결 완료  /  마지막은 실제 웹캠 확인',font=font(25),fill=gray)
rows=[('A','기존 기능·빈칸 확인','완료','관절 추출 → 인식 → 행동 판정의 실제 연결 점검'),('B','설명과 공개 정답 연결','완료','손모양 · 위치 · 방향 · 반복 · 끝상태를 필수항목으로'),('C','원인별 보완·학습','완료','사과의 불필요한 반대손 의존성 / 준비와 주요동작 분리'),('D','정상·오류·미관측 비교','공개 검사 완료','정상 유지 · 반대 방향 교정 · 부분 수행은 완료 금지'),('E','화면·퀘스트 연결','완료','초록 확인 / 주황 교정 / 회색 보류 → 사과 → 감사'),('F','실제 촬영·최종 시연','사용자 확인 남음','공개 촬영루프 완료 / 실제 웹캠·재시도·화면 대기시간 확인')]
for i,(key,title,status,desc)in enumerate(rows):
 y=176+i*145;d.rounded_rectangle((92,y,1335,y+111),14,fill=light,outline=navy,width=2)
 d.rounded_rectangle((115,y+22,173,y+80),9,fill=navy);d.text((132,y+26),key,font=font(31,True),fill='white');d.text((198,y+15),title,font=font(30,True),fill=navy);d.text((1005,y+22),status,font=font(23,True),fill=gray);d.text((198,y+64),desc,font=font(23),fill=gray)
 if i<5:d.line((715,y+112,715,y+133),fill=navy,width=3);d.polygon([(707,y+126),(723,y+126),(715,y+138)],fill=navy)
d.text((92,1060),'문제가 보이면 원인에 맞는 B·C로 돌아가 공개 자료로 수정 → D 재확인',font=font(25,True),fill=navy)
d.text((92,1120),'안녕하세요: 참고 1명의 단계 연습 연결 · TCN 인식·다인 검증은 남음',font=font(25),fill=gray)
d.text((92,1170),'기존 C·T3-F 유지  |  사용자 촬영은 학습·튜닝에 사용하지 않음',font=font(23),fill=gray)
im.save(f)
s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=record['next']);s[U.name].update(status=record['status'],updated_at=now,completion_checkpoint=str(U/'CHECKPOINT.json'),report=str(U/'RESULT_KO.txt'),graph=str(f),web_scope=record['scope'],next_single_action=record['next']);save(O/'STATE.json',s)
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8');h='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';t=t.replace(h,h+'\n\n### 2026-11-16 촬영후 기초시연 A~E 연결 / 실제 웹캠 확인 대기\n사과·감사 공개정상/필수항목·교정·회색·퀘스트연결 완료. 8821최종서버 ready, 공개3표현×2화면 실제RTMW촬영루프 자동종료·재검출0 확인. 안녕하세요는 기존모델재사용 단계연습(한명/TCN없음/실전불가),24파생조건·3화면 유지. 브라우저교정/부분회색/정상재생확인. 실제사용자웹캠·화면대기시간은남음. 근거 capture_feedback_completion_20261116/CHECKPOINT.json·RESULT_KO.txt·TASK_GRAPH.png. C·T3-F/100공통보존,개인자료학습0.\n',1);h='## 9. 다음 단일 작업';t=t.replace(h,h+'\n\n### 2026-11-16 최종 사용자 촬영 확인\n8821 02연습에서사과·감사 정상→교정→재시도,03실전사과→감사 확인. 카메라는사용자가시작. 진단만확인하고학습/튜닝/선택금지. 문제시공개재현→원인보완→D검증. 안녕하세요는단계연습유지;다인/TCN없어전체완료로주장하지않음. 공개검증·서버/UI·기록자율작업은완료.\n',1);p.write_text(t,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}
## 재개 승인 결과
A~E 사과·감사 제한 시연 완료. 안녕하세요 단계연습 연결(1공개수행자/TCN없음/실전통과불가). F 공개촬영 전체경로완료, 실제사용자웹캠확인남음. 학습을 계속 돌릴 상황으로 임의 해석하지 말 것.
## 먼저 읽기
research/front_keyphase_v1/ops/capture_feedback_completion_20261116/CHECKPOINT.json, RESULT_KO.txt, plan.json, TASK_GRAPH.png.
## 실행
8821 일반환경서버 listener {server['pid']}, 카메라꺼짐. 실제PID/idle재확인할것. hello_capture_v1/web/server_stdout.log와server_stderr.log. 웹최종import hello_web_adapter→apple_web_adapter→감사→기존G6. 사과/감사TCNtop1+필수동작정상API확인. 공개재생은퀘스트불가. 02연습사과선택·카메라꺼짐준비. 공개교정/부분회색/안녕하세요정상/DTW재생브라우저확인. feedback.js영상metadata후시점재설정 보완.
## 과학적 범위
사과8FIT+7개발확인정상/70좌표통제.16/17보완후더이상미사용아님. 감사6정상. 안녕하세요1원본24조건·3화면,정상2단계확인. 역방향/닫힌손쓸기일부는unknown으로남음;모든오류교정완료라말하지말것. 모델본문/C·T3-F동결,사과별도101번째TCN출력. 사용자학습/튜닝/선택0.
## 다음
사용자 실제촬영 대기. 사과·감사 정상→차이교정→정상재시도/퀘스트,자동종료·실제화면대기시간. 기존비동기실제촬영질문답변없음. 답변없다고실제카메라자동열지말것. 안녕하세요범위질문미응답시권장대로단계연습만연결한상태. HDD실제분리시연은미확인,Python W읽기차단하사과·감사 RTMW/모델/참고로드통과. 새광범위학습/LLM/실시간확장안함.
''',encoding='utf-8')
hist=R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with hist.open('a',encoding='utf-8')as f:f.write(f'\n\n[{now}] 촬영후 기초시연 A~E: 감사·사과 정상완료/교정/회색/퀘스트 연결. 손모양·설명·행동시작끝 정답으로 작은판정부 보강,사과 상반신 입력의 반대손 의존성 수정. 안녕하세요는1공개원본 단계연습으로명확히제한. 원래100공통/C·T3-F보존. 공개 실제RTMW촬영루프 3표현×2화면확인, 물리웹캠은사용자확인대기. 구현/검증Codex root,새독립전문가없음. 근거 {U}/CHECKPOINT.json.\n')
print(json.dumps(dict(checkpoint=str(U/'CHECKPOINT.json'),graph=str(U/'TASK_GRAPH.png'),status=record['status']),ensure_ascii=False))
