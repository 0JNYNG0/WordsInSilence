from pathlib import Path
import json,hashlib,datetime,shutil,ast
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;ROOT=O.parents[2]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 at=datetime.datetime.now().astimezone().isoformat();ad=read(D/'web/ADMISSION.json')
 assert ad['admitted'] and all(sha(p)==h for p,h in ad['pins'].items())
 assert all(sha(p)==h for p,h in ad['checks'].items())
 http=read(D/'web/HTTP_CHECK.json');assert http['passed'];pid=http['server']['pid']
 stop=read(D/'checks/STOP_CHECK.json');short=read(D/'checks/SHORT_CHECK.json');normal=read(D/'web/CONSUMER_CHECK.json');cap=read(D/'web/CAPTURE_CHECK.json')
 speeds={k:dict(total=sum(r['kind']==k for r in short['rows']),before=sum(r['kind']==k and r['before']==2 for r in short['rows']),after=sum(r['kind']==k and r['after']==2 for r in short['rows'])) for k in ['speed_1.25','speed_1.5','speed_2']}
 summary=dict(status='PUBLIC_STOP_SHORT_CYCLE_VERIFIED_WEB_APPLIED_USER_CAPTURE_PENDING',at=at,server=http['server'],training_runs=0,old_pins_preserved=ad['prior_pins_preserved'],total_pins=len(ad['pins']),private_inputs_used=False,stop=dict(public_crop_cases=len(stop['rows']),before=sum(r['before']['first_stop']is not None for r in stop['rows']),after=sum(r['after']['first_stop']is not None for r in stop['rows']),synthetic_safety_controls=len(stop['controls']),all_safe=all(r['passed']for r in stop['controls'])),short_repetition=dict(public_original_videos=17,speed_conditions=speeds,controlled_incomplete_and_missing_cases=20,all_controls_passed=short['passed'],scope='Speed-altered existing observations; not new human recordings.'),normal_preserved=len(normal['rows']),capture_rows=[{k:r[k]for k in ['word','kind','auto_stop','overall_pass','frames','stop_to_result_seconds']}for r in cap['rows']],model_change='Frozen motion ensemble additionally examines 3-frame windows alongside original 7-frame observations. No new weights.',limitations=['Very fast/hidden endpoints may remain unknown','Public crop replay is not physical seated webcam confirmation','Stop dwell750ms and browser display are separate from calculation times','Not an accuracy estimate for all words or users'])
 write(D/'RESULT.json',summary)
 graph=dict(goal='정면 웹캠 사과·감사: 시범 → 촬영 → 자동 종료 → 교정/통과/관측 부족 → 재시도·퀘스트',status=summary['status'],previous=str(G/'selective_rtmw_anchor_repair_20261117/GRAPH.json'),nodes=[dict(id='R1',label='앉은 자세 자동 종료',status='public_verified',evidence='checks/STOP_CHECK.json'),dict(id='R2',label='짧은 반복의 실제 도달',status='public_verified_with_limits',evidence='checks/SHORT_CHECK.json'),dict(id='R3',label='정상·오류·속도 통합 및 웹',status='applied_verified',evidence='web/ADMISSION.json'),dict(id='R4',label='실제 촬영·교정 재시도·퀘스트',status='software_checked_user_capture_pending',evidence='checks/DEMO_CHECK.json')],edges=[['R1','R3'],['R2','R3'],['R3','R4'],['R4','public_reproduction_if_new_issue']],user_needed=['평소 앉은 자세에서 사과·감사 자동 종료와 정상 결과 확인','이후 명확한 손모양 오류 안내와 교정 재시도·퀘스트 완주 확인'],deferred=['새 단어 확대','LLM','Unity','물리적 HDD 분리 확인'])
 write(D/'GRAPH.json',graph)
 text=f'''자동 종료·짧은 반복 보완 — {at}

1. 이전과 달라진 점
손이 화면 아래로 내려갔는데도 검출기가 남기는 작은 손 점무리 때문에 종료가 막히는 문제를 보완했다. 손 미검출만으로 종료하지 않는다. 시작 때 확인한 양팔 대기 자세로 돌아오고 기존 0.75초 유지 조건을 만족해야 한다. 고정 가로선으로 사용자 영상을 잘라내거나 피드백 관절값을 삭제하지 않았다.
빠른 왕복을 긴 분석 구간이 하나로 섞는 현상을 공개 자료에서 확인했다. 기존 움직임 신경망 가중치는 그대로 두고 7프레임 분석에 3프레임 분석을 더했다. 앞 동작 뒤에 실제로 멀어지는 움직임과 다시 도달한 관측이 있어야 새 반복으로 센다. 예상 횟수만큼 채우거나 DTW로 없는 행동을 만들지 않는다.

2. 확인한 결과
- 공개 정면 6영상에서 만든 상반신 화면 16조건: 준비·종료 가능 7 → 16. 이 가운데 4조건은 영상 비율도 유지했다. 실제 앉은 사용자의 신규 촬영 결과가 아니다.
- 별도 한손/양손 교차·짧은 가림·낮은 손 유지·관측 전체 누락·지연 등 안전 제어15개 통과. 진행 중 조기 종료 없음.
- 공개 감사17영상의 시간 간격을 줄인 시험: 1.25배 두 반복 확인11→14, 1.5배3→14, 2배0→6. 각각17조건이며 모두 같은 공개 영상에서 파생한 시험이다. 2배 등 매우 빠른 경우에는 남은 한계가 있다.
- 한 번만 하기·자세만 유지·두 번째 접근에서 멈춤·관절 누락20조건은 두 번 완료로 잘못 올리지 않았다.
- 기존 사과·감사32정상 영상의 세부 판정 및 TCN 상위3후보 모두 유지. 손모양 차이·관측 부족·부분 수행 카드와 퀘스트 제어 통과.
- 공개 영상의 실제 촬영 코드 재생4회 모두 자동 종료. 정상은 종료 후 계산 약0.25~0.28초, 필요한 부분 재검출 포함 약0.67초. 종료 판정 대기0.75초 및 브라우저 표시 시간은 별도다. 실제 사용자 대기시간 측정은 아직 안 했다.

3. 웹 적용과 사용자 확인
http://127.0.0.1:8821/ 에 적용, 실제 서버PID {pid}. 카메라 자동 시작 없음. 같은 주소를 새로고침하면 된다. 실제 HTTP 공개 재생2개와 이전 화면 스크립트 보존 확인. 종료 보류 이유와 시작 기준을 auto_end_diagnostic.json으로 저장한다.
R1/R2/R3 완료. R4의 프로그램 퀘스트·로컬 자산 점검은 완료했으나, 평소 앉은 실제 촬영에서 자동 종료 및 교정 재시도·퀘스트를 완주하는 확인은 남았다. 사용자의 확인이 필요한 단계는 이 부분이다. 공개 시험 통과를 실제 촬영 완료로 표시하지 않았다.

4. 보존·검증 범위
C·T3-F, 100단어 본체 및 기존 모델 가중치 유지. 기존95개 핀 전부 동일, 현재116개 핀 확인. 추가 학습0회. 사용자 촬영 읽기·학습·튜닝·후보 선택0. 모든 신규 입력은 공개 정면 자료;5방향은 아바타 전용 유지. 코드·별도 실행 검사는 같은 Codex가 수행했고 독립 수어 전문가 검증은 아니다.
앞으로 새 실패가 확인되면 사용자 촬영은 진단만 하고 공개 정면 자료에서 재현해 공통 원인을 고친다. 지금은 필요한 실제 촬영 확인을 기다리며 검증된 화면을 유지한다.
'''
 (D/'RESULT_KO.txt').write_text(text,encoding='utf-8')
 # A compact diagram for continuity with the approved work graph.
 from PIL import Image,ImageDraw,ImageFont
 im=Image.new('RGB',(1600,390),'white');draw=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
 draw.text((48,26),'자동 종료 · 짧은 반복 보완 현황',font=font(34),fill='#152b44')
 labels=[('R1  자동 종료','공개 조건 검증 완료','손 내림과 팔의 대기 자세'),('R2  짧은 반복','공개 조건 검증 완료','짧은 왕복 + 실제 도달'),('R3  통합 검증','웹 적용 완료','정상·오류·속도 유지'),('R4  실제 촬영','사용자 확인 필요','앉아서 촬영 → 재시도')]
 for i,(title,status,detail)in enumerate(labels):
  x=48+i*390;fill='#183750' if i<3 else '#f0f3f6';ink='white'if i<3 else '#183750'
  draw.rounded_rectangle((x,105,x+340,290),18,fill=fill,outline='#183750',width=2)
  draw.text((x+20,123),title,font=font(26),fill=ink);draw.text((x+20,175),status,font=font(24),fill=ink);draw.text((x+20,233),detail,font=font(19),fill=ink)
  if i<3:draw.line((x+349,200,x+378,200),fill='#183750',width=3);draw.polygon([(x+382,200),(x+373,194),(x+373,206)],fill='#183750')
 draw.text((48,328),'실제 웹캠 확인 전까지 전체 완료로 표시하지 않습니다.',font=font(24),fill='#475669');im.save(D/'GRAPH.png')
 next_action='새 보완을 실제 앉은 웹캠에서 사과·감사 정상 촬영으로 확인. 자동 종료·판정·화면 대기를 진단하고 이후 손모양 오류→교정 재시도·퀘스트 완주. 사용자 자료는 진단만, 실패 보완은 공개 정면 재현 후 진행.'
 for name in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md']:
  dst=D/'before'/name
  if not dst.exists():shutil.copy2(ROOT/name,dst)
 statep=O/'STATE.json';dst=D/'before/STATE.json'
 if not dst.exists():shutil.copy2(statep,dst)
 state=read(statep);state['current_stage']=summary['status'];state['next_single_action']=next_action;state['updated_at']=at
 work=state['capture_feedback_completion_20261116'];work.update(status=summary['status'],updated_at=at,next_single_action=next_action,current_graph=str(D/'GRAPH.json'),current_work_graph=str(D/'GRAPH.json'),current_checkpoint=str(D/'RESULT.json'),active_feedback_runtime=str(D/'serve.py'),runtime_pid=pid)
 work['seated_stop_short_cycle_20261117']=dict(report=str(D/'RESULT.json'),text=str(D/'RESULT_KO.txt'),admission=str(D/'web/ADMISSION.json'),graph=str(D/'GRAPH.json'),summary=summary['status'],training_runs=0,user_data_used=False,production_changed=True,actual_webcam_pending=True)
 write(statep,state)
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';content=p.read_text('utf-8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
 content=content.replace(h8,h8+f'\n\n### 2026-11-17 자동 종료·짧은 반복 공개 보완 및 웹 적용\n공개 정면 상반신16조건 준비/종료7→16, 안전제어15통과. 감사17영상 파생속도시험1.25배11→14/1.5배3→14/2배0→6;부분·누락20제어오승격없음. 기존32정상·TCN보존. 새 가중치 학습 없이 기존 움직임 판정부7+3프레임 근거 연결 및 화면하단 손 추정점 종료 오판 보완. 양팔 대기 복귀750ms유지. 공개 실제촬영코드4회자동종료, 정상계산0.25~0.28초/재관측0.67초(종료대기·브라우저제외). 사용자 자료사용0/C·T3-F/95기존핀유지. 새웹 {D/"serve.py"},8821 PID{pid};HTTP공개2통과,카메라꺼짐. R1/R2/R3완료,R4실제앉은촬영/교정재시도/퀘스트확인남음. 보고{D/"RESULT_KO.txt"},그래프{D/"GRAPH.json"}. 매우빠름·가림은여전히보류가능,독립전문가검증아님.\n',1)
 content=content.replace(h9,h9+f'\n\n### 2026-11-17 보완 웹 실제 촬영 확인\n{next_action}\n현재서버 http://127.0.0.1:8821/ · R4사용자확인전전체완료아님. 학습/운영모델재선택불필요.\n',1);p.write_text(content,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {at}
목표 정면 웹캠 사과·감사 시범→촬영→자동종료→빠른 교정/통과/관측부족→재시도·퀘스트.5방향은아바타전용.
사용자승인 1~4단계 수행. R1자동종료/R2짧은반복/R3통합공개검증·웹적용 완료. R4소프트웨어퀘스트9제어/로컬GLB확인 완료, 실제앉은촬영 및 교정재시도·퀘스트완주는 사용자 확인 대기. 작업을 중단하라는 요청이 아님. 실제촬영필요 안내질문 발송.
근거 {D/'RESULT_KO.txt'} / RESULT.json / GRAPH.json / web/ADMISSION.json. 그래프그림 GRAPH.png.
공개 정면6영상 상반신16조건 준비·종료7→16; 한손/양손가림·교차·저위손유지등15안전제어통과. 감사17영상 속도파생1.25배11→14/1.5배3→14/2배0→6. 부분·정지·미관측20제어오승격없음. 실제 사람의 새 오류촬영 정확도 아님. 기존32정상·TCN후보일치. 매우빠름/숨은도달은남은한계.
변경: 화면하단 잘린/축소된 손 추정점은 정상 손높음 근거에서 구분하되 양팔 시작대기복귀750ms유지. 관절 피드백입력 자르지않음. 기존 움직임 신경망7프레임에3프레임창을 추가,이전도달후 실제분리+접근+관측도달일때만 새반복추가. 사후코드검토에서단순구간비중첩만으로충분하지않음을확인해 분리관측필수보완. 새가중치학습0.
공개촬영경로4회자동종료. 정상 종료후계산0.25~0.28초, 필요재관측14호출포함0.67초. 브라우저ACK/실제손내림대기 별도. auto_end_diagnostic.json에 다음촬영시작기준·종료보류이유저장.
실행 START_WEB_DEMO.cmd→frontal_demo_launcher.py→{D/'serve.py'},8821실제PID{pid},일반Windows환경,카메라꺼짐. selective_retry_v3유지 + stop_short_cycle_version=seated_rest_dual_window_v1. HTTP2공개통과. PID재사용추정금지,재개시확인.
다음 {next_action}
개인 촬영 읽기/학습/튜닝/후보선택0. 기존C·T3-F/100단어본체/가중치/95핀유지(총116핀). 총기존학습75변경없음. 독립전문가검증아님. 모델확대/LLM/Unity는미룸. 실행자산C존재,물리HDD분리촬영미검증.
''',encoding='utf-8')
 for p in D.glob('*.py'):ast.parse(p.read_text('utf-8'))
 assert read(statep)['current_stage']==summary['status']
 assert all(sha(p)==h for p,h in ad['pins'].items())
 write(D/'CLOSEOUT_CHECK.json',dict(passed=True,at=at,pins=len(ad['pins']),prior_pins=95,model_training_runs=0,private_data_read=False,docs_synchronized=True,physical_camera=False,remaining_user_confirmation=True))
 print('Recorded',summary['status'],'PID',pid)
if __name__=='__main__':main()
