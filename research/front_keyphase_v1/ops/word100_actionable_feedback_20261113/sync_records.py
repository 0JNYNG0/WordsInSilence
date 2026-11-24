from pathlib import Path
from datetime import datetime
import hashlib,json,shutil

U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(U/'FINAL_VERIFIED.json')['passed']
now=datetime.now().astimezone().isoformat()
backup=U/'before_records';backup.mkdir(exist_ok=True)
gfile=O/'word100_web_completion_graph_20261111/TASK_GRAPH.json'
files=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json',gfile]
for p in files:
    dest=backup/p.name
    if not dest.exists():shutil.copy2(p,dest)

summary='기존4단어 행동 교정 전달 우선. 관측 차이를 시각·손·짧은 행동 지시로 묶고 중복 제거/짧은 재생을8821에 연결. 저장93조건·정상60새경고0·합성오류9조건12문구·계약207/화면6/최종43검사 통과. 정확도 개선 주장 아님. 물리 이동 반전불변18학습은 반전불일치6→0이나 짧은 실제 이동1원천항목 신규손실로 미채택. G2 REWORK 유지, 새3단어/연출보다 기존4피드백 우선.'
next_task='G2 짧은 실제 이동과 유지의 관측정보 손실·설명구간 오연결을 원천항목 단위로 분리하고, 직접 대응 정답을 사용한 고정 후보의 정상 속도·교차·짧은 동작 및 방향/순서/누락 통합 보존을 검증한다. 평균 점수 임계값/같은 학습 재실행 금지. 통과한 판단만8821 교정 안내에 연결. 새3단어는 후순위; AIHub 안녕하세요 선택 유지. 실제 촬영은 사용자 시험이며 학습 금지.'
record=dict(status='COACHING_WEB_VERIFIED_MOTION_CANDIDATE_NOT_ADOPTED',updated_at=now,
    summary=summary,next=next_task,report=str(U/'RESULT_KO.txt'),graph=str(U/'GRAPH.json'),
    verification=str(U/'FINAL_VERIFIED.json'),server=read(O/'web_tutorial_live_integration_20261113/SERVER.json'),
    training_runs=18,training_running=False,private_used=False,model_replaced=False,
    full_common100_retraining=False,sealed_evaluation_opened=False,
    motion_candidate_adopted=False,correction_delivery_web_applied=True,
    whole_project_complete=False,author_and_verifier='same root Codex; separate saved-output checks; not independent KSL expert')
write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_BOUNDED_UNIT',running_training=False,rerun_completed_training=False,server=record['server'],next=next_task))
write(U/'GRAPH.json',dict(at=now,objective='기존4단어에서 근거 있는 행동 교정 안내 제공',
    stages=[dict(id='G0/G1',status='PASS',title='자료·설명·직접 정답 정리'),
    dict(id='G2',status='REWORK',title='행동 구간과 설명 연결',result='반전 불일치 개선, 짧은 움직임 신규 손실로 후보 미채택'),
    dict(id='G3',status='PENDING_INTEGRATED_VALIDATION',title='모양·방향·순서·누락 종합 판단'),
    dict(id='delivery',status='PASS_LIMITED_CONTRACT',title='확인한 차이→짧은 교정 문장→그 순간 재생',url='http://127.0.0.1:8821/',independent_accuracy_proven=False),
    dict(id='G4',status='PENDING',title='새 공개 입력 최종 평가'),
    dict(id='G5/G6',status='PARTIAL',title='웹 표시 연결 완료 / 실제 카메라·교정 정확도 확인 남음')],
    next=next_task))
graph=read(gfile)
graph.setdefault('parallel_delivery_updates',[]).append(dict(at=now,unit=U.name,verification=str(U/'FINAL_VERIFIED.json'),summary=summary,does_not_complete_G2_G3_G4=True))
graph['next_task']=next_task
assert next(n for n in graph['nodes'] if n['id']=='G2')['status']=='REWORK'
write(gfile,graph)
s=read(O/'STATE.json');s[U.name]=record;s['updated_at']=now
assert s['current_stage']=='word100_web_completion_graph_20261111'
s['current_followup']=U.name
w=s['web_tutorial_live_integration_20261113'];w['server']=record['server'];w['coaching_update']=str(U/'RESULT_KO.txt');w['next']=next_task;w['updated_at']=now
write(O/'STATE.json',s)
context=R/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8-sig')
title='### 2026-11-13 기존4단어 교정 문구·구간 재생 연결 / 물리 이동 후보 미채택'
assert title not in text
current=title+'\n\n'+summary+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `FINAL_VERIFIED.json`, `GRAPH.json`. 웹8821 PID'+str(record['server']['pid'])+'; 실제 웹캠 미개방. 기존 C·T3-F/사용자자료 보호. Codex 구현·별도검산이며 독립전문가 검증 아님.\n\n'
following=title+'\n\n'+next_task+'\n이번 고정18학습과 과거6+18학습 완료결과 재실행 금지. 전체 그래프 G2 REWORK 유지. 현재 전달 기능만 웹 반영했으며 모델 교체 없음.\n\n'
for heading,entry in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',current),('## 9. 다음 단일 작업',following)]:
    marker=heading+'\n\n';assert marker in text;text=text.replace(marker,marker+entry,1)
context.write_text(text,encoding='utf-8')
handoff=f'''# 최신 인수인계 — 2026-11-13 기존4단어 행동 교정 전달 우선

사용자 마감 촉박. 새 단어/연출보다 기존4단어의 행동 교정 문구를 먼저 완성하는 범위로 진행.
{summary}

본선 current_stage=word100_web_completion_graph_20261111 유지. G0/G1 PASS, G2 REWORK, G3/G4 통합 미완료. 전달 부품의 통과를 연구 전체 통과로 올리지 않는다.
현재 단위: {U}
보고/검증: RESULT_KO.txt, GRAPH.json, FINAL_VERIFIED.json. root Codex 작성·별도 계약/산술 검산이며 독립 한국수어 전문가 정답 아님.

웹: http://127.0.0.1:8821/ (현재 PID {record['server']['pid']}, 재개 전 실제 포트/PID 확인)
서버: web_tutorial_live_integration_20261113/server.py. python=word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. -B -u 절대 server.py 경로로 숨김 실행. 현재 카메라 꺼짐. 사용자 촬영 중이면 서버를 재시작하지 않는다.
웹 수정: coaching.py의 근거 제한 문구를 server.py 결과에 추가, feedback.js의 실제 시간 이동·짧은 구간 재생, live.js의 MOA 요약. 원래 점수/판단 카드/모델/DTW 미변경. 기존8820/8814 보존.
검증: 저장93조건(12공개영상 변형, 독립93영상 아님), 정상60새교정0, 합성오류9조건12문구, 계약207, Node 화면계약6, 최종43. 공개 저장 정상/불확실 브라우저 표시·재생 확인. 공개 관절 실제 추론연결1회, 퀘스트 미진행. 실제 웹캠/실제 오류 교정 정확도 미검증. BROWSER_VERIFIED/WEB_COACHING_*.png 참조.

AI 후보: motion_invariant.py 고정6기록제외×3seed=18학습 완료. MOTION_FROZEN/PRETRAIN/TRAINED/COMPARISON/SAVED_VERIFIED 보존. 시간반전 불일치6조건→0이나 짧은 실제 이동1원천항목 신규손실로 미채택. 모델교체 없음. 전체100단어 재학습 아님. 동일 학습 재실행·개발점수로 임계값 조정 금지. 이전 word100_feedback_recovery_20261113의 shape6+motion18완료도 재실행 금지.
새손실: WORD2523 REAL03 held_posture_review82:R. 기존 공통손실: WORD0739 REAL02 main_1:76:R / WORD1133 REAL06 main_0:48:R. 관측의 물리이동 존재와 의미적 행동·완료·설명 연결을 혼동하지 않는다.

다음 한 작업: {next_task}
다음 후보 설계 전 위 원천항목과 직접 정답의 정보 부족/연결 문제부터 확인한다. 봉인 평가 미개봉 유지. 임의 오답 주석 금지.

지원 상세 피드백4: 조상1567/이집트2523/순종0739/잘하다1133. 모든 손가락·접촉·반복·누락을 검증한 상태는 아니다. 특히 엄지와 손끝 모으기·완료의 의미 판단/자동구간 연결은 남아 있다.
추가 목표 보존: AIHub 안녕하세요 SEN0354, 과일사과0099, 감사1290. 숫자5 제외. 안녕하세요와 NIKL 인사 임의병합 금지. 새3단어는 현재 우선순위 뒤. 아바타는 조상1단어 시험이며 다른단어 아바타 동작 미지원.
보호: C·T3-F/공통100/원본 읽기전용, 사용자·멘토 촬영 학습·튜닝·선택 금지. 새 다운로드·외부전송·Git 없음. 물리 카메라는 사용자 직접 시험만.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
assert all(read(O/'STATE.json')[U.name][k]==record[k] for k in ['status','model_replaced','training_runs'])
assert title in context.read_text('utf-8')
write(U/'RECORDS_SYNCED.json',dict(passed=True,at=now,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},G2='REWORK',whole_project_complete=False))
print(json.dumps(dict(synced=True,unit=U.name,server_pid=record['server']['pid']),ensure_ascii=False))
