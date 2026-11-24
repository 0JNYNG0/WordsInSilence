from pathlib import Path
import json,datetime,shutil
R=Path.cwd();O=R/'research/front_keyphase_v1/ops';D=O/'capture_feedback_completion_20261116/moa_coaching_room_20261117'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
(D/'before_current_docs').mkdir(exist_ok=True)
for p in paths:
 dst=D/'before_current_docs'/p.name
 if not dst.exists():shutil.copy2(p,dst)
load=lambda p:json.loads((D/p).read_text(encoding='utf-8-sig'))
role=load('checks/ROLE_CHECK.json');truth=load('checks/DIRECT_TRUTH_CHECK.json');cap=load('web/CAPTURE_CHECK.json')
result=dict(at=now,status='PUBLIC_ENDPOINT_CANDIDATE_VERIFIED_NOT_DEPLOYED_UI_FLIGHT_REVIEW_PENDING',implementation='Codex local',verification='Same Codex separate public regression and camera-mocked browser checks; not independent expert review',role_cases=len(role['rows']),improved=role['improved'],truth_frames=len(truth['rows']),before_correct=truth['before_correct'],after_correct=truth['after_correct'],partial_controls=len(load('checks/REVIEWED_PARTIAL_CHECK.json')['rows']),normal_cases=len(load('web/CONSUMER_CHECK.json')['rows']),card_cases=len(load('web/CARD_CHECK.json')['rows']),public_capture_cases=len(cap['rows']),training_runs=0,private_data_read=False,weights_changed=False,production_ui_changed=False,production_backend_changed=False,storyboard=str(D/'qa/MOA_FLIGHT_TRAIL_STORYBOARD.png'),animation=str(D/'qa/MOA_FLIGHT_TRAIL_PREVIEW.gif'),pending=['User approval of revised downward/forward/upward smoke-trail UI','Complete endpoint admission/hash preservation and actual HTTP integration','Validate retained recording endpoint before reopening preview','Deploy only admitted backend; UI only after explicit user approval'],active_server='http://127.0.0.1:8821/',active_entrypoint=str(D.parent/'rest_stability_endpoint_20261117/serve.py'),preview_server='http://127.0.0.1:8822/',scope='Public frontal observations and derived tempo/partial cases, not new-user accuracy. Five views avatar-only.')
(D/'CHECKPOINT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
summary='기존8821은 자동종료 흔들림 보완 버전 그대로 유지. 별도 moa_coaching_room_20261117에서 관측 역할 분리 후보의 공개88조건 검증(속도변형4조건 개선), 직접149프레임143정답 유지, 부분16/정상32/카드7/공개캡처4 통과. 가중치 추가학습0·사용자자료 접근0. 아직 admission/실제HTTP통합 전으로 미적용. 새 화면은 분리된8822에서 준비; 사용자 기존 안개 시안 반려 후 MOA 고개숙임→아래/앞 타원급상승→아랫부분 연기꼬리 급확장 시안을 실제 렌더해 이미지/GIF 제시. 기존8821 UI 적용은 명시적 승인 대기.'
next_action='수정 비행 시안의 사용자 답변을 반영한다. 그동안 판정 후보의 동결해시/admission과 녹화 보존 HTTP 검증을 마무리한다. 공개 검증을 통과한 판정 보완만 연결하고, 새 UI·전환은 사용자 승인 이후 연결한다. 촬영은 사용자만 시작, 개인 자료는 학습·튜닝·선택 금지.'
(D/'RESULT_KO.txt').write_text(summary+'\n'+next_action+'\n근거: CHECKPOINT.json, checks/ROLE_CHECK.json, checks/DIRECT_TRUTH_CHECK.json, checks/REVIEWED_PARTIAL_CHECK.json, web/CONSUMER_CHECK.json, web/CARD_CHECK.json, web/CAPTURE_CHECK.json, qa/FLIGHT_CAPTURE.json.\n판정 개선은 기존 학습된 관측과 코드 연결 보완이며 새 신경망 학습이 아니다. 합성 속도/누락 시험은 자연 오류·새사용자 정확도를 의미하지 않는다.\n',encoding='utf-8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig')
for n,heading,body in [(8,'2026-11-17 MOA 비행 시안 수정·공개 판정 후보 검증, 아직 미적용',summary),(9,'2026-11-17 시안 승인 대기와 판정 통합 검증',next_action)]:
 lines=t.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if l.startswith(f'## {n}.'));lines.insert(i+1,f'\n### {heading}\n{body}\n근거 {D / "CHECKPOINT.json"}.\n\n');t=''.join(lines)
p.write_text(t,encoding='utf-8')
s=json.loads((O/'STATE.json').read_text(encoding='utf-8-sig'));s['current_stage']=result['status'];s['next_single_action']=next_action
k=s['capture_feedback_completion_20261116'];k['status']=result['status'];k['updated_at']=now;k['next_single_action']=next_action;k['moa_coaching_room_20261117']=result
(O/'STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'# 최신 인수인계 — {now}\n목표: 정면 웹캠 사과·감사 시범→촬영→자동종료→교정/관측부족→재시도·퀘스트. 5방향은 아바타 전용.\n{summary}\n\n현재8821 엔트리: {result["active_entrypoint"]}. 이전 PID3212는 재개 시 실제 확인 필요. 별도8822 시안 서버는 D/SERVER_8822.json 확인. 시안은 실제 카메라 없이 모의 촬영 API로 점검했다.\n다음 한 작업: {next_action}\n현재 작업 폴더 D={D}\n구현된 새 후보: probe_endpoint_roles.py, role_runtime.py, thanks_runtime/worker.py. 공개 검증은 완료했으나 D/web/ADMISSION.json 및 serve.py의 RoleCapture 연결은 아직 없다.\n새 화면: site/scene.js, hologram.js, companion.css, renderer.js, live.js, index.html. 촬영 종료 후 실시간 미리보기 유지 및 별도 결과 창 때문에 session-recording 보존 경로 검증 필요. 기존 사용자 녹화 읽기 없음.\n사용자 최신 지시: 비행기처럼 살짝 고개를 숙이고 아래→앞→위로 타원 반곡선을 그려 화면 밖으로 빠져나갈 것. 아랫부분 연기가 궤적에 남다가 빠르게 커질 것. 시안 승인 전 기존8821 UI 적용 금지. 전환 시작 시 대화/메뉴 숨김. 4컷 그림과 GIF 생성완료.\n시안: {result["storyboard"]}\n움직임: {result["animation"]}\n근거: D/CHECKPOINT.json, D/RESULT_KO.txt와 checks/, web/, qa/. 구현 및 별도 검산 담당은 같은 Codex, 독립 전문가 검증 아님. 실제 사용자 시연 미완료.\n',encoding='utf-8')
print(json.dumps(dict(recorded=True,role_cases=result['role_cases'],normal_cases=result['normal_cases'],pending_ui_approval=True),ensure_ascii=False))
