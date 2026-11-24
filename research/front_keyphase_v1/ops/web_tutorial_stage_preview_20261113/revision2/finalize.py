from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, re

root=Path(__file__).resolve().parents[5]
unit=Path(__file__).resolve().parents[1]
out=unit/'revision2'
assert (root/'WordsInSilence_PROJECT_CONTEXT.md').exists(), root
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
status='UI_REVISION2_PREVIEW_VERIFIED_AI_NOT_CONNECTED'
site=unit/'site'
checks=[]
def check(name, value):
    checks.append({'name':name,'passed':bool(value)})
    assert value,name
def read(p): return p.read_text(encoding='utf-8-sig')
def dump(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
active=['index.html','style-v2.css','app-v2.js','hologram.js','quest.js','renderer.js']
content='\n'.join(read(site/p) for p in active)
check('active_entry_uses_revision2', 'app-v2.js' in read(site/'index.html') and 'style-v2.css' in read(site/'index.html'))
check('no_camera_or_analysis_calls', not re.search(r'getUserMedia|MediaRecorder|/api/|WebSocket',content))
check('no_remote_dependencies', not re.search(r'https?://',content))
check('preview_and_camera_disclosures', '실제 인식 판정은 아직 연결 전' in content and '카메라 연결 전' in content)
check('old_circle_mask_not_active', 'clip-path:circle' not in content.replace(' ','') and 'clipPath' not in read(site/'app-v2.js'))
check('eye_anchors_share_robot_transform', 'robotRoot[12+r]' in content)
for p in ['PRACTICE.jpg','QUEST_FIRST.jpg','QUEST_COMPLETE.jpg','EYE_PROJECTION.jpg','test_quest.mjs']:
    check('artifact_'+p,(out/p).exists() and (out/p).stat().st_size>0)
for p in ['app.js','index.html','style.css']:
    check('before_backup_'+p,(out/'before'/p).exists())
verified={
 'time':stamp,'status':status,'checks':checks,'passed':True,
 'implementation':'root Codex','verification':'same Codex: Node state tests, source checks and CUA visual inspection; no independent human validation',
 'node_tests':'PASS: ordered progression, wrong word rejection, duplicate pending click, invalid/stale token rejection, exact-once completion, reset',
 'syntax_check':'app-v2.js PASS after eye-anchor correction; hologram.js PASS before final UI pass',
 'browser_observed':['1280x720 practice with central Ch29/MOA and small bottom-center camera placeholder','eye light originates at both mesh eyes and opens projected grid space','practice exit returns to white','quest 인사→사과→5 green nodes and links','all-complete event once and disabled success button','quest reset returns to 인사','native narrow 355x638 practice layout','no console warning/error observed'],
 'not_tested':['real camera capture','TCN or feedback integration','gesture correctness for 인사/사과/5','runtime inference latency','real mobile hardware and cross-browser performance'],
 'corrections':['quest header/success text overlapped forehead: compacted and moved up','narrow practice controls covered chest: moved to side','beam source was above eyes: corrected model-local centers and shared root transform'],
 'scope':{'training_runs':0,'model_inference_calls':0,'camera_opened':False,'downloads':0,'baseline_changed':False,'operating_web_changed':False},
 'hashes':{p:hashlib.sha256((site/p).read_bytes()).hexdigest() for p in active},
 'server_note':'Existing localhost preview remains reachable in CUA. Earlier listener observed PID 286948. Final sandbox CIM query unavailable and sandbox netstat empty; not treated as server failure or re-started.'
}
dump(out/'VERIFIED.json',verified)
summary='사용자 수정 요청에 따라 로봇 눈빛 투사·가상 연습실·실전 퀘스트 화면을 보완했다. 원형 덮개를 양눈 광원/빛의 투사/격자 공간 전환으로 교체하고 복귀는 아래로 걷히는 스캔 연출로 분리했다. 연습은 중앙 Ch29+오른쪽 MOA+하단 중앙 작은 촬영 화면 자리로 변경했다. 실전은 인사→사과→5 순서의 원과 연결선이 초록색으로 채워지고 마지막에 완료 이벤트가 1회 발생한다. 성공은 명시된 연출 시험 버튼으로만 발생하며 실제 카메라·TCN·피드백 API는 미연결이다.'
next_action='수정된 8820 연출 미리보기의 사용자 확인을 받는다. 실제 촬영/TCN/피드백 연결은 별도 다음 구현이며 아직 완료로 취급하지 않는다. 인사·사과·5는 퀘스트 UI 예시로, 현재 모델 지원/교육용 아바타 동작 완료를 뜻하지 않는다. 피드백 본선은 word100_feedback_recovery_20261113/NEXT_PLAN_KO.txt의 G2 재개점을 보존하고 완료6+18학습은 재실행하지 않는다. 기존 C·T3-F/운영웹과 사용자자료 보호 유지.'
report=f'''웹 가상 연습실 수정 2차 — {stamp}

{summary}

바뀐 점
- 만나기: 기존 MOA·한글 도트 타이핑 유지.
- 전환: 두 눈 위치에서 빛이 나와 화면 앞쪽으로 퍼지고 격자 연습 공간으로 전환. 나올 때는 아래쪽으로 빛이 걷힘.
- 연습: 커다란 좌우 영상 칸 제거. 중앙 3D 캐릭터/옆 MOA/가운데 아래 작은 내 모습 자리.
- 실전: 인사 → 사과 → 5. 성공 원→연결선 왼쪽에서 오른쪽 채움→다음 원 강조→모두 완료 이벤트.
- 중복 성공, 순서 건너뛰기, 재시작 뒤 늦게 도착하는 이전 성공 신호 방지.

확인한 것
Node 퀘스트 상태 검사 통과. 실제 브라우저에서 전환/연습/복귀/실전 3단계/완료/다시보기 확인.
1280×720와 좁은 355×638 화면 확인. 마지막 관측 콘솔 경고·오류 없음.
실제 캡처: PRACTICE.jpg, EYE_PROJECTION.jpg, QUEST_FIRST.jpg, QUEST_COMPLETE.jpg.
작성과 검증은 같은 Codex가 수행. 독립 사람·한국수어 전문가 검수가 아님.

아직 연결하지 않은 것
카메라 창은 자리 표시다. 실제 촬영·인식·피드백 요청은 발생하지 않는다.
실전 성공은 '성공 연출 확인' 시험 버튼의 결과다. 단어를 실제로 판정한 결과가 아니다.
캐릭터 동작은 기존 공개 '조상' 적용 시험만 연습에서 재생. 인사/사과/5 예시로 잘못 재생하지 않는다.
새 학습·추론·다운로드·사용자 자료 접근 없음. 기존 C·T3-F와 운영 웹은 수정하지 않았다.

수정 이력/한계
중간 검수에서 퀘스트 문구/캐릭터 머리 겹침, 좁은 창의 가슴 위 조작 버튼 겹침, 눈보다 높은 광원 시작점을 발견해 보완했다.
처음 파일 교체 patch 형식 오류는 쓰기 없이 실패했고 별도 v2 파일 작성으로 복구했다.
최종 제한환경에서 CIM 포트 조회는 실패하고 netstat에는 포트가 보이지 않았다. CUA 실제 localhost 화면은 정상 확인했으며 서버 중복 실행은 하지 않았다.
실제 기기별 3D 성능과 AI 분석 속도는 이번 화면 검증 범위가 아니다.

다음
{next_action}
'''
(out/'RESULT_KO.txt').write_text(report,encoding='utf-8')

server=json.loads(read(unit/'SERVER.json'))
server.update(pid=286948,scope='static effects and quest UI preview only; no camera/inference',revision='revision2',url='http://127.0.0.1:8820/',browser_reachable_verified_at=stamp,pid_note='last observed listener earlier in this revision; sandbox final process recheck unavailable')
dump(unit/'SERVER.json',server)

state_path=root/'research/front_keyphase_v1/ops/STATE.json'
state=json.loads(read(state_path))
previous=state['web_tutorial_stage_preview_20261113'].copy()
dump(out/'STATE_BEFORE.json',previous)
state['web_tutorial_stage_preview_20261113'].update(status=status,updated_at=stamp,result=str(out/'RESULT_KO.txt'),verification=str(out/'VERIFIED.json'),server=server,ai_connected=False,camera_connected=False,training_resumed=False,revision=2,next=next_action,summary=summary,previous_status=previous['status'])
state['next_single_action']=next_action
state['last_updated']=stamp
dump(state_path,state)

context_path=root/'WordsInSilence_PROJECT_CONTEXT.md'
context=read(context_path)
heading='### 2026-11-13 웹 연출 수정2 — 눈빛 투사·캐릭터 중심 연습·실전 순서 UI'
evidence='근거: `research/front_keyphase_v1/ops/web_tutorial_stage_preview_20261113/revision2/RESULT_KO.txt`, `revision2/VERIFIED.json`, 실제 화면 캡처4장.'
for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+' Node 상태 검사·CUA 화면 검증 완료. 모델 성능 향상/교육용 완료 아님.'),('## 9. 다음 단일 작업',next_action)]:
    assert marker in context
    context=context.replace(marker,marker+'\n\n'+heading+'\n'+body+'\n'+evidence+'\n',1)
context_path.write_text(context,encoding='utf-8')
(root/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-13 웹 연출 수정2 완료

{summary}

사용자의 이번 직접 수정 요청이 이전 그림 승인 대기를 대체했다. 구현한 범위는 연출/화면/시험 퀘스트뿐이며 학습 재개나 실제 인식 연결을 하지 않았다.

로컬 미리보기: http://127.0.0.1:8820/
현재 단위: {unit}
결과/검증: revision2/RESULT_KO.txt, revision2/VERIFIED.json.
실제 스크린샷: revision2/PRACTICE.jpg, EYE_PROJECTION.jpg, QUEST_FIRST.jpg, QUEST_COMPLETE.jpg.
이전 STORYBOARD_PROPOSAL.png는 생성 콘셉트 그림으로 실제 화면 캡처가 아니다.

실행: 기존 정적 서버 재사용. SERVER.json의 PID286948은 이번 수정 중 앞서 관측한 값. 마지막 제한환경 CIM/포트 확인 불가였지만 CUA localhost 화면 정상 확인. 다음 재개시 실제 세션 확인하고 중복 실행 금지.
사이트 진입 index.html → app-v2.js/style-v2.css. hologram.js(눈빛), quest.js(순서/완료). revision2/before에 이전 파일 보존. 실제 MOA/Ch29/65뼈/8스킨가중치 자산 재사용, 외부 다운로드 없음.

검증: root Codex 작성, Node 상태/문법 검사와 CUA 실제 화면 확인. 별도 사람/수어 전문가 검증이 아니다. 중복/순서/초기화와 이벤트1회 검증, 1280×720·355×638 배치 확인. 실제 카메라·TCN·피드백·AI속도·교육용 동작 정확도 미검증.

다음: {next_action}

아바타 직전 결과: avatar_motion_smoothing_20261113/RESULT_KO.txt. 공개 조상 REAL02 5뷰 추정3D·65뼈 유지·손목 제한 보완, 교육용완성 아님. 정확 얼굴관계·큰손가락회전·접촉/충돌 미해결.
피드백 본선: word100_feedback_recovery_20261113. 손모양6+물리이동18 비교 검산 완료/미채택, G2보완→G3통합→G4평가 순서 보존. 공통100/기존 C·T3-F/운영/원본 보존, 사용자·멘토자료 학습·튜닝·후보선택 금지.
''',encoding='utf-8')

# Closeout consistency checks: do not launch model work for documentation.
saved=json.loads(read(state_path))
assert saved['web_tutorial_stage_preview_20261113']['status']==status
assert saved['current_stage']==state['current_stage']
assert heading in read(context_path) and next_action in read(root/'NEW_CHAT_HANDOFF.md')
assert Path(saved['web_tutorial_stage_preview_20261113']['verification']).exists()
print('PASS: revision2 source/artifact checks, report, STATE/context/handoff synchronization')
