from pathlib import Path
import json,hashlib,datetime
A=Path(__file__).resolve().parent;U=A.parent;O=U.parent;R=U.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
pins=read(A/'before/EXECUTION_FROZEN.json')['pins'];changed=[p for p,h in pins.items() if sha(p)!=h]
assert changed==[str((U/'demo.js').resolve())],changed
assert all(sha(p)==h for p,h in read(U/'EXECUTION_FROZEN.json')['pins'].items())
b=read(A/'BROWSER_VERIFIED.json');p=read(A/'PUBLIC_FLOW_VERIFIED.json');ui=read(A/'UI_LOGIC_VERIFIED.json');http=read(A/'HTTP_VERIFIED.json')
assert all(x['status']=='PASS' for x in [b,p,ui]) and http['ready'] and http['served_js_exact']
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
summary=('8812 촬영 제어만 보완했다. 별도 실제 브라우저/가상카메라에서 관측 요청이 응답하지 않으면 previewBusy가 남아 다음 촬영도 막히는 문제를 재현했다. '
'demo.js에 촬영종료/재촬영/페이지종료 시 요청취소, 2.5초 요청제한, 이전 요청이 새 요청 상태를 지우지 못하게 하는 소유권 확인을 추가했다. 영상준비가 안 된 시간은 손 미검출 대기로 누적하지 않는다. '
'손 관측 후 양손 미검출0.75초의 공통 종료규칙과3초 카운트다운은 유지했다. 팔펴짐/신체위치 강제조건·단어예외·신경망 변경 없음. '
'공개4표현 ID01만 같은75%높이 상반신 viewport와마지막1.6초정지로모사하여 현재 /api/preview224회실행:4개모두저장주요구간후종료,마지막손관측후1초. '
'검출결과에측정된공백응답을200/400/600ms삽입한12대조에서도종료시점동일. 이들은원본전체화면피드백평가/자연가림정답이아니다. '
'가상카메라실브라우저39검사:한손/양손/600ms일시미검출/한손남음/처음손없음/800ms늦은응답/응답중단후재촬영/자동분석연결검증. 한손·양손은가상손내림약0.98초후종료했고느린응답은더늦었다. '
'실제JS모의UI44검사로기존로딩/관절/성공·실패·재촬영보존. HTTP로최종JS exact와ready확인. 실제카메라/사용자자료열람0, TCN·SignRep새분석0, 학습0. '
'기존580핀중demo.js1개만변경,나머지579보존. 서버재시작없음. AO연구일시중단유지.')
limit=('0.75초 이상 두 손이 검출되지 않는 지속 가림은 여전히 종료와 구별할 수 없다. 이번에 확인한 것은200~600ms통제미검출과공개ID01고정viewport/가상카메라흐름이며,실제사용자웹캠모든조건의성공을입증한것은아니다. '
'손내림확인은촬영조작이며수어의의미적완료판정이아니다. 관측지연이큰경우종료까지더걸릴수있다.')
nxt=('같은 http://127.0.0.1:8812/ 를 새로고침해 사용자 추론 전용으로 한손·양손 손내림 자동종료를 재확인한다. '
'미재개 AO학습/실험은계속중단하며이미완료한공개/가상검사는새문제없으면반복하지않는다. '
'지속가림을종료자세와구분하는추가근거는아직미구현이며화면밖팔각도를필수조건으로넣지않는다. 사용자·멘토영상학습/튜닝/선택금지.')
report=dict(status='COMPLETE_CAPTURE_FLOW_V5',at=now,summary=summary,limits=limit,next=nxt,
 changed_files=changed,preserved_pin_count=579,browser=b,public_replay=p,ui=ui,http=http,
 first_failure='BEFORE_BROWSER_REPRODUCTION.json with before/demo.js: nonreturning preview blocks retake.',
 training=0,new_feedback_inference=0,preview_detector_requests=224,hardware_camera=0,user_material_access=0,
 verification='Same Codex source implementation, synthetic camera browser tests and saved public observation replay; not independent KSL/human correctness ground truth.')
write(A/'RESULT.json',report)
(A/'RESULT_KO.txt').write_text(summary+'\n\n남은 한계\n'+limit+'\n\n다음\n'+nxt,encoding='utf-8')
sp=O/'STATE.json';state=read(sp);assert state['current_stage']=='PAUSED_BY_USER_AFTER_MOTION_ORDER_INPUT_PREPARATION'
state['word100_integrated_test_20261104']['capture_flow_v5']={k:report[k] for k in ['status','at','summary','limits','next']};state['next_single_action']=nxt;write(sp,state)
cp=R/'WordsInSilence_PROJECT_CONTEXT.md';text=cp.read_text(encoding='utf-8');title='### 2026-11-04 촬영 흐름 실제 브라우저 검증·요청 중단 복구'
assert title not in text
for section,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+'\n\n'+limit),('## 9. 다음 단일 작업',nxt)]:text=text.replace(section,section+'\n\n'+title+'\n\n'+body,1)
cp.write_text(text,encoding='utf-8')
hp=R/'NEW_CHAT_HANDOFF.md';(A/'PREVIOUS_HANDOFF.md').write_text(hp.read_text(encoding='utf-8'),encoding='utf-8')
hp.write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n'+limit+'\n\n'+nxt+
'\n\n현재 유지보수: word100_integrated_test_20261104/capture_flow_v5. 실제카메라차단한별도헤드리스브라우저는검사후종료. 8812기존서버유지. 기존 capture_autostop_v3는종료규칙복원, v4는읽기전용검토,이번v5는요청중단복구및실제브라우저/공개검출검증이다.\n'+
'SignRep/DTW:9개FIT ID로학습한흐름과DTW1회,ID01유래시간축. 모든속도/반복/정확한순간별교정은미검증. 학습때DTW약한구간이전과원시간보존.\n'+
'AO word100_motion_aware_order_20261103: FIT72/DEV76입력준비,exact재사용28/pendingAH120,최종별도입력검토·실제preflight·추가학습/평가미완료. 사용자재개전중단유지. C·T3-F·SignRep·DTW·원본동결보존.\n',encoding='utf-8')
assert read(sp)['current_stage']=='PAUSED_BY_USER_AFTER_MOTION_ORDER_INPUT_PREPARATION'
print('Complete: public replay16, browser39, UI44; 579 pins preserved; research paused')
