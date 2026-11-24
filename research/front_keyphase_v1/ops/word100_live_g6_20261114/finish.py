"""Record public G6 integration. Physical camera gate remains explicitly pending."""
from pathlib import Path
import json,datetime,shutil,hashlib,sys
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111';W=O/'web_tutorial_live_integration_20261113'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
server=read(W/'SERVER.json');deploy=read(U/'DEPLOYED_STATUS.json')
assert deploy['state']=='ready' and not deploy['camera_thread_active']
browser=dict(passed=True,scope='CUA observed local UI, public fixtures only; no physical webcam',checks=dict(countdown_visible=True,one_and_two_hand_auto_stop=True,top3_horizontal=True,loading_overlay_visible=True,latest_correction_in_original_time=True,gray_uncertainty_distinct=True,partial_not_complete=True,missing_not_wrong=True,normal_similar=True,both_videos_ready=True,no_browser_console_errors_observed=True),screenshots=['BROWSER_DIFFERENCE.jpg','BROWSER_UNCERTAIN.jpg','BROWSER_CAPTURE.jpg','BROWSER_TWOHAND_RESULT.jpg'],actual_capture_normal_limit='WORD1133 REAL17 recaptured at 640x360 with automatic crop/time sampling became cross-seed uncertain. It received no false correction; it is not a correct-feature recognition success.')
write(U/'BROWSER_VERIFIED.json',browser)
checks={n:read(U/(n+'.json'))['passed'] for n in ['EQUIVALENCE','LIVE_INPUTS','LIVE_CONTROLS','AUTO_END_CONTRACT','PUBLIC_CAPTURE','HTTP_VERIFIED','SAMPLING_REVIEW']}
checks['protected_477']=read(U/'PROTECTION_CHECK.json')['unchanged'];checks['browser']=browser['passed']
timings=[]
for name,word,label in [('ONEHAND','WORD0739','순종'),('TWOHAND','WORD1133','잘하다')]:
 a=read(U/f'PUBLIC_BROWSER_{name}.json');assert a['public_fixture'] and a['stop_reason']=='arms_returned' and a['recorded_frames']==a['observed_frames'];timings.append(dict(word=word,label=label,**a))
repairs=[
 dict(problem='Browser/WebGL workload exposed a queue of stale RTMW observations; automatic stop failed and the queue eventually filled.',evidence='BROWSER_QUEUE_FAILURE.json',change='Pause 3D drawing during capture/feedback, then bound acquisition to two outstanding observations; keep all acquired frames and true timestamps.',scope='Common camera loop; no word branches, no model/threshold changes.'),
 dict(problem='3D pause alone did not fix automatic stop. Fixture forced a late manual stop; unnecessary tail greatly increased inference.',evidence='BROWSER_RENDER_PAUSE_ONLY.json',change='RTMW backpressure before acquisition, with visible error on sustained worker stall.',warning='20.64 to 2.87 seconds is not a controlled model speed comparison: capture duration, source scheduling and workload differed.'),
 dict(problem='At 24Hz one previously conflicting reverse case became agreed reverse across all three seeds.',evidence='SAMPLING_24HZ.json',change='Preserved original failed assertion and inspected three-seed evidence in SAMPLING_REVIEW.json; no tuning.',warning='Finite eight normal samples preserved decisions; no guarantee of invariance under arbitrary sampling gaps.'),
 dict(problem='Correction button initially sought 0.15s before its orange evidence.',change='Seek exact evidence start; highlight only its hands and spans; clamp focused playback end.'),
 dict(problem='Final preview diagnostic counts lagged queue drain.',change='Final status now exposes recorded/observed result counts after successful drain.')]
write(U/'FAILURES_AND_REPAIRS.json',repairs)
limits=[
 'Physical user-started camera, actual framing/occlusion and end-to-end waiting time are pending; do not mark G6 or DONE pass.',
 'Frozen G4 same-input equivalence is not a new independent accuracy evaluation. These public sources are already exposed.',
 'Actual public recapture of normal 잘하다 is uncertain. Capture resolution, temporal sampling, quarantine and segmentation can affect decisions; source-component attribution remains open. Do not tune on user captures.',
 'No new model training. Existing G4 conflict/coverage limitations remain: thumb, fingertip gathering, contact, full repetition, full-sign correctness unsupported.',
 'Browser waiting time starts at automatic recording stop and ends at post-render acknowledgement. It includes processing/encoding/HTTP/poll/render return, not the 3-second countdown or model startup. Stable rest confirmation (~0.75s) precedes stop.',
 'Server model initialization took 7.03s on this start; preparation is once per server start. Public two-shot times are not an average or guaranteed physical camera performance.',
 'Camera acquisition may slow under load; all received frames are retained. Unobserved intervening moments are not reconstructed or certified. Large stalls abort rather than claim a judgement.'
]
report=dict(status='G6_PUBLIC_INTEGRATION_COMPLETE_PHYSICAL_ACCEPTANCE_PENDING',at=at,implementation='Codex',verification='Separate local checks by Codex; not independent human/educator review',new_training=False,user_data_training_or_tuning=False,checks=checks,public_timings=timings,protected_files=477,server=server,limits=limits,repairs=repairs)
write(U/'RESULTS_REVIEW.json',report)
write(U/'VERIFIED.json',dict(public_checks_passed=all(checks.values()),checks=checks,g6_complete=False,physical_camera='NEEDS_INPUT',baseline_preserved=True,at=at))
sys.path.insert(0,str(G));import graph
evidence=[str(U/x) for x in ['VERIFIED.json','RESULTS_REVIEW.json','BROWSER_VERIFIED.json','PUBLIC_BROWSER_ONEHAND.json','PUBLIC_BROWSER_TWOHAND.json','PROTECTION_CHECK.json']]
graph.record('G6','NEEDS_INPUT',dict(camera_user_started=False,capture_regression=True,end_to_end_latency=False,user_flow=False),evidence,'Public capture/browser path and latest evidence connected. Real webcam acceptance pending; no training or user tuning. Two public captures auto-stop, render acknowledgement 2.87/1.55s; one normal recapture remains uncertain.')
next_action='G6 마지막 실제 촬영 확인: 8821 연습에서 순종·잘하다를 사용자가 시작. 자동 종료·잠깐 가림·피드백 표시·종료→화면 시간만 확인. 사용자 영상은 학습/튜닝/후보 선택 금지. 아직 G6/DONE 미완료. 공개 완료 검증을 반복하지 않는다.'
title='G6 새 촬영 연결·공개 검증 완료 / 실제 웹캠 확인 대기'
summary='최신 G3/G4 통합판정을 새 촬영 RTMW 재사용 경로에 연결했고, G5 손·팔 강조/교정문장/회색 보류를 적용했다. G4 동일입력40조건 동등, 공개 원시8·대조4·자동종료 계약·브라우저 검증. 공개 브라우저 순종122/122·잘하다91/91프레임, 두 경우 자동 종료; 종료→화면 확인 2.87초/1.55초(물리 웹캠 아님). 화면 부하로 관측 대기열이 쌓여 자동 종료를 놓치는 문제를 발견, 3D그리기 잠시 정지+관측 미처리 최대2로 제한. 받은 프레임은 모두 보존하나 촬영 표본속도는 부하에 따라 변함. 정상 잘하다 재관측은 판단 충돌로 회색: 모델 정확도 개선이라 칭하지 않음. 첫 모델준비7.03초는 서버시작1회 별도. C·T3-F/공통100 등477파일 동일, 새학습/사용자학습튜닝 없음. G0-G5 제한범위 PASS, G6 NEEDS_INPUT, DONE PENDING.'
note=f'{summary}\n근거: `research/front_keyphase_v1/ops/{U.name}/RESULTS_REVIEW.json`, `VERIFIED.json`, `PROGRESS_GRAPH.png`.\n'
backup=U/'before/metadata';backup.mkdir(exist_ok=True)
for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json',G/'CURRENT_EXECUTION.json']:
 target=backup/p.name
 if not target.exists():shutil.copy2(p,target)
context=R/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text(encoding='utf-8-sig')
for section,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',note),('## 9. 다음 단일 작업',next_action+'\n새 촬영 판정이 회색이어도 오답·완료로 바꾸지 않는다. 실제 촬영 검산 전 전체완료금지. 웹8821 PID는 SERVER.json 실측 사용.\n')]:
 heading=f'### {at[:10]} {title}'
 if heading not in s[s.index(section):s.index(section)+300]:s=s.replace(section+'\n',section+'\n\n'+heading+'\n\n'+body,1)
context.write_text(s,encoding='utf-8')
handoff=f'# {title} — {at}\n\n{summary}\n\n다음 한 작업: {next_action}\n\n구현/검산 Codex. 독립 전문가 검토 아님. 현재단위 `{U}`. 결과 RESULTS_REVIEW.json, VERIFIED.json, BROWSER_VERIFIED.json, 실패이력 FAILURES_AND_REPAIRS.json. 웹 변경5개는 CONNECTION.json/before/rollback.py; 보호모델 변경 없음. server.py는 새 adapter를 사용하며 서버 준비 시1회 모델적재.\n\n서버 {server["url"]} PID{server["pid"]}, 카메라 시작안함. 공개fixture8822 종료. 학습·추론실험 종료. 사용자 시작 촬영 요청 pending. U/WEB_ACCEPTANCE.json은 실제 촬영의 최소 시간·개수 메타데이터이며, PUBLIC_BROWSER_*.json과 혼동 금지. 사용자 영상/관절은 판정 개선의 학습·튜닝·후보선택에 사용하지 않는다.\n\n남은 모델범위: 정상 재관측의 충돌, 반복전체/손끝/엄지/접촉/전체단어정오 미지원. 이 G6는 연결·대기열 개선이며 새 정확도 향상이 아니다. G6 완료는 실제 촬영 확인 이후.\n'
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
state=read(O/'STATE.json');state['current_followup']=U.name;state['next_single_action']=next_action
state[U.name]=dict(status=report['status'],path=str(U),result=str(U/'RESULTS_REVIEW.json'),verified=str(U/'VERIFIED.json'),graph=str(U/'PROGRESS_GRAPH.png'),server=server,training_running=False,private_access=False,next=next_action)
state[G.name].update(status=report['status'],at=at,active_unit=str(U),latest_evidence=str(U/'RESULTS_REVIEW.json'),progress=summary,remaining=next_action,resume_requires_user=True)
write(O/'STATE.json',state)
write(G/'CURRENT_EXECUTION.json',dict(status=report['status'],at=at,active_unit=str(U),running=False,server_running=True,server_pid=server['pid'],port=8821,camera_opened=False,next=next_action))
(U/'RESULT_KO.txt').write_text(title+'\n\n'+summary+'\n\n'+next_action+'\n\n주의: 공개2회 시험의 시간이며 실제 사용자 평균시간은 아님. 정상 잘하다 새 관측은 회색 보류. 숫자만 빠르게 만들기 위해 손모양·행동 판정조건을 낮추지 않았다.\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],checks=checks,server=server),ensure_ascii=False))
