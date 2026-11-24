from pathlib import Path
import json, hashlib, shutil, datetime, ast
D = Path(__file__).resolve().parent
ROOT = next(p for p in D.parents if (p / 'WordsInSilence_PROJECT_CONTEXT.md').exists())
O = D.parent.parent
read = lambda p: json.loads(Path(p).read_text('utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, q):
    Path(p).write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding='utf-8')

now = datetime.datetime.now().astimezone().isoformat()
admission = read(D / 'web/ADMISSION.json')
assert all(sha(p) == h for p, h in admission['pins'].items())
assert read(D / 'web/HTTP_CHECK.json')['passed']
assert admission['prior_pins_preserved'] == 116
capture = read(D / 'web/CAPTURE_CHECK.json')
server = read(D / 'SERVER.json')
stop = read(D / 'checks/STOP_CHECK.json')
summary = ('자동 종료 흔들림 보완만 웹 적용. 공개 정면 상반신 16조건 정상 종료 유지, '
           '공개 관측에 순간 손점 오판을 넣은 32조건 종료 0→32, 안전 대조18 통과. '
           '기존32영상 TCN·피드백 및 화면7검사 유지. 실제 촬영 코드를 공개영상으로 구동한4회 자동종료. '
           '종료 후 결과계산 정상0.37~0.40초, 선택적 재검출 포함0.99초; 대기·브라우저 표시 제외. '
           '끝 상태 보완 후보는88조건 개선0, 직접 확인한 분리 자세 기반 부분동작16조건 오통과 없으나 미채택. '
           '감사의 접근2후보 끝 미확인 문제는 미해결. 학습0, 사용자자료 접근0, 이전116핀·C·T3-F 유지.')
next_action = ('감사의 접근 움직임은 있으나 끝 상태가 확인되지 않는 원인을 공개 정면 자료에서 재현한다. '
               '정상 두 도달과 실제로 닿기 전에 멈춘 구간의 직접 정답을 구분하고, '
               '관절 관측 문제인지 도달 자세 분류 문제인지 먼저 확인한다. '
               '관측 문제면 중요한 끝 구간의 제한된 재검출을, 분류 문제면 해당 공개 정답 보강·학습을 각각 검증한다. '
               '근거 없는 완료 기준 완화는 금지. 새 자동 종료의 실제 촬영 확인은 사용자 가능 시 진단만 진행한다.')
graph = dict(goal='정면 웹캠 사과·감사 시범 → 촬영 → 자동 종료 → 교정/통과/관측 부족 → 재시도·퀘스트',
             status='REST_STABILITY_WEB_APPLIED_ENDPOINT_REWORK',
             previous=str(D.parent/'seated_stop_short_cycle_20261117/user_retest_1734/GRAPH.json'),
             nodes=[dict(id='R1', label='자동 종료 흔들림 보완', status='public_verified_web_applied_actual_recheck_pending', evidence=str(D/'checks/STOP_CHECK.json')),
                    dict(id='R2', label='반복 동작의 끝 확인', status='candidate_no_demonstrated_gain_not_admitted', evidence=str(D/'checks/ENDPOINT_CHECK.json')),
                    dict(id='R3', label='정상·오류·속도 및 웹', status='passed_for_admitted_stop_change', evidence=str(D/'web/HTTP_CHECK.json')),
                    dict(id='R4', label='실제 촬영·교정 재시도·퀘스트', status='not_completed')],
             edges=[['R1','R3'],['R2','R3'],['R3','R4']], next_single_action=next_action)
write(D/'GRAPH.json',graph)
result = dict(status=graph['status'], at=now, summary=summary, public_scope='Known public frontal clips and derived faults; not blind accuracy.',
              stop_normal_conditions=16, derived_jitter_conditions=32, before_jitter_stops=0, after_jitter_stops=32,
              safety_controls=18, unchanged_public_normal_clips=32, unchanged_cards=7,
              capture_rows=capture['rows'], browser_paint_measured=False, stop_dwell_in_latency=False,
              endpoint_candidate_admitted=False, endpoint_improved_cases=0, endpoint_comparison_cases=88, reviewed_partial_controls=16,
              web_admitted=True, runtime=server, camera_opened=False, physical_camera_rechecked=False,
              training_runs_this_task=0, historical_training_runs=75, private_data_read_this_task=False,
              private_training=False, private_tuning=False, private_candidate_selection=False,
              previous_frozen_files_preserved=116, C_T3F_preserved=True, next_single_action=next_action,
              implementation_and_verification='Same Codex: separate public controls, frozen replay, actual capture-code fixture, HTTP and hash checks. Not independent human validation.')
write(D/'RESULT.json',result)
text = f'''자동 종료 보완 — {now}

변경 전: 팔을 내리고 있어도 손점이 순간 위쪽으로 잘못 잡히면 종료 대기가 처음부터 다시 시작됐습니다.
변경 후: 팔이 대기 자세이고 손목이 아래를 향할 때만 아주 짧은 손점 흔들림을 제한적으로 기다립니다. 흔들린 시간은 종료 대기 시간에 넣지 않습니다. 실제로 손을 다시 올리거나 팔이 대기 자세에서 벗어나면 대기를 다시 시작합니다.

검증: 공개 정면 상반신16조건 정상 종료 유지. 그 공개 관측에 순간 검출 흔들림을 넣은32시험은 종료0→32로 개선. 실제 손 재진입·교차·가림 등 안전대조18 통과. 이는32명의 사용자시험이 아닙니다.
기존32개 공개 영상의 인식·피드백 결과 유지. 정상·손가락 차이·관측부족·부분 동작 등의 화면7검사 유지.
공개 영상을 실제 촬영 코드로 처리한4시험 모두 자동 종료. 최종 정상2시험의 종료 후 계산0.37~0.40초, 선택적 재검출 시험0.99초. 손내림 대기와 브라우저 표시 시간 제외. 일부 관절을 가린 사과 시험은 기존처럼 확인을 보류했습니다.

남은 문제: 감사에서 접근 움직임 두 번은 찾지만 끝 상태를 확인하지 못하는 경우. 다른 관측 근거를 합치는 후보를88조건과 직접 확인한 분리 자세 기반 부분동작16조건에서 검사했지만 개선 사례가 없어 적용하지 않았습니다. 공개 정상 영상에서는 같은 실패가 재현되지 않아 해결됐다고 말할 수 없습니다.

웹: http://127.0.0.1:8821/ (새로고침 필요). 서버PID {server['pid']}, 카메라 꺼짐. 실제 새 촬영에서 종료가 안정됐는지는 아직 확인 전입니다. 반복 동작 끝 문제는 남아 있으므로 전체 기능 완료가 아닙니다.
이번 작업은 종료 코드 보완이며 추가 신경망 학습은 없습니다. 기존 C·T3-F와116개 동결 파일 유지. 사용자 촬영은 이번 작업에서 읽거나 학습·튜닝·후보 선택에 사용하지 않았습니다.

다음 작업: {next_action}
작성·검증: Codex 구현과 별도 공개 대조/통합/HTTP/해시 검산. 독립 전문가 검증은 아닙니다.
'''
(D/'RESULT_KO.txt').write_text(text,encoding='utf-8')

# Preserve current canonical records, then insert only the new latest entries.
before=D/'before';before.mkdir(exist_ok=True)
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
    backup=before/p.name
    if not backup.exists():shutil.copy2(p,backup)
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8-sig')
entry=f'### 2026-11-17 자동 종료 순간 관측 흔들림 보완 적용\n{summary}\n현재 서버 {D / "serve.py"}, 8821 PID{server["pid"]}. 새 실제촬영 확인 전이며 전체완료 아님. 근거 {D / "RESULT_KO.txt"}, {D / "GRAPH.json"}.\n\n'
marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
assert marker in s;s=s.replace(marker,marker+'\n'+entry,1)
marker='## 9. 다음 단일 작업\n';assert marker in s
s=s.replace(marker,marker+'\n### 2026-11-17 종료 보완 적용 후 남은 끝 상태 문제\n'+next_action+'\n이미 시도한 후보는 개선0으로 미채택. 같은 검사를 반복하거나 사용자에게 같은 실패의 재촬영을 계속 요구하지 않는다.\n\n',1)
context.write_text(s,encoding='utf-8')
state=read(O/'STATE.json');state.update(current_stage=graph['status'], next_single_action=next_action, updated_at=now, last_updated=now)
t=state['capture_feedback_completion_20261116']
t.update(status=graph['status'], updated_at=now, current_graph=str(D/'GRAPH.json'), current_work_graph=str(D/'GRAPH.json'),
         active_feedback_runtime=str(D/'serve.py'),runtime_pid=server['pid'],next_single_action=next_action,
         report=str(D/'RESULT_KO.txt'),training_runs_current_task=0,training_runs_current_task_admitted=0)
t['rest_stability_endpoint_20261117']=dict(report=str(D/'RESULT.json'),text=str(D/'RESULT_KO.txt'),graph=str(D/'GRAPH.json'),
       admission=str(D/'web/ADMISSION.json'),summary=summary,training_runs=0,private_data_used=False,
       web_applied=True,endpoint_candidate_admitted=False,actual_webcam_pending=True)
write(O/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}
목표: 정면 웹캠 사과·감사 시범→촬영→자동종료→정상/교정/관측부족→재시도·퀘스트. 5방향은 아바타 전용.
{summary}
현재 최신 서버: {D/'serve.py'} · http://127.0.0.1:8821/ · 실제 서버PID {server['pid']}. 일반 실행환경에서 교체, 카메라 꺼짐. START_WEB_DEMO.cmd→frontal_demo_launcher.py도 새 서버로 연결.
버전: bounded_weak_rest_pause_v2. 이전 seated_rest_dual_window_v1 및 selective_front_rtmw_retry_v3 유지. 새122핀 중 이전116핀 보존. 가중치 변경0, 총학습75 유지.
R1 공개 검증/웹 적용 완료·실제 확인 전. R2 끝 상태 후보 개선0으로 미채택. R3 이번 변경 통합 통과. R4 전체 실제 시연 미완료.
이번 작업에서는 사용자자료를 읽지 않았으며 학습·튜닝·후보선택에도 사용하지 않음. 이전 사용자3건은 직전 user_retest_1734의 진단 기록으로만 유지. 실제 새 촬영 요청은 사용자 가능 시; 같은 실패를 반복 촬영시키지 않는다.
다음 한 작업: {next_action}
근거: {D/'RESULT_KO.txt'}, {D/'RESULT.json'}, {D/'GRAPH.json'}, checks/ 및 web/ 검증JSON.
종료 보완은 규칙 코드이며 AI 추가학습이 아니다. 끝 상태 후보의 2D 가까움은 물리적 접촉 정답이 아니다. 구현과 별도 검사는 같은 Codex, 독립 전문가 검증 아님. 반복 검사 대신 공개 실패 원인을 먼저 확보.
진행 중 실험 명령 없음. 운영 웹 서버만 실행. 다음 재개시 PID/상태 확인 후 이어받기.
''',encoding='utf-8')
ast.parse((ROOT/'frontal_demo_launcher.py').read_text('utf-8'))
assert str(D/'serve.py').replace('\\','/') in (ROOT/'frontal_demo_launcher.py').read_text('utf-8').replace('\\','/') or 'rest_stability_endpoint_20261117/serve.py' in (ROOT/'frontal_demo_launcher.py').read_text('utf-8')
assert all(sha(p)==h for p,h in admission['pins'].items())
write(D/'CLOSEOUT_CHECK.json',dict(passed=True,at=now,preserved_pins=len(admission['pins']),state_matches=read(O/'STATE.json')['current_stage']==graph['status'],runtime_pid=server['pid'],report_exists=(D/'RESULT_KO.txt').is_file(),handoff_latest=True))
print(summary)
print('CLOSEOUT PASS',server['pid'])
