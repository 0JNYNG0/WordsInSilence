from pathlib import Path
import json,hashlib,datetime,ast
M=Path(__file__).resolve().parent;G=M.parent;O=G.parent;R=O.parents[2];L=G/'public_temporal_visibility_20261116'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 now=datetime.datetime.now().astimezone().isoformat();comp=read(M/'learned_checks/RESULT.json');reg=read(M/'final_regression/RESULT.json');extra=read(M/'final_extra/RESULT.json');http=read(M/'web/HTTP_CHECK.json');cap=read(M/'web/CAPTURE_CHECK.json');pins=read(M/'web/ADMISSION.json')['pins'];parent=read(L/'web/ADMISSION.json')['pins']
 assert comp['failed']==0 and reg['shape_new_wrong']==reg['shape_lost_correct']==reg['distinction_failed']==reg['temporal_failed']==0 and not reg['known_error_loss']and extra['passed']and read(M/'FASTPATH_CHECK.json')['passed']and read(M/'FINAL_EPISODE_CONTROLS.json')['passed']
 assert read(M/'final_regression/confirmation/PATTERN_CHECK.json')['passed']
 for n in ['CONSUMER_CHECK','CARD_CHECK','ENDPOINT_CARD_CHECK','UNCERTAIN_LOCATION_CHECK','COVERAGE_CARD_CHECK','CAPTURE_CHECK']:assert read(M/'web'/f'{n}.json')['passed'],n
 assert http['cycle_observability_active']and not http['camera_active']and all(r['overall_pass']for r in http['rows'])
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())and all(pins[p]==h for p,h in parent.items())
 for p in list(M.glob('*.py'))+list((M/'runtime').glob('*.py')):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
 changes=[r for r in comp['rows']if r['before']!=r['after']or r['events_before']!=r['events_after']];remaining=[dict(id=r['id'],unknown=[k for k,v in r['after'].items()if v!='supported'])for r in comp['rows']if not all(v=='supported'for v in r['after'].values())];training=read(M/'motion/FROZEN.json');times=[{k:r[k]for k in ['phase','word','auto_end','frames','observed','redetection','stop_to_result_seconds']}for r in cap['rows']]
 def timing(word):
  a,b=[r for r in times if r['word']==word];return f"{a['stop_to_result_seconds']:.3f} → {b['stop_to_result_seconds']:.3f}초"
 stats=dict(known_public_conditions=len(comp['rows']),all_required_supported_before=comp['full_before'],all_required_supported_after=comp['full_after'],changed_conditions=len(changes),new_normal_warnings=0,lost_previously_observed_support=0,newly_inspected_unknown_conditions=[r['id']for r in comp['rows']if r['coverage_extended_unknown']],added_actual_cycles=sum(r['events_after']-r['events_before']for r in comp['rows']),all_old_event_bounds_preserved=True,shape_truth_anchors=237,shape_additional_wrong=0,shape_lost_correct=0,error_controls=40,distinct_patterns=6,temporal_controls=48,new_episode_controls=28,side_partial_hold=8,missing_digit=12,additional_camera_views=2)
 result=dict(status='COMPLETE_CYCLE_OBSERVABILITY_WEB_APPLIED',at=now,stats=stats,training=dict(actual_runs=3,admitted_runs=3,fit_intervals=len(training['rows']),fit_people=sorted(set(r['person']for r in training['rows'])),steps_per_seed=400,method='Three fixed-seed small MLP motion observer fine-tuned from prior weights with seven palm-diameter-normalized hand-edge/other-palm distance features. Supplements only unresolved original motion states. No common backbone/TCN/exact-finger retraining.',original_heldout_interval_check=read(M/'motion/CHECK.json')|{'rows':None},selection_scope='Reused public development conditions, not unbiased final test.'),corrections=['Associate existing arrival anchors with measured near episodes rather than rejecting a next cycle covered by the old long interval. Require real separated/approaching/posture evidence; no requested-count-driven event creation.','Calibrate non-collapsing projected palm diameter using the same17 prior-reviewed FIT frames from9people. Measures2D vicinity, not physical contact.','Supplement short motion observations using FIT-only learned hand-edge trends, with independently observed body relation and actual target arrival.','Preserve already observed support; newly found second cycle with hidden fingers remains gray at its actual interval.','Skip fallback calculations when required motion is already confirmed;54public outputs retain decisions and source spans.'],changes=changes,remaining=remaining,additional_views=extra['rows'],limitations=['52 reused public conditions include camera/resolution variants, not52 independent people or real-user accuracy.','Additional8L/R camera views were source-reviewed before initial candidate output then reused in final comparison; frontal person8 isFIT. Not an untouched new-person holdout.','16L gains a real second cycle; first-cycle shape evidence remains, second-cycle left shape becomes explicitly unknown. This is expanded inspection coverage, not a false new error or a forced green result.','8L retains a prior extra event proposal and unconfirmed left shape/target; remains incomplete. Exact cycle count for severely occluded viewpoints is not guaranteed.','Hidden individual fingers remain unknown. Remaining8known conditions and extra8L need better observable evidence/labels, not guessed pose filling.','Synthetic public truncation/missing/finger substitution controls do not establish all real human-error performance.','Runtime timing is public-video camera replay through actualRTMW, not physical webcam or browser paint.','Common code has no word/person exceptions, but this new repetition evidence is directly tested in gratitude-type repeated-hand interactions; not proof of universal KSL accuracy.'],capture_timing=times,pins=len(pins),parent_pins=len(parent),web=dict(url='http://127.0.0.1:8821/',pid=http['server']['pid'],entrypoint=str(M/'serve.py'),launcher='START_WEB_DEMO.cmd -> cycle_demo_launcher.py',camera_active=False),private_recordings_read_or_used=False,implementation_actor='Codex',verification_actor='Same Codex separate source review, executable comparison, serialized output and hash checks. No independent agent or KSL expert.',evidence=dict(comparison='learned_checks/RESULT.json',regression='final_regression/RESULT.json',controls='FINAL_EPISODE_CONTROLS.json',extra='final_extra/RESULT.json',optimization='FASTPATH_CHECK.json',capture='web/CAPTURE_CHECK.json',cards='web/COVERAGE_CARD_CHECK.json',http='web/HTTP_CHECK.json'))
 write(M/'RESULT.json',result);write(M/'PRESERVATION_VERIFIED.json',dict(passed=True,parent_pins=len(parent),all_pins=len(pins),all_hashes_match=True,private_data=False))
 graph=dict(at=now,nodes=[dict(id='A',label='원본 영상·관절·움직임·연결 원인 구분',status='complete'),dict(id='B',label='공통 구간 연결 보완·공개FIT 움직임 미세조정',status='complete'),dict(id='C',label='정상·오류·부분·가림·속도 비교 검증',status='complete'),dict(id='D',label='교정·확인 부족 표시 / 자동 종료·시간 / 웹 적용',status='complete'),dict(id='OBS',label='심한 가림·옆시점 손가락 근거 부족',status='remaining_observation_limit'),dict(id='USER',label='실제 촬영 정상→오류→교정 재시도',status='pending_user_not_learning')],edges=[['A','B'],['B','C'],['C','A'],['C','D'],['D','USER']]);write(M/'GRAPH.json',graph)
 report=f'''반복 동작과 가림 구분 보완 — {now}

이전 → 현재
- 두 번 했는데 하나로 묶이던 공개 옆시점5조건에서 두 번의 실제 접근을 확인했습니다. 앞 동작의 구간이 길어서 다음 동작을 덮는 문제와 짧은 복귀를 움직임 모델이 놓치는 문제를 나눠 고쳤습니다.
- 알려진 공개52조건에서 요구 항목 전체 확인 {comp['full_before']} → {comp['full_after']}. 서로 다른52명이 아니라 여러 시점·해상도를 포함한 재사용 공개자료 비교입니다. 새로2회가 확인된16L은 두 번째 왼손이 불분명해 전체통과로 만들지 않았습니다.
- 기존 정면17조건/기존 정상 통과/기존 동작 경계 유지. 손모양237정답 추가오판·정답소실0. 오류40·손가락패턴6·시간/누락48·추가반복28·측면부분8·손가락누락12조건 통과.

무엇을 고쳤나
1. 공통 연결 코드: 전체 구간이 겹친다는 이유로 다음 반복을 버리지 않고, 실제 떨어짐과 다시 다가와 도착한 순간을 구분합니다. 옆시점에서 짧아지는 손바닥 한 축 대신 실제 관측한 손바닥 전체 크기로 거리를 정규화했습니다. 기존공개FIT17프레임 기준만 사용했습니다.
2. 작은 움직임 판정부 학습: 공개FIT9명의 기존48동작정답으로3seed미세조정(각400회). 손 중심 이동에 손날→다른 손바닥의 상대거리 변화를 함께 입력했습니다. 원래 모델이 확실히 판단한 값은 유지하고, 불확실한 짧은 이동만 보조합니다. 공통100단어/Conv+GRU본체를 다시 학습한 것은 아닙니다.
3. 가림 처리: 첫 번째 손모양이 확인되었어도, 새로 찾은 두 번째 구간의 손가락이 가려졌으면 그 부분만 회색 ‘확인 부족’으로 표시합니다. 숨은 손가락을 정답으로 채우지 않습니다.
4. 속도: 이미 충분히 확인된 동작에서는 추가 보완 계산을 건너뜁니다.54개공개조건의 판정·근거구간이 그대로인지 확인했습니다.

웹 적용 확인
공개영상 카메라 재생의 종료→결과계산: 사과 {timing('WORD0099')}, 감사 {timing('WORD1290')}.
자동 종료/3초시작/모든수신프레임관측/재검출0 유지. 실제 웹캠·브라우저 화면표시까지 측정한 값은 아닙니다.
손모양 오류는 해당 손과 짧은 교정 문장, 한 번만 한 것은 미완료, 가림은 실제 불확실 구간에 회색으로 표시됨을 확인했습니다.

남은 한계
심한 옆시점의 손모양8조건은 아직 확인 부족입니다. 추가8L영상은 기존 여분 행동후보와 왼손·위치 불확실이 남아 미통과입니다. 충분한 관측 없이 통과 기준을 완화하지 않았습니다. 모든 수어와 실사용자 오류를 검증한 결과는 아닙니다.

최신화면 http://127.0.0.1:8821/ 새로고침. 카메라 꺼짐.
C·T3-F/100단어TCN/공통Conv+GRU본체/정확손가락판정부 보존. 사용자촬영 읽기·학습·튜닝·선택0. 새다운로드/외부전송/커밋0.
Codex직접구현 + 별도 실행/해시/직렬화검증. 별도에이전트·수어전문가검증은 아님.
근거: 같은폴더 RESULT.json, GRAPH.json, learned_checks, final_regression, final_extra, web.
''';(M/'RESULT_KO.txt').write_text(report,encoding='utf8')
 next_action='최신8821 사과·감사 실제 촬영 정상→명확한 오류→교정 재시도를 사용자가 시작하면 진단한다(학습/튜닝/후보선택 금지). 공개 잔여 옆시점은 손가락의 실제 관측/정답 근거와 마무리를 동작으로 보는 여분 후보를 분리해 다루며, 검증된 정상과속도를 유지한다. 숨은 상태나2D접촉을 임의확정하지 않는다.'
 state=read(O/'STATE.json');p=state['capture_feedback_completion_20261116'];accounted=p.get(M.name,{}).get('training_runs_accounted',False);state.update(current_stage='PUBLIC_CYCLE_OBSERVABILITY_WEB_APPLIED',last_updated=now,next_single_action=next_action);p.update(status=state['current_stage'],updated_at=now,report=str(M/'RESULT_KO.txt'),training_runs=p['training_runs']+(0 if accounted else 3),training_runs_current_task=3,training_runs_current_task_admitted=3,current_work_graph=str(M/'GRAPH.json'),current_checkpoint=str(M/'PROGRESS.json'),next_single_action=next_action,web_admitted=True,private_data_used=False);p[M.name]=dict(status=result['status'],at=now,path=str(M),result=str(M/'RESULT.json'),report=str(M/'RESULT_KO.txt'),graph=str(M/'GRAPH.json'),stats=stats,training_runs=3,training_runs_accounted=True,pid=http['server']['pid'],session_id=None,private_data=False);write(O/'STATE.json',state);write(M/'PROGRESS.json',dict(status=result['status'],at=now,live_server_pid=http['server']['pid'],training_and_checks_finished=True,next_single_action=next_action))
 context=R/'WordsInSilence_PROJECT_CONTEXT.md';t=context.read_text(encoding='utf-8-sig');heading='### 2026-11-17 실제 반복 구간 분리·짧은 복귀 학습 / 웹 적용';assert heading not in t
 summary=f'공개52기존조건전체확인38→44,추가반복5조건(16L새2번째왼손unknown유지),기존정면17/정상/경계보존. 긴구간중첩으로다음반복누락을실제도착/떨어짐으로보완;FIT17프레임손바닥전체크기정규화;FIT9명48움직임정답3seed미세조정으로짧은복귀보완.100단어/공통Conv+GRU/C·T3-F/정확손가락동결.237손모양추가오판/정답소실0,40오류/6패턴/48시간누락/28반복/8부분/12손가락누락통과.8LR추가공개시점검증후재사용,새사람holdout아님.54조건계산단축동일판정확인. 웹12쌍TCN/DTW/정상/교정/가림/부분표시유지. 공개촬영사과{timing("WORD0099")},감사{timing("WORD1290")} 종료→계산,실제웹캠표시시간아님.8821 PID{http["server"]["pid"]} cameraoff/cycle_demo_launcher;부모177/전체190핀보존.누적훈련72,개인자료0. 남은8옆시점손모양unknown/추가8L여분후보·관측부족. 근거 {M.relative_to(R).as_posix()}/RESULT.json·RESULT_KO.txt·GRAPH.json.'
 t=t.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+heading+'\n'+summary+'\n',1);t=t.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-17 반복 보완 적용 이후\n'+next_action+'\n',1);context.write_text(t,encoding='utf8')
 (R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}

## 완료·현재 적용
{M}
RESULT.json / RESULT_KO.txt / GRAPH.json. A원인분리→B공통연결/소형움직임미세조정→C정상오류검증→D웹적용 완료.
{summary}

## 방법·주의
CycleLesson: 실제떨어짐→근접에피소드와기존arrivalanchor연결,옛전체구간중첩으로새반복배제않음. 준비단계는별도몸관계근거필요. LearnedCycleLesson:원래motionunknown만48FIT정답학습된손날거리관측으로보조,목표횟수로이벤트생성않음. 이미동작확인되면보완추론생략(FASTPATH54동일판정).
16L첫손모양근거유지/새2번째손모양unknown은검사범위확장임. GATE_CLARIFICATION참조.정상통과를unknown으로바꾼사례없음.숨은손가락추정없음.
중간comparison은준비3회오류포함,qualified는수정기하후보,learned_checks최종판정근거. FASTPATH_METADATA_CHECK초기실패는scale_basis진단문자열차이;최종FASTPATH_CHECK는그문자열만제외하고모든판정/근거/경계동일. UNOPTIMIZED_CONSUMER_CHECK이전시간,최종CONSUMER_CHECK/CAPTURE_CHECK가속도근거.
8LR추가시점은초기후보점검후재사용/정면FIT이므로새사람미사용일반화주장금지.8L기존3후보/shape,targetunknown남음.

## 실행·보존
http://127.0.0.1:8821/ PID{http['server']['pid']} ready/cameraoff. START_WEB_DEMO.cmd→cycle_demo_launcher.py→{M/'serve.py'}.
실제카메라미개방;공개재생4회로자동종료/관측재사용검증.사용자촬영읽기/학습/튜닝/후보선택0.새다운로드/외부전송/커밋0.Codex구현+분리검증,별도에이전트/수어전문가없음.훈련/검증종료,서버만유지.

## 다음 한 작업
{next_action}
''',encoding='utf8');print('FINALIZED',stats,len(pins),flush=True)
if __name__=='__main__':main()
