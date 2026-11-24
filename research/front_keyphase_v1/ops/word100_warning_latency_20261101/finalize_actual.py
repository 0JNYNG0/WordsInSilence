from pathlib import Path
import json,datetime,statistics,hashlib
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=read(U/'BENCHMARK.json');rows=[]
for clip in dict.fromkeys(r['clip'] for r in b['runs']):
 a=[r['wall_seconds'] for r in b['runs'] if r['clip']==clip and r['condition']=='serial'];c=[r['wall_seconds'] for r in b['runs'] if r['clip']==clip and r['condition']=='candidate']
 rows.append(dict(clip=clip,serial=a,candidate=c,serial_mean=statistics.mean(a),candidate_mean=statistics.mean(c),reduction_percent=(1-statistics.mean(c)/statistics.mean(a))*100))
for n in ['review/BENCHMARK_VERIFICATION.json','review/DETAIL_VERIFICATION.json','ENCODER_GATE.json','LIVE_FINAL_VERIFICATION.json','LIVE_ROLLBACK_VERIFICATION.json']:assert read(U/n)['status'].startswith('PASS'),n
for p,h in read(U/'FROZEN.json')['source_pins'].items():assert sha(p)==h,p
assert read(U/'EXPLICIT_GATE_TIMEOUT.json')['status']=='TIMEOUT_NO_PARTIAL_ACCEPTED' and not (U/'EXPLICIT_GATE.json').exists()
stop=read(U/'SERVER_STOP_VERIFICATION.json');assert not stop['new8803_listeners'] and not stop['stopped_process_exists'] and stop['old8802_pid']==26000
web={k:read(U/f'{k}_LIVE_UI_VERIFICATION.json')['request_timing']['analysis_seconds'] for k in ['CANDIDATE','PRELOAD','ROLLBACK']}
trace=[json.loads(x) for x in (U/'EXPLICIT_WINDOW_TRACE.jsonl').read_text().splitlines()];finished=[x for x in trace if x['event']=='CALL_END']
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
summary=f'정상 공개7경고를 손모양전환1·2D방향불안정2·팔위치4로 구분. 원판정/수치/경로/짧은피크를 보존하며 근거종류·시점상세추가,96저장조건 검산. 중복decode제거/CPU준비-GPU실행겹침의 동일2편각2회 분석기평균 {rows[0]["serial_mean"]:.2f}→{rows[0]["candidate_mean"]:.2f}초,{rows[1]["serial_mean"]:.2f}→{rows[1]["candidate_mean"]:.2f}초 및109창 입력/출력exact 확인. 그러나 실제웹 첫요청{web["CANDIDATE"]:.2f}초,시작import이동{web["PRELOAD"]:.2f}초,복귀후{web["ROLLBACK"]:.2f}초로 GPU호출지연재발. 전용InferRequest도45초timeout(부분미채택). 성능안정화/경고제거 미달,새8803종료. 기존8802 PID26000/C/T3F/운영/개인자료보존.'
next_action='최우선은 고정 입력·모델의 반복 GPU호출 지연 원인 분리. 창별로그초반0.13~0.17초→후반4~6초, 정확한원인미확정. 완료실험재실행/모델학습확대금지. 별도사전고정시험으로 기존사용가능CPU경로와GPU입력버퍼복사/요청소유·자원사용을 작은동일입력에서비교하여 동일출력/시간상한함께확인. 드라이버·전원설정변경/다운로드자동확대X. 경고7은DEV문턱완화로없애지말고 공개FIT에서 손목/RGB회전근거·전환단계·중요부위 관계정답확보. 사용자촬영학습/튜닝/선택금지.'
report=f'''이번 결과: 일부 계산 개선은 확인했지만, 처리시간과 경고7개를 모두 해결하지는 못했다.
{now}

경고7개의 내용
- 순종1개: 손을 펼쳤다가 접는 전환 순간에 판정부 값이 크게 달랐다. 대표 차이는 범위 안이지만 한 시점의 최대값으로 경고했다. 정렬 시차/추정오류/실제 차이 중 원인을 확정할 정답이 부족하다.
- 잘하다2개: 카메라를 향한 주먹에서 손목 검출점이 손 안쪽으로 옮겨지며 화면상 방향값이 뒤집히는 근거가 보였다. 2D 방향 차이를 실제 손목 회전 오류라고 단정할 수 없다.
- 조상4개: 팔 높이·벌림 차이가 보였다. 의미를 해치는 오류인지 확인되지 않았다.
13공개 영상74정확한 원프레임을 확인했다. Codex 영상검토이며 한국수어 전문가의 독립적인 교육정답은 아니다.

보완한 내용
각 차이가 손모양 모델의 추정 차이인지,2D 방향/팔 위치의 측정 차이인지 구분하고 정확한 시간대를 추가했다. 단1시점 차이도 삭제하지 않았다.
원래 경고·수치·DTW 경로·같은 두 전체 참고·미관측 정보를 모두 보존했다. 96저장 조건 검산을 통과했다. 새 원본영상96편 시험이라는 뜻은 아니다.
경고는7개 그대로다. 설명 정보가 개선됐으며, 오경고7개 제거/교육 정확도 향상을 주장하지 않는다. 교차의 좌우정보는 보존했지만 교차만 별도로 전문가 정답표시한 정확도 시험은 아니다.

속도 후보의 고정 비교
같은 영상을 모두 읽는 중복 작업4→3회. CPU입력준비와 GPU계산의 대기를 줄였다. GPU는 단일 순차실행이고 준비큐만4개다.
모든 원프레임/PTS/ROI/16프레임 RGB창/13프레임 판정부 문맥/224변환/정밀도/얼굴·손·팔 모델은 유지했다.
같은 영상에서 기존/후보를 A-B-B-A로 각각2회 비교했다.
순종 평균 {rows[0]['serial_mean']:.2f}→{rows[0]['candidate_mean']:.2f}초, {rows[0]['reduction_percent']:.1f}% 감소.
이집트 평균 {rows[1]['serial_mean']:.2f}→{rows[1]['candidate_mean']:.2f}초, {rows[1]['reduction_percent']:.1f}% 감소.
전체 관측·좌우배정·보정값·모델출력이 동일했다. 별도로109개 창의 입력 바이트와 모든 출력도 일치했다.
이는 영상분석기 시간이며 업로드·DTW·웹응답 전체 시간이 아니다. 두 영상각2회, 첫serial 초기모듈비용 및 이 노트북 환경 변동의 한계가 있다.

중요한 새 문제와 미적용 이유
실제 웹 같은 공개 '잘하다' 첫 요청 {web['CANDIDATE']:.2f}초.
라이브러리 준비를 시작 단계로 옮긴 추가 시험 {web['PRELOAD']:.2f}초. 변경 철회 후에도 {web['ROLLBACK']:.2f}초로 재발했다. 따라서 import이동만이 원인이라고 단정할 수 없다.
지연은 주로 OpenVINO 계산 호출 내부였다. 완료한 세 요청의 출력과 상세 비교는 같았지만 시간이 매우 불안정했다.
전용 실행요청도 시험했으나55창 중 {len(finished)}개 완료 후45초 제한에서 중단했다. 불완전 결과를 채택하지 않았다. 초반 호출0.13~0.17초→후반4~6초로 길어졌다. GPU kernel시간 그 자체가 아니라 동기호출과 결과반환의 시간이며 정확한 원인은 미확정이다.
공식 API는 전용요청 사용을 권고하지만 이번 지연의 원인을 증명하지 않는다: https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.CompiledModel.html
과거15.7/31.5초는 서로 다른 입력/실행이라 후보와 직접 전후 비교하지 않았다. 속도안정화/3초 목표를 달성하지 못했다.

현재 적용 상태
새8803실험 서버는 종료했다. 기존8802 PID26000,8801기록재생, C·T3-F, 운영기본은 그대로다. 후보 코드와 상세근거·화면은 연구폴더에 남겼으며 기본 환경에 적용하지 않았다.
이번 추가학습0. 사용자/멘토촬영은 학습·튜닝·후보선택·참고자료에 사용하지 않았으며 공개 자료만 시험했다.
Codex 별도검산: 속도/관측1,187개, 상세근거142,188개 PASS. 이 숫자는 사람 교육정답이나 정확도 표본수가 아니다.
최초 문자열배열 검산 오류는 보존·수정하고 완료4회 추론을 재실행하지 않았다. import 감시검사의 과도한 비용은 별도 중단기록 후 가벼운 검사로 바꿨다. 추가 GPU지연 및 시간제한 기록도 삭제하지 않았다.

다음 우선순위
{next_action}
최종목표는 설명과 여러 참고로 중요한 손모양·위치·움직임·순서 차이를 알려주는 것이다. 이번에 관측 차이와 수어 오류를 분리했지만 의미판정과 반복실행 지연은 남아 있다.

근거: BENCHMARK.json, ENCODER_GATE.json, warning_audit/WARNING_REVIEW.json, warning_audit/DETAIL_REPORT.json, LIVE_FINAL_VERIFICATION.json, LIVE_ROLLBACK_VERIFICATION.json, EXPLICIT_WINDOW_TRACE.jsonl, EXPLICIT_GATE_TIMEOUT.json, review/.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
complete=dict(status='DIAGNOSIS_COMPLETE_CANDIDATE_NOT_DEPLOYED_RUNTIME_UNRESOLVED',updated_at=now,summary=summary,paired_latency=rows,web_timings=web,explicit_completed_windows=len(finished),explicit_timeout_seconds=45,normal_warnings_before=7,normal_warnings_after=7,accuracy_improvement_claim=False,training_runs=0,user_inputs_used=0,protected_models_changed=False,experimental_server_stopped=True,original8802_pid=26000,next_action=next_action,result=str(U/'RESULT_KO.txt'))
with (U/'COMPLETION.json').open('x',encoding='utf8') as f:json.dump(complete,f,ensure_ascii=False,indent=2)
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig');t=t.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n### 2026-11-01 경고 원인·입력 보존 검증 — GPU 지연 미해결, 운영 미적용\n'+summary,1);t=t.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-01 다음 단일 작업 — 반복 GPU 지연 원인 분리\n'+next_action,1);p.write_text(t,encoding='utf8')
p=O/'STATE.json';d=read(p);d['current_stage']=complete['status'];d['next_single_action']=next_action;d[U.name]=complete;p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_action+'\n\n현재단위 '+U.name+'. 모든실행종료. 새8803종료, 기존8802 PID26000(실제확인필요),8801이전재생유지. 8회benchmark/109창gate/96조건상세/공개웹3회완료;전용요청45초timeout전체결과없음. 완료추론재실행금지. 진단완료·지연미해결이지문제해결완료아님. 최종 RESULT_KO.txt/COMPLETION 기준. finalize_unexecuted_draft.py는미실행초안이고현황아님. 사용자·멘토자료/C/T3F/운영보존,새모델/다운로드/외부전송/Git0. 별도Codex검산≠사람교육정답.\n',encoding='utf8')
with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+now+' 경고 원인과 입력 보존 후보 — GPU 지연 발견\n'+summary+'\n방향성: 목표유지. 관측차이를언어오류로과장하지않고,프레임을없애지않는중복/대기제거를시험. 처음속도개선만으로채택하지않고 실제웹184/131초지연도보존·보고했다. 모델/학습인원추가보다실행원인분리가우선이다.\n다음: '+next_action+'\n근거: '+str(U/'RESULT_KO.txt')+'\n')
assert read(O/'STATE.json')[U.name]['status']==complete['status']
print(complete['status'])
