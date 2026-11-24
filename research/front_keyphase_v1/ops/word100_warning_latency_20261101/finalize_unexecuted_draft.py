from pathlib import Path
import json,datetime,statistics,hashlib
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
live=read(U/'LIVE_FINAL_VERIFICATION.json');assert live['status']=='PASS'
bench=read(U/'BENCHMARK.json');guard=read(U/'review/DETAIL_VERIFICATION.json');bv=read(U/'review/BENCHMARK_VERIFICATION.json')
assert guard['status'].startswith('PASS') and bv['status'].startswith('PASS')
assert read(U/'ENCODER_GATE.json')['status']=='PASS'
for p,h in read(U/'FROZEN.json')['source_pins'].items():assert sha(p)==h,p
rows=[]
for clip in dict.fromkeys(r['clip'] for r in bench['runs']):
 old=[r['wall_seconds'] for r in bench['runs'] if r['clip']==clip and r['condition']=='serial']
 new=[r['wall_seconds'] for r in bench['runs'] if r['clip']==clip and r['condition']=='candidate']
 rows.append(dict(clip=clip,serial=old,candidate=new,serial_mean=statistics.mean(old),candidate_mean=statistics.mean(new),reduction_percent=(1-statistics.mean(new)/statistics.mean(old))*100))
before=live['before_request']['analysis_seconds'];after=live['after_request']['analysis_seconds'];startup=live['after_startup']
preload=startup['warmup']['capture_import_preload']['seconds']
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
summary=f'정상 공개자료의 7경고를 손모양전환1·2D방향불안정2·팔위치4로 구분했다. 원경고를 정상으로 바꾸지 않고 근거종류·정확한 차이시점을 별도 표시했다. 중복decode제거/입력준비-GPU계산겹침으로 동일2영상 각2회 평균 {rows[0]["serial_mean"]:.2f}→{rows[0]["candidate_mean"]:.2f}초,{rows[1]["serial_mean"]:.2f}→{rows[1]["candidate_mean"]:.2f}초(분석기만), 전체관측/출력exact. 별도같은공개웹영상 firstrequest {before:.2f}→{after:.2f}초, import준비 {preload:.2f}초를startup에분리. 8803 별도연구데모 적용, 8802/C/T3F/운영/사용자자료 보존. 7경고자동해결·교육정확도향상·3초달성은아님.'
next_action='남은차이를정오로바꾸기전에 공개FIT영상에서 손목/손바닥/RGB 근거로 실제회전과2D투영불안정 구분, 손모양전환단계와 중요부위 관계의 확인정답을 확보한다. DEV7에맞춘문턱완화/표현예외/추가무근거재학습금지. 속도는현재동일정보계산후보유지, 다음별도시험은촬영중관측준비를진행해촬영종료후대기를줄일수있는지 전체frame/PTS/ROI동등성기준으로설계. 완료8회benchmark/encoder/96조건재실행X; 사용자자료학습/튜닝/선택X.'
text=f'''작업 결과 — 정상 차이 경고 원인과 처리시간 보완
{now}

핵심: 계산은 빨라졌고, 경고의 근거는 더 정확하게 설명할 수 있게 됐다.
그러나 정상 영상의 경고7개를 자동으로 해결했거나 수어 피드백 AI가 완성됐다는 결과는 아니다.

1. 경고7개의 실제 내용
순종1: 손을 펼쳤다가 접는 전환에서 판정부 값이 순간적으로 크게 달랐다. 전체 대표 차이는 범위 안이지만 한 시점의 최대값 때문에 경고했다. 실제 중요한 변화인지/정렬 시차인지/모델 오류인지 분리 정답은 아직 부족하다.
잘하다2: 주먹이 카메라를 향할 때 검출된 손목점이 손 안쪽으로 이동해 화면상의 방향값이 뒤집히는 근거를 확인했다. 화면의 두 점으로 만든 방향 차이를 실제 손목 회전 오류라고 말할 수는 없다.
조상4: 팔 높이·벌림의 차이가 보였다. 그것이 의미를 해치는 차이라는 정답은 없다. 영상 간 좌표 차이와 교육적인 틀림을 구분해야 한다.
13공개 영상에서74정확한 원프레임을 확인했다. Codex 영상검토이며 독립적인 한국수어 전문가의 교육정답이 아니다.

2. 무엇을 바꿨나
같은 영상을 불필요하게 다시 읽던 과정을 4회→3회로 줄였다.
영상 입력을 준비하는 CPU와 순서대로 계산하는 GPU가 서로 불필요하게 기다리지 않게 했다. GPU는 한 번에 한 창을 계산하며 준비 큐만4개다.
얼굴·손·팔 모델, 전체 프레임,16프레임 영상창,224픽셀 변환,정밀도,13프레임 판정부 문맥은 그대로다.
각 차이에 관측 종류와 정확한 시간 구간을 추가했다. 2D 방향은 실제3D 회전으로 확정하지 않으며 짧은 차이도 삭제하지 않는다. 숫자가 작은/불확실한 구간을 맞았다고 바꾸지 않았다.
첫 요청에서 쓰던 capture 라이브러리 import를 서버 준비 단계로 옮겼다. 그 비용은 없어지는 것이 아니라 시작할 때 지불한다. 영상별 검출·추적 상태는 매 요청 새로 만든다.

3. 같은 영상에서 시간 비교
순종: 기존2회 {rows[0]['serial'][0]:.2f}/{rows[0]['serial'][1]:.2f}초 → 보완2회 {rows[0]['candidate'][0]:.2f}/{rows[0]['candidate'][1]:.2f}초. 평균 {rows[0]['serial_mean']:.2f}→{rows[0]['candidate_mean']:.2f}초({rows[0]['reduction_percent']:.1f}% 감소).
이집트: 기존2회 {rows[1]['serial'][0]:.2f}/{rows[1]['serial'][1]:.2f}초 → 보완2회 {rows[1]['candidate'][0]:.2f}/{rows[1]['candidate'][1]:.2f}초. 평균 {rows[1]['serial_mean']:.2f}→{rows[1]['candidate_mean']:.2f}초({rows[1]['reduction_percent']:.1f}% 감소).
위 값은 영상 분석기 시간이며 업로드·참고 비교·화면을 포함한 전체 대기시간이 아니다. 두 영상/각2회의 작은 이 노트북 시험이다. 첫 serial에는 import 등의 초기비용이 더 들어갔을 수 있다. 각 영상 내부 ABBA만 비교했다.
실제 웹의 같은 '잘하다' 공개 영상: import준비 이동 전 {before:.2f}초 → 이동 후 {after:.2f}초(서버의 전체 분석; 업로드/시작은 별도). 한 번씩의 통합 확인이며 차이 전부를 import 효과로 단정하지 않는다.
이번 서버 시작 총 {startup['total_startup_seconds']:.2f}초, 그 안의 capture import {preload:.2f}초를 별도 기록했다. 준비비용을 숨기지 않았다.
과거15.7/31.5초는 서로 다른 입력/실행이라 이번 후보와 직접 비교해 큰 개선율을 만들지 않았다. 과거31초의 원인을 이번 결과만으로 단정하지 않는다. 몸·손 관측과 RGB 특징 계산이 여전히 주된 비용이다. 현재 약3초 목표는 달성하지 못했다.

4. 중요한 정보는 유지됐나
두 공개 영상의8번 처리에서 관측·좌우배정·보정값·모델 출력이 동일했다. 별도로 같은ROI/PTS로109개 RGB 창의 입력 바이트와 모든 출력을 정확히 비교했다.
기존96저장 조건의 원래 판정·경로·점수·미관측·단1시점 차이를 모두 보존했다. 속도변경/짧은변화/손누락 대조를 포함한다. 이96개가96개의 새로운 원본영상 시험이라는 뜻은 아니다.
정상 영상의 경고는7개 그대로다. 경고를 지워서 성능이 좋아진 것처럼 만들지 않았다. 교차 동작의 좌우정보 보존은 확인했지만, 교차만 별도로 전문가가 정답표시한 정확도 시험은 아니다.
같은 웹 영상의 import이동 전후도 원래 배열·상세 비교·근거가 동일했다. 반복 검산은 독립적인 사람 수어 정답을 뜻하지 않는다.

5. 실제 적용 범위
별도 연구 데모: http://127.0.0.1:8803/
기존8802와 C·T3-F 및 운영 기본은 그대로다. 이번 추가학습은0이며 사용자 촬영은 학습·튜닝·후보선택·참고자료에 사용하지 않았다. 공개 자료만 시험했다.
원래 경고/최대차이 버튼은 유지하고 새 '수치 차이가 나타난 시점 확인'에서 짧은 차이의 위치를 볼 수 있다.
실행/구현root 및 Codex 에이전트; 별도검산 결과와 실제 성능 측정을 구별한다.
검산 코드의 문자열 배열 처리 실패는 첫 오류를 보존해 수정했다. 이미 끝난 첫4회 추론은 다시 돌리지 않고 결과를 이어 사용했다. import 검사감시기 과도한 비용도 중단기록 후 가벼운 검사로 바꿨으며 실제 모델 실패가 아니다.

6. 남은 문제와 다음 목표
{next_action}
최종 목표는 계속 '설명과 여러 참고를 이용해 어느 손의 어떤 모양·방향·위치·움직임을 어떤 순서로 했고 무엇이 중요한 차이인지 알려주는 것'이다. 이번은 관측 계산과 경고 해석 기반의 개선이며 단어 인식률 상승이나 교육적 정오 모델 완성을 주장하지 않는다.

근거: BENCHMARK.json, ENCODER_GATE.json, LIVE_FINAL_VERIFICATION.json, warning_audit/WARNING_REVIEW.json, warning_audit/DETAIL_REPORT.json, review/BENCHMARK_VERIFICATION.json, review/DETAIL_VERIFICATION.json.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
server=read(U/'SERVER_ACTIVE.json')
complete=dict(status='COMPLETE_DIAGNOSIS_LOSSLESS_SPEED_AND_EVIDENCE_DETAIL_WITH_LIMITS',updated_at=now,summary=summary,paired_latency=rows,first_web_seconds=before,preloaded_web_seconds=after,normal_warnings_before=7,normal_warnings_after=7,accuracy_improvement_claim=False,training_runs=0,user_inputs_used=0,protected_models_changed=False,research_server=dict(port=8803,pid=server['listener_pid']),next_action=next_action,result=str(U/'RESULT_KO.txt'))
with (U/'COMPLETION.json').open('x',encoding='utf8') as f:json.dump(complete,f,ensure_ascii=False,indent=2)
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig');t=t.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n### 2026-11-01 정상 경고 원인·입력 보존 속도 개선 완료\n'+summary,1);t=t.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-01 다음 단일 작업 — 남은 차이의 의미와 촬영 종료 후 대기\n'+next_action,1);p.write_text(t,encoding='utf8')
p=O/'STATE.json';d=read(p);d['current_stage']=complete['status'];d['next_single_action']=next_action;d[U.name]=complete;p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_action+f'\n\n현재단위 {U.name}. 전체학습/평가/8회속도시험/109창동등성/96조건상세검산/공개웹2회시험종료. 재실행금지. 연구8803 actualPID{server["listener_pid"]}, 이전8802 PID26000보존. 실제프로세스/health확인후이어받기. 현재7경고는해결0,정보분리개선과속도개선만확인. 결과 RESULT_KO.txt/COMPLETION 및 review. 사용자자료학습/튜닝/선택/새모델다운로드/외부전송/Git/운영교체없음. Codex검산≠독립사람정답.\n',encoding='utf8')
with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+now+' 정상 경고의 원인과 입력 보존 속도 개선\n'+summary+'\n방향성: 목표는 유지. 관측 차이와 수어 오류를 혼동하지 않도록 근거해석을 분리하며, 속도를 위해 프레임/짧은동작을 삭제하지 않고 실행순서의 중복과대기를줄였다. 모델·학습인원 추가가아닌병목별수정. 경고7해결/전체피드백완성으로부풀리지않음.\n다음: '+next_action+'\n근거: '+str(U/'RESULT_KO.txt')+'\n')
assert read(O/'STATE.json')[U.name]['status']==complete['status']
assert all((U/p).exists() for p in ['RESULT_KO.txt','COMPLETION.json','LIVE_FINAL_VERIFICATION.json','PRELOAD_LIVE_UI_VERIFICATION.json'])
print(json.dumps(dict(status=complete['status'],summary=summary),ensure_ascii=False))
