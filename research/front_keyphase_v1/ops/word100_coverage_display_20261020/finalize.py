from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;B=U.parents[1];ROOT=B.parents[1];OPS=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
f=read(U/'FROZEN.json');v=read(U/'PUBLIC_VALIDATION.json');it=read(U/'INTEGRATION.json');qa=read(U/'UI_QA.json');http=read(U/'HTTP_CHECK.json');stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
assert all(x['status']=='PASS' for x in [v,it,qa,read(U/'SELFTEST.json')]) and http['status']=='OK'
assert not (U/'FINAL_AUDIT.json').exists()
for p,h in f['protected'].items():assert sha(p)==h,p
for p,h in read(U/'IMPLEMENTATION.json').items():assert sha(p)==h,p
for n,h in f['previous_results'].items():assert sha(OPS/'word100_missing_phase_validation_20261020'/n)==h,n
text='''DTW 차이 수치 + 관측 비율·미관측 시간 표시 완료

웹 적용: http://127.0.0.1:8765/practice.html?model=t3f (C 선택도 동일 표시)
새로고침 후 결과 분석의 손가락 굽힘/손 위치/이동 궤적 차이 아래에 새 관측 정보가 표시된다. 기존 점수, 참고범위, C/T3-F 및 모델선택은 유지했다.

표시 항목
1. 좌우 손별 촬영 전체 관측비율(실제 프레임 시각을 이용한 시간 가중 추정).
2. 손이 처음부터 마지막으로 보인 구간의 관측비율. 전체 촬영과 나란히 표시하므로 준비·마무리 때문에 낮아진 비율을 구별할 수 있다. 양손 중 하나라도 관측된 첫끝이며 의미상 핵심구간의 확정은 아니다.
3. 파랑=계산 가능한 관절 근거,회색=근거 없음,줄무늬=프레임 사이 긴 시간 공백. 가장 긴 미관측 시간과 시작~끝 초단위 목록.
4. 비교하지 않는 손은 기존 공개 참고 역할을 그대로 유지. 사용자에게 안 보인다는 이유로 새롭게 제외하지 않는다.

해석
관측80%는 촬영 시간의 약80%에서 해당 항목의 계산 근거를 확보했다는 뜻이며,80% 올바르게 수어를 했다는 뜻은 아니다. 손목 위치는 관측되었어도 손가락이 충분히 보이지 않으면 굽힘 관측비율이 더 낮을 수 있다. 현재 운영 DTW의 관측조건(몸 기준 손목/19각도 계산 가능)을 그대로 썼다. 이전 부분각도 연구 후보를 점수에 몰래 연결하지 않았다.
관측비율은 질의 촬영의 원래 프레임 기준이다. 모든 프레임이 DTW 점수에 사용됐다는 뜻은 아니며,참고영상과 실제로 짝지어 비교한 비율과 다르다. 원래 점수/참고 범위는 관측된 부분의 차이로 해석한다.
미관측 원인은 정상 손내림,사용하지 않는 손,겹침,검출 실패 등일 수 있다. 원인을 단정하거나 수행오류/감점/자동실패를 추가하지 않았다. 화면 밖 손이 실제 무엇을 했는지는 알 수 없다.

공개 검증
- AI Hub100단어1199영상(TRAIN899/DEV300),기존 DTW 원관측과 sensor관측 두 경로=2398입력. 14388항목·손 결과의 시간 적분을 별도 계산과 비교했다. 최대 백분율 오차2.85e-14.
- 고정100단어의TRAIN01각2경로200영상,24544프레임에서 기존 frame_features와 새 관측가능 마스크 일치. 점수/정렬 계산부 변경 없음.
- 시간축0.75배/1.5배4796대조:관측비율 불변,미관측 초만 비례 변경. 반복된 관측2398대조:시간 집계 일치,오류판정추가 없음. 이 반복은 합성 관측 대조이며 모든 수어에서 반복횟수 변경이 언어적으로 허용된다는 검증은 아니다. 실제 빨리 촬영했을 때 검출성능까지 같다는 뜻도 아니다.
- 이전 전체원본의 안정적 손내림 근거가 있고sensor에는손점이없는121882손프레임(2391영상·손단위)을 확인. 표시를 단순 관측정보로 유지하고 자동 오류/필수손 제외 판단에 쓰지 않는다. 원본도 MediaPipe 추정이며 의미정답은 아니다.
- 전체손 및 짧은구간 가림5319항목 검사,실제근거제거4511조건. 새 관측누락이 가려지거나 관측비율을 올리지 않음. 전체손 관측없음0%는0차이점수와 구별.
- 자체10검사,별도공개산술·기존기능마스크 검산,3모델 결과전달 불변검사 PASS. 같은Codex별도구현검산이며 다른 에이전트 검토 아님.
- 브라우저 공개자료화면:3항목/좌우6막대,미관측목록,선택손,기존점수·참고상태 유지,390px모바일 넘침없음. 카메라/실제사용자기록 요청0. 예시의 순위는 UI용 가상값이며 새 인식결과 아님.

적용과 보존
관측정보 추가 모듈 및 결과전달·표시 파일만 수정. C/T3-F 가중치6개,DTW계산/역할/참고은행 등26보호파일과 과거결과 해시 불변. 서버62068,카메라없음,새화면파일전달 확인. 추가학습0/인식보조0/사용자·멘토자료 검증접근0/보류14~18원시접근0.

검증 도중 복구
최초 브라우저 모의출처가 안전한 출처가 아니어서 초기화 hook이 생성되지 않았다. 해당기록을 보존하고 localhost를 모의출처로 사용하여 같은 화면검사PASS. 기존모델오류아님.
공개 결과전달 검사 후 in-process app import/TestClient 단계가 반환되지 않아 그검증세션을중단했다. 원인미확정이며 최초증거보존. 같은공개자료의3모델전달/가림검사를직접함수로완료하고,실제서버health·정적파일전달별도확인. 사용자촬영API를실행해시험하지는않았다.

남은 일
실제 웹캠 새촬영에서 표시를 확인할 수 있다. 이번 작업은 관측정보의 정확한 전달이며 인식률향상/올바른수어판정 성능 개선을 측정한 것이 아니다. 정상 속도·반복의 언어적 허용 여부와 필수동작실제생략 판정은 여전히 별도다. 다음은 표시결과를함께확인하고,필요하면 공개동작에서 관측근거와단계차이를연결하는 평가계획검토. 추가학습은결과확인전진행하지않는다.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>관측 시간 표시 · WordsInSilence</title><style>*{box-sizing:border-box}body{font-family:"Malgun Gothic",Arial,sans-serif;background:#f1f5f8;color:#20384c;margin:0;line-height:1.8}main{max-width:1060px;margin:auto;padding:40px 24px 70px}h1{font-size:36px;line-height:1.4;margin:12px 0 18px;letter-spacing:-1px}h2{font-size:23px;margin-top:0}.tag{font-weight:bold;color:#347997}.card{background:white;padding:28px;border:1px solid #dce5ec;border-radius:16px;margin:22px 0}.lead{font-size:19px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}.grid img{max-width:100%;display:block;border:1px solid #dce5ec;border-radius:8px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce5ec}th{background:#f1f5f8}.note{border-left:4px solid #337e9d;padding:10px 16px;background:#eef5f8}a{color:#16668b}.small{font-size:13px;color:#4c6376}li{margin:8px 0}.scroll{overflow:auto}@media(max-width:680px){main{padding:20px 14px}.card{padding:20px}.grid{grid-template-columns:1fr}h1{font-size:29px}}</style><main>
<div class="tag">WORDS IN SILENCE · 2026.10.20</div><h1>차이 수치와 함께<br>관측한 비율·놓친 시간대를 표시합니다</h1><p class="lead">기존 C·T3-F와 DTW 점수는 그대로 두고, 각 항목 아래에 <b>어느 구간을 확인할 수 있었는지</b> 추가했습니다.</p><p><a href="http://127.0.0.1:8765/practice.html?model=t3f">웹 테스트 화면 열기</a> · 새로고침 후 결과 분석에서 확인할 수 있습니다.</p>
<section class="card"><h2>어떻게 읽으면 되나요?</h2><div class="grid"><img src="WEB_METRIC.png" alt="기존 손가락 굽힘 차이 수치 아래에 좌우 관측 비율과 시간대가 추가된 화면"><div><p><b>관측80%</b>는 그 시간만큼 계산에 필요한 관절 근거가 있었다는 뜻입니다.80% 올바르게 수행했다는 뜻은 아닙니다.</p><ul><li><b>촬영 전체:</b> 준비·마무리까지 포함한 비율</li><li><b>손이 처음~마지막 보인 구간:</b> 양손 중 하나라도 관측된 첫끝 구간의 비율</li><li><b>파랑:</b> 해당 항목의 계산 근거가 있었음</li><li><b>회색:</b> 확인할 근거가 없음</li><li><b>줄무늬:</b> 프레임 사이 긴 시간 공백</li></ul><div class="note">손내림·가림·검출 실패를 자동으로 수행오류 처리하지 않습니다. 볼 수 없던 구간은 판단할 수 없음으로 남깁니다.</div><p class="small">공개 ‘믿다’ REAL12 관측 기반 화면 예시. 현재 운영 DTW는 기존 관측 경로를 유지합니다. 관측비율은 원래 프레임의 시간 가중 추정이며 DTW에서 실제 비교한 대응쌍 비율과 다릅니다.</p></div></div></section>
<section class="card"><h2>놓친 구간도 초 단위로 확인합니다</h2><div class="grid"><img src="WEB_SENSOR_GAPS.png" alt="왼손과 오른손의 미관측 시작 끝 초단위 목록"><div><p>미관측 목록을 펼치면 <b>0.00~1.48초</b>처럼 시간대를 볼 수 있습니다. 전체 관측비율이 낮아도 준비·마무리 때문인지, 손이 보인 구간 안에서도 공백이 있는지 나눠 확인할 수 있습니다.</p><p>손 위치를 알 수 있어도 손가락을 충분히 볼 수 없으면 굽힘 관측비율은 더 낮을 수 있습니다. 세 항목을 따로 표시하는 이유입니다.</p><p>사용하지 않는 손은 기존 공개 참고 역할을 그대로 따릅니다. 사용자에게 보이지 않았다는 이유로 필요한 손을 제외하지 않습니다.</p><p class="small">왼쪽은 같은 공개 자료의 sensor관측을 사용한 시험 화면입니다. 운영 경로를 sensor로 바꾼 것이 아닙니다. 화면 밖에 있는 손의 실제 동작은 알 수 없습니다.</p></div></div></section>
<section class="card"><h2>공개 자료로 확인한 범위</h2><div class="scroll"><table><tr><th>검증</th><th>범위</th><th>결과</th></tr><tr><td>관측 시간 집계</td><td>100단어1199영상 × 2관측경로</td><td>14388항목·손 집계 일치</td></tr><tr><td>기존 계산 조건 일치</td><td>고정200영상24544프레임</td><td>기존 관측 가능 여부와 일치</td></tr><tr><td>빠르게·느리게 시간 변환</td><td>4796대조</td><td>관측비율 유지, 초 단위만 비례 변경</td></tr><tr><td>관측 순서 전체 반복</td><td>2398대조</td><td>시간 집계 일치, 수행오류 판정 추가 없음</td></tr><tr><td>전체손·짧은 관측 가림</td><td>5319항목 검사</td><td>누락을 관측으로 채우거나 비율을 높이지 않음</td></tr><tr><td>기존 결과 전달·화면</td><td>M0·C·T3-F / 공개 화면</td><td>점수·참고범위·비교손 유지</td></tr></table></div><p>원본에서 안정적인 손내림 근거가 있던 sensor미관측121882손프레임도 확인했습니다. 미관측은 관측정보로만 남기고 자동 오답이나 감점에 쓰지 않습니다.</p><div class="note">속도·반복 대조는 같은 공개 관절 관측의 시간 구조를 바꾼 시험입니다. 실제 빠른 촬영의 검출성능, 또는 특정 수어의 반복횟수를 바꿔도 의미가 같은지까지 검증한 것은 아닙니다.</div></section>
<section class="card"><h2>그대로 유지한 것과 남은 한계</h2><p>C·T3-F 가중치, 기존 DTW 점수·정렬·참고범위는 변경하지 않았습니다. 사용자·멘토 촬영은 이번 검증에 접근하지 않았고, 추가 학습과 인식 보조도 진행하지 않았습니다.</p><p>이번 개선은 <b>수치가 어떤 관측에 근거했는지 설명하는 기능</b>입니다. 낮은 차이가 전체 동작의 정확함을 보장하지 않는다는 점을 드러내지만, 실제로 필수 동작을 생략했는지 판별하는 기능은 아직 아닙니다.</p><p>다음은 이 표시를 함께 확인한 뒤, 공개 자료의 단계 차이를 어떻게 평가할지 검토하는 순서입니다. 추가 학습은 결과 확인 후 결정합니다.</p></section>
<details class="card"><summary>검증 근거와 실행 정보</summary><p>자체10검사,별도산술/기존함수 검산,3모델 결과전달 불변,카메라없는 브라우저 검사 PASS. 동일 Codex가 별도 검산을 수행했습니다. 보호26파일과 이전 결과 불변. 웹 서버 PID62068,카메라 비활성 확인.</p><p>최초 UI검사의 모의출처 초기화 문제는 localhost 모의출처로 수정 후PASS. in-process app import/HTTP검사 멈춤은 증거를 남기고 같은 공개자료의 직접 결과전달 검증으로 복구했습니다. 실제서버는 health/정적파일 전달을 확인했으며 사용자 촬영API로 시험하지 않았습니다.</p><a href="PUBLIC_VALIDATION.json">공개 검증</a> · <a href="INTEGRATION.json">결과전달 검증</a> · <a href="UI_QA.json">화면 검증</a> · <a href="HTTP_CHECK.json">서버 확인</a> · <a href="RESULT_KO.txt">상세 텍스트</a></details></main></html>'''
(U/'RESULT_KO.html').write_text(doc,'utf-8')
write(U/'VISUAL_REVIEW.json',{'status':'PASS','inspected':['WEB_METRIC.png','WEB_SENSOR_GAPS.png','WEB_MOBILE.png'],'reviewer':'Codex','notes':'Paired score+coverage, gap list, mobile layout legible; no runtime camera/private record was loaded.'})
write(U/'CURRENT_EXECUTION.json',{'public_session':96470,'public_exit':0,'integration_initial_session':12977,'integration_initial':'stopped stalled in-process app/TestClient; first evidence preserved','integration_recovery_exit':0,'ui_initial_session':19287,'ui_initial':'stopped failed harness; first evidence preserved','ui_recovery_exit':0,'research_processes_running':False,'web_server_pid':62068})
write(U/'STATUS.json',{'status':'COMPLETE_DISPLAY_DEPLOYED_NO_TRAINING','at':stamp})
artifacts=['FROZEN.json','IMPLEMENTATION.json','PUBLIC_VALIDATION.json','INTEGRATION.json','SELFTEST.json','UI_QA.json','HTTP_CHECK.json','VISUAL_REVIEW.json','SERVER.json','RESULT_KO.html','RESULT_KO.txt','FIRST_INTEGRATION_STALL.json','FIRST_ERROR_ui.txt']
final={'status':'PASS','at':stamp,'protected_files_unchanged':len(f['protected']),'previous_results_unchanged':True,'model_weights_unchanged':6,'display_deployed':True,'dtw_numeric_scores_changed':False,'new_training':0,'private_data_read':0,'camera_opened':False,'implementation_and_verification':'Codex, independent scalar and old-function comparison by same operator','artifact_sha256':{n:sha(U/n) for n in artifacts}}
write(U/'FINAL_AUDIT.json',final)
next_action='사용자와 새 관측비율·미관측시간 표시를 확인한다. 정상 내림/속도/반복을 수행오류로 단정하지 않는다. 다음 후보는 공개자료의 관측근거와 실제단계차이 평가계획 검토이며 추가학습·인식보조는 결과확인후. C/T3-F와 DTW수치 유지.'
state=read(OPS/'STATE.json');state.update(current_stage='WORD100_COVERAGE_DISPLAY_DEPLOYED_REVIEW',updated_at=stamp,updated_date_kst='2026-10-20',activity_state='AWAITING_USER_REVIEW',next_single_action=next_action,resume_command=next_action)
state['word100_coverage_display_20261020']={'status':'COMPLETE_DISPLAY_DEPLOYED_NO_TRAINING','at':stamp,'unit':str(U),'public_validation':v,'integration':it,'ui':qa,'protected_files':len(f['protected']),'display_deployed':True,'model_weights_unchanged':6,'dtw_numeric_scores_unchanged':True,'new_training':0,'private_data_read':0,'server':read(U/'SERVER.json'),'report':str(U/'RESULT_KO.html'),'next_single_action':next_action};write(OPS/'STATE.json',state)
title='### 2026-10-20 DTW 관측비율·미관측시간 표시 웹 적용 — 점수·모델 유지'
entry=f'''{title}
갱신 {stamp}. ops/word100_coverage_display_20261020. 사용자 승인 관측비율/놓친시간 표시와 공개대조 검증 완료. 기존DTW의수치/경로/참고범위/C/T3-F유지. 새coverage모듈+전달adapter+practice표시만수정,웹적용완료.
- 각항목좌우:전체촬영 관측시간비율,첫끝any-hand구간 비율,가장긴미관측초,시간대막대·미관측시작끝목록. 파랑관측/회색미관측/줄무늬긴촬영간격. 1.5medianΔ 반경nearest-cell시간추정,좌표보간없음. 기존19각도/몸기준손목근거기준,부분각도후보채택없음. 질의raw근거≠DTW대응쌍비교비율/정답률.
- 공개100단어TRAIN899/DEV300×legacy/sensor=2398.14388항목손시간별도검산(오차2.85e-14),TRAIN01고정200영상24544프레임기존feature조건일치. 속도0.75/1.5대조4796 비율유지·시간비례,반복2398 시간검산/오류판정추가0. 실제빠른촬영검출/수어반복허용정답검증아님.
- 기존전체원본낮은안정근거&sensor손점없음121882손프레임/2391영상손단위확인,자동오류/새필수손제외없음. 마스크5319항목검사/실근거제거4511. 자체10PASS/3모델결과전달불변/공개브라우저3항목6막대·모바일PASS. 동일Codex별도코드검산,다른agent검토아님.
- 보호26파일(가중치6·DTW계산·참고역할은행등)/과거결과해시불변. 사용자·멘토읽기0,14~18입력읽기0,학습/인식보조0. 기존server43256카메라비활성확인→server62068정상재시작,health/새정적파일전달확인. 사용자새촬영은미실행.
- 최초UI모의출처초기화오류보존→localhost수정PASS. in-process app import/TestClient멈춤12977중단/원인미확정/증거보존→같은공개자료직접adapter검증PASS·실제health/static확인. 전체사용자촬영HTTP시험주장금지. 숫자조건/모델변경없음. RESULT_KO.html/txt,PUBLIC_VALIDATION,INTEGRATION,UI_QA,HTTP_CHECK,FINAL_AUDIT.

'''
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=ctx.read_text('utf-8-sig');assert title not in txt
txt=txt.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+entry,1).replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n'+title+'\n'+next_action+'\n\n',1);ctx.write_text(txt,'utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 차이 수치와 관측 근거를 함께 설명하는 DTW 웹 표시. 마지막 완료: 관측 시간 비율·놓친 시간대 표시와 공개검증,웹반영.

- 단위 `{U}`. 보고 RESULT_KO.html/txt,PUBLIC_VALIDATION,INTEGRATION,UI_QA,HTTP_CHECK,FINAL_AUDIT.
- 신규word100_observation_coverage.py / observation-display.js. 기존web_analysis+practicejs/html/css만허용수정. C/T3-F6가중치·DTW/역할/은행등26파일보존. 원래수치/정렬/참고범위불변.
- 각항목좌우전체/첫끝anyhand관측비율·가장긴미관측초·시간막대·구간목록. 실제timestamp nearest-cell시간추정,1.5medianΔ반경밖공백불명,보간없음. 기존19각도gate그대로/부분각도후보미적용. 미관측=수행오류아님.
- 공개1199×2=2398,14388항목손집계·200영상24544프레임기존마스크일치. 속도4796대조비율동일/시간비례,합성반복2398대조오류판정추가없음.5319가림검사. 실제빠른촬영검출/언어적반복허용검증아님.
- 공개원본손내림근거&sensor누락121882손프레임확인,자동오답/새손제외없음. 자체10/별도검산/3모델전달동일/UI6막대모바일PASS. 같은Codex별도검산이며다른agent아님.
- 사용자멘토자료검증미접근,14~18미접근,학습/인식보조0. research작업완료. 웹server62068(일반권한실행)활성,기존43256카메라비활성확인후재시작. 카메라열지않음,health/static검사만. http://127.0.0.1:8765/practice.html?model=t3f 새로고침.
- 최초UI모의출처초기화문제→localhost복구PASS. in-process app/TestClient멈춤12977중단·최초증거보존,동일공개자료direct adapter검산완료. 실제사용자촬영HTTP검증미실행. 원인미확정항목을모델실패로보고하지않음.

다음 한 작업: {next_action}
''','utf-8')
assert read(OPS/'STATE.json')['current_stage']=='WORD100_COVERAGE_DISPLAY_DEPLOYED_REVIEW'
print('FINALIZED',stamp)
