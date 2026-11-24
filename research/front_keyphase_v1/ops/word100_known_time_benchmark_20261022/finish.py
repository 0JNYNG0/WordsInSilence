import json,re
import study as d
U,O,ROOT=d.U,d.O,d.ROOT
f=d.read(U/'FROZEN.json');v=d.read(U/'VERIFY.json');a=d.read(U/'ANALYSIS.json');rc=d.read(U/'REPORT_CHECK.json')
assert v['status']=='PASS' and rc['status']=='PASS' and not (U/'RUNNING.json').exists()
assert not (U/'FIRST_ERROR.txt').exists() and not (U/'VERIFY_FIRST_ERROR.txt').exists()
d.prev.prev.prev.pipe.check(f['pins']);d.prev.prev.prev.pipe.check(f['protected'])
roots=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in roots:assert d.sha(p)==d.sha(U/'before'/p.name),'Concurrent root edit: '+str(p)
doc=(U/'RESULT_KO.html').read_text('utf8');links=re.findall(r'(?:src|href)="([^"]+)"',doc)
for link in links:assert (U/link).is_file()
at=d.now()
next_action=('사용자와 알려진 시간변환 시험 결과를 공동 검토한다. 같은원본 정상시간변형에서 기존DTW는시점오차작음·이전plateau경고과다였고, '
 '주변각도범위 후보가 경고81/108→1/108로줄였으나 짧은구간16/81·긴구간48/81만지속경고하여자동감점/맞음판정미채택. '
 '다음 제안은 확실한 모양변화·손위치/팔궤적변화·관측부족을 분리하고, 다른사람 공개참고에서 정상동작의경고/보류가과하지않은지 공통검증하는 계획. '
 '단조변화의중간생략은빠른표본과구분불가할수있어억지판정금지. 특정단어예외·결과보고문턱튜닝금지. '
 'C·T3-F·웹유지,사용자/멘토촬영학습·튜닝·선택금지. 추가학습은결과공동확인후.')
entry=f'''### 2026-10-22 알려진 시간변환·구간제거 검증 — 정상 경고 감소·누락 판단은 불완전
- 단위 `{U.name}`. 사용자 같은공개영상 시간변환/누락 구분 시험 승인. 공개 TRAIN3단어(수어·후원·멀다)×9명27원본관절입력,원본자신참고. 새RGB추출/새사람인식/100단어검증 아님.
- 실행전 FROZEN297조건: identity27+정상속도/비균일속도/정지108+주석core25/50/75%위치의3frame·20%길이제거각81. 모두저장·제외0. 변환원본index/삭제정답은평가만접근,진단후보에는미제공.
- 기존DTW band.2/가중치/입력보존. 새후보는연결된각도와바로이웃관측의min/max범위를참고와비교,동일각도3연속frame차이만지속경고. 연구용이며순서/실제자세복원/의미정오보장아님. 문턱사후조정없음.
- 정상시간변형108에서기존경고81→새1. fast075 27→0/27,slow15 27→0/27,pace27→1/27,pause0→0/27. identity27은모두0. 변화근거구간의기존DTW평균시점오차.16~.26원본frame(정지0),측정정상변화시점모두2frame이내. 경로개선/TCN정확도향상결과아님.
- 구간제거지속경고3frame16/81,core20%48/81. 해당경고는제거시간안에있고보존시간번짐0. 나머지짧은65=3연속미만48/비교가능부범위내12/전부비교근거없음5,긴33=29/2/2. 범위내도부분미관측가능. 필수수어동작누락정답률아님.
- 정상오경고후원REAL09왼손각도index3 f52~54: 시점정렬근접하나합성재표본이짧은각도변화를완화,약6.90~13.85도범위차이. 사용자잘못아님. 단조움직임중간제거와빠른표본은구분불가가능,별도단순입력한계검사보존.
- 참고expected시간/손역할/양성손내림 모든297조건동일. 좌우/관절/기존모델원본유지. 새누락경고는각도기반으로팔/위치누락판정아님. 근거부족을맞음으로처리하지않으며자동감점/운영적용미채택.
- verify.py 독립DP최적누적비용·변환/마스크·범위/경고·시점검산297PASS,6단순입력검사PASS. 기존비용산식검증재사용. 동일Codex별도코드검산/의미전문가정답아님. 그래프4·원본3시트9장면시각검토,정적HTML12링크검사,브라우저렌더미실행.
- 보호32핀및기존입력/결과핀유지. 학습/분류추론/사용자멘토자료접근/운영교체0. 모든실행종료. FROZEN/candidate/study/RESULT/VERIFY/ANALYSIS/CASE_REVIEW/RESULT_KO.html·txt/FINAL_AUDIT/SYNC_CHECK.
'''
summary=dict(normal_time_variants=108,normal_previous_warning=81,normal_local_warning=1,identity=27,identity_warning=0,
 short=dict(n=81,localized=16,remaining=a['groups']['skip3']['counts']),long=dict(n=81,localized=48,remaining=a['groups']['skip20pct']['counts']),
 timing_conditions=v['conditions'],recognition_accuracy_measured=False,same_source_only=True,semantic_essentiality_known=False)
audit=dict(status='KNOWN_TIME_VERIFIED_RESEARCH_ONLY_NOT_DEPLOYED',at=at,implementer='Codex',verifier=v['verifier'],
 scope=dict(words=3,people=9,source_clips=27,cases=297),results=summary,verification=str(U/'VERIFY.json'),
 protected=len(f['protected']),training=0,classifier_inference=0,user_data_access=0,operational_changes=False,running=None,next=next_action,
 execution=['study.py freeze exit0','study.py run exit0 (exec18868)','verify.py exit0 (exec87176)','analyze.py exit0','report.py exit0'],
 report_check=rc,first_failure=None,artifacts={str(p):d.sha(p) for p in U.iterdir() if p.is_file()})
d.write(U/'FINAL_AUDIT.json',audit)
p=roots[0];t=p.read_text('utf-8-sig');mark='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';assert mark in t;t=t.replace(mark,mark+'\n\n'+entry,1)
mark='## 9. 다음 단일 작업';assert mark in t;t=t.replace(mark,mark+'\n\n### 2026-10-22 알려진 시간변환 시험 결과 공동 검토\n'+next_action+'\n',1);p.write_text(t,encoding='utf8')
state=d.read(roots[2]);state.update(updated_date_kst=at,current_stage='WORD100_KNOWN_TIME_VERIFIED_RESEARCH_ONLY',next_single_action=next_action)
state[U.name]=dict(status=audit['status'],at=at,scope=audit['scope'],results=summary,training=0,user_data_access=0,operational_changes=False,protected=32,report=str(U/'RESULT_KO.html'),audit=str(U/'FINAL_AUDIT.json'),verify=str(U/'VERIFY.json'),next=next_action,running=None)
roots[2].write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
roots[1].write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 같은 공개 원본의 대응 시점을 아는 시험으로 정상 속도 차이와 구간 누락 진단을 구분한다.
마지막 완료: `{U.name}` 297조건 동결실험·독립검산·그래프보고 완료. 정상경고81/108→1/108,제거구간3frame16/81·20%48/81. 연구용으로만보존하며C·T3-F·웹유지.
다음 한 작업: {next_action}

{entry}

작업담당 Codex구현/별도코드수식검산. FROZEN조건/코드해시실행전고정,사후튜닝없음. 학습/분류추론/사용자촬영접근/운영교체0.
freeze/run/verify/analyze/report 모두종료0, RUNNING없음. 완료결과 재실행금지. 원본/W:/sources/과거동결결과 읽기전용.
주근거 RESULT_KO.html, VERIFY, ANALYSIS, CASE_REVIEW, FROZEN, FINAL_AUDIT, SYNC_CHECK.
같은영상좌표재표본으로검출노이즈도공유하는쉬운시험임을반드시명시. 의미상필수동작누락정답률/새사람인식률로확대금지.
추가학습은결과공동확인후. 사용자테스터자료는학습·튜닝·모델선택금지.
''',encoding='utf8')
d.prev.prev.prev.pipe.check(f['protected'])
for p in roots:assert U.name in p.read_text('utf8')
assert d.read(roots[2])[U.name]['results']==summary
d.write(U/'SYNC_CHECK.json',dict(status='PASS',at=d.now(),roots={str(p):d.sha(p) for p in roots},protected=32,running=None))
print('FINISHED',U/'RESULT_KO.html')
