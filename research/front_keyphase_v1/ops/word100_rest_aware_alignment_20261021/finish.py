import json,re
from pathlib import Path
import study as d
U,O,ROOT=d.U,d.O,d.ROOT
f=d.read(U/'FROZEN.json');r=d.read(U/'RESULT.json');v=d.read(U/'VERIFY.json');s=d.read(U/'STRESS.json');a=d.read(U/'DIAGNOSIS.json')
d.prev.pipe.check(f['pins']);d.prev.pipe.check(f['protected'])
assert v['status']=='PASS' and len(s['whole'])==54 and len(s['short'])==27
assert not (U/'RUNNING.json').exists()
roots=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in roots:assert d.sha(p)==d.sha(U/'before'/p.name),'Concurrent root edit: '+str(p)
doc=(U/'RESULT_KO.html').read_text('utf8')
links=re.findall(r'(?:src|href)="([^"]+)"',doc)
for p in links:assert (U/p).is_file()
assert '<script' not in doc
at=d.now()
next_action=('사용자와 손별 손내림 시간정렬 세 조건 결과를 검토한다. 새 정렬은 동작시점 개선 미입증·단계전환 회귀로 미채택. '
 '다음 제안은 범위내관측과 실제활동/안정손내림/모름의 진단을 분리하고, 짧은 전환·누락이 주변 장면으로 대체되는 정렬을 표시하는 공통 후보 계획. '
 '참고시간분모는 고정하고 정상속도·반복·교차 대조를 함께 유지한다. 결과로 문턱을 완화하지 않는다. '
 'C·T3-F·운영웹 유지, 사용자·멘토자료 학습/튜닝/선택 금지. 추가학습은 결과 공동확인 후 별도 진행.')
entry=f'''### 2026-10-21 손별 손내림 시간정렬 세 조건 검증 — 계산 소폭 개선·정렬 회귀로 미채택
- 단위 `{U.name}`. 사용자 설명확인 후 검증 재개 승인. 저장된 세 조건 결과를 재실행 없이 검산·원본검토했다. 추가학습/운영교체 없음.
- 공개 TRAIN3단어(수어/후원/멀다)×9명27영상,자기사람제외8참고216쌍,3조건648결과. 기존158입력/80정상·교차대조는 이전 보존근거 재사용이지 이번100단어 새 DTW시험 아님.
- 기존373→계산만378→정렬까지378/675 일부비교각도항목(정답률 아님). 후원27→32/270,수어238/270·멀다108/135유지. 기존373항목손실0,297부족 지속. 증가5는 계산효과이며 정렬추가이득0. 공통표시10각도중앙값변화 최대1.31도는정확성증거아님.
- 경로216중8후원쌍변경,208동일. 원본 REAL04→REAL09 f58→f40은 펴진단계 더유사하나 f61/f64→f42는 이미굽힌손을 펴진참고에 맞추는 회귀(기존f48더유사). 정상전체시점개선미입증·새정렬운영미채택.
- 자동active는범위내점 존재이지수어활동정답아님. 원본에서내려놓은왼손도경계안관측됨. 새25쌍-손프레임경고를25실제오류로집계하지않음. 26원본장면묶음78패널/9시트검토;독립수어전문가/전체시점정답주석아님.
- 전수648결과에서참고각도시간범위고정. 전체손누락54조건432재정렬에서도범위/가중치분모유지·누락손비교가능0. 짧은핵심3저장frame누락27조건216경로중91변경,3240쌍-각도중관측비율상승39/비교가능전환14/반대전환108. 주변장면재연결위험잔존;실제필수동작생략정오시험아님.
- 별도수식3,215,596비용셀·최적경로비용·648쌍분모/커버리지·27영상요약검산PASS. 최초검산float32평균/float64나눗셈차이는검산코드만정정,원오류/정정근거보존. 후보/문턱/입력/저장결과수정없음.
- 보호32핀유지,좌표/좌우/각도상태불변. 학습/분류추론/사용자자료접근/운영변경0. FROZEN/RESULT/VERIFY/STRESS/DIAGNOSIS/VISUAL_REVIEW/RESULT_KO.html·txt/FINAL_AUDIT. 모든실행종료. HTML정적링크25·원본9시트·그래프시각검사완료,브라우저렌더미실행.
'''
audit=dict(status='VERIFIED_MIXED_TIMING_NOT_ADOPTED',at=at,implementer='Codex',
 verifier='Same Codex independent arithmetic and raw-source visual review; no independent semantic ground truth',
 authorization='Latest user: validate saved3conditions, normalrest/crossing and fixedrange; no training until joint review',
 comparison=r['comparison'],verify=v,stress=dict(whole_conditions=54,whole_paths=432,whole_absent_hand_ready=0,
 short_conditions=27,short_paths=216,short_changed=91,short_coverage_increases=39,short_ready_increases=14,short_ready_losses=108),
 path_changes=8,training=0,classifier_inference=0,user_data_access=0,operational_changes=False,
 protected=len(f['protected']),running=None,next=next_action,
 static_links=len(links),visual_review=dict(raw_sheets=9,triplets=26,availability_plot=True,path_plot_example=True,browser_render=False),
 first_failure='VERIFY_FIRST_ERROR.txt; arithmetic precision corrected verifier only, VERIFIER_CORRECTION.json',
 artifacts={str(p):d.sha(p) for p in U.iterdir() if p.is_file()})
d.write(U/'FINAL_AUDIT.json',audit)
p=roots[0];txt=p.read_text('utf-8-sig')
mark='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';assert mark in txt;txt=txt.replace(mark,mark+'\n\n'+entry,1)
mark='## 9. 다음 단일 작업';assert mark in txt;txt=txt.replace(mark,mark+'\n\n### 2026-10-21 손내림 시간정렬 결과 공동 검토\n'+next_action+'\n',1)
p.write_text(txt,encoding='utf8')
state=d.read(roots[2]);state.update(updated_date_kst=at,current_stage='WORD100_REST_ALIGNMENT_VERIFIED_MIXED_NOT_ADOPTED',next_single_action=next_action)
state[U.name]=dict(status=audit['status'],at=at,scope=dict(words=3,people=9,clips=27,pairs=216,conditions=3),
 results=r['comparison'],path_changes=8,unchanged_paths=208,training=0,user_data_access=0,operational_changes=False,protected=32,
 short_missing=dict(coverage_increase_rows=39,ready_increase_rows=14,ready_lost_rows=108),
 report=str(U/'RESULT_KO.html'),audit=str(U/'FINAL_AUDIT.json'),next=next_action,running=None)
roots[2].write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
roots[1].write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 손별 손내림을 DTW 정렬에 반영해 실제 동작시점이 개선되는지, 손누락에도 참고범위가 유지되는지 검증.
마지막 완료: `{U.name}` 세조건 검증완료. 일부비교373→378→378/675. 새정렬추가이득없음·정상단계전환회귀로미채택. 기존C·T3-F·웹유지.
다음 한 작업: {next_action}

{entry}

학습/사용자촬영접근/운영교체0. 모든 실행 종료, RUNNING 없음.
study.py freeze/run은 앞선 승인턴 종료. 검증은 사용자가 설명 확인 후 재개 승인해 완료. verify.py verify는 첫 float32 검산불일치 보존 후 계산자료형만 정정하여종료0; stress/analyze_visuals/report도종료0. 저장실험결과재실행없음.
근거는단위 RESULT_KO.html,VERIFY,STRESS,DIAGNOSIS,VISUAL_REVIEW,VERIFIER_CORRECTION,FINAL_AUDIT,SYNC_CHECK. 이전동결자료/원본/W:/sources는읽기전용.
추가학습은 결과를 함께 확인한 다음. 사용자테스터자료는학습·튜닝·모델선택금지.
''',encoding='utf8')
d.prev.pipe.check(f['protected'])
for p in roots:assert U.name in p.read_text('utf8')
assert d.read(roots[2])[U.name]['results']==r['comparison']
d.write(U/'SYNC_CHECK.json',dict(status='PASS',roots={str(p):d.sha(p) for p in roots},protected=32,running=None))
print('FINISHED',U/'RESULT_KO.html')
