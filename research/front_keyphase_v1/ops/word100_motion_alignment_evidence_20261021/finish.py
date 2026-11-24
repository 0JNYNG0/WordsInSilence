import json,re
import study as d
U,O,ROOT=d.U,d.O,d.ROOT
f=d.read(U/'FROZEN.json');v=d.read(U/'VERIFY.json');a=d.read(U/'ANALYSIS.json');rc=d.read(U/'REPORT_CHECK.json')
assert v['status']=='PASS' and rc['status']=='PASS' and not list(U.glob('*RUNNING.json'))
d.prev.prev.pipe.check(f['pins']);d.prev.prev.pipe.check(f['protected'])
assert not (U/'FIRST_ERROR.txt').exists() and not (U/'VERIFY_FIRST_ERROR.txt').exists()
roots=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in roots:assert d.sha(p)==d.sha(U/'before'/p.name),'Concurrent root edit: '+str(p)
doc=(U/'RESULT_KO.html').read_text('utf8');links=re.findall(r'(?:src|href)="([^"]+)"',doc)
for link in links:assert (U/link).is_file()
at=d.now()
next_action=('사용자와 관측/변화/안정손내림/근거부족 분리 및 정렬구간 경고 결과를 공동 확인한다. '
 '누락 시간표는 진단용 보존, 여러순간→한장면 경고는 정상속도에도 과다하여 자동감점·오류판정 미채택. '
 '다음 제안은 같은 공개 원본의 알려진 시간변환으로 대응 정답이 있는 정상속도 대조를 먼저 만들고, 확실히 관측된 모양 변화의 전후 순서 보존을 진단하는 공통 계획. '
 '단순 경로압축횟수나5명합의만으로 의미오류를 확정하지 않는다. 참고범위 유지·정상한손/손내림/교차 보존. '
 '결과공동확인 전 추가학습/운영적용 없음. C·T3-F·웹 유지, 사용자/멘토 촬영 학습·튜닝·선택 금지.')
entry=f'''### 2026-10-21 손 관측·변화·누락 시간 분리 — 진단 검증완료·자동 오류 판정 미채택
- 단위 `{U.name}`. 사용자 “우선 그렇게 진행해줘” 승인 후 별도 연구용 구현/검산/그래프 보고완료. 추가학습/운영교체 없음.
- 공개 TRAIN 수어·후원·멀다3단어×9명27영상. 기존3조건648쌍 재사용+누락648쌍+정상변형1080쌍=2376진단. 새100단어 인식검증 아님. 기존80정상/교차 입력보존근거 재사용.
- 손관측/위치변화/각도변화/긍정적안정손내림/근거부족을 분리. 두검출값의범위초과는기하근거이지보정된신뢰확률/의미정답아님. 이전active→rest 신규25쌍-손프레임 모두보임·변화근거없음(위치0/각도0/positive-rest0). 손관측을의미활동으로오해하지않도록진단개선,25개독립오류수아님.
- 원래촬영시간의근거부족시간표를경로와별도보존. 짧은3frame누락27조건1215각도셀 모두표시. 전체손누락54조건중필수45조건30195핵심각도셀표시·선택손9조건필수화없음. 모든참고expected분모/기존좌표/좌우/각도상태/저장3조건경로불변. 기존점수·커버리지반등계산을고친것이아니라누락을별도노출한것.
- 5/8참고동일손각도시간경고합의: 원래14/27,길이0.75배19/27,1.5배27/27,0.5초정지19/27. 속도/정지정상변형81중65경고,따라서자동오류판정미채택. 반복2/3각14/27은참고/경로도함께복사한보존검사이지반복횟수정답검증아님.
- 알려진후원REAL04 f61문제는경고남음;f64관측근거0각도는15부족으로별도표시. REAL06 f28문제는개별참고경고11각도이나5명합의0이라미포착. 경고감소/합의만으로정렬정확성향상주장불가. 분류정확도/실제시간정렬향상미측정.
- verify.py 별도acos/쌍별구간겹침수식으로98970프레임-손-각도셀·2376저장결과/원본경로/참고분모검산PASS,의미있는7불변조건PASS. 동일Codex검산이며별도전문가정답검증아님. 새속도DTW는저장경로단조성/진단검산이지독립재최적화아님.
- 보호32핀및전체기존입력/결과핀유지. FROZEN/candidate/study/ORIGINAL/MISSING/CONTROLS/VERIFY/ANALYSIS/RESULT_KO.html·txt/FINAL_AUDIT/SYNC_CHECK. 그래프5개·기존원본시트1개시각검사,HTML정적10링크검사·브라우저렌더미실행. 모든실행종료. 사용자자료접근/학습/분류추론/운영변경0.
'''
audit=dict(status='DIAGNOSTIC_VERIFIED_AUTOMATIC_ERROR_NOT_ADOPTED',at=at,implementer='Codex',verifier=v['verifier'],
 authorization='Latest user 우선 그렇게 진행해줘; prior plan motion vs visibility and local alignment substitution diagnosis; no training until review.',
 scope=dict(words=3,people=9,clips=27,original_pairs=648,missing_pairs=648,control_pairs=1080),
 result=dict(old_flags=a['old_visibility_flags'],missing=v['missing'],original=v['original_summary'],controls=v['control_summary'],known=a['known_transition_checks']),
 verification=str(U/'VERIFY.json'),protected=len(f['protected']),training=0,classifier_inference=0,user_data_access=0,operational_changes=False,
 visualization=rc,running=None,first_failure=None,next=next_action,
 commands_completed=['study.py freeze:0','study.py original:0','study.py missing:0','study.py controls:0','verify.py:0','analyze.py:0','report.py:0'],
 artifacts={str(p):d.sha(p) for p in U.iterdir() if p.is_file()})
d.write(U/'FINAL_AUDIT.json',audit)
p=roots[0];txt=p.read_text('utf-8-sig');mark='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';assert mark in txt
txt=txt.replace(mark,mark+'\n\n'+entry,1);mark='## 9. 다음 단일 작업';assert mark in txt
txt=txt.replace(mark,mark+'\n\n### 2026-10-21 관측·변화·누락 시간 진단 결과 공동 검토\n'+next_action+'\n',1);p.write_text(txt,encoding='utf8')
state=d.read(roots[2]);state.update(updated_date_kst=at,current_stage='WORD100_MOTION_ALIGNMENT_DIAGNOSTIC_VERIFIED_NOT_ADOPTED',next_single_action=next_action)
state[U.name]=dict(status=audit['status'],at=at,scope=audit['scope'],results=audit['result'],training=0,user_data_access=0,operational_changes=False,protected=32,
 report=str(U/'RESULT_KO.html'),audit=str(U/'FINAL_AUDIT.json'),verify=str(U/'VERIFY.json'),next=next_action,running=None)
roots[2].write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
roots[1].write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 손의 관측과 변화 근거·안정손내림을 분리하고, 누락 시간이 정렬에 가려지지 않도록 진단한다.
마지막 완료: `{U.name}` 연구용 구현/검증/그래프보고. 짧은누락27조건의시간표보존. 경고는정상속도에도과다하여자동오류판정미채택. C·T3-F·운영웹유지.
다음 한 작업: {next_action}

{entry}

담당 Codex 구현+별도수식검산. 학습/분류추론/사용자촬영접근/운영교체0. 모든실행종료, RUNNING없음.
freeze/original/missing/controls/verify/analyze/report 종료0. 완료실험 재실행·문턱조정 금지. 원본/W:/sources/과거동결자료 읽기전용.
주근거 RESULT_KO.html, VERIFY, ANALYSIS, FROZEN, FINAL_AUDIT, SYNC_CHECK. 구현용 candidate.py와 study.py는실행전동결해시유지.
정상대조가있어도임의의1→다수연결은정상속도정렬에서도발생함. 다음에는대응시간을아는공개합성대조부터검토할것. 27영상외범용효과미확인.
추가학습은 결과를 함께 확인한 다음 진행. 사용자테스터자료는학습·튜닝·모델선택금지.
''',encoding='utf8')
d.prev.prev.pipe.check(f['protected'])
for p in roots:assert U.name in p.read_text('utf8')
assert d.read(roots[2])[U.name]['results']==audit['result']
d.write(U/'SYNC_CHECK.json',dict(status='PASS',at=d.now(),roots={str(p):d.sha(p) for p in roots},protected=32,running=None))
print('FINISHED',U/'RESULT_KO.html')
