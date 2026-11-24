from pathlib import Path
import json,hashlib,datetime,runpy
U=Path(__file__).resolve().parent;O=U.parent;R=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
assert not (U/'FINAL_VERIFY.json').exists()
files=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
assert all(sha(p)==sha(U/'before'/p.name) for p in files),'Concurrent root change'
assert read(U/'INDEPENDENT_VERIFY.json')['status']=='PASS_SAVED_CONTRACT_VERIFICATION'
assert read(U/'CONTRACT_VERIFY.json')['status']==read(U/'UI_VERIFY.json')['status']=='PASS'
f=read(U/'FROZEN.json');b=read(U/'BUNDLE.json')
assert all(sha(p)==h for p,h in b['source_pins'].items())
assert all(sha(p)==h for p,h in f['protected'].items())
bp=U/'build_report.py';t=bp.read_text(encoding='utf-8');t=t.replace('전체안보임10묶음도비교창을줄이지않는다.','한 특징의 13표본 전체가 보류된10묶음도 비교창을 줄이지 않는다. 손 전체가 물리적으로 안 보인다는 뜻은 아니다.')
bp.write_text(t,encoding='utf-8');runpy.run_path(str(bp),run_name='__main__')
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
summary='''현재 방법의 방향·한계와 이후 4단계를 정리하고, 첫 후속인 공통 손모양 특징의 시간 기록 연결을 완료했다. 기존 공개12표현·기록ID3개의36개 고정창,468프레임·936특징값을 원래 단어/사람 분리16/8/8/4와 연결했다. 745확인/191보류를 그대로 보존하고 같은값 관측82묶음/미관측32묶음, 변화가능12구간/같은끝값사이미관측8구간을 기록했다. 앞5/뒤1/전체특징보류10묶음도 명시하며 null때 비교창을 줄이지 않는다. 두특징0/0은 주먹/같은전체손모양을 뜻하지 않고1/1모순은 실제0건. 동일원본38이미지·76항목은54확인일치/13한쪽보류/9양쪽보류/0직접충돌이며 양출처를 덮어쓰지 않았다. AI 관찰 초안의 연결·보존 검증이지 정답이나 성능 향상 증명이 아니다. 읽기전용 관측 어댑터에 단어/Gloss/평가정답을 제외하고, 모순시 원본을 보존하며 비교값만 보류한다. 별도코드 검산18항목 및 자체936결측/72전체보류/288시간변환계약·짧은변화·순서검사 통과. 원본RGB468개/고유원본프레임538키의분리·출처와보호32핀확인. 연구화면72구간/특징상태desktop/mobile검증. 학습0/새검출0/새DTW0/운영변경0/사용자멘토접근0.'''
next_action='''다음 단일 작업은 작은 공통 특징 관측모델 비교에 필요한 정답과 평가 묶음의 신뢰도를 확보하는 것이다. 현재 연구창과2특징만으로 전체수형·언어적필수동작을 안다고 하지 않는다. 기존 공개 자료에서 이미지와 매칭된 독립적인 물리적 손특징 주석을 재사용할 수 있는지 확인하고, 현재 AI초안·관절자동검출과 정답의 출처를 구분한다. 적합한 기존 주석이 없으면 무한 검출기/임계값 실험 대신 실제 손전체이동·회전/손가락변화/모호구간,각 특징양음성을 포함한 작은 확인묶음을 고정하여 필요한 사람 검토 범위를 최소화한다. 출처·관찰범위·분리와정답이확보된특징만 현재RGB/관절/RGB+시간의동일조건비교로 연결한다. 현재묶음을 독립사람정답으로 승격하거나 순서정보를 지우지 않는다. 통과조건은 오답과확인가능범위를함께개선,보류만늘려오류수감소는미채택. 이후3단계는동일시점·여러참고DTW에서정상동작의잘못된교정/명확한오류놓침/관측부족을별도검증,4단계는고정후웹별도후보실사용평가다. 이번단위는계획과1단계연결완료이며2~4모델학습·DTW운영·웹교체미실행. C·T3-F보존,사용자멘토학습튜닝선택금지. 활성실행없음. 승인된일반준비·검증재승인은불필요하고새자료다운로드/외부전송/운영교체는승인범위를확인한다.'''
title='2026-10-27 공통 손모양 특징 시간묶음 연결 및 이후 계획 검증 완료'
evidence=f'research/front_keyphase_v1/ops/{U.name}/RESULT_KO.html'
ctx=files[0].read_text(encoding='utf-8-sig')
for header,text in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
 assert ctx.count(header)==1
 ctx=ctx.replace(header,header+'\n\n### '+title+'\n'+at+'\n'+text+'\n근거: '+evidence+', BUNDLE.json, INDEPENDENT_VERIFY.json, CONTRACT_VERIFY.json, NEXT_PLAN.json, UI_VERIFY.json.\n',1)
files[0].write_text(ctx,encoding='utf-8')
state=read(files[2]);state.update(updated_date_kst=at,current_stage='SHARED_SHAPE_SEQUENCE_BUNDLE_COMPLETE_LABEL_VALIDATION_NEXT',next_single_action=next_action)
state[U.name]={'status':'COMPLETE_READ_ONLY_FEATURE_SEQUENCE_AND_PLAN_NOT_NEW_MODEL_GAIN','at':at,'summary':summary,'next':next_action,'evidence':evidence,
 'counts':b['counts'],'exact_static_join_counts':b['exact_static_join']['counts'],'roles':b['roles'],
 'implementation':'Codex root: frozen representation, join, read-only adapter, contracts, report/UI/docs.',
 'verification':'linguistic_eval_review separate saved-contract reconstruction and pixel/source audit; root metamorphic contracts and browser QA. Not independent human truth.',
 'training_ready':False,'training_runs':0,'operating_changes':0,'user_mentor_access':0,'active_execution':None}
dump(files[2],state)
files[1].write_text(f'''# WordsInSilence 최신 인수인계

{at}

## 현재 목표와 마지막 완료
{summary}

## 다음 한 작업
{next_action}

## 중요한 해석
- 현재의2특징은 검지만폄/네손가락폄이다. 둘다0이라고 주먹은아니며, 같은0/0이이어져도 전체손모양유지라고 하지 않는다. 기존TCN입력이2특징뿐이라는 뜻도아니다.
- 기록된동일특징run은 관측표본묶음. 프레임사이연속상태·정확한동작단계·교육적정오아님. 각창13프레임만,전체수어단어범위로확대금지.
- 전체보류10은 한특징13표본보류10묶음이며 물리적으로손전체가사라졌다는판정아님.
- 동일시점76쌍은중앙36×2뿐아니라T14의다른두시점4칸도포함. one_unknown은관찰확실성차이로유지. 두초안의값은원본그대로보존.
- 모든원본은기존TCN TRAIN및이미검토한영상이다. 현재작업으로C/T3-F인식률·DTW교정정확도향상주장불가.
- 루트자체검사:936값복원,936단일결측,72전체특징결측,288타임스케일,짧은pulse/순서/모순/단어메타데이터독립성. 모두저장계약시험으로실제속도인식률과다름.
- linguistic_eval_review가producerimport없이18계약검사,468RGB/38중복픽셀/538고유source프레임의역할보존,86핀·보호32검산. 같은AI별도코드이며독립사람정답아님.
- 검산기최초2스키마오류(옛event명,source.sha256경로)는INDEPENDENT_VERIFY.json prior_attempts보존. 원본/BUNDLE변경없음.
- 앞단하드차단후보미채택유지. 220개기존일치보류를개선으로보고하지않고기준을점수에맞춰높이지않음.
- 코드/보고/연구어댑터만이단위에있다. 운영TCN/DTW/웹·원본·사용자자료변경없음. 학습/새검출/새DTW/개인자료접근/다운로드/Git0. 실행완료,서버·자동화새실행없음.

## 근거
- `{evidence}`: 현재방향/한계,4단계와통과기준,원본특징타임라인.
- `{U.name}/BUNDLE.json`: 고정36창·936특징·동일시점대조.
- `{U.name}/OBSERVATION_EXPORT.json`: 단어힌트없는읽기전용초안인터페이스. 자동예측모델아님.
- `{U.name}/INDEPENDENT_VERIFY.json`: 별도계약·픽셀·분리검산.
- `{U.name}/CONTRACT_VERIFY.json`, `UI_VERIFY.json`: 자체계약/브라우저검사.
- `{U.name}/NEXT_PLAN.json`: 이후4단계및미채택조건.

이번단위완료와실제프로젝트성능개선을혼동하지않는다. 다음은신뢰가능특징정답확보를통해작은모델비교로넘기는단계다.
''',encoding='utf-8')
assert all(sha(p)==h for p,h in f['protected'].items())
assert all(U.name in p.read_text(encoding='utf-8-sig') for p in files)
dump(U/'FINAL_VERIFY.json',{'status':'PASS','at':at,'docs':{str(p):sha(p) for p in files},'bundle_sha256':sha(U/'BUNDLE.json'),'report_sha256':sha(U/'RESULT_KO.html'),'protected_pins':len(f['protected']),'implemented_stage':1,'remaining_stages':[2,3,4],'model_gain_established':False,'active_execution':None})
print('Complete: stage1 linked and verified; staged plan recorded; C/T3-F unchanged.')
