from pathlib import Path
import json,hashlib,shutil,datetime,re
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
assert (ROOT/'AGENTS.md').is_file()
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();unit=U.name
ver=read(U/'FINAL_VERIFICATION.json');assert ver['status']=='PASS'
for p,h in ver['pins'].items():assert sha(p)==h
statepath=ROOT/'research/front_keyphase_v1/ops/STATE.json';state=read(statepath)
assert state['current_stage']=='SIGNREP_GPU_RUNTIME_VERIFIED_KSL_TRAINING_NOT_STARTED','Do not overwrite concurrent newer state'
before=U/'before';before.mkdir(exist_ok=False)
for path in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',statepath]:shutil.copy2(path,before/path.name)
status='SIGNREP_TEMPORAL_HEAD_SIX_RUNS_COMPLETE_MIXED_RESULTS_NOT_DEPLOYED'
summary='공개12표현/기록ID3명/36영상(학습16/평가20),4351프레임1932창. 108정지장면+36개13프레임구간RGB재검토, 불확실17항목추가보류, 확인1112셀(학습461/평가651). Codex임시물리주석이며독립사람/교육적정답아님. SignRep본체고정+5항목시간판정부, 관절기준과영상추가각3seed/160epoch총6회완료. 배운표현새사람균형정확도91.74→97.06%(+5.32%p), 새표현기존사람62.60→60.38%, 새표현새사람66.68→61.36%, 전체73.24→71.42%(-1.82%p). 변화7구간전후정답1~3→4~5개. 전체회귀로운영채택보류. GPU원본RGB+저장관절+판정부중앙3.19초(2.96~3.67),관절추출/DTW/웹제외,초기준비별도3.20초. 입력36개검산+저장결과427검사PASS. C/T3F/웹/DTW수정0,개인자료접근0,다운로드0. 실행모두종료.'
next_action='이번6회고정결과를유지하고, 기존공개학습자료에서 같은손모양/위치관계가다른표현에도분포하도록 주석지원범위를보강한다. 특히왼손모양·얼굴/양손관계·짧은변화와이동방향시작끝을분리. 필수단계/반복/교육적정오를기존Codex물리주석이나단어설명만으로정답승격금지. 이번20영상은이미살펴본개발자료로유지하고새평가대상확보후비교조건동결. 평가점수로현재후보재튜닝·완료결과재학습금지. 새모델순회/다운로드/개인자료/운영교체없음. 사용자에게혼합결과보고후다음확장범위를정한다.'
entry=dict(status=status,at=at,unit=str(U),summary=summary,next_single_action=next_action,operator='Codex local implementation and separate arithmetic verification; no independent human ground truth',private_data_access=0,production_changes=0,runs=6,verification_checks=427)
state['updated_date_kst']=at;state['current_stage']=status;state['next_single_action']=next_action;state[unit]=entry;save(statepath,state)
path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=path.read_text(encoding='utf-8-sig')
title='### 2026-10-29 SignRep 시간별 판정부 — 6회 비교 완료, 새 표현 회귀로 미채택'
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
 assert text.count(heading)==1
 text=text.replace(heading,heading+'\n\n'+title+'\n'+at+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+unit+'/RESULT_KO.txt, LABEL_REVIEW.json, INPUT_VERIFICATION.json, SCORES.json, FINAL_VERIFICATION.json.\n',1)
path.write_text(text,encoding='utf8')
handoff=f'''# WordsInSilence 최신 인수인계

{at}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 사용자가 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고, 중요한 차이와 허용 가능한 차이를 구분해 알려준다.

현재 단위: research/front_keyphase_v1/ops/{unit}
현재 상태: {status}

{summary}

이번 비교는 기존운영TCN/DTW의점수를재측정한것이아니다. 관절만쓰는새시간판정부와 관절+고정SignRep를 같은임시주석으로비교한것이다. 학습표현8개×2명, 평가:새사람8/새표현기존사람8/새표현새사람4. 학습두조건100%는일반화증거아님. 본체추가학습0,47,749개판정부(관절기준23,141개)만각3회학습. 5항목은검지만펴짐/네손가락펴짐/화면상겹침으로, 주먹·접촉·필수반복·교육적정오전체를표현하지않는다. 변화7구간의전후모양맞힘은정확한시점/필수단계누락검출이아님.

다음 단일 작업: {next_action}

주석/입력: old DRAFT_LABELS와BUNDLE원본보존. static477기존known중17불확실추가보류, 기존unknown채움없음. 모두108정지장면/36연속구간재검토. FIRST_ROLE_SCHEMA_FAILURE.json은fit대신fit_candidate키를읽어야했던계약코드오류의최초기록;수정후클래스지원통과. 입력은native전체영상관절에서고정상반신범위로mask, RGB도같은영역밖픽셀차단후224입력. 모델768특징은16프레임창/stride2를중앙시각기준보간한표현이지관측복구나의미정답이아님.

완료실행: prepare_inputs.py exec88837, train_compare.py exec34892, runtime_and_diagnostics.py exec2623 모두종료0. 완료결과재추론/재채점/재학습하지않는다. INPUTS/TRAIN_READY/TRAINING/SCORES/RUNTIME/FINAL_VERIFICATION의해시와원자료를재사용한다. 보고서36개시간그래프와연속장면제공. 별도검산427은같은Codex별도코드검산이며독립사람검토가아님.

기존SignRep실행환경·가중치·IntelGPU변환IR는 word100_signrep_feasibility_20261029를읽기전용재사용. 모델SHA f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b 유지. 노트북CoreUltra7 258V/32GB/Arc140V,torchCPU+OpenVINO GPU. 새모델/설치/드라이버변경없음.

보호: C·T3-F/운영웹·DTW 그대로. 사용자·멘토촬영 접근/학습/튜닝/선택금지. 원본W:/sources/과거동결결과읽기전용. 카메라/외부전송/Git없음. 이번보고후다음범위를확인하며자동재학습하지않는다.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
save(U/'AUTHORIZATION.json',dict(at=at,source='Current direct user request',request='공개 한국수어 시간별 손모양·움직임 주석 검증, 문제없으면 SignRep 작은 시간별 판정부 학습/기존 방식 비교, 처리시간 보고',scope='Existing public data only; provisional physical-feature scope explicitly separated from unavailable semantic-stage truth',preserve=['C','T3-F','operating web/DTW','user/mentor footage'],new_downloads=0))
save(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE',at=at,sessions=[dict(id=i,exit_code=0) for i in [88837,34892,2623]],active=None,unit=str(U),report=str(U/'RESULT_KO.html')))
save(U/'DOCUMENT_SYNC.json',dict(at=at,pins={str(p):sha(p) for p in [path,ROOT/'NEW_CHAT_HANDOFF.md',statepath]},result_pins={str(U/n):sha(U/n) for n in ['RESULT_KO.html','RESULT_KO.txt','FINAL_VERIFICATION.json']},status=status))
print('Synchronized latest §8/§9, STATE, NEW_CHAT_HANDOFF; prior contents backed up.')
