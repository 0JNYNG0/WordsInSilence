from pathlib import Path
import json,hashlib,datetime,runpy
U=Path(__file__).resolve().parent; O=U.parent; R=U.parents[3]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert R.name=='WordsInSilence',R
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
assert all(sha(p)==sha(U/'before'/p.name) for p in paths),'Concurrent root documentation change'
for n,expected in [('UI_VERIFY.json','PASS'),('METRICS_ROOT_VERIFY.json','PASS'),('review/GUARD_VERIFY.json','PASS_INDEPENDENT_SAVED_ARITHMETIC_REPLAY')]:
 assert read(U/n)['status']==expected,(n,read(U/n).get('status'))
pins=read(U/'FROZEN.json')['pins'];protected=read(O/'word100_face_relation_alignment_20261027/FROZEN.json')['protected']
assert all(sha(p)==h for p,h in pins.items())
assert all(sha(p)==h for p,h in protected.items())
bp=U/'build_report.py';s=bp.read_text(encoding='utf-8')
s=s.replace('독립적으로 알려진 값을 하나씩 숨긴 AI 관찰 초안 대조 시험이다.','현재 RGB에 AI가 붙인 특징값을 매번 한 칸씩 따로 가린 대조 시험이다.')
s=s.replace('이번 4개 충돌과 새 결과는','이전 시간문맥 시험의 4개 진단 사례와 새 결과는')
s=s.replace('인위적 가림 시험','특징값 한 칸 가림 시험')
bp.write_text(s,encoding='utf-8');runpy.run_path(str(bp),run_name='__main__')
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
summary='''사용자가 제시한 손모양 상태와 변화 과정의 분리를 구현하고, 짧은 변화를 덮어쓰지 않도록 RGB·2D 관절 변화 신호를 추가한 후보를 검증했다. 이전/다음 버튼은 실제 연속 원본 1프레임(약 0.033초)이며 모아보기와 같은 이미지임을 확인했다. K번호는 임의 식별자였다. 새 연구 화면은 실제 프레임/시간 및 이전·현재·다음 사진을 로딩 완료 후 함께 바꾸며 느린 연속 보기를 제공한다. 새 6표현×3기록ID의 18구간·234RGB를 동결했다. 468특징칸의 셔플 이미지 초안은 411확인/57보류. 알려진 348칸을 한 칸씩 가린 비교에서 기존 앞뒤 연결은 259일치/0충돌/89보류, 변화 차단 후보는 39일치/0충돌/309보류였다. 기존 일치 220개까지 보류하므로 후보 미채택. 과거 진단 279칸에서는 194일치/1충돌/84보류→92일치/0충돌/187보류로 짧은 펼침 충돌 1개를 막았지만 일치 102개도 보류했다. 실제 새 미확인57칸의 추정은 기존7개→후보0개로 실제 복구 향상은 확인하지 못했다. 2개 안정 구간의 24연결로 고정한 2D각 3.0341도/RGB잔차 0.10799 기준은 정상 회전·투영·관절 흔들림까지 막는 한계가 있었다. 결과에 맞춘 기준 변경은 없다. 직접 관측411+334칸을 보존하고 상태 변화12구간/같은 끝 상태 사이 미확인8구간을 별도로 기록했다. 이는 정답 증명이나 수어 정오 판정이 아니다. 원본·보호핀32개 불변, 432인접점수 산술과 규칙 별도 구현 및 새뷰어468상태/1404사진슬롯 검산 통과. 같은 AI의 관찰 초안/독립코드 검산이며 독립 사람 정답은 아니다. C·T3-F·운영 웹·DTW 유지, 학습/새검출/개인자료사용0.'''
next_action='''다음 단일 작업은 기존 공통 특징 묶음에 손모양 유지·변화·관측 불가의 시간 범위를 연결하고, 보이는 핵심 손모양과 그 순서가 보존되는지를 평가하는 것이다. 이번 하드 차단 후보는 미채택으로 보존하고 같은 평가 자료에서 기준을 조절해 성능을 만들지 않는다. 손 전체 이동·회전과 실제 손가락 모양 변화를 분리할 신뢰 가능한 특징 정답을 먼저 확인한다. 모든 중간 프레임을 강제 복구하거나 ‘네 손가락 폄 아님’을 주먹으로 치환하지 않는다. 원본 시간·좌표·현재 관측·추정·변화 구간을 구분하고, 구간의 언어적 중요성을 자동으로 무시하지 않는다. 검증 가능한 정답이 확보되면 기존 표현/기록ID 분리를 유지해 작은 RGB+시간 특징 모델 시험으로 연결한다. 목표 단어/Gloss를 관측 정답 힌트로 넣거나 같은 AI 추정을 독립 정답으로 승격하지 않는다. C·T3-F 및 운영 웹/DTW 보존, 사용자·멘토 자료의 학습·튜닝·선택 금지. 승인된 일반 준비·검증 재승인 불필요, 운영 교체는 별도 범위. 이번 단위 실행 종료, 활성 학습/검출/서버 없음.'''
evidence=f'research/front_keyphase_v1/ops/{U.name}/RESULT_KO.html'
title='2026-10-27 손모양 상태·변화 구간 및 원본 프레임 검증 완료 — 변화 차단 후보 미채택'
ctx=paths[0].read_text(encoding='utf-8-sig')
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
 assert ctx.count(header)==1
 ctx=ctx.replace(header,header+'\n\n### '+title+'\n'+at+'\n'+body+'\n근거: '+evidence+', COMPARISON.json, METRICS_ROOT_VERIFY.json, review/GUARD_VERIFY.json, UI_VERIFY.json.\n',1)
paths[0].write_text(ctx,encoding='utf-8')
state=read(paths[2]);state['updated_date_kst']=at;state['current_stage']='TEMPORAL_TRANSITION_REPRESENTATION_VERIFIED_HARD_GUARD_NOT_ADOPTED';state['next_single_action']=next_action
d=read(U/'COMPARISON.json')
state[U.name]={'status':'COMPLETE_HARD_GUARD_NOT_ADOPTED_NO_RECOGNITION_OR_FEEDBACK_GAIN_ESTABLISHED','at':at,'summary':summary,'next':next_action,'evidence':evidence,'decision':read(U/'DECISION.json'),'fresh_masked_comparison':d['fresh_summary'],'old_diagnostic_comparison':d['old_diagnostic_summary'],'implementation':'Codex root: freeze/evaluation/representation/report/UI; linguistic_eval_review: RGB extraction/flow/pose metrics; local_feature_inventory: original frame and previous viewer audit; ksl_source_audit: shuffled current-RGB drafts.','verification':'Root separate raw arithmetic, ksl_source_audit separate guard implementation, local_feature_inventory center image review. Same AI family, not independent human ground truth.','active_execution':None,'training_ready':False,'operating_changes':0,'user_mentor_access':0}
dump(paths[2],state)
handoff=f'''# WordsInSilence 최신 인수인계

{at}

## 현재 목표와 마지막 완료
{summary}

## 다음 한 작업
{next_action}

## 구현·검증·한계
- root: 조건 동결, 상태·변화·미관측 기록, 비교 집계, 연구 화면, 별도 원시 점수 산술 검산 및 문서 갱신.
- linguistic_eval_review: 새234원본 이미지 추출, 저장 관측에서 RGB 흐름·2D각 변화432연결 계산, 별도 scalar 산술 검산.
- ksl_source_audit: 순서를 숨긴234이미지/23확대보기의 특징 초안, 별도 코드로348신규/279과거 마스킹과20구간 검산, 보고 문구 검토.
- local_feature_inventory: 이전 버튼432클릭/450완료상태·234모아보기타일 감사, T15/T17의26원본RGB 대조. 별도18중심 이미지 초안 대조는30일치/5양쪽보류/1한쪽보류.
- 보호된 직접 관측411+334는 설계 보존 검사. 정상 손/교차 좌표나 좌우 배정을 바꾸지 않았으며, 검출 정확도를 새로 증명한 것은 아니다.
- 현재 ‘C 앞뒤 규칙’은 운영 TCN C와 다른 이름. 현재 연구는 손모양 특징 시간근거 비교이며 TCN 학습이나 DTW 운영 적용이 아니다.
- 새216연결: 변화 신호 없음42/변화 또는 시점 흔들림124/근거부족50. RGB+pose유효166중각도초과117. 이 원시 점수만으로 실제 손가락 변화와 회전/가림/검출 지터의 원인을 확정하지 않는다.
- 새 ID2/3의224마스킹: 기존160일치/0충돌/64보류→후보17일치/0충돌/207보류. 사람1의 안정24연결로 정한 기준이며 미관측 손의 정답 없는 복원을 성능으로 세지 않는다.
- 원본버튼은 실제1프레임이다. K 식별자는 임의순서. 이전 로딩 중 설명 선변경은 확인됐지만 잘못된 이전 픽셀이 새 설명과 표시됐다는 현상은 재현되지 않았다. 새 화면은 decode 후 사진/시간 동시 갱신.
- 첫 실패들은 frame_audit/ 및 metrics/FIRST_ARITHMETIC_REVIEW_ERROR.json에 보존. 파일URI 감사의 해시 이동/초기상태 문제와 cp949 읽기 문제를 수정했으며 원본/라벨/평가조건은 바꾸지 않았다.
- 코드 검증은 별도 담당/별도 구현이며 모두 같은 AI 계열. 실제 수어 사용자·교육자의 독립 정답 또는 교육적 허용 기준 검증이 아니다.
- 학습0, 새 검출기0, 운영 교체0, 사용자/멘토 자료0, 다운로드/외부전송/Git0. 이번 단위 세션 종료; 새 서버나 자동화 없음.

## 근거
- `{evidence}`: 실제 원본3장/시간순13장/느린 재생 및 상태·변화 표시.
- `{U.name}/COMPARISON.json`: 최초 원시 비교 결과 보존. 재채점하지 않음.
- `{U.name}/FROZEN.json`, `metrics/CALIBRATION.json`: 사전 조건 및 안정구간 기준.
- `{U.name}/METRICS_ROOT_VERIFY.json`, `metrics/ARITHMETIC_REVIEW.json`: 432연결 산술 검산.
- `{U.name}/review/GUARD_VERIFY.json`: 별도 규칙 구현 및 집계 검산.
- `{U.name}/frame_audit/FRAME_AUDIT.json`, `UI_VERIFY.json`: 이전 원본 프레임 감사와 새 화면468상태 검증.

후보의 실패를 RGB·시간 모델 전체의 불가능으로 일반화하지 않는다. 이번 단위는 완료했지만 실제 가림 복구·TCN·DTW 성능 향상은 입증하지 못했다.
'''
paths[1].write_text(handoff,encoding='utf-8')
for p in paths: assert U.name in p.read_text(encoding='utf-8-sig')
assert all(sha(p)==h for p,h in protected.items())
dump(U/'FINAL_VERIFY.json',{'status':'PASS','at':at,'root_docs':{str(p):sha(p) for p in paths},'report_sha256':sha(U/'RESULT_KO.html'),'comparison_sha256':sha(U/'COMPARISON.json'),'protected_pins':len(protected),'report_review':'ksl_source_audit reviewed counts and scope; independent-truth ambiguity and old four-conflict source wording corrected; no scoring change.','ui':'468 states / 1404 image slots tested; screenshots reviewed by root; later changes prose only.','active_execution':None})
print('Documentation synchronized; candidate not adopted; existing models unchanged.')
