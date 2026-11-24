"""Finalize verified local integration records. No inference, training, or camera."""
import json, hashlib
from pathlib import Path
from datetime import datetime

V=Path(__file__).resolve().parent;U=V.parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
at=datetime.now().astimezone().isoformat()
cmp=read(V/'PUBLIC_COMPARISON_FT.json');review=read(V/'INDEPENDENT_FT_REVIEW.json')
admit=read(V/'ADMITTED.json');ui=read(U/'site/UI_REVIEW.json');fixture=read(V/'SERVICE_FIXTURE_ACTION_CHECK.json');web=read(V/'WEB_SERVICE_CHECK.json')
assert cmp['passes']==cmp['total']==16 and cmp['code_unchanged']
assert review['passed'] and all(c['passed']for c in review['checks'])
assert ui['tests']['passed'] and ui['completion_display_tests']['passed'] and fixture['passed'] and web['passed']
for name,digest in cmp['code_hashes'].items():assert sha(U/name)==digest,name
for name,digest in ui['hashes'].items():assert sha(U/'site'/name)==digest,name
assert sha(U/'server.py')==fixture['server_sha256']
assert sha(Path(admit['checkpoint']))==admit['sha256']==web['candidate_sha256']
base=O/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt'
assert sha(base)==admit['base_sha256']
assert (V/'WEB_VERIFIED.jpg').stat().st_size>1000

roles=dict(word_recognition_tcn=False,existing_action_temporal_conv_bigru=True,
    learned_handshape=True,publicly_finetuned_movement_head=True,learned_direction_axes=[2,0],
    explicit_stage_and_repetition_rules=True,greeting_stage_network_trained=False)
verified=dict(at=at,status='VERIFIED_LIMITED_ACTION_MODEL_CONNECTION',passed=True,
    implementation={'root':'integration, supervised source approval, public comparison, service/UI verification, records',
        'streaming_architecture_review':'existing-model adapter and isolated movement-head fine-tune',
        'live_action_ui':'stable message presentation and UI tests',
        'hello_contract_review':'independent code, split, source-label and candidate audit; not human KSL expert'},
    model_roles=roles,base_sha256=admit['base_sha256'],head_sha256=admit['sha256'],
    regression_cases=dict(passed=16,total=16,unit='controlled conditions from the same public reference, not 16 independent videos'),
    independent_code_checks=len(review['checks']),text_presentation_checks=len(ui['tests']['checks']),
    completion_presentation_checks=len(ui['completion_display_tests']['checks']),
    public_fake_device_lifecycle=fixture['passed'],real_browser_public_replay=web['passed'],
    physical_camera_opened=False,physical_camera_latency_verified=False,private_data_used=False,
    original_models_replaced=False,old_C_T3_F_changed=False,whole_sign_accuracy=None,new_person_accuracy=None,
    initial_failure_preserved='PUBLIC_COMPARISON.json: 11/16 after unadapted strict model connection',
    known_limit='10fps input still misses the brief first sweep; held rather than fabricated completion',
    evidence={p.name:sha(p) for p in [V/'PUBLIC_COMPARISON_FT.json',V/'INDEPENDENT_FT_REVIEW.json',
        V/'SERVICE_FIXTURE_ACTION_CHECK.json',V/'WEB_SERVICE_CHECK.json',U/'site/UI_REVIEW.json',V/'finetune/RESULTS.json']},
    current_code_hashes={str(p.relative_to(U)):sha(p)for p in [U/'coach.py',U/'action_runtime.py',U/'server.py',
        U/'site/app.js',U/'site/index.html',U/'site/style.css',U/'site/presenter.js']})
write(V/'VERIFIED.json',verified)
write(V/'FINAL_DECISION.json',dict(at=at,supersedes_intermediate_plan='REVISION_DECISION.json remains history; its axis0 diagnostic-only/no-training proposal is superseded by the fixed public-only fine-tune and final regression',
    decision='Admit copied movement head and existing Conv+GRU to 8822 only after public regression and independent audit',
    training='One fixed 400-step run; only two Linear layers in motion_head. Original/shared temporal Conv+GRUs frozen. Historical public FIT replay plus one approved reference with nine reviewed direction/hold intervals. DEV excluded from optimizer.',
    sparse_input='Generic 3 genuine distinct-time points, all observed, >=33333us, use existing 9-sample interpolated feature representation; never claim interpolation creates sensor observations',
    probability_threshold_changed=False,source_reference_count=1,
    no_whole_sign_or_generalization_claim=True,admitted_axes=[2,0],first_failure_preserved=True))

text='''기존 행동 분석 모델 실시간 연결 · 안내 문장 안정화

이번에 달라진 점
이전 화면은 학습된 손모양 판정과 위치·반복 진행 규칙으로 두 동작을 확인했다.
지금은 기존 100단어 공통 시간 합성곱+양방향 GRU의 특징과 움직임 판정부도 실제로 사용한다.
팔을 따라 이동한 방향, 양손이 내려갔다가 올라오는 방향을 모델이 뒷받침해야 단계가 진행된다.
단계 순서, 시작·끝 조건, 반복 횟수 세기는 아직 명시적인 진행 규칙이다.
단어 인식 TCN은 이 연습 화면에서 사용하지 않는다. 기존 C·T3-F와 다른 시험 화면은 바꾸지 않았다.

왜 작은 추가 학습을 했는가
그대로 연결한 첫 시험에서는 짧은 정상 반복을 모델이 정지로 보아 완료를 막았다. 첫 비교 16조건 중 11조건만 유지됐다.
입력 정규화 오류와 학습 자료 범위를 확인했다. 기존 직접 높이 정답에는 이번처럼 두 주먹을 짧게 반복하는 사례가 부족했다.
공개 인사 원본의 9개 방향·유지 구간을 영상으로 다시 확인하고 기존 공개 학습 자료와 함께 방향 판정부 복사본만 400회 갱신했다.
공유 시간 합성곱+GRU·손모양 신경망·원본 체크포인트는 그대로다. 새 외부 다운로드, 사용자 촬영 학습·튜닝은 없다.
한 영상의 원래 속도/간격 축소/역방향 파생은 서로 독립된 사람의 자료가 아니다.

검증 결과
같은 공개 자료의 정상, 첫 동작만 수행, 반대 방향, 주먹 정지, 한 번만 반복, 한 손만 움직임, 손 누락,
동작 중 손모양 치환, 시간 간격·크기·위치 변화 등을 포함한 16개 조건이 최종 기대 결과와 일치했다.
15fps 간격으로 남긴 공개 관측에서도 두 단계가 완료됐다. 10fps에서는 짧은 첫 동작을 놓쳐 보류하는 한계가 그대로다.
기존 개발용 32개 방향/정지 구간은 최상위 판정 32개 일치를 유지했다. 100단어 전체 정확도 수치가 아니다.
정상 공개 원본은 보완 전후 같은 수신 시점에서 단계가 완료됐다. 이번 결과는 인식률 향상 증명이 아니라
학습된 움직임 근거를 추가하면서 기존 공개 조건을 유지한 연결 검증이다.
별도 검토자가 실제 가중치 변경 범위, FIT/DEV 분리, 미래 프레임 미사용, 관측 누락·초기화·양손 조건을 확인했다.
공개 영상으로 대신한 카메라 모의에서 3초 준비→관측→완료, 이전 실행 결과 차단을 확인했다.
실제 브라우저에서도 공개 재생 0/2→2/2와 새 문장 배치를 확인했다. 실제 카메라는 이번 검증에서 열지 않았다.

글씨가 정신없이 바뀌던 문제
큰 안내는 같은 내용이 약 0.35초 지속될 때 바꾸며, 보통 한 문장은 최소 약 1.4초 읽을 시간을 준다.
잠깐의 누락에는 새 지시를 쏟아내지 않고, 불확실함이 약 0.6초 이어지면 관측 확인 안내로 바꾼다.
세부 교정 내용이 계속 흔들리면 일정 시간 후 한 가지 공통 안내를 보여준다.
단계 완료, 재시작, 종료, 오류는 즉시 반영한다. 실제 판정·관절 강조는 숨기지 않고, 관측 부족은 초록색 성공으로 유지하지 않는다.
기본 화면은 큰 문장 한 개와 현재 확인 항목만 보인다. 모델 상태와 기술 설명은 접힌 상세 안에 둔다.
이 시간은 읽기 편한 화면 표시 설정이며, 정답 판정 임계값이나 학습 결과를 바꾼 것이 아니다.

속도 해석
공개 좌표를 이미 읽은 상태에서 손모양+행동 모델+진행 계산의 중앙값은 프레임당 약 30.6→43.0ms였다.
이 수치는 RTMW 검출·웹캠·전송·화면 렌더링을 제외한다. 약 43ms만 기다리면 화면 결과가 나온다는 뜻이 아니다.
최근 3관측 뒤의 시점을 보고 앞뒤 흐름을 함께 사용한다. 실제 처리 프레임률이 낮으면 표시 지연도 커질 수 있다.
실제 웹캠의 전체 지연·새 사용자의 성공률은 직접 확인이 남았다.

남은 범위와 다음 한 작업
http://127.0.0.1:8822/ 새로고침 후 실제 웹캠에서 안내의 읽기 편함, 두 단계 진행, 관측 지연을 확인한다.
사용자 촬영은 RAM에서만 처리하며 학습·튜닝·후보 선택에 쓰지 않는다. 문제가 보이면 공개 자료로 재현하여 보완한다.
엄지·정확한 손바닥 방향·실제 접촉·여러 사람의 안녕하세요 전체 정답 판정은 아직 검증하지 않았다.
기존 모델의 공통 움직임 기능을 연결한 것이지, 모델이 안녕하세요의 모든 설명과 구간을 자율적으로 이해했다는 뜻은 아니다.
현재 판정 결과 whole_word_correct는 null을 유지한다.

근거: action_integration/VERIFIED.json, PUBLIC_COMPARISON_FT.json, INDEPENDENT_FT_REVIEW.json,
finetune/RESULTS.json, SERVICE_FIXTURE_ACTION_CHECK.json, WEB_SERVICE_CHECK.json, site/UI_REVIEW.json.
최초 실패와 중간 대안은 삭제하지 않고 보존했다. 구현/검증은 Codex와 별도 에이전트가 수행했으며 수어 전문가 검수는 아니다.
'''
(V/'RESULT_KO.txt').write_text(text,encoding='utf-8');(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
graph=dict(at=at,scope='별도 8822 실시간 인사 연습: 학습된 공통 움직임 관측 연결',
    new_training='Copied movement head only; original shared temporal model frozen',private_training=False,model_roles=roles,
    nodes=[dict(id='A',label='공개 원본과 방향·정지 정답 확인',status='complete_single_reference'),
        dict(id='B',label='기존 시간 합성곱+GRU 연결과 방향 판정부 보강',status='complete_limited_direction_scope'),
        dict(id='C',label='공개 정상·오류 비교와 별도 검토',status='complete_controlled_conditions'),
        dict(id='D',label='웹 연결·문장 안정화·공개 재생 확인',status='complete'),
        dict(id='E',label='실제 웹캠의 두 단계 진행·안내·지연 확인',status='pending_user_test'),
        dict(id='F',label='다인 인사·엄지·손방향·접촉·전체 단어 검증',status='not_complete')],
    edges=[['A','B'],['B','C'],['C','D'],['D','E'],['E','F']],
    failure_loops=[dict(from_node='C',to='A',reason='실제 정답·학습근거 확인 후 필요한 부분만 보강'),
        dict(from_node='E',to='B',reason='사용자 촬영으로 튜닝하지 않고 공개 자료로 재현·검증')],
    release='Existing C/T3-F unchanged. Word TCN intentionally outside guided practice. whole_word_correct=null.')
write(U/'TASK_GRAPH.json',graph)
next_action='8822에 기존 시간 합성곱+GRU와 공개 보강 방향 판정부 연결·문장 안정화·공개 비교 완료. 다음은 실제 웹캠에서 안녕하세요 두 단계 진행과 안내 읽기 편함·전체 관측 지연을 확인. 개인 촬영은 학습/튜닝/후보 선택 금지, 문제는 공개 자료로 재현. 다인·엄지·손방향·접촉·전체 단어 검증은 별도 남음. C·T3-F 유지.'
entry=dict(at=at,status='ACTION_MODEL_CONNECTED_PUBLIC_VERIFIED_CAMERA_PENDING',report=str(V/'RESULT_KO.txt'),
    graph=str(U/'TASK_GRAPH.json'),verification=str(V/'VERIFIED.json'),url='http://127.0.0.1:8822/',pid=web['listener_pid'],
    model_roles=roles,training_runs=1,training_scope='copied motion_head only',private_training=False,reference_count=1,
    whole_word_correct=None,physical_camera_tested=False,independent_review=str(V/'INDEPENDENT_FT_REVIEW.json'),next_single_action=next_action)
s=read(O/'STATE.json');s['hello_live_action_integration_20261116']=entry;s['hello_live_coach_20261115']=dict(entry)
s['next_single_action']=next_action;write(O/'STATE.json',s)
context=R/'WordsInSilence_PROJECT_CONTEXT.md';old=context.read_text(encoding='utf-8-sig')
header='### 2026-11-16 실시간 행동 모델 연결·문장 안정화 완료 / 실제 웹캠 확인 남음'
section='''\n\n'''+header+'''\n별도8822에 기존100공통 시간합성곱+양방향GRU 특징과 방향판정부를 실제 연결했다. 첫 연결은 짧은 정상반복을 정지로 보아 공개16조건 중11만 유지. 공개SEN0354 REAL01의9방향/정지구간을 영상확인하고 기존FIT과 함께 복사한motion_head만 고정400step 보강했다. 공유Conv+GRU·손모양·원본가중치/C·T3-F는동결. 최종 공개16조건 유지, 기존DEV32방향구간32일치유지,별도코드검토23통과. 독립새사람정확도아님. 15fps완료/10fps첫동작보류한계 유지. 단계순서·반복횟수·시작끝은명시규칙,whole_word_correct=null,단어TCN은연습미연결.\n큰교정문장 안정화(지속350ms/읽기1400ms/불확실600ms),모델상태는접힌상세로이동. 실제판정·강조색을숨기지않으며완료·초기화는즉시표시. 공개카메라모의/브라우저0→2단계확인,실제카메라미개방. 서버8822일반실행 PID'''+str(web['listener_pid'])+''', 현재실행은SERVER.json과대조. 공개캐시좌표계산중앙30.6→43.0ms는검출/전송/카메라/렌더제외. 사용자전체지연미측정.\n근거 research/front_keyphase_v1/ops/hello_live_coach_20261115/action_integration/VERIFIED.json·RESULT_KO.txt·ADMITTED.json. Codex통합/streaming_architecture_review모델/live_action_ui화면/hello_contract_review독립감사. 사용자촬영학습·튜닝·후보선택없음. 최초실패와중간대안보존.\n'''
if header not in old:
 old=old.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'+section,1)
 old=old.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-16 실시간 행동 모델 연결 후속\n'+next_action+'\n',1)
 context.write_text(old,encoding='utf-8')
handoff='''# 최신 목표 — 학습된 동작 모델을 이용한 한국수어 피드백 게임

## 마지막 완료 / 현재 범위
2026-11-16 사용자 승인: 기존 행동 분석 시간 합성곱+GRU를 별도 안녕하세요 실시간 연습8822에 연결. 순간 검출변화로 교정글씨가 번잡하다는 후속 요청도 반영.
현재 기존100단어공통 Conv+양방향GRU 특징+손모양보조+추가학습 방향판정부 사용. 팔쓸기 축2, 양손높이 축0의 학습방향이 뒷받침해야 단계진행. 순서·반복횟수·시작끝은 명시규칙. SEN0354 단계/전체단어를 모델이 자율인식한 것은 아님. word TCN은 이 연습에 미사용, 기존 C·T3-F·8821 유지.

## 이번 학습과 검증
첫 엄격 연결은 정상짧은반복을hold로보아11/16만유지. 원본입력정규화와학습정답범위확인 후, 한사람공개SEN0354 REAL01의9개방향/정지구간을RGB확인하고 기존공개FIT과함께 motion_head 복사본만 seed42/400step 학습. 공유Conv+GRU/손모양/원본파라미터동결. DEV/개인촬영학습·튜닝·후보선택없음. 원본/시간간격축소/역방향파생은독립자료아님.
최종공개16조건 유지(정상/중단/역방향/누락/한손반복/모양치환/크기위치속도). 기존DEV32방향구간32일치유지. 15fps완료,10fps짧은첫동작보류. 최초실패보존. 별도감사23항목통과. 모델전체새사람정확도또는언어적전체정답으로보고금지.

## 화면 / 실행
큰문장1개+현재칩. 텍스트지속350ms/읽기1400ms/불확실600ms는UI설정일뿐판정완화아님. 관절강조·실제검사상태즉시;완료/종료/초기화우선. 모델상태는접힌상세. 공개모의카메라3초준비→실시간→완료/이전세션출력차단,브라우저공개0→2확인. 실제웹캠은이번검증에서열지않음.
http://127.0.0.1:8822/ 일반실행 listenerPID'''+str(web['listener_pid'])+''' (launcher5208). 재개시 SERVER.json과실제포트프로세스확인. sandbox에서일반프로세스가안보일수있어없다고단정금지. 카메라는사용자버튼에서만열림. 촬영/관절/판정은RAM만. 학습프로세스종료.
기존캐시좌표기반CPU계산중앙30.6→43.0ms(손모양+모델+진행),검출/카메라/전송/렌더제외. 실제웹캠전체지연미검증.

## 최신 근거
단위: research/front_keyphase_v1/ops/hello_live_coach_20261115/
action_integration/RESULT_KO.txt, VERIFIED.json, FINAL_DECISION.json, ADMITTED.json, PUBLIC_COMPARISON_FT.json, INDEPENDENT_FT_REVIEW.json, WEB_SERVICE_CHECK.json, SERVICE_FIXTURE_ACTION_CHECK.json. site/UI_REVIEW.json, TASK_GRAPH.json.
원본모델 word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt SHA80aba3b28006270308834bf5ba88277d7b26c8dce01c11fc962ce5c21b8cea17。
후보방향머리 action_integration/finetune/CANDIDATE.pt SHA810f3b0dd922e83ac12045e848e660e4c2d6ef8b6190a6eb49a15cac542ef378。
CONTRACT.json·REVISION_DECISION.json은당시계획이다. 후속FINAL_DECISION·ADMITTED가최종;axis0진단전용/학습없음이라는중간대안은최종아님.

## 남은 범위 / 다음 한 작업
'''+next_action+'''
엄지·정확한손바닥방향·실제접촉·다인정답은미완료. 한사용자성공도일반화보장아님. 기존4단어는공통회귀,실제시연목표안녕하세요SEN0354/사과WORD0099/감사WORD1290유지. 인사원본1개는C의 runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4. W의1TB는WORD33ZIP, SEN다인자료아직없음;자동다운로드하지않음. 감사·사과후속연구는 basic_demo_feedback_20261115/acquisition_motion_followup에보존.

## 구현·검증 역할
Codex root 통합/공개비교/서버/브라우저/기록, streaming_architecture_review 모델연결·고정작은학습, live_action_ui 문장안정화, hello_contract_review 별도데이터분리·원본가중치·코드감사. 수어전문가정답검수아님.
Python: research/front_keyphase_v1/ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8 -B.
현재 웹 STATE는 research/front_keyphase_v1/ops/STATE.json. 루트Ops/STATE는Unity계보. 문서최신키만읽고과거보고와혼동금지.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
assert read(O/'STATE.json')['hello_live_action_integration_20261116']['pid']==web['listener_pid']
assert header in context.read_text(encoding='utf-8')
print(json.dumps(dict(passed=True,at=at,records_updated=True,public_regression='16/16 controlled conditions',listener_pid=web['listener_pid'],physical_camera=False),ensure_ascii=False))

if __name__=='__main__':pass
