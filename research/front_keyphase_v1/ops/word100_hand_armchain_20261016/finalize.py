"""Record verified outcome and synchronize latest handoff. No model execution."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
LAB = U.parents[1]

def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def new_json(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

assert not (U/'COMPLETION.json').exists()
frozen = read(U/'CANDIDATE_FREEZE.json')
for p,h in frozen['pins'].items():
    assert sha(p) == h, p
r = read(U/'REPLAY_RESULT.json')
v = read(U/'INDEPENDENT_VERIFICATION.json')
a = v['saved_output_audit']
assert r['counts_vs_previous'] == {'unchanged':6720}
assert a['status'] == 'PASS' and a['frames'] == 5072 and a['conditions'] == 40
assert v['synthetic']['tests'] == 67 and v['synthetic']['status'] == 'PASS'
assert not a['all_changes_vs_joint'] and not a['previous_assigned_raw_changes']
model = LAB/'runs/word100_tcn_v1/whole_context/best.pt'
model_hash = 'e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'
assert sha(model) == model_hash
stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
summary = ('어깨·팔꿈치·손목의 최근 실제 연결을 미배정 손의 추가 근거로 쓰는 공통 연구 후보 구현·독립 검증 완료. '
    '공개 TRAIN20영상 원본/상반신40조건,5,072프레임·손6,720건 모두 직전joint v3와 동일: 추가 회복0·새 제외0·좌우변경0. '
    '기존 믿다2건·냉커피1건 회복 유지. 합성/회귀67검사,별도 코드반례3건,검출순서반전40조건,보호파일1,248개 검산 통과. '
    '실제 추가 효과는 확인되지 않아 연구 후보로 보존하며 운영 미채택·학습 미진행. '
    '남은 유효raw10건은 최초 팔 연결 확보 전이며,소복57/응접실46/팔십구43의 몸 오른손목은 좌표는 있으나 visibility0.491717/0.404972/0.411181로 기존0.5 품질기준 미달. '
    '사용자자료·계수·원본·crop·TCN·DTW·운영웹 변경 없음.')
next_action = ('사용자와 추가회복0·새회귀0 및 초기 손 배정 한계를 함께 검토한다. '
    '다음 보완 후보는 공개 TRAIN의 첫 손 진입 구간에서 실제 손 검출기의 손목·현재 어깨/팔꿈치·시간 연속성을 교차 확인하는 공통 초기 배정이다. '
    '몸 손목 품질이 낮은 이유와 원본/상반신 동일 프레임 차이를 먼저 분리하고, 단순 신뢰 문턱 하향·팔 직선 연장·빈 손 생성은 하지 않는다. '
    '새 후보는 별도 조건 고정 후 검증하며,사용자 결과 확인 전 학습·전체899재추출·운영모델/DTW교체는 시작하지 않는다.')
report = f'''팔 연결 기반 좌우 손 배정 보완 결과
갱신: {stamp}

결론
요청한 공통 팔 연결 보완은 연구용 코드로 구현하고 검증했다. 그러나 기존 문제 영상의 추가 회복은 0건이었다. 기존 배정이 바뀌지 않아 이 표본의 새 회귀도 0건이다. 문제 해결 또는 인식률 개선으로 보고하지 않으며 학습과 운영 적용은 하지 않았다.

수정한 내용
1. 기존 어깨너비·몸 손목·손바닥 추적에, 실제로 함께 확인된 어깨→팔꿈치→손목 연결 이력을 추가했다.
2. 현재 팔꿈치 위치와 최근 실제 전완 관계를 비교해 미배정 손 후보의 좌우 판단을 보조한다. 같은 손 중복 후보도 함께 비교한다.
3. 최근100ms/두 프레임 이내,현재 팔 관측·기존 손 추적·좌우 label·경쟁 후보가 일관될 때만 사용한다. 특정 단어/사람 분기는 없다.
4. 기존에 배정된 손과 기존 추적 상태를 바꾸지 않는다. 새 회복으로 팔 이력을 재갱신하지 않으며 새 손 좌표를 만들지 않는다.

실제 비교
자료: 이전과 동일한 공개 학습20영상의 원본/상반신40조건. 같은 저장 MediaPipe 원검출을 사용했다.
프레임5,072 / 손 검출6,720 / 직전 후보와 동일6,720 / 추가 회복0 / 새 제외0 / 좌우변경0.
이전 믿다 원본71·상반신56의 왼손,냉커피 상반신52의 오른손 회복 유지.
팔 연결 가설은3,586프레임에서 생성되어 기능 작동을 확인했으나,남은 문제 손의 추가 회복으로 이어지지 않았다.

왜 해결되지 않았는가
미배정29건 중18건은 같은 손의 대안/중복 후보 억제,10건은 아직 확인된 팔 연결 이력이 없는 유효 손 후보,1건은 손바닥 경계 유효성 미달이었다.
소복57,응접실46,팔십구43은 어깨·팔꿈치가 유효하고 몸 오른손목 좌표도 출력된다. 그러나 몸 손목 visibility가 각각0.491717,0.404972,0.411181로 기존0.5 기준에 못 미친다. 이때 손 검출기는 실제 손 후보를 별도로 내고 있다. 몸 손목 좌표가 아예 없는 상황과 다르다.
이 문제 프레임 이전에 신뢰 가능한 전체 팔 연결이 한 번도 확보되지 않았으므로 이번 '최근 확인한 팔 연결 유지' 방식으로 초기 좌우를 결정할 수 없었다. 이력 보존 시간을 늘려도 존재하지 않는 과거 이력이 생기지는 않는다.
소복의 중복 후보 동률 보류,응접실·팔십구의 초기 손 보류는 남는다. 믿다/냉커피 주변의 원검출 자체 부재,즐겁다 경계 손도 해결되지 않았다.

검증
구현: Codex root. 설계/코드/결과 해석: armchain_review. 재생/원인/영상: armchain_replay. 독립 검산: armchain_verify.
67개 합성/회귀 검사 통과,검토자 별도 반례3개 통과. 40조건 검출 순서를 뒤집어도 물리적 배정 동일. 기존 좌표140,511점·몸 관절·마스크·시간 정보 보존,보호파일1,248개 해시 일치.
세 문제 장면의 visibility/경계/관측 여부를 CAUSE_AUDIT.json에 분리 기록. 소복/응접실 실제 전후 영상의 팔 연결 표시판을 root도 확인했다.

한계 및 다음 작업
{next_action}
이 결과는 입력 보정의 구현/보존 검사이며 수어 정확도·손21점 정확도·새 사용자 일반화를 검증한 것이 아니다. 기존 사용자 촬영은 학습·튜닝·후보선택에 사용하지 않았다.

근거
SCOPE.json / CANDIDATE_FREEZE.json / REPLAY_RESULT.json / INDEPENDENT_VERIFICATION.json
CAUSE_AUDIT.json / EMPIRICAL_REVIEW.md / VISUAL_REVIEW.md / PREREQUISITE_IMAGE_INDEX_v2.json
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
decision = dict(status='VERIFIED_IMPLEMENTATION_NO_EMPIRICAL_GAIN_NOT_ADOPTED',updated_kst=stamp,
    summary=summary,next_single_action=next_action,candidate_sha256=sha(U/'assignment.py'),
    model_sha256=model_hash,training=0,classifier_inference=0,new_detector_runs=0,
    production_changes=0,user_data_used=False,additional_recoveries=0,new_drops=0,new_side_changes=0,
    units=dict(clips=20,conditions=40,frames=5072,raw_hands=6720),independent_tests=67,
    acceptance='Implementation and preservation verified; no measured benefit, existing problems unresolved; not training-ready declaration')
new_json(U/'DECISION.json',decision)

# Preserve the four latest shared documents before the authorized state update.
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',LAB/'ops/STATE.json',ROOT/'Docs/WEBCAM_IMPROVEMENT_TODO.md']
(U/'before').mkdir()
for p in docs:
    shutil.copyfile(p,U/'before'/p.name)
context=docs[0].read_text('utf-8')
header8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
header9='## 9. 다음 단일 작업'
assert context.count(header8)==1 and context.count(header9)==1
context=context.replace(header8,header8+'\n\n### 2026-10-16 팔 연결 보완 구현·검증 — 추가 회복 없음, 초기 배정 검토 대기\n'+summary+'\n근거 `ops/word100_hand_armchain_20261016/RESULT_KO.txt`, `DECISION.json`, `INDEPENDENT_VERIFICATION.json`, `CAUSE_AUDIT.json`.')
context=context.replace(header9,header9+'\n\n### 2026-10-16 다음 — 팔 연결 결과 상호 검토 후 초기 손 배정 범위 확정\n'+next_action+'\n이번 사용자가 구현·검증까지 직접 승인한 팔 연결 후보는 완료했으며 아래 이전 제안 대기 기록보다 이 항목이 최신이다.')
stale='현재 추가 정정: 다음 팔 연결 보완은 이미TCN에포함된팔꿈치관절을새로넣는작업이아니라, 인식전 좌우배정에도팔연결을활용하는검토다. 사용자의실제범위/장면검토후기능단위를확정한다.'
context=context.replace(stale,'이전 설명 기록: 팔꿈치는 이미 TCN 인식 입력에 포함되어 있고, 이번 수정은 인식 전 손 좌우 배정에 팔 연결을 추가한 연구 후보다. 최신 구현 결과는 위 항목을 따른다.')
docs[0].write_text(context,encoding='utf-8')
state=read(docs[2]);state['updated_date_kst']=stamp
state['current_stage']='WORD100_ARMCHAIN_VERIFIED_NO_ADDITIONAL_GAIN_REVIEW'
state['next_single_action']=next_action
entry=dict(decision,unit='ops/'+U.name)
# Put current unit near the top while keeping past task evidence unchanged.
out={}
for k,val in state.items():
    out[k]=val
    if k=='current_stage':out[U.name]=entry
docs[2].write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}. 사용자 승인: 어깨·팔꿈치·손목 연결을 이용한 공통 좌우 손 배정 보완과 기존 문제 영상 검증. 학습은 결과를 사용자와 검토한 다음에만 진행.

## 마지막 완료
{summary}

## 정확한 판정
- 연구용 팔 연결 보완 구현/검증 완료, 실영상 추가 개선0. 운영 미채택이며 학습 준비 완료 선언이 아니다.
- 20영상 원본/crop40조건,5,072프레임·6,720검출 전부 직전joint v3와 동일. 기존 믿다2·냉커피1 회복과 기존 배정 보존.
- 미배정29raw 중18은 같은 쪽 대안/중복 억제,10은 초기 팔 이력 미확보,1은 경계 유효성 미달. 유효10건은7클립9프레임이며 과거 이력이 만료된 것이 아니라 아직 한 번도 만들어지지 않은 시점이다.
- 소복57·응접실46·팔십구43은 몸 손목 좌표가 아예 없는 것이 아니다. observed=true,finite,inbounds이지만 visibility기준 미달. 손 검출기 raw와 Pose 몸 손목의 품질을 혼동하지 않는다.
- 팔 위치 가설은3,586프레임에서 계산됐지만 실제 추가 회복0. 단어 예외/문턱 완화 없음. 새 코드 결과로 기존 track이나 arm anchor를 자기강화하지 않음.
- 독립67검사,별도 코드반례3건,40조건 순서반전 불변,보호1,248파일,출력140,511점/마스크/시간 보존 PASS. 21점 정확도나 수어 인식률 향상은 미검증.

## 다음 한 작업
{next_action}

## 근거와 담당
단위 `research/front_keyphase_v1/ops/{U.name}`. RESULT_KO.txt,DECISION.json,COMPLETION.json,INDEPENDENT_VERIFICATION.json,CAUSE_AUDIT.json,EMPIRICAL_REVIEW.md,VISUAL_REVIEW.md.
구현/최종문서/영상확인 root; 독립설계·코드·결과검토 armchain_review; 재생·원인·영상 armchain_replay; 독립검산 armchain_verify.
실행 완료,이번 후보의 진행 중 명령/학습 없음. 같은 재생을 중복 실행하지 않는다. 이전후보/원시자료/운영코드 보존.

## 유지되는 웹·모델 상태
기존 진단 URL http://127.0.0.1:8767/ 는이전joint v3(새후보와이번40조건동일)의 실제검출근거. 문서상서버 PID45392/exec6195,`ops/word100_hand_viewport_clarify_20261016/public`;이번작업에서서버재시작/수정없으며재개시실제상태확인.
새 팔 연결 표시판은이번단위 `PREREQUISITE_IMAGE_INDEX_v2.json`의소복57·응접실46. actualinput전체를표시,추가검출/crop변경없음.
현재crop은상단y=0을유지하지만좌우도잘림. 향후자료준비에서사용자의상단/좌우동작공간보존요구를확인한다.
TCN `runs/word100_tcn_v1/whole_context/best.pt` SHA {model_hash}. 기존 학습9명899/개발3명300,96.67%는개발검증이다. 사용자12회Top1 6/12·Top5 8/12는실사용기록이며학습·튜닝·후보선택금지.
TCN은몸9점+손42점280채널,어깨·팔꿈치·손목을이미포함한다. 이번코드는이관절을손배정의추가근거로사용하는연구후보이며모델입력의몸관절추가작업이아니다.
'''
docs[1].write_text(handoff,encoding='utf-8')
todo=docs[3].read_text('utf-8').replace('목표3건 보완·검증 완료 / 추가 표본 잔여 결함 상호 검토 대기','팔 연결 후보 검증 완료·추가회복0 / 초기 손 배정 잔여 문제 검토 대기')
todo+='\n\n## 2026-10-16 2단계 팔 연결 보완 결과\n\n'+summary+'\n\n다음 한 작업: '+next_action+'\n\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`.\n'
docs[3].write_text(todo,encoding='utf-8')

new_json(U/'ROOT_ACCEPTANCE.json',dict(status='PASS',timestamp=stamp,
    frozen_pins=len(frozen['pins']),model_sha256=sha(model),root_visual_review=['소복57 전후','응접실46 전후'],
    decision='No measured benefit, preserve research candidate, review before training',
    document_hashes={str(p):sha(p) for p in docs}))
new_json(U/'COMPLETION.json',dict(decision,authors=['root','armchain_review','armchain_replay','armchain_verify'],
    artifacts={str(p.relative_to(U)):sha(p) for p in U.rglob('*') if p.is_file() and '__pycache__' not in p.parts},
    documents={str(p):sha(p) for p in docs}))
assert read(docs[2])[U.name]['status']==decision['status']
assert read(docs[2])['next_single_action']==next_action
print(json.dumps({'status':decision['status'],'documents_updated':4,'model_sha256':sha(model),'training':0},ensure_ascii=False))
