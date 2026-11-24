from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib
V=Path(__file__).resolve().parent;U=V.parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
verify=read(V/'VERIFICATION.json');replay=read(V/'REPLAY.json');live=read(V/'LIVE_API_CHECK.json');capture=read(V/'CAPTURE_API_CHECK.json');candidate=read(V/'CANDIDATE_RESULTS.json')
assert verify['passed']and replay['passed']and len(replay['rows'])==30
assert live['state']=='done'and live['completed']==[True,True]
assert len(capture)==2 and all(r['new_shared_guard']and not r['quest_advanced']for r in capture)
assert read(V/'STATE_CHECKS.json')['passed']and read(V/'DIAGNOSTIC_CHECKS.json')['passed']
assert not candidate['admission_pass']
for r in read(V/'PROTECTED_WEIGHTS.json'):assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
files=[O/'word100_finger_repair_20261115/observation_quality.py',O/'word100_finger_repair_20261115/finger_runtime.py',O/'basic_demo_feedback_20261115/shape_consensus.py',U/'coach.py',U/'diagnostic_ledger.py']
record=dict(at=at,status='PUBLIC_CAUSES_REPRODUCED_SHARED_NUMERIC_GUARD_APPLIED_MODEL_UNCERTAINTY_REMAINS',
 implementation='Codex root',verification='Codex separate regression; previous independently authored state tests rerun by root. No new agent/expert review.',
 code={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},new_training=False,private_data_read=False,private_training=False,
 shared_input_guard='Collapsed palm axis or all finger-chain points coincident are unavailable, not flexed. Nondegenerate geometry is NOT proof of correct tracking.',
 model_vote_change_admitted=False,candidate_reason='Removing original coarse veto raised historical definite mistakes1->3; retain current combination.',
 public_regression=dict(source_videos=27,frames=3270,hand_finger_decision_changes=0,live_conditions=30,source_note='Prior public sources/controlled variants, not new-user accuracy'),
 web=dict(capture_end='8821 new shared guard seen through two public API replays',live='8822 public replay completes2/2 with separated model reasons',physical_camera=False),
 remaining=['Unknown hand shapes not solved. Known RTMW adjacent-finger misassignment remains.','Old/new hand classifiers disagree on normal extended neighboring fingers.','Current numerical guard does not reconstruct points or improve linguistic recognition by itself.','No claim user frustration or personal33% uncertain resolved.','Hello remains single-reference guided prototype, not full-sign correctness/new-user generalization.'])
write(V/'VERIFIED.json',record)
write(V/'ADMITTED.json',dict(at=at,status='public_regression_passed',scope='shared numerical observation guard and bounded operational diagnostic reasons only',files=record['code'],weights_changed=False,thresholds_changed=False,whole_sign_accuracy=None))
source_review=dict(reviewer='Codex visual review, not independent sign expert',cases=[
 dict(clip='SEN0354_REAL01',frames=[36,42,48,51,54,60,66,75],finding='RGB shows open sweep, transition, then closed long-finger posture. Selected stable intervals all observed correctly in4 view variants.',images=['native_review.jpg','letterbox_review.jpg']),
 dict(clip='WORD2700_REAL10',frame=66,finding='Folded little finger is assigned points near adjacent extended finger; current observer outputs all4 extended. Existing source review confirmed on RGB/overlay. Guard does not solve this.',image=str(O/'word100_finger_error_split_20261115/review/diagnostic/NIA_SL_WORD2700_REAL10_F_1.jpg')),
 dict(clip='WORD1290_REAL16/17',frames=[75,95,102],finding='Visible long fingers extend with neighboring fingers together. Overlay approximately follows respective visible chains; precise3D truth unavailable. Original and expanded coarse observers confidently disagree; current result abstains.',image='model_disagreement_review.jpg'),
 dict(clip='WORD2551_REAL12',frame=35,finding='Previous source audit reconfirmed short actual index extension with neighboring frames bent. Visible index chain approximately follows actual finger. Known classifier error remains; not claimed repaired.',image=str(O/'word100_finger_error_split_20261115/review/multiview/historic_overlay.jpg'))],
 limits='No true joint-quality automatic oracle. Model confidence/disagreement is operational diagnosis, not proof that coordinates are right. Synthetic collapsed-chain probe is a separate numerical defect and not the demonstrated cause of prior user stress.')
write(V/'SOURCE_REVIEW.json',source_review)
next_action=('두 사용 방식은 유지하되 손모양 공통 판정을 먼저 보강한다. 공개 학습 수행자의 나란히 편 손가락·굽힘·실제 짧은 변화 정답을 관절 품질과 분리하여 보강하고, 기존/보강 판정부가 엇갈리는 경계를 추가 학습할지 고정 비교한다. '
 '이미 본 REAL16/17을 새 최종검증으로 재사용하지 않는다. 정상과 실제 다른 손모양을 모두 유지하는 후보만 촬영후/실시간 양쪽에 연결한다. '
 '8822 실시간 단계규칙·상하머리 보완은8821에 자동 적용된 것으로 말하지 않는다. 개인 촬영 학습·튜닝·선택 금지, C·T3-F 유지.')
report='''손모양 불확실 원인 분리와 두 사용 방식 정리

이전 작업은 무의미해지지 않았다.
8821 촬영 후 평가: 촬영하면서 RTMW관절 저장 → 촬영 종료 → TCN단어 후보 / 학습된 행동구간·방향·손가락 관측 / DTW비교재생 / 교정 안내.
8822 실시간 연습: 지금까지 들어온 RTMW관절 → 짧은 과거창의 손모양·Conv+GRU방향 관측 → 설명 기반 단계진행 규칙 → 즉시 색상·안내. 단어TCN은 쓰지 않는다.
공통 RTMW입력·100공통 행동인코더 계보·개별손가락 관측을 재사용한다. 두 모드의 방향 판정부, 행동구간 연결·완료 규칙, DTW사용은 같지 않다. 최근 실시간 상하 방향 학습과 진행조건 수정이 기존 평가에 자동 적용된 것은 아니다.
이번 공통 손가락 입력검사는 양쪽 코드와 재시작한 서버에 연결했다. 별도 작은 손모양보조는8822/기초단어 연구용이며8821전체 판정을 대체하지 않는다.

공개 원인 확인
1. 안녕하세요 한 원본의 native/640letterbox/상반신crop/별도crop에서 선택한 안정 손모양52손시점씩은 모두 기대모양이었다. 일부 카메라크기로 바꾼 것만으로 사용자 모양불확실을 재현했다고 하지 않는다.7관측창과전체영상 계산도16대응시점씩 같았다.
2. 공개 육십팔 WORD2700 REAL10 f66 등에서는 접힌 새끼손가락 점이 옆 펼친 손가락을 따라가 모델이 펼침으로 읽었다. 관절이 틀린 입력 원인. 아직 이 실제 오추정을 복구하지 못했다.
3. 감사 REAL16/17 공개17손시점 중5보류를 재현했다. 비교적 손을 따라간 점에 대해 이전 손한쪽 모델은 굽힘,보강모델은 폄으로 서로 확신하여 합의가 막힌다. 단순 관절 미검출과 다르다. 원본·오버레이를 재검토했으며 수어 전문가의 독립검수가 아니다.
4. 순수 분류문제의 기존 짧은 검지폄 WORD2551 REAL12 f35도 원본 검토와 이전 진단을 확인했다. 앞뒤 문맥이 실제짧은모양을 덮는 문제는 여전히 남는다.
5. 손모양을 바꾸는 도중의 불확실은 무조건 오답이나 잘못된 관절로 정답붙이지 않는다.

실제로 고친 부분
관절점이 화면 안에 있다는 이유만으로, 손가락4점이 같은 점에 뭉친 입력까지 사용하던 결함을 보완했다. 이전에는 검지4점을 같은 곳에 놓아도 굽힘으로 확정했다. 이제 해당 손가락은 관측불가,손전체 합의로도 이를 억지로 채우지 않는다. 손바닥 기준축이 사라진 입력도 보류한다. 크기·거리·굽힘각도의 임의임계값이 아니라 수치적으로 길이가 없는 입력을 제외한다.
관절점이 정상처럼 보이지만 실제 옆 손가락을 따라가는 경우까지 검출해내는 기능은 아니다. 수치검사통과=정확한관절이라는 뜻이 아니다.
RAM진단은 관절정보 부족/수치적으로 모양을 알 수 없음/모델의 확신 부족/모델 간 의견 불일치를 구분한다. 사용자 스켈레톤·영상·확률배열은 진단에 저장하지 않는다. 원인 이름은 운영상태이며 사용자가 실제 틀렸다는 정답이 아니다.

시도 후 제외한 변경
기존 손모양모델의 거부를 없애면 공개기존237손시점에서227맞음/1오판/9보류 →231맞음/3오판/3보류였다. 새로운 오판2개가 생겨 적용하지 않았다. 별도 기존공개10/16자료에서는 일부 보류감소가 있었지만, 좋은 숫자만으로 교체하지 않았다. 추가학습·임계값변경 없이1개정해진조합만 비교했다.

검증과 적용
공개27영상3270프레임의 양손 기존손가락/손모양/가용성 판정은 모두 유지. 기존실시간28조건과 양손각각의검지점붕괴2조건=30조건 통과.74입력·모델보호 검사,기존독립작성 상태34개를root재실행,진단4+기존14검사 통과.
8821 공개 순종·잘하다 API분석에 새공통입력검사 버전 확인,공개재생으로 퀘스트를 진행시키지 않음.8822 공개인사재생2단계완료와 모델불일치/확신부족별 진단확인. 실제카메라를열지 않았다.
새모델학습/가중치변경 없음. C·T3-F/기존방향머리/100공통인코더 보존. 이번 작성·검증은Codex root;새에이전트 또는 전문가 독립검토는없다.

남은 한계와 방향
이번은 입력안전성과 원인진단 개선이다. 사용자 촬영의33%불확실 해결,실사용편의성 향상,전체수어정확도 향상으로 표현하지 않는다.
제출 목표는 계속 '사용자의 동작에 근거있는 교정을 주는 게임'이다. 실시간은 배우는 중의 도움,촬영후평가는 도움없이 한 번 수행한 결과 확인에 둔다. 두개의새제품을따로만들기보다 공통손모양·방향·관측판정을 먼저 검증하고 각모드에서 쓰도록 한다.
다음 작업은 공개 정상폄/굽힘/짧은변화의 관측품질-손모양 정답을 분리하여 공통판정부의 불일치 원인을 보강하는 것. 정상확인범위를 줄여 통과시키거나,사용자촬영으로 정답을 맞추지 않는다. 동시에 안녕하세요단일참고의 전체정답·다인일반화 한계와 기존 촬영후평가 미완료사항을 별도 유지한다.
'''
(V/'RESULT_KO.txt').write_text(report,encoding='utf-8')
graph=read(U/'TASK_GRAPH.json');graph['latest_repair']=str(V/'VERIFIED.json');graph['handshape_cause_review']=[
 dict(id='H1',label='관절/모델/모양전환 원인 분리',status='complete_public_diagnostic'),
 dict(id='H2',label='점 붕괴 입력 보류·원인진단',status='complete'),
 dict(id='H3',label='공개 회귀·두 경로 연결',status='complete'),
 dict(id='H4',label='정상 손모양의 모델 불일치 해결',status='remaining'),
 dict(id='H5',label='실제 사용자 편의성·시연 완료',status='not_claimed')]
graph['mode_contract']=dict(shared=['RTMW/normalized joints','individual finger observer','frozen common action encoder lineage'],capture_end=['TCN','full episode association','DTW playback','correction evidence'],live=['received-prefix learned direction','guided stage controller','immediate visual cues'],automatic_cross_mode_training=False)
write(U/'TASK_GRAPH.json',graph)
server=read(U/'SERVER.json');server.update(pid=35552,launcher_pid=23820,started_at=at,execution='normal',latest_verification=str(V/'VERIFIED.json'),camera_opened=False);write(U/'SERVER.json',server)
state=read(O/'STATE.json');state['last_updated']=at;state['next_single_action']=next_action
state['hello_handshape_cause_review_20261116']=dict(status=record['status'],report=str(V/'RESULT_KO.txt'),verified=str(V/'VERIFIED.json'),graph=str(U/'TASK_GRAPH.json'),new_training=False,web_ports=[8821,8822],current_pids={'8821':33996,'8822':35552},user_recordings_used=False,next_single_action=next_action)
if 'hello_live_coach_20261115'in state:state['hello_live_coach_20261115']['latest_followup']=str(V/'VERIFIED.json')
write(O/'STATE.json',state)
context=R/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text(encoding='utf-8');note='''
### 2026-11-16 손모양 불확실 원인 분리·공통 입력 보완 / 두 사용방식 정리
공개 인사4화면조건의 핵심모양은 유지. 감사16/17의5보류는 기존/보강손모양모델의 의견불일치 재현,육십팔10의옆손가락잘못추적도 원본재검토. 기존거부제거 후보는237손시점227/1/9→231/3/3(맞음/오판/보류)로새오판2발생하여미적용. 추가학습없음.
손가락점/손바닥축이한점으로붕괴해도정상입력처럼쓰던수치결함을공통부에서보류처리. 실제옆손가락추적복구는아님. 모델불일치/확신부족/수치관측부족 RAM진단분리.27공개영상3270프레임손모양/개별손가락판정유지,기존28+붕괴2=30실시간조건통과.74검사/이전독립작성상태34root재실행/진단4+14.원본모델/C·T3-F유지·개인자료미사용.
8821/8822대기상태확인후재시작,8821공개2영상새공통검사버전·8822공개2단계완료확인.실제카메라미개방.현재listener33996/35552.공통손가락부수정은양쪽연결;최근실시간방향학습/진행규칙은기존촬영후평가에자동반영된것아님.각모드는연습중도움/종료후평가로유지.불확실해결·사용편의성·전체완료는미확인.
근거 hello_live_coach_20261115/handshape_cause_review_20261116/VERIFIED.json·RESULT_KO.txt.구현/검증Codex root,새에이전트·전문가검토없음.

'''
s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'+note,1)
s=s.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-11-16 공통 손모양 판정 후속\n'+next_action+'\n\n',1);context.write_text(s,encoding='utf-8')
handoff=f'''# 최신 인수인계 — 한국수어 피드백 게임

## 현재 {at}
손모양 불확실을 공개 원본/관절/모델 출력으로 분리했다. 안녕하세요 기존4화면조건에서 안정모양52시점씩은 맞았다. 감사REAL16/17의17정답중5보류는 이전/보강손모양모델 불일치,육십팔REAL10 새끼는옆손가락을잘못추적했다. 이실제오추정과모델불일치는아직남음.보류를없애는단일후보는기존오판1→3으로미적용.

## 이번 적용
공통FingerObserver에서손가락4점또는손바닥기준축이한점으로뭉친입력을관측불가처리.단순붕괴가아닌잘못된관절을자동복구한것아님.두경로공유코드반영및서버재시작.실시간RAM진단에model_disagreement/model_confidence_insufficient/geometry원인분리.정확한관절판정기아님.
공개27영상3270프레임기존손모양/손가락유지,실시간28기존+붕괴2=30통과.74자체검사·이전독립작성상태34root재실행·진단4+14.가중치/임계값/새학습없음.기존C·T3-F/100공통/손모양/방향모델보존.사용자촬영읽기·학습·튜닝·후보선택없음.

## 두 사용방식 — 혼동 금지
8821 촬영하면서RTMW추출→종료후TCN/행동구간·방향·손가락/DTW재생/피드백.
8822 실시간RTMW→과거관측의손모양/공통Conv+BiGRU+별도방향머리→설명기반2단계규칙/즉시안내.단어TCN미사용,전체정답whole_word_correct=null.
공통개별손가락판정과인코더계보재사용.최근8822의상하방향학습·진행조건이8821에자동적용된것아님.현재실시간을기존전체평가대체완성품으로말하지않는다.이번수치입력보호만양쪽공통반영.

## 실행과 검증
8821 listener33996(launcher36772),8822 listener35552(launcher23820),일반실행/카메라버튼전미개방.현재SERVER.json과실제포트확인후이어받기.
8821공개순종·잘하다API에서numeric-v1-20261116확인,퀘스트강제진행없음.8822공개재생done/2단계완료,새모델불일치진단확인.실제웹캠편의성개선은미확인.
결과단위 research/front_keyphase_v1/ops/hello_live_coach_20261115/handshape_cause_review_20261116/
RESULT_KO.txt·VERIFIED.json·SOURCE_REVIEW.json·CANDIDATE_RESULTS.json·VERIFICATION.json·REPLAY.json·CAPTURE_API_CHECK.json·LIVE_API_CHECK.json.로그server8821/8822.out/err.log.新상하모델적용은상위seated_diagnosis_20261116/VERTICAL_ADMITTED.json유지.
작성/실행/검증Codex root,이번새에이전트/전문가검토없음.과거결과파일덮어쓰지않음.

## 다음 한 작업
{next_action}

## 남은 목표/한계
게임기초목표 안녕하세요SEN0354/감사WORD1290/사과WORD0099,기존4단어공통회귀.인사다인원본미확보(1참고만),W1TB는WORD묶음.엄지/정확한손방향/접촉/다인전체정답/실시간10fps짧은동작한계.실시간사용자5세션의모양불확실33%는오류율아니며이번수치수정으로해결됐다고하지않음.그5회스칼라진단은seated_diagnosis의DIAGNOSTIC_ONLY.json에진단용보존;학습에금지.이전학습은public상하방향머리400step,이번학습0회.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
print('RECORDS VERIFIED AND UPDATED')
