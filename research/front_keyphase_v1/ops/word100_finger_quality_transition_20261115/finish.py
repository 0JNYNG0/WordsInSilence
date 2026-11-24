from quality_common import *
from datetime import datetime
from zoneinfo import ZoneInfo
v=read(U/'VERIFIED.json');assert v['artifact_checks_passed'];assert not read(U/'DECISION.json')['web_connected'];now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
report='''손가락 관측 혼선과 실제 손모양 변화 구분 — 2026-11-15

결론
공개 원본 정답 보강, 두 가지 학습 구조, 기존 방식과 별도 확인 자료 비교를 마쳤다. 일부 관절 혼선은 걸러졌으나, 새 영상에서도 도움이 된다는 근거가 부족하여 이번 후보는 웹에 연결하지 않았다. 기존 G6-finger-20261115는 그대로다.

무엇을 만들었나
손가락마다 두 가지를 따로 배우는 작은 시간 합성곱+GRU 판정부를 만들었다.
① 관절선이 실제 손가락을 따라가는가 / 옆 손가락 등 잘못된 부위를 따라가는가.
② 앞뒤 원본을 볼 때 손가락 모양을 유지하는가 / 실제로 굽히거나 펴는 중인가.
손이 이동·회전하는 것만으로 손모양이 바뀌었다고 가르치지 않았다. 두 항목은 별개라 동시에 발생할 수도 있다. 다만 이번 정답에는 같은 손가락에서 두 문제가 동시에 확인된 예가 0개이므로, 동시 발생 구분 성능을 입증한 것은 아니다.
기존 단어·행동 모델을 재학습한 것이 아니라, 기존 손가락 판정을 보조할 관측 판정부를 새로 지도학습한 실험이다. 입력은 손 크기로 정규화한 관절 관계와 앞뒤 7프레임이다. 단어명·수행자명·예상 정답 손모양을 모델 입력으로 주지 않았다. 양손/네 손가락이 공통 가중치를 쓴다. 엄지는 이번 범위 밖이다. RGB는 학습 정답 확인에만 사용했고, 실행 모델은 관절을 입력받았다.
오류로 의심되는 관절에서 나온 기존 손가락 판단을 '확인 불가'로 보류할 수만 있게 했다. 실제 좌표나 정답 손모양을 만들어 채우거나, 새로운 오답 카드를 만들지 않는다.

자료와 정답
- FIT: 공개 6명(01·03·04·06·08·09), 18영상, 4표현(화상·멀다·육십팔·팔십육)의 126시점. 같은 손모양도 손을 돌리거나 위치를 옮기는 정상 사례를 포함했다.
- 품질 정답 486개 중 명확한 관절 혼선 16개, 변화 정답 485개 중 실제 손가락 변화 70개. 1시점의 검지·중지·약지·새끼를 각각 세므로 영상 개수와 다른 수치다. 불명확한 정답은 제외했다.
- 개발14: 4영상 36시점, 손모양 143판정. 후보 선택에 사용.
- 별도 확인15: 4영상 40시점, 손모양 160판정. 후보/기준 고정 후 원본 정답을 만들고, 예측은 마지막에 한 번 확인했다. 이 결과로 재튜닝하지 않았다.
- 14·15는 조사한 기존 세부 손가락 후보 자료에 없었으나, 과거 TCN 평가에는 사용된 공개 자료다. 프로젝트 전체에서 처음 본 데이터라고 부르지 않는다.
- 정답은 Codex가 원본 RGB 앞뒤 장면과 관절선을 확인해 붙인 연구용 정답이다. 한국수어 전문가나 독립 사람의 검수는 아니다. 순간 동작 관찰이지 수어 언어적 정오의 보증은 아니다.
- 기존 100단어 공통 모델과 C·T3-F는 수정하지 않았다. 이번 보조 학습을 100단어 전체 정밀 정답 학습으로 설명하면 안 된다.

실제 비교 — 모두 '시점별 손가락 판단', 단어 인식률 아님
                  기존 맞음/오판/보류       새 품질보조 맞음/오판/보류
개발 143항목        123 / 6 / 14            123 / 4 / 16
별도 확인 160항목   141 / 1 / 18            141 / 1 / 18
기존 개발 967항목   910 / 1 / 56            870 / 1 / 96

개발 자료의 오판 2개는 정답을 알아낸 것이 아니라, 틀린 확신을 보류로 바꾼 것이다. 기존에 맞던 910개 중 40개(4.40%)도 추가 보류됐다. 별도 확인에서는 개선이 없었다. 그 확인 자료의 고정 시점에는 명확한 관절 혼선이 0개여서, 새 사람의 관절 혼선을 잘 거른다는 긍정 증거도 확보하지 못했다.
왼손은 개발40항목37맞음/3보류, 확인44항목38맞음/6보류로 그대로였다. 모두 '편 손→검지' 사례이며, 왼손 약지·새끼 독립 조합까지 검증한 것은 아니다.

학습 보완 과정
첫 구조(3seed)는 개발에서 관절 혼선 7개를 모두 놓쳤고, 모양을 유지한 12항목을 변화로 잘못 판단했다.
정답을 66→126시점으로 보강하고, 관절 품질과 모양 변화의 신경망을 분리한 두 번째 구조(3seed)를 학습했다. 품질 분기는 개발 혼선 7개 중 2개를 잡았으며 정상 관절의 혼선 오판은 0개였다. 변화 분기는 실제 변화 15개 중 7개를 잡았으나, 유지 7개를 변화로 오해했다. 별도 확인 변화17개 중5개 검출, 유지1개 오해였다. 전체 정확도만 보면 항상 '유지'라고 하는 단순 기준보다도 좋아졌다고 할 수 없다. 변화 분기를 웹에 쓰지 않았다.
품질+변화를 함께 적용하면 기존 맞음910개 중60개가 보류돼 사전 5% 손실 기준을 넘었다. 품질만 쓰는 후보를 먼저 고정했지만, 별도 확인 오판 감소 기준을 충족하지 못해 그 후보도 미연결이다.

남은 오류의 종류가 다르다
- 개발에서 남은 4개는 새끼손가락 관절이 약지 쪽을 따라가는데도 품질 분기가 잡지 못한 경우다. 정상처럼 연속되는 잘못된 관절선은 '갑자기 튄 값'만 찾는 것으로 해결되지 않는다.
- 별도 확인에 남은 1개는 팔십육 REAL15의 f35 새끼손가락이다. 원본 검토상 관절선이 명백히 다른 손가락을 따른 예는 아니었는데 기존 손모양 판정부가 '폄'으로 오판했다. 따라서 관절 품질 검사만으로는 모든 손모양 오판을 해결할 수 없다.
- 손을 카메라 쪽으로 돌릴 때 생기는 2D 모양 변화와 실제 손가락 굽힘을 변화 분기가 아직 혼동한다. 이 관찰은 실패 사례에 대한 설명이며 신경망 내부 원인을 완전히 증명한 것은 아니다.

정상과 기존 기능 보호
기존 공개 122시험 조건을 재생했다. 정상64조건에 새 손가락 교정 경고0, 기존 동작/방향/순서 판정과 손가락 상세20개 유지. 나머지에는 공개 영상을 편집·변형한 오류 시험이 포함되며 실제 학습자의 다양한 실수를 검증한 것은 아니다.
추가 두 분기 계산은 CPU에서 한 손 전체 시퀀스 기준 중앙값 약14ms였다. 모든 보조부 포함 재생 계산 중앙값 약92ms이며 RTMW·기존 본체·DTW·웹 표시·촬영 시간은 제외다. 전체 사용자 대기시간으로 설명하면 안 된다.
보호518파일 동일, 별도 저장 검산2727항목 통과. 이것은 절차/산출물 검사 통과이며 새 모델의 출시 기준 통과가 아니다. 구현과 저장 산출물 검산은 같은 Codex가 수행했다.
개인 촬영 접근·학습·튜닝·후보선택0, 카메라/새 서버 시작0, 다운로드0. 기존 웹/모델을 유지했으므로 사용자 촬영 동작이나 화면은 이번에 바뀌지 않았다.

다음 한 작업
새 후보를 또 임계값으로 맞추기 전에, 관절 품질이 나쁜 사례와 관절은 타당하지만 손모양 분류가 틀린 사례를 분리해 공개 연속 구간의 정답 범위를 보강한다. 다른 표현·다른 사람·왼손에서 실제 혼선 양성 사례가 포함된 별도 확인 묶음을 원본 검토로 먼저 확보한다. 현재 14·15·17·18 결과는 이후 독립 평가로 재사용하지 않는다. 아래 NEXT_PLAN.json에 구체적인 분기와 중단/통과 조건을 남겼다.

근거 파일
FIT_LABELS_V2.json / DIAGNOSTIC_LABELS.json / CONFIRMATION_LABELS.json
TRAIN_FROZEN*.json / TRAINED*.json / V2_RATIONALE.json
diagnostic_RESULTS_V2.json / confirmation_reserved_RESULTS_V2.json / HISTORIC_RESULTS_V2.json
ERROR_REVIEW.json / SHADOW_REPLAY.json / DECISION.json / VERIFIED.json
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
write(U/'NEXT_PLAN.json',dict(status='DESIGNED_NOT_EXECUTED',evidence=['ERROR_REVIEW.json','VERIFIED.json','DECISION.json'],next_single_action='보류 증가40항목과 남은 오판을 관절 혼선/손모양 분류/투영 변화로 분리하고, FIT 공개 연속 구간 및 별도 확인의 양성 품질 정답부터 확보',steps=[dict(id='A',action='실제 교정 문장을 만드는 지속 오류와 단일 시점 오류를 구분한다. 기존 최소 지속 조건을 그대로 적용해 사용자 피드백에 영향을 주는 실패를 먼저 분류한다.',evidence='이번143항목 오판감소2는 단일 손가락 관측 단위, 공개122최종카드는 전후동일. 이 차이를 숨기지 않는다.'),dict(id='B',action='기존 FIT01..09의 다른 표현에서 정상 밀착/겹침, 잘못 붙은 관절, 실제 굽힘, 손 회전을 원본 연속 화면으로 구분. 왼손 독립조합·검지 전면·서로 다른 손가락 오류 포함. 독립 확인 묶음은 모델 예측 없이 먼저 양성/음성 지지량 확인.',gate='동일 손가락 품질오류와 실제 변화의 동시 사례가 없으면 그 능력 주장 금지. 충분한 공개 양성이 확보되지 않으면 추가학습/교정범위확대 보류. 사람/원본클립 단위 분리.'),dict(id='C',action='원인이 품질이면 품질 분기, 정상 관절의 잘못된 모양이면 기존 손가락 분류기, 투영 회전이면 변화 분기를 해당 FIT 정답으로 학습. 좌표만으로 구분 근거가 겹치면 정답 좌표를 상상해 보간하지 않고 불확실로 남긴다.',gate='단어별 예외·정답명 입력·14/15/17/18 재튜닝 금지. 새 대형 영상 모델/다운로드를 자동 추가하지 않는다.'),dict(id='D',action='새 확인에서 틀린 교정 감소와 정상 판단 유지, 실제 표시되는 문장/강조/시간을 함께 검증한 후보만 웹 연결.',gate='관측 오판 감소만으로 완료 선언하지 않는다. 실제 새 사용자 카메라는 사용자가 재개할 때 별도 확인.' )],constraints=['C/T3F/common100 and current web preserved','No private footage in any training/tuning/selection','No new HTML if no clear improvement']))
write(U/'FAILURES.json',dict(scientific=['v1 no wrong-chain detection;12stable falsely changing','v2 quality only reduces diagnostic2 but not reserve1;40correct historic cells lost to unknown','v2 combined loses60historic correct, exceeds5% gate','reserve fixed anchors contain0confirmed wrong-chain positives, cannot demonstrate sensitivity','no independent KSL expert labels; same-finger simultaneous quality+change positive0'],implementation_resolved=['extract missing model_pins before extraction','review syntax else112; duplicate raw_sha256; source fallbacks for alreadycached public sources','local file-name prefix missing in view_image call, corrected','reserve_labels missing closingbrace; dependent evaluation failed before predictions; fixed and executed once successfully'],unaffected=['No archived result or frozen input/model changed','No user recordings used','No camera or server started'],host='Get-CimInstance denied; extraction/training completions confirmed by tool sessions and output manifests; no duplicate long run'))
summary='관절 혼선/실제 손가락 변화를 따로 감독하는 공통 Conv+GRU2구조×3seed 학습·비교 완료. FIT6명18영상126시점. 품질보조 개발143항목 오판6→4(정답123유지), 별도15확인160항목1→1. 기존개발910맞음중40추가보류; 함께쓰면60보류라5%초과. 변화분기 회전혼동 잔존. 확인품질양성0으로 일반화근거부족;웹미연결 G6-finger-20261115유지. 공개122재생 정상64새경고0,상세20/기존판정유지. 보호518/저장검산2727PASS. C/T3F/common100불변,개인자료0;독립전문가검수아님.'
state=read(O/'STATE.json');state['current_followup']=U.name;state['next_single_action']=read(U/'NEXT_PLAN.json')['next_single_action'];state[U.name]=dict(status='COMPLETE_COMPARISON_NOT_ADMITTED',at=now,summary=summary,report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFIED.json'),decision=str(U/'DECISION.json'),next_plan=str(U/'NEXT_PLAN.json'),web_changed=False,user_recordings_used=False,pid=None,exec_session=None);write(O/'STATE.json',state)
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
old='### 2026-11-15 손가락 관측 혼선·변화 분리 작업 진행\n\n공개 원본으로 관절 혼선과 손모양 전환 정답을 분리 보강→작은 공통 관측 판정부 학습→미사용15확인→정상회귀→통과 기능만 연결. `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`. 학습01..09/진단14/예약15, 이전17/18 재튜닝 금지. C·T3-F·사용자자료 보호.'
assert doc.count(old)==2
doc=doc.replace(old,'### 2026-11-15 관절 혼선·손모양 변화 분리 학습 비교 완료 / 웹 미연결\n\n'+summary+'\n\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `ERROR_REVIEW.json`, `DECISION.json`, `VERIFIED.json`.',1)
doc=doc.replace(old,'### 2026-11-15 품질 오류와 모양 분류 오류를 분리한 연속 정답 보강\n\n'+state['next_single_action']+'. 공개 원본으로 먼저 확인하고, 다음 새 독립 확인에는 실제 관절 혼선 양성이 있는지 예측 전에 검토한다. 현재14/15/17/18은 이제 노출자료이며 독립확인으로 재사용하지 않는다. 동결된 이번 두학습/확인/122회귀 반복금지. 현웹/C·T3-F/100단어 공통학습은 유지, 개인자료 사용금지. 다음세부계획은 `research/front_keyphase_v1/ops/'+U.name+'/NEXT_PLAN.json`(미실행).',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
handoff='''# 관절 혼선·손모양 변화 학습 비교 완료 / 이번 후보 미연결

'''+summary+'''

결과: `research/front_keyphase_v1/ops/'''+U.name+'''/RESULT_KO.txt`와 DECISION.json/VERIFIED.json.
현재 후보 고정 뒤 예약15 예측을 완료했다. 14·15·17·18은 이제 손가락 연구에 노출됐으므로 새 독립 평가/튜닝으로 쓰지 않는다. 기존 P/H 동결 산출물 보존.

다음 한 작업: '''+state['next_single_action']+'''. NEXT_PLAN.json은 설계만, 아직 미실행. 먼저 실제 교정에 영향을 주는 지속 오류와 개별 시점 오류를 구분한다. 신규 대형 모델/외부 다운로드/사용자 자료 사용 없음.

이번 실행: 두 구조 각3seed=6학습 종료, 예약확인 완료, 공개122회귀 종료. 진행 프로세스/exec 세션 없음. 이번에 서버/카메라 시작 안 함; 기존 웹 G6-finger-20261115 유지. 실제 사용자 촬영 확인은 사용자 보류 상태 그대로. HTML 미생성.
구현·학습·원본 RGB 주석·저장 검산 모두 Codex. 독립 한국수어 전문가 검수는 아님. 정답은 연구용 관찰 라벨이고 언어적 올바름 보증은 아님.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8');write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_COMPARISON_NOT_ADMITTED',at=now,pid=None,exec_session=None,web_changed=False,next_plan_status='DESIGNED_NOT_EXECUTED'))
print(summary)
