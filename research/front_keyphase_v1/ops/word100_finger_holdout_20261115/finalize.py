from holdout_common import *
from datetime import datetime
from zoneinfo import ZoneInfo
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();v=read(U/'VERIFIED.json');assert v['passed'];selection=read(U/'SELECTION.json');confirmation=read(U/'CONFIRMATION.json');replay=read(U/'SHADOW_REPLAY.json');assert selection['selected']=='unchanged'
cause=dict(diagnostic=[dict(clip_id=r['clip_id'],frame=r['frame'],finger='little',truth='visibly flexed',previous='extended',image=str(U/'ERROR_OVERLAY.jpg'),finding='RGB shows flexed little finger; RTMW distal chain follows nearby extended ring-finger region. Shared coordinate input can mislead multiple observers. No exact corrective coordinates are fabricated.')for r in read(U/'diagnostic_old/RESULTS.json')['rows']if r['wrong']],reserved=[dict(clip_id='NIA_SL_WORD2718_REAL18_F',frame=35,fingers=['index','middle'],image=str(U/'RESERVED_ERROR.jpg'),finding='RGB sequence changes extended→flexing; binary observer still reports extended at the annotated flexing anchor. This is a transition-observation mismatch, not evidence that the public sign itself is incorrect. Exact causal contribution of temporal pooling versus 2D projection not isolated.')],input_scores='RTMW raw score is not a calibrated probability; no invented confidence threshold applied',learned_limits='Adding32FITanchors improved some diagnostic outputs but not reserved errors. Correcting a confidently hallucinated joint chain is not guaranteed by more pose-only labels.',reviewer='Codex RGB+raw-overlay inspection; not independent KSL expert; sparse long-finger posture labels, no semantic correctness labels')
write(U/'ERROR_CAUSES.json',cause)
search=read(U/'LEFT_COVERAGE_SEARCH.json');search['findings']={'WORD1486':'Right hand only in viewed active samples; no new left finger combination','WORD2309':'Right hand only in viewed active samples; no new left finger combination','WORD0678':'Right hand only in viewed active samples; no new left finger combination','WORD2572':'Left extended long fingers; no clearly separate ring/little flexion combination in viewed samples'};search['limits']='Twelve RGB samples per4clips, not exhaustive full-dataset search. No claim that other left combinations do not exist. No extra training/prediction selection from these clips.';write(U/'LEFT_COVERAGE_FINDINGS.json',search)
write(U/'FAILURES_AND_REPAIRS.json',dict(items=[dict(kind='implementation',problem='Preparing new sample availability accidentally used21point visibility instead of4finger mask; shape assertion stopped before training.',repair='Use same4finger features availability as old pipeline; rerun prepared successfully; no mislabeled optimization ran.'),dict(kind='measurement',problem='Front-view category totals include all4longfingers and would overstate index recognition.',repair='Report index-only10decisions:4correct6abstained0wrong; keep all-finger group metrics separately.'),dict(kind='candidate',problem='New standalone fine-tuned head reduces diagnostic5errors to2 but historicalDEV1→11errors and reserve2→5errors.',repair='Do not replace observer; evaluated contradiction-only and required-agreement supplements.'),dict(kind='integration',problem='Required-agreement supplement diagnostic5→2errors but reserved2→2; historicalcorrect910→909, reservedcorrect217→214.',repair='Predeclared diagnostic zero-error gate not met. No new model or threshold connected; original web code unchanged. No post-reserve tuning.'),dict(kind='coverage',problem='Left labels cover open/index configurations, not isolated ring/little left-hand mixtures.',repair='RGB-only search4additionalpublicclips; no suitable new distinct left mixture confirmed in selected samples; leave this scope explicitly unverified.')]))
next_action='기존 정답을 유지하면서 손가락 관측의 혼선과 자세 전환을 따로 표시할 학습 정답을 FIT 공개자료에서 만들고, 검증에 노출되지 않은 별도 자료를 예약한 뒤 관측 신뢰도 판정부를 비교한다. 왼손 약지·새끼 독립 조합은 실제 공개 원본을 확보해 별도로 검증한다. 단어별 예외·17/18 재튜닝·웹 교정 확대 금지.'
text='''공개 손가락 검증·보완 학습 결과 (2026-11-15)

결론
요청한 공개 검증, 원인 확인, 추가 학습, 기존 방식 비교를 완료했다.
새 모델은 통과 기준을 충족하지 못해 웹에 연결하지 않았다. 기존 제한 손가락 보조와 C·T3-F/공통100모델은 그대로다.

무엇을 확인했나
- 세부 손가락 후보 선택 자료에 없던 공개16영상(8표현×수행자17/18)을 사용했다. 이 영상들은 과거 TCN 평가 이력이 있으므로 프로젝트 전체에서 처음 쓰는 자료는 아니다.
- 원본 RGB에서12개 고정시점을 각 영상별로 먼저 살펴봤다. 192개 후보시점 중 모양을 정할 수 있는120시점에 손가락474개 판정 정답을 남겼다. 가려지거나 애매한 항목은 미지정했다. 전체 연속영상은 모델에 입력했지만 모든 프레임에 정답을 붙인 시험은 아니다.
- 수행자17의8영상은 원인 진단,18의8영상은 수정안 고정 뒤 확인에만 사용했다. 둘 다 학습에 넣지 않았다.

현재 방식 결과
- 왼손: 편 손/검지만 편 손 조합에서153개 손가락 항목 중140개 정답,13개 보류,0개 오판. 약지와 새끼가 서로 다르게 굽은 왼손 조합까지 검증한 결과는 아니다.
- 정면을 향한 검지 자체:10시점 중4정답,6보류,0오판. 17은5시점 모두 보류,18은4정답·1보류였다. 한 사람의 연속시점은 독립된 새 사람 시험이 아니다.
- 전체: 진단225개 항목181정답/5오판/39보류, 별도 확인249개217정답/2오판/30보류.

남은 오판의 근거
1. 굽힌 새끼손가락의 RTMW 관절이 옆의 펴진 약지를 따라 잘못 잡혔다. 잘못된 좌표를 여러 모델이 함께 믿는 문제가 있었다.
2. 손가락을 굽히기 시작한 중간 시점을 아직 편 상태로 읽는 경우가 있었다. 공개 수어 자체가 틀렸다는 뜻은 아니다. 시간창 영향과 정면 투영의 기여를 아직 분리해 증명하지 못했다.
원본/관절 비교: ERROR_OVERLAY.jpg, RESERVED_ERROR.jpg. ERROR_CAUSES.json에 시점과 범위를 기록했다.

무엇을 학습했나
- 이전 FIT 수행자03/06/09의 공개6영상에서 부족한 손가락 조합32시점을 RGB로 확인해 추가했다. 기존 정답688시점을 보존해 총720시점 중 FIT476시점만 최적화에 사용했다(개발244시점은 제외).
- 기존 손가락 시간합성곱+양방향GRU 가중치3개(seed42/43/44)를 각300회 미세조정했다. 기존 정규화·신뢰기준을 고정했다. 사람·단어 이름·목표 정답은 손가락 관측기의 입력이 아니다. 엄지는 이번 판정에서 제외했다.
- 공통100단어 뇌 전체를 다시 학습한 것이 아니라 작은 손가락 관측부만 조정한 시험이다.

이전과 비교
                          이전   새 단독모델   기존+합의 확인
진단 자료 오판              5        2              2
기존 개발자료 오판          1       11              1
따로 남긴 확인자료 오판     2        5              2
합의 확인은 기존 판정을 정답으로 바꾸는 기능이 아니라, 서로 다르면 보류하는 기능이다. 진단 오판 감소를 실제 정답 증가로 설명하지 않는다.
개발자료의 기존 정답1개와 별도 확인의 기존 정답3개가 합의 확인에서 보류됐다. 별도 확인 성능 개선도 없어 웹에 추가하지 않았다. 확인자료 점수를 본 뒤 임계값을 바꾸거나 다시 학습하지 않았다.

정상 동작과 기존 기능
- 기존 공개122조건(정상·속도/교차 조건64 포함)을 보완 후보로 다시 계산했다. 정상의 새 손가락 경고0, 기존 방향·순서·구간 판정 유지. 기존 손가락 상세20개도 유지됐다.
- 오류 조건에는 공개 관절에서 만든 인위적 오류가 포함된다. 실제 사용자의 잘못된 수어 검출률이라고 말하지 않는다.
- 추가 움직임+손가락 계산 중앙값 약0.077초,95백분위 약0.103초였다. 관절 추출·DTW·촬영·화면 표시는 제외한 보조 계산 시간이다. 새 모델은 배포하지 않아 실제 웹 속도가 바뀌었다고 주장하지 않는다.
- 보호파일518개 해시 동일. 사용자 촬영은 열거나 학습·튜닝·선택에 사용하지 않았다. 웹 코드·모델 연결 그대로, 카메라와 서버 새 실행 없음.

남은 범위와 다음 한 작업
왼손 약지/새끼 독립 조합, 정면 검지의 보류, 관절 혼선, 자세 전환 시점은 남았다.
더 많은 단어를 무조건 늘리기보다, 올바르게 검출된 손가락과 옆 손가락을 잘못 따라간 관측을 구분하는 정답, 펴짐→굽힘 전환 정답을 FIT 자료에서 따로 마련하는 것이 다음 순서다. 실제 원본으로 확인할 수 없는 손가락을 상상해 보충하지 않는다. 검증에 노출된17/18영상으로 추가 튜닝하지 않는다.

구현/학습/별도 산출물 검산: Codex. 독립 수어 전문가 검수는 아님.
VERIFIED.json 616검사 통과는 실험 기록·분리·보호·회귀 검사 통과이며 새 모델 배포 통과가 아니다.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
write(U/'CONNECT_GATE.json',dict(new_models_admitted=False,current_web='G6-finger-20261115 unchanged',reason='No eligible diagnostic candidate before reserve release; reserve confirms no improvement of conservative guard',web_files_unchanged=True,C_T3F_unchanged=True,common100_unchanged=True,user_recordings_used=False,source_of_decision=str(U/'SELECTION.json'),confirmation=str(U/'CONFIRMATION.json')))
write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETED_VALIDATION_TRAINING_COMPARISON_NO_NEW_WEB_MODEL',at=now,pid=None,active_training=False,camera_started=False,next_action=next_action))
state=read(O/'STATE.json');state['current_followup']=U.name;state['next_single_action']=next_action;state[U.name]=dict(status='COMPLETED_NO_NEW_MODEL_ADMITTED',at=now,report=str(U/'RESULT_KO.txt'),verified=str(U/'VERIFIED.json'),plan=str(U/'FROZEN.json'),sources=16,extra_fit_anchors=32,front_index=dict(correct=4,unknown=6,wrong=0,total=10),left=dict(correct=140,unknown=13,wrong=0,total=153),web_changed=False,protection_count=518,next_action=next_action);write(O/'STATE.json',state)
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
current='''### 2026-11-15 손가락 후보 선택 밖 공개 검증 완료 · 새 모델 미연결

공개16영상(8표현×17/18) RGB 정답을 먼저 고정했다. 진단17과 확인18을 분리하고, FIT3/6/9의6영상32정답을 추가해 손가락 합성곱+GRU3seed를 미세조정했다. 기존개발967항목의 오판1→단독후보11로 증가, 별도확인249항목도2→5라 단독교체 불가. 합의보조는 진단5→2오판이나 확인2→2, 보류증가로 미연결. 기존 웹 보조 유지. 정면검지 자체10시점4정답/6보류/0오판(다른손가락을분모에섞지않음), 왼손 편손·검지조합153항목140정답/13보류/0오판; 왼손 약지·새끼 독립조합은 미검증. 원인은 새끼 관절이 옆약지를 따르는 관측 혼선 및 굽히기 전환 시점의 잘못된 상태. 정확한 모델 내부 원인까지 증명한 것은 아님.

공개122조건 재생, 정상64새경고0, 기존 판정/상세20유지. 보호518동일, 사용자자료 미사용. 과거TCN평가자료임을 공개; 프로젝트전역미사용 아님. 구현/학습/별도검산 Codex, 독립전문가 검수 아님. 새 서버/카메라 시작 없음, 사용자촬영 확인은 계속 보류.
근거: `research/front_keyphase_v1/ops/word100_finger_holdout_20261115/RESULT_KO.txt`, `VERIFIED.json`, `CONFIRMATION.json`, `ERROR_CAUSES.json`.

'''
doc=doc.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+current,1)
doc=doc.replace('## 9. 다음 단일 작업\n\n','## 9. 다음 단일 작업\n\n### 2026-11-15 손가락 관측 혼선·자세 전환 분리\n\n'+next_action+'\n\n이번16영상과 추가검색4영상은 이제 검토 이력에 기록됨. 새 학습은17/18로하지않고 기존FIT역할만 사용. 독립 검증 자료 다시 예약. 근거: `research/front_keyphase_v1/ops/'+U.name+'/ERROR_CAUSES.json`, `LEFT_COVERAGE_FINDINGS.json`.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('''# 손가락 공개 검증·추가 학습·비교 완료

현재 목표: 설명의 행동 구간과 사용자의 실제 손모양·움직임을 연결해 근거 있는 교정만 웹에 제공. 사용자 촬영은 학습/튜닝/선택 금지. C·T3-F와 공통100 보존.

최신 결과: `research/front_keyphase_v1/ops/word100_finger_holdout_20261115/RESULT_KO.txt`.
- 후보용으로 쓰지 않았던 공개16영상으로 진단17/확인18 분리. 과거 TCN평가 이력 있음.
- 기존FIT6영상32시점 추가, 작은 손가락Conv+GRU3seed 미세조정. 기존 단독판정 오판 증가, 합의보조도 확인자료 개선 없음 → 새 모델 웹 미연결. 기존 `G6-finger-20261115` 그대로.
- 정면검지10시점4정답6보류0오판; 왼손은 편손/검지조합153항목140정답13보류0오판. 약지/새끼 독립 왼손조합 미검증. 추가4공개영상 RGB검색에서도 적합한 조합을 확인하지 못했으며 전체 데이터에 없다고 단정하지 않음.
- RTMW 새끼관절이 약지를 따르는 오류와 펴짐→굽힘 전환 상태 오류 확인. 손가락 부분상태 오류가 실제 공개 수어가 틀렸다는 뜻은 아님.
- 122공개조건 재생(정상64새경고0), 보호518동일,616산출물검사. 새 모델 배포 통과 아님. 실제웹캠/사용자오류 성능/새 end-to-end 대기시간 시험은 안 함.
- W: 미연결, C: 공개캐시 사용. 학습/추출 프로세스 모두 종료, 새 서버 미실행. 과거 PID를 살아 있다고 가정하지 말 것.

다음 한 작업: '''+next_action+'''

구현·실행·별도 산출물 검산 Codex, 독립 전문가 검토 아님. 최신 상세 STATE 키 `word100_finger_holdout_20261115`; 오래된 인수인계로 덮어쓰지 말 것.
''','utf-8')
print('recorded completion, no web/model changes')
