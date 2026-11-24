from common_split import *
from datetime import datetime
from zoneinfo import ZoneInfo
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();v=read(U/'VERIFIED.json');audit=read(U/'SOURCE_TRUTH_AUDIT.json');cfg=read(U/'CANDIDATE_FROZEN.json')
for p,h in cfg['pins'].items():assert sha(p)==h
diag=read(U/'diagnostic_SOURCE_AUDITED_RESULTS.json')['metrics'];historic=read(U/'historic_RESULTS.json')['metrics'];reserved=read(U/'confirmation_reserved_RESULTS.json')['metrics'];replay=read(U/'REPLAY_RESULTS.json')['summary']
decision=dict(at=now,status='COMPLETE_RESEARCH_NOT_ADMITTED',web_changed=False,admitted_web_model='G6-finger-20261115',metrics=dict(historic=historic,diagnostic_source_audited=diag,confirmation_reserved=reserved),replay=replay,release_reasons=['quality/combined reserved set has no confidently confirmed corrupt-chain positive; sensitivity beyond the diagnostic actor is unproven','shape_only increases historic confident errors1to13','compatible_shape improves some abstentions but does not meet the frozen requirement to reduce diagnostic wrong counts','combined still abstains on11previously correct historic items; positive evidence insufficient to justify adopting the quality branch'],finger_model_only=True,C_T3F_changed=False,common100_changed=False,user_footage_used=False,independent_expert_review=False)
write(U/'DECISION.json',decision)
nextplan=dict(goal='Preserve brief genuine finger states while distinguishing them from corrupted tracks; improve both labels and per-frame decision before any wider web activation',evidence=['WORD2551 REAL12 f35 source index truly extended, track consistent, current temporal model says flexed','Holding center input fixed and replacing context changes fine-tuned shape probabilities0.003..0.011 to0.969..0.986; mechanistic clue only','WORD2748 REAL10 f61 label corrected1to0 after side-view audit; label quality is a separate failure channel','35training corrupt-chain cells:34little1index; no middle/ring positives; reserved corrupt-chain support0'],steps=[dict(id='A',task='Audit comparable FIT-only short extension/bending sequences using existing front+side public takes; identify center posture and whether neighbors change, and review start/end timing. No assigning labels by word names or copied DTW intervals.',gate='Each supervised center has source evidence; actual transitions and unobservable chains marked separately; no private/DEV/reserved optimization'),dict(id='B',task='Add per-frame center-state supervision or a center-state branch while retaining temporal Conv+GRU. Compare genuine brief extension with transient tracking errors. Do not blindly smooth or fill using the expected sign.',gate='Fixed model alternatives and thresholds before new diagnostic outputs; old common100/C/T3F remain frozen'),dict(id='C',task='Prepare a fixed additional public clip set not used for current candidate selection, with confirmed corrupt-chain positives and parallel-open negatives on both hands. Keep all current10/16 results as exposed diagnostics, not new final validation.',gate='Positive support and label quality checked before new models; report wrong, correct and abstained separately, retained true transitions and false corrections'),dict(id='D',task='Evaluate combined decisions and replay normal/speed/crossing/error conditions through existing web contract. Only release a candidate passing predeclared normal-preservation and quality-evidence gates.',gate='No new normal warning, preserve20oldfinger details/coarse/direction/order, source integrity; actual camera check remains user-deferred')],scope='Common observation features, no word-specific thresholds. Public source multi-view is for annotation review, not extra runtime camera requirement.')
write(U/'NEXT_PLAN.json',nextplan)
write(U/'FAILURES_AND_REPAIRS.json',dict(items=[dict(issue='Training distillation target was created with torch.inference_mode',repair='Changed teacher computation to no_grad before first optimizer update; no completed model existed; restarted fixed run'),dict(issue='One read-only cause-inspection shell command used a relative Python path from a subdirectory',repair='Used absolute interpreter path; no data changed'),dict(issue='Verifier typo else16 caused parse error',repair='Corrected spacing; verifier executed961checks'),dict(issue='Initial frontal-only diagnostic label WORD2748 REAL10 f61 index1 was wrong',repair='Side-view source audit establishes actual flexion0; original labels/results retained, separate corrected counts applied identically to baseline/candidates; models not retrained or reselected'),dict(issue='Reserved16 had no confirmed corrupt-chain positive',repair='Do not invent quality truth. Research training continued with explicit pretraining clarification; quality deployment remains blocked')]))
report='''관절 오류와 손모양 판단 오류를 나눈 보완 — 2026-11-15

핵심 결과
이번 정답 보강·추가 학습·기존/별도 공개 비교·전체 피드백 재검사를 마쳤다. 일부 잘못된 단정을 줄였지만, 새 관절 품질 판정부를 웹에 넣을 근거는 아직 부족하다. 현재 웹 G6-finger-20261115를 그대로 유지했다. C·T3-F와100단어 공통 모델, 기존 DTW/행동 구간 판정도 변경하지 않았다.

무엇을 구분했나
1. 손가락 점이 옆 손가락을 따라간 경우: 손모양을 억지로 학습시키지 않고, 그 관측을 믿을 수 있는지 따로 학습했다. 새로 확인한 잘못된/불확실한 관절 항목은 손모양 학습에서 제외했다.
2. 점은 실제 손가락을 따라가는데 모델이 잘못 읽는 경우: 기존 공유 손가락 시간 합성곱+GRU의 가중치를 이어서 학습했다. 손목과 손가락 사이 상대 관계를 쓰며, 단어명·기대 손모양·사람 번호를 입력하지 않는다.
3. 정답 주석 자체가 틀린 경우도 따로 찾았다. WORD2748 REAL10 f61 검지는 측면에서 실제로 굽혀진 상태였다. 처음 정면만 보고 붙인 '펴짐'1을 '굽힘'0으로 정정했다. 기존 동결 주석·수치는 덮어쓰지 않고 SOURCE_TRUTH_AUDIT와 별도 재집계를 남겼다. 학습/후보 선택은 이 정정으로 다시 하지 않았다.

자료와 학습
HDD연결 후 새 확인용 공개 정면 원본12개(6표현×수행자10/16)를 확보하고 모든 프레임의 기존 RTMW관절을 추출했다. 학습은 수행자01~09, 개발 비교10, 별도 확인16으로 분리했다.10/16은 이번 손가락 후보에는 새 비교지만 과거 TCN평가에 사용된 적 있어 프로젝트 전체에서 처음 보는 사람이라고 말할 수 없다.
이번에 연구용 손모양·관절 품질 정답195손 시점을 원본 RGB/앞뒤 장면과 대조했다: FIT96(15영상),개발48(6영상),별도51(6영상). 서로 다른 손가락4개를 각각 평가한다. 모델 예측을 보고 정답을 붙이지 않았다. 주석은 Codex의 시각 검토이며 수어 전문가의 독립 검수가 아니다.
기존 FIT444손 시점을 유지하고 중복2개를 정리해 총538손 시점/104공개영상/21표현/9명으로 손가락 판정부를 추가 학습했다. 이는 100단어 공통 모델 전체를 다시 학습한 것이 아니다.
손모양 모델3개는 기존 가중치에서250단계씩 미세조정했다. 관절 품질 모델3개는 별도350단계씩 학습했다. 고정 초기값42/43/44를 모두 쓰며 가장 잘 나온 한 회차만 고르지 않았다. 품질 학습의 실제 잘못된 관절35항목 중34는 새끼손가락,1은 검지라서 아직 균형이 부족하다.

비교 결과 (한 시점의 손가락 하나=1항목; 영상/단어 정답률이 아님)
                         기존:맞음/오판/보류    새 결합 후보:맞음/오판/보류
기존 개발 자료967항목       910 / 1 / 56          903 / 1 / 63
새 개발 자료187항목         146 / 4 / 37          145 / 1 / 41
별도 확인 자료192항목       173 / 2 / 17          178 / 0 / 14
새 개발자료는 측면 검토 후 정정한 주석으로 양쪽을 같은 조건에서 다시 센 값이다. 정정 전은145/5/37→144/2/41이었다. 이1건 감소는 모델 개선이 아닌 주석 정정이다.
새 개발 오판4개 중3개와 별도 확인 오판2개는 '올바른 손모양으로 복구'한 것이 아니라 '불확실하므로 단정하지 않음'으로 바뀌었다. 별도 확인에서 더 맞힌6개는 이전에 보류하던 항목이다. 기존 개발에서는 새로4개를 맞혔지만, 원래 맞던11개를 보류했다. 이 손실도 숨기지 않았다.
직전 품질 후보는 같은 기존 개발 자료에서 원래 맞던40개를 보류했으므로, 이번11개로 정상 판정 손실은 줄었다. 그래도 잘못된 관절을 정확히 모두 복구한 것은 아니다.
새 손모양 모델을 혼자 쓰면 기존 개발 오판1→13으로 늘었다. 기존 손 전체 판정부와 함께 확인하면 오판1유지/맞음910→914였지만, 새 개발의 오판4개는 그대로였다. 이 때문에 단독 교체나 기준 완화는 하지 않았다.

정상·기존 기능 보존
공개122시험 조건(정상64포함)을 다시 실행했다. 정상 동작에 새 손가락 경고0, 기존 구체적 손가락 안내20개 유지. 원래 동작 구간·순서·방향·거친 손모양 근거도 그대로였다. 속도 변형·누락·역순 등을 포함한 공개 대조 시험이며 실제 학습자의 다양한 오류를 모두 검증한 결과는 아니다.
보호 대상518파일 해시, 학습/비교 수행자 분리, 손가락 정답/관측 마스크, 모델 가중치 실제 변경, 결과 재계산, 좌표 이동·크기·화면 내 회전에 대한 특징 불변성 등961자체 검사를 통과했다. 독립 전문가 검증으로 표시하지 않았다.

남은 원인과 적용 판단
오래 남은 WORD2551 REAL12 f35는 정면·측면·앞뒤 장면과 관절선을 확인해도 검지 폄 정답을 유지한다. 현재 모델은 이를 굽힘으로 읽는다. 중앙 관절을 그대로 두고 앞뒤를 중앙 프레임 복제로 바꾸는 원인 확인 시험에서는 새 손가락 모델3개의 폄 확률이0.003~0.011에서0.969~0.986으로 바뀌었다. 앞뒤 변화의 영향으로 짧게 펴진 순간을 놓칠 가능성을 보여준다. 실제 영상 개선이나 새로운 정확도 증명은 아니며 이 복제 입력은 학습/후보 선택에 쓰지 않았다.
별도 확인 영상에서는 '좌표가 확실히 잘못됨'이라고 확정할 수 있는 품질 정답이0개였다. 손모양 정답은 있어도 가려진 손끝의 실제 좌표를 정면만으로 확인할 수 없는 장면을 억지로 오류 정답으로 만들지 않았다. 따라서 관절 오류 감지 능력이 다른 수행자에게도 유지된다는 통과 근거는 부족하다.
새 후보는 웹 미적용. 단순히 실패하고 원점으로 돌아간 결과는 아니다: 정상 판정 손실을40→11로 줄인 후보, 실제 손모양 오류와 검출 오류를 나눈 정답, 잘못 붙인 주석 정정, 짧은 실제 상태를 문맥이 덮는 원인 단서가 남았다. 다만 웹 성능이 좋아졌다고 말할 단계는 아니다.

다음 한 작업
짧게 실제로 펴진 순간과 검출점만 튄 순간을 구분할 정답을 FIT공개영상에서 측면 자료와 함께 보강한다. 그 후 중앙 시점의 자세와 앞뒤 흐름을 각각 배우고 함께 판단하는 보완을 비교한다. 현재10/16결과를 새 최종 검증처럼 재사용하지 않는다. 별도 고정 공개 묶음에는 양손의 명확한 관절 오류와 정상 나란히 편 손가락을 모두 포함해, 또 모두 보류하는 방식으로 좋아 보이지 않는지 확인한다. 구체적 A~D는 NEXT_PLAN.json.

개인 촬영은 읽거나 학습·튜닝·후보 선택에 쓰지 않았다. 새 카메라/다운로드/서버 변경 없음. 실제 사용자 촬영 확인은 기존 보류 요청 유지. 모델·추론 코드·연구자료는 이번 결과 폴더에 격리. 수행 담당과 자체 검증은 Codex이며 독립 검증자 없음.
근거: LABEL_VERIFIED.json, TRAIN_INPUTS.json, TRAIN_CONFIG.json, TRAINED.json, diagnostic_SOURCE_AUDITED_RESULTS.json, historic_RESULTS.json, confirmation_reserved_RESULTS.json, SOURCE_TRUTH_AUDIT.json, CLEAN_CHAIN_DIAGNOSTIC.json, REPLAY_RESULTS.json, VERIFIED.json, DECISION.json.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
summary='관절 오류/손모양 판정 오류 분리·정답195손시점·추가학습6회·공개122재검사 완료. 새 개발187항목 오판4→1,예약192항목2→0이나 해당 오판은 보류로 전환. 기존967에서 정상정답11개 보류(직전40보다 감소), 품질예약 오류양성0로 새후보 웹미적용. 검지 정답1건 측면검토 정정(양쪽 동일 재집계). 실제 관절이 맞는 짧은 검지폄1건 오판은 남음.'
nexttext='짧은 실제 손가락 자세와 관절 추적 오류의 FIT정답을 측면까지 확인해 보강하고, 중앙시점 자세/시간문맥을 나누는 후보와 미사용 공개 오류양성 포함 묶음을 사전 고정해 비교. 현재10/16은 이미 노출된 진단자료로만 사용. 기존웹 G6-finger-20261115·C/T3F·100단어 공통·DTW 보존, 사용자촬영 미사용.'
s=read(O/'STATE.json');s['current_followup']=U.name;s['next_single_action']=nexttext;s[U.name]=dict(status=decision['status'],at=now,result=str(U/'RESULT_KO.txt'),decision=str(U/'DECISION.json'),verified=str(U/'VERIFIED.json'),next_plan=str(U/'NEXT_PLAN.json'),summary=summary,web_changed=False);write(O/'STATE.json',s)
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for h,content in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nexttext)]:doc=doc.replace(h+'\n\n',h+'\n\n### 2026-11-15 관절 오류·분류 오류 분리 보완 완료 / 웹 유지\n\n'+content+' 근거 `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `NEXT_PLAN.json`. 구현·원본검토·자체 검증 Codex, 독립 전문가 검수 없음.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# 관절 오류·손모양 분류 오류 분리 보완 완료\n\n'+summary+'\n\n- 결과: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`\n- 다음: '+nexttext+'\n- 범위: 공통100 기반 동결, 손가락538FIT시점/104영상/21표현/9명. 품질 모델 양성35중34새끼/1검지. 3seed미세조정+3seed품질신규.\n- 현웹 `G6-finger-20261115` 유지, 새모델 연결 안함. 실제 웹캠 확인은 사용자가 미룬 상태.\n- side영상 근거로 REAL10 WORD2748 f61 검지1→0정정. 초기동결파일 그대로, `diagnostic_SOURCE_AUDITED_RESULTS.json`에 재집계. 미래 학습은 정정 근거를 참조하되 이 DEV시점은 학습 금지.\n- REAL12 WORD2551 f35 진짜폄/관절일치/모델굽힘 유지. 중앙반복 원인probe는 성능·학습 근거가 아닌 시간문맥 영향 확인용.\n- 수행/자체검증 Codex, 독립 검증자 없음. 진행중 학습 프로세스 없음. HDD공개원본 읽기만, 필요한 영상은 이번 C:결과폴더에 복사.\n','utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status=decision['status'],at=now,pid=None,processes_running=False,next=str(U/'NEXT_PLAN.json')))
for p in [U/'DECISION.json',U/'NEXT_PLAN.json',U/'CURRENT_EXECUTION.json',O/'STATE.json']:read(p)
for p,h in read(U/'PROTECTED.json').items():assert sha(p)==h
print(summary)
