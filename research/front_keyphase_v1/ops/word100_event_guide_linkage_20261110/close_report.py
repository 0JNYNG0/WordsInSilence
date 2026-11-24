"""Document verified research outcomes without modifying old or deployed models."""
from link_core import U,O,ROOT,read,write,sha
from datetime import datetime,timezone,timedelta
import json
r=read(U/'FINAL_RESULT.json');v=read(U/'FINAL_VERIFICATION.json')
assert v['passed'] and v['result_sha256']==sha(U/'FINAL_RESULT.json')
audit={
 'scope':'Post-result source RGB and saved movement-path audit; not new training labels or independent validation.',
 'source_inputs':str(U/'WARNING_RGB_INPUTS.json'),
 'cases':[
  dict(clip_id='NIA_SL_WORD0739_REAL13_F',kind='omit_final_event',finding='Original source71~82 is the early hand-near-face/downward transition. New context falsely called it later upright forearm, then treated lowering as reversed raising/order. Do not count this warning as omission detection.',final='new warning vetoed; prior uncertain restored'),
  dict(clip_id='NIA_SL_WORD0739_REAL12_F',kind='reversed_core_auto_order',finding='Raw reversed motion has a falling forearm, but original and new semantic context estimates disagree. Direction alone does not establish the intended guide-action identity.',final='new correction vetoed; evidence retained'),
  dict(clip_id='NIA_SL_WORD0739_REAL13_F',kind='reversed_core_auto_order',finding='Reversed source67..112 maps analysis89..103 to source90..76: chest-to-face upward movement, opposite saved guide instruction to lower toward chest. Frozen movement head and relative wrist trajectory agree; context conflict veto does not reject this evidence.',final='one added opposite-direction candidate warning retained; not a new proof of whole-sign error')],
 'first_candidate_result':str(U/'RESULT.json'),'first_candidate_result_sha256':sha(U/'RESULT.json'),
 'final_result':str(U/'FINAL_RESULT.json'),'final_result_sha256':sha(U/'FINAL_RESULT.json')}
write(U/'CAUSAL_AUDIT.json',audit)
report='''찾은 움직임을 설명의 행동과 연결한 결과

결론: 실제 분석 경로에서 반대 방향을 추가로 찾은 사례 1개가 생겼습니다.
정상 동작을 더 많이 지적하지 않았고, 기존 손모양·방향 판정도 유지했습니다.
다만 순서와 누락을 모두 정확하게 설명하는 단계까지 완성된 것은 아닙니다. 연구 후보만 보존했고 웹 기본 모델은 교체하지 않았습니다.

1. 무엇이 달라졌나
이전: 기존 비교 구간 안에서만 “어떤 행동인가”를 찾았습니다. 구간이 비거나 두 행동이 겹치면, 새로 찾은 움직임을 활용하지 못했습니다.
현재: 영상 전체에서 신체 관계를 먼저 찾고 → 찾은 이동 구간을 설명의 행동에 연결하고 → 방향·순서를 따로 검사합니다.
예를 들면 “가슴 앞에 있는 팔”을 찾은 뒤, 그 구간의 손이 아래로 갔는지 위로 갔는지를 확인합니다.
손모양이 틀려도 그 움직임 자체가 비교에서 빠지지 않도록, 행동 연결 입력과 손모양 판정을 분리했습니다.
실제로 손가락 좌표만 바꾼 검사에서 행동 연결에 들어가는 신체·손목 특징이 정확히 동일함을 확인했습니다. 실제 검출기의 모든 오차가 서로 독립이라는 뜻은 아닙니다.

설명은 기존 가이드의 행동·방향·필수 여부를 구조화해 연결했습니다. 새 모델은 자유로운 한국어를 읽는 LLM이 아닙니다.
“반대팔을 따라 쓸기”와 “그 동작으로 손목에 도착한 모습”처럼 같은 행동의 일부인 것은 별개의 필수 단계로 강제하지 않습니다.
판정 함수에 단어 번호별 예외는 넣지 않았습니다. 단어마다 다른 실제 설명을 자료로 연결하는 것과, 특정 오답을 숨기는 예외 코드는 구분했습니다.

2. 무엇을 학습했나
공통 학습은 공개 100단어·899영상(기록 ID01~09)을 유지했습니다. 모든 학습 회차에 899영상을 사용했습니다.
설명 연결을 위한 세부 정답은 기존 공개 4표현·36영상에서 직접 확인한 569시점을 재사용했습니다.
배경, 얼굴 앞의 손, 가슴 앞의 팔, 세워진 전완, 반대팔 부근, 반대 손목 부근, 오른손이 높은 모습, 왼손이 높은 모습, 모인 양손, 벌어진 양손으로 나눴습니다.
이것은 100단어 전체에 세밀한 행동 정답이 완비됐다는 뜻은 아닙니다. 공통 학습 범위와 현재 피드백 검증 범위는 다릅니다.
기존 시간 합성곱+양방향 GRU의 복사본과 새 행동 연결 출력을 3개 고정 난수값으로 각각 12회 추가 학습했습니다. 대표값42는 결과 확인 전에 정했고 가장 좋은 결과만 고르지 않았습니다.
앞뒤를 뒤집은 자료는 “같은 모습도 다른 순서에 나타날 수 있다”는 대조 학습에만 썼고, 올바른 단어 수행 정답으로 가르치지 않았습니다.
기존 손모양·방향 가중치, 경계 검출 가중치, C·T3-F와 현재 TCN은 바꾸지 않았습니다.

3. 이전 → 현재
공개 4단어×기록 ID11~13의 12개 원본과, 그 자료를 변형한 조건들을 비교했습니다.
아래 교차 3개는 정상 12개 안에 포함됩니다. 변형 조건은 각각 다른 새 사람의 영상이 아닙니다.

시험                           이전                         현재
정상 원본 12개                 잘못된 경고 0                0 유지(비슷함9·보류3)
느림/빠름 24조건               잘못된 경고 0                0 유지(비슷함17·보류7)
정상 양손 교차 3개             3개 비슷함                   그대로
기존 손모양/방향 오류 9조건     9개 경고                     그대로
주요 구간을 역순으로 만든 9개   경고3·비슷함1·보류5          경고4·비슷함1·보류4
마지막 동작 삭제 6개           6개 보류, 정상 통과0         그대로
마지막 동작 관절 가림 6개      6개 보류, 정상 통과0         그대로
전체 손 관절 미관측 12개       12개 보류, 정상 통과0        그대로

새로 남긴 경고는 “순종” 역순 시험 하나에서, 가슴 쪽으로 내려와야 하는 흐름이 위로 올라가는 방향이라는 근거입니다.
기존에는 해당 부분을 보류했지만, 이제 신체 관계와 실제 이동을 연결해 반대 방향 후보를 표시했습니다.
같은 변화가 3개 고정 학습에서 모두 나왔습니다. 그렇다고 새 영상 3개에서 각각 성공했다는 뜻은 아닙니다.
순수한 순서 오류 정답률이 3/9→4/9로 개선됐다고 표현하면 부정확합니다. 역순 변형 시험에서 방향 차이를 하나 더 잡은 결과입니다.

보조 진단: 원본 4영상의 직접 확인한 67시점에서 신체 관계 일치는 기존 출력 환산62/67, 새 모델64/67(다른 두 학습64/67·62/67)이었습니다.
67시점은 기록 ID11의 4영상에서 분명하게 볼 수 있는 장면을 고른 작은 표본입니다. 전체 수어 피드백 정확도가 95%라는 뜻이 아닙니다.
기존은 목표 단어로 후보가 제한된 출력이고 새 모델은 전체 관계 종류에서 고르는 출력이라, 이 수치는 보조 진단입니다.
검토한 이동 7구간 모두 해당 설명의 움직임 후보와 연결됐습니다. 부분 구간도 포함되므로, 정확한 시작·끝이나 완전 수행을 7/7 확인한 것은 아닙니다.

4. 도중에 발견해서 바로 보완한 문제
첫 후보는 새 경고를 3개 냈습니다. 하지만 원본을 확인하니 그중 하나는 앞부분의 얼굴 앞 자세를 뒤의 팔 세우기 동작으로 잘못 연결했습니다.
“마지막 동작을 삭제한 시험”에서 경고했다고 해서, 누락을 제대로 배웠다고 세면 안 되는 사례였습니다.
이를 바로잡기 위해 기존 관측과 새 행동 연결이 충돌하면 새로운 교정을 보류하는 공통 검사를 추가했습니다.
다른 역순 사례 하나도 행동 연결 근거가 충돌해 보류했습니다. 원래 관측값과 후보 구간은 버리지 않고 모두 남겼습니다.
따라서 최종 보고에는 근거가 남은 추가 경고 1개만 포함했습니다. 첫 결과도 RESULT.json에 보존했습니다.
이 보완은 개발 자료의 오류를 본 뒤 추가한 것이므로, 최종 수치를 새 독립 시험 성능으로 주장하지 않습니다.

5. DTW·속도·기존 기능
DTW 정렬은 유지했습니다. 기존 저장된 시간 대응, 프레임 목록, TCN 결과와 손모양·방향 근거가 바뀌지 않았는지 확인했습니다.
이번에 새로 계산한 것은 행동 관계 판정부와 구간 연결, 일부 연결 구간의 방향 확인입니다. 기존 TCN/DTW 전체 추론을 반복하지 않았습니다.
추가 계산 중앙값은 약0.037초, 최대 약0.183초였습니다(충돌 보류 후처리 전). 관절 검출·기존 특징 준비·TCN·DTW·화면 표시는 제외했습니다. 웹 전체 응답시간이 이만큼이라는 뜻은 아닙니다.
사용자·멘토 촬영은 접근하거나 학습·튜닝·모델 선택에 사용하지 않았습니다. 카메라, 다운로드, 외부 전송, 운영 교체도 하지 않았습니다.

6. 아직 남은 문제
- 역순 시험 하나는 여전히 잘못된 “비슷함”으로 남고, 4개는 보류입니다.
- 마지막 동작 삭제와 가림을 정상으로 넘기지는 않지만, “정말 생략했다”와 “관측하지 못했다”를 자신 있게 구분해 설명하지는 못합니다.
- 얼굴 앞 자세와 나중의 팔 세우기처럼 비슷한 모습이 다른 행동에 등장할 때, 동작의 역할을 착각할 수 있습니다. 충돌 검사는 이런 경우를 숨기지 않고 보류하는 보완입니다.
- 공통 코드로 구현했지만, 새로 늘어난 최종 경고는 현재 한 표현에서만 확인됐습니다. 모든 단어에서 개선됐다고 말할 수 없습니다.
- 주석 검토와 검산은 같은 Codex가 수행했습니다. 한국수어 전문가의 독립 검수가 아닙니다. 시험도 반복 사용한 과거 개발 자료입니다.

7. 다음에 집중할 일
실제 움직임의 앞·중간·뒤 모습을 하나의 행동으로 묶는 정답을 보강해, 비슷한 한 자세만으로 행동을 잘못 배정하는 문제를 줄입니다.
“앞 동작에서 같은 모습이 나옴”, “뒤 동작을 실제 수행함”, “도중에 중단함”, “잠깐 관절이 가려짐”을 서로 다른 학습 근거로 연결해야 합니다.
정상 동작의 방향과 순서를 먼저 관측한 뒤 설명과 비교하며, 정답 순서에 맞춰 후보를 억지로 선택하지 않습니다.
세부 주석을 여러 표현과 기록 ID에 보강하고, 기존 학습·선택에 쓰지 않은 공개 평가 자료의 가능성을 다시 감사합니다.
최종 경고와 누락 설명의 이득이 넓게 확인되기 전에는 웹 기본 모델을 교체하지 않습니다.

완료 범위: 정답 연결 감사, 추가 학습, 234자동 비교, 오류 근거 확인·공통 충돌 보완, 저장 결과 별도 검산, 프로젝트 이력 기록.
최종 근거: FINAL_RESULT.json, FINAL_VERIFICATION.json, CAUSAL_AUDIT.json,
LINK_LABELS.json, GUIDE_CONTRACTS.json, HANDSHAPE_IDENTITY_CHECK.json.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
next_text='비슷한 한 자세를 다른 행동으로 배정하는 문제를 줄이기 위해 공개FIT의 행동별 앞·중간·뒤 관계와 실제 진행/중단/가림 대조 정답을 보강한다. 현재 관계판정부·경계·기존모양방향을 고정기준으로 두고 사건 전체의 역할 연결을 검증한다. 올바른 순서로 강제배정/단어예외/DEV문턱튜닝 금지. 새평가 출처·기존노출 감사 후 비교,100단어공통/C/T3F/웹/개인자료보존. 완료3학습·234비교·234저장후처리 재실행금지.'
write(U/'NEXT_PLAN.json',dict(status='NEXT_UNIT_NOT_STARTED',next=next_text,remaining=['same body pose in different action roles','one false similar and four uncertain reversed fixtures','omission vs observation lack','broader distinct evaluation evidence'],acceptance='Preserve normal/speed/crossing and wrong-shape evidence; improve role/direction/order/omission feedback beyond one expression; no forced assignment or false completion.'))
summary='100단어899공통+공개4표현36FIT의569관측시점으로 신체관계9종+배경 연결 Conv+GRU3seed×12회 추가학습. 기존DTW창독립 연결·선택도착과필수행동분리·손모양오류가사건을삭제하지않는기능검사완료. 역사DEV78조건×3seed: 첫새경고3중앞동작오배정1발견,기존관측충돌보류공통후처리로2경고보류. 최종역순9의경고3→4(방향근거새1),비슷함1·보류4; 정상12/속도24/교차3/기존오류9/삭제가림24유지.67관측시점기존환산62→64/67(타seed64·62),새독립평가아님. 1797초기/2134최종저장검산PASS(같은Codex). 실제누락·순서완전해결아님,새경고한표현만. C/T3F/현재TCN/웹미교체·개인자료0.'
title='### 2026-11-10 관측 사건·설명 연결 학습과 제한 방향 개선 검증 완료'
cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';ct=cp.read_text('utf-8-sig');oldtitle='### 2026-11-10 관측 사건과 설명 행동 연결 진행';assert ct.count(oldtitle)==2
for sec,txt in [('## 8.',summary),('## 9.',next_text)]:
 s=ct.index(oldtitle,ct.index(sec));e=ct.index('\n### ',s+len(oldtitle));ct=ct[:s]+title+'\n\n'+txt+'\n근거 '+str(U/'RESULT_KO.txt')+'\n\n'+ct[e:]
cp.write_text(ct,'utf-8');stamp=datetime.now(timezone(timedelta(hours=9))).isoformat();st=read(O/'STATE.json');st.update(current_stage=U.name,next_single_action=next_text,last_updated=stamp);st[U.name]=dict(status=r['status'],summary=summary,next=next_text,report=str(U/'RESULT_KO.txt'),result=str(U/'FINAL_RESULT.json'),verification=str(U/'FINAL_VERIFICATION.json'),execution=str(U/'CURRENT_EXECUTION.json'),private_access=False,default_change=False);write(O/'STATE.json',st)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+next_text+'\n\n현재단위 '+str(U)+'\n학습·평가·저장후처리·검산 종료. 현재 단위 실행 중 명령 없음. 웹/카메라 미실행.\n첫 결과 RESULT.json 보존, 최종 판단은 FINAL_RESULT.json/FINAL_VERIFICATION.json/RESULT_KO.txt.\n','utf-8')
with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:
 f.write('\n\n'+title+'\n'+summary+'\n변경 이유: 빈 DTW 비교창과 겹친 단계후보 때문에 새로찾은 사건을 설명에 연결못함. 관측관계 독립학습과 자료형 가이드 연결로 분리.\n실행: 100공통 유지·직접관측569 재사용·4개DEV67시점새검토·3학습·234비교. 역순 대조는 정상 수행정답으로 사용하지 않음.\n중간 오류: 첫 추가경고3개를 성과로 확정하지 않고 원본RGB/시간경로 점검,앞부분얼굴자세를뒤팔세움으로 오배정 확인. 기존관측과 새연결의 충돌 보류를 공통후처리로 추가,재학습/문턱탐색0.\n검증: 같은Codex별도산술,독립사람검수아님. 최종새경고1이며순서/누락의전체해결아님. 단어별예외0·100단어세부성능주장없음.\n다음: '+next_text+'\n근거 '+str(U/'RESULT_KO.txt')+'\n')
write(U/'CURRENT_EXECUTION.json',dict(status=r['status'],pid=None,completed_at=stamp,training_running=False,evaluation_running=False,default_changed=False,next_unit_started=False))
with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=stamp,kind='COMPLETE_DOCUMENTED',result_sha256=sha(U/'FINAL_RESULT.json')),ensure_ascii=False)+'\n')
print(json.dumps(dict(report=str(U/'RESULT_KO.txt'),status=r['status'],verification=v['count']),ensure_ascii=False))
