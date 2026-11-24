"""Root current-result checkpoint; frozen experiment outputs remain untouched."""
from pathlib import Path
import json,sys
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U
read=lambda n:json.loads((U/n).read_text(encoding='utf-8-sig'))
r=read('PROCESS_EVALUATION.json');a=read('PROCESS_AUTO_EVALUATION.json');d=read('PROCESS_DECISION.json');t=read('PROCESS_TRAIN_RESULT.json')
assert d['status']=='NOT_ADOPTED_FIXED_GATES_FAILED'
assert read('PROCESS_AUTO_EVALUATION_VERIFIED.json')['status'].startswith('PASS')
title='과정 판정부 학습·자동 구간 평가 완료 및 미채택 원인 분석'
summary=(
 '사용자재개에따라작업계속/전체목표미완료. 기존4표현 학습형실제데모8809와C/T3F/8804/8808보존. '
 'AB FIT217직접조건/216고유입력,DEV80조건/80입력 준비·검산완료. '
 'seed42/final80epoch 작은과정판정부47980param 한회학습완료(57.33초); 기존SignRep/모델고정. '
 'rawFIT217+DEV80 새평가1회, 저장검산45936checks PASS. DEVknown75중68 raw정답(90.67%), source-query macroBA81.39%; '
 '실제양성30 high정답12/high오답4/보류14, 부분45 high정답42/high오답1/보류2. raw90.67을실사용피드백정확도라하지않음. '
 'saved297raw재사용한연결근거검증 및자동full-native60 새평가1회, 별도184463checks PASS. '
 'DEV실제자격확인3/30, 부분잘못확인1/45, 관측부족잘못확인0/5. auto1/15,제안범위가알려진과정을놓친수0/15. '
 '잘린2523ID12 middle_splice 원55→59연결을 실제과정처럼확인; seam을알려준진단만0오확인이나실제runtime는1유지. '
 '사전21gate중5통과16미달로미채택/웹교체없음. 구C/TCN인식률이나전체수어정오점수가아님. '
 'retention_direction_review가저장값만으로 rawprocess/point/interval/manual-vs-auto 복합병목분석중. '
 '데이터준비반복·threshold/seed/epoch재탐색없음. 사용자/멘토/H14/15/ID16미사용. HTML추가없음. '
 '같은Codex별도검산이며독립수어전문가정답아님. 전체학습형피드백목표진행중.'
)
nxt=(
 'AB PROCESS_DECISION 및 PROCESS_RESULT_KO.txt 확인. 완료학습1/80epoch·raw297·auto60/qualification297·검산 재실행금지. '
 'retention_direction_review의 RESULTS_PROCESS_CAUSE.json 실제상태확인후이어받기(중복진단금지). '
 '핵심질문: raw90.67이지만 qualified3/30·auto1/15가된원인을 rawprocess 확신부족/출발도착 point근거/자동선택interval motion/동시병목으로분리. '
 'auto제안15범위는모두기존양성chain포함이므로 단순DTW검색범위확대만으로해결된다고하지않음. '
 '직접motion학습구간과 setup-run첫프레임부터terminal까지실행구간의유지·문맥불일치는가설; 저장값으로검증후단일실질보완계획. '
 '정상주먹흔들림·유지·교차·짧은변화보존,중간합성cut을언어오답으로승격금지,정확히선택한구간정답미검증표시. '
 '필요보완은새명시계약으로학습입력과실행조건을맞추고비교; 기존결과를임계값하향/재채점으로구제금지. '
 '현재훈련/추출없음. 같은기존C/T3F/웹/개인촬영금지보존. 작은검증을전체goalcomplete로올리지않음.'
)
active=dict(training=None,extraction=None,new_candidate_adopted=False,
 diagnosis_agent='retention_direction_review',diagnosis_artifact='RESULTS_PROCESS_CAUSE.json',
 completed_sessions=[29228],demo_port_preserved=8809)
# Replace only this exact latest entry on a subsequent status sync.
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf8')
for heading,body in [('### 2026-11-02 '+title,summary),('### 2026-11-02 '+title+' 이어받기',nxt)]:
 needle=heading+'\n'
 if needle in text:
  start=text.index(needle)+len(needle);ends=[i for i in [text.find('\n### ',start),text.find('\n## ',start)] if i>=0]
  end=min(ends) if ends else len(text);text=text[:start]+'\n'+body+'\n\n'+text[end:]
context.write_text(text,encoding='utf8')
sync_flow.sync(title,summary,nxt,'PROCESS_CANDIDATE_TRAINED_EVALUATED_NOT_ADOPTED_CAUSE_REVIEW',active)
report='''현재 상황 — 2026-11-02

목표: 여러 사람의 공개 한국수어와 설명을 미리 배운 신경망이 새 영상의 중요한 손모양·신체 관계·움직임·순서를 파악하고, 설명 문구와 영상 시간대를 연결해 피드백한다.

진행 위치
4표현(순종·잘하다·조상·이집트)의 자료·설명 연결, 신경망 학습과 일부 SignRep 가중치 미세조정, 배운 기준 흐름에 대한 DTW, 실제 새 영상 데모 연결은 수행했다. 매번 참고 영상9편을 각각 비교하는 구조가 아니다. 현재는 중요한 과정 누락과 정상적인 차이를 믿을 만하게 구별하는 능력을 보완 중이다. 모든 단어, 전체 정오, 필수 반복 횟수, 실사용자 일반화가 완성된 상태는 아니다. 임의의 전체 완료율은 붙이지 않는다.

‘시작·끝 모습’이 뜻하는 것
촬영본 전체의 맨 처음과 끝이 아니라 주요 동작 하나의 출발과 도착이다. 예를 들어 ‘오른손을 왼팔 위쪽에 놓고 팔을 따라 손목으로 이동’한다면, 위팔에 놓은 모습과 손목에 도착한 모습만이 아니라 그 사이 이동 과정이 보이는지도 확인한다. 단순한 손올림·손내림이나 주요 동작 사이 연결 부분만을 맞추는 작업이 아니다.

이번 실제 실행
공개 FIT36영상의217비교 조건(고유입력216), 별도 등록ID의 DEV12영상80조건을 준비·검산했다. 각 원본에서 만든 변형이므로297명의 독립 시험이 아니다. 작은 과정 판정부 하나를80회 학습했다(학습57.33초). 기존 모델은 고정했다. 관측 부족을 잘못한 행동의 정답으로 바꾸지 않았다.

결과
정해 둔 비교 구간에서는 판정 가능한75조건 중68개를 단순 점수 기준으로 구분했다(90.7%). 하지만 확신 있게 맞춘 정상 과정은30개 중12개였고, 정상4개·부분1개를 확신 있게 잘못 구분했다.
출발 모습·이동·도착의 연결 근거까지 확인하면 정상30개 중3개만 충분히 확인했다. 일부만 남은45조건 중1개를 연결된 과정으로 잘못 확인했다. 영상 근거가 가려진5조건은 모두 보류했다.
실제 데모처럼 전체 영상에서 자동으로 찾은 범위에서는 정상15과정 중1개만 확인했다. 15개 범위 모두에 검토된 정상 과정은 들어 있었다. 따라서 단순히 동작 구간을 못 찾은 것만으로 설명할 수 없다.
기존 방식에는 이 ‘연결 과정’의 동일한 점수가 없으므로, 이전 정확도에서 몇% 향상했다고 주장하지 않는다. 이번 결과는 C·T3-F 단어 인식률도 전체 수어 정답률도 아니다.

판단과 다음 작업
새 후보는 사전에 정한 기준21개 중16개에 미달해 현재 웹에 적용하지 않았다. 기존 기능은 유지했다. 단순 점수90.7%만 보고 성공이라고 말할 수 없다.
저장 결과로 전체 과정 점수, 출발·도착 모습 판단, 이동 구간 판단이 각각 어디서 막히는지 확인 중이다. 특히 모델이 학습한 이동 구간과 자동으로 검사한 구간의 길이·준비 자세 포함 여부가 다른지는 확인할 가설이다. 원인이 확인된 부분을 공통 방식으로 보완하며, 결과에 맞춰 점수 기준을 낮추거나 같은 학습을 반복하지 않는다.

속도와 보존
현재 실제 데모의 공개 영상1편 기록은 분석8.77초/버튼부터표시9.55초다. 평균이나 이번 후보의 개선 속도가 아니다. 이번에는 새 실사용 전체 처리시간 측정을 하지 않았다.
기존 C·T3-F 및 기존 웹 보존. 사용자·멘토 촬영은 학습·튜닝·선택에 사용하지 않았다. 추가 HTML은 만들지 않았다. 검토·별도 산술검산은 Codex이며 독립 한국수어 전문가 판정은 아니다. 전체 목표는 아직 진행 중이다.
'''
(U/'PROCESS_RESULT_KO.txt').write_text(report,encoding='utf8')
(U/'PROGRESS_KO.txt').write_text(report,encoding='utf8')
print('Verified result and latest project/handoff/state synchronized; no model or inference run')
