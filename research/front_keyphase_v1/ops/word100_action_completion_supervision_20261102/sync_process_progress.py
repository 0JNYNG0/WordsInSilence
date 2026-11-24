"""Root-only current progress, preserving historical checkpoints."""
from pathlib import Path
import json,sys
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U
read=lambda n:json.loads((U/n).read_text(encoding='utf-8-sig'))
v=read('PROCESS_VIEW_TARGETS_INDEPENDENT.json');assert v['status'].startswith('PASS') and v['views']==217
p=read('PROCESS_INPUT_PROGRESS.json');complete=p['status']=='COMPLETE'
dev=[w for w in ['0739','1133','1567','2523'] if (U/f'REVIEW_DEV_{w}.json').exists()]
dev_views=(U/'DEV_PARTIAL_VIEWS_MANIFEST.json').exists()
partial_done=[w for w in ['0739','1133','1567','2523'] if (U/f'REVIEW_DEV_PARTIAL_{w}.json').exists()]
dev_inputs=(U/'DEV_PROCESS_INPUTS_VERIFIED.json').exists()
train_progress=read('PROCESS_TRAIN_PROGRESS.json') if (U/'PROCESS_TRAIN_PROGRESS.json').exists() else None
train_state=('학습 미시작' if train_progress is None else '학습 '+str(train_progress['status'])+' '+str(train_progress.get('epoch',0))+'/80')
title='과정 관측 정답 검산 완료·실제 편집 입력 및 개발 검토'
summary=(
 '전체 목표 active. 4표현 학습형 데모8809(9등록FIT/3DEV, learned prototype1개DTW)와 C/T3F/8804/8808 보존. '
 'AB FIT36영상45과정의 원본·부분·속도·여백·관측누락217view 직접검토 연결 완료: 과정지지79/부분근거128/미확인10. '
 '부분128은 이 화면에 연결된 표적과정이 관측되지 않았다는 뜻이며 수행자 수어오류 정답이 아니다. '
 '별도 저장값 재구성검산 PASS217/순간관계2360/온전한interval472; 미확인·ID08변이·대체반복보존. '
 '1567 반복의 다른 온전한구간이 남은4편집positive와1blackoutpositive를 보존해 editkind=오답을 금지했다. '
 '실제 원픽셀216고유영상 준비·모든픽셀/PTS exact검증 완료. '
 f"새 실제영상 MP joint_v3+SignRep 입력은 {p['done']}/216 {'완료' if complete else '실행중'}(GPU.0, session24461, PID27692; 실제상태 우선). "
 '83short입력은16slot내부끝화면반복뿐 원시간/원프레임은추가하지않으며 성능검증아님. '
 f"DEV12공개영상15과정 native직접검토완료 {','.join(dev)}. "+
 ('DEV80직접검토·별도검산 완료: 과정지지30/부분45/미확인5, sparse691/interval179, 원시간2011연결0us. ' if len(partial_done)==4 else 'DEV비교조건 직접검토중. ')+
 ('DEV실제입력80/80 추출·저장검산완료(GPU.0),2011rows/short22/blank49. ' if dev_inputs else '')+
 'process_head47980param 구현·실제FIT입력 preflightPASS, '+train_state+'. '+
 '새정확도향상은평가·별도검산전미확인. 기준latency공개1편분석8.774s·버튼9.551s, 평균아님. '
 '준비숫자를성능점수로사용하지않음. 전체완료/필수반복/교육정오/새사용자일반화미완료. '
 '검토와별도검산은Codex이며독립수어전문가정답아님.'
)
nxt=(
 'AB FIT216 및 DEV80 실제입력·직접정답 검산완료; 추출/시각검토 재실행금지. '
 'PROCESS_AUTO_NATIVE_MANIFEST는기존full-native input/result/path 48개에서60query 단일proposal. 기존 canonical FIT01 guidehull+공통12context, 수동DEV경계사용금지. '
 'retention 자동proposal/qualification/evaluation 코드와 predicate 최종readiness 실제상태확인. PROCESS_FINAL_READINESS_REVIEW 통과뒤 freeze_process_comparison.py, train_process_head.py seed42/final80epoch 한회, evaluate_process_head.py FIT217/DEV80 한회, verify_process_evaluation_saved.py 저장검산, evaluate_process_auto.py 자동full-native60 및 saved297qualification. '
 '현재 '+train_state+'. PROCESS_TRAIN_PROGRESS/실제PID/세션확인후중복금지. 완료평가 재채점/재추론금지. '
 'DEV편집은실제남은과정review로1/0/unknown 확정하고예측전동결. speed다른타임라인의같은픽셀검토재사용은표시하고실제속도성능으로주장금지. DEV는반복개발이므로최종독립시험이라하지않음. '
 'retention PROCESS_TRAINING_DESIGN_REVIEW 및 process_head/train_process_head 코드검토확인. '
 '한후보seed42/final80epoch, RGB768+bodyrelative59+mask+typedquery, unknownloss0/sourcequery정규화/기존모델고정. '
 '수동window뿐아니라기존prototype DTW의full-native 자동후보경로·고정gate·DEV정답을동결한뒤실제학습/새평가한번. '
 '단어ID/편집종류/시간비율/시작끝존재만으로완료판정금지; seam/길이/padding shortcut·정상변이·관측부족따로검증. '
 '완료old평가/AApublicjob 재실행금지, 사용자·멘토/H14/15/ID16 금지, 새다운로드/외부전송/Git/운영교체없음. '
 '대형파생W,작은코드보고C. 현재진행도 PROGRESS_KO.txt 갱신; HTML추가없음. 전체goalactive.'
)
active=dict(training=train_progress,fit_extraction=None if complete else dict(session_id=24461,pid=27692,done=p['done'],total=216),
 development_native_review_words_complete=dev,development_view_preparation='complete80' if dev_views else 'pending',
 development_partial_review_complete=partial_done,development_inputs_verified=dev_inputs,demo_port=8809)
# This checkpoint is mutable current status, not a frozen experiment result.
# Update only this named entry when rerun; preserve all older history entries.
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text(encoding='utf8')
for h,body in [('### 2026-11-02 '+title,summary),('### 2026-11-02 '+title+' 이어받기',nxt)]:
 needle=h+'\n'
 if needle in s:
  start=s.index(needle)+len(needle);ends=[i for i in [s.find('\n### ',start),s.find('\n## ',start)] if i>=0]
  end=min(ends) if ends else len(s);s=s[:start]+'\n'+body+'\n\n'+s[end:]
context.write_text(s,encoding='utf8')
sync_flow.sync(title,summary,nxt,'PROCESS_DIRECT_TARGETS_VERIFIED_INPUTS_AND_DEV_PREPARATION',active)
report=(
 '현재 진행도 — 2026-11-02\n\n'
 '우리 목표: 여러 사람이 수행한 공개 한국수어와 설명을 미리 배운 신경망이 새 영상과 목표 단어를 받고, 중요한 손모양·신체 관계·움직임·순서를 설명 문구와 영상 시간대에 연결해 알려준다.\n\n'
 '현재 위치: 핵심 기능을 연결한 작은 실제 데모를 만들었고, 중요한 동작의 수행 여부를 더 믿을 만하게 판단하도록 보완 중이다. 전체 완료율의 객관적인 분모가 없어 임의의 백분율은 붙이지 않는다.\n\n'
 '1. 자료·동작 설명 연결 — 4표현(순종·잘하다·조상·이집트)에서 연결했다. 모든 단어 준비 완료는 아니다.\n'
 '2. 신경망 지도학습·미세조정 — 실제로 수행했다. SignRep 마지막 투영층 일부도 조정했다. 미세조정 자체의 추가 이득은 아직 입증하지 못했다.\n'
 '3. 새 영상 분석·DTW 정렬 — 실제 데모8809에 연결했다. 배운 기준 흐름에 맞추며 참고 영상9편을 매번 각각 비교하지 않는다.\n'
 '4. 설명·시간대별 피드백 — 일부 손모양·신체 관계·움직임·앞뒤 관계를 표시한다. 단어 전체의 정오, 모든 짧은 누락, 필수 반복 횟수까지 판단하는 상태는 아니다.\n'
 '5. 중요한 차이와 정상 차이 구분 — 현재 집중 보완 중이다. 주요 동작 하나의 출발·도착 모습만 남은 경우와, 그 사이 움직임도 실제 관측된 경우를 나눠 가르친다. 촬영 전체의 처음·끝이나 단순 연결 구간만을 뜻하지 않는다.\n'
 '6. 새 사람·환경·넓은 단어 범위와 속도 — 추가 검증과 개선이 남아 있다. 현재 개발 점수를 실사용자 정확도나 출시 신뢰성으로 표현하지 않는다.\n\n'
 '이번에 진행한 일\n'
 '공개 학습36영상의45과정에서 만든217개 비교 조건을 검토하고 정답 기록을 연결했다. 중요한 움직임이 보이는 경우79, 일부 근거만 남은 경우128, 판단 근거가 부족한 경우10으로 구분했다. 같은 원본을 변형한 자료이므로217명이거나217개의 독립 시험이 아니다.\n'
 '중간을 잘랐어도 다른 반복 동작이 온전히 남아 있는 경우를 발견해 정상적인 관측 근거로 보존했다. 잘린 종류만 보고 오답을 붙이지 않았다. 미확인은 오답으로 학습하지 않는다.\n'
 '순간 관계2360개와 온전히 남은 움직임 구간472개의 원본·시간·연결을 별도 계산으로 확인했다. 구간 정답을 모든 프레임 정답으로 복제하지 않았다.\n'
 f"216개 실제 비교 영상의 관절·영상 특징 입력 준비는 {p['done']}/216 {'완료' if complete else '진행 중'}이다. 잘라내기 전 원본의 영상 특징을 복사하지 않고 실제 남은 화면을 다시 분석한다.\n"+
 ('저장된 실제5082프레임의 입력·원시간·미관측 표시와2160개 출력 파일도 별도 검산을 통과했다.\n' if (U/'PROCESS_INPUTS_VERIFIED.json').exists() else '')+
 '학습과 다른 공개 등록ID11~13의 비교80조건도 직접검토·원시간·입력검산을 마쳤다. 과정 근거30/부분45/미확인5로 구분했다.\n'
 '새 작은 판정부와 실제 학습 입력의 기본 검사는 통과했다. 현재 '+train_state+'. 이번 정확도 개선 결과는 아직 없다.\n\n'
 '남은 핵심 작업\n'
 '비교 정답·실제 입력 검산을 마친 뒤 작은 과정 판정부를 한 번의 고정 조건으로 학습한다. 정상·부분 수행·관측 부족을 구분하는지, 자동으로 찾은 구간에서도 작동하는지 평가한다. 통과한 경우 실제 데모에 연결해 전체 처리시간까지 확인한다.\n'
 '현재 공개 영상 한 편 실측: 분석8.774초, 버튼부터 표시9.551초(GPU). 평균값도 이번 속도 개선치도 아니다. 새 순서 판정부32ms보다 관절·영상 관측이 주된 병목이다.\n\n'
 '기존 C·T3-F 및 이전 데모 보존. 사용자·멘토 촬영은 학습·튜닝·후보 선택에 사용하지 않는다. 검토와 검산은 Codex 및 별도 에이전트이며, 독립 한국수어 전문가 검토로 표현하지 않는다. 전체 목표는 진행 중이다.\n'
)
(U/'PROGRESS_KO.txt').write_text(report,encoding='utf8')
print('Progress synchronized; extraction',p['done'],'/216; development reviews',dev)
