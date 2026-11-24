from pathlib import Path
import json,hashlib,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
P=Path(__file__).resolve().parent;O=P.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert ROOT.name=='WordsInSilence'
 checks=read(P/'FINAL_CHECK.json');follow=read(P/'FOLLOWUP_CHECK.json')
 assert checks['status']==follow['status']=='PASS'
 results=read(P/'EVALUATION_COMPLETE.json');audit=read(P/'FAILURE_AUDIT.json');now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
 next_plan=[
  dict(step=1,task='동작을 찾는 정답과 잘했는지 판단하는 정답을 분리',why='정상 모습과 달라졌을 때 단계 확률까지 낮아져 비교 구간이 사라지는 사례를 확인했다.',
       implementation='공개 FIT의 이미 확인한 구간에 시도한 동작 종류·시작/끝·손별 모양·이동 방향·관측 여부를 별도로 기록한다. 같은 움직임에서 손모양만 바꾼 대조는 단계 위치를 유지하되 모양 차이로 표시한다. 반대 방향/삭제는 의미까지 유지되는지 확인한 대조만 사용하고, 임의로 정상 라벨을 붙이지 않는다.',
       gate='FIT 내부 수행자 분리로 확인하고, 평가 전 조건·입력·경계 정답을 고정한다. DEV의 특정 실패 시점이나 특정 단어를 예외로 외우지 않는다.'),
  dict(step=2,task='모든 동작 후보의 근거를 남기며 별도 학습',why='최고 확률의 짧은 한 구간만 고르면 뒤쪽의 반대 손모양 근거가 사라졌다.',
       implementation='현재 Conv+GRU 공통 손특징을 보존하고 단계 찾기/관측된 방향/특징 차이 감독을 분리한다. 후보 시간구간을 여러 개 유지하고 그 안의 모양·방향 근거를 모은다. 후보 사이의 미관측 시간을 보간해서 성공 근거로 만들지 않는다. 구간 찾기 실패와 손 미검출의 이유도 구분한다.',
       gate='새 연결부는 정상·역방향·모양 변경·복합 변화·교차·속도 차이·관측 부족을 함께 통과해야 한다. 형상 분류 성능만 좋아졌다고 연결 완료로 보지 않는다.'),
  dict(step=3,task='정말 빠진 동작과 잠깐 안 보이는 동작을 나눠 검사',why='기존 시간 일부 삭제는 중요한 동작 전체 삭제의 정답이 아니어서 누락 검출률을 주장할 수 없다.',
       implementation='검토한 FIT 사건 시작/끝을 이용해 전체 사건 삭제와 관절만 가림을 별도 대조로 만든다. 검토 가능한 짧은 변화·반복을 먼저 포함한다. 순서/반복/끝났는지 판단은 사건 목록으로 비교하고 DTW가 빈 사건을 채우지 못하게 한다.',
       gate='사건이 관측되면 존재, 삭제 근거가 충분하면 누락 후보, 관측 부족이면 미확인으로 구분한다. 횟수가 명시되지 않은 설명에서 임의의 반복 횟수를 정답으로 만들지 않는다.'),
  dict(step=4,task='통과한 범위만 시각 피드백으로 연결',why='사용자가 무엇을 어떻게 고칠지 바로 보여주는 것이 최종 목적이다.',
       implementation='참고/사용자 영상을 나란히 두고 실제 차이 구간을 짧게 반복한다. 해당 손·팔만 강조하고 한 문장으로 안내한다. 단계 색과 판단 색을 분리하며, 미구현 특징을 촬영 실패로 안내하지 않는다. 순수 신경망 시간과 촬영 종료→화면 결과 시간을 별도로 잰다.',
       gate='현 기준보다 정상 오경고·중요 차이 누락·처리시간이 악화되지 않는지 공개 분리 자료로 검증한다. 독립 평가 자료 가능성을 출처/중복/노출 감사로 먼저 확인하며, 기존 DEV 반복 결과를 새 일반화 성능으로 포장하지 않는다.')]
 write(P/'NEXT_PLAN.json',dict(objective='관측한 모양, 동작이 일어난 시간·방향·순서를 따로 배우고 연결해 근거 있는 피드백 제공',steps=next_plan,new_detector_or_big_model=False,word_specific_exception=False,private_training=False,expert_correctness_not_claimed=True))
 model_lines=[]
 for name,r in results['models'].items():
  model_lines.append(f"{name}: 단계 {r['stage']['correct']}/84, 단계별 평균 재현율 {audit['models'][name]['stage']['macro_recall']*100:.2f}%, 판정부만 중앙값 {r['head_seconds']['median']*1000:.2f}ms")
 report='''이번 작업의 결론

사용자가 제안한 역할 분리는 타당하다. 이번에는 실제로 손모양 관측 모델을 고정하고 단계/차이 판정부를 별도 추가 학습했다. 그러나 판정부를 나누고 상대 위치 입력을 늘리는 것만으로는 개선되지 않았다. 단계 찾기와 정오 판단의 학습 정답까지 분리해야 한다는 구체적인 문제가 확인됐다.
새 모델은 웹에 적용하지 않았다. 대신 놓친 근거를 숨기고 비슷하다고 말하는 연결 문제를 찾아, 반대 근거가 남아 있으면 판단을 보류하는 공통 보완도 별도 구현·검증했다. 이 보완은 오류를 정확히 교정하는 기능의 완성이 아니며 연구 후보에만 적용했다.

무엇을 했나
손모양을 읽는 공통 Conv+GRU 모델의 가중치를 고정했다. 같은 공개 4단어, 기록 ID01~09의 36영상으로 별도 판정부를 학습하고 ID11~13의 12영상으로 비교했다. 기록 ID와 원본 SHA가 분리되어 있으며 외형으로 다른 사람을 보증하지 않는다.
조건 A는 기존 관측 특징, 조건 B는 같은 특징에 몸통 기준 손·팔 위치와 프레임 사이 변화량을 추가했다. 각 조건은 고정된 세 seed로 한 번씩, 총 6회 학습했다. 기존 손모양 출력은 두 조건이 동일하게 공유한다. 새 100단어 전체 재학습은 하지 않았다.
평가는 기존 RGB 검토의 84시점과 129개 공개 입력 조건으로 진행했다. 129는 서로 다른 129사람/원본 영상이 아니라 같은 12개 개발 영상의 정상·상반신 범위·속도 변화·손모양 변형·역방향·관절 가림·시간 삭제 파생 조건 수다. 손모양/방향 변형은 합성 관절 시험이며 실제 사람이 틀린 영상의 정확도가 아니다. RGB 정답 검토와 별도 산술 검증은 같은 Codex가 수행한 것이며 수어 전문가의 독립 정답은 아니다.

핵심 결과
현재 웹 판정부 저장 결과: 71/84시점에서 단계 일치.
기존 관측 특징에 맞춰 별도 학습한 대표 A: 68/84.
손·팔 상대 위치와 변화를 더한 대표 B: 66/84.
세 seed 범위도 A 67~70/84, B 66~69/84여서 새 모델의 단계 성능 향상은 확인하지 못했다. 최고 seed를 사후 선택하지 않았다. 이전 보고의 72/84→71/84는 공통 관측 모델 내부 단계 출력 비교이고, 이번 71/84는 실제 웹의 별도 단계 판정부 출력이다. 분모가 같아도 출력 계보가 다르므로 혼동하면 안 된다.
대표 A/B 모두 정상 원본 12영상에서 새 차이 경고는 0이었다. 전체 필수 카드의 확인 특징이 비슷함은 8영상, 불확실한 카드 포함은 4영상이었다. 이는 단어 전체 정답 판정이 아니다.
방향만 반대로 한 3개 합성 시험에서 A는 3개 차이, B는 1개 차이·2개 미확인이었다. 모양만 바꾼 3개에서 A는 3개 차이, B는 2개 차이·1개 잘못된 비슷함이었다. 모양+방향 복합 3개는 두 조건 모두 차이를 표시했지만 이것만으로 모든 성분을 올바르게 찾았다고 볼 수 없다.
손 관측을 전부 지운 12개는 여섯 모델 모두 비슷함 0, 미확인 12였다. 이를 신규 개선으로 주장하지 않는다.
기존 작은 방향 신경망도 자동 구간에 연결해 별도 확인했다. 복합 변화에서 기하학적 반대 방향 근거와 신경망의 정상 방향 출력이 충돌하는 1사례가 남았다. 따라서 웹의 기존 상대 방향 계산을 이 신경망으로 바꾸지 않았다.
기존 시간 삭제 24개는 반복 일부를 줄이거나 경계만 잘라냈을 수 있다. 동작 전체 누락 정답이 확보된 24개라고 보고하지 않는다.

실제로 확인한 원인
공개 잘하다 ID12의 합성 손모양 변경 시험에서 두 조건은 똑같은 손모양 출력값을 받았다.
앞 후보 구간 59~62프레임: 오른손 펴기 평균 확률 0.837, 닫기 0.010.
뒤 후보 구간 75~82프레임: 펴기 0.004, 닫기 0.989.
기존 연결 함수는 단계 확률의 최고점이 가장 높은 후보 하나만 고른다. B는 앞의 짧은 구간을 선택했고, 뒤쪽 닫힌 손 근거가 있었는데도 검사가 그곳까지 가지 않아 비슷함으로 끝났다. 모양 AI가 아예 몰랐던 것과 구간 선택이 근거를 버린 것은 다르다. 이 수치는 신경망의 확률이지 사람이 보증한 정확도 퍼센트가 아니다.
역방향 ID11/13에서는 양손 관측이 있는 프레임이 각각39개였지만 필요한 단계의 최대 확률은0.313/0.003으로 낮아 구간을 찾지 못했다. 사진/관절이 전부 없어서 생긴 문제는 아니었다. 정상 모습 위주의 단계 학습이 오류 동작을 단계 밖으로 밀어낼 가능성이 있으나, 그 학습 원인 자체를 인과적으로 입증한 것은 아니다.

추가로 고친 부분
단어 이름에 따른 예외를 만들지 않고 모든 카드의 다른 후보 구간도 각각 확인하게 했다. 선택하지 않은 후보에서 기존 기준으로 확인되는 반대 손모양 근거가 있으면 비슷함을 미확인으로 낮춘다. 후보 사이의 빈 시간을 채우지 않으며, 어느 후보가 진짜 동작인지 모르는 상태에서 오답으로 단정하지 않는다.
저장된 6모델×129조건에 적용한 결과 B의 세 seed 모두 동일한 1개 합성 손모양 사례만 비슷함→미확인으로 바뀌었다. 정상·상반신 범위·속도·관절 가림 조건의 결과는 유지됐다. 실제 독립 새 자료 성능 향상이나 정답 교정 성공으로 주장하지 않는다. 기존 고정 비교를 덮어쓰지 않고 FOLLOWUP_RESULT로 분리했다.

현재 AI가 하는 일 / 아직 부족한 일
1) 손 관측: 검지만 펴짐, 네 손가락 펴짐, 닫힘 같은 일부 기본 모양과 양손 상대 높이를 읽는다. 모든 손모양·엄지/손끝 접촉을 안다고 할 수 없다.
2) 단계: 상세 피드백 4단어의 설명을 구조화한 구간/특징/순서와 연결한다. 문장 전체를 사람처럼 이해하는 언어 모델은 아니다. 공통 관측을100단어로 배웠어도100단어 상세 피드백이 완성된 것은 아니다.
3) 방향: 현재 웹은 몸통·반대팔 기준의 상대 이동 계산도 사용한다. 작은 신경망의 방향 출력은 시험 중이며 모든 회전·반복을 검증하지 않았다.
4) 위치: 현재280입력은 몸통9관절+양손이다. RTMW의 상세 얼굴점은 원본 결과에 있으나 현재 신경망 입력에는 전부 들어가지 않는다. 코끝/광대/이마의 구체적 접촉 피드백이 완성됐다고 할 수 없다. 필요한 표현에서 기존 얼굴점을 선택해 쓰는 후속 설계가 필요하며 표정은 범위 밖이다.
5) 정렬: DTW는 같은 동작을 비교하고 재생할 시간을 연결한다. 없는 동작을 만들어 주거나 정상이라는 정답을 제공하는 기능이 아니다.

사용자에게 보여줄 방식(다음 적용 설계이며 실제 이번 판정 예시가 아님)
참고 영상 | 사용자 영상 → 차이가 있는 짧은 구간 반복 → 해당 손·팔 강조 → 한 문장 안내.
모양 차이: ‘팔을 쓸어내리는 부분에서 오른손을 펴보세요.’
방향 차이: 화면 화살표와 함께 ‘팔꿈치 쪽에서 손목 쪽으로 쓸어보세요.’
빠진 동작: 근거가 충분할 때만 ‘이 동작까지 이어서 해보세요.’
미확인: ‘이 부분은 아직 판단하기 어려워요.’ 관측 부족인지 모델이 아직 배우지 못한 특징인지 구분한다.
동작 단계 색상 막대와 판정 색상 막대를 분리한다. 초록은 확인한 특징이 비슷함, 주황은 근거 있는 차이, 회색/무늬는 미확인이다. 전체95%정답 같은 검증되지 않은 점수는 쓰지 않는다.

다음 작업의 우선순위
'''+ '\n'.join(f"{s['step']}. {s['task']}\n   근거: {s['why']}\n   작업: {s['implementation']}\n   통과: {s['gate']}" for s in next_plan)+'''

이 순서가 필요한 이유
단계 찾기 → 모양·방향 확인 → 순서·누락 확인 → 짧은 시각 안내를 연결한다. 틀린 모양으로 팔을 쓸었다면 ‘쓸기 동작을 시도한 구간’은 찾되 모양은 틀린 것으로 가르쳐야 한다. 잘못된 동작을 정상으로 배우게 하는 것이 아니다. 단지 정상과 다르다는 이유로 검사할 구간 자체가 사라지지 않게 하는 것이다. 전체 학습량을 무작정 늘리거나 새 거대 모델을 찾는 것보다, 이번에 확인한 연결 손실과 정답 부족을 먼저 해결한다.

검증·속도·적용 상태
'''+ '\n'.join(model_lines)+f'''
추가 판정부만 약1~2ms 수준이며, 공통 신경망·관절 추출·DTW·화면 표시를 제외한 값이다. 전체 웹 응답 시간을 이번에 다시 측정하지 않았다.
고정 비교 저장 결과 {checks['checks']}개 산술/해시 확인, 후속 보완 {follow['checks']}개 확인 통과. 이는 독립 전문가 검증이 아니다.
이번 변경은 새 실험 폴더 안의 학습·평가·후속 보완 코드와 문서다. 기존 C·T3-F·현재TCN·웹피드백 모델·웹 화면은 유지했다. 사용자/멘토 자료 접근·학습·튜닝·후보 선택0, 다운로드0, 카메라0, 서버 시작0, 운영 교체0. 완료된 훈련/평가는 반복하지 않았다.
명확한 전체 성능 개선은 없어 새 HTML은 만들지 않았다.

설계 참고 근거
MS-TCN(CVPR2019)은 시간별 동작 구분을 별도의 학습 문제로 다룬다. 역할 분리의 일반적인 근거이지 한국수어 교정 성능을 증명하거나 새 모델을 도입한 근거는 아니다.
https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html
Gradient Surgery(NeurIPS2020)는 여러 목표를 동시에 학습할 때 간섭 가능성을 다룬다. 이번 회귀의 원인이 그 간섭이라고 입증한 것은 아니고 PCGrad를 도입하지도 않았다.
https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
핵심 판단은 본 폴더 FAILURE_AUDIT, FOLLOWUP_RESULT, EVALUATION_COMPLETE, FINAL_CHECK, FOLLOWUP_CHECK의 실제 공개 저장 결과에 근거한다.
'''
 (P/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 status='COMPLETE_SEPARATE_HEADS_NO_ACCURACY_GAIN_CONFLICT_GUARD_VALIDATED_OFFLINE'
 write(P/'RESULT.json',dict(status=status,updated_at=now,baseline_stage=results['baseline_stage'],models=results['models'],followup=read(P/'FOLLOWUP_RESULT.json'),verification=[str(P/'FINAL_CHECK.json'),str(P/'FOLLOWUP_CHECK.json')],report=str(P/'RESULT_KO.txt'),next_plan=str(P/'NEXT_PLAN.json'),deployed=False,private_used=False,expert_ground_truth=False))
 write(P/'CURRENT_EXECUTION.json',dict(status=status,pid=None,training_active=False,evaluation_active=False,next='Separate attempted-event localization from feature correctness supervision; see NEXT_PLAN.json. Do not rerun completed comparison.'))
 summary='손모양 관측 모델을 고정하고 단계/차이 판정부 2조건×3seed 추가 학습 및 공개129입력 조건 비교 완료. 36FIT/12역사적DEV,84검토시점에서 현재 웹 별도 단계출력71→관측특징만68/상대위치추가66(대표42), 개선 없음. 후자는 역방향3중2구간 미검출·모양변형3중1비슷함 오류. 저장출력 감사로 최고점 후보 하나만 선택해 뒤쪽 반대 손모양 근거를 버리는 연결 문제 확인. 공통 후속 보완은 다른 후보에 반대 근거가 있으면 비슷함→미확인으로 보류; 6×129저장조건에서 해당1사례만 세seed 공통 수정, 정상/속도/가림 결과 유지. 전체손누락12는 모두 비슷함0. 고정비교2108+후속2153 검산 통과. 학습 성능 향상/정확한 교정 완성/독립 새평가를 주장하지 않음. 연구 후보만 변경, 웹·C·T3F·현재TCN 유지, 개인자료 접근0, 새HTML 없음.'
 nxt='다음 단일작업: 공개 FIT에서 시도한 동작 구간과 잘 수행한 모양·방향·순서 정답을 분리한다. 오류 모습 때문에 비교 구간 자체가 사라지지 않도록 정상/대조의 사건 위치와 정오를 별도 감독하고, 여러 후보 구간의 근거를 보존한다. 실제 전체 사건 삭제 대조와 관절만 미관측 대조를 구분해 누락 정답을 보강한다. 세부4단계는 word100_separate_phase_learning_20261109/NEXT_PLAN.json. FIT 내부 수행자 분리로 조건을 고정하고 기존DEV는 회귀 진단으로만 사용한다. 완료6학습/6평가/후속결과 재실행 및 사후 문턱·seed 선택 금지. 새모델/검출기/개인자료/카메라/운영교체로 확대하지 않는다.'
 # Preserve previous state before synchronizing current documentation.
 before=P/'before';before.mkdir(exist_ok=True)
 for f in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
  target=before/f.name
  if not target.exists():shutil.copy2(f,target)
 title='### 2026-11-09 모양과 단계 분리 학습·후보 구간 근거 손실 확인'
 ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text(encoding='utf-8')
 assert title not in text
 for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+' 근거: `research/front_keyphase_v1/ops/'+P.name+'/RESULT_KO.txt`. 같은 Codex의 별도 검산이며 독립 전문가 검증 아님.'),('## 9. 다음 단일 작업',nxt)]:
  assert marker in text;text=text.replace(marker,marker+'\n\n'+title+'\n\n'+body,1)
 ctx.write_text(text,encoding='utf-8')
 handoff='# WordsInSilence 최신 인수인계\n\n'+title+'\n\n'+summary+'\n\n'+nxt+'\n\n전체 목표: 한국수어 설명과 공개 참고의 행동을 배워, 목표 표현에 대해 손별 모양·상대 위치·방향·순서·누락의 중요한 차이를 시각적으로 알려준다. 단계 찾기와 정상 판정을 같은 것으로 학습하지 않는다. 자세한 설명은 RESULT_KO.txt. 이번 추가 판정부 속도는 모델 일부만 잰 값이며 전체 웹 속도가 아니다. 기존 웹8814의 최근 재생 수정은 유지, 이번에는 웹 변경·서버 시작·촬영을 하지 않았다.\n\n현재 모든 신규 학습/평가 명령은 종료. 실행 상태 '+str(P/'CURRENT_EXECUTION.json')+'; 검증 FINAL_CHECK.json/FOLLOWUP_CHECK.json. 이번 코드는 실험 후보만이며 웹 모델 교체 없음. sources 원본과 이전 동결 결과는 읽기 전용, 사용자·멘토자료 접근/학습/튜닝/선택 금지.\n'
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
 state=read(O/'STATE.json');state['current_stage']=P.name;state['next_single_action']=nxt;state['updated_at']=now
 state={P.name:dict(status=status,report=str(P/'RESULT_KO.txt'),result=str(P/'RESULT.json'),execution=str(P/'CURRENT_EXECUTION.json'),verification=str(P/'FINAL_CHECK.json'),followup_verification=str(P/'FOLLOWUP_CHECK.json'),summary=summary,next=nxt,new_model_deployed=False,web_code_changed=False),**state};write(O/'STATE.json',state)
 for f in ['RESULT.json','CURRENT_EXECUTION.json','NEXT_PLAN.json','FOLLOWUP_FROZEN.json']:read(P/f)
 current=read(O/'STATE.json');assert current['current_stage']==P.name and current['next_single_action']==nxt
 assert title in ctx.read_text(encoding='utf-8') and nxt in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf-8')
 write(P/'DOCUMENT_CHECK.json',dict(status='PASS',time=now,root_context=True,state=True,handoff=True,report_sha256=sha(P/'RESULT_KO.txt'),no_html=True))
 print(json.dumps(dict(status=status,report=str(P/'RESULT_KO.txt'),documents='synchronized'),ensure_ascii=False))
if __name__=='__main__':main()
