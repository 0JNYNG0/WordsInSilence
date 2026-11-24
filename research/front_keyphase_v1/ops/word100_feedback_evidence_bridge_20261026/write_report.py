from pathlib import Path
import json,html,hashlib
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
v=read(U/'VERIFIED.json');a=read(U/'CAUSAL_VERIFIED.json');r=read(U/'RESULT.json')
assert v['status']==a['status']=='PASS'
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
txt='''DTW 관측 근거 연결과 정상 속도 경고 원인 검증

전체 개선 목표는 진행 중입니다. 기존 C·T3-F와 운영 웹·DTW는 변경하지 않았습니다.

무엇을 구현했나
연구용 DTW 결과를 손별·각도별·시간별로 전달하는 연결부를 만들었습니다. 순간 모양·위치를 비교할 수 있는지, 앞뒤 움직임까지 비교할 수 있는지를 따로 표시합니다. 한 손가락이 불확실하다고 다른 손가락이나 팔의 관측까지 없애지 않습니다. 참고 영상에서 필요한 시간 범위는 고정하며, 관측 부족을 0차이나 정상 수행으로 바꾸지 않습니다.
직접 관측한 점, 시간 추정, 대략적인 손 영역, 좌우 신원의 근거를 구분하는 입력 계약도 구현했습니다. YOLO·HaMeR라는 모델 이름만으로 정확한 손가락 근거로 승격하지 않습니다. 이 계약의 경우분리 검사는 통과했지만, 모든 검출기를 실제 웹에 연결한 상태는 아닙니다. 이번 재생 자료는 이전 공개 관측 근거를 재사용했습니다.

무엇을 검증했나
기존 공개 TRAIN 27영상(3단어·9개 촬영자 ID)에서 준비한 2,808개 대조를 연결부로 재생했습니다. 새로 촬영·검출하거나 독립 최종 평가를 한 것은 아닙니다. 117,936개 손·항목 결과의 시간 구간과 비율을 별도 코드로 검산했습니다.
정상 원본·속도 변형의 다른 사람 비교 1,080쌍에서 순간 비교 가능 1,148,965칸을 전부 보존했습니다. 그중 430,407칸은 연속 비교가 어렵지만 순간 비교는 유지됩니다. 이 가운데 기존 순간 차이 36,116칸도 지우지 않았습니다. 이는 새 손을 복구하거나 정확도를 높였다는 수치가 아니라 기존 근거를 통합 과정에서 잃지 않았다는 결과입니다.
인위적인 손·팔 누락 135개 대조 모두 참고 비교 범위를 유지했습니다. 근거 없는 결과는 미판정으로 남았습니다. 기존 후원 교차 12프레임의 손 모양·손바닥 근거 부족은 유지하고 팔꿈치·손목 96좌표칸은 비교 가능 상태를 유지했습니다. 실제 교차 손가락의 정답성을 새로 검증한 것은 아닙니다.

새로 분리해 확인한 문제
같은 동작의 속도만 바꾼 108개 시험과 동일 입력 27개를 이용했습니다. 현재 DTW 연결, 실제 대응 시점을 정답으로 알려준 연결을 비교했습니다. 동시에 재표본된 관절값과 원본에서 그 시점의 값을 가져온 경우를 나눠 네 조건으로 확인했습니다. 원본 대응 정보는 진단에만 사용했습니다.
네 조건 모두 비교 가능한 항목에서 정상 속도 변형의 세부 경고는 기존 580개, 원본 값만 복원 573개, 대응 시점만 정확히 맞춤 157개, 둘 다 정답 정보로 복원 0개였습니다. 항목은 참고 프레임×손×좌표/각도×앞·중간·뒤/변화량이며 사람·영상 수나 인식률이 아닙니다. 1,091개 항목은 네 조건의 근거가 동시에 충족되지 않아 이 원인 비교에서 제외됐고, 원래 결과에서는 유지됐습니다.
이는 작은 시점 차이가 세부 변화량 경고에 큰 영향을 줄 수 있다는 근거입니다. 실제 사용자의 정답 대응 시점을 알 수는 없으므로, 이 580→157을 시스템 성능 개선이라고 주장할 수 없습니다. 또 시점 정답만 썼을 때 사라진 경고와 새로 생긴 경고가 함께 있어 단일 원인으로 전부 설명하지 않습니다.

다음 작업
정답 대응 정보 없이, 한 손의 앞·중간·뒤를 같은 방향으로 반 프레임만 함께 이동시키는 제한된 비교를 시험합니다. 관절값과 원래 순간 차이는 그대로 두고, 조금 다른 시간 연결만으로 설명되는 차이를 별도로 표시합니다. 새로운 차이가 생기는 이동은 채택하지 않고, 자료가 없는 구간도 연결하지 않습니다. 정상 속도 경고와 짧은 구간 제거 신호가 각각 얼마나 달라지는지 공개 자료에서 함께 확인합니다. 특정 단어 예외나 결과를 보고 문턱 조정은 하지 않습니다.

현재 의미와 한계
TCN 단어 인식과 DTW 관측 근거를 별도로 다루는 연구 연결부입니다. 운영 웹 적용이나 검증된 교정 문장 생성까지 완료된 것은 아닙니다. 위치가 보인다고 손가락 모양까지 안다고 판단하지 않습니다. 정상 범위를 벗어난 수치도 검출·시간 정렬의 차이일 수 있으므로 아직 자동 교정 지시로 사용하지 않습니다.
이전 경계 보강 연구 후보는 별도로 유지합니다. 같은 DEV의 합성 경계 시험에서 제한적 개선이 있었으나 정상 평균은 같고 개별 회귀가 있습니다. 사용자 웹캠 개선은 새로 측정하지 않았습니다.
구현·별도 수치 검산 담당: 동일 Codex. 2,037,700개 원인 진단 수치를 다른 산식으로 확인했고 최대 차이는 약2.84e-14였습니다. 독립 사람의 손 위치 정답 또는 수어 전문가 검증이 아닙니다. 사용자·멘토 자료 접근/학습/튜닝0, 새 검출·TCN 추론·학습0, 운영 변경0입니다.
'''
(U/'RESULT_KO.txt').write_text(txt,encoding='utf-8')
body=''.join('<p>'+html.escape(s).replace('\n','<br>')+'</p>' for s in txt.split('\n\n'))
(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>DTW 근거 연결 검증</title><style>body{max-width:1000px;margin:48px auto;padding:24px;font:18px/1.8 system-ui;background:#f5f7fb;color:#172433}p:first-child{font-size:30px;font-weight:700}p{background:white;padding:22px;border-radius:12px}</style><body>'+body+'</body></html>',encoding='utf-8')
summary='DTW 연구 근거 연결부 구현·2808기존공개대조 재생/117936손항목 검산PASS. 순간·연속근거 분리,비교시간범위고정,누락135·정상교차 보존. 정상108속도변형의 공통근거 세부경고580→정답시점만157;원본값만573,둘다0. 원인진단이며실제개선아님. 사용자자료0/운영미변경/전체goal active.'
nextstep='정답시간정보없이 공통 반프레임 시간오프셋 후보를 연구검증: 한손의앞중뒤를같은방향으로움직여시점민감차이와남는차이분리. 순간정보/고정분모/관측부족보존;새경고추가금지,단어예외금지;정상속도와짧은삭제조건 모두평가. 기존공개개발묶음만사용;사용자/멘토/추가199/200접근금지. 운영교체금지.'
context=R/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8-sig')
for prefix,title,content in [('## 8.','### 2026-10-26 DTW 관측 근거 연결·정상 속도 경고 원인 검증',summary),('## 9.','### 2026-10-26 지속 개선 — 공통 시간 오프셋 연구 후보',nextstep)]:
    if title not in s:
        i=s.index('\n',s.index(prefix));s=s[:i+1]+'\n'+title+'\n'+content+'\n\n'+s[i+1:]
context.write_text(s,encoding='utf-8')
statep=O/'STATE.json';state=read(statep);state.update(updated_at=at,current_stage='AUTONOMOUS_FEEDBACK_TEMPORAL_SENSITIVITY_NEXT',next_single_action=nextstep)
state[U.name]=dict(status='VERIFIED_RESEARCH_BRIDGE_AND_CAUSAL_DIAGNOSIS',summary=summary,report=str(U/'RESULT_KO.html'),running_processes=[],next=nextstep,goal_complete=False)
statep.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n현재: '+summary+'\n\n다음: '+nextstep+'\n\n현재 연구폴더: '+str(U)+'\nRESULT/VERIFIED/CAUSAL_VERIFIED/RESULT_KO를재사용. exec13370/77902종료0. 이전word100_boundary_aux_20261026 경계보강3회/대조3회 완료, 제한적DEV합성개선·정상평균동일·운영미적용. C/T3F/운영/원본보존. goalactive,별도heartbeat없음.\n',encoding='utf-8')
with (U/'DONE.json').open('x',encoding='utf-8') as h:json.dump(dict(at=at,status='VERIFIED',larger_goal_complete=False,next=nextstep,report=str(U/'RESULT_KO.html')),h,ensure_ascii=False,indent=2)
print(summary)
