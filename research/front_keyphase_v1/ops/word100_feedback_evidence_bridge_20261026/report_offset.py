from pathlib import Path
import json,html,hashlib
from datetime import datetime,timezone,timedelta
R0=Path(__file__).resolve().parent;U=R0/'offset_candidate';O=R0.parent;R=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
v=read(U/'VERIFIED.json');assert v['status']=='PASS'
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
lines=['반 프레임 시간 민감성 시험 — 자동 교정 기준으로는 미채택',
 '전체 개선 작업은 계속 진행 중입니다. 기존 C·T3-F·운영 웹·DTW는 변경하지 않았습니다.',
 '기존 공개27영상·3단어·9개 촬영자 ID의 2,808개 저장 대조를 다시 활용했습니다. 별도의 새 사용자 시험은 아닙니다. 실제 동작의 정답 시점을 후보에게 알려주지 않았습니다.',
 '한 손의 앞·중간·뒤를 같은 방향으로 반 프레임씩 옮겨 보는 연구 진단을 구현했습니다. 한 손의 모든 비교 가능한 각도·좌표에 같은 시간 이동을 사용합니다. 관측이 줄어들거나 새로운 차이가 생기는 이동은 채택하지 않습니다. 원래 순간 비교와 원래 연속 차이는 전부 보존했습니다.',
 '정상 속도 차이: 같은 원본을 빠르게·느리게 바꾼108개 시험에서 새 차이 표시가 있던 항목은552→269개였습니다. 사람 수나 인식률이 아닌 참고시간×손×각도/좌표 칸 수입니다. 서로 다른 사람의 정상 속도864쌍에서는40,685→38,101개로 줄었으나, 원본에서의 차이가 없어지면서 상대적으로 새 표시가 되는5,766칸도 있었습니다. 전체 경고가 줄었다고 모든 경우가 좋아진 것은 아닙니다.',
 '짧은 동작 신호 손실: 원본 중간3프레임을 제거한81개 변형을 같은 원본과 비교했을 때, 제거 시간대에 새 차이를 보인 경우가63→41개로 줄었습니다. 다른8명 참고 중5개 이상에 새 차이가 나타난 경우도70→64개였습니다. 세부적으로9개는 신호를 잃고3개는 새로 얻었습니다. 이81개는 인위적인 시간 제거이며 의미상 필수 동작의 정답 주석이 아닙니다.',
 '판단: 정상적인 시간 차이에 민감한 부분을 찾는 데는 도움이 되지만, 이를 잘 맞음으로 바꾸면 짧은 변화도 가릴 수 있습니다. 따라서 원래 차이를 지우거나 감점을 취소하는 기능으로 채택하지 않습니다. 시간 연결에 따라 결과가 달라질 수 있다는 진단 정보로만 보존합니다. 특정 단어 예외나 문턱 재조정은 하지 않았습니다.',
 '단순 경고 감소와 신뢰할 수 있는 개선은 다릅니다. 코드가 실행되고 검산을 통과해도 동작 피드백 기준으로 채택될 수 있다는 의미는 아닙니다. 683,208개 손·시간 창의 선택을 별도 코드로 재검산했고, 관측 근거를 잃는9,865개 제안을 거부했음을 확인했습니다. 정상 손·교차의 기존 순간 정보와 비교 범위는 변하지 않았습니다.',
 '중요한 한계: 각 시간 창은 서로 다른 이동을 고를 수 있습니다. 따라서 이 결과는 영상 전체를 일관되게 다시 정렬한 새 DTW 경로가 아닙니다. 이 경로로 수어 인식률을 높였다고 주장하지 않습니다. 관측 부족이나 추정 손 모양도 정상 수행으로 바뀌지 않습니다.',
 '다음 작업: 기존 공개 영상에서 저장해 둔 MediaPipe 재검출·선택적 YOLO·HaMeR 근거를 위치/정밀 손 모양/좌우 신원/시간 추정으로 실제 연결합니다. 이미 확실한 정상 손을 바꾸지 않고, 손 영역만 찾은 경우를 정확한 손목·손가락으로 승격하지 않습니다. 누락 보완과 피드백에 쓸 수 있는 근거가 실제로 얼마나 늘어나는지, 해결되지 않은 좌우 충돌은 무엇인지 같은 자료로 비교합니다. 10개 손목·신원 충돌을 억지로 확정하지 않습니다.',
 '운영에 이미 켜져 있는 YOLO와 혼동하지 않습니다. 현재 YOLO·HaMeR는 연구 후보이며 운영 C·T3-F 웹의 기본 관측은 그대로입니다. 사용자·멘토 촬영, 추가199/200 자료 접근0, 새 학습0, 운영 변경0입니다.',
 '담당과 검증: 동일 Codex 구현 및 별도 저장 산술 검산. 수어 전문가의 의미 판정이나 독립 사람 손 관절 정답 검증은 아닙니다.']
text='\n\n'.join(lines)+'\n';(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
body=''.join('<p>'+html.escape(x)+'</p>' for x in lines)
(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>시간 민감성 검증</title><style>body{max-width:1050px;margin:40px auto;padding:24px;background:#f5f7fb;color:#172433;font:18px/1.8 system-ui}p{padding:20px;background:white;border-radius:12px}p:first-child{font-size:30px;font-weight:700}</style><body>'+body+'</body></html>',encoding='utf-8')
summary='DTW 근거연결2808대조 검산PASS 후 반프레임 공통오프셋 후보까지완료. 같은원본 정상속도 신규차이552→269이나3프레임삭제신호63/81→41/81,다른8참고5개이상70→64(9lost/3gained). 자동오류완화·정상판정 미채택,원래차이보존+시점민감정보만연구보관. 공개개발자료만,운영모델/사용자자료불변,goalactive.'
nextstep='공개 저장 MP재검출/선택적YOLO/HaMeR 실출력을 관측근거연결부에 연결. 위치·손가락·좌우신원·시간추정별 실제 증거를 보존하고 영역추적을 정밀점으로 승격금지. 정상손/교차/기존좌표/비교범위 보존과 실제로추가된 비교근거를 검산;10신원충돌 강제해결금지. 현재미사용새publicGT가필요한부분명시. 운영C/T3F/DTW/웹 교체금지,사용자/멘토/추가199/200 접근·학습·튜닝금지. 작은후보실험종료로goal완료처리금지.'
with (U/'DONE.json').open('x',encoding='utf-8') as h:json.dump(dict(at=at,status='VERIFIED_DIAGNOSTIC_NOT_ACCEPTED_FOR_CORRECTION',goal_complete=False,next=nextstep),h,ensure_ascii=False,indent=2)
context=R/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8-sig')
for prefix,title,content in [('## 8.','### 2026-10-26 시간 민감성 후보 검증 — 경고 감소와 짧은 변화 손실',summary),('## 9.','### 2026-10-26 지속 개선 — 실제 검출 보완 근거 연결',nextstep)]:
    if title not in s:
        i=s.index('\n',s.index(prefix));s=s[:i+1]+'\n'+title+'\n'+content+'\n\n'+s[i+1:]
context.write_text(s,encoding='utf-8')
statep=O/'STATE.json';state=read(statep);state.update(updated_at=at,current_stage='AUTONOMOUS_PUBLIC_DETECTOR_EVIDENCE_INTEGRATION_NEXT',next_single_action=nextstep)
state[R0.name].update(status='BRIDGE_VERIFIED_OFFSET_DIAGNOSTIC_ONLY',summary=summary,offset_result=str(U/'RESULT_KO.html'),running_processes=[],next=nextstep,goal_complete=False)
statep.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n현재: '+summary+'\n\n다음: '+nextstep+'\n\n현재 연구폴더: '+str(R0)+'\n근거연결 RESULT/VERIFIED/CAUSAL_VERIFIED,offset_candidate/RESULT/VERIFIED/RESULT_KO 재사용. exec13370·77902·29494·94989 종료0. 새학습·운영교체없음. 이전경계보강3회/대조3회 결과는word100_boundary_aux_20261026(합성제한개선/정상평균동일/개별회귀). C/T3F/운영/원본보존. goalactive,별도heartbeat없음.\n',encoding='utf-8')
# Check persisted latest documentation and local links, without browser automation.
assert read(statep)['current_stage']=='AUTONOMOUS_PUBLIC_DETECTOR_EVIDENCE_INTEGRATION_NEXT'
assert nextstep in (R/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
assert summary in context.read_text('utf-8')
assert len(read(U/'RESULT.json')['rows'])==2808
with (U/'SYNC_CHECK.json').open('x',encoding='utf-8') as h:json.dump(dict(status='PASS',html='generated; static checks only, not browser-render inspected',goal_complete=False),h,ensure_ascii=False)
print('Verified research result saved; broader goal remains active.')
