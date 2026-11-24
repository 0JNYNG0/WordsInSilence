from pathlib import Path
from datetime import datetime
import json,shutil
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
before=U/'before_records';before.mkdir(exist_ok=True)
for p in [O/'STATE.json',R/'NEW_CHAT_HANDOFF.md',R/'WordsInSilence_PROJECT_CONTEXT.md']:
 if not(before/p.name).exists():shutil.copy2(p,before/p.name)
now=datetime.now().astimezone().isoformat()
summary='자동구간에 도달상태 부품 연결 중. 실제 설명의 이동/도달 구간 전달 누락 확인 및 공통 연결 보완. 정상/오류 통합비교·원래 G2 기준 감사 진행 중. 기존 웹과 C·T3-F 변경 없음.'
nxt='독립적으로 찾은 모든 행동과 도달/방향/손모양 증거를 유지한 채 정상·오류 고정 비교 및 별도검산. G2/G3 실제 통과 전 봉인8/웹 승격 금지.'
s=read(O/'STATE.json');s['current_followup']=U.name;s['next_single_action']=nxt;s['updated_at']=now
s[U.name]=dict(status='ACTIVE_NORMAL_ERROR_INTEGRATION',summary=summary,next=nxt,unit=str(U),training_running=False,evaluation_running=False,whole_project_complete=False)
write(O/'STATE.json',s)
write(U/'CURRENT_EXECUTION.json',dict(status='ACTIVE_IMPLEMENTATION',updated_at=now,training_running=False,evaluation_running=False,next=nxt))
(R/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-14 도달상태·자동구간 통합 진행 중\n\n'+summary+'\n\n현재 단위: '+str(U)+'\n'+nxt+'\n\nPLAN.json, INITIAL_COMPARISON.json(웹93+원래role234), CONTEXT_COMPARISON.json(웹93) 완료; 재실행 금지. G2 REWORK, G3 이후 미통과. 기존 goal_state_learning 6학습/383조건 완료결과 읽기전용. 477보호파일 유지. 공개FIT02~07/노출probe08~09/기존DEV11~13만 사용, final8 봉인. 사용자·멘토자료/다운로드/외부전송/Git/카메라 금지. 기존웹 자동재시작 없음. 운영/100공통신경망/C·T3-F 변경 없음. 같은Codex별도검산은독립사람정답이아님.\n\n진행 기록: CONTEXT는 기존 direction계약의 이동+도달 phase_groups 및 카드 .65 경계에서 연속섬별 관측, 앞 1인접샘플만 포함. 정답상태나 예상순서로 구간선택하지 않음. 직접 관측 상태는 전체 수행완료가 아니며 웹표시 false. 원래 graph의G2 baseline/계약은 별도 검증해야 함.\n',encoding='utf-8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-14 도달 상태·자동구간 통합 / 정상·오류 검증 진행 중'
if title not in t:
 for head,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
  marker=head+'\n\n';assert marker in t;t=t.replace(marker,marker+title+'\n\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`, `CURRENT_EXECUTION.json`.\n\n',1)
 p.write_text(t,encoding='utf-8')
print('Active checkpoint saved; graph not promoted.')
