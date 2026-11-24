from pathlib import Path
import json, hashlib, shutil
from datetime import datetime
from zoneinfo import ZoneInfo
U=Path(__file__).resolve().parent; O=U.parent; R=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
assert not (U/'PLAN.json').exists()
plan=dict(at=now,authorization='User approves all four audited steps; identify individual fingers instead of requiring handshape names; connect sentence clauses to observed action intervals.',
 steps=[dict(id='A',task='Audit public finger labels and clause-to-action contracts; recheck RGB; retain unknowns'),dict(id='B',task='Compare known source intervals with automatically selected intervals; separate observation failure from localization failure, including joint errors'),dict(id='C',task='Train only evidence-justified shared auxiliary heads with frozen common100 backbone; fixed final steps and seeds; no word exception'),dict(id='D',task='Paired public regression, simultaneous-error/visibility/chronology checks, source/hash verification, report adoption gate')],
 constraints=['No LLM or 3D work','C/T3F, common100 backbone, current web unchanged during experiment','No private/user/mentor recordings','No new download','No hidden fingers filled from dictionary or class name','No corrected/error condition flags in inference inputs','No treating synthetic controls as real learner error accuracy','No independent test claim on previously exposed public sources'],
 learning=dict(seeds=[42,43,44],finger_steps=600,lr=.001,weight_decay=.001,window_frames=7,confidence=.85,source_person_split='FIT01-09, historical DEV11-13; source/clip-disjoint',candidate='hand-local temporalConv+GRU adapter with common100 frozen contextual features; common hands/fingers; no wordID',checkpoint_selection='fixed final step, all seeds reported'),
 gates=['No false finger labels from negative coarse labels','No normal known-feature regression for release','Do not improve by abstention alone','Keep all automatic candidates including wrong-shape attempts','Unknown observations never turned into error or completion','Preserve native timeline and independent order evidence','No web promotion unless all relevant gates pass'],
 implementation='Codex',verification='Separate local saved-output/input/hash audit by same Codex; not independent KSL expert',web_promotion=False,whole_project_complete=False)
write(U/'PLAN.json',plan)
b=U/'before_start';b.mkdir()
for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,b/p.name)
prot=read(O/'word100_handshape_transfer_20261113/FROZEN.json')['protected']
# Include every current model/code pin already protected by the final live gate.
for name in ['FROZEN.json','VERIFIED.json','CONNECTION.json']:
 p=O/'word100_live_g6_20261114'/name
 if p.exists():prot[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
write(U/'PROTECTED.json',prot)
state=read(O/'STATE.json');state['current_followup']=U.name
state['next_single_action']='공개 설명·손가락 정답 감사→확인구간/자동구간 원인분리→필요 판정부 학습→복합 오류·정상 회귀 검증. LLM/3D 제외; 실제 사용자 촬영은 보류 유지.'
state[U.name]=dict(status='IN_PROGRESS',at=now,plan=str(U/'PLAN.json'),training_running=False,web_changed=False,private_access=False)
write(O/'STATE.json',state)
title='### 2026-11-14 손가락별 근거·설명 행동구간 연결 보완 승인'
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for h in ['## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 9. 다음 단일 작업']:
 doc=doc.replace(h+'\n\n',h+'\n\n'+title+'\n\n'+state['next_single_action']+' 신경망 판단과 수동 구조화 설명을 구분한다. 기존 C·T3-F/공통100/현재 웹 보존, 사용자 자료 학습·튜닝 금지.\n근거: `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# 설명 연결·손가락별 판단 원인분리 진행 중\n\n'+state['next_single_action']+'\n\n현재 단위 '+str(U)+'; PLAN.json과 CURRENT_EXECUTION.json 확인. 사용자 최신 지시로 공개자료 연구 재개. G6 실제 웹캠 확인은 사용자 선택대로 보류, 카메라 자동 시작 금지. G0~G5 제한 완료가 전체 오류 검출 완성을 뜻하지 않는다. 모델 결과와 조건문/설명계약을 구분해 보고한다. 현재 웹을 새 후보로 교체하지 않으며 C·T3-F 보존. 사용자·멘토 자료 접근 없음.\n','utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='A_SOURCE_AUDIT',pid=None,at=now))
print(str(U))
