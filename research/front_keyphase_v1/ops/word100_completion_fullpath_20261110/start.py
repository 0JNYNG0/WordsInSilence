from common import *
import shutil
assert not (F/'AUTHORIZATION.json').exists()
(F/'before').mkdir(exist_ok=True)
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,F/'before'/p.name)
write(F/'AUTHORIZATION.json',dict(request='그럼 그 개선 방안대로 작업을 바로 진행해줘.',date_kst='2026-11-10',scope=['Public RTMW raw replay through actual consumer.raw, TCN, feedback, frozen completion guard and DTW','Audit unused public recordings before frozen supplementary validation','Public-only isolated presentation if justified'],prohibited=['C/T3F/currentTCN/default web replacement','user/mentor data','training or selecting parameters on evaluation','camera','downloads','external transfer','Git'],training=False,implementation='Codex local',verification='Same Codex separate saved-output checks, not independent expert'))
write(F/'CURRENT_EXECUTION.json',dict(status='PREPARING_RAW_REPLAY_AND_EXPOSURE_AUDIT',training=None))
s=read(O/'STATE.json');s['current_stage']=F.name;s['next_single_action']='공개 RTMW 원본의 실제 분석경로 연결과 새 공개 촬영의 노출·중복 감사를 수행';s[F.name]=dict(status='PREPARING',execution=str(F/'CURRENT_EXECUTION.json'));write(O/'STATE.json',s)
print('STARTED',F.name)
