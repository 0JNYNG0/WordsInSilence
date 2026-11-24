from common import *
import re,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
assert not(U/'PLAN.json').exists()
manifests=list(read(H/'EXPOSURE_AUDIT.json')['manifests'])+[str(p)for p in [H/'FROZEN.json',H/'LEFT_COVERAGE_SEARCH.json',Q/'FROZEN.json',Q/'FIT_LABELS_V2.json',N/'LABELS.json']]
used=set()
for f in manifests:used.update(re.findall(r'NIA_SL_WORD\d+_REAL\d+_F',Path(f).read_text('utf-8-sig')))
available={w:[p for p in range(10,19)if f'NIA_SL_{w}_REAL{p:02d}_F'not in used]for w in ['WORD2551','WORD2748','WORD0071','WORD1290']}
write(U/'EXPOSURE_AUDIT.json',dict(manifests={f:sha(f)for f in manifests},unused_clip_combinations=available,caveat='Fine-finger candidate-selection exposure only. These people/words have past TCN use; not globally unseen people.'))
plan=dict(at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),goal='Separate instantaneous finger posture from temporal change so brief real extension is not overwritten by context',steps=['A source RGB/side audit of FIT brief changes and fixed unused public clip pairs; quality mask not imputed truth','B shared center-state MLP plus separate temporal Conv+GRU change supervision. Reuse frozen100 common/base pipeline. Fixed3 seeds','C compare center-only and context-gated auxiliary against admitted baseline; historic and newly fixed diagnostics, then sealed confirmation','D public122 normal/speed/crossing/error replay. Only passing branch may connect; no camera/private input'],gates=['No extra confident errors or loss of correct states on historic and fresh confirmation; actual brief extension improvement must be correct not merely abstained','No unconfirmed track repair or padding missing points. Unknown source targets excluded. Bad-track counts reported separately','No new normal warnings, preserve existing20details/coarse/direction/order on122 replay','No word/person/gloss inputs or thresholds; FIT people01..09 only; no current/exposed holdout optimization','Fix architecture/config/rules before diagnostic predictions; no test-driven threshold adjustment'],limitations='Codex source annotation/self-verification, not independent expert. Visible projected state only; thumb and exact3D excluded.')
write(U/'PLAN.json',plan);write(U/'PROTECTED.json',read(N/'PROTECTED.json'));(U/'before').mkdir()
for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
s=read(O/'STATE.json');s['current_followup']=U.name;s['next_single_action']='中央時点の指形と前後変化を分離した公開正解監査・学習・比較';s[U.name]=dict(status='SOURCE_AUDIT',plan=str(U/'PLAN.json'));write(O/'STATE.json',s)
(R/'NEW_CHAT_HANDOFF.md').write_text('# 손가락 현재 모양·앞뒤 변화 분리 보완 진행\n\n현재 계약 `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`. 공개 FIT 정답 검토부터 시작. 이전 결과 before/ 보존. 기존 웹/C·T3-F/100단어 유지, 개인 촬영 미사용.\n','utf-8')
print(available)
