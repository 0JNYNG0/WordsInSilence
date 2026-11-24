from pathlib import Path
import json,hashlib,shutil
from datetime import datetime
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not(U/'PLAN.json').exists()
(U/'before').mkdir()
for name in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md']:shutil.copy2(R/name,U/'before'/name)
shutil.copy2(O/'STATE.json',U/'before/STATE.json')
write(U/'PROTECTED.json',read(O/'word100_finger_center_temporal_20261115/PROTECTED.json'))
write(U/'PLAN.json',dict(at=datetime.now().astimezone().isoformat(),priority='Working sign-learning game comes before presentation packaging. User explicitly corrected previous prioritization.',
 goal='Capture -> auto end -> fast evidence-based decision -> verified lesson success OR localized correction/reason OR uncertain retry -> next attempt/quest.',
 scope='Current4word supported web scope; keep100word TCN/common model and C/T3F; no private training or camera auto-start.',
 graph=[dict(id='F0',task='Inspect current game success and feature-evidence coverage',gate='Distinguish recognition-only progression from action-correctness, enumerate missing positive/negative evidence'),dict(id='F1',task='Unified outcome with positive criteria, corrections, uncertainty and completion evidence',gate='Top1 match alone cannot pass; missing/partial cannot become completed through DTW; preserve observed errors'),dict(id='F2',task='Repair necessary evidence/learning gaps on public sources',gate='Correct normal performances progress and relevant controlled errors do not; do not merely block everything'),dict(id='F3',task='Connect qualified outcome to practice/quest/retry UI',gate='Exactly-once progression, localized evidence and short reason; unknown is not failure'),dict(id='F4',task='Physical camera auto-end, load and whole waiting time acceptance',gate='User-started one/twohand normal/error tests, no private tuning; verify portable run'),dict(id='F5',task='Record working demo and produce submission evidence',gate='Do not call static/synthetic playback a successful user demo')],
 validation='Fixed existing public122 archive, supplemented public raw as needed, new training must separately freeze FIT/holdout before running. Frozen saved predictions may diagnose missing interface coverage, not prove new unseen accuracy.',
 forbidden=['top1-only success called action success','absence of warnings treated as complete','unknown filled as true','per-word exception to force pass','changing threshold/model on user footage','labeling unfinished research as completed game'],
 current_gap='QuestProgress.consume currently increments solely from TCN top1. Feedback whole_sign_correct is None. Static-role cards lack shape observations; alternating repetition and gather/spread completion unresolved.',
 implementation='Codex',verification='Separate Codex checks, not independent expert'))
print('functional completion contract saved')
