import json,hashlib
from pathlib import Path
from datetime import datetime
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_hand_region_uncertainty_20261017'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
scope=read(B/'SCOPE.json');check=read(B/'INDEPENDENT_CHECK.json')
assert check['status']=='PASS'
for p,h in {**scope['pins'],**check['protected_pins']}.items():assert sha(p)==h
dependencies={str(p):sha(p) for p in [U/'separate.py',B/'region.py',O/'word100_hand_visibility_20261017/visibility.py',B/'PREDICTIONS.json',B/'VISUAL_LABELS.json',B/'INDEPENDENT_CHECK.json']}
result={'at':datetime.now().astimezone().isoformat(timespec='seconds'),
 'authorization':'User approved common separation of position from hand-shape/overlap uncertainty, preserve useful existing cases, public validation only, no training until joint result review.',
 'implementation':'Local Codex; independent saved-output contract audit by separate code, not another agent/human.',
 'design':'No new numeric thresholds. Current arm-chain position projection is independent of region uncertainty; observed/partial/unobserved shape is separate; old proximity cue distinguished from actual region-supported suspicion. No aggregate state, cause certainty, landmark recovery, mask/DTW/exclusion or training connection.',
 'comparison':'Use previous frozen60 public TRAIN videos/15926 hand states and141 manual labels as known regression evidence, NOT fresh unseen validation. Separate table for earlier38 challenge samples. Preserve old wrong cases in report. Stream adapter checked on two complete public clips, chosen now: WORD0500_REAL06 and WORD0071_REAL04. No new TCN or detector runs.',
 'acceptance':['All existing mapped location states/evidence preserved across15926 hand states','Region uncertainty cannot overwrite location under metamorphic perturbations',
               'Missing shape never reported as observed, region estimate never as recovered landmarks','Proximity hint never represented as confirmed occlusion',
               'Report residual location errors and unknowns; no physical accuracy gain inferred from structural preservation'],
 'prohibited':['user/mentor recordings','new training','classifier inference','new detector','camera','DEV/heldout','production changes','Git','downloads','coefficient sweeps'],
 'dependencies':dependencies,'source_pins':scope['pins'],'protected_pins':check['protected_pins'],
 'records':scope['records'],'stream_validation_clips':['NIA_SL_WORD0500_REAL06_F','NIA_SL_WORD0071_REAL04_F']}
with (U/'SCOPE.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'clips':len(result['records']),'parameters_changed':0,'frozen':True}))
