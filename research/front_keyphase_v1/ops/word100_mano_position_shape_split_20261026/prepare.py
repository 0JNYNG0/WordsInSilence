from pathlib import Path
import hashlib,json
from datetime import datetime,timezone,timedelta
from candidate import PARAMS
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_mano_identity_recheck_20261026'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
f=read(P/'FROZEN.json')
pins=[{'path':str(p),'sha256':sha(p)} for p in P.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'before_closeout' not in str(p)]
for name in ['word100_forearm_contour_20261018/candidate.py','word100_mano_duplicate_guard_20261025/candidate_v2.py','word100_mano_duplicate_guard_20261025/REQUESTED_DIAGNOSTIC_V2.json']:
 p=O/name;pins.append({'path':str(p),'sha256':sha(p)})
f.update(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),task='Separate coarse hand region/arm position evidence from precise finger shape evidence',implementation='Codex local',params=PARAMS,candidate_sha256=sha(U/'candidate.py'),runner_sha256=sha(U/'run.py'),prior_pins=pins,
 protocol='All 243 saved HaMeR requests plus all complete original observations in 812 public frames; same 10 identity conflicts, normal crossing20frames/control94frames and two fixed contact/overlap broad scenes. Existing clean anchor masks reused. Full-frame640px image tracking; no input ROI before observation. Arm contour from prior frozen algorithm; region ShiTomasi+LK distinct from 21 joint-point tracking. Nearest clean anchor each side before/after within.5sec only. No class-specific rules, no sweeps. No new neural inference/training.',
 parameter_origin='Optical thresholds inherited from image_tracking_pair; window.5sec inherited from identity_recheck. Hand convex hull feature region excludes opposite observed hull; region distance uses existing.12shoulder engineering gate. No GT calibration. Contour parameters reused exactly.',
 acceptance='Review every newly supported identity in 10 conflict frames and both fixed broad scenes. Region does not establish fingers or exact palm center. Any contradicted coarse identity fails adoption. Keep coordinates and masks byte-equivalent, no production update; report evidence availability separately from correctness.',
 evaluation='Diagnostic re-use of public TRAIN, not independent final accuracy. Raw coords preserved; research sidecars only. User and mentor inputs excluded. C/T3F and production protected.')
with (U/'FROZEN.json').open('x',encoding='utf-8') as q:json.dump(f,q,ensure_ascii=False,indent=2)
print('Frozen',len(pins),'previous files')
