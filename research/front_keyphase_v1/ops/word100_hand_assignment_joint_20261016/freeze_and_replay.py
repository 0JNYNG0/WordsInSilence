"""Pin candidate before the single real-input comparison, preserving failures."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,traceback
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
assert not (U/'CANDIDATE_FREEZE.json').exists()
names=['SCOPE.json','assignment.py','research_landmarker.py','POINT_CONTRACT.json',
       'replay.py','PRE_REPLAY_REVIEW.json','CODE_REVIEW.md','BROADENED_TRAIN_PLAN.json']
pins={str(U/n):sha(U/n) for n in names}
new(U/'CANDIDATE_FREEZE.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),pins=pins,
    status='FROZEN_BEFORE_ANY_CANDIDATE_REAL_INPUT_REPLAY',
    additional_safety='Independent synthetic stale-opposite counterexample repaired before real-input comparison; no threshold sweep.',
    training=0,classifier_inference=0))
try:
    import replay
    replay.run()
except Exception as exc:
    new(U/'REPLAY_FIRST_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    raise
