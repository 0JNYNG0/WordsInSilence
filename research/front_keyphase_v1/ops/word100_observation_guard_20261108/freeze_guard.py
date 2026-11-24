from prepare import *
from guard import CONFIG,VERSION
assert not (U/'GUARD_FROZEN.json').exists()
plan=read(U/'INPUT_PLAN.json')
cal=[r for r in plan['rows'] if r['role']=='calibration']
assert all((U/'crops'/(r['clip_id']+'.npz')).exists() for r in cal)
write(U/'GUARD_FROZEN.json',dict(version=VERSION,config=CONFIG,calibration_clips=[r['clip_id'] for r in cal],
    basis='ID01/02 geometry profiles only: normal in-frame palm/arm up to about0.64, span/arm1.10, rootgap0.20 in inspected quantiles; relaxed engineering cutoffs0.8/1.4/0.5. Weak score ratio alone cannot reject. Boundary evidence also requires lowered forearm and3 nearby observations. Raw scores are not visibility probabilities.',
    point_of_no_return='Frozen before any candidate TCN evaluation or noncalibration crop inspection. Do not change these constants using regression results.',
    pins={str(U/n):sha(U/n) for n in ['guard.py','geometry.py','INPUT_PLAN.json']},
    acceptance='No correct-to-wrong full DEV300 TCN regression. Preserve reviewed normal crossing keyframes. Improve cropped absent-hand contamination and net cropped recognition without left-only/word-specific behavior. All gates must pass for deployment.',
    limits='Cannot detect every anatomically coherent identity swap or fix fine-finger inaccuracies. Masked evidence cannot be counted as matching; no smoothing or filled-in coordinate claims.',training=False,private_used=False))
print(VERSION,'frozen',len(cal),'calibration clips')
