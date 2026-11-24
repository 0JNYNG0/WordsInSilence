"""New coordinate-basis experiment; reuse the identical original architecture.

No old files or models are changed. Prepared sources supply the canonicalized
63-vector; normalization is still learned only from the fold's training people.
"""
from pathlib import Path
import importlib.util
import sys

U=Path(__file__).resolve().parent
ORIGINAL=U.parent/'hand_transition_learning'
spec=importlib.util.spec_from_file_location('_rotation_original_hand_core',ORIGINAL/'hand_core.py')
original=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
for name in ['np','torch','nn','read','write','sha','stamp','HandObserver','padded']:
    globals()[name]=getattr(original,name)
T=U.parent/'typed_shape_transition_review'
ATTRIBUTES=list(original.ATTRIBUTES)
sys.dont_write_bytecode=True

def preserve_failure(stage):
    import traceback,time
    U.mkdir(exist_ok=True,parents=True)
    p=U/f'{stage}_FAILURE_{time.time_ns()}.txt'
    p.write_text(traceback.format_exc(),encoding='utf-8')

