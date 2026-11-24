"""Run once after root freezes code; preserve first failure without retry."""
from pathlib import Path
import json,sys,traceback
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
assert (U/'CANDIDATE_FREEZE.json').exists(),'Root must freeze candidate before real replay'
assert not (U/'REPLAY_RESULT.json').exists(),'Reuse completed result'
assert not (U/'REPLAY_FIRST_FAILURE.json').exists(),'Review preserved failure before recovery'
try:
    import replay
    replay.run()
except Exception as exc:
    with (U/'REPLAY_FIRST_FAILURE.json').open('x',encoding='utf-8') as f:
        json.dump(dict(error=repr(exc),traceback=traceback.format_exc(),training=0),f,ensure_ascii=False,indent=2)
    raise
