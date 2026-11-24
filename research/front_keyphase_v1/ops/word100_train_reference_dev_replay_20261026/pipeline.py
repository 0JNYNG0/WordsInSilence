"""Sequential authorized research work; persistent checkpoints prevent duplicate execution."""
from pathlib import Path
import subprocess, sys, json, datetime, os
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent
def record(name,data):
    (U/name).write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),**data),ensure_ascii=False),encoding='utf8')
assert not(U/'PIPELINE_RUNNING.json').exists()
record('PIPELINE_RUNNING.json',dict(pid=os.getpid()))
for args,marker in [(['study.py','freeze'],'FROZEN.json'),(['study.py','calibrate'],'BOUNDS_DONE.json'),(['study.py','run'],'DONE.json'),(['verify.py'],'VERIFIED.json')]:
    if (U/marker).exists():
        print('EXISTING_STAGE_NOT_REEXECUTED',marker,flush=True);continue
    stage='_'.join(args);record('PIPELINE_STATUS.json',dict(stage=stage,status='RUNNING'))
    print('START',stage,flush=True)
    result=subprocess.run([sys.executable,'-B',str(U/args[0]),*args[1:]],cwd=U)
    record('PIPELINE_STATUS.json',dict(stage=stage,status='EXIT',exit_code=result.returncode))
    if result.returncode:
        record('PIPELINE_FAILED.json',dict(stage=stage,exit_code=result.returncode));sys.exit(result.returncode)
    assert (U/marker).exists()
record('PIPELINE_DONE.json',dict(status='PAIR_AND_INDEPENDENT_ARITHMETIC_COMPLETE'))
(U/'PIPELINE_RUNNING.json').unlink()
