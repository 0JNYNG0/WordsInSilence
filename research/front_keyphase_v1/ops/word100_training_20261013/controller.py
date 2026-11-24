"""Continue prepared chunks then verify/train/recount without duplicate or silent restart."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import U,OUT,RUN,read,write,sha,pin_check

def run_script(name,args=()):
    log=U/(name.replace('.py','')+'_controller.log')
    write(U/'CONTROLLER_PROGRESS.json',dict(status='RUNNING',pid=os.getpid(),script=name,log=str(log)))
    with log.open('a',encoding='utf-8') as out:
        p=subprocess.run([sys.executable,str(U/name),*args],stdout=out,stderr=subprocess.STDOUT,check=False)
    if p.returncode:raise RuntimeError(f'{name} failed exit{p.returncode}; see{log}')

def main():
    f=pin_check();assert read(U/'INPUT_CHECK_FIRST6.json')['status']=='PASS'
    assert read(U/'SMOKE.json')['status']=='PASS'
    expected=read(U/'EXECUTION_CODE_PINS.json')
    assert all(sha(p)==h for p,h in expected.items())
    start=time.monotonic();write(U/'CONTROLLER_PROCESS.json',dict(pid=os.getpid(),started_at=time.time(),executable=sys.executable))
    try:
        for chunk in range(10):
            n=len(list((OUT/'records').glob('*.json')))
            if n==1199:break
            assert time.monotonic()-start<14400,'Whole preparation exceeded4h budget'
            run_script('prepare.py',['--limit','300','--max-seconds','7200'])
            now=len(list((OUT/'records').glob('*.json')));assert now>n,'No preparation progress'
        assert len(list((OUT/'records').glob('*.json')))==1199
        if not (U/'INPUT_CHECK.json').exists():run_script('verify_inputs.py')
        assert read(U/'INPUT_CHECK.json')['status']=='PASS'
        if not (U/'MEDIA_CHECK.json').exists():run_script('verify_media.py')
        assert read(U/'MEDIA_CHECK.json')['status']=='PASS'
        run_script('sync_state.py',['TRAINING'])
        if not (RUN/'COMPARISON.json').exists():run_script('train.py')
        if not (U/'METRICS_CHECK.json').exists():run_script('verify_metrics.py')
        assert read(U/'METRICS_CHECK.json')['status']=='PASS'
        write(U/'CONTROLLER_PROGRESS.json',dict(status='COMPLETE_AWAITING_REPORT',pid=os.getpid(),elapsed_seconds=time.monotonic()-start))
        write(U/'EXIT.json',dict(exit_code=0,finished_at=time.time()))
    except Exception as e:
        if not (U/'FIRST_CONTROLLER_FAILURE.json').exists():write(U/'FIRST_CONTROLLER_FAILURE.json',dict(message=str(e),trace=traceback.format_exc()))
        write(U/'EXIT.json',dict(exit_code=1,finished_at=time.time(),error=str(e)));raise

if __name__=='__main__':main()
