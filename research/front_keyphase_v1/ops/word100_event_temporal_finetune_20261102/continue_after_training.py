"""Single bounded continuation of an already-running, identity-checked trainer."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,os,sys,time,subprocess,traceback,psutil
U=Path(__file__).resolve().parent
PID=31156;CREATED=1793598091.4662573
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def live():
    try:
        p=psutil.Process(PID)
        if abs(p.create_time()-CREATED)>0.01:return False
        assert any(Path(x).resolve()==(U/'train.py').resolve() for x in p.cmdline()[1:] if not x.startswith('-'))
        return p.is_running()
    except psutil.NoSuchProcess:return False

def main():
    assert not (U/'DRIVER_LAUNCH.json').exists()
    alive=live()
    assert alive or (U/'TRAINING_RESULT.json').exists()
    for name in ['EVALUATION.json','VERIFIED.json','CAUSE_AUDIT.json','RESULT_KO.txt']:assert not (U/name).exists()
    with (U/'DRIVER_LAUNCH.json').open('x',encoding='utf8') as f:
        json.dump(dict(at=now(),pid=os.getpid(),training_pid=PID,training_created=CREATED,training_live_confirmed=alive,
            steps=['evaluate.py','verify.py','diagnose.py','finalize.py'],no_training_restart=True,no_model_adoption=True),f,indent=2)
    write(U/'DRIVER_PROGRESS.json',dict(at=now(),status='WAITING_FOR_LIVE_TRAINER',pid=os.getpid(),trainer=PID))
    while live():time.sleep(10)
    assert read(U/'TRAIN_PROGRESS.json')['status']=='COMPLETE'
    assert len(read(U/'TRAINING_RESULT.json')['runs'])==4
    for script in ['evaluate.py','verify.py','diagnose.py','finalize.py']:
        write(U/'DRIVER_PROGRESS.json',dict(at=now(),status='RUNNING',step=script,pid=os.getpid()))
        print('Starting',script,flush=True)
        with (U/(script+'.log')).open('x',encoding='utf8') as log:
            result=subprocess.run([sys.executable,'-B',str(U/script)],cwd=str(U.parents[3]),stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(script,result.returncode)
        print('Completed',script,flush=True)
    write(U/'DRIVER_PROGRESS.json',dict(at=now(),status='COMPLETE',pid=os.getpid(),whole_goal_complete=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (U/'DRIVER_FIRST_FAILURE.txt').exists():(U/'DRIVER_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        write(U/'DRIVER_PROGRESS.json',dict(at=now(),status='FAILED',pid=os.getpid()))
        raise
