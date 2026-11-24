"""Serial preparation continuation only. Never starts training or approves visual evidence."""
from pathlib import Path
import sys,json,datetime,time,subprocess,os,hashlib,traceback
U=Path(__file__).resolve().parent;I=U.parent/'word100_wrist_only_inputs_20261026'
sys.stdout.reconfigure(encoding='utf-8')
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,d):
    with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def status(stage,**kw):
    d=dict(at=now(),stage=stage,**kw);tmp=U/'CONTINUATION_STATUS.tmp';tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf8');os.replace(tmp,U/'CONTINUATION_STATUS.json')
    print(json.dumps(d),flush=True)
def child(unit,args,logname):
    status('RUN_CHILD',unit=str(unit),args=args)
    with (U/logname).open('x',encoding='utf8') as log:
        r=subprocess.run([sys.executable,'-B',*args],cwd=unit,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,(str(unit),args,r.returncode,logname)
def main():
    assert not (U/'CONTINUATION_RUNNING.json').exists()
    write(U/'CONTINUATION_RUNNING.json',dict(at=now(),pid=os.getpid(),script_sha256=sha(Path(__file__)),train_frozen_sha256=sha(I/'FROZEN.json'),dev_frozen_sha256=sha(U/'FROZEN.json')))
    assert read(U/'SMOKE_REVIEW.json')['ready_for_full']
    status('WAIT_EXISTING_TRAIN_PREPARATION',expected_views=1798)
    started=time.monotonic()
    while True:
        assert not list(I.glob('ERROR_*.json')),'TRAIN prep failure must be inspected before continuation.'
        if not (I/'RUNNING.json').exists():
            assert len(list((I/'records').glob('*.json')))==1798,'TRAIN preparation ended incomplete.'
            events=[json.loads(s) for s in (I/'EVENTS.jsonl').read_text('utf8').splitlines() if s.strip()]
            assert events[-1]['stage']=='BATCH_COMPLETE' and events[-1]['batch']=='all' and events[-1]['total']==1798
            break
        assert time.monotonic()-started<14400,'Wait exceeded 4h; inspect live process, never remove TRAIN lock.'
        time.sleep(30)
    if not (I/'ALL_VERIFIED.json').exists():child(I,['verify.py','all'],'TRAIN_VERIFICATION_LOG.txt')
    assert read(I/'ALL_VERIFIED.json')['status']=='PASS' and read(I/'ALL_VERIFIED.json')['frozen_sha256']==sha(I/'FROZEN.json')
    assert not (U/'RUNNING.json').exists()
    if len(list((U/'records').glob('*.json')))<600:child(U,['prepare.py','all'],'DEV_PREPARATION_LOG.txt')
    assert len(list((U/'records').glob('*.json')))==600 and not (U/'RUNNING.json').exists()
    if not (U/'ALL_VERIFIED.json').exists():child(U,['verify.py','all'],'DEV_VERIFICATION_LOG.txt')
    assert read(U/'ALL_VERIFIED.json')['status']=='PASS'
    status('INPUTS_PREPARED_NUMERICALLY_VERIFIED_VISUAL_REVIEW_REQUIRED',training_started=False)
    (U/'CONTINUATION_RUNNING.json').unlink()
if __name__=='__main__':
    try:main()
    except Exception:
        write(U/('CONTINUATION_FAILURE_'+now().replace(':','_')+'.json'),dict(error=traceback.format_exc()));raise
