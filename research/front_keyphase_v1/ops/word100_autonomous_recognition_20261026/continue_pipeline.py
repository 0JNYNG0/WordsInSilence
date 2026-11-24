"""Resume after existing preparation. No duplicate extraction, retries or model tuning."""
from common import *
import subprocess,time,os,traceback

def main():
 f=U/'fresh199';lock=U/'AUTONOMOUS_DRIVER_LOCK.json';write(lock,dict(pid=os.getpid(),at=now()),new=True)
 steps=[('verify_inputs',f/'verify_inputs.py',f/'INPUTS_VERIFIED.json'),('evaluate',f/'evaluate.py',f/'T3F_20261015_SCORES.json'),('aggregate',U/'aggregate_fresh.py',f/'COMPARISON.json'),('verify_scores',U/'verify_fresh.py',f/'SCORES_VERIFIED.json'),('report',U/'report.py',U/'COMPLETION.json')]
 pins={str(p):sha(p) for _,p,_ in steps};pins[str(U/'common.py')]=sha(U/'common.py');pins[str(U/'FROZEN.json')]=sha(U/'FROZEN.json');pins[str(f/'EVAL_CONTEXT.json')]=sha(f/'EVAL_CONTEXT.json')
 write(U/'PIPELINE_FROZEN.json',dict(at=now(),pins=pins,steps=[name for name,_,_ in steps],no_model_tuning=True,training=0),new=True)
 try:
  started=time.monotonic()
  while not (f/'PREP_COMPLETE.json').exists() or (f/'PREP_RUNNING.json').exists():
   assert not (f/'FIRST_PREP_ERROR.json').exists(),'Input first error preserved: inspect before continuing'
   assert time.monotonic()-started<7200,'Preparation not completed in2h; inspect running task, do not clear other lock'
   time.sleep(15)
  assert read(f/'PREP_COMPLETE.json')['records']==199
  for name,script,out in steps:
   for p,h in pins.items():assert sha(p)==h,p
   if out.exists():raise RuntimeError('Existing step result: inspect and reuse manually instead of duplicate '+name)
   write(U/'DRIVER_STATUS.json',dict(at=now(),stage=name,pid=os.getpid()),new=False)
   log=U/f'driver_{name}.log'
   with log.open('x',encoding='utf8') as stream:
    done=subprocess.run([sys.executable,'-X','utf8',str(script)],cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
   assert done.returncode==0,f'{name} failed with{done.returncode}; retained {log}'
   assert out.exists(),name
  write(U/'DRIVER_STATUS.json',dict(at=now(),stage='COMPLETE_PENDING_REPORT_REVIEW_AND_HANDOFF',pid=os.getpid()),new=False)
  print('PIPELINE COMPLETE; final review and handoff pending',flush=True)
 except BaseException as exc:
  err=dict(at=now(),error=str(exc),traceback=traceback.format_exc())
  if not (U/'FIRST_DRIVER_ERROR.json').exists():write(U/'FIRST_DRIVER_ERROR.json',err)
  write(U/'DRIVER_STATUS.json',dict(at=now(),stage='ERROR_PRESERVED',error=str(exc)),new=False)
  raise
 finally:
  if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()
if __name__=='__main__':main()
