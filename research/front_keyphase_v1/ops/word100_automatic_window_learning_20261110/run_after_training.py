from pathlib import Path
import json,sys,subprocess,time,os,traceback
U=Path(__file__).resolve().parent
def status(phase): (U/'CHAIN_STATUS.json').write_text(json.dumps(dict(pid=os.getpid(),phase=phase,time=time.time())),encoding='utf-8')
try:
 status('waiting_for_training');start=time.time()
 while not (U/'TRAINING_COMPLETE.json').exists():
  if list(U.glob('FAILURE_*.txt')):raise RuntimeError('Training failure exists; inspect before continuing')
  if time.time()-start>1800:raise TimeoutError('Training wait exceeded30minutes')
  time.sleep(2)
 for script,done in [('evaluate.py','EVALUATION_COMPLETE.json'),('fullpath.py','FULLPATH_COMPLETE.json'),('verify.py','VERIFICATION.json')]:
  if (U/done).exists():continue
  status(script)
  with (U/(script+'.log')).open('w',encoding='utf-8') as log:
   run=subprocess.run([sys.executable,'-X','utf8',str(U/script)],stdout=log,stderr=subprocess.STDOUT)
  if run.returncode:raise RuntimeError(script+' failed; see '+str(U/(script+'.log')))
 status('completed');print('Training, component evaluation, full-path evaluation and saved-result verification completed.')
except BaseException:
 status('failed');(U/f'CHAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
