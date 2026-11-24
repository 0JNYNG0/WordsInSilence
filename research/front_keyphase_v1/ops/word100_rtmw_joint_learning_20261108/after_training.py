"""Task-owned continuation of fixed comparisons; no app automation."""
from common import *
import subprocess,traceback

def main():
 while not (U/'TCN_TRAINING_COMPLETE.json').exists():
  failed=[p for pattern in ['EXTRACTION_FAILURE_*.txt','CHAIN_FAILURE_*.txt','TCN_FAILURE_*.txt'] for p in U.glob(pattern)]
  if failed:raise RuntimeError('Training chain failure needs review: '+','.join(p.name for p in failed))
  time.sleep(5)
 runtime=O/'word100_signrep_feasibility_20261029/runtime/Scripts/python.exe'
 for script,done,exe in [
  ('compare_tcn.py','TCN_COMPARISON.json',sys.executable),
  ('time_models.py','MODEL_TIMING.json',sys.executable),
  ('stream_trained_models.py','TRAINED_STREAM_COMPLETE.json',str(runtime)),
  ('verify_results.py','FINAL_VERIFICATION.json',sys.executable),
  ('report_results.py','RESULT.json',sys.executable)]:
  assert not (U/done).exists(),'Completed task must not be rerun: '+done
  event('final_chain_stage',script=script)
  with (U/(script+'.log')).open('a',encoding='utf-8') as log:
   subprocess.run([exe,'-X','utf8',str(U/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
 event('all_fixed_comparisons_complete')
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'FINAL_CHAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
