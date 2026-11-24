"""This task's foreground-owned continuation; not an app automation."""
from common import *
import subprocess,traceback
def main():
 while not (U/'EXTRACTION_COMPLETE.json').exists():
  if list(U.glob('EXTRACTION_FAILURE_*.txt')):raise RuntimeError('Extraction failure needs review')
  time.sleep(5)
 for script in ['verify_inputs.py','check_tcn.py','train_tcn.py']:
  with (U/(script+'.log')).open('a',encoding='utf-8') as log:subprocess.run([sys.executable,'-X','utf8',str(U/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
 event('tcn_chain_complete')
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'CHAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
