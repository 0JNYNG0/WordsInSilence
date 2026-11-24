"""Resolve only completion metadata after all six saved scores passed recount."""
from common import *
import shutil,time

def main():
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins']);r=read(U/'COMPARISON_INDEPENDENT.json');check(r['pins'])
 assert r['status']=='PASS_SAVED_SCORE_RECOUNT' and r['predictions_verified']==2400
 assert read(U/'EVAL_EXIT.json')['exit_code']==1 and not (U/'EVAL_RUNNING.json').exists()
 error=(U/'FIRST_EVALUATION_ERROR.txt').read_text(encoding='utf-8-sig')
 assert 'PermissionError' in error and 'STATUS.json.writing' in error and error.count('EVALUATION_MODEL_SAVED')==6
 for name in ['STATUS.json','STATUS.json.writing']:
  p=U/name
  if p.exists():shutil.copyfile(p,U/'before'/('after_eval_error_'+name))
 recovery={'at':now(),'status':'PASS_SAVED_RESULTS_COMPLETE_STATUS_ONLY_RECOVERY','original_exit_code':1,'failure':'Windows denied STATUS.json atomic replacement after all six models were saved and evaluation lock removed.','saved_scores_verified':2400,'models_completed':6,'new_inference':0,'training':0,'model_or_frozen_code_changes':0,'score_pins':r['pins'],'first_error_sha256':sha(U/'FIRST_EVALUATION_ERROR.txt'),'original_exit_sha256':sha(U/'EVAL_EXIT.json'),'comparison_sha256':sha(U/'COMPARISON_INDEPENDENT.json')}
 write(U/'EVALUATION_RECOVERY.json',recovery,new=True)
 for i in range(3):
  try:event(stage='EVALUATION_VERIFIED_STATUS_RECOVERED',models=6,predictions=2400,original_exit=1,new_inference=0);break
  except PermissionError:
   if i==2:raise
   time.sleep(.5)
 print(recovery['status'])

if __name__=='__main__':main()
